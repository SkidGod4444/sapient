// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Asynchronous action chunking: run the policy in the background while the
//! robot executes the actions it already has.
//!
//! A chunk takes ~0.6 s (Apple M4) to ~3.3 s (Raspberry Pi 5) to compute and
//! holds 50 actions. Executing a chunk and only then computing the next one
//! makes the robot stand still for the whole inference time between chunks.
//! [`AsyncActions`] instead asks for the next chunk as soon as the queue of
//! pending actions falls to a threshold, keeps executing the queue meanwhile,
//! and merges the new chunk in when it arrives (the scheme of LeRobot's async
//! inference, Shukor et al. 2025):
//!
//! * every action carries the **step** it is meant for, counted in actions
//!   EXECUTED, not in control ticks: action `i` of a chunk computed from the
//!   observation taken after `s` executed actions is for step `s + i`. A tick
//!   on which the robot stalls does not move it, so it does not age the plan;
//! * when a chunk arrives, actions for steps the robot already executed are
//!   dropped (they were planned from a now-stale observation);
//! * where the new chunk overlaps actions still queued, the two are combined
//!   ([`Aggregate`]).
//!
//! The robot's control loop calls [`AsyncActions::tick`] once per control
//! period. A tick never blocks on inference: if no action is queued for the
//! current step it returns `None` (a **stall** — the robot holds its pose).
//! The stall rate at a given control rate is the number that says whether a
//! machine is fast enough to drive a robot with this policy.

use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};

use crate::vla::VlaPipeline;

/// One observation for the policy.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Preprocessed camera images (see [`VlaPipeline::preprocess_rgb`]).
    pub images: Vec<Vec<f32>>,
    pub task: String,
    pub state: Vec<f32>,
}

/// Anything that turns an observation into an action chunk (`steps` rows).
/// Implemented for [`VlaPipeline`]; tests use a fake.
///
/// `queued` holds the actions still waiting to execute when the chunk was
/// requested. It is empty unless [`Aggregate::Continue`] is set; a policy that
/// supports it returns a chunk whose first rows equal `queued` and whose
/// remaining rows continue them.
pub trait ChunkPolicy: Send + Sync + 'static {
    fn predict_chunk(
        &self,
        obs: &Observation,
        seed: u64,
        queued: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>>;
}

impl ChunkPolicy for VlaPipeline {
    fn predict_chunk(
        &self,
        obs: &Observation,
        seed: u64,
        queued: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        let chunk = if queued.is_empty() {
            self.predict(&obs.images, &obs.task, &obs.state, seed)?
        } else {
            let flat: Vec<f32> = queued.iter().flatten().copied().collect();
            self.predict_continuing(
                &obs.images,
                &obs.task,
                &obs.state,
                &self.noise_for_seed(seed),
                None,
                &flat,
            )?
        };
        Ok((0..chunk.steps).map(|i| chunk.row(i).to_vec()).collect())
    }
}

/// How an arriving chunk is combined with actions already queued for the same
/// control steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregate {
    /// The newer chunk replaces the queued actions (it saw a newer observation).
    Latest,
    /// The mean of the queued and the new action (smoother hand-over).
    Average,
    /// The queued actions are sent with the request and the policy generates
    /// a chunk that keeps them and continues them (hard inpainting — see
    /// `SmolVla::sample_actions_inpaint`). In LIBERO with a simulated delay of
    /// 0.64 chunk it completed 9 of 30 episodes where `Latest` completed 4 and
    /// synchronous execution 14; at a quarter chunk it made no difference.
    Continue,
}

/// When to ask for the next chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    /// Pick it from the measured inference latency — see [`auto_trigger`].
    Auto,
    /// Ask when at most this fraction of a chunk is still queued (LeRobot's
    /// `chunk_size_threshold`). 0 = only when the queue is empty — synchronous
    /// execution (wait for each chunk, then execute all of it); 1 = keep one
    /// inference running at all times.
    Fraction(f32),
}

impl std::str::FromStr for Threshold {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let f: f32 = s
            .parse()
            .map_err(|_| anyhow!("threshold must be `auto` or a number in 0..=1, got {s:?}"))?;
        if !(0.0..=1.0).contains(&f) {
            return Err(anyhow!("threshold must be in 0..=1, got {f}"));
        }
        Ok(Self::Fraction(f))
    }
}

impl std::fmt::Display for Threshold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Fraction(x) => write!(f, "{x}"),
        }
    }
}

/// The queue length at or below which [`Threshold::Auto`] asks for a new chunk,
/// given the inference latency `d` in control ticks and `c` actions per chunk.
///
/// With chunks aligned by executed actions, a chunk requested with `q` actions
/// queued arrives `d` ticks later carrying `c − min(q, d)` usable actions.
/// * `d ≤ c/2`: asking just before the queue runs out (`q ≈ d`, plus a margin
///   for latency jitter) never stalls and computes the fewest chunks. Asking
///   earlier only burns more compute.
/// * `d > c/2`: no trigger avoids stalls, and synchronous execution (`0`: ask
///   only when empty) is chosen. For `c/2 < d < c` asking at once would stall
///   less (`1 − c/(2d)` against `d/(c+d)`), but it completes fewer tasks: in
///   LIBERO-Spatial with a simulated delay of 32 ticks and `c = 50`, immediate
///   requests stalled 27% of ticks instead of 45% and succeeded in 4 of 30
///   episodes instead of 14 (paired p = 0.006) — probably because each chunk
///   then contributes only its late actions, planned from an observation `d`
///   ticks old (not yet tested by an ablation). For
///   `d ≥ c` asynchronous execution also stalls more. Pass a fixed
///   `Threshold::Fraction` to trade success for fewer stalls.
///
/// Measured on a Pi 5 (d = 17, 34, 50, 100 ticks at 5, 10, 15, 30 Hz; 120–300 s
/// runs, at least 22 chunks each) both the synchronous and the immediate-request
/// stall rates matched the formulas within one point. Earlier 40 s runs read the
/// synchronous rate 2.7–6 points low because they held only 3–8 chunk cycles.
pub fn auto_trigger(d: u64, c: usize) -> usize {
    let d = d as usize;
    if 2 * d > c {
        0
    } else {
        (d + d / 5 + 2).min(c)
    }
}

/// Settings for [`AsyncActions`].
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    pub threshold: Threshold,
    pub aggregate: Aggregate,
    /// Seed of the first chunk's start noise; each later chunk uses the next.
    pub seed: u64,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            threshold: Threshold::Auto,
            aggregate: Aggregate::Latest,
            seed: 0,
        }
    }
}

/// What one [`AsyncActions::tick`] did.
#[derive(Debug, Clone)]
pub struct Tick {
    /// Control ticks so far (wall-clock periods), this one included.
    pub tick: u64,
    /// Index of the action executed on this tick (= actions executed before
    /// it), or of the action awaited if this tick stalled.
    pub step: u64,
    /// The action to execute now, or `None` when nothing is queued for this
    /// step (stall).
    pub action: Option<Vec<f32>>,
    /// An observation was sent for inference on this tick.
    pub requested: bool,
    /// Chunks merged into the queue on this tick, with their inference time.
    pub arrived: Vec<Duration>,
}

/// Counters over the life of an [`AsyncActions`].
#[derive(Debug, Clone, Default)]
pub struct AsyncStats {
    pub ticks: u64,
    pub stalls: u64,
    /// Stalls before the first chunk arrived (start-up, not steady state).
    pub startup_stalls: u64,
    pub chunks: u64,
    /// Actions dropped because their step had passed when the chunk arrived.
    pub dropped: u64,
    /// Inference time of every chunk.
    pub latencies: Vec<Duration>,
    /// The same, in control ticks between request and arrival.
    pub latency_ticks: Vec<u64>,
    /// Queue length that triggered the most recent request (0 = synchronous).
    pub last_trigger: usize,
}

impl AsyncStats {
    /// Stall fraction after the first chunk arrived.
    pub fn steady_stall_rate(&self) -> f64 {
        let steady = self.ticks.saturating_sub(self.startup_stalls);
        if steady == 0 {
            0.0
        } else {
            (self.stalls - self.startup_stalls) as f64 / steady as f64
        }
    }
}

struct Request {
    obs: Observation,
    step: u64,
    seed: u64,
    queued: Vec<Vec<f32>>,
}

struct Reply {
    step: u64,
    took: Duration,
    actions: Result<Vec<Vec<f32>>>,
}

/// Background action chunking for one policy and one robot.
pub struct AsyncActions {
    cfg: AsyncConfig,
    chunk_len: usize,
    tx: Option<Sender<Request>>,
    rx: Receiver<Reply>,
    worker: Option<JoinHandle<()>>,
    /// (step, action), consecutive steps from the front.
    queue: VecDeque<(u64, Vec<f32>)>,
    /// Actions executed so far = the step of the next action to execute.
    step: u64,
    pending: bool,
    /// Tick count when the in-flight request was sent.
    request_tick: u64,
    next_seed: u64,
    stats: AsyncStats,
}

impl AsyncActions {
    /// Start the inference worker. `chunk_len` is the policy's actions per
    /// chunk (`VlaPipeline::chunk_len`).
    pub fn new<P: ChunkPolicy>(policy: Arc<P>, chunk_len: usize, cfg: AsyncConfig) -> Self {
        let (tx, worker_rx) = channel::<Request>();
        let (worker_tx, rx) = channel::<Reply>();
        let worker = std::thread::Builder::new()
            .name("sapient-vla-async".into())
            .spawn(move || {
                for req in worker_rx {
                    let started = Instant::now();
                    let actions = policy.predict_chunk(&req.obs, req.seed, &req.queued);
                    let reply = Reply {
                        step: req.step,
                        took: started.elapsed(),
                        actions,
                    };
                    if worker_tx.send(reply).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn VLA worker thread");
        Self {
            next_seed: cfg.seed,
            cfg,
            chunk_len,
            tx: Some(tx),
            rx,
            worker: Some(worker),
            queue: VecDeque::new(),
            step: 0,
            pending: false,
            request_tick: 0,
            stats: AsyncStats::default(),
        }
    }

    /// Actions queued from the current step on.
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    pub fn stats(&self) -> &AsyncStats {
        &self.stats
    }

    /// One control period. `observe` is called only when a new chunk is
    /// requested on this tick (it should capture the current camera frames and
    /// state). Never blocks on inference.
    pub fn tick(&mut self, observe: impl FnOnce() -> Result<Observation>) -> Result<Tick> {
        let mut arrived = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(reply) => {
                    self.pending = false;
                    arrived.push(reply.took);
                    self.merge(reply.step, reply.actions?);
                    self.stats.chunks += 1;
                    self.stats.latencies.push(reply.took);
                    self.stats
                        .latency_ticks
                        .push(self.stats.ticks - self.request_tick);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    return Err(anyhow!("VLA worker thread stopped"))
                }
            }
        }

        let mut requested = false;
        let trigger = self.trigger();
        if !self.pending && self.queue.len() <= trigger {
            self.stats.last_trigger = trigger;
            self.request_tick = self.stats.ticks;
            let obs = observe()?;
            let queued = match self.cfg.aggregate {
                Aggregate::Continue => self.queue.iter().map(|(_, a)| a.clone()).collect(),
                _ => Vec::new(),
            };
            let req = Request {
                obs,
                step: self.step,
                seed: self.next_seed,
                queued,
            };
            self.next_seed = self.next_seed.wrapping_add(1);
            self.tx
                .as_ref()
                .ok_or_else(|| anyhow!("VLA worker shut down"))?
                .send(req)
                .map_err(|_| anyhow!("VLA worker thread stopped"))?;
            self.pending = true;
            requested = true;
        }

        let action = match self.queue.front() {
            Some((s, _)) if *s == self.step => self.queue.pop_front().map(|(_, a)| a),
            _ => None,
        };
        let step = self.step;
        if action.is_some() {
            self.step += 1;
        }
        self.stats.ticks += 1;
        if action.is_none() {
            self.stats.stalls += 1;
            if self.stats.chunks == 0 {
                self.stats.startup_stalls += 1;
            }
        }
        Ok(Tick {
            tick: self.stats.ticks,
            step,
            action,
            requested,
            arrived,
        })
    }

    /// Queue length at or below which a new chunk is requested.
    fn trigger(&self) -> usize {
        match self.cfg.threshold {
            Threshold::Fraction(f) => (f.clamp(0.0, 1.0) * self.chunk_len as f32) as usize,
            // Worst of the last few latencies, so one fast chunk does not make
            // the next request late. Before the first measurement: ask when
            // empty (the start-up request).
            Threshold::Auto => {
                let recent = self.stats.latency_ticks.iter().rev().take(5).max();
                recent.map_or(0, |&d| auto_trigger(d, self.chunk_len))
            }
        }
    }

    /// Merge a chunk computed from the observation at `obs_step`.
    fn merge(&mut self, obs_step: u64, actions: Vec<Vec<f32>>) {
        for (i, a) in actions.into_iter().enumerate() {
            let t = obs_step + i as u64;
            if t < self.step {
                self.stats.dropped += 1;
                continue;
            }
            let front = self.queue.front().map(|(s, _)| *s).unwrap_or(t);
            let pos = t.checked_sub(front).map(|p| p as usize);
            match pos {
                Some(p) if p < self.queue.len() => {
                    let slot = &mut self.queue[p].1;
                    match self.cfg.aggregate {
                        // With `Continue` the chunk's rows for queued steps
                        // are the queued actions themselves.
                        Aggregate::Latest | Aggregate::Continue => *slot = a,
                        Aggregate::Average => {
                            for (o, n) in slot.iter_mut().zip(&a) {
                                *o = 0.5 * (*o + n);
                            }
                        }
                    }
                }
                Some(p) if p == self.queue.len() => self.queue.push_back((t, a)),
                // Cannot happen with one request in flight (the queue is
                // always the consecutive run of steps from the current one);
                // if it ever does, the newest plan wins.
                _ => {
                    self.queue.clear();
                    self.queue.push_back((t, a));
                }
            }
        }
    }
}

impl Drop for AsyncActions {
    fn drop(&mut self) {
        // Closing the request channel ends the worker's loop; an inference in
        // progress finishes first.
        self.tx.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// Result of [`simulate`]: a control loop driven at a fixed rate against a
/// real policy, with a static scene.
#[derive(Debug, Clone)]
pub struct SimulationReport {
    pub hz: f64,
    pub seconds: f64,
    pub stats: AsyncStats,
}

/// Drive [`AsyncActions`] at `hz` for `seconds` of wall-clock time, feeding the
/// same `images` every request and the last executed action back as the state
/// (a stand-in for a robot that tracks its commands). Measures stalls and
/// inference latency under the real timing of this machine.
#[allow(clippy::too_many_arguments)]
pub fn simulate<P: ChunkPolicy>(
    policy: Arc<P>,
    chunk_len: usize,
    images: Vec<Vec<f32>>,
    task: &str,
    state: Vec<f32>,
    hz: f64,
    seconds: f64,
    cfg: AsyncConfig,
) -> Result<SimulationReport> {
    let mut runner = AsyncActions::new(policy, chunk_len, cfg);
    let period = Duration::from_secs_f64(1.0 / hz);
    let ticks = (seconds * hz).round() as u64;
    let start = Instant::now();
    let mut current = state;
    for k in 0..ticks {
        let tick = runner.tick(|| {
            Ok(Observation {
                images: images.clone(),
                task: task.to_string(),
                state: current.clone(),
            })
        })?;
        if let Some(a) = tick.action {
            current = a[..current.len().min(a.len())].to_vec();
        }
        // Sleep to the next period boundary (no drift accumulation).
        let next = start + period * (k as u32 + 1);
        if let Some(wait) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
    }
    Ok(SimulationReport {
        hz,
        seconds,
        stats: runner.stats().clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns `chunk` actions whose single value encodes (observation step,
    /// index): `obs.state[0] * 1000 + i`, after sleeping `delay`.
    struct Fake {
        chunk: usize,
        delay: Duration,
    }

    impl ChunkPolicy for Fake {
        fn predict_chunk(
            &self,
            obs: &Observation,
            _seed: u64,
            _queued: &[Vec<f32>],
        ) -> Result<Vec<Vec<f32>>> {
            std::thread::sleep(self.delay);
            Ok((0..self.chunk)
                .map(|i| vec![obs.state[0] * 1000.0 + i as f32])
                .collect())
        }
    }

    fn obs(step: u64) -> Result<Observation> {
        Ok(Observation {
            images: vec![],
            task: String::new(),
            state: vec![step as f32],
        })
    }

    /// Tick until the worker has answered (bounded wait), counting stalls.
    fn wait_for_chunk(r: &mut AsyncActions) -> Vec<Tick> {
        let mut ticks = Vec::new();
        let start = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(2));
            let step = r.step;
            let t = r.tick(|| obs(step)).unwrap();
            let got = !t.arrived.is_empty();
            ticks.push(t);
            if got || start.elapsed() > Duration::from_secs(5) {
                return ticks;
            }
        }
    }

    #[test]
    fn waiting_does_not_age_the_plan() {
        // The robot stalls during the first inference; it has not moved, so the
        // chunk starts at its first action and nothing is dropped.
        let policy = Arc::new(Fake {
            chunk: 10,
            delay: Duration::from_millis(20),
        });
        let cfg = AsyncConfig {
            threshold: Threshold::Fraction(0.0),
            ..Default::default()
        };
        let mut r = AsyncActions::new(policy, 10, cfg);
        let ticks = wait_for_chunk(&mut r);
        assert!(ticks[0].requested);
        assert!(ticks.len() > 1, "inference should take several ticks");
        let last = ticks.last().unwrap();
        assert_eq!(last.step, 0);
        assert_eq!(last.action.as_ref().expect("action on arrival")[0], 0.0);
        assert_eq!(r.stats().dropped, 0);
        assert_eq!(r.stats().startup_stalls, ticks.len() as u64 - 1);
    }

    #[test]
    fn early_request_keeps_the_queue_full() {
        // A new chunk is requested while 45 of 50 actions are still queued
        // (≥ 180 ms at 4 ms per tick); inference takes 20 ms, so after
        // start-up the robot should never stall.
        let policy = Arc::new(Fake {
            chunk: 50,
            delay: Duration::from_millis(20),
        });
        let cfg = AsyncConfig {
            threshold: Threshold::Fraction(0.9),
            ..Default::default()
        };
        let mut r = AsyncActions::new(policy, 50, cfg);
        wait_for_chunk(&mut r);
        let mut last = None;
        for _ in 0..200 {
            std::thread::sleep(Duration::from_millis(4));
            let step = r.step;
            let t = r.tick(|| obs(step)).unwrap();
            let a = t.action.expect("no stall after start-up");
            // Every action was planned for exactly this step by some chunk:
            // value = obs_step·1000 + (step − obs_step).
            let v = a[0] as u64;
            let (obs_step, i) = (v / 1000, v % 1000);
            assert_eq!(obs_step + i, t.step, "action for the wrong step");
            if let Some(prev) = last {
                assert!(obs_step >= prev, "older chunk overrode a newer one");
            }
            last = Some(obs_step);
        }
        assert!(r.stats().chunks > 10);
        assert_eq!(r.stats().steady_stall_rate(), 0.0);
        // Actions executed while a chunk was being computed are dropped from it.
        assert!(r.stats().dropped > 0);
    }

    #[test]
    fn sequential_threshold_stalls_for_every_inference() {
        // threshold 0: a new chunk is requested only when the queue is empty,
        // so each chunk boundary costs a full inference of stalls — the
        // synchronous behaviour.
        let policy = Arc::new(Fake {
            chunk: 5,
            // Long enough that slow CI sleeps (up to ~10 ms per tick) still
            // leave a majority of stalled ticks.
            delay: Duration::from_millis(80),
        });
        let cfg = AsyncConfig {
            threshold: Threshold::Fraction(0.0),
            ..Default::default()
        };
        let mut r = AsyncActions::new(policy, 5, cfg);
        wait_for_chunk(&mut r);
        for _ in 0..150 {
            std::thread::sleep(Duration::from_millis(2));
            let step = r.step;
            r.tick(|| obs(step)).unwrap();
        }
        assert!(
            r.stats().steady_stall_rate() > 0.3,
            "{:?}",
            r.stats().steady_stall_rate()
        );
        // Every chunk is executed in full.
        assert_eq!(r.stats().dropped, 0);
    }

    /// Records the queue it is sent and, like a continuing policy, returns
    /// those rows first and its own plan after them.
    struct Continuing {
        seen: std::sync::Mutex<Vec<usize>>,
    }

    impl ChunkPolicy for Continuing {
        fn predict_chunk(
            &self,
            obs: &Observation,
            _seed: u64,
            queued: &[Vec<f32>],
        ) -> Result<Vec<Vec<f32>>> {
            self.seen.lock().unwrap().push(queued.len());
            std::thread::sleep(Duration::from_millis(12));
            let mut rows = queued.to_vec();
            rows.extend((rows.len()..10).map(|i| vec![obs.state[0] + i as f32]));
            Ok(rows)
        }
    }

    #[test]
    fn continue_sends_the_queue_and_latest_does_not() {
        let run = |aggregate| {
            let policy = Arc::new(Continuing {
                seen: std::sync::Mutex::new(Vec::new()),
            });
            let mut r = AsyncActions::new(
                policy.clone(),
                10,
                AsyncConfig {
                    threshold: Threshold::Fraction(0.5),
                    aggregate,
                    ..Default::default()
                },
            );
            wait_for_chunk(&mut r);
            for _ in 0..60 {
                std::thread::sleep(Duration::from_millis(3));
                let step = r.step;
                r.tick(|| obs(step)).unwrap();
            }
            let seen = policy.seen.lock().unwrap().clone();
            seen
        };
        let sent = run(Aggregate::Continue);
        assert!(sent.len() > 2, "expected several requests, got {sent:?}");
        assert_eq!(sent[0], 0, "nothing is queued for the first request");
        assert!(
            sent[1..].iter().any(|&n| n > 0),
            "later requests carry the queued actions: {sent:?}"
        );
        assert!(run(Aggregate::Latest).iter().all(|&n| n == 0));
    }

    #[test]
    fn auto_trigger_regimes() {
        // Fast: just in time with a margin.
        assert_eq!(auto_trigger(10, 50), 14);
        assert_eq!(auto_trigger(0, 50), 2);
        // Exactly half a chunk is still stall-free: ask at once.
        assert!(auto_trigger(25, 50) >= 25);
        // More than half a chunk: synchronous (asking at once would stall
        // less but completes fewer tasks — see the doc comment).
        assert_eq!(auto_trigger(26, 50), 0);
        assert_eq!(auto_trigger(33, 50), 0);
        assert_eq!(auto_trigger(50, 50), 0);
        assert_eq!(auto_trigger(99, 50), 0);
        assert_eq!("auto".parse::<Threshold>().unwrap(), Threshold::Auto);
        assert_eq!(
            "0.5".parse::<Threshold>().unwrap(),
            Threshold::Fraction(0.5)
        );
        assert!("1.5".parse::<Threshold>().is_err());
    }

    #[test]
    fn auto_never_stalls_and_computes_less_than_eager() {
        // 20 ms inference, ~4 ms ticks → d ≈ 5–6 ticks, far below 50.
        let run = |threshold| {
            let policy = Arc::new(Fake {
                chunk: 50,
                delay: Duration::from_millis(20),
            });
            let mut r = AsyncActions::new(
                policy,
                50,
                AsyncConfig {
                    threshold,
                    ..Default::default()
                },
            );
            wait_for_chunk(&mut r);
            for _ in 0..250 {
                std::thread::sleep(Duration::from_millis(4));
                let step = r.step;
                r.tick(|| obs(step)).unwrap();
            }
            r.stats().clone()
        };
        let auto = run(Threshold::Auto);
        let eager = run(Threshold::Fraction(1.0));
        // Just in time means a 20% + 2-tick margin over the worst recent
        // latency. On a shared CI runner one scheduler hiccup can exceed it
        // (seen: 2 stalls in 251 ticks on macOS x86_64), so allow up to 2% here;
        // real-hardware stall rates are measured with `sapient act --simulate`.
        assert!(
            auto.steady_stall_rate() <= 0.02,
            "auto stalled {:.1}% of ticks",
            auto.steady_stall_rate() * 100.0
        );
        assert!(
            auto.chunks * 2 < eager.chunks,
            "auto {} vs eager {} chunks",
            auto.chunks,
            eager.chunks
        );
    }

    #[test]
    fn auto_goes_synchronous_when_inference_outlasts_a_chunk() {
        // 5-action chunks, 80 ms inference, 2 ms ticks: d ≫ c.
        let policy = Arc::new(Fake {
            chunk: 5,
            delay: Duration::from_millis(80),
        });
        let mut r = AsyncActions::new(policy, 5, AsyncConfig::default());
        wait_for_chunk(&mut r);
        for _ in 0..150 {
            std::thread::sleep(Duration::from_millis(2));
            let step = r.step;
            r.tick(|| obs(step)).unwrap();
        }
        assert_eq!(r.stats().last_trigger, 0);
        assert_eq!(
            r.stats().dropped,
            0,
            "synchronous chunks are executed in full"
        );
    }

    #[test]
    fn average_aggregation_blends_overlap() {
        let policy = Arc::new(Fake {
            chunk: 4,
            delay: Duration::from_millis(1),
        });
        let mut r = AsyncActions::new(
            policy,
            4,
            AsyncConfig {
                threshold: Threshold::Fraction(1.0),
                aggregate: Aggregate::Average,
                seed: 0,
            },
        );
        r.queue = VecDeque::from(vec![(0, vec![10.0]), (1, vec![20.0]), (2, vec![30.0])]);
        r.merge(1, vec![vec![0.0], vec![0.0], vec![0.0], vec![8.0]]);
        let got: Vec<(u64, f32)> = r.queue.iter().map(|(s, a)| (*s, a[0])).collect();
        assert_eq!(
            got,
            vec![(0, 10.0), (1, 10.0), (2, 15.0), (3, 0.0), (4, 8.0)]
        );
    }
}
