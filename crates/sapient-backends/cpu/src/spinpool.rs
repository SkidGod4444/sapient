// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Persistent spin-wait worker pool for the decode hot path (llama.cpp-style).
//!
//! One decoded token runs ~200+ GEMV parallel regions — one per matmul call —
//! and every rayon region pays worker wake + park latency (µs-scale futex
//! round-trips), ~230 barriers per token. Here the workers stay HOT for the
//! duration of a generation: between ops they spin (bounded iterations, then
//! park on a condvar), so dispatching an op during decode costs a few atomic
//! operations instead of thread wakeups. This is the shape of llama.cpp's
//! ggml threadpool, which is where its ~1.6× multicore decode-scaling edge
//! over per-GEMV fork/join was measured to come from (docs/BENCHMARKS.md).
//!
//! ## Contract
//! - [`run(n_chunks, &f)`](SpinPool::run) executes `f(0..n_chunks)` exactly
//!   once each, in parallel (the caller participates), returning only after
//!   ALL chunks complete. Chunks must touch disjoint data — the same contract
//!   as `par_chunks_mut`.
//! - Concurrent publishers serialize on a lock: a `rayon::join` of two
//!   matmuls degrades to two back-to-back fully-parallel ops (never a
//!   deadlock, and each op still uses the whole pool).
//! - [`enabled()`] is false while the thermal governor is shedding cores
//!   (spinning workers would defeat the backoff) and when `SAPIENT_SPINPOOL=0`
//!   (escape hatch + the A/B lever). Callers fall back to rayon.
//!
//! ## Op-handoff protocol (seqlock-style)
//! The generation counter is EVEN when an op is published and ODD while
//! the slot is being rewritten. A publisher (holding `publish`):
//! 1. bumps `gen` to ODD — closes the door: no worker can newly join,
//! 2. waits for `active == 0` (workers still inside the previous op leave),
//! 3. rewrites the op slot + chunk counters,
//! 4. bumps `gen` to EVEN and wakes parked workers.
//!
//! A worker joins by registering in `active` FIRST and then making its
//! authoritative `gen` read (both SeqCst): if that read saw the old even
//! generation it happened before the odd bump, so the publisher's drain in
//! step 2 observes the registration and waits the worker out; if it happened
//! after, the worker sees ODD and backs out. Either way a worker can never
//! copy the slot while it is being rewritten. (The original order — drain
//! BEFORE the odd bump — left exactly that window, and it segfaulted in
//! practice under rapid park/wake cycling: a late joiner validated a
//! still-unchanged gen and copied a mid-rewrite slot. The stress test below
//! pins the fixed behavior.)

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Spin iterations before a worker parks on the condvar. MEASURED optimum on
/// M4 (llama-1B Q4_K_M decode sweep: 0 → 54.2 tok/s, 4k → 63.6, 16k → 59.1,
/// 50k → 52.3, 200k hot-spin → 38.6): ~4k iterations (~10–20 µs) catches the
/// back-to-back GEMVs inside a layer, while parking through the longer serial
/// phases (attention, sampling). Long spins actively HURT — workers burning
/// cores during serial phases steal the package power budget / scheduler
/// slots from the critical-path thread. Overridable via
/// `SAPIENT_SPINPOOL_SPINS`.
const DEFAULT_SPIN_ITERS: u64 = 4_000;

#[derive(Clone, Copy)]
struct OpSlot {
    call: unsafe fn(*const (), usize),
    ctx: *const (),
    n_chunks: usize,
    /// Chunks per claimed block (guided scheduling granularity).
    block: usize,
}

impl Default for OpSlot {
    fn default() -> Self {
        unsafe fn noop(_: *const (), _: usize) {}
        OpSlot {
            call: noop,
            ctx: std::ptr::null(),
            n_chunks: 0,
            block: 1,
        }
    }
}

/// Pad each hot atomic to its own cache line (128 B covers Apple Silicon's
/// line pairs). Without this, `generation` — which every idle worker spins
/// on — shares a line with `completed`/`next_chunk`, so every chunk
/// completion invalidates the spinners' line and every spin-load contends
/// the completer's store: measured ~2× decode REGRESSION on M4 before
/// padding. This is why llama.cpp's threadpool pads its counters.
#[repr(align(128))]
struct Pad<T>(T);

pub struct SpinPool {
    /// Seqlock generation: even = published, odd = slot being rewritten.
    generation: Pad<AtomicU64>,
    op: UnsafeCell<OpSlot>,
    /// GUIDED claiming: participants grab contiguous BLOCKS of chunks off
    /// this counter (~3 blocks per participant). Measured rationale: v1
    /// per-chunk claiming load-balanced the M4's P/E cores (+5% vs rayon)
    /// but its ~112 RMWs/op on two hot lines plus scattered per-thread
    /// chunk order was a 2× regression on 14-core Thor; v2 static shares
    /// fixed Thor (2× → −8%) but equal shares made every op wait for the
    /// slowest E-core on M4 (45 → 31 tok/s, nothing left to steal). Guided
    /// blocks keep v2's contiguity at ~4× less claim traffic than v1 while
    /// letting fast cores take more blocks (v1's balance).
    next_block: Pad<AtomicUsize>,
    /// Completed BLOCKS (one increment per block, not per chunk).
    completed: Pad<AtomicUsize>,
    /// Workers currently inside an op (validated slot copy → last share).
    active: Pad<AtomicUsize>,
    /// Serializes publishers.
    publish: Mutex<()>,
    sleep: Mutex<()>,
    wake: Condvar,
    parked: Pad<AtomicUsize>,
    workers: usize,
    spin_iters: u64,
}

// SAFETY: the op slot is only written by the publisher while `gen` is odd and
// `active == 0` (enforced by the protocol above); workers only read it after
// validating an even, unchanged `gen` while holding an `active` reference.
unsafe impl Sync for SpinPool {}

impl SpinPool {
    fn new(workers: usize, spin_iters: u64) -> &'static SpinPool {
        let pool = Box::leak(Box::new(SpinPool {
            generation: Pad(AtomicU64::new(0)),
            op: UnsafeCell::new(OpSlot::default()),
            next_block: Pad(AtomicUsize::new(0)),
            completed: Pad(AtomicUsize::new(0)),
            active: Pad(AtomicUsize::new(0)),
            publish: Mutex::new(()),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
            parked: Pad(AtomicUsize::new(0)),
            workers,
            spin_iters,
        }));
        for w in 0..workers {
            let p: &'static SpinPool = pool;
            std::thread::Builder::new()
                .name(format!("sapient-spin-{w}"))
                .spawn(move || p.worker_loop())
                .expect("spawn spinpool worker");
        }
        pool
    }

    /// Execute this op's blocks: claim contiguous blocks of `op.block`
    /// chunks off the shared counter until none remain. Dynamic (fast cores
    /// take more blocks — the M4 P/E balance) yet contiguous within each
    /// block (the Thor prefetch locality). One completion increment per
    /// block. A parked worker simply claims nothing — no deadlock.
    fn execute_blocks(&self, op: &OpSlot) {
        loop {
            let b = self.next_block.0.fetch_add(1, Ordering::Relaxed);
            let lo = b * op.block;
            if lo >= op.n_chunks {
                break;
            }
            let hi = (lo + op.block).min(op.n_chunks);
            for c in lo..hi {
                // SAFETY: ctx outlives the op (the publisher blocks in
                // `run` until every block completes).
                unsafe { (op.call)(op.ctx, c) };
            }
            self.completed.0.fetch_add(1, Ordering::Release);
        }
    }

    fn worker_loop(&self) {
        // macOS demotes CPU-burning threads (priority decay) and prefers
        // E-cores for them; a demoted worker holding a claimed chunk stalls
        // the whole op barrier — measured as the spin pool scaling BACKWARDS
        // with thread count (10 threads slower than 4). Pin the QoS class so
        // the scheduler treats spin-waiting workers as latency-sensitive,
        // the same thing ggml's threadpool does on Apple.
        #[cfg(target_os = "macos")]
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
        }
        let mut seen = 0u64; // last even generation this worker ran
        loop {
            // ── Wait for a new published (even) generation ──────────────
            let mut spins = 0u64;
            let g = loop {
                let g = self.generation.0.load(Ordering::Acquire);
                if g % 2 == 0 && g != seen {
                    break g;
                }
                spins += 1;
                if spins > self.spin_iters {
                    let mut guard = self.sleep.lock().unwrap();
                    self.parked.0.fetch_add(1, Ordering::SeqCst);
                    loop {
                        let g = self.generation.0.load(Ordering::Acquire);
                        if g % 2 == 0 && g != seen {
                            break;
                        }
                        guard = self.wake.wait(guard).unwrap();
                    }
                    self.parked.0.fetch_sub(1, Ordering::SeqCst);
                    break self.generation.0.load(Ordering::Acquire);
                }
                std::hint::spin_loop();
            };
            if g % 2 != 0 || g == seen {
                continue; // woke on an odd/stale gen — re-enter the wait loop
            }

            // ── Enter the op: register FIRST, then the authoritative read ──
            // The registration must be visible to the publisher's drain
            // before we commit to reading the slot; SeqCst on both sides
            // gives the total order the safety argument needs.
            self.active.0.fetch_add(1, Ordering::SeqCst);
            if self.generation.0.load(Ordering::SeqCst) != g {
                // Door closed (odd) or a newer op — back out, retry.
                self.active.0.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            // SAFETY: gen is even and unchanged since we registered in
            // `active`; the next publisher waits for active == 0 before
            // touching the slot, so this copy is of a stable, published op.
            let op = unsafe { *self.op.get() };
            self.execute_blocks(&op);
            self.active.0.fetch_sub(1, Ordering::SeqCst);
            seen = g;
        }
    }

    /// Execute `f(0..n_chunks)` in parallel across the pool + this thread.
    /// Returns after every chunk has run. Chunks must write disjoint data.
    pub fn run<F: Fn(usize) + Sync>(&self, n_chunks: usize, f: &F) {
        unsafe fn thunk<F: Fn(usize) + Sync>(ctx: *const (), c: usize) {
            // SAFETY: ctx is the &F passed to `run`, alive until `run` returns.
            unsafe { (*(ctx as *const F))(c) }
        }
        if n_chunks == 0 {
            return;
        }
        if n_chunks == 1 || self.workers == 0 {
            for c in 0..n_chunks {
                f(c);
            }
            return;
        }

        // The publisher runs the token's SERIAL phases (norms, RoPE,
        // sampling) between ops. If the workers are QoS-pinned but the
        // publisher isn't, macOS runs the one thread doing critical-path
        // work at the LOWEST priority in the process — pin it too, once
        // per thread.
        #[cfg(target_os = "macos")]
        {
            thread_local! {
                static QOS_PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
            }
            QOS_PINNED.with(|p| {
                if !p.get() {
                    unsafe {
                        libc::pthread_set_qos_class_self_np(
                            libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
                            0,
                        );
                    }
                    p.set(true);
                }
            });
        }

        let _g = self.publish.lock().unwrap();
        // ORDER MATTERS — odd bump BEFORE the active drain. A worker joins by
        // registering in `active` and THEN reading `gen` (its authoritative
        // check): if that read precedes this bump it saw the old even gen and
        // its registration is visible to the drain below (we wait it out);
        // if it follows the bump it sees odd and backs out. Draining FIRST
        // left a window — sample active==0, a late worker registers,
        // validates the still-unchanged gen, and copies the slot WHILE we
        // rewrite it (torn copy → dangling ctx → the measured SIGSEGV).
        self.generation.0.fetch_add(1, Ordering::SeqCst); // → odd: door closed
        while self.active.0.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
        // SAFETY: gen is odd (no new joiner passes its authoritative check)
        // and active == 0 (everyone inside has left) — no worker can be
        // reading the slot.
        let participants = self.workers + 1;
        // Block size is TOPOLOGY-dependent (both directions measured):
        // heterogeneous P/E cores (M-series) need block = 1 — an E-core
        // claiming a late multi-chunk block adds a straggler tail to every
        // op (llama-1B lm_head block≈8 measured 62 → 44 tok/s on M4);
        // homogeneous server ARM wants ~3 blocks/participant — per-chunk
        // claim+completion RMW traffic was Thor's 2× regression, and guided
        // blocks took it to +8% OVER rayon. SAPIENT_SPINPOOL_BLOCK overrides.
        let block = block_size_override().unwrap_or({
            if cfg!(target_os = "macos") {
                1
            } else {
                (n_chunks / (3 * participants)).max(1)
            }
        });
        let n_blocks = n_chunks.div_ceil(block);
        let op = OpSlot {
            call: thunk::<F>,
            ctx: f as *const F as *const (),
            n_chunks,
            block,
        };
        unsafe {
            *self.op.get() = op;
        }
        self.next_block.0.store(0, Ordering::Relaxed);
        self.completed.0.store(0, Ordering::Relaxed);
        self.generation.0.fetch_add(1, Ordering::SeqCst); // → even: published
        if self.parked.0.load(Ordering::SeqCst) > 0 {
            let _sg = self.sleep.lock().unwrap();
            self.wake.notify_all();
        }

        // Participate from the calling thread.
        self.execute_blocks(&op);
        // Block writes become visible via the Release increments above.
        while self.completed.0.load(Ordering::Acquire) < n_blocks {
            std::hint::spin_loop();
        }
    }
}

/// The process-global pool: `rayon::current_num_threads() - 1` workers (the
/// publishing thread participates), matching the task budget `gemv_chunk`
/// computes from the same figure. Lazily spawned on first use.
pub fn pool() -> &'static SpinPool {
    static POOL: OnceLock<&'static SpinPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let spins = std::env::var("SAPIENT_SPINPOOL_SPINS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_SPIN_ITERS);
        // SAPIENT_SPINPOOL_WORKERS decouples the pool size from rayon's —
        // the diagnostic lever for isolating pool-internal cost from
        // two-pool mixing (e.g. RAYON_NUM_THREADS=1 + WORKERS=9 runs all
        // GEMV parallelism on the spin pool alone).
        let workers = std::env::var("SAPIENT_SPINPOOL_WORKERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| rayon::current_num_threads().saturating_sub(1));
        SpinPool::new(workers, spins)
    })
}

/// `SAPIENT_SPINPOOL_BLOCK`: fixed chunks-per-claimed-block override for the
/// guided scheduler (topology experiments).
fn block_size_override() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SAPIENT_SPINPOOL_BLOCK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
    })
}

/// Worker threads + the participating publisher — the parallelism the task
/// count should be sized for when the pool is active (`gemv_chunk` uses it).
pub fn parallelism() -> usize {
    pool().workers + 1
}

/// Whether the spin pool should be used for this dispatch. Off when
/// `SAPIENT_SPINPOOL=0` (A/B lever / escape hatch) and while the thermal
/// governor is shedding cores — parked rayon workers shed heat, spinning
/// workers don't, so governed decode must stay on the rayon path.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| {
        std::env::var("SAPIENT_SPINPOOL")
            .map(|v| v != "0")
            // Measurement-driven default (2026-07-10, guided v4): the win
            // scales with thread count — more per-token fork/join tax to
            // reclaim. M4 +7.7% (llama-1B) / +0.9% (qwen); Thor 14-core
            // +5.3%; Pi 5 4-core −4% (at ~100 ms/token there is no tax to
            // reclaim). So: ON for macOS and for Linux/aarch64 at ≥ 8
            // threads (Thor in, Pi out); everything else (x86, Windows —
            // unmeasured) stays opt-in. SAPIENT_SPINPOOL=1/0 overrides.
            .unwrap_or_else(|_| {
                cfg!(target_os = "macos")
                    || (cfg!(all(target_os = "linux", target_arch = "aarch64"))
                        && rayon::current_num_threads() >= 8)
            })
    });
    on && crate::thermal::effective_threads() >= rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn runs_every_chunk_exactly_once() {
        let pool = pool();
        for round in 0..200 {
            let n = 1 + (round % 61);
            let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            pool.run(n, &|c| {
                hits[c].fetch_add(1, Ordering::SeqCst);
            });
            for (c, h) in hits.iter().enumerate() {
                assert_eq!(h.load(Ordering::SeqCst), 1, "round {round} chunk {c}");
            }
        }
    }

    #[test]
    fn concurrent_publishers_serialize() {
        let pool = pool();
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..100 {
                        let hits: Vec<AtomicU32> = (0..37).map(|_| AtomicU32::new(0)).collect();
                        pool.run(37, &|c| {
                            hits[c].fetch_add(1, Ordering::SeqCst);
                        });
                        assert!(hits.iter().all(|h| h.load(Ordering::SeqCst) == 1));
                    }
                });
            }
        });
    }

    #[test]
    fn survives_park_and_wake() {
        let pool = pool();
        let hits: Vec<AtomicU32> = (0..16).map(|_| AtomicU32::new(0)).collect();
        pool.run(16, &|c| {
            hits[c].fetch_add(1, Ordering::SeqCst);
        });
        // Sleep well past any spin budget so workers park, then dispatch again.
        std::thread::sleep(std::time::Duration::from_millis(300));
        let hits2: Vec<AtomicU32> = (0..16).map(|_| AtomicU32::new(0)).collect();
        pool.run(16, &|c| {
            hits2[c].fetch_add(1, Ordering::SeqCst);
        });
        assert!(hits
            .iter()
            .chain(hits2.iter())
            .all(|h| h.load(Ordering::SeqCst) == 1));
    }
}

#[cfg(test)]
mod perf_probe {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::time::Instant;

    // Ground-truth probe: parallel speedup of the pool itself, outside the
    // engine. Run: SAPIENT_SPINPOOL_WORKERS=9 cargo test -p sapient-backends-cpu \
    //   --release --lib perf_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn pool_speedup_probe() {
        let pool = pool();
        let n_chunks = 40usize;
        let work_per_chunk = 400_000u64;
        let sink: Vec<AtomicU64> = (0..n_chunks).map(|_| AtomicU64::new(0)).collect();
        let busy = |c: usize| {
            let mut x = c as u64 ^ 0x9e3779b97f4a7c15;
            for i in 0..work_per_chunk {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
            }
            sink[c].store(x, Ordering::Relaxed);
        };
        // Warm both paths once.
        pool.run(n_chunks, &busy);
        for c in 0..n_chunks {
            busy(c);
        }

        let t = Instant::now();
        for _ in 0..50 {
            for c in 0..n_chunks {
                busy(c);
            }
        }
        let serial = t.elapsed();

        let t = Instant::now();
        for _ in 0..50 {
            pool.run(n_chunks, &busy);
        }
        let par = t.elapsed();
        println!(
            "workers={} serial={serial:?} pool={par:?} speedup={:.2}x",
            pool.workers,
            serial.as_secs_f64() / par.as_secs_f64()
        );

        // Tiny-op dispatch overhead: near-empty chunks, 10k ops.
        let t = Instant::now();
        for _ in 0..10_000 {
            pool.run(n_chunks, &|c| {
                sink[c].store(c as u64, Ordering::Relaxed);
            });
        }
        println!(
            "10k near-empty ops: {:?} ({:.1} µs/op)",
            t.elapsed(),
            t.elapsed().as_secs_f64() * 1e6 / 10_000.0
        );
    }
}

#[cfg(test)]
mod stress {
    use super::*;
    use std::sync::atomic::AtomicU64;

    // Reproducer class for the op-boundary ABA: a tiny spin budget makes
    // workers park/wake around every op, and back-to-back ops with
    // heap-owned closure state make a stale/torn slot read fatal (the
    // SIGSEGV this test exists to prevent regressing).
    #[test]
    fn rapid_ops_with_constant_parking() {
        let pool = SpinPool::new(4, 50); // parks after ~50 spins
        for round in 0..5_000 {
            let n = 2 + (round % 13);
            let payload: Vec<u64> = (0..n as u64).map(|v| v + round as u64).collect();
            let acc: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            pool.run(n, &|c| {
                acc[c].store(payload[c] * 2, Ordering::SeqCst);
            });
            for c in 0..n {
                assert_eq!(
                    acc[c].load(Ordering::SeqCst),
                    payload[c] * 2,
                    "round {round} chunk {c}"
                );
            }
        }
    }
}
