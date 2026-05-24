//! `BatchScheduler` trait and implementations.

use std::time::Duration;

use crate::request::{Batch, Request};
use sapient_core::error::Result;

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Determines when and how to form batches from a stream of requests.
pub trait BatchScheduler: Send + Sync + std::fmt::Debug {
    /// Submit a request to the scheduler.
    fn submit(&mut self, req: Request);

    /// Try to form a batch.  Returns `None` if not enough requests yet.
    fn try_form_batch(&mut self) -> Option<Batch>;

    /// Force-flush any pending requests into a batch (e.g., on shutdown).
    fn flush(&mut self) -> Option<Batch>;
}

// ── StaticBatchScheduler ──────────────────────────────────────────────────────

/// Forms batches of exactly `batch_size` requests.  Pads the last batch if
/// `pad_to_size` is true.
#[derive(Debug)]
pub struct StaticBatchScheduler {
    batch_size: usize,
    queue: Vec<Request>,
}

impl StaticBatchScheduler {
    pub fn new(batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        Self { batch_size, queue: Vec::new() }
    }
}

impl BatchScheduler for StaticBatchScheduler {
    fn submit(&mut self, req: Request) {
        self.queue.push(req);
    }

    fn try_form_batch(&mut self) -> Option<Batch> {
        if self.queue.len() >= self.batch_size {
            let batch: Vec<Request> = self.queue.drain(..self.batch_size).collect();
            Some(Batch::new(batch))
        } else {
            None
        }
    }

    fn flush(&mut self) -> Option<Batch> {
        if self.queue.is_empty() {
            None
        } else {
            let batch = self.queue.drain(..).collect();
            Some(Batch::new(batch))
        }
    }
}

// ── DynamicBatchScheduler ─────────────────────────────────────────────────────

/// Forms batches using a time-window strategy:
///   - Emit a batch when `max_batch_size` is reached, OR
///   - When `max_wait` has elapsed since the first request in the current window.
///
/// This implements the micro-batching strategy from Inference Engineering:
/// trade a small, bounded latency increase for significant throughput gain.
#[derive(Debug)]
pub struct DynamicBatchScheduler {
    max_batch_size: usize,
    max_wait: Duration,
    queue: Vec<Request>,
    window_start: Option<std::time::Instant>,
}

impl DynamicBatchScheduler {
    pub fn new(max_batch_size: usize, max_wait: Duration) -> Self {
        Self {
            max_batch_size,
            max_wait,
            queue: Vec::new(),
            window_start: None,
        }
    }

    /// Standard 5ms window as described in the spec.
    pub fn standard() -> Self {
        Self::new(64, Duration::from_millis(5))
    }
}

impl BatchScheduler for DynamicBatchScheduler {
    fn submit(&mut self, req: Request) {
        if self.window_start.is_none() {
            self.window_start = Some(std::time::Instant::now());
        }
        self.queue.push(req);
    }

    fn try_form_batch(&mut self) -> Option<Batch> {
        let should_emit = self.queue.len() >= self.max_batch_size
            || self.window_start.map_or(false, |t| t.elapsed() >= self.max_wait);

        if should_emit && !self.queue.is_empty() {
            let size = self.queue.len().min(self.max_batch_size);
            let batch: Vec<Request> = self.queue.drain(..size).collect();
            if self.queue.is_empty() {
                self.window_start = None;
            }
            metrics::histogram!("sapient.batch.size").record(batch.len() as f64);
            Some(Batch::new(batch))
        } else {
            None
        }
    }

    fn flush(&mut self) -> Option<Batch> {
        if self.queue.is_empty() {
            None
        } else {
            self.window_start = None;
            let batch = self.queue.drain(..).collect();
            Some(Batch::new(batch))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_req() -> Request {
        Request::new(HashMap::new())
    }

    #[test]
    fn static_scheduler_forms_batch() {
        let mut sched = StaticBatchScheduler::new(2);
        sched.submit(make_req());
        assert!(sched.try_form_batch().is_none());
        sched.submit(make_req());
        let b = sched.try_form_batch().unwrap();
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn dynamic_scheduler_timeout() {
        let mut sched = DynamicBatchScheduler::new(8, Duration::from_millis(1));
        sched.submit(make_req());
        std::thread::sleep(Duration::from_millis(5));
        let b = sched.try_form_batch().unwrap();
        assert_eq!(b.len(), 1);
    }
}
