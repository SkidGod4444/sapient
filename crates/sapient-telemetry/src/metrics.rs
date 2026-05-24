//! Metrics helpers — thin wrappers for per-op counters and histograms.

pub use metrics::{counter, gauge, histogram};

/// Record a gauge for current memory pool usage.
pub fn record_pool_usage(used_bytes: usize, capacity_bytes: usize) {
    metrics::gauge!("sapient.pool.used_bytes").set(used_bytes as f64);
    metrics::gauge!("sapient.pool.capacity_bytes").set(capacity_bytes as f64);
    if capacity_bytes > 0 {
        metrics::gauge!("sapient.pool.utilization")
            .set(used_bytes as f64 / capacity_bytes as f64);
    }
}

/// Record queue depth (for latency analysis).
pub fn record_queue_depth(depth: usize) {
    metrics::gauge!("sapient.scheduler.queue_depth").set(depth as f64);
}
