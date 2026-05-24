//! `Telemetry` trait and implementations.

use std::time::{Duration, Instant};
use tracing::{debug, info};

// ── TelemetryConfig ───────────────────────────────────────────────────────────

/// Configuration for telemetry output.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub mode: TelemetryMode,
    pub otlp_endpoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelemetryMode {
    None,
    Console,
    Otlp,
    Prometheus { port: u16 },
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self { mode: TelemetryMode::None, otlp_endpoint: None }
    }
}

// ── Telemetry trait ───────────────────────────────────────────────────────────

/// Instrumentation interface for the SAPIENT engine.
///
/// All methods have a default no-op implementation so callers compile with
/// zero overhead when using `NoOpTelemetry`.
pub trait Telemetry: Send + Sync + std::fmt::Debug {
    /// Called at the start of a graph execution.
    fn on_execution_start(&self, _graph_name: &str, _batch_size: usize) {}

    /// Called at the end of a graph execution.
    fn on_execution_end(&self, _graph_name: &str, _duration: Duration) {}

    /// Per-operator latency.
    fn on_op_executed(&self, _op_name: &str, _duration: Duration) {}

    /// Memory allocation event.
    fn on_alloc(&self, _bytes: usize, _device: &str) {}

    /// Memory free event.
    fn on_free(&self, _bytes: usize, _device: &str) {}

    /// Batch formed.
    fn on_batch_formed(&self, _size: usize, _wait_us: u64) {}

    /// Request queued.
    fn on_request_queued(&self) {}

    /// Request completed.
    fn on_request_completed(&self, _latency_us: u64) {}
}

// ── NoOpTelemetry ─────────────────────────────────────────────────────────────

/// Zero-overhead telemetry (default).  All methods are no-ops, so the
/// compiler should eliminate all call sites completely.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpTelemetry;

impl Telemetry for NoOpTelemetry {}

// ── ConsoleTelemetry ──────────────────────────────────────────────────────────

/// Prints telemetry events to stderr via `tracing`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConsoleTelemetry;

impl Telemetry for ConsoleTelemetry {
    fn on_execution_start(&self, graph_name: &str, batch_size: usize) {
        debug!(graph = graph_name, batch = batch_size, "▶ execution start");
    }

    fn on_execution_end(&self, graph_name: &str, duration: Duration) {
        let us = duration.as_micros() as f64;
        info!(graph = graph_name, latency_us = us, "✓ execution end");
        metrics::histogram!("sapient.execution.latency_us").record(us);
    }

    fn on_op_executed(&self, op_name: &str, duration: Duration) {
        let us = duration.as_micros() as f64;
        debug!(op = op_name, latency_us = us, "  op executed");
        metrics::histogram!("sapient.op.latency_us").record(us);
    }

    fn on_alloc(&self, bytes: usize, _device: &str) {
        metrics::counter!("sapient.alloc.bytes").increment(bytes as u64);
    }

    fn on_free(&self, bytes: usize, _device: &str) {
        metrics::counter!("sapient.free.bytes").increment(bytes as u64);
    }

    fn on_batch_formed(&self, size: usize, wait_us: u64) {
        debug!(batch_size = size, wait_us = wait_us, "batch formed");
        metrics::histogram!("sapient.batch.size").record(size as f64);
        metrics::histogram!("sapient.batch.wait_us").record(wait_us as f64);
    }

    fn on_request_queued(&self) {
        metrics::counter!("sapient.requests.queued").increment(1u64);
    }

    fn on_request_completed(&self, latency_us: u64) {
        metrics::histogram!("sapient.request.latency_us").record(latency_us as f64);
    }
}
