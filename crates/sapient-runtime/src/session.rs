//! `InferenceSession` — the primary user-facing API.
//!
//! A session owns a compiled graph, a backend, a scheduler, and a telemetry
//! hook.  It exposes `run()` for single inference and `run_batch()` for
//! explicit batching.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, info, instrument};

use sapient_core::{Tensor};
use sapient_core::error::{Result, SapientError};
use sapient_backends_cpu::backend::{CpuBackend, ExecutionBackend};
use sapient_scheduler::{
    Batcher, DynamicBatchScheduler, Executor, Request, Response,
};
use sapient_telemetry::{ConsoleTelemetry, NoOpTelemetry, Telemetry};

use crate::model::{Model, ModelConfig};

// ── SessionOptions ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SessionOptions {
    /// Pool size for CPU backend (bytes).
    pub cpu_pool_bytes: usize,
    /// Enable console telemetry.
    pub telemetry: bool,
    /// Max dynamic batch size.
    pub max_batch_size: usize,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            cpu_pool_bytes: 256 * 1024 * 1024, // 256 MiB
            telemetry: false,
            max_batch_size: 64,
        }
    }
}

// ── InferenceSession ──────────────────────────────────────────────────────────

/// Thread-safe inference session.
pub struct InferenceSession {
    model:    Arc<Model>,
    backend:  Arc<dyn ExecutionBackend>,
    telemetry: Arc<dyn Telemetry>,
    opts:     SessionOptions,
}

impl InferenceSession {
    /// Create a new session from a loaded model.
    pub fn new(model: Model, opts: SessionOptions) -> Result<Self> {
        let backend: Arc<dyn ExecutionBackend> =
            Arc::new(CpuBackend::new(opts.cpu_pool_bytes));

        let telemetry: Arc<dyn Telemetry> = if opts.telemetry {
            Arc::new(ConsoleTelemetry)
        } else {
            Arc::new(NoOpTelemetry)
        };

        Ok(Self {
            model: Arc::new(model),
            backend,
            telemetry,
            opts,
        })
    }

    /// Run a single forward pass.
    ///
    /// `inputs` maps input name → tensor.
    #[instrument(skip_all, fields(model = %self.model.name))]
    pub fn run(&self, inputs: HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        let start = Instant::now();
        self.telemetry.on_request_queued();
        self.telemetry.on_execution_start(&self.model.name, 1);

        let result = self.backend.execute(&self.model.graph, inputs);

        let dur = start.elapsed();
        self.telemetry.on_execution_end(&self.model.name, dur);
        self.telemetry.on_request_completed(dur.as_micros() as u64);

        result
    }

    /// Run a batch of input maps, merging and splitting automatically.
    ///
    /// Returns one `Vec<Tensor>` per request.
    pub fn run_batch(
        &self,
        batch: Vec<HashMap<String, Tensor>>,
    ) -> Result<Vec<Vec<Tensor>>> {
        let batch_size = batch.len();
        let start = Instant::now();
        self.telemetry.on_batch_formed(batch_size, 0);

        // Convert to Requests and use the batcher to merge.
        let requests: Vec<Request> = batch.into_iter().map(Request::new).collect();
        let request_batch = sapient_scheduler::request::Batch::new(requests);

        let batcher = Batcher::new();
        let merged = batcher.merge_inputs(&request_batch)?;

        let outputs = self.backend.execute(&self.model.graph, merged)?;

        let dur = start.elapsed();
        self.telemetry.on_execution_end(&self.model.name, dur);

        // Broadcast outputs to each requester (proper splitting is model-specific).
        Ok(vec![outputs; batch_size])
    }

    /// Reference to the underlying model.
    pub fn model(&self) -> &Model { &self.model }

    /// Backend name.
    pub fn backend_name(&self) -> &str { self.backend.name() }
}

impl std::fmt::Debug for InferenceSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceSession")
            .field("model", &self.model.name)
            .field("backend", &self.backend.name())
            .finish()
    }
}
