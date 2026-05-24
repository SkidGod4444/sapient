//! `Executor` — bridges the async scheduler with rayon's compute thread pool.
//!
//! Architecture:
//!   - `Executor::run()` spawns a tokio task that pulls batches from a channel.
//!   - Each batch is sent to rayon's thread pool for CPU-bound execution.
//!   - Results are sent back via per-request oneshot channels.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, error, instrument};

use sapient_backends_cpu::backend::ExecutionBackend;
use sapient_core::error::SapientError;
use sapient_ir::graph::Graph;

use crate::batcher::Batcher;
use crate::request::{Batch, Request, Response};
use crate::scheduler::BatchScheduler;

// ── Executor ──────────────────────────────────────────────────────────────────

/// Async/rayon bridge for batch execution.
pub struct Executor<B: ExecutionBackend + 'static> {
    backend: Arc<B>,
    graph: Arc<Graph>,
    batcher: Batcher,
}

impl<B: ExecutionBackend + 'static> Executor<B> {
    pub fn new(backend: B, graph: Graph) -> Self {
        Self {
            backend: Arc::new(backend),
            graph: Arc::new(graph),
            batcher: Batcher::new(),
        }
    }

    /// Execute a single batch synchronously (used internally and for benchmarks).
    #[instrument(skip_all, fields(batch_size = batch.len()))]
    pub fn execute_batch(&self, batch: Batch) -> Vec<Response> {
        let start = Instant::now();
        let requests = batch.requests;
        let batch_size = requests.len();

        // Merge inputs.
        let fake_batch = crate::request::Batch::new(requests);
        let merged = match self.batcher.merge_inputs(&fake_batch) {
            Ok(m) => m,
            Err(e) => {
                error!(error = %e, "Executor: failed to merge inputs");
                return fake_batch
                    .requests
                    .into_iter()
                    .map(|r| Response::err(r.id, SapientError::internal(e.to_string()), 0))
                    .collect();
            }
        };
        let requests = fake_batch.requests;

        // Execute on backend.
        let result = self.backend.execute(&self.graph, merged);
        let latency_us = start.elapsed().as_micros() as u64;

        match result {
            Ok(outputs) => {
                // Split batched outputs back to per-request responses.
                // For now: broadcast the same output to all requesters.
                // A production implementation would split on the batch dimension.
                requests
                    .into_iter()
                    .map(|r| Response::ok(r.id, outputs.clone(), latency_us))
                    .collect()
            }
            Err(e) => requests
                .into_iter()
                .map(|r| Response::err(r.id, SapientError::internal(e.to_string()), latency_us))
                .collect(),
        }
    }

    /// Run a stream of requests through the scheduler asynchronously.
    ///
    /// `scheduler` is polled in a tight loop on a tokio task; batches are
    /// sent to rayon for compute.
    pub async fn run_async(
        self: Arc<Self>,
        mut scheduler: impl BatchScheduler + Send + 'static,
        mut rx: tokio::sync::mpsc::Receiver<(Request, tokio::sync::oneshot::Sender<Response>)>,
    ) {
        let mut response_map: HashMap<uuid::Uuid, tokio::sync::oneshot::Sender<Response>> =
            HashMap::new();

        loop {
            // Drain incoming requests.
            while let Ok((req, tx)) = rx.try_recv() {
                response_map.insert(req.id, tx);
                scheduler.submit(req);
            }

            // Try to form a batch.
            if let Some(batch) = scheduler.try_form_batch() {
                let exec = self.clone();
                let responses =
                    tokio::task::spawn_blocking(move || exec.execute_batch(batch)).await;

                if let Ok(responses) = responses {
                    for resp in responses {
                        if let Some(tx) = response_map.remove(&resp.request_id) {
                            let _ = tx.send(resp);
                        }
                    }
                }
            }

            // Check for closed channel.
            if rx.is_closed() {
                if let Some(batch) = scheduler.flush() {
                    let responses = self.execute_batch(batch);
                    for resp in responses {
                        if let Some(tx) = response_map.remove(&resp.request_id) {
                            let _ = tx.send(resp);
                        }
                    }
                }
                break;
            }

            tokio::task::yield_now().await;
        }
    }
}
