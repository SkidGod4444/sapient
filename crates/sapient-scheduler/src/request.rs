// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `Request` and `Response` types for inference scheduling.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use sapient_core::error::SapientError;
use sapient_core::Tensor;

// ── Request ───────────────────────────────────────────────────────────────────

/// A single inference request.
#[derive(Debug)]
pub struct Request {
    /// Unique request identifier.
    pub id: Uuid,
    /// Named input tensors.
    pub inputs: HashMap<String, Tensor>,
    /// Optional hard deadline — exceeded requests return `DeadlineExceeded`.
    pub deadline: Option<Instant>,
    /// Priority (0 = lowest, 255 = highest).
    pub priority: u8,
    /// If true, use the streaming (low-latency) path.
    pub stream: bool,
}

impl Request {
    pub fn new(inputs: HashMap<String, Tensor>) -> Self {
        Self {
            id: Uuid::new_v4(),
            inputs,
            deadline: None,
            priority: 0,
            stream: false,
        }
    }

    pub fn with_deadline(mut self, timeout: Duration) -> Self {
        self.deadline = Some(Instant::now() + timeout);
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn is_expired(&self) -> bool {
        self.deadline.map_or(false, |d| Instant::now() > d)
    }
}

// ── Response ──────────────────────────────────────────────────────────────────

/// Result of executing a single request.
#[derive(Debug)]
pub struct Response {
    pub request_id: Uuid,
    pub outputs: Result<Vec<Tensor>, SapientError>,
    /// Wall-clock time from request creation to response ready.
    pub latency_us: u64,
}

impl Response {
    pub fn ok(request_id: Uuid, outputs: Vec<Tensor>, latency_us: u64) -> Self {
        Self {
            request_id,
            outputs: Ok(outputs),
            latency_us,
        }
    }

    pub fn err(request_id: Uuid, error: SapientError, latency_us: u64) -> Self {
        Self {
            request_id,
            outputs: Err(error),
            latency_us,
        }
    }
}

// ── Batch ─────────────────────────────────────────────────────────────────────

/// A formed batch of requests ready for execution.
pub struct Batch {
    pub requests: Vec<Request>,
    /// Formed at this instant (for metrics).
    pub formed_at: Instant,
}

impl Batch {
    pub fn new(requests: Vec<Request>) -> Self {
        Self {
            requests,
            formed_at: Instant::now(),
        }
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}
