//! `Batcher` — collects requests, applies padding, bucketing, and forms input
//! tensors ready for batched graph execution.

use std::collections::HashMap;

use sapient_core::error::{Result, SapientError};
use sapient_core::{DType, Shape, Tensor};

use crate::request::{Batch, Request};

// ── Batcher ───────────────────────────────────────────────────────────────────

/// Combines a batch of single requests into merged input tensors.
///
/// For sequence models the batcher groups by similar lengths (bucketing) and
/// left-pads shorter sequences.  For fixed-shape models it simply stacks.
pub struct Batcher {
    /// Bucket boundaries for sequence batching.
    buckets: Vec<usize>,
}

impl Batcher {
    /// Create with no bucketing (all shapes stacked directly).
    pub fn new() -> Self {
        Self { buckets: vec![] }
    }

    /// Create with sequence-length buckets (e.g., [64, 128, 256, 512]).
    pub fn with_buckets(buckets: Vec<usize>) -> Self {
        let mut b = buckets;
        b.sort_unstable();
        Self { buckets: b }
    }

    /// Pad a 1-D sequence to `target_len` with `pad_value` (left-pad).
    pub fn left_pad_f32(seq: &[f32], target_len: usize, pad_value: f32) -> Vec<f32> {
        if seq.len() >= target_len {
            seq[seq.len() - target_len..].to_vec()
        } else {
            let mut padded = vec![pad_value; target_len - seq.len()];
            padded.extend_from_slice(seq);
            padded
        }
    }

    /// Merge a batch of single-input requests into one stacked tensor per key.
    ///
    /// All requests must have the same input keys and compatible shapes (only
    /// the batch dimension is stacked).
    pub fn merge_inputs(&self, batch: &Batch) -> Result<HashMap<String, Tensor>> {
        if batch.is_empty() {
            return Ok(HashMap::new());
        }

        // Collect all input keys.
        let keys: Vec<String> = batch.requests[0].inputs.keys().cloned().collect();
        let mut merged = HashMap::new();

        for key in &keys {
            let tensors: Vec<&Tensor> = batch
                .requests
                .iter()
                .map(|r| {
                    r.inputs.get(key).ok_or_else(|| {
                        SapientError::internal(format!("request missing input key '{key}'"))
                    })
                })
                .collect::<Result<_>>()?;

            // Find target length (for sequence dimension = last dim or dim 1).
            let target_len = self.bucket_for(tensors.iter().map(|t| t.numel()).max().unwrap_or(0));

            // Determine per-element shape (all dims except batch).
            let elem_shape = tensors[0].shape().dims().to_vec();

            // Stack tensors along new batch dimension 0.
            let batch_size = tensors.len();
            let elem_numel: usize = elem_shape.iter().product();

            let mut stacked = Vec::with_capacity(batch_size * elem_numel);
            for t in &tensors {
                let data = t.as_f32_slice();
                if data.len() == elem_numel {
                    stacked.extend_from_slice(data);
                } else {
                    // Pad to elem_numel (sequence padding).
                    let padded = Self::left_pad_f32(data, elem_numel, 0.0);
                    stacked.extend_from_slice(&padded);
                }
            }

            let mut out_dims = vec![batch_size];
            out_dims.extend_from_slice(&elem_shape);
            merged.insert(
                key.clone(),
                Tensor::from_f32(&stacked, Shape::new(out_dims))?,
            );
        }

        Ok(merged)
    }

    fn bucket_for(&self, len: usize) -> usize {
        for &b in &self.buckets {
            if b >= len {
                return b;
            }
        }
        len // No bucket ≥ len; use exact length.
    }
}

impl Default for Batcher {
    fn default() -> Self {
        Self::new()
    }
}
