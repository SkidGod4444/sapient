//! KV-cache for incremental autoregressive decoding.
//!
//! Each decoder layer maintains a `LayerKVCache` that grows as tokens
//! are generated. At decode step `t`, the cache holds K and V for
//! positions [0, t-1], so only the new token's QKV needs to be computed.

use sapient_core::Tensor;
use std::collections::HashMap;

// ── LayerKVCache ──────────────────────────────────────────────────────────────

/// Cached key and value tensors for one decoder layer.
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Accumulated keys — shape grows as (batch, n_kv_heads, seq_k, head_dim).
    pub keys: Vec<Tensor>,
    /// Accumulated values — same shape.
    pub values: Vec<Tensor>,
}

impl LayerKVCache {
    pub fn empty() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Append a new key/value slice and return the current sequence length.
    pub fn append(&mut self, k: Tensor, v: Tensor) -> usize {
        self.keys.push(k);
        self.values.push(v);
        self.keys.len()
    }

    /// Current cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.keys.len()
    }

    /// Clear the cache (e.g., start a new conversation).
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}

// ── KVCache ───────────────────────────────────────────────────────────────────

/// Full KV cache for all decoder layers.
#[derive(Debug, Clone)]
pub struct KVCache {
    layers: Vec<LayerKVCache>,
}

impl KVCache {
    /// Create an empty KV cache for `n_layers` decoder layers.
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerKVCache::empty()).collect(),
        }
    }

    pub fn layer(&self, idx: usize) -> &LayerKVCache {
        &self.layers[idx]
    }
    pub fn layer_mut(&mut self, idx: usize) -> &mut LayerKVCache {
        &mut self.layers[idx]
    }

    /// Sequence length of the first layer (all layers have the same length).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    /// Clear the entire cache (new conversation / context reset).
    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
    }

    /// Number of layers in the cache.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_grows() {
        let mut cache = KVCache::new(4);
        assert_eq!(cache.seq_len(), 0);
        let k = Tensor::zeros(vec![1, 2, 1, 64], sapient_core::DType::F32).unwrap();
        let v = Tensor::zeros(vec![1, 2, 1, 64], sapient_core::DType::F32).unwrap();
        cache.layer_mut(0).append(k.clone(), v.clone());
        cache.layer_mut(0).append(k, v);
        assert_eq!(cache.layer(0).seq_len(), 2);
    }

    #[test]
    fn kv_cache_clear() {
        let mut cache = KVCache::new(2);
        let t = Tensor::zeros(vec![1, 1, 1, 64], sapient_core::DType::F32).unwrap();
        cache.layer_mut(0).append(t.clone(), t);
        assert_eq!(cache.seq_len(), 1);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }
}
