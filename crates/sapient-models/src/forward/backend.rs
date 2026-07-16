// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Backend dispatch for native LLM forward passes.

use anyhow::Result;
use sapient_core::Tensor;

use super::common;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LlmBackendKind {
    Cpu,
    Metal,
    /// Cross-platform GPU via wgpu/WGSL (Vulkan/DX12/Metal — Intel/AMD/Nvidia/Apple).
    Wgpu,
    #[default]
    Auto,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacGpuSupport {
    pub available: bool,
    pub backend: &'static str,
    pub reason: &'static str,
}

pub fn mac_gpu_support() -> MacGpuSupport {
    MlxLlmOps::support()
}

impl std::str::FromStr for LlmBackendKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "wgpu" | "vulkan" | "dx12" | "gpu" => Ok(Self::Wgpu),
            "auto" => Ok(Self::Auto),
            other => anyhow::bail!("unsupported generation backend '{other}'"),
        }
    }
}

impl std::fmt::Display for LlmBackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Metal => write!(f, "metal"),
            Self::Wgpu => write!(f, "wgpu"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

pub trait LlmBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn linear_3d(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor>;

    /// Linear projection with an optional bias added over the last dimension.
    /// Backend-agnostic: computes `linear_3d` then folds in the bias on the host,
    /// so every backend gets correct bias handling for free.
    fn linear_3d_bias(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let y = self.linear_3d(x, weight)?;
        match bias {
            None => Ok(y),
            Some(b) => common::add_bias_last_dim(&y, b),
        }
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor>;
    fn silu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu(&self, x: &Tensor) -> Result<Tensor>;
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn apply_rope_positions(&self, x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor>;

    /// RoPE over only the first `rotary_dim` channels (Phi partial rotary).
    /// Computed on the CPU reference kernel for all backends — it is cheap and
    /// avoids backend-specific partial-rotary support.
    fn apply_rope_partial(
        &self,
        x: &Tensor,
        positions: &[usize],
        base: f32,
        rotary_dim: usize,
    ) -> Result<Tensor> {
        common::apply_rope_partial(x, positions, base, rotary_dim)
    }

    fn gqa_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor>;
    fn logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>>;

    /// Compute logits for ALL positions in the sequence.
    /// Returns `seq_len` vectors each of length `vocab_size`.
    /// Default impl delegates to the CPU reference kernel; backends may override.
    fn all_logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<Vec<f32>>> {
        common::all_logits_from_hidden(hidden, lm_head)
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuLlmBackend;

impl LlmBackend for CpuLlmBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn linear_3d(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        common::linear_3d(x, weight)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        common::rms_norm(x, weight, eps)
    }

    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        common::layer_norm(x, weight, bias, eps)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        common::silu(x)
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        common::gelu(x)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        common::add(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        common::mul(a, b)
    }

    fn apply_rope_positions(&self, x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
        common::apply_rope_positions(x, positions, base)
    }

    fn gqa_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor> {
        common::gqa_attention(q, k, v, n_kv_heads, causal)
    }

    fn logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>> {
        common::logits_from_hidden(hidden, lm_head)
    }
}

#[derive(Debug, Default, Clone)]
pub struct MetalLlmBackend {
    cpu: CpuLlmBackend,
    mlx: MlxLlmOps,
}

impl MetalLlmBackend {
    pub fn is_available() -> bool {
        MlxLlmOps::is_available()
    }
}

/// Debug aid: `SAPIENT_MLX_DISABLE=linear_3d,rms_norm` forces the listed ops
/// onto the CPU reference kernel (`all` disables every MLX op). Lets a
/// wrong-numbers MLX kernel be bisected per-op without rebuilding.
fn mlx_disabled(op: &str) -> bool {
    use std::sync::OnceLock;
    static LIST: OnceLock<Vec<String>> = OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("SAPIENT_MLX_DISABLE")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    });
    list.iter().any(|s| s == op || s == "all")
}

/// Debug aid: `SAPIENT_MLX_VERIFY=1` cross-checks MLX linear results against
/// the CPU reference kernel (see `MetalLlmBackend::linear_3d`).
fn mlx_verify() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SAPIENT_MLX_VERIFY").is_ok_and(|v| v == "1"))
}

impl LlmBackend for MetalLlmBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn linear_3d(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        if mlx_disabled("linear_3d") {
            return self.cpu.linear_3d(x, weight);
        }
        let result = self.mlx
            .linear_3d(x, weight)
            .or_else(|e| {
                tracing::warn!(op = "linear_3d", error = %e, "MLX op failed; using CPU reference kernel");
                self.cpu.linear_3d(x, weight)
            });
        // Debug aid: `SAPIENT_MLX_VERIFY=1` cross-checks the first calls against
        // the CPU reference kernel and prints the divergence per weight tensor.
        if mlx_verify() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALLS: AtomicUsize = AtomicUsize::new(0);
            let n = CALLS.fetch_add(1, Ordering::Relaxed);
            if n < 64 {
                if let (Ok(mlx_t), Ok(cpu_t)) = (&result, self.cpu.linear_3d(x, weight)) {
                    let a = mlx_t.to_f32_cow();
                    let b = cpu_t.to_f32_cow();
                    let mut max_abs = 0f32;
                    let mut norm = 0f64;
                    for (va, vb) in a.iter().zip(b.iter()) {
                        max_abs = max_abs.max((va - vb).abs());
                        norm += (*vb as f64) * (*vb as f64);
                    }
                    let rms = (norm / b.len() as f64).sqrt();
                    eprintln!(
                        "[mlx-verify] linear_3d #{n} w={:?} dtype={:?} max_abs_diff={max_abs:.4} cpu_rms={rms:.4}",
                        weight.shape().dims(),
                        weight.dtype(),
                    );
                }
            }
        }
        result
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        if mlx_disabled("rms_norm") {
            return self.cpu.rms_norm(x, weight, eps);
        }
        self.mlx.rms_norm(x, weight, eps).or_else(|e| {
            tracing::warn!(op = "rms_norm", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.rms_norm(x, weight, eps)
        })
    }

    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        if mlx_disabled("layer_norm") {
            return self.cpu.layer_norm(x, weight, bias, eps);
        }
        self.mlx.layer_norm(x, weight, bias, eps).or_else(|e| {
            tracing::warn!(op = "layer_norm", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.layer_norm(x, weight, bias, eps)
        })
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        if mlx_disabled("silu") {
            return self.cpu.silu(x);
        }
        self.mlx.silu(x).or_else(|e| {
            tracing::warn!(op = "silu", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.silu(x)
        })
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        if mlx_disabled("gelu") {
            return self.cpu.gelu(x);
        }
        self.mlx.gelu(x).or_else(|e| {
            tracing::warn!(op = "gelu", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.gelu(x)
        })
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if mlx_disabled("add") {
            return self.cpu.add(a, b);
        }
        self.mlx.add(a, b).or_else(|e| {
            tracing::warn!(op = "add", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.add(a, b)
        })
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if mlx_disabled("mul") {
            return self.cpu.mul(a, b);
        }
        self.mlx.mul(a, b).or_else(|e| {
            tracing::warn!(op = "mul", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.mul(a, b)
        })
    }

    fn apply_rope_positions(&self, x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
        if mlx_disabled("rope") {
            return self.cpu.apply_rope_positions(x, positions, base);
        }
        self.mlx
            .apply_rope_positions(x, positions, base)
            .or_else(|e| {
                tracing::warn!(op = "rope", error = %e, "MLX op failed; using CPU reference kernel");
                self.cpu.apply_rope_positions(x, positions, base)
            })
    }

    fn gqa_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor> {
        if mlx_disabled("gqa_attention") {
            return self.cpu.gqa_attention(q, k, v, n_kv_heads, causal);
        }
        // Run on the Metal GPU. MlxLlmOps::gqa_attention builds an explicit
        // causal mask for prefill (seq_q > 1) so the KV-cache offset case
        // (seq_q < seq_k at decode) is handled correctly.
        self.mlx
            .gqa_attention(q, k, v, n_kv_heads, causal)
            .or_else(|e| {
                tracing::warn!(op = "gqa_attention", error = %e,
                    "MLX attention failed; falling back to CPU");
                self.cpu.gqa_attention(q, k, v, n_kv_heads, causal)
            })
    }

    fn logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>> {
        if mlx_disabled("logits") {
            return self.cpu.logits_from_hidden(hidden, lm_head);
        }
        self.mlx.logits_from_hidden(hidden, lm_head).or_else(|e| {
            tracing::warn!(op = "logits", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.logits_from_hidden(hidden, lm_head)
        })
    }
}

/// Converts a `Tensor` to an `mlx_rs::Array`, caching the result by buffer pointer so
/// that weight tensors (which have a stable `Arc<CpuBuffer>` address) are uploaded to
/// the GPU exactly once across all tokens, instead of re-converted on every `linear_3d`
/// call.  Activation tensors are ephemeral (different pointer each step) and never
/// accumulate in the cache.
#[cfg(feature = "mlx")]
type MlxWeightCache =
    std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<usize, mlx_rs::Array>>>;

#[cfg(feature = "mlx")]
type MlxQuantCache = std::sync::Arc<
    parking_lot::Mutex<
        std::collections::HashMap<usize, (mlx_rs::Array, mlx_rs::Array, mlx_rs::Array)>,
    >,
>;

#[derive(Clone)]
struct MlxLlmOps {
    /// Shared weight cache: `buffer_ptr → GPU Array`. Clones share the same cache so
    /// the MetalLlmBackend (which clones MlxLlmOps per call site) reuses uploads.
    #[cfg(feature = "mlx")]
    cache: MlxWeightCache,
    /// Quantized weight cache: buffer_ptr → (Wq, scales, biases) in MLX Q4 format.
    /// Populated lazily on first use; shared across MlxLlmOps clones.
    #[cfg(feature = "mlx")]
    quant_cache: MlxQuantCache,
}

impl std::fmt::Debug for MlxLlmOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxLlmOps").finish()
    }
}

// Can't use #[derive(Default)] because the `cache` and `quant_cache` fields are cfg-gated on
// the `mlx` feature and derive doesn't understand that.
#[allow(clippy::derivable_impls)]
impl Default for MlxLlmOps {
    fn default() -> Self {
        Self {
            #[cfg(feature = "mlx")]
            cache: std::sync::Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
            #[cfg(feature = "mlx")]
            quant_cache: std::sync::Arc::new(parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }
}

#[cfg(target_os = "macos")]
impl MlxLlmOps {
    fn support() -> MacGpuSupport {
        #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
        {
            MacGpuSupport {
                available: true,
                backend: "mlx",
                reason: "Apple Silicon with MLX feature enabled",
            }
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64", not(feature = "mlx")))]
        {
            MacGpuSupport {
                available: false,
                backend: "cpu",
                reason: "Apple Silicon detected, but Sapient was built without the sapient-models/mlx feature",
            }
        }

        #[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
        {
            MacGpuSupport {
                available: false,
                backend: "cpu",
                reason: "MLX GPU execution requires Apple Silicon; Intel Macs use CPU",
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            MacGpuSupport {
                available: false,
                backend: "cpu",
                reason: "MLX GPU execution is only available on macOS",
            }
        }
    }

    fn is_available() -> bool {
        Self::support().available
    }

    #[cfg(feature = "mlx")]
    fn to_shape(dims: &[usize]) -> Result<Vec<i32>> {
        dims.iter()
            .map(|&d| {
                i32::try_from(d)
                    .map_err(|_| anyhow::anyhow!("shape dimension too large for MLX: {d}"))
            })
            .collect()
    }

    /// Convert a `Tensor` to an `mlx_rs::Array`.
    ///
    /// For tensors with stable buffer addresses (weights stored in `Arc<CpuBuffer>`) this
    /// returns a cached copy — the upload to GPU happens only on the first call.  Activations
    /// have a fresh allocation each decode step so they are converted without caching.
    #[cfg(feature = "mlx")]
    fn to_array(&self, tensor: &Tensor) -> Result<mlx_rs::Array> {
        let ptr_key = tensor.as_bytes().as_ptr() as usize;

        // Fast path: already uploaded.
        {
            let guard = self.cache.lock();
            if let Some(arr) = guard.get(&ptr_key) {
                return Ok(arr.clone());
            }
        }

        // Slow path: convert and cache.
        let shape = Self::to_shape(tensor.shape().dims())?;
        let data = tensor.to_f32_cow();
        let arr = mlx_rs::Array::from_slice(data.as_ref(), &shape);

        // Only cache when the tensor looks like a weight (> 1 KiB and numel > 256).
        // This avoids caching tiny scalars or ephemeral activation buffers that happen
        // to share the same size.
        let numel = tensor.numel();
        if numel > 256 {
            self.cache.lock().insert(ptr_key, arr.clone());
        }

        Ok(arr)
    }

    /// Convert without caching — for activation tensors created fresh each step.
    ///
    /// `to_f32_cow` on a non-contiguous tensor (e.g. KV-cache slices from
    /// `slice_axis`) returns the full backing buffer, which is far larger than the
    /// tensor's logical `numel`. `Array::from_slice` asserts `data.len == shape.product`
    /// and panics. We limit the slice to `numel` elements to prevent the assert.
    ///
    /// For contiguous tensors (the common case for activations and weights) the
    /// buffer length already equals `numel` so this limit is a no-op.
    #[cfg(feature = "mlx")]
    fn to_array_uncached(tensor: &Tensor) -> Result<mlx_rs::Array> {
        let shape = Self::to_shape(tensor.shape().dims())?;
        let numel = tensor.numel();
        let cow = tensor.to_f32_cow();
        // Limit to the logical element count so non-contiguous view tensors (KV
        // cache slices) don't overflow the MLX assert.
        let data = &cow[..numel.min(cow.len())];
        Ok(mlx_rs::Array::from_slice(data, &shape))
    }

    /// Convert a weight tensor to MLX's native Q4 quantized format, caching the result.
    ///
    /// Uses `mlx_rs::ops::quantize()` (group_size=64, bits=4) — the same format
    /// mlx-lm uses.  The resulting (Wq, scales, biases) tuple is stored in
    /// `quant_cache` keyed by buffer pointer so quantization runs only once per weight.
    ///
    /// Only applicable for tensors with `in_dim % 64 == 0` (required by MLX quantize).
    /// Returns `None` when the weight is too small or the shape is incompatible.
    #[cfg(feature = "mlx")]
    fn to_quantized_array(
        &self,
        tensor: &Tensor,
    ) -> Result<Option<(mlx_rs::Array, mlx_rs::Array, mlx_rs::Array)>> {
        // Debug aid: `SAPIENT_MLX_NO_QUANT=1` forces the F32 matmul path so the
        // MLX requantization (quantize + quantized_matmul) can be ruled in/out.
        {
            use std::sync::OnceLock;
            static NO_QUANT: OnceLock<bool> = OnceLock::new();
            if *NO_QUANT
                .get_or_init(|| std::env::var("SAPIENT_MLX_NO_QUANT").is_ok_and(|v| v == "1"))
            {
                return Ok(None);
            }
        }
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Ok(None);
        }
        let (out_dim, in_dim) = (dims[0], dims[1]);
        // MLX quantize requires columns (in_dim) divisible by group_size (64)
        // and both dimensions divisible by 32.
        if in_dim % 64 != 0 || out_dim % 32 != 0 || tensor.numel() < 512 {
            return Ok(None);
        }

        let ptr_key = tensor.as_bytes().as_ptr() as usize;
        {
            let guard = self.quant_cache.lock();
            if let Some(qma) = guard.get(&ptr_key) {
                return Ok(Some(qma.clone()));
            }
        }

        // Dequantize / convert to F32, then apply MLX quantization.
        let shape = Self::to_shape(&[out_dim, in_dim])?;
        let numel = tensor.numel();
        let cow = tensor.to_f32_cow();
        let data = &cow[..numel.min(cow.len())];
        let w_f32 = mlx_rs::Array::from_slice(data, &shape);

        let (wq, scales, biases) = mlx_rs::ops::quantize(&w_f32, 64i32, 4i32)?;
        // Force GPU materialization so subsequent quantized_matmul calls reuse
        // resident GPU memory rather than re-executing the quantize graph node.
        mlx_rs::transforms::eval([&wq, &scales, &biases])?;
        let result = (wq, scales, biases);
        self.quant_cache.lock().insert(ptr_key, result.clone());
        Ok(Some(result))
    }

    #[cfg(feature = "mlx")]
    fn to_tensor(array: mlx_rs::Array) -> Result<Tensor> {
        let shape: Vec<usize> = array
            .shape()
            .iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| anyhow::anyhow!("negative MLX shape dimension: {d}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let data = array.as_slice::<f32>().to_vec();
        Tensor::from_f32(&data, shape).map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn linear_3d(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (x, weight);
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let dims = x.shape().dims();
            if dims.len() != 3 {
                anyhow::bail!("linear_3d expects [batch, seq, hidden]");
            }
            let (batch, seq, in_dim) = (dims[0], dims[1], dims[2]);
            let w_dims = weight.shape().dims();
            if w_dims.len() != 2 {
                anyhow::bail!("linear weight must be 2-D");
            }
            let out_dim = w_dims[0];
            if w_dims[1] != in_dim {
                anyhow::bail!("linear weight in_dim mismatch: {} vs {in_dim}", w_dims[1]);
            }

            // x is a fresh activation (ephemeral); weight is a stable weight (cache it).
            let x_arr =
                Self::to_array_uncached(x)?.reshape(&Self::to_shape(&[batch * seq, in_dim])?)?;

            // Prefer quantized matmul: avoids dequantization, same throughput as mlx-lm.
            if let Some((wq, scales, biases)) = self.to_quantized_array(weight)? {
                let y = mlx_rs::ops::quantized_matmul(
                    &x_arr, &wq, &scales, &biases,
                    true, // transpose=true: weight is [out, in], we want x @ W.T
                    64i32, 4i32,
                )?;
                return Self::to_tensor(y.reshape(&Self::to_shape(&[batch, seq, out_dim])?)?);
            }
            // Fallback: standard F32 matmul (small tensors, embedding, lm_head, etc.)
            let w_arr = self.to_array(weight)?.transpose()?;
            let y = x_arr.matmul(&w_arr)?;
            Self::to_tensor(y.reshape(&Self::to_shape(&[batch, seq, out_dim])?)?)
        }
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (x, weight, eps);
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let x = Self::to_array_uncached(x)?;
            let weight = self.to_array(weight)?;
            Self::to_tensor(mlx_rs::fast::rms_norm(&x, &weight, eps)?)
        }
    }

    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (x, weight, bias, eps);
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let x = Self::to_array_uncached(x)?;
            let weight = self.to_array(weight)?;
            let bias = bias.map(|b| self.to_array(b)).transpose()?;
            Self::to_tensor(mlx_rs::fast::layer_norm(
                &x,
                Some(&weight),
                bias.as_ref(),
                eps,
            )?)
        }
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = x;
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let x = Self::to_array_uncached(x)?;
            Self::to_tensor(mlx_rs::nn::silu(&x)?)
        }
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = x;
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let x = Self::to_array_uncached(x)?;
            Self::to_tensor(mlx_rs::nn::gelu(&x)?)
        }
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (a, b);
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let a = Self::to_array_uncached(a)?;
            let b = Self::to_array_uncached(b)?;
            Self::to_tensor(a.add(&b)?)
        }
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (a, b);
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let a = Self::to_array_uncached(a)?;
            let b = Self::to_array_uncached(b)?;
            Self::to_tensor(a.multiply(&b)?)
        }
    }

    fn apply_rope_positions(&self, x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (x, positions, base);
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let dims = x.shape().dims();
            if dims.len() != 4 {
                anyhow::bail!("RoPE expects [batch, heads, seq, head_dim]");
            }
            if positions.is_empty() {
                anyhow::bail!("RoPE positions cannot be empty");
            }
            let offset = i32::try_from(positions[0])
                .map_err(|_| anyhow::anyhow!("RoPE position too large for MLX"))?;
            let contiguous = positions
                .iter()
                .enumerate()
                .all(|(i, &p)| p == positions[0] + i);
            if !contiguous {
                anyhow::bail!("MLX RoPE requires contiguous positions");
            }

            let x = Self::to_array_uncached(x)?;
            Self::to_tensor(mlx_rs::fast::rope(
                &x,
                dims[3] as i32,
                // `traditional = false` → rotate-half (NeoX/HF) convention, which
                // matches how Llama/Qwen/Phi weights are trained and what the CPU
                // kernel does. `true` (interleaved/GPT-J) produces garbage here.
                false,
                Some(base),
                1.0,
                offset,
                None::<&mlx_rs::Array>,
            )?)
        }
    }

    fn logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (hidden, lm_head);
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let dims = hidden.shape().dims();
            let hidden_size = dims[2];
            let seq = dims[1];
            let h = hidden.to_f32_cow();
            let last = &h[(seq - 1) * hidden_size..seq * hidden_size];
            let h_last = mlx_rs::Array::from_slice(last, &[1, hidden_size as i32]);

            // Prefer quantized matmul for lm_head when eligible.
            if let Some((wq, scales, biases)) = self.to_quantized_array(lm_head)? {
                let logits = mlx_rs::ops::quantized_matmul(
                    &h_last, &wq, &scales, &biases,
                    true, // transpose=true: lm_head is [vocab, hidden], we want h @ W.T
                    64i32, 4i32,
                )?;
                return Ok(logits.as_slice::<f32>().to_vec());
            }
            // Fallback: standard F32 matmul (small vocab or non-aligned dimensions).
            let head = self.to_array(lm_head)?.transpose()?;
            let logits = h_last.matmul(&head)?;
            Ok(logits.as_slice::<f32>().to_vec())
        }
    }

    /// Grouped-query attention (GQA) on the Metal GPU via MLX.
    ///
    /// MLX's `fast::scaled_dot_product_attention` dispatches to an optimised Metal
    /// kernel when `seq_q = 1` (every decode step).  At prefill we build an explicit
    /// additive causal mask in order to correctly handle `seq_q < seq_k` (cached
    /// prefix) — MLX's built-in `"causal"` mode would assume square attention.
    fn gqa_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (q, k, v, n_kv_heads, causal);
            anyhow::bail!("{}", Self::support().reason);
        }

        #[cfg(feature = "mlx")]
        {
            let qs = q.shape().dims().to_vec();
            let ks = k.shape().dims().to_vec();
            let (_batch, n_heads, seq_q, head_dim) = (qs[0], qs[1], qs[2], qs[3]);
            let seq_k = ks[2];
            let scale = 1.0 / (head_dim as f32).sqrt();

            // GQA (n_heads > n_kv_heads): mlx_rs 0.25.3's fast::scaled_dot_product_attention
            // does not correctly handle grouped-query attention when query head count ≠
            // key/value head count — it produces garbage logits. Fall back to the
            // verified CPU reference kernel for all GQA models (Qwen2.5, Llama 3.x,
            // Mistral). Standard MHA (n_heads == n_kv_heads) can use MLX directly.
            if n_heads != n_kv_heads {
                anyhow::bail!(
                    "GQA (n_heads={n_heads} ≠ n_kv_heads={n_kv_heads}): using CPU attention"
                );
            }

            // q: [batch, n_heads, seq_q, head_dim]
            // k: [batch, n_kv_heads, seq_k, head_dim]
            // MLX SDPA expects [batch, heads, seq, dim].
            let q_arr = Self::to_array_uncached(q)?;
            let k_arr = Self::to_array_uncached(k)?;
            let v_arr = Self::to_array_uncached(v)?;

            // Build the causal mask when needed.
            // - seq_q = 1 (decode): every cached key is in the past → no masking needed.
            // - seq_q > 1 (prefill): need a [seq_q, seq_k] upper-triangular -inf mask.
            let mask_arr: Option<mlx_rs::Array> = if causal && seq_q > 1 {
                let offset = seq_k.saturating_sub(seq_q);
                let mut data = vec![0.0f32; seq_q * seq_k];
                for qi in 0..seq_q {
                    for ki in 0..seq_k {
                        if ki > qi + offset {
                            data[qi * seq_k + ki] = f32::NEG_INFINITY;
                        }
                    }
                }
                Some(mlx_rs::Array::from_slice(
                    &data,
                    &[seq_q as i32, seq_k as i32],
                ))
            } else {
                None
            };

            // IntoOption<ScaledDotProductAttentionMask> is implemented for
            // Option<ScaledDotProductAttentionMask>, so wrap our optional mask.
            let mlx_mask = mask_arr
                .as_ref()
                .map(mlx_rs::fast::ScaledDotProductAttentionMask::from);
            let out_arr = mlx_rs::fast::scaled_dot_product_attention(
                &q_arr, &k_arr, &v_arr, scale, mlx_mask,
            )?;

            // out_arr: [batch, n_heads, seq_q, head_dim]
            Self::to_tensor(out_arr)
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl MlxLlmOps {
    fn support() -> MacGpuSupport {
        MacGpuSupport {
            available: false,
            backend: "cpu",
            reason: "MLX GPU execution is only available on macOS",
        }
    }

    fn is_available() -> bool {
        false
    }

    fn linear_3d(&self, _x: &Tensor, _weight: &Tensor) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn rms_norm(&self, _x: &Tensor, _weight: &Tensor, _eps: f32) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn layer_norm(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        _eps: f32,
    ) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn silu(&self, _x: &Tensor) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn gelu(&self, _x: &Tensor) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn add(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn apply_rope_positions(
        &self,
        _x: &Tensor,
        _positions: &[usize],
        _base: f32,
    ) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn gqa_attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _n_kv_heads: usize,
        _causal: bool,
    ) -> Result<Tensor> {
        anyhow::bail!("MLX is only available on macOS")
    }

    fn logits_from_hidden(&self, _hidden: &Tensor, _lm_head: &Tensor) -> Result<Vec<f32>> {
        anyhow::bail!("MLX is only available on macOS")
    }
}

#[derive(Debug, Clone)]
pub enum LlmBackendDispatch {
    Cpu(CpuLlmBackend),
    Metal(MetalLlmBackend),
}

impl LlmBackendDispatch {
    /// Returns `true` when the backend is CPU-only (thread-safe for concurrent
    /// compute calls). Returns `false` for GPU backends (Metal/MLX) whose
    /// command buffers do not support concurrent encoding from multiple threads.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    pub fn from_kind(kind: LlmBackendKind) -> Result<Self> {
        Self::from_kind_with_model_bytes(kind, 0)
    }

    /// Like `from_kind` but gates Metal on SDPA shader availability for `head_dim`.
    ///
    /// MLX ships a fixed set of pre-compiled Metal SDPA shaders (head_dims: 32, 64,
    /// 96, 128, 256).  Selecting Metal for an unsupported head_dim would panic at
    /// inference time with "Unable to load function sdpa_vector_float_N_N".
    /// - Auto → silently falls back to CPU with a tracing::info log.
    /// - Metal (explicit) → returns a user-readable error with a `--backend cpu` hint.
    pub fn from_kind_with_head_dim(kind: LlmBackendKind, head_dim: usize) -> Result<Self> {
        if MetalLlmBackend::is_available() && !mlx_sdpa_supported_head_dim(head_dim) {
            match kind {
                LlmBackendKind::Metal => anyhow::bail!(
                    "Metal/MLX does not support head_dim={head_dim} for this model architecture \
                     (supported: 32, 64, 96, 128, 256). Run with `--backend cpu` instead."
                ),
                LlmBackendKind::Auto => {
                    tracing::info!(
                        head_dim,
                        "auto-backend: CPU (Metal SDPA has no pre-compiled shader for \
                         head_dim={head_dim}; supported: 32, 64, 96, 128, 256)"
                    );
                    return Ok(Self::Cpu(CpuLlmBackend));
                }
                // Wgpu uses its own engine (not this per-op dispatch); treat as CPU here.
                LlmBackendKind::Cpu | LlmBackendKind::Wgpu => {}
            }
        }
        Self::from_kind(kind)
    }

    /// Select a backend, optionally accounting for the model's weight footprint.
    ///
    /// `model_bytes` is the total weight size in bytes (0 = unknown).  On Apple
    /// Silicon (unified memory), Metal is chosen for Auto when the model fits
    /// with a 1.5× KV-cache headroom factor; otherwise CPU is used to avoid
    /// swapping GPU memory which kills throughput.
    pub fn from_kind_with_model_bytes(kind: LlmBackendKind, model_bytes: u64) -> Result<Self> {
        match kind {
            // Wgpu uses its own engine (not this per-op dispatch); CPU fallback here.
            LlmBackendKind::Cpu | LlmBackendKind::Wgpu => Ok(Self::Cpu(CpuLlmBackend)),
            LlmBackendKind::Auto if MetalLlmBackend::is_available() => {
                let fits = metal_memory_fits(model_bytes);
                if fits {
                    tracing::debug!(
                        model_bytes,
                        "auto-backend: Metal (model fits in unified memory)"
                    );
                    Ok(Self::Metal(MetalLlmBackend::default()))
                } else {
                    tracing::info!(
                        model_bytes,
                        "auto-backend: CPU (model too large for Metal GPU memory headroom — \
                         use --backend metal to force GPU anyway)"
                    );
                    Ok(Self::Cpu(CpuLlmBackend))
                }
            }
            LlmBackendKind::Auto => Ok(Self::Cpu(CpuLlmBackend)),
            LlmBackendKind::Metal if MetalLlmBackend::is_available() => {
                Ok(Self::Metal(MetalLlmBackend::default()))
            }
            LlmBackendKind::Metal => {
                let support = mac_gpu_support();
                anyhow::bail!(
                    "Metal/MLX generation backend is unavailable: {}",
                    support.reason
                )
            }
        }
    }
}

/// Returns true when MLX ships a pre-compiled Metal SDPA shader for `head_dim`.
///
/// MLX hard-codes a discrete set of (head_dim, head_dim) shader variants. Requesting
/// any other value causes a runtime panic: "Unable to load function sdpa_vector_float_N_N".
/// Callers should check this before selecting the Metal backend for a model.
pub fn mlx_sdpa_supported_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 32 | 64 | 96 | 128 | 256)
}

/// Returns true when `model_bytes` fit in the Apple Silicon unified memory pool
/// with 1.5× headroom for KV cache and activations.  Returns true when
/// `model_bytes == 0` (unknown size) so we don't block models we can't measure.
fn metal_memory_fits(model_bytes: u64) -> bool {
    if model_bytes == 0 {
        return true;
    }
    let total_ram = total_system_ram_bytes();
    // Reserve 2 GB for the OS + app overhead; require 1.5× headroom for the model.
    let usable = total_ram.saturating_sub(2 * 1024 * 1024 * 1024);
    model_bytes as f64 * 1.5 <= usable as f64
}

/// Total system RAM in bytes via `sysctl hw.memsize` (macOS) or `/proc/meminfo`.
/// Returns 0 on failure (treated as unknown → Metal allowed).
pub fn total_system_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok();
        if let Some(out) = output {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(n) = s.trim().parse::<u64>() {
                    return n;
                }
            }
        }
    }
    0
}

impl LlmBackend for LlmBackendDispatch {
    fn name(&self) -> &'static str {
        match self {
            Self::Cpu(b) => b.name(),
            Self::Metal(b) => b.name(),
        }
    }

    fn linear_3d(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        match self {
            Self::Cpu(b) => b.linear_3d(x, weight),
            Self::Metal(b) => b.linear_3d(x, weight),
        }
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        match self {
            Self::Cpu(b) => b.rms_norm(x, weight, eps),
            Self::Metal(b) => b.rms_norm(x, weight, eps),
        }
    }

    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        match self {
            Self::Cpu(b) => b.layer_norm(x, weight, bias, eps),
            Self::Metal(b) => b.layer_norm(x, weight, bias, eps),
        }
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Cpu(b) => b.silu(x),
            Self::Metal(b) => b.silu(x),
        }
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Cpu(b) => b.gelu(x),
            Self::Metal(b) => b.gelu(x),
        }
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        match self {
            Self::Cpu(backend) => backend.add(a, b),
            Self::Metal(backend) => backend.add(a, b),
        }
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        match self {
            Self::Cpu(backend) => backend.mul(a, b),
            Self::Metal(backend) => backend.mul(a, b),
        }
    }

    fn apply_rope_positions(&self, x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
        match self {
            Self::Cpu(b) => b.apply_rope_positions(x, positions, base),
            Self::Metal(b) => b.apply_rope_positions(x, positions, base),
        }
    }

    fn gqa_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Cpu(b) => b.gqa_attention(q, k, v, n_kv_heads, causal),
            Self::Metal(b) => b.gqa_attention(q, k, v, n_kv_heads, causal),
        }
    }

    fn logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(b) => b.logits_from_hidden(hidden, lm_head),
            Self::Metal(b) => b.logits_from_hidden(hidden, lm_head),
        }
    }

    fn all_logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Cpu(b) => b.all_logits_from_hidden(hidden, lm_head),
            Self::Metal(b) => b.all_logits_from_hidden(hidden, lm_head),
        }
    }
}
