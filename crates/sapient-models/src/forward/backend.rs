//! Backend dispatch for native LLM forward passes.

use anyhow::Result;
use sapient_core::Tensor;

use super::common;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LlmBackendKind {
    Cpu,
    Metal,
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
            Self::Auto => write!(f, "auto"),
        }
    }
}

pub trait LlmBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn linear_3d(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor>;
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
    fn gqa_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_kv_heads: usize,
        causal: bool,
    ) -> Result<Tensor>;
    fn logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>>;
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

    fn fallback(&self, op: &str) {
        tracing::warn!(
            op = op,
            "native Metal LLM kernel is not implemented yet; using CPU reference kernel"
        );
    }
}

impl LlmBackend for MetalLlmBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn linear_3d(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        self.mlx
            .linear_3d(x, weight)
            .or_else(|e| {
                tracing::warn!(op = "linear_3d", error = %e, "MLX op failed; using CPU reference kernel");
                self.cpu.linear_3d(x, weight)
            })
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
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
        self.mlx.layer_norm(x, weight, bias, eps).or_else(|e| {
            tracing::warn!(op = "layer_norm", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.layer_norm(x, weight, bias, eps)
        })
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        self.mlx.silu(x).or_else(|e| {
            tracing::warn!(op = "silu", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.silu(x)
        })
    }

    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        self.mlx.gelu(x).or_else(|e| {
            tracing::warn!(op = "gelu", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.gelu(x)
        })
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.mlx.add(a, b).or_else(|e| {
            tracing::warn!(op = "add", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.add(a, b)
        })
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.mlx.mul(a, b).or_else(|e| {
            tracing::warn!(op = "mul", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.mul(a, b)
        })
    }

    fn apply_rope_positions(&self, x: &Tensor, positions: &[usize], base: f32) -> Result<Tensor> {
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
        // Keep this on the CPU reference path for now. Sapient's CPU attention
        // mask handles cached decoding with q_len < kv_len; using MLX's
        // causal shortcut without an offset would corrupt generation.
        self.fallback("gqa_attention");
        self.cpu.gqa_attention(q, k, v, n_kv_heads, causal)
    }

    fn logits_from_hidden(&self, hidden: &Tensor, lm_head: &Tensor) -> Result<Vec<f32>> {
        self.mlx.logits_from_hidden(hidden, lm_head).or_else(|e| {
            tracing::warn!(op = "logits", error = %e, "MLX op failed; using CPU reference kernel");
            self.cpu.logits_from_hidden(hidden, lm_head)
        })
    }
}

#[derive(Debug, Default, Clone)]
struct MlxLlmOps;

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

    #[cfg(feature = "mlx")]
    fn to_array(tensor: &Tensor) -> Result<mlx_rs::Array> {
        let shape = Self::to_shape(tensor.shape().dims())?;
        Ok(mlx_rs::Array::from_slice(tensor.as_f32_slice(), &shape))
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

            let x_arr = Self::to_array(x)?.reshape(&Self::to_shape(&[batch * seq, in_dim])?)?;
            let w_arr = Self::to_array(weight)?.transpose()?;
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
            let x = Self::to_array(x)?;
            let weight = Self::to_array(weight)?;
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
            let x = Self::to_array(x)?;
            let weight = Self::to_array(weight)?;
            let bias = bias.map(Self::to_array).transpose()?;
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
            let x = Self::to_array(x)?;
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
            let x = Self::to_array(x)?;
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
            let a = Self::to_array(a)?;
            let b = Self::to_array(b)?;
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
            let a = Self::to_array(a)?;
            let b = Self::to_array(b)?;
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

            let x = Self::to_array(x)?;
            Self::to_tensor(mlx_rs::fast::rope(
                &x,
                dims[3] as i32,
                true,
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
            let h = hidden.as_f32_slice();
            let last = &h[(seq - 1) * hidden_size..seq * hidden_size];
            let h_last = mlx_rs::Array::from_slice(last, &[1, hidden_size as i32]);
            let head = Self::to_array(lm_head)?.transpose()?;
            let logits = h_last.matmul(&head)?;
            Ok(logits.as_slice::<f32>().to_vec())
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
    pub fn from_kind(kind: LlmBackendKind) -> Result<Self> {
        match kind {
            LlmBackendKind::Cpu => Ok(Self::Cpu(CpuLlmBackend)),
            LlmBackendKind::Auto if MetalLlmBackend::is_available() => {
                Ok(Self::Metal(MetalLlmBackend::default()))
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
}
