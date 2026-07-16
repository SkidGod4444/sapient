// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `DType` — element data-type enum for tensors.

use crate::error::{Result, SapientError};
use serde::{Deserialize, Serialize};

/// The scalar data-type stored in a tensor buffer.
///
/// Quantized variants store packed blocks rather than individual elements.
/// Use `byte_count(numel)` to compute storage — not `numel * element_size()`.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DType {
    /// 32-bit IEEE float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// Brain float 16.
    BF16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// Boolean (stored as u8, 0 or 1).
    Bool,
    /// 4-bit quantized, ggml Q4_0 block layout.
    /// 32 weights/block, 18 bytes/block (f16 scale + 16 packed nibble bytes).
    Q4_0,
    /// 8-bit quantized, ggml Q8_0 block layout.
    /// 32 weights/block, 34 bytes/block (f16 scale + 32 × i8).
    Q8_0,
    /// K-quant 4-bit (Q4_K_M family). 256 weights/block, 144 bytes/block.
    /// d(f16) + dmin(f16) + 12B scales + 128B nibbles. Dequant on-the-fly.
    Q4_K,
    /// K-quant 5-bit. 256 weights/block, 176 bytes/block.
    Q5_K,
    /// K-quant 6-bit. 256 weights/block, 210 bytes/block.
    Q6_K,
    /// Q4_K repacked for multi-row CPU GEMV (SAPIENT-internal, never on disk):
    /// groups of 4 consecutive rows have their 144-byte super-blocks
    /// block-interleaved — [r0.b0, r1.b0, r2.b0, r3.b0, r0.b1, …] — so a 4-row
    /// dot kernel reads ONE contiguous stream instead of four row-strided ones.
    /// Same bytes-per-weight as Q4_K (pure permutation). Produced at load by
    /// the CPU engine for heap-resident 2-D weights with rows % 4 == 0.
    Q4_K_R4,
    /// Q6_K repacked for multi-row CPU GEMV — same 4-row block-interleaved
    /// scheme as [`DType::Q4_K_R4`], over 210-byte Q6_K super-blocks.
    Q6_K_R4,
}

/// Weights per small quantized block (Q4_0, Q8_0).
pub const QUANT_BLOCK_SIZE: usize = 32;
/// Bytes per Q4_0 block.
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Bytes per Q8_0 block.
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Weights per K-quant block (Q4_K, Q5_K, Q6_K).
pub const K_QUANT_BLOCK_SIZE: usize = 256;
/// Bytes per Q4_K block.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Bytes per Q5_K block.
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// Bytes per Q6_K block.
pub const Q6_K_BLOCK_BYTES: usize = 210;

impl DType {
    /// Size in bytes of a single element for non-quantized dtypes.
    /// **Not valid for Q4_0 / Q8_0** — use `byte_count(numel)` instead.
    #[inline]
    pub const fn element_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::Bool => 1,
            // Quantized types have sub-1 or fractional bytes/element; use byte_count.
            DType::Q4_0
            | DType::Q8_0
            | DType::Q4_K
            | DType::Q4_K_R4
            | DType::Q5_K
            | DType::Q6_K
            | DType::Q6_K_R4 => 0,
        }
    }

    /// Required alignment (in bytes) for this dtype.
    #[inline]
    pub const fn alignment(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
            DType::Q4_0 | DType::Q8_0 => 2,
            DType::Q4_K | DType::Q4_K_R4 | DType::Q5_K | DType::Q6_K | DType::Q6_K_R4 => 2,
        }
    }

    /// Bytes per quantized block. Panics for non-quantized dtypes.
    #[inline]
    pub const fn block_bytes(self) -> usize {
        match self {
            DType::Q4_0 => Q4_0_BLOCK_BYTES,
            DType::Q8_0 => Q8_0_BLOCK_BYTES,
            DType::Q4_K | DType::Q4_K_R4 => Q4_K_BLOCK_BYTES,
            DType::Q5_K => Q5_K_BLOCK_BYTES,
            DType::Q6_K | DType::Q6_K_R4 => Q6_K_BLOCK_BYTES,
            _ => panic!("block_bytes() called on non-quantized dtype"),
        }
    }

    /// Elements per quantized block (32 for Q4_0/Q8_0, 256 for K-quants).
    #[inline]
    pub const fn block_numel(self) -> usize {
        match self {
            DType::Q4_0 | DType::Q8_0 => QUANT_BLOCK_SIZE,
            DType::Q4_K | DType::Q4_K_R4 | DType::Q5_K | DType::Q6_K | DType::Q6_K_R4 => {
                K_QUANT_BLOCK_SIZE
            }
            _ => panic!("block_numel() called on non-quantized dtype"),
        }
    }

    /// Total byte count for `numel` elements. Works for all dtypes.
    #[inline]
    pub fn byte_count(self, numel: usize) -> usize {
        match self {
            DType::Q4_0 => numel / QUANT_BLOCK_SIZE * Q4_0_BLOCK_BYTES,
            DType::Q8_0 => numel / QUANT_BLOCK_SIZE * Q8_0_BLOCK_BYTES,
            DType::Q4_K | DType::Q4_K_R4 => numel / K_QUANT_BLOCK_SIZE * Q4_K_BLOCK_BYTES,
            DType::Q5_K => numel / K_QUANT_BLOCK_SIZE * Q5_K_BLOCK_BYTES,
            DType::Q6_K | DType::Q6_K_R4 => numel / K_QUANT_BLOCK_SIZE * Q6_K_BLOCK_BYTES,
            _ => numel * self.element_size(),
        }
    }

    /// True for all ggml block-quantized dtypes.
    #[inline]
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            DType::Q4_0
                | DType::Q8_0
                | DType::Q4_K
                | DType::Q4_K_R4
                | DType::Q5_K
                | DType::Q6_K
                | DType::Q6_K_R4
        )
    }

    /// Is this a floating-point dtype?
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    /// Is this an integer dtype?
    #[inline]
    pub const fn is_integer(self) -> bool {
        matches!(self, DType::I32 | DType::I64 | DType::U8 | DType::Bool)
    }

    /// Human-readable short name.
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::Bool => "bool",
            DType::Q4_0 => "q4_0",
            DType::Q8_0 => "q8_0",
            DType::Q4_K => "q4_k",
            DType::Q4_K_R4 => "q4_k_r4",
            DType::Q5_K => "q5_k",
            DType::Q6_K => "q6_k",
            DType::Q6_K_R4 => "q6_k_r4",
        }
    }

    /// Parse from a string (case-insensitive). Also available via `s.parse::<DType>()`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        s.parse()
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for DType {
    type Err = crate::error::SapientError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(DType::F32),
            "f16" | "float16" => Ok(DType::F16),
            "bf16" | "bfloat16" => Ok(DType::BF16),
            "i32" | "int32" => Ok(DType::I32),
            "i64" | "int64" => Ok(DType::I64),
            "u8" | "uint8" => Ok(DType::U8),
            "bool" => Ok(DType::Bool),
            "q4_0" => Ok(DType::Q4_0),
            "q8_0" => Ok(DType::Q8_0),
            "q4_k" | "q4_k_m" | "q4_k_s" => Ok(DType::Q4_K),
            "q5_k" | "q5_k_m" | "q5_k_s" => Ok(DType::Q5_K),
            "q6_k" => Ok(DType::Q6_K),
            other => Err(SapientError::TypeMismatch {
                expected: "a valid dtype".to_owned(),
                got: other.to_owned(),
            }),
        }
    }
}

// ── ONNX numeric type code mapping ──────────────────────────────────────────

impl DType {
    /// Map from ONNX TensorProto::DataType integer.
    pub fn from_onnx_dtype(code: i32) -> Result<Self> {
        match code {
            1 => Ok(DType::F32),
            2 => Ok(DType::U8),
            5 => Ok(DType::I32),
            7 => Ok(DType::I64),
            9 => Ok(DType::Bool),
            10 => Ok(DType::F16),
            16 => Ok(DType::BF16),
            other => Err(SapientError::TypeMismatch {
                expected: "a supported ONNX dtype".into(),
                got: format!("ONNX code {other}"),
            }),
        }
    }

    /// Map to ONNX TensorProto::DataType integer.
    pub fn to_onnx_dtype(self) -> i32 {
        match self {
            DType::F32 => 1,
            DType::U8 => 2,
            DType::I32 => 5,
            DType::I64 => 7,
            DType::Bool => 9,
            DType::F16 => 10,
            DType::BF16 => 16,
            // No standard ONNX code for ggml quant types.
            DType::Q4_0
            | DType::Q8_0
            | DType::Q4_K
            | DType::Q4_K_R4
            | DType::Q5_K
            | DType::Q6_K
            | DType::Q6_K_R4 => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_sizes() {
        assert_eq!(DType::F32.element_size(), 4);
        assert_eq!(DType::I64.element_size(), 8);
        assert_eq!(DType::Bool.element_size(), 1);
    }

    #[test]
    fn byte_count() {
        assert_eq!(DType::F32.byte_count(10), 40);
    }

    #[test]
    fn from_str_roundtrip() {
        for (s, dt) in [
            ("f32", DType::F32),
            ("f16", DType::F16),
            ("bf16", DType::BF16),
            ("i32", DType::I32),
            ("i64", DType::I64),
            ("u8", DType::U8),
            ("bool", DType::Bool),
        ] {
            assert_eq!(DType::from_str(s).unwrap(), dt);
        }
    }

    #[test]
    fn onnx_roundtrip() {
        for dt in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::Bool,
        ] {
            let code = dt.to_onnx_dtype();
            assert_eq!(DType::from_onnx_dtype(code).unwrap(), dt);
        }
    }
}
