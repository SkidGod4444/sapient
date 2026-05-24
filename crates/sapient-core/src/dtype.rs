//! `DType` — element data-type enum for tensors.

use crate::error::{Result, SapientError};
use serde::{Deserialize, Serialize};

/// The scalar data-type stored in a tensor buffer.
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
}

impl DType {
    /// Size in bytes of a single element.
    #[inline]
    pub const fn element_size(self) -> usize {
        match self {
            DType::F32  => 4,
            DType::F16  => 2,
            DType::BF16 => 2,
            DType::I32  => 4,
            DType::I64  => 8,
            DType::U8   => 1,
            DType::Bool => 1,
        }
    }

    /// Required alignment (in bytes) for this dtype.
    #[inline]
    pub const fn alignment(self) -> usize {
        match self {
            DType::F32  => 4,
            DType::F16  => 2,
            DType::BF16 => 2,
            DType::I32  => 4,
            DType::I64  => 8,
            DType::U8   => 1,
            DType::Bool => 1,
        }
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
            DType::F32  => "f32",
            DType::F16  => "f16",
            DType::BF16 => "bf16",
            DType::I32  => "i32",
            DType::I64  => "i64",
            DType::U8   => "u8",
            DType::Bool => "bool",
        }
    }

    /// Parse from a string (case-insensitive).
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "float32"  => Ok(DType::F32),
            "f16" | "float16"  => Ok(DType::F16),
            "bf16" | "bfloat16" => Ok(DType::BF16),
            "i32" | "int32"    => Ok(DType::I32),
            "i64" | "int64"    => Ok(DType::I64),
            "u8"  | "uint8"    => Ok(DType::U8),
            "bool"             => Ok(DType::Bool),
            other => Err(SapientError::TypeMismatch {
                expected: "a valid dtype".to_owned(),
                got: other.to_owned(),
            }),
        }
    }

    /// Compute the total byte count for `numel` elements of this dtype.
    #[inline]
    pub fn byte_count(self, numel: usize) -> usize {
        numel * self.element_size()
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ── ONNX numeric type code mapping ──────────────────────────────────────────

impl DType {
    /// Map from ONNX TensorProto::DataType integer.
    pub fn from_onnx_dtype(code: i32) -> Result<Self> {
        match code {
            1  => Ok(DType::F32),
            2  => Ok(DType::U8),
            5  => Ok(DType::I32),
            7  => Ok(DType::I64),
            9  => Ok(DType::Bool),
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
            DType::F32  => 1,
            DType::U8   => 2,
            DType::I32  => 5,
            DType::I64  => 7,
            DType::Bool => 9,
            DType::F16  => 10,
            DType::BF16 => 16,
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
            ("f32", DType::F32), ("f16", DType::F16), ("bf16", DType::BF16),
            ("i32", DType::I32), ("i64", DType::I64), ("u8", DType::U8), ("bool", DType::Bool),
        ] {
            assert_eq!(DType::from_str(s).unwrap(), dt);
        }
    }

    #[test]
    fn onnx_roundtrip() {
        for dt in [DType::F32, DType::F16, DType::BF16, DType::I32, DType::I64, DType::U8, DType::Bool] {
            let code = dt.to_onnx_dtype();
            assert_eq!(DType::from_onnx_dtype(code).unwrap(), dt);
        }
    }
}
