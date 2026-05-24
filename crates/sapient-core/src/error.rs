//! `SapientError` — unified error type for the entire SAPIENT engine.

use thiserror::Error;

/// The central error enum for all SAPIENT operations.
#[derive(Debug, Error)]
pub enum SapientError {
    // ── Shape / Type errors ──────────────────────────────────────────────────
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Rank mismatch: expected {expected}, got {got}")]
    RankMismatch { expected: usize, got: usize },

    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    #[error("Incompatible shapes for broadcasting: {lhs:?} and {rhs:?}")]
    BroadcastError { lhs: Vec<usize>, rhs: Vec<usize> },

    // ── Graph / IR errors ────────────────────────────────────────────────────
    #[error("Graph contains a cycle — execution is impossible")]
    CyclicGraph,

    #[error("Node {0:?} not found in graph")]
    NodeNotFound(String),

    #[error("Graph validation failed: {0}")]
    InvalidGraph(String),

    #[error("Shape inference failed for op '{op}': {reason}")]
    ShapeInferenceFailed { op: String, reason: String },

    // ── Backend errors ───────────────────────────────────────────────────────
    #[error("Backend '{backend}' does not support op '{op}'")]
    UnsupportedOp { backend: String, op: String },

    #[error("Backend error from '{backend}': {message}")]
    BackendError { backend: String, message: String },

    #[error("No suitable backend found for execution")]
    NoBackendAvailable,

    // ── Memory / allocation errors ───────────────────────────────────────────
    #[error("Allocation failed: requested {bytes} bytes (alignment {align})")]
    AllocationFailed { bytes: usize, align: usize },

    #[error("Buffer size mismatch: expected {expected} bytes, got {got}")]
    BufferSizeMismatch { expected: usize, got: usize },

    #[error("Memory pool exhausted — consider increasing pool capacity")]
    PoolExhausted,

    // ── Format / IO errors ───────────────────────────────────────────────────
    #[error("ONNX parse error: {0}")]
    OnnxParseError(String),

    #[error("GGUF parse error: {0}")]
    GgufParseError(String),

    #[error("Safetensors parse error: {0}")]
    SafetensorsParseError(String),

    #[error("Unsupported model format: {0}")]
    UnsupportedFormat(String),

    #[error("Model not found at path '{0}'")]
    ModelNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    // ── Scheduling / runtime errors ──────────────────────────────────────────
    #[error("Request timed out (deadline exceeded)")]
    DeadlineExceeded,

    #[error("Batch scheduler is shut down")]
    SchedulerShutdown,

    #[error("Runtime is not initialized — call Session::new() first")]
    UninitializedRuntime,

    // ── Telemetry errors ─────────────────────────────────────────────────────
    #[error("Telemetry export failed: {0}")]
    TelemetryError(String),

    // ── Catch-all ────────────────────────────────────────────────────────────
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Convenience alias used throughout the codebase.
pub type Result<T> = std::result::Result<T, SapientError>;

// ── Helper constructors ──────────────────────────────────────────────────────

impl SapientError {
    /// Create a backend error with context.
    pub fn backend(backend: impl Into<String>, message: impl Into<String>) -> Self {
        Self::BackendError {
            backend: backend.into(),
            message: message.into(),
        }
    }

    /// Create an unsupported-op error.
    pub fn unsupported_op(backend: impl Into<String>, op: impl Into<String>) -> Self {
        Self::UnsupportedOp {
            backend: backend.into(),
            op: op.into(),
        }
    }

    /// Create an internal error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = SapientError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![2, 4],
        };
        let s = e.to_string();
        assert!(s.contains("Shape mismatch"));
        assert!(s.contains("[2, 3]"));
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let e: SapientError = io_err.into();
        assert!(matches!(e, SapientError::Io(_)));
    }
}
