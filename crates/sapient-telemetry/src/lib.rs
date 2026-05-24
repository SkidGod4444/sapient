//! SAPIENT telemetry — built-in observability, not bolted-on.
//!
//! Implements the `Telemetry` trait with:
//!   - `NoOpTelemetry`:  zero-overhead default
//!   - `ConsoleTelemetry`: prints spans to stderr (useful in dev)
//!   - `OtelTelemetry` (feature "otel"): OTLP export
//!
//! All per-op instrumentation points are in `InstrumentedSession` in
//! `sapient-runtime` — telemetry crate only provides the traits and impls.

pub mod metrics;
pub mod profiler;
pub mod telemetry;

pub use profiler::{ChromeTracer, Span};
pub use telemetry::{ConsoleTelemetry, NoOpTelemetry, Telemetry, TelemetryConfig};

/// Initialise a global `tracing` subscriber (JSON or pretty).
pub fn init_tracing(json: bool) {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if json {
        fmt().with_env_filter(filter).json().init();
    } else {
        fmt().with_env_filter(filter).pretty().init();
    }
}
