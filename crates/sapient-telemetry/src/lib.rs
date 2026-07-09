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

/// Like [`init_tracing`], but with a caller-chosen default filter and a plain
/// human format. Long-running commands (`sapient serve`) route here so their
/// operational logs ("loading model…", "evicted…") reach the console even
/// without `--verbose` — a server that loads multi-GB models in silence looks
/// hung from the client side. `RUST_LOG` still overrides.
pub fn init_tracing_with_default(default: &str) {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .init();
}

/// Initialise a global `tracing` subscriber (JSON or pretty).
///
/// When `verbose` is false, tracing output is disabled so chat/pull stay clean.
/// Set `RUST_LOG=info` to override.
pub fn init_tracing(json: bool, verbose: bool) {
    use std::io;
    use tracing_subscriber::{fmt, EnvFilter};

    let default = if verbose { "info" } else { "off" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));

    if json {
        fmt().with_env_filter(filter).json().init();
    } else if verbose {
        fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_file(false)
            .with_line_number(false)
            .init();
    } else {
        fmt().with_env_filter(filter).with_writer(io::sink).init();
    }
}
