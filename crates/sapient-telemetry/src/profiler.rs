//! Chrome trace format profiler.
//!
//! Generates a `trace.json` compatible with `chrome://tracing` for visual
//! flame-graph profiling of execution.

use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

// ── Span ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Span {
    pub name:     String,
    pub category: String,
    pub start_us: u64,
    pub dur_us:   u64,
    pub pid:      u32,
    pub tid:      u64,
}

impl Span {
    pub fn new(name: impl Into<String>, category: impl Into<String>, start: Instant, dur: Duration) -> Self {
        Self {
            name:     name.into(),
            category: category.into(),
            start_us: start.elapsed().as_micros() as u64,
            dur_us:   dur.as_micros() as u64,
            pid:      std::process::id(),
            tid:      thread_id(),
        }
    }
}

fn thread_id() -> u64 {
    // Stable numeric thread ID via hash of OS thread ID string.
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    format!("{:?}", std::thread::current().id()).hash(&mut h);
    h.finish()
}

// ── ChromeTracer ──────────────────────────────────────────────────────────────

/// Accumulates spans and writes them in Chrome trace format.
#[derive(Debug, Default)]
pub struct ChromeTracer {
    spans: Mutex<Vec<Span>>,
}

impl ChromeTracer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, span: Span) {
        self.spans.lock().unwrap().push(span);
    }

    pub fn record_now(&self, name: impl Into<String>, category: impl Into<String>, dur: Duration) {
        self.record(Span {
            name:     name.into(),
            category: category.into(),
            start_us: 0, // simplified: real implementation uses a monotonic base
            dur_us:   dur.as_micros() as u64,
            pid:      std::process::id(),
            tid:      thread_id(),
        });
    }

    /// Write the collected spans to a `trace.json` file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        #[derive(Serialize)]
        struct TraceEvent {
            name: String,
            cat:  String,
            ph:   char,
            ts:   u64,
            dur:  u64,
            pid:  u32,
            tid:  u64,
        }

        let spans = self.spans.lock().unwrap();
        let events: Vec<TraceEvent> = spans
            .iter()
            .map(|s| TraceEvent {
                name: s.name.clone(),
                cat:  s.category.clone(),
                ph:   'X', // complete events
                ts:   s.start_us,
                dur:  s.dur_us,
                pid:  s.pid,
                tid:  s.tid,
            })
            .collect();

        #[derive(Serialize)]
        struct TraceFile { traceEvents: Vec<TraceEvent> }
        let json = serde_json::to_string_pretty(&TraceFile { traceEvents: events })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        fs::write(path, json)
    }
}
