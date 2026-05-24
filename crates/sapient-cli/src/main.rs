//! SAPIENT CLI — `sapient run | bench | inspect | serve`

mod server;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use sapient_core::Tensor;
use sapient_runtime::{InferenceSession, Model, ModelConfig, SessionOptions};
use sapient_telemetry::init_tracing;

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "sapient",
    version = env!("CARGO_PKG_VERSION"),
    about   = "SAPIENT Inference Engine — high-performance edge inference",
    long_about = None,
)]
struct Cli {
    /// Enable verbose / debug output.
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Output structured JSON logs.
    #[arg(long, global = true)]
    json_logs: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model with JSON input.
    Run {
        /// Path to the model file (ONNX, GGUF).
        model: PathBuf,

        /// Path to input JSON file.
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Path to write output JSON (defaults to stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Backend: cpu | metal | vulkan.
        #[arg(short, long, default_value = "cpu")]
        backend: String,

        /// Enable console telemetry.
        #[arg(long)]
        telemetry: bool,
    },

    /// Benchmark a model across batch sizes.
    Bench {
        /// Path to the model file.
        model: PathBuf,

        /// Comma-separated batch sizes (e.g. 1,4,8,16).
        #[arg(long, default_value = "1,4,8,16")]
        batch_sizes: String,

        /// Backend to benchmark.
        #[arg(short, long, default_value = "cpu")]
        backend: String,

        /// Number of warmup iterations.
        #[arg(long, default_value = "10")]
        warmup: usize,

        /// Number of benchmark iterations.
        #[arg(long, default_value = "100")]
        iters: usize,
    },

    /// Print graph structure in DOT format.
    Inspect {
        /// Path to the model file.
        model: PathBuf,

        /// Write DOT to file instead of stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Start an HTTP inference server.
    Serve {
        /// Path to the model file.
        model: PathBuf,

        /// Port to listen on.
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Backend to use.
        #[arg(short, long, default_value = "cpu")]
        backend: String,

        /// Number of worker threads.
        #[arg(short, long, default_value = "4")]
        workers: usize,
    },
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);

    if cli.verbose {
        // Already handled by RUST_LOG, but we could set it here programmatically.
    }

    match cli.command {
        Commands::Run { model, input, output, backend, telemetry } => {
            run_command(model, input, output, backend, telemetry)?;
        }
        Commands::Bench { model, batch_sizes, backend, warmup, iters } => {
            bench_command(model, &batch_sizes, backend, warmup, iters)?;
        }
        Commands::Inspect { model, output } => {
            inspect_command(model, output)?;
        }
        Commands::Serve { model, port, backend, workers } => {
            server::serve(model, port, backend, workers).await?;
        }
    }

    Ok(())
}

// ── run ───────────────────────────────────────────────────────────────────────

fn run_command(
    model_path: PathBuf,
    input_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    backend: String,
    telemetry: bool,
) -> Result<()> {
    let config = ModelConfig { backend: backend.clone(), ..Default::default() };
    let model = Model::load(&model_path, config).context("failed to load model")?;
    let opts = SessionOptions { telemetry, ..Default::default() };
    let session = InferenceSession::new(model, opts).context("failed to create session")?;

    // Load inputs.
    let inputs: HashMap<String, Tensor> = if let Some(p) = input_path {
        let json = std::fs::read_to_string(&p).context("reading input JSON")?;
        parse_input_json(&json).context("parsing input JSON")?
    } else {
        // No inputs — run with empty (useful for constant-output models).
        HashMap::new()
    };

    info!(backend = %backend, "running inference");
    let start = Instant::now();
    let outputs = session.run(inputs).context("inference failed")?;
    let elapsed_ms = start.elapsed().as_millis();

    println!("Inference completed in {elapsed_ms}ms");
    println!("Outputs: {} tensor(s)", outputs.len());
    for (i, t) in outputs.iter().enumerate() {
        println!("  [{i}] shape={} dtype={}", t.shape(), t.dtype());
    }

    // Write output.
    if let Some(out_path) = output_path {
        let json = serialise_outputs(&outputs);
        std::fs::write(&out_path, json).context("writing output JSON")?;
        println!("Output written to {}", out_path.display());
    }

    Ok(())
}

// ── bench ─────────────────────────────────────────────────────────────────────

fn bench_command(
    model_path: PathBuf,
    batch_sizes: &str,
    backend: String,
    warmup: usize,
    iters: usize,
) -> Result<()> {
    let sizes: Vec<usize> = batch_sizes
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let config = ModelConfig { backend: backend.clone(), optimize: true, ..Default::default() };
    let model = Model::load(&model_path, config).context("failed to load model")?;
    let session = InferenceSession::new(model, SessionOptions::default())?;

    println!("┌─────────────┬────────────────┬───────────────┬───────────────┐");
    println!("│ Batch Size  │ Median (ms)    │ P99 (ms)      │ Throughput/s  │");
    println!("├─────────────┼────────────────┼───────────────┼───────────────┤");

    for &bs in &sizes {
        let inputs = make_dummy_inputs(&session, bs);

        // Warmup.
        for _ in 0..warmup {
            let _ = session.run(inputs.clone());
        }

        // Benchmark.
        let mut latencies_us: Vec<u64> = (0..iters)
            .map(|_| {
                let t = Instant::now();
                let _ = session.run(inputs.clone());
                t.elapsed().as_micros() as u64
            })
            .collect();

        latencies_us.sort_unstable();
        let median_ms = latencies_us[latencies_us.len() / 2] as f64 / 1000.0;
        let p99_ms    = latencies_us[latencies_us.len() * 99 / 100] as f64 / 1000.0;
        let tps       = bs as f64 / (median_ms / 1000.0);

        println!(
            "│ {:11} │ {:14.2} │ {:13.2} │ {:13.0} │",
            bs, median_ms, p99_ms, tps
        );
    }

    println!("└─────────────┴────────────────┴───────────────┴───────────────┘");
    Ok(())
}

// ── inspect ───────────────────────────────────────────────────────────────────

fn inspect_command(model_path: PathBuf, output_path: Option<PathBuf>) -> Result<()> {
    let config = ModelConfig { optimize: false, infer_shapes: false, ..Default::default() };
    let model = Model::load(&model_path, config).context("loading model")?;

    let dot = model.graph.to_dot();

    if let Some(p) = output_path {
        std::fs::write(&p, &dot).context("writing DOT file")?;
        println!("DOT graph written to {}", p.display());
    } else {
        print!("{dot}");
    }

    println!("\nGraph: {} nodes, {} edges", model.graph.node_count(), model.graph.edges.len());
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_input_json(json: &str) -> Result<HashMap<String, Tensor>> {
    #[derive(serde::Deserialize)]
    struct InputSpec {
        shape: Vec<usize>,
        data:  Vec<f32>,
    }

    let map: HashMap<String, InputSpec> = serde_json::from_str(json)
        .context("invalid input JSON format")?;

    let mut tensors = HashMap::new();
    for (name, spec) in map {
        let t = Tensor::from_f32(&spec.data, spec.shape).context("building tensor")?;
        tensors.insert(name, t);
    }
    Ok(tensors)
}

fn serialise_outputs(outputs: &[Tensor]) -> String {
    use serde_json::json;
    let arr: Vec<serde_json::Value> = outputs
        .iter()
        .map(|t| json!({
            "shape": t.shape().dims(),
            "dtype": t.dtype().to_string(),
            "data": t.as_f32_slice(),
        }))
        .collect();
    serde_json::to_string_pretty(&arr).unwrap_or_default()
}

fn make_dummy_inputs(session: &InferenceSession, _batch_size: usize) -> HashMap<String, Tensor> {
    // Build a dummy input for each graph input node.
    let mut inputs = HashMap::new();
    for &id in &session.model().graph.inputs {
        use sapient_ir::node::Node;
        if let Some(Node::Input { name, shape, dtype, .. }) = session.model().graph.get(id) {
            let shape = shape.clone().unwrap_or_else(|| sapient_core::Shape::new([1]));
            let dtype = dtype.unwrap_or(sapient_core::DType::F32);
            if let Ok(t) = Tensor::zeros(shape, dtype) {
                inputs.insert(name.clone(), t);
            }
        }
    }
    inputs
}
