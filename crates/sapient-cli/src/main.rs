//! SAPIENT CLI — chat, pull, run, bench, inspect, serve

mod hub;
mod server;

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sapient_generate::Pipeline;
use sapient_runtime::{InferenceSession, Model, ModelConfig, SessionOptions};
use sapient_telemetry::init_tracing;
use sapient_tokenizers::ChatMessage;
use tracing::info;

use sapient_core::Tensor;

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "sapient",
    version = env!("CARGO_PKG_VERSION"),
    about   = "SAPIENT Inference Engine — run HuggingFace models locally",
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
    /// Interactive chat with a HuggingFace model.
    Chat {
        /// HuggingFace model ID (e.g. `microsoft/phi-2`).
        model: String,
    },

    /// Download a model from HuggingFace Hub to the local cache.
    Pull {
        /// HuggingFace model ID.
        model: String,
    },

    /// List models in the local HuggingFace cache.
    List,

    /// Show architecture and config info for a HuggingFace model.
    Info {
        /// HuggingFace model ID.
        model: String,
    },

    /// Save a HuggingFace access token for gated models.
    Login {
        /// Token value (otherwise read from stdin).
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
    },

    /// Run text generation or file-based inference.
    Run {
        /// HuggingFace model ID or path to a model file (ONNX, GGUF).
        model: String,

        /// Prompt for HuggingFace models (required for Hub IDs).
        #[arg(short, long)]
        prompt: Option<String>,

        /// Path to input JSON file (file-based models only).
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Path to write output JSON (file-based models only).
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Backend: cpu | metal | vulkan (file-based models only).
        #[arg(short, long, default_value = "cpu")]
        backend: String,

        /// Enable console telemetry.
        #[arg(long)]
        telemetry: bool,
    },

    /// Benchmark a model across batch sizes (file-based models).
    Bench {
        /// HuggingFace model ID or path to a model file.
        model: String,

        #[arg(long, default_value = "1,4,8,16")]
        batch_sizes: String,

        #[arg(short, long, default_value = "cpu")]
        backend: String,

        #[arg(long, default_value = "10")]
        warmup: usize,

        #[arg(long, default_value = "100")]
        iters: usize,
    },

    /// Print graph structure in DOT format (file-based models).
    Inspect {
        /// HuggingFace model ID or path to a model file.
        model: String,

        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Start an HTTP inference server (file-based models).
    Serve {
        /// HuggingFace model ID or path to a model file.
        model: String,

        #[arg(short, long, default_value = "8080")]
        port: u16,

        #[arg(short, long, default_value = "cpu")]
        backend: String,

        #[arg(short, long, default_value = "4")]
        workers: usize,
    },
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);

    match cli.command {
        Commands::Chat { model } => chat_command(&model).await,
        Commands::Pull { model } => pull_command(&model).await,
        Commands::List => list_command(),
        Commands::Info { model } => info_command(&model).await,
        Commands::Login { token } => login_command(token.as_deref()),
        Commands::Run {
            model,
            prompt,
            input,
            output,
            backend,
            telemetry,
        } => run_command(&model, prompt, input, output, backend, telemetry).await,
        Commands::Bench {
            model,
            batch_sizes,
            backend,
            warmup,
            iters,
        } => bench_command(&model, &batch_sizes, backend, warmup, iters).await,
        Commands::Inspect { model, output } => inspect_command(&model, output).await,
        Commands::Serve {
            model,
            port,
            backend,
            workers,
        } => {
            let model_path = hub::resolve_model_path(&model).await?;
            server::serve(model_path, port, backend, workers).await
        }
    }
}

// ── Hub commands ──────────────────────────────────────────────────────────────

async fn chat_command(model: &str) -> Result<()> {
    println!("Loading {model}...");
    let pipeline = Pipeline::from_pretrained(model)
        .await
        .with_context(|| format!("failed to load model '{model}'"))?;

    let arch = pipeline.arch();
    println!("Ready — {model} ({arch:?}). Type /exit to quit.\n");

    let mut history: Vec<ChatMessage> = Vec::new();
    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut line = String::new();
        let n = io::stdin().read_line(&mut line)?;
        if n == 0 {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "/exit" | "/quit" | "/q") {
            break;
        }

        history.push(ChatMessage::user(line));
        print!("sapient> ");
        io::stdout().flush()?;

        let reply = pipeline.chat(&history).await.context("generation failed")?;
        println!("{reply}\n");
        history.push(ChatMessage::assistant(reply));
    }

    Ok(())
}

async fn pull_command(model: &str) -> Result<()> {
    println!("Pulling {model}...");
    let files = hub::pull_model(model).await?;
    println!("✓ Cached {model}");
    println!("  config:    {}", files.config_path.display());
    if let Some(tok) = &files.tokenizer_path {
        println!("  tokenizer: {}", tok.display());
    }
    for w in &files.weight_paths {
        println!("  weights:   {}", w.display());
    }
    Ok(())
}

fn list_command() -> Result<()> {
    let models = hub::list_cached_models()?;
    if models.is_empty() {
        println!("No cached models yet. Download one with:");
        println!("  sapient pull microsoft/phi-2");
        return Ok(());
    }
    println!("Cached models ({}):", models.len());
    for m in models {
        println!("  {m}");
    }
    Ok(())
}

async fn info_command(model: &str) -> Result<()> {
    let info = hub::fetch_model_info(model)
        .await
        .with_context(|| format!("failed to fetch info for '{model}'"))?;

    println!("Model:      {model}");
    println!("Type:       {}", info.model_type);
    println!("Arch:       {:?}", info.arch);
    println!("Layers:     {}", info.num_hidden_layers);
    println!("Hidden:     {}", info.hidden_size);
    println!("Heads:      {}", info.num_attention_heads);
    println!("KV heads:   {}", info.num_key_value_heads);
    println!("Vocab:      {}", info.vocab_size);
    println!("Context:    {}", info.max_position_embeddings);
    Ok(())
}

fn login_command(token: Option<&str>) -> Result<()> {
    let token = match token {
        Some(t) if !t.is_empty() => t.to_owned(),
        _ => {
            print!("HuggingFace token (hf_...): ");
            io::stdout().flush()?;
            let mut buf = String::new();
            io::stdin().read_line(&mut buf)?;
            buf.trim().to_owned()
        }
    };

    let path = hub::save_hf_token(&token)?;
    println!("✓ Token saved to {}", path.display());
    println!("  Gated models (Llama, Gemma, etc.) are now accessible.");
    Ok(())
}

// ── run ───────────────────────────────────────────────────────────────────────

async fn run_command(
    model: &str,
    prompt: Option<String>,
    input_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    backend: String,
    telemetry: bool,
) -> Result<()> {
    if hub::looks_like_hub_model_id(model) {
        let prompt = prompt.with_context(|| {
            format!(
                "Hub model '{model}' requires --prompt.\n\
                 For interactive chat use: sapient chat {model}"
            )
        })?;
        println!("Loading {model}...");
        let pipeline = Pipeline::from_pretrained(model).await?;
        let output = pipeline.generate(&prompt).await?;
        println!("{output}");
        return Ok(());
    }

    let model_path = PathBuf::from(model);
    let config = ModelConfig {
        backend: backend.clone(),
        ..Default::default()
    };
    let model = Model::load(&model_path, config).context("failed to load model")?;
    let opts = SessionOptions {
        telemetry,
        ..Default::default()
    };
    let session = InferenceSession::new(model, opts).context("failed to create session")?;

    let inputs: HashMap<String, Tensor> = if let Some(p) = input_path {
        let json = std::fs::read_to_string(&p).context("reading input JSON")?;
        parse_input_json(&json).context("parsing input JSON")?
    } else {
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

    if let Some(out_path) = output_path {
        let json = serialise_outputs(&outputs);
        std::fs::write(&out_path, json).context("writing output JSON")?;
        println!("Output written to {}", out_path.display());
    }

    Ok(())
}

// ── bench ─────────────────────────────────────────────────────────────────────

async fn bench_command(
    model: &str,
    batch_sizes: &str,
    backend: String,
    warmup: usize,
    iters: usize,
) -> Result<()> {
    let model_path = hub::resolve_model_path(model).await?;
    let sizes: Vec<usize> = batch_sizes
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let config = ModelConfig {
        backend: backend.clone(),
        optimize: true,
        ..Default::default()
    };
    let model = Model::load(&model_path, config).context("failed to load model")?;
    let session = InferenceSession::new(model, SessionOptions::default())?;

    println!("┌─────────────┬────────────────┬───────────────┬───────────────┐");
    println!("│ Batch Size  │ Median (ms)    │ P99 (ms)      │ Throughput/s  │");
    println!("├─────────────┼────────────────┼───────────────┼───────────────┤");

    for &bs in &sizes {
        let inputs = make_dummy_inputs(&session, bs);

        for _ in 0..warmup {
            let _ = session.run(inputs.clone());
        }

        let mut latencies_us: Vec<u64> = (0..iters)
            .map(|_| {
                let t = Instant::now();
                let _ = session.run(inputs.clone());
                t.elapsed().as_micros() as u64
            })
            .collect();

        latencies_us.sort_unstable();
        let median_ms = latencies_us[latencies_us.len() / 2] as f64 / 1000.0;
        let p99_ms = latencies_us[latencies_us.len() * 99 / 100] as f64 / 1000.0;
        let tps = bs as f64 / (median_ms / 1000.0);

        println!(
            "│ {:11} │ {:14.2} │ {:13.2} │ {:13.0} │",
            bs, median_ms, p99_ms, tps
        );
    }

    println!("└─────────────┴────────────────┴───────────────┴───────────────┘");
    Ok(())
}

// ── inspect ───────────────────────────────────────────────────────────────────

async fn inspect_command(model: &str, output_path: Option<PathBuf>) -> Result<()> {
    let model_path = hub::resolve_model_path(model).await?;
    let config = ModelConfig {
        optimize: false,
        infer_shapes: false,
        ..Default::default()
    };
    let model = Model::load(&model_path, config).context("loading model")?;

    let dot = model.graph.to_dot();

    if let Some(p) = output_path {
        std::fs::write(&p, &dot).context("writing DOT file")?;
        println!("DOT graph written to {}", p.display());
    } else {
        print!("{dot}");
    }

    println!(
        "\nGraph: {} nodes, {} edges",
        model.graph.node_count(),
        model.graph.edges.len()
    );
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_input_json(json: &str) -> Result<HashMap<String, Tensor>> {
    #[derive(serde::Deserialize)]
    struct InputSpec {
        shape: Vec<usize>,
        data: Vec<f32>,
    }

    let map: HashMap<String, InputSpec> =
        serde_json::from_str(json).context("invalid input JSON format")?;

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
        .map(|t| {
            json!({
                "shape": t.shape().dims(),
                "dtype": t.dtype().to_string(),
                "data": t.as_f32_slice(),
            })
        })
        .collect();
    serde_json::to_string_pretty(&arr).unwrap_or_default()
}

fn make_dummy_inputs(session: &InferenceSession, _batch_size: usize) -> HashMap<String, Tensor> {
    let mut inputs = HashMap::new();
    for &id in &session.model().graph.inputs {
        use sapient_ir::node::Node;
        if let Some(Node::Input {
            name, shape, dtype, ..
        }) = session.model().graph.get(id)
        {
            let shape = shape
                .clone()
                .unwrap_or_else(|| sapient_core::Shape::new([1]));
            let dtype = dtype.unwrap_or(sapient_core::DType::F32);
            if let Ok(t) = Tensor::zeros(shape, dtype) {
                inputs.insert(name.clone(), t);
            }
        }
    }
    inputs
}
