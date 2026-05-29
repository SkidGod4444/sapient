//! SAPIENT CLI — chat, pull, run, bench, inspect, serve

mod hub;
mod progress;
mod server;
mod ui;
mod update;

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use sapient_generate::{mac_gpu_support, GenerationBackend, LoadOptions, Pipeline};
use sapient_hub::LoadOptions as HubLoadOptions;
use sapient_runtime::{InferenceSession, Model, ModelConfig, SessionOptions};
use sapient_telemetry::init_tracing;
use sapient_tokenizers::ChatMessage;

use sapient_core::Tensor;
use sapient_hub::HubClient;

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

        /// Generation backend: auto | cpu | metal.
        #[arg(short, long, default_value = "auto")]
        backend: String,
    },

    /// Download a model from HuggingFace Hub to the local cache.
    Pull {
        /// HuggingFace model ID.
        model: String,
    },

    /// List models downloaded to the local cache.
    List,

    /// List all models SAPIENT supports (the registry catalog).
    #[command(visible_aliases = ["available", "catalog"])]
    Models,

    /// Remove one cached model from this device.
    #[command(name = "rm", visible_aliases = ["remove"])]
    Rm {
        /// HuggingFace model ID to remove (e.g. `microsoft/phi-2`).
        model: String,
    },

    /// Remove cached models from this device.
    Reset {
        /// HuggingFace model ID to remove (omit to clear all cached models).
        model: Option<String>,

        /// Skip confirmation when clearing all cached models.
        #[arg(short = 'y', long)]
        yes: bool,

        /// Only remove incomplete downloads (`.sync.part` / `.lock` files).
        #[arg(long)]
        stale: bool,
    },

    /// Show architecture and config info for a HuggingFace model.
    Info {
        /// HuggingFace model ID.
        model: String,
    },

    /// Show available local inference backends.
    BackendInfo,

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

        /// Backend: auto | cpu | metal | vulkan.
        #[arg(short, long, default_value = "auto")]
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

    /// Update sapient to the latest release from GitHub.
    Update {
        /// Reinstall even if already on the latest version.
        #[arg(long)]
        force: bool,

        /// Install the Apple Silicon Metal (GPU) build.
        #[arg(long, conflicts_with = "cpu")]
        metal: bool,

        /// Install the CPU build (skip Metal even on Apple Silicon).
        #[arg(long)]
        cpu: bool,
    },
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.json_logs, cli.verbose);

    match dispatch(cli).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            ui::failure(format!("{e:#}"));
            std::process::ExitCode::FAILURE
        }
    }
}

async fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Chat { model, backend } => {
            chat_command(model.as_str(), &backend, cli.verbose).await
        }
        Commands::Pull { model } => pull_command(model.as_str(), cli.verbose).await,
        Commands::List => list_command(),
        Commands::Models => models_command(),
        Commands::Rm { model } => rm_command(model.as_str()),
        Commands::Reset { model, yes, stale } => reset_command(model.as_deref(), yes, stale),
        Commands::Info { model } => info_command(model.as_str()).await,
        Commands::BackendInfo => backend_info_command(),
        Commands::Login { token } => login_command(token.as_deref()),
        Commands::Run {
            model,
            prompt,
            input,
            output,
            backend,
            telemetry,
        } => {
            run_command(
                model.as_str(),
                prompt,
                input,
                output,
                backend,
                telemetry,
                cli.verbose,
            )
            .await
        }
        Commands::Bench {
            model,
            batch_sizes,
            backend,
            warmup,
            iters,
        } => bench_command(model.as_str(), &batch_sizes, backend, warmup, iters).await,
        Commands::Inspect { model, output } => inspect_command(model.as_str(), output).await,
        Commands::Serve {
            model,
            port,
            backend,
            workers,
        } => {
            let model_path = hub::resolve_model_path(model.as_str()).await?;
            server::serve(model_path, port, backend, workers).await
        }
        Commands::Update { force, metal, cpu } => {
            let variant = if metal {
                Some(update::Variant::Metal)
            } else if cpu {
                Some(update::Variant::Cpu)
            } else {
                None
            };
            update::run_update(force, variant)
        }
    }
}

// ── Hub commands ──────────────────────────────────────────────────────────────

async fn chat_command(model: &str, backend: &str, verbose: bool) -> Result<()> {
    let backend_kind = parse_generation_backend(backend)?;
    let backend_label = backend_kind.to_string();
    let mut load_opts = LoadOptions {
        backend: backend_kind,
        ..LoadOptions::default()
    };

    let is_local_gguf = model.ends_with(".gguf") || std::path::Path::new(model).is_file();

    // If the model isn't cached and we need to download, show a live download
    // progress bar with bytes + speed instead of a silent spinner.
    let already_cached = is_local_gguf
        || hub::list_cached_models()
            .unwrap_or_default()
            .iter()
            .any(|m| m == model);

    let dl_handle = if !already_cached && !verbose {
        let hub_check = HubClient::with_options(sapient_hub::LoadOptions {
            quiet: true,
            ..Default::default()
        })?;
        let total_bytes = hub_check.repo_total_bytes(model).await.unwrap_or(0);
        let blobs_dir = HubClient::blobs_dir_for_model(model);
        load_opts.hub.quiet = true; // progress bar takes over
        Some(progress::start_download_progress(
            model,
            blobs_dir,
            total_bytes,
        ))
    } else {
        load_opts.hub.quiet = false;
        None
    };

    let load_spinner =
        (already_cached && !verbose).then(|| ui::spinner(format!("loading {model}…")));
    if verbose {
        eprintln!("Loading {model} with backend {backend_label}…");
    }

    let pipeline = if is_local_gguf {
        // Local GGUF file: load directly without Hub download.
        match Pipeline::from_gguf_with_backend(model, load_opts.backend).await {
            Ok(p) => p,
            Err(e) => {
                if let Some(h) = dl_handle {
                    h.finish_error();
                }
                if let Some(pb) = load_spinner {
                    pb.finish_and_clear();
                }
                return Err(e).with_context(|| format!("failed to load GGUF '{model}'"));
            }
        }
    } else {
        match Pipeline::from_pretrained_with_opts(model, load_opts).await {
            Ok(p) => p,
            Err(e) => {
                if let Some(h) = dl_handle {
                    h.finish_error();
                }
                if let Some(pb) = load_spinner {
                    pb.finish_and_clear();
                }
                return Err(e).with_context(|| format!("failed to load model '{model}'"));
            }
        }
    };

    if let Some(h) = dl_handle {
        h.finish_success(model);
    }
    if let Some(pb) = load_spinner {
        pb.finish_and_clear();
    }

    let arch = format!("{:?}", pipeline.arch());
    ui::print_chat_banner(model, &arch, &backend_label);

    let mut history: Vec<ChatMessage> = Vec::new();
    loop {
        ui::write_user_prompt()?;

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
        if matches!(line, "/help" | "/?") {
            ui::print_chat_help();
            continue;
        }
        if matches!(line, "/clear" | "/reset") {
            history.clear();
            ui::hint("conversation history cleared");
            continue;
        }

        history.push(ChatMessage::user(line));

        // Spinner shows "thinking" until the first token arrives, then is cleared
        // and replaced by the assistant prompt + streamed reply.
        let think = ui::spinner("thinking…");
        let start = std::time::Instant::now();
        let mut stream = pipeline.chat_stream(&history).await;
        let mut reply = String::new();
        let mut first = true;
        while let Some(token) = stream.next().await {
            if first {
                think.finish_and_clear();
                ui::write_assistant_prompt()?;
                first = false;
            }
            print!("{token}");
            reply.push_str(&token);
            io::stdout().flush()?;
        }
        if first {
            think.finish_and_clear();
        }
        println!();
        if !reply.is_empty() {
            let tokens = pipeline
                .tokenizer()
                .encode(&reply)
                .map(|t| t.len())
                .unwrap_or(0);
            ui::print_gen_stats(tokens, start.elapsed());
        }
        history.push(ChatMessage::assistant(reply));
    }

    Ok(())
}

async fn pull_command(model: &str, verbose: bool) -> Result<()> {
    if verbose {
        println!("Pulling {model}…");
    }

    // Fetch the total expected size and blobs dir *before* starting the download
    // so the progress bar can show a real percentage.
    let hub = HubClient::with_options(sapient_hub::LoadOptions {
        quiet: false,
        ..Default::default()
    })?;

    let total_bytes = hub.repo_total_bytes(model).await.unwrap_or(0);
    let blobs_dir = HubClient::blobs_dir_for_model(model);

    let handle = progress::start_download_progress(model, blobs_dir, total_bytes);

    let result = if verbose {
        hub::pull_model_with_options(
            model,
            HubLoadOptions {
                quiet: false,
                ..Default::default()
            },
        )
        .await
    } else {
        hub::pull_model(model).await
    };

    match result {
        Ok(files) => {
            handle.finish_success(model);
            if verbose {
                println!("  config:    {}", files.config_path.display());
                if let Some(tok) = &files.tokenizer_path {
                    println!("  tokenizer: {}", tok.display());
                }
                for w in &files.weight_paths {
                    println!("  weights:   {}", w.display());
                }
            }
            ui::hint(format!("start chatting with:  sapient chat {model}"));
            Ok(())
        }
        Err(e) => {
            handle.finish_error();
            Err(e)
        }
    }
}

fn list_command() -> Result<()> {
    let models = hub::list_cached_models()?;
    if models.is_empty() {
        ui::hint("No models downloaded yet.");
        println!("  Pull one with:  sapient pull openhorizon/phi-2");
        println!("  See all models: sapient models");
        return Ok(());
    }

    let catalog = sapient_hub::registry::catalog();
    let rows: Vec<Vec<String>> = models
        .iter()
        .map(|alias| {
            let meta = catalog.iter().find(|m| m.alias == alias);
            vec![
                alias.clone(),
                meta.map(|m| m.family.to_string()).unwrap_or_default(),
                meta.map(|m| m.params.to_string()).unwrap_or_default(),
            ]
        })
        .collect();

    println!("\nDownloaded models ({})\n", models.len());
    ui::print_table(&["MODEL", "FAMILY", "SIZE"], &rows);
    println!();
    Ok(())
}

fn models_command() -> Result<()> {
    let catalog = sapient_hub::registry::catalog();
    let cached = hub::list_cached_models().unwrap_or_default();

    let rows: Vec<Vec<String>> = catalog
        .iter()
        .map(|m| {
            let status = if cached.iter().any(|c| c == m.alias) {
                "downloaded".to_string()
            } else if m.gated {
                "gated".to_string()
            } else {
                "—".to_string()
            };
            vec![
                m.alias.to_string(),
                m.family.to_string(),
                m.params.to_string(),
                status,
            ]
        })
        .collect();

    println!("\nSupported models ({})\n", catalog.len());
    ui::print_table(&["MODEL", "FAMILY", "SIZE", "STATUS"], &rows);
    println!();
    ui::hint("run any of these with:  sapient chat <model>");
    Ok(())
}

fn rm_command(model: &str) -> Result<()> {
    let bytes = hub::clear_cached_model(model)?;
    ui::success(format!(
        "Removed {model} from cache ({} freed).",
        hub::format_bytes(bytes)
    ));
    Ok(())
}

fn reset_command(model: Option<&str>, yes: bool, stale_only: bool) -> Result<()> {
    if stale_only {
        let freed = hub::clear_stale_downloads()?;
        if freed == 0 {
            println!("No incomplete downloads found.");
        } else {
            println!(
                "Removed incomplete downloads ({} freed).",
                hub::format_bytes(freed)
            );
        }
        return Ok(());
    }

    if let Some(model_id) = model {
        let bytes = hub::clear_cached_model(model_id)?;
        println!(
            "Removed {model_id} from cache ({} freed).",
            hub::format_bytes(bytes)
        );
        return Ok(());
    }

    let models = hub::list_cached_models()?;
    if models.is_empty() {
        println!("No cached models to remove.");
        return Ok(());
    }

    if !yes {
        print!(
            "Remove all {} cached model(s)? This cannot be undone. [y/N] ",
            models.len()
        );
        io::stdout().flush()?;

        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        let answer = answer.trim().to_ascii_lowercase();
        if answer != "y" && answer != "yes" {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let (count, bytes) = hub::clear_all_cached_models()?;
    println!(
        "Removed {count} cached model(s) ({} freed).",
        hub::format_bytes(bytes)
    );
    Ok(())
}

async fn info_command(model: &str) -> Result<()> {
    let info = hub::fetch_model_info(model)
        .await
        .with_context(|| format!("failed to fetch info for '{model}'"))?;

    ui::print_logo();
    println!();
    ui::info_row("model", model);
    ui::info_row("type", &info.model_type);
    ui::info_row("arch", format!("{:?}", info.arch));
    ui::info_row("layers", info.num_hidden_layers);
    ui::info_row("hidden", info.hidden_size);
    ui::info_row("heads", info.num_attention_heads);
    ui::info_row("kv heads", info.num_key_value_heads);
    ui::info_row("vocab", info.vocab_size);
    ui::info_row("context", info.max_position_embeddings);
    println!();
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
    ui::success(format!("Token saved to {}", path.display()));
    ui::hint("Gated models (Llama, Mistral, etc.) are now accessible.");
    Ok(())
}

fn backend_info_command() -> Result<()> {
    let ram = query_system_ram_bytes();
    fn query_system_ram_bytes() -> u64 {
        #[cfg(target_os = "macos")]
        {
            if let Ok(out) = std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
            {
                if let Ok(s) = std::str::from_utf8(&out.stdout) {
                    if let Ok(n) = s.trim().parse::<u64>() {
                        return n;
                    }
                }
            }
        }
        0
    }
    let _ = ram; // may be 0 on non-mac
    let ram = query_system_ram_bytes();
    let ram_gb = ram as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("\n  {}", console::style("Hardware").bold());
    if ram > 0 {
        println!(
            "    Unified memory:  {:.0} GB{}",
            ram_gb,
            if cfg!(target_os = "macos") {
                " (Apple Silicon shared CPU/GPU pool)"
            } else {
                ""
            }
        );
    }

    println!("\n  {}", console::style("Backends").bold());
    println!("    cpu    always available");

    let gpu = mac_gpu_support();
    if gpu.available {
        #[cfg(target_os = "macos")]
        let device = sapient_backends_metal::MlxBackend::device_name()
            .unwrap_or_else(|| "Apple Silicon".to_string());
        #[cfg(not(target_os = "macos"))]
        let device = "GPU".to_string();

        let auto_target = if ram > 0 {
            format!(
                "metal ({device}) — fits up to {:.0} GB models",
                (ram as f64 - 2e9) / (1.5 * 1e9)
            )
        } else {
            format!("metal ({device})")
        };
        println!("    metal  available — {device}");
        println!("    auto   → {auto_target}");
        println!(
            "\n  {} sapient chat --backend metal <model>",
            console::style("Tip:").dim()
        );
    } else {
        println!("    metal  unavailable ({})", gpu.reason);
        println!("    auto   → cpu");
    }
    println!();

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
    verbose: bool,
) -> Result<()> {
    if hub::looks_like_hub_model_id(model) {
        let prompt = prompt.with_context(|| {
            format!(
                "Hub model '{model}' requires --prompt.\n\
                 For interactive chat use: sapient chat {model}"
            )
        })?;
        let mut load_opts = LoadOptions::default();
        load_opts.hub.quiet = false; // Always show progress!
        load_opts.backend = parse_generation_backend(&backend)?;
        if verbose {
            eprintln!("Loading {model} with backend {}…", load_opts.backend);
        } else {
            eprintln!("Loading model {model}…");
        }
        let pipeline = Pipeline::from_pretrained_with_opts(model, load_opts).await?;
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

fn parse_generation_backend(value: &str) -> Result<GenerationBackend> {
    value
        .parse()
        .with_context(|| format!("invalid backend '{value}'; expected auto, cpu, or metal"))
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
