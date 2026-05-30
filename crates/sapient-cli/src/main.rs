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
use sapient_generate::{
    mac_gpu_support, GenerationBackend, LoadOptions, Pipeline, SpeculativePipeline,
};
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

        /// Load weights from disk on demand via memory-mapping.
        /// Enabled automatically when the model file is larger than available RAM.
        /// Use this flag to force it on smaller devices (e.g. Raspberry Pi).
        #[arg(long)]
        mmap: bool,

        /// Enable speculative decoding with an auto-selected small draft model
        /// (prefers smollm2-135m, falls back to qwen2.5-0.5b).
        /// Expected speedup: 2-4× on generation. Both models are downloaded if needed.
        #[arg(long)]
        speculative: bool,

        /// Draft model to use with --speculative (default: auto-selected).
        #[arg(long, requires = "speculative")]
        draft_model: Option<String>,
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

    /// LLM generation benchmark: measures load time, TTFT, tok/s, and peak RAM.
    /// Outputs a side-by-side comparison table suitable for competing with Ollama.
    #[command(name = "bench-llm", visible_aliases = ["bllm"])]
    BenchLlm {
        /// Model alias (e.g. `openhorizon/qwen2.5-0.5b-q4`) or local .gguf path.
        model: String,

        /// Prompt to use for generation (same prompt repeated across runs).
        #[arg(
            short,
            long,
            default_value = "Explain quantum entanglement in one sentence."
        )]
        prompt: String,

        /// Maximum tokens to generate per run.
        #[arg(long, default_value = "50")]
        max_tokens: usize,

        /// Number of generation runs (more = better statistics).
        #[arg(long, default_value = "3")]
        runs: usize,

        /// Force memory-mapped weight loading.
        #[arg(long)]
        mmap: bool,

        /// Generation backend: auto | cpu | metal.
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Output raw JSON (for scripted comparisons with Ollama).
        #[arg(long)]
        json: bool,
    },

    /// Print graph structure in DOT format (file-based models).
    Inspect {
        /// HuggingFace model ID or path to a model file.
        model: String,

        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Start an OpenAI-compatible HTTP server for LLM inference.
    ///
    /// Models are loaded on-demand from API requests — no model is required at startup.
    /// Optionally pre-load a model for zero-latency on the first request.
    ///
    /// Exposes: GET /v1/models, POST /v1/chat/completions (streaming + non-streaming),
    /// POST /v1/completions. Compatible with any OpenAI client library.
    Serve {
        /// Optional model to pre-load at startup (e.g. `openhorizon/qwen2.5-1.5b-q4`).
        /// If omitted, models are loaded on-demand when first requested via the API.
        model: Option<String>,

        /// Port to listen on.
        #[arg(short, long, default_value = "11435")]
        port: u16,

        /// Generation backend: auto | cpu | metal.
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Load weights via mmap (auto-enabled when model > available RAM).
        #[arg(long)]
        mmap: bool,
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
        Commands::Chat {
            model,
            backend,
            mmap,
            speculative,
            draft_model,
        } => {
            chat_command(
                model.as_str(),
                &backend,
                cli.verbose,
                mmap,
                speculative,
                draft_model.as_deref(),
            )
            .await
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
        Commands::BenchLlm {
            model,
            prompt,
            max_tokens,
            runs,
            mmap,
            backend,
            json,
        } => {
            bench_llm_command(
                model.as_str(),
                &prompt,
                max_tokens,
                runs,
                mmap,
                &backend,
                json,
            )
            .await
        }
        Commands::Inspect { model, output } => inspect_command(model.as_str(), output).await,
        Commands::Serve {
            model,
            port,
            backend,
            mmap,
        } => server::serve_llm(model.as_deref(), port, &backend, mmap).await,
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

async fn chat_command(
    model: &str,
    backend: &str,
    verbose: bool,
    force_mmap: bool,
    speculative: bool,
    draft_model: Option<&str>,
) -> Result<()> {
    // If speculative decoding is requested, branch into the speculative path.
    if speculative {
        return chat_speculative_command(model, backend, verbose, draft_model).await;
    }

    let backend_kind = parse_generation_backend(backend)?;
    let backend_label = backend_kind.to_string();
    let mut load_opts = LoadOptions {
        backend: backend_kind,
        force_mmap,
        ..LoadOptions::default()
    };

    let is_local_gguf = model.ends_with(".gguf") || std::path::Path::new(model).is_file();

    // A model is "already cached" only if it is fully downloaded — no .sync.part
    // files in the blobs directory. A partial download is treated as not cached so
    // we show the download progress bar and hf-hub resumes from where it left off.
    let already_cached = is_local_gguf
        || (hub::list_cached_models()
            .unwrap_or_default()
            .iter()
            .any(|m| m == model)
            && !hub::has_stale_downloads(model));

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
        let gguf_result = if load_opts.force_mmap {
            Pipeline::from_gguf_mmap_with_backend(model, load_opts.backend).await
        } else {
            Pipeline::from_gguf_with_backend(model, load_opts.backend).await
        };
        match gguf_result {
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
    let effective_backend = if pipeline.is_mmap() {
        format!("{backend_label} · mmap")
    } else {
        backend_label.clone()
    };
    ui::print_chat_banner(model, &arch, &effective_backend);

    // Hint: if the user loaded a full-precision safetensors model, suggest the
    // GGUF-quantized alternative which is 4-10× faster on CPU.
    if !model.contains("gguf") && !model.contains("-q4") && !model.contains("-q8") {
        // Look for a quantized alias in the registry (e.g. phi-2 → phi-2-q4)
        let gguf_hint = sapient_hub::registry::catalog()
            .iter()
            .find(|m| {
                (m.alias.contains("-q4") || m.alias.contains("-q8"))
                    && (m.alias.contains(
                        model.rsplit('/').next().unwrap_or(model)
                            .trim_end_matches("-instruct")
                            .trim_end_matches("-chat"),
                    ))
            })
            .map(|m| m.alias);
        if let Some(faster) = gguf_hint {
            ui::hint(format!(
                "For 4-8× faster inference use the quantized version:  sapient chat {faster}"
            ));
        }
    }

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
        let mut ttft: Option<std::time::Duration> = None;
        while let Some(token) = stream.next().await {
            if first {
                ttft = Some(start.elapsed());
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
            ui::print_gen_stats(tokens, start.elapsed(), ttft);
        }
        history.push(ChatMessage::assistant(reply));
    }

    Ok(())
}

// ── Speculative chat ──────────────────────────────────────────────────────────

async fn chat_speculative_command(
    model: &str,
    backend: &str,
    verbose: bool,
    draft_model: Option<&str>,
) -> Result<()> {
    let _ = parse_generation_backend(backend)?; // validate backend string early

    if verbose {
        eprintln!("Loading target model {model} (speculative mode)…");
        if let Some(d) = draft_model {
            eprintln!("Using draft model: {d}");
        } else {
            eprintln!("Auto-selecting draft model…");
        }
    }

    let load_spinner = if verbose {
        None
    } else {
        Some(ui::spinner(format!("loading {model} + draft…")))
    };

    let pipeline = match draft_model {
        Some(draft) => SpeculativePipeline::new(model, draft, 5).await,
        None => SpeculativePipeline::with_auto_draft(model, 5).await,
    }
    .with_context(|| format!("failed to load speculative pipeline for '{model}'"))?;

    if let Some(pb) = load_spinner {
        pb.finish_and_clear();
    }

    ui::print_chat_banner(model, "speculative", backend);

    let mut history: Vec<ChatMessage> = Vec::new();
    loop {
        ui::write_user_prompt()?;

        let mut line = String::new();
        let n = std::io::stdin().read_line(&mut line)?;
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

        let think = ui::spinner("thinking…");
        let start = std::time::Instant::now();
        let mut stream = pipeline.chat_stream(&history).await;
        let mut reply = String::new();
        let mut first = true;
        let mut ttft: Option<std::time::Duration> = None;

        use futures::StreamExt;
        while let Some(token) = stream.next().await {
            if first {
                ttft = Some(start.elapsed());
                think.finish_and_clear();
                ui::write_assistant_prompt()?;
                first = false;
            }
            print!("{token}");
            reply.push_str(&token);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
        if first {
            think.finish_and_clear();
        }
        println!();

        if !reply.is_empty() {
            // We don't have direct tokenizer access from SpeculativePipeline, so
            // approximate token count by whitespace splitting.
            let tokens = reply.split_whitespace().count();
            ui::print_gen_stats(tokens, start.elapsed(), ttft);
        }
        history.push(ChatMessage::assistant(reply));
    }

    Ok(())
}

// ── Pull ──────────────────────────────────────────────────────────────────────

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
            // Detect ENOSPC (os error 28) and auto-clean partial downloads,
            // then re-surface with a clear "what to do" message.
            let msg = e.to_string();
            if msg.contains("os error 28")
                || msg.contains("No space left on device")
                || msg.contains("ENOSPC")
            {
                // Auto-clean the orphaned .sync.part files left by the failed download.
                let freed = hub::clear_stale_downloads().unwrap_or(0);
                let freed_str = if freed > 0 {
                    format!(
                        "\n  {} of incomplete download files were automatically removed.",
                        hub::format_bytes(freed)
                    )
                } else {
                    String::new()
                };
                anyhow::bail!(
                    "Disk full while downloading '{model}'.{freed_str}\n\n\
                     To free more space:\n\
                     \n  sapient reset --stale          # remove all partial downloads\
                     \n  sapient reset {model}           # remove this model entirely\
                     \n  sapient reset                  # clear all cached models\
                     \n\nOr retry with a smaller quant (Q4_K_M uses ~half the disk of Q8_0)."
                );
            }
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

// ── bench-llm ─────────────────────────────────────────────────────────────────

/// Current process resident set size in bytes.
/// Linux: reads /proc/self/status VmRSS.
/// macOS: spawns `ps -o rss= -p PID` (no libc required).
fn resident_set_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Ok(kb) = rest.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                        return kb * 1024;
                    }
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        let pid = std::process::id();
        if let Ok(out) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
        {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(kb) = s.trim().parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
async fn bench_llm_command(
    model: &str,
    prompt: &str,
    max_tokens: usize,
    runs: usize,
    force_mmap: bool,
    backend: &str,
    json_out: bool,
) -> Result<()> {
    let backend_kind = parse_generation_backend(backend)?;

    // ── Load model (timed) ──────────────────────────────────────────────────
    let load_start = Instant::now();
    let opts = LoadOptions {
        backend: backend_kind,
        force_mmap,
        generation: sapient_generate::GenerationConfig {
            max_new_tokens: max_tokens,
            ..Default::default()
        },
        ..LoadOptions::default()
    };

    let load_spinner = (!json_out).then(|| ui::spinner(format!("loading {model}…")));
    let pipeline = Pipeline::from_pretrained_with_opts(model, opts)
        .await
        .with_context(|| format!("failed to load model '{model}'"))?;
    let load_ms = load_start.elapsed().as_millis() as u64;

    if let Some(pb) = load_spinner {
        pb.finish_and_clear();
    }

    let backend_label = format!(
        "{}{}",
        backend,
        if pipeline.is_mmap() { " · mmap" } else { "" }
    );

    // ── Benchmark runs ──────────────────────────────────────────────────────
    let messages = vec![ChatMessage::user(prompt)];
    let mut run_results: Vec<ui::BenchRun> = Vec::with_capacity(runs);
    let mut json_runs: Vec<serde_json::Value> = Vec::with_capacity(runs);

    for i in 0..runs {
        pipeline.reset_cache();

        let gen_start = Instant::now();
        let mut stream = pipeline.chat_stream(&messages).await;

        let mut reply = String::new();
        let mut ttft_ms = 0u64;
        let mut first = true;

        while let Some(chunk) = stream.next().await {
            if first {
                ttft_ms = gen_start.elapsed().as_millis() as u64;
                first = false;
            }
            reply.push_str(&chunk);
        }

        let elapsed_ms = gen_start.elapsed().as_millis() as u64;
        let total_tokens = pipeline
            .tokenizer()
            .encode(&reply)
            .map(|t| t.len())
            .unwrap_or_default();
        let tps = if elapsed_ms > 0 {
            total_tokens as f64 / (elapsed_ms as f64 / 1000.0)
        } else {
            0.0
        };

        run_results.push(ui::BenchRun {
            run: i + 1,
            ttft_ms,
            tps,
            total_tokens,
        });
        json_runs.push(serde_json::json!({
            "run": i + 1,
            "ttft_ms": ttft_ms,
            "elapsed_ms": elapsed_ms,
            "total_tokens": total_tokens,
            "tps": (tps * 10.0).round() / 10.0,
        }));
    }

    let peak_rss_mb = resident_set_bytes() / (1024 * 1024);

    // ── Output ──────────────────────────────────────────────────────────────
    if json_out {
        let mean_ttft = if run_results.is_empty() {
            0
        } else {
            run_results.iter().map(|r| r.ttft_ms).sum::<u64>() / run_results.len() as u64
        };
        let mean_tps = if run_results.is_empty() {
            0.0
        } else {
            run_results.iter().map(|r| r.tps).sum::<f64>() / run_results.len() as f64
        };
        let out = serde_json::json!({
            "model": model,
            "backend": backend_label,
            "mmap": pipeline.is_mmap(),
            "load_time_ms": load_ms,
            "prompt": prompt,
            "runs": json_runs,
            "summary": {
                "mean_ttft_ms": mean_ttft,
                "mean_tps": (mean_tps * 10.0).round() / 10.0,
                "peak_rss_mb": peak_rss_mb,
            }
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        ui::print_bench_table(model, &backend_label, load_ms, &run_results, peak_rss_mb);
    }

    Ok(())
}
