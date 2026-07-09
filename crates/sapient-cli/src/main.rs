//! SAPIENT CLI — chat, pull, run, bench, inspect, serve

mod hub;
mod markdown;
mod progress;
mod server;
mod stats;
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
    detect_devices, mac_gpu_support, recommend_backend, GenerationBackend, LoadOptions, Pipeline,
    SpeakPipeline, SpeculativePipeline, TranscribeOptions, TranscribePipeline, ORPHEUS_VOICES,
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

        /// Generation backend: auto | cpu | metal | wgpu.
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

        /// Print replies as raw Markdown text instead of rendering it in the
        /// terminal (rendering is on by default on interactive terminals).
        #[arg(long)]
        raw: bool,

        /// Run a single chat turn for this prompt and exit (non-interactive).
        /// Applies the model's chat template + end-of-turn stopping, so unlike
        /// `run` the reply is a clean, bounded answer — and prints only the reply
        /// to stdout, so it's easy to script (e.g. feed into `sapient speak`).
        #[arg(short, long)]
        prompt: Option<String>,
    },

    /// Transcribe an audio file to text with a Whisper model (speech-to-text).
    #[command(visible_aliases = ["stt", "asr"])]
    Transcribe {
        /// Whisper model alias or repo id (e.g. `whisper-base`, `openai/whisper-small`).
        model: String,

        /// Path to the audio file (WAV/FLAC/MP3/OGG/M4A).
        audio: PathBuf,

        /// Generation backend: auto | cpu | metal | wgpu.
        /// `auto` uses the binary's compiled accelerator (GPU build → GPU).
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Source language code (e.g. `en`, `fr`). Omit to auto-detect.
        #[arg(long)]
        language: Option<String>,

        /// Translate to English instead of transcribing in the source language.
        #[arg(long)]
        translate: bool,

        /// Emit timestamp tokens and use them to re-seek long audio (>30 s).
        #[arg(long)]
        timestamps: bool,

        /// Beam width (1 = greedy, the default). Higher is slower but can be
        /// more accurate.
        #[arg(long, default_value_t = 1)]
        beam_size: usize,
    },

    /// Real-time voice conversation: mic → speech-to-text → LLM → reply.
    /// Requires the `audio-io` build feature (mic capture).
    Converse {
        /// LLM model alias/repo for the chat replies.
        model: String,

        /// Whisper STT model for transcription.
        #[arg(long, default_value = "whisper-base")]
        stt: String,

        /// Generation backend: auto | cpu | metal | wgpu.
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Force the spoken language (otherwise auto-detected per utterance).
        #[arg(long)]
        language: Option<String>,

        /// System prompt to seed the conversation.
        #[arg(long)]
        system: Option<String>,

        /// Also speak the reply aloud (text replies stream either way). Off by default.
        #[arg(long)]
        speak: bool,

        /// Voice engine for spoken replies (with --speak): `kokoro` (Kokoro-82M,
        /// real-time on CPU — default) or `orpheus` (Orpheus-3B, richer but slow,
        /// not real-time). NOTE: this is the text-to-SPEECH model; `--stt` is the
        /// separate speech-to-TEXT (Whisper) model.
        #[arg(long, default_value = "kokoro")]
        tts: String,

        /// Run a single turn from a WAV/audio file instead of the live mic (no
        /// microphone needed), printing per-stage timing (STT / LLM / TTS). Use it
        /// to benchmark the converse pipeline on headless/mic-less devices.
        #[arg(long)]
        input: Option<PathBuf>,
    },

    /// Synthesise speech from text with a TTS model (text-to-speech).
    #[command(visible_aliases = ["tts", "say"])]
    /// Ask a vision-language model about an image (SmolVLM — Phase 12)
    See {
        /// Image file (png/jpeg/webp).
        image: PathBuf,

        /// Question / instruction about the image.
        #[arg(short, long, default_value = "Describe this image.")]
        prompt: String,

        /// VLM model alias or Idefics3-family repo id.
        #[arg(short, long, default_value = "smolvlm-256m")]
        model: String,

        /// Maximum new tokens in the answer.
        #[arg(long, default_value_t = 192)]
        max_tokens: usize,
    },

    Speak {
        /// Orpheus model alias or repo id (e.g. `orpheus-3b`).
        model: String,

        /// Text to speak.
        text: String,

        /// Output WAV file path.
        #[arg(short, long, default_value = "speech.wav")]
        output: PathBuf,

        /// Voice: tara | leah | jess | leo | dan | mia | zac | zoe.
        #[arg(long, default_value = "tara")]
        voice: String,

        /// Generation backend: auto | cpu | metal | wgpu.
        /// `auto` uses the binary's compiled accelerator (GPU build → GPU).
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Write the WAV but don't play it through the speaker.
        #[arg(long)]
        no_play: bool,
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
    #[command(hide = true)]
    BackendInfo,

    /// Detect all CPUs and GPUs, show memory/bandwidth, and recommend backends.
    ///
    /// Reports which models fit in GPU memory, whether hybrid CPU+GPU execution
    /// is possible, and expected tok/s for common model sizes.
    #[command(hide = true)]
    Devices,

    /// Live resource monitor — CPU cores, RAM, and disk used by SAPIENT.
    #[command(visible_aliases = ["top", "monitor"])]
    Stats,

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

        /// Backend: auto | cpu | metal | wgpu.
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Enable console telemetry.
        #[arg(long)]
        telemetry: bool,
    },

    /// Benchmark a model across batch sizes (file-based models).
    #[command(hide = true)]
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
    #[command(name = "bench-llm", visible_aliases = ["bllm"], hide = true)]
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

        /// Generation backend: auto | cpu | metal | wgpu.
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

        /// Generation backend: auto | cpu | metal | wgpu.
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Load weights via mmap (auto-enabled when model > available RAM).
        #[arg(long)]
        mmap: bool,

        /// Keep up to N most-recently-used models resident in memory, so
        /// switching back to a recent model is instant (no reload). LRU-evicted.
        #[arg(long, default_value = "3")]
        max_models: usize,

        /// Cap total resident model memory (GB). 0 = derive from system RAM.
        /// Evicts least-recently-used models when exceeded (in addition to --max-models).
        #[arg(long, default_value = "0")]
        cache_gb: f64,

        /// Max concurrent inferences (admission control). 0 = auto (CPU count, capped).
        #[arg(long, default_value = "0")]
        max_concurrency: usize,

        /// Serve every model with speculative decoding (target = requested model,
        /// draft = auto-selected small model unless --draft-model is given).
        #[arg(long)]
        speculative: bool,

        /// Draft model to use with --speculative (default: auto-selected).
        #[arg(long, requires = "speculative")]
        draft_model: Option<String>,
    },

    /// Update sapient to the latest release from GitHub.
    Update {
        /// Reinstall even if already on the latest version.
        #[arg(long)]
        force: bool,

        /// Install the Apple Silicon Metal (GPU) build.
        #[arg(long, conflicts_with_all = ["cpu", "gpu"])]
        metal: bool,

        /// Install the cross-platform GPU build (wgpu: Vulkan/DX12 — Intel/AMD/Nvidia).
        #[arg(long, conflicts_with = "cpu")]
        gpu: bool,

        /// Install the CPU build (skip any GPU build).
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
            raw,
            prompt,
        } => {
            chat_command(
                model.as_str(),
                &backend,
                cli.verbose,
                mmap,
                speculative,
                draft_model.as_deref(),
                raw,
                prompt.as_deref(),
            )
            .await
        }
        Commands::Transcribe {
            model,
            audio,
            backend,
            language,
            translate,
            timestamps,
            beam_size,
        } => {
            transcribe_command(
                model.as_str(),
                &audio,
                &backend,
                language,
                translate,
                timestamps,
                beam_size,
            )
            .await
        }
        Commands::Converse {
            model,
            stt,
            backend,
            language,
            system,
            speak,
            tts,
            input,
        } => {
            converse_command(
                model.as_str(),
                stt.as_str(),
                &backend,
                language,
                system,
                speak,
                tts.as_str(),
                input,
            )
            .await
        }
        Commands::See {
            image,
            prompt,
            model,
            max_tokens,
        } => see_command(&image, &prompt, &model, max_tokens).await,
        Commands::Speak {
            model,
            text,
            output,
            voice,
            backend,
            no_play,
        } => speak_command(model.as_str(), &text, &output, &voice, &backend, no_play).await,
        Commands::Pull { model } => pull_command(model.as_str(), cli.verbose).await,
        Commands::List => list_command(),
        Commands::Models => models_command(),
        Commands::Rm { model } => rm_command(model.as_str()),
        Commands::Reset { model, yes, stale } => reset_command(model.as_deref(), yes, stale),
        Commands::Info { model } => info_command(model.as_str()).await,
        Commands::BackendInfo => backend_info_command(),
        Commands::Devices => devices_command(),
        Commands::Stats => stats::run().await,
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
            max_models,
            cache_gb,
            max_concurrency,
            speculative,
            draft_model,
        } => {
            server::serve_llm(
                model.as_deref(),
                port,
                &backend,
                mmap,
                max_models,
                cache_gb,
                max_concurrency,
                speculative,
                draft_model.as_deref(),
            )
            .await
        }
        Commands::Update {
            force,
            metal,
            gpu,
            cpu,
        } => {
            let variant = if metal {
                Some(update::Variant::Metal)
            } else if gpu {
                Some(update::Variant::Gpu)
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

/// Read one line of chat input with a bracketed-paste-aware line editor.
///
/// Plain `stdin().read_line()` returns the instant it sees a newline, so any
/// pasted text that contains (or ends with) `\n` is submitted immediately —
/// often before the user presses Enter. `rustyline` enables bracketed-paste
/// mode, so a paste is inserted into the edit buffer as literal text (newlines
/// included) and only a real Enter key submits.
///
/// Returns `Ok(None)` on EOF / Ctrl-C / Ctrl-D — the caller should break.
fn read_chat_line(editor: &mut rustyline::DefaultEditor) -> Result<Option<String>> {
    use rustyline::error::ReadlineError;
    match editor.readline(&ui::user_prompt_str()) {
        Ok(line) => {
            if !line.trim().is_empty() {
                let _ = editor.add_history_entry(line.as_str());
            }
            Ok(Some(line))
        }
        Err(ReadlineError::Interrupted | ReadlineError::Eof) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[allow(clippy::too_many_arguments)]
async fn chat_command(
    model: &str,
    backend: &str,
    verbose: bool,
    force_mmap: bool,
    speculative: bool,
    draft_model: Option<&str>,
    raw: bool,
    prompt: Option<&str>,
) -> Result<()> {
    // If speculative decoding is requested, branch into the speculative path.
    // (One-shot --prompt is handled by the standard path below.)
    if speculative && prompt.is_none() {
        return chat_speculative_command(model, backend, verbose, draft_model, raw).await;
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

    // One-shot mode: run a single chat turn (chat template + end-of-turn EOS, so
    // the reply is bounded and well-formed) and print ONLY the reply to stdout so
    // it's clean to capture in a script. Status/spinners already go to stderr.
    if let Some(p) = prompt {
        let reply = pipeline
            .chat(&[ChatMessage::user(p)])
            .await
            .context("one-shot chat generation failed")?;
        println!("{}", reply.trim());
        return Ok(());
    }

    let arch = format!("{:?}", pipeline.arch());
    // Use the pipeline's own label — shows "metal+cpu hybrid (24/32 layers on GPU)"
    // in hybrid mode, or the plain backend name otherwise.
    let display_label = pipeline.backend_display_label();
    let effective_backend = if pipeline.is_mmap() {
        format!("{display_label} · mmap")
    } else {
        display_label
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
                        model
                            .rsplit('/')
                            .next()
                            .unwrap_or(model)
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

    let mut editor = rustyline::DefaultEditor::new()?;
    let mut history: Vec<ChatMessage> = Vec::new();
    loop {
        let Some(input) = read_chat_line(&mut editor)? else {
            break;
        };

        let line = input.trim();
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
        // and replaced by the assistant prompt + the live Markdown-rendered reply.
        let think = ui::spinner("generating…");
        let start = std::time::Instant::now();
        let mut stream = pipeline.chat_stream(&history).await;
        let mut renderer = markdown::StreamRenderer::new(raw);
        let mut first = true;
        let mut ttft: Option<std::time::Duration> = None;
        while let Some(token) = stream.next().await {
            if first {
                ttft = Some(start.elapsed());
                think.finish_and_clear();
                ui::write_assistant_prompt()?;
                renderer.begin()?;
                first = false;
            }
            renderer.push(&token)?;
        }
        if first {
            think.finish_and_clear();
        } else {
            renderer.finish()?;
        }
        let reply = renderer.into_text();
        if !reply.trim().is_empty() {
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
    raw: bool,
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

    let mut editor = rustyline::DefaultEditor::new()?;
    let mut history: Vec<ChatMessage> = Vec::new();
    loop {
        let Some(input) = read_chat_line(&mut editor)? else {
            break;
        };

        let line = input.trim();
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

        let think = ui::spinner("generating…");
        let start = std::time::Instant::now();
        let mut stream = pipeline.chat_stream(&history).await;
        let mut renderer = markdown::StreamRenderer::new(raw);
        let mut first = true;
        let mut ttft: Option<std::time::Duration> = None;

        use futures::StreamExt;
        while let Some(token) = stream.next().await {
            if first {
                ttft = Some(start.elapsed());
                think.finish_and_clear();
                ui::write_assistant_prompt()?;
                renderer.begin()?;
                first = false;
            }
            renderer.push(&token)?;
        }
        if first {
            think.finish_and_clear();
        } else {
            renderer.finish()?;
        }

        let reply = renderer.into_text();
        if !reply.trim().is_empty() {
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

/// Estimate the download size (GB) of a catalog model from its `params` label
/// (e.g. "7B Q4_K_M", "360M Q8_0", "1.5B"). Uses bits-per-weight for the quant:
/// Q4_K≈4.8, Q5_K≈5.6, Q6_K≈6.6, Q8_0≈8.5, otherwise BF16/F16 safetensors ≈16.
fn estimate_download_gb(params: &str) -> f64 {
    let lower = params.to_ascii_lowercase();
    // Parameter count (billions). Accept "7b", "0.5b", "360m".
    let mut billions = 0.0f64;
    for tok in lower.split_whitespace() {
        if let Some(num) = tok.strip_suffix('b') {
            if let Ok(v) = num.parse::<f64>() {
                billions = v;
                break;
            }
        } else if let Some(num) = tok.strip_suffix('m') {
            if let Ok(v) = num.parse::<f64>() {
                billions = v / 1000.0;
                break;
            }
        }
    }
    if billions <= 0.0 {
        return 0.0;
    }
    let bpw = if lower.contains("q4_k") || lower.contains("q4_0") {
        4.8
    } else if lower.contains("q5_k") || lower.contains("q5_0") {
        5.6
    } else if lower.contains("q6_k") {
        6.6
    } else if lower.contains("q8_0") {
        8.5
    } else {
        16.0 // safetensors F16/BF16
    };
    // GB = params × bits/weight ÷ 8 bits/byte (≈ GiB; close enough for a label).
    billions * bpw / 8.0
}

/// Render an approximate GB string, e.g. "~4.4 GB" (or "<0.1 GB" / "—").
fn fmt_gb(gb: f64) -> String {
    if gb <= 0.0 {
        "—".to_string()
    } else if gb < 0.1 {
        "<0.1 GB".to_string()
    } else {
        format!("~{gb:.1} GB")
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
            // Actual on-disk size for downloaded models.
            let bytes = hub::cached_model_size(alias);
            let disk = if bytes > 0 {
                format!("{:.1} GB", bytes as f64 / 1e9)
            } else {
                "—".to_string()
            };
            vec![
                alias.clone(),
                meta.map(|m| m.family.to_string()).unwrap_or_default(),
                meta.map(|m| m.params.to_string()).unwrap_or_default(),
                disk,
            ]
        })
        .collect();

    println!("\nDownloaded models ({})\n", models.len());
    ui::print_table(&["MODEL", "FAMILY", "SIZE", "ON DISK"], &rows);
    println!();
    Ok(())
}

fn models_command() -> Result<()> {
    use sapient_hub::registry::ModelCategory;

    let catalog = sapient_hub::registry::catalog();
    let cached = hub::list_cached_models().unwrap_or_default();

    let row_for = |m: &sapient_hub::registry::SupportedModel| -> Vec<String> {
        let status = if cached.iter().any(|c| c == m.alias) {
            "downloaded".to_string()
        } else if m.gated {
            "gated".to_string()
        } else {
            "—".to_string()
        };
        // Show the real on-disk size if downloaded, else an estimate.
        let cached_bytes = hub::cached_model_size(m.alias);
        let download = if cached_bytes > 0 {
            format!("{:.1} GB", cached_bytes as f64 / 1e9)
        } else {
            fmt_gb(estimate_download_gb(m.params))
        };
        vec![
            m.alias.to_string(),
            m.family.to_string(),
            m.params.to_string(),
            download,
            status,
        ]
    };

    println!("\nSupported models ({})", catalog.len());

    // Group into capability sections so STT/TTS models are clearly separated
    // from chat models (and from each other).
    let sections = [
        (ModelCategory::Chat, "sapient chat <model>"),
        (
            ModelCategory::SpeechToText,
            "sapient transcribe <model> <audio>",
        ),
        (
            ModelCategory::TextToSpeech,
            "sapient speak <model> \"<text>\"",
        ),
        (
            ModelCategory::Vision,
            "sapient see <image> -p \"<question>\"",
        ),
    ];

    for (cat, run_hint) in sections {
        let rows: Vec<Vec<String>> = catalog
            .iter()
            .filter(|m| m.category() == cat)
            .map(row_for)
            .collect();
        if rows.is_empty() {
            continue;
        }
        println!("\n{} ({})\n", cat.label(), rows.len());
        ui::print_table(&["MODEL", "FAMILY", "SIZE", "DOWNLOAD", "STATUS"], &rows);
        ui::hint(format!("run with:  {run_hint}"));
    }
    println!();
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

// ── devices ───────────────────────────────────────────────────────────────────

fn devices_command() -> Result<()> {
    let profile = detect_devices();

    println!();
    println!(
        "  {}",
        console::style("⚡ SAPIENT Device Report").cyan().bold()
    );
    println!();

    // CPU + memory block
    print!("{}", profile.report());

    // Hybrid note
    if profile.unified_memory {
        println!(
            "\n  {}  Apple Unified Memory — zero-copy between CPU and Metal GPU",
            console::style("Hybrid:").green().bold()
        );
        println!("          All layers run on Metal when model fits; otherwise CPU+Metal split.");
    }

    // Windows GPU hint
    #[cfg(target_os = "windows")]
    {
        use sapient_generate::device::ComputeApi;
        let has_cuda = profile
            .gpus
            .iter()
            .any(|g| g.apis.contains(&ComputeApi::Cuda));
        let has_dx12 = profile
            .gpus
            .iter()
            .any(|g| g.apis.contains(&ComputeApi::DirectX12));
        if has_cuda {
            println!(
                "\n  {}  NVIDIA GPU detected with CUDA — DirectML/Vulkan compute backend planned.",
                console::style("Note:").yellow().bold()
            );
        } else if has_dx12 {
            println!(
                "\n  {}  GPU with DirectX 12 detected — Vulkan/DX12 compute backend planned.",
                console::style("Note:").yellow().bold()
            );
        }
    }

    // Recommendations
    println!("{}", profile.recommendations());

    // Current auto-backend decision
    let cached = hub::list_cached_models().unwrap_or_default();
    if !cached.is_empty() {
        println!("  Your downloaded models:");
        for m in &cached {
            // Look up model size from registry to compute recommendation
            let plan = recommend_backend(&profile, 0, 32); // 0 = unknown size
            println!("    {m:<36} → {}", console::style(plan.label()).dim());
        }
        println!();
    }

    // Usage tips
    if profile.gpus.iter().any(|g| {
        use sapient_generate::device::ComputeApi;
        g.apis.contains(&ComputeApi::Metal)
    }) {
        println!("  {}", console::style("Tips:").bold());
        println!("    sapient chat --backend metal <model>   # force Metal GPU");
        println!("    sapient chat --backend cpu   <model>   # force CPU");
        println!("    sapient chat --backend auto  <model>   # auto-select (default)");
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
        .with_context(|| format!("invalid backend '{value}'; expected auto, cpu, metal, or wgpu"))
}

// ── transcribe (speech-to-text) ─────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn transcribe_command(
    model: &str,
    audio: &std::path::Path,
    backend: &str,
    language: Option<String>,
    translate: bool,
    timestamps: bool,
    beam_size: usize,
) -> Result<()> {
    if !audio.exists() {
        anyhow::bail!("audio file not found: {}", audio.display());
    }
    let backend_kind = parse_generation_backend(backend)?;

    let loading = ui::spinner(format!("loading {model}…"));
    let pipeline = std::sync::Arc::new(
        TranscribePipeline::from_pretrained_with_backend(model, backend_kind).await?,
    );
    drop(loading);

    let opts = TranscribeOptions {
        language,
        translate,
        timestamps,
        beam_size,
        ..Default::default()
    };

    // Stream decoded text to stdout as the model produces it.
    use futures::StreamExt;
    use std::io::Write;
    let mut stream = pipeline.transcribe_stream(audio, opts).await?;
    let mut any = false;
    while let Some(delta) = stream.next().await {
        any = true;
        print!("{delta}");
        std::io::stdout().flush().ok();
    }
    if any {
        println!();
    }
    Ok(())
}

// ── speak (text-to-speech) ──────────────────────────────────────────────────────

/// True for the Kokoro-82M TTS aliases (the non-autoregressive, real-time path).
fn is_kokoro_model(model: &str) -> bool {
    let m = model.to_lowercase();
    m == "kokoro" || m == "kokoro-82m" || m.contains("kokoro")
}

async fn see_command(
    image: &std::path::Path,
    prompt: &str,
    model: &str,
    max_tokens: usize,
) -> Result<()> {
    if !image.exists() {
        anyhow::bail!("image not found: {}", image.display());
    }
    let loading = ui::spinner(format!("loading {model}…"));
    let mut vlm = sapient_generate::VlmPipeline::from_pretrained(model).await?;
    drop(loading);

    let thinking = ui::spinner("looking at the image…");
    let image = image.to_path_buf();
    let prompt_owned = prompt.to_string();
    let (answer, stats) = tokio::task::spawn_blocking(move || {
        vlm.answer_with_stats(&image, &prompt_owned, max_tokens)
    })
    .await??;
    drop(thinking);
    println!("{answer}");
    eprintln!(
        "⏱ vision {} ms · prefill {} ms ({} tok) · decode {} tok in {} ms ({:.1} tok/s)",
        stats.vision_ms,
        stats.prefill_ms,
        stats.prompt_tokens,
        stats.gen_tokens,
        stats.decode_ms,
        stats.decode_tps()
    );
    Ok(())
}

async fn speak_command(
    model: &str,
    text: &str,
    output: &std::path::Path,
    voice: &str,
    backend: &str,
    no_play: bool,
) -> Result<()> {
    if is_kokoro_model(model) {
        return speak_kokoro_command(text, output, voice, no_play).await;
    }
    // `speak` is text-to-SPEECH. A speech-to-text (Whisper) model has no TTS
    // weights and would otherwise fail cryptically ("architecture Whisper does
    // not yet have a native forward engine"), so catch the mix-up here.
    if let Some(m) = sapient_hub::registry::lookup(model) {
        use sapient_hub::registry::ModelCategory;
        match m.category() {
            ModelCategory::TextToSpeech => {}
            ModelCategory::SpeechToText => anyhow::bail!(
                "'{model}' is a speech-to-text (Whisper) model, not a text-to-speech \
                 model. To synthesize speech use a TTS model: sapient speak kokoro-82m \
                 \"<text>\" (real-time) or sapient speak orpheus-3b \"<text>\". \
                 To transcribe audio with '{model}', use: sapient transcribe."
            ),
            ModelCategory::Chat => anyhow::bail!(
                "'{model}' is a {} text-generation model, not a text-to-speech model. \
                 Use a TTS model: sapient speak kokoro-82m \"<text>\" or \
                 sapient speak orpheus-3b \"<text>\". To chat with '{model}', use: \
                 sapient chat {model}.",
                m.family
            ),
            ModelCategory::Vision => anyhow::bail!(
                "'{model}' is a vision-language model, not a text-to-speech model. \
                 Use it with: sapient see <image> --model {model} -p \"<question>\"."
            ),
        }
    }
    if !ORPHEUS_VOICES.contains(&voice) {
        anyhow::bail!(
            "unknown voice '{voice}' — choose one of: {}",
            ORPHEUS_VOICES.join(", ")
        );
    }
    let backend_kind = parse_generation_backend(backend)?;

    let loading = ui::spinner(format!("loading {model}…"));
    let pipeline = SpeakPipeline::from_pretrained_with_backend(model, backend_kind).await?;
    drop(loading);

    let synth = ui::spinner(format!("synthesising ({voice})…"));
    let text = text.to_owned();
    let voice = voice.to_owned();
    let (samples, sample_rate) = tokio::task::spawn_blocking(move || {
        let sr = pipeline.sample_rate();
        let samples = pipeline.speak(&text, &voice)?;
        Ok::<_, anyhow::Error>((samples, sr))
    })
    .await
    .context("speak task panicked")??;
    drop(synth);

    write_and_play_speech(&samples, sample_rate, output, no_play).await
}

/// Write `samples` to `out` as a 16-bit WAV and — unless `no_play` — play them
/// through the default output device. Playback needs the `audio-io` feature; a
/// build without it (or a headless box with no output device) just writes the
/// file and prints a note. Mirrors `converse`'s drain-then-tail wait so the
/// last buffer finishes before the command returns.
async fn write_and_play_speech(
    samples: &[f32],
    sample_rate: u32,
    out: &std::path::Path,
    no_play: bool,
) -> Result<()> {
    sapient_generate::write_wav(out, samples, sample_rate)?;
    let secs = samples.len() as f32 / sample_rate.max(1) as f32;
    println!(
        "✓ wrote {} ({:.1}s, {} Hz)",
        out.display(),
        secs,
        sample_rate
    );

    if no_play || samples.is_empty() {
        return Ok(());
    }

    #[cfg(feature = "audio-io")]
    {
        let samples = samples.to_vec();
        tokio::task::spawn_blocking(move || play_samples_blocking(&samples, sample_rate))
            .await
            .context("playback task panicked")??;
    }
    #[cfg(not(feature = "audio-io"))]
    {
        ui::hint(
            "this build has no audio output (compiled without `audio-io`); saved to file only",
        );
    }
    Ok(())
}

/// Blocking playback of mono `samples` through the default output device, waiting
/// for the queue to drain (+ a short tail) before returning. Runs inside
/// `spawn_blocking` since it sleeps. A missing output device is a soft failure
/// (the WAV is already written) — we print a hint instead of erroring.
#[cfg(feature = "audio-io")]
fn play_samples_blocking(samples: &[f32], sample_rate: u32) -> Result<()> {
    use sapient_generate::SpeakerPlayback;
    let player = match SpeakerPlayback::default_output() {
        Ok(p) => p,
        Err(e) => {
            ui::hint(format!("no audio output device ({e}); saved to file only"));
            return Ok(());
        }
    };
    let spin = ui::spinner("playing…".to_string());
    player.submit(samples, sample_rate)?;
    // Wait for the device callback to consume the queue, then a small tail so the
    // last buffer actually finishes on the hardware.
    while player.pending_secs() > 0.05 {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    std::thread::sleep(std::time::Duration::from_millis(250));
    drop(spin);
    Ok(())
}

/// Synthesize with Kokoro-82M (non-autoregressive, real-time). Pulls the
/// converted safetensors mirror (or `SAPIENT_KOKORO_DIR`), runs the pure-Rust
/// StyleTTS2 + ISTFTNet forward, and writes a 24 kHz WAV.
async fn speak_kokoro_command(
    text: &str,
    output: &std::path::Path,
    voice: &str,
    no_play: bool,
) -> Result<()> {
    use sapient_generate::converse::Tts;
    use sapient_generate::{KokoroTts, DEFAULT_KOKORO_VOICE};
    // `--voice` defaults to the Orpheus default ("tara"); map that to Kokoro's.
    let voice = if voice == "tara" {
        DEFAULT_KOKORO_VOICE
    } else {
        voice
    };

    let loading = ui::spinner("loading kokoro-82m…".to_string());
    let tts = KokoroTts::from_default().await?.with_voice(voice);
    drop(loading);

    let synth = ui::spinner(format!("synthesising ({voice})…"));
    let text = text.to_owned();
    let (samples, sr) = tokio::task::spawn_blocking(move || {
        let sr = tts.sample_rate();
        let samples = tts.synthesize(&text)?;
        Ok::<_, anyhow::Error>((samples, sr))
    })
    .await
    .context("kokoro speak task panicked")??;
    drop(synth);

    write_and_play_speech(&samples, sr, output, no_play).await
}

// ── converse (real-time speech-to-speech) ──────────────────────────────────────

// `audio-io` is a default feature, so the official release binaries (macOS,
// Windows, x86_64 Linux) ship `converse`. This stub only compiles for builds
// made with `--no-default-features` (e.g. the aarch64-linux release, which skips
// the ALSA dependency to keep the cross-compile clean).
#[cfg(not(feature = "audio-io"))]
#[allow(clippy::too_many_arguments)]
async fn converse_command(
    _model: &str,
    _stt: &str,
    _backend: &str,
    _language: Option<String>,
    _system: Option<String>,
    _speak: bool,
    _tts: &str,
    _input: Option<PathBuf>,
) -> Result<()> {
    anyhow::bail!(
        "this build was compiled without live audio I/O. Rebuild with the \
         `audio-io` feature (it is on by default): `cargo build --release -p \
         sapient-cli` — on Linux install the ALSA dev headers first (`sudo \
         apt-get install libasound2-dev`)."
    )
}

#[cfg(feature = "audio-io")]
#[allow(clippy::too_many_arguments)]
async fn converse_command(
    model: &str,
    stt: &str,
    backend: &str,
    language: Option<String>,
    system: Option<String>,
    speak: bool,
    tts: &str,
    input: Option<PathBuf>,
) -> Result<()> {
    use std::io::{IsTerminal, Write};

    use sapient_generate::{
        microphone_guidance, open_privacy_settings, request_microphone, ConversePipeline,
        EnergyVad, KokoroTts, MicCapture, MicPermission, NoopTts, Pipeline, SpeakPipeline,
        SpeakerPlayback, TranscribePipeline, Tts, VadConfig,
    };

    let backend_kind = parse_generation_backend(backend)?;

    // `--stt` is the speech-to-TEXT (Whisper) model. A text-to-SPEECH model like
    // orpheus-3b has no Whisper weights and would fail cryptically ("no weights
    // found"), so catch the mix-up here with a clear message.
    if let Some(m) = sapient_hub::registry::lookup(stt) {
        if m.family != "Whisper" {
            anyhow::bail!(
                "--stt expects a speech-to-text (Whisper) model, but '{stt}' is a {} \
                 (text-to-speech) model. Use --stt whisper-base | whisper-tiny | \
                 whisper-small. To choose the spoken voice, use: --speak --tts {}.",
                m.family,
                m.family.to_lowercase()
            );
        }
    }

    // Spoken-voice engine for `--speak`: kokoro (real-time) or orpheus (slow).
    let tts_engine = match tts.trim().to_lowercase().as_str() {
        "kokoro" | "kokoro-82m" => "kokoro",
        "orpheus" | "orpheus-3b" | "openhorizon/orpheus-3b" => "orpheus",
        other => anyhow::bail!("unknown --tts '{other}'; choose: kokoro | orpheus"),
    };

    // Ask the OS for microphone access up front (macOS shows the consent prompt;
    // Windows/Linux have no per-app prompt → Unknown, handled by the level meter).
    // Skipped for --input (WAV benchmark): no mic is touched in that path.
    if input.is_none() {
        match tokio::task::block_in_place(request_microphone) {
            MicPermission::Denied => {
                eprintln!(
                    "⚠️  Microphone access is denied.\n   {}",
                    microphone_guidance()
                );
                open_privacy_settings();
                return Ok(());
            }
            MicPermission::Granted | MicPermission::Unknown => {}
        }
    }

    let loading = ui::spinner(if speak {
        format!("loading {stt} + {model} + {tts_engine}…")
    } else {
        format!("loading {stt} + {model}…")
    });
    // Load all models concurrently at startup (their downloads overlap) and keep
    // them resident for the whole session — no mid-conversation model load. The
    // spoken-voice engine is chosen by --tts: Kokoro-82M (non-autoregressive →
    // real-time on CPU) or Orpheus-3B (autoregressive → richer but ~0.18× real-time).
    let llm_opts = LoadOptions {
        backend: backend_kind,
        ..Default::default()
    };
    let (stt_res, llm_res, tts_res) = tokio::join!(
        TranscribePipeline::from_pretrained_with_backend(stt, backend_kind),
        Pipeline::from_pretrained_with_opts(model, llm_opts),
        async {
            if !speak {
                return Ok::<Option<std::sync::Arc<dyn Tts>>, anyhow::Error>(None);
            }
            if tts_engine == "orpheus" {
                let sp =
                    SpeakPipeline::from_pretrained_with_backend("orpheus-3b", backend_kind).await?;
                Ok(Some(std::sync::Arc::new(sp) as std::sync::Arc<dyn Tts>))
            } else {
                let k = KokoroTts::from_default().await?;
                Ok(Some(std::sync::Arc::new(k) as std::sync::Arc<dyn Tts>))
            }
        },
    );
    let stt_pipe = stt_res?;
    let llm = llm_res?;
    let tts: std::sync::Arc<dyn Tts> = tts_res?.unwrap_or_else(|| std::sync::Arc::new(NoopTts));
    drop(loading);

    let mut converse = ConversePipeline::new(stt_pipe, llm, tts);
    match system {
        Some(s) => converse = converse.with_system(s),
        // Keep replies short and natural for a conversational voice cadence.
        None if speak => {
            converse = converse.with_system(
                "You are a voice assistant. Reply in one or two short, natural sentences.",
            );
        }
        None => {}
    }
    if let Some(l) = language {
        converse = converse.with_language(l);
    }

    // ── --input: one-shot WAV turn (no mic) — benchmark the full pipeline ────
    if let Some(wav) = input {
        let utt = sapient_audio::io::load_audio(&wav, 16_000)
            .with_context(|| format!("loading input audio {wav:?}"))?;
        let utt_secs = utt.len() as f32 / 16_000.0;
        println!("input: {} ({utt_secs:.2}s @ 16 kHz)", wav.display());

        let t_stt = std::time::Instant::now();
        let transcript = converse.transcribe_utterance(&utt)?;
        let stt_ms = t_stt.elapsed().as_millis();
        if transcript.is_empty() {
            anyhow::bail!("transcript was empty — is the input speech?");
        }
        ui::converse_you(&transcript);

        ui::converse_assistant_prefix();
        let mut audio_len = 0usize;
        let turn = converse
            .respond_streaming(
                &transcript,
                |tok| {
                    print!("{tok}");
                    let _ = std::io::stdout().flush();
                },
                |samples, _rate| audio_len += samples.len(),
            )
            .await?;
        println!();

        let reply_secs = audio_len as f32 / turn.audio_sample_rate as f32;
        let tps = if turn.gen_ms > 0 {
            turn.gen_tokens as f32 / (turn.gen_ms as f32 / 1000.0)
        } else {
            0.0
        };
        println!("\n── converse benchmark (one turn, --input) ──");
        println!(
            "  STT   {:>6} ms   ({:.1}× realtime, {utt_secs:.2}s audio)",
            stt_ms,
            utt_secs / (stt_ms as f32 / 1000.0).max(1e-3)
        );
        println!(
            "  LLM   {:>6} ms   ({} tok, {tps:.1} tok/s)",
            turn.gen_ms, turn.gen_tokens
        );
        if speak {
            let rtf = (turn.tts_ms as f32 / 1000.0) / reply_secs.max(1e-3);
            println!(
                "  TTS   {:>6} ms   ({tts_engine}: {reply_secs:.2}s audio, RTF {rtf:.2})",
                turn.tts_ms
            );
        }
        println!(
            "  TTFT  {:>6} ms   (reply start → first token)",
            turn.ttft_ms
        );
        if let Some(fa) = turn.first_audio_ms {
            println!("  audio {:>6} ms   (reply start → first audio chunk)", fa);
        }
        let total = stt_ms + turn.gen_ms + turn.tts_ms;
        println!("  total {:>6} ms", total);
        return Ok(());
    }

    // Mic stays on this thread (cpal Stream is !Send); the future runs on the
    // main runtime (not spawned), so holding it across awaits is fine.
    let cap = MicCapture::default_input()?;
    let src_rate = cap.sample_rate();
    let frames = cap.frames();
    // Speaker is opened only for --speak (and only if an output device exists).
    let player = if speak {
        match SpeakerPlayback::default_output() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("warning: no speaker output ({e}); replies will be text-only");
                None
            }
        }
    } else {
        None
    };

    let frame_samples = (src_rate / 50).max(1) as usize; // 20 ms VAD frames at device rate
    let new_vad = || {
        EnergyVad::new(VadConfig {
            frame_samples,
            // 400 ms hangover (default 600) — the streaming loop wants a snappy
            // end-of-turn; incremental STT below hides the transcription cost.
            silence_hang_frames: 20,
            // Real-mic hardening: ≥240 ms of speech, and the utterance's MEAN
            // energy must look like speech — quiet echo tails otherwise reach
            // Whisper, which hallucinates on non-speech.
            min_utterance_frames: 12,
            min_mean_rms: 0.02,
            ..VadConfig::default()
        })
    };
    let mut vad = new_vad();
    let mut buf: Vec<f32> = Vec::new();

    // Incremental STT: while the user is still talking, a background worker
    // re-transcribes the utterance-so-far (snapshots every ~0.5 s). At
    // end-of-utterance only the silence hangover is new audio, so the last
    // incremental transcript is used directly — STT leaves the critical path.
    let live = converse.live_stt();
    let mut frames_since_feed = 0u32;

    let backend_label = converse.backend_label();
    ui::converse_banner(src_rate, stt, model, &backend_label, speak, tts_engine);

    // Prewarm STT: the first Whisper pass pays one-time setup (mel plan, cache
    // alloc, page-in) — ~2 s of extra latency that used to land on the user's
    // FIRST utterance. Burn it on silence now instead.
    {
        let warm = ui::spinner("warming up…");
        let _ = converse.transcribe_utterance(&vec![0.0f32; 8_000]);
        drop(warm);
    }

    // Live mic meter + a one-time hint if the mic looks dead (the terminal lacks
    // microphone permission, so cpal delivers only silence).
    let tty = std::io::stdout().is_terminal();
    let mut peak = 0.0f32;
    let mut meter_tick = 0u32;
    let mut total_frames = 0u64;
    let mut warned_silent = false;
    let mut meter_shown = false;
    let clear_meter = |shown: &mut bool| {
        if *shown {
            print!("\r\x1b[2K");
            let _ = std::io::stdout().flush();
            *shown = false;
        }
    };

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => { clear_meter(&mut meter_shown); ui::converse_bye(); break; }
            chunk = frames.recv_async() => {
                let Ok(chunk) = chunk else { break };
                buf.extend_from_slice(&chunk);
                while buf.len() >= frame_samples {
                    let frame: Vec<f32> = buf.drain(..frame_samples).collect();
                    // Live input level (independent of the VAD) so the user can see the mic is hot.
                    let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
                    peak = peak.max(rms);
                    total_frames += 1;
                    meter_tick += 1;
                    if tty && meter_tick % 5 == 0 {
                        print!("{}", ui::mic_meter_line(rms));
                        let _ = std::io::stdout().flush();
                        meter_shown = true;
                    }
                    // ~5 s in with no real signal → the mic is almost certainly muted/denied.
                    if !warned_silent
                        && total_frames > (src_rate as u64 / frame_samples as u64) * 5
                        && peak < 1e-3
                    {
                        warned_silent = true;
                        clear_meter(&mut meter_shown);
                        ui::converse_warn(microphone_guidance());
                    }

                    let finalized = vad.push(&frame);
                    if finalized.is_none() && vad.in_speech() {
                        frames_since_feed += 1;
                        if frames_since_feed >= 25 {
                            // ~every 0.5 s of speech: hand the utterance-so-far
                            // to the background transcriber (16 kHz mono).
                            frames_since_feed = 0;
                            let snap =
                                sapient_audio::io::resample(vad.speech_so_far(), src_rate, 16_000)?;
                            live.feed(snap);
                        }
                    }
                    if let Some(utt) = finalized {
                        clear_meter(&mut meter_shown);
                        frames_since_feed = 0;
                        // Resample the finalized utterance (device rate → 16 kHz) for STT.
                        let utt16 = sapient_audio::io::resample(&utt, src_rate, 16_000)?;
                        let audio_secs = utt16.len() as f32 / 16_000.0;

                        ui::converse_status("transcribing…");
                        let t_stt = std::time::Instant::now();
                        // Prefer the incremental transcript when it covers the
                        // utterance minus the trailing hangover (~0.5 s tolerance);
                        // otherwise fall back to one full pass.
                        let (inc_text, covered) = live.settle(std::time::Duration::from_millis(900));
                        let stt_hidden = !inc_text.is_empty()
                            && utt16.len().saturating_sub(covered) <= 8_000;
                        let transcript = if stt_hidden {
                            inc_text
                        } else {
                            converse.transcribe_utterance(&utt16)?
                        };
                        live.reset();
                        let stt_elapsed = t_stt.elapsed();

                        if transcript.is_empty() {
                            ui::converse_note("didn't catch that — try again");
                        } else {
                            ui::converse_you(&transcript);
                            ui::converse_stt_stats(audio_secs, stt_elapsed);
                            ui::converse_assistant_prefix();
                            // Stream the reply: text token-by-token, and audio
                            // **sentence-by-sentence as it's generated** (no-op
                            // without --speak). `respond_streaming` synthesizes
                            // each sentence the moment it completes and emits it
                            // here, so the speaker starts ~after the first sentence
                            // instead of after the whole reply — time-to-first-audio
                            // no longer scales with reply length.
                            let turn = converse
                                .respond_streaming(
                                    &transcript,
                                    |tok| {
                                        print!("{tok}");
                                        let _ = std::io::stdout().flush();
                                    },
                                    |samples, rate| {
                                        if let Some(p) = &player {
                                            let _ = p.submit(samples, rate);
                                        }
                                    },
                                )
                                .await?;
                            println!();
                            let tts_info = if speak && !turn.audio.is_empty() {
                                Some((
                                    std::time::Duration::from_millis(turn.tts_ms as u64),
                                    turn.audio.len() as f32 / turn.audio_sample_rate as f32,
                                ))
                            } else {
                                None
                            };
                            ui::converse_gen_stats(
                                turn.gen_tokens,
                                std::time::Duration::from_millis(turn.gen_ms as u64),
                                tts_info,
                            );
                            // Per-stage latency (Phase 10.5): what the user waited.
                            {
                                let stt_part = if stt_hidden {
                                    "stt hidden (streamed)".to_string()
                                } else {
                                    format!("stt {} ms", stt_elapsed.as_millis())
                                };
                                let audio_part = match turn.first_audio_ms {
                                    Some(ms) => format!(" · first audio {ms} ms"),
                                    None => String::new(),
                                };
                                ui::converse_note(&format!(
                                    "⏱ {stt_part} · first token {} ms{audio_part}",
                                    turn.ttft_ms
                                ));
                            }
                            // Drain queued speaker audio — but keep listening
                            // for barge-in. There is no real AEC, so the gate is
                            // ECHO-REFERENCED: the speaker tracks its own played
                            // envelope (`expected_bleed`), and the mic must beat
                            // α·(what's playing RIGHT NOW) — a reply that gets
                            // louder mid-sentence raises the bar with itself.
                            // (A start-of-turn calibration still lost to loud
                            // later sentences — live-mic field report #2.)
                            if let Some(p) = &player {
                                let mut consec_loud = 0u32;
                                let mut frames_seen = 0u32;
                                let mut alpha = 0.0f32; // mic ← speaker coupling
                                'drain: while p.pending_secs() > 0.05 {
                                    while let Ok(chunk) = frames.try_recv() {
                                        buf.extend_from_slice(&chunk);
                                    }
                                    while buf.len() >= frame_samples {
                                        let f: Vec<f32> = buf.drain(..frame_samples).collect();
                                        let rms = (f.iter().map(|s| s * s).sum::<f32>()
                                            / f.len() as f32)
                                            .sqrt();
                                        let exp = p.expected_bleed();
                                        frames_seen += 1;
                                        let threshold =
                                            (alpha * exp * 2.2 + 0.03).max(0.05);
                                        let loud = rms > threshold;
                                        // Learn the coupling from frames that are
                                        // NOT candidate speech (they're pure
                                        // bleed); never learn from loud frames —
                                        // those may be the human.
                                        if !loud && exp > 0.02 {
                                            alpha = alpha.max(rms / exp);
                                        }
                                        if frames_seen <= 20 {
                                            continue; // settle coupling ~400 ms
                                        }
                                        if loud {
                                            consec_loud += 1;
                                        } else {
                                            consec_loud = 0;
                                        }
                                        if consec_loud >= 15 {
                                            // ~300 ms sustained above the live
                                            // bleed estimate → human talking.
                                            p.clear();
                                            ui::converse_note("(interrupted — listening)");
                                            break 'drain;
                                        }
                                    }
                                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                                }
                                // Echo-tail grace: let the device buffer + room
                                // reverb decay before we listen again, so the
                                // tail can't become a phantom "utterance".
                                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                                p.reset_reference();
                            }
                        }

                        // Discard anything captured during STT/LLM/TTS (including our
                        // own speaker output) so it isn't treated as the next
                        // utterance, and reset the segmenter for a fresh turn.
                        while frames.try_recv().is_ok() {}
                        buf.clear();
                        vad = new_vad();
                        peak = 0.0;
                        total_frames = 0;
                    }
                }
            }
        }
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
