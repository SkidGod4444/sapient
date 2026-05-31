//! OpenAI-compatible HTTP inference server.
//!
//! Models are loaded on demand: the first request for a model triggers download
//! and load; subsequent requests reuse the in-memory pipeline. The N most-recently
//! -used models stay resident (LRU, bounded by `--max-models` and a RAM budget),
//! so switching back to a recent model is instant — unlike Ollama, which keeps one
//! model and cold-reloads on every switch. Pick a model via the `"model"` field.
//!
//! Routes: GET /v1/models, POST /v1/chat/completions, POST /v1/completions,
//! GET /v1/health.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tracing::info;

use sapient_generate::{GenerationConfig, LoadOptions, Pipeline, SamplingStrategy};
use sapient_tokenizers::ChatMessage;

// ── Multi-model LRU cache ─────────────────────────────────────────────────────
//
// Unlike Ollama (one resident model, cold reload on every switch), we keep the N
// most-recently-used models resident, bounded by a memory budget. Switching back
// to a recent model is then instant — no download, no re-quantization, no reload.
// Each cached model is an `Arc<CachedModel>`, so a streaming request keeps using
// its model after the cache lock is released (and even if the model is evicted
// mid-request — the Arc keeps it alive until the stream finishes).

/// A model held resident in the cache.
struct CachedModel {
    model_id: String,
    pipeline: Pipeline,
    /// Resident-size estimate (bytes) used for budget-based eviction.
    bytes: u64,
}

/// Bounded LRU cache of loaded models. Most-recently-used is at the back.
struct ModelCache {
    entries: VecDeque<Arc<CachedModel>>,
    max_models: usize,
    budget_bytes: u64,
    used_bytes: u64,
}

impl ModelCache {
    fn new(max_models: usize, budget_bytes: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            max_models: max_models.max(1),
            budget_bytes,
            used_bytes: 0,
        }
    }

    /// Promote an already-resident model to MRU and return it.
    fn touch(&mut self, id: &str) -> Option<Arc<CachedModel>> {
        let pos = self.entries.iter().position(|m| m.model_id == id)?;
        let m = self.entries.remove(pos).expect("position just found");
        self.entries.push_back(m.clone());
        Some(m)
    }

    /// Insert a freshly-loaded model at MRU, evicting LRU entries until both the
    /// count and byte budgets are satisfied. Returns the evicted model IDs.
    /// The just-inserted model is never evicted (it stays at the back).
    fn insert(&mut self, entry: Arc<CachedModel>) -> Vec<String> {
        if self.touch(&entry.model_id).is_some() {
            return Vec::new(); // already present (concurrent load race)
        }
        self.used_bytes += entry.bytes;
        self.entries.push_back(entry);

        let mut evicted = Vec::new();
        while self.entries.len() > 1
            && (self.entries.len() > self.max_models || self.used_bytes > self.budget_bytes)
        {
            if let Some(old) = self.entries.pop_front() {
                self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
                evicted.push(old.model_id.clone());
                // `old` (Arc) drops here unless an in-flight request still holds it.
            }
        }
        evicted
    }

    fn mru_id(&self) -> Option<String> {
        self.entries.back().map(|m| m.model_id.clone())
    }

    fn ids(&self) -> Vec<String> {
        self.entries.iter().map(|m| m.model_id.clone()).collect()
    }
}

#[derive(Clone)]
struct ServeState {
    cache: Arc<Mutex<ModelCache>>,
    /// Serializes model loads so two concurrent first-requests for the same model
    /// don't both download/load it, and loads don't thrash each other.
    load_lock: Arc<Mutex<()>>,
    /// Admission control: bounds the number of inferences running at once so a
    /// burst of requests queues fairly instead of oversubscribing the CPU/GPU and
    /// exploding thread/memory use. Excess requests await a permit.
    inference_sem: Arc<tokio::sync::Semaphore>,
    backend: String,
    force_mmap: bool,
}

impl ServeState {
    /// Return the resident model for `model_id`, loading (and LRU-evicting) it if
    /// needed. The cache lock is NOT held during the (slow) load or during
    /// inference, so cache hits and other models' requests aren't blocked.
    async fn get_or_load(&self, model_id: &str) -> Result<Arc<CachedModel>> {
        // Fast path: already resident.
        if let Some(m) = self.cache.lock().await.touch(model_id) {
            return Ok(m);
        }

        // Slow path: serialize loads, then re-check (a concurrent request may have
        // loaded it while we waited on the load lock).
        let _load = self.load_lock.lock().await;
        if let Some(m) = self.cache.lock().await.touch(model_id) {
            return Ok(m);
        }

        info!("loading model '{model_id}'…");
        let backend_kind = crate::parse_generation_backend(&self.backend)?;
        let opts = LoadOptions {
            backend: backend_kind,
            force_mmap: self.force_mmap,
            ..LoadOptions::default()
        };
        let mut pipeline = Pipeline::from_pretrained_with_opts(model_id, opts)
            .await
            .with_context(|| format!("failed to load model '{model_id}'"))?;
        // Reuse the KV cache across requests sharing a prompt prefix (multi-turn
        // chat, shared system prompts) — skips re-prefilling the whole history.
        pipeline.enable_prefix_cache();

        let bytes = estimate_model_bytes(model_id);
        let entry = Arc::new(CachedModel {
            model_id: model_id.to_string(),
            pipeline,
            bytes,
        });

        let evicted = self.cache.lock().await.insert(entry.clone());
        for id in &evicted {
            info!("evicted '{id}' from model cache (LRU)");
        }
        info!(
            "model '{model_id}' ready ({:.1} GB; resident models: {})",
            bytes as f64 / 1e9,
            self.cache.lock().await.entries.len(),
        );
        Ok(entry)
    }
}

/// Estimate a model's resident memory (bytes) for budgeting. Uses the on-disk
/// (download) size as a proxy for the mmap'd weights; falls back to a default
/// when the size is unknown (e.g. a not-yet-listed local path).
fn estimate_model_bytes(model_id: &str) -> u64 {
    let disk = crate::hub::cached_model_size(model_id);
    if disk > 0 {
        disk
    } else {
        2 * 1024 * 1024 * 1024 // 2 GB fallback
    }
}

/// Total physical RAM in bytes (for the default cache budget). Best-effort.
fn total_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(v) = s.trim().parse::<u64>() {
                    return v;
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb) = rest.trim().split_whitespace().next() {
                        if let Ok(v) = kb.parse::<u64>() {
                            return v * 1024;
                        }
                    }
                }
            }
        }
    }
    0
}

// ── OpenAI request/response types ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<OAIMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stop: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stop: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OAIMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: OAIMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<DeltaChoice>,
}

#[derive(Serialize)]
struct DeltaChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize, Default)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: &'static str,
}

#[derive(Serialize, Default)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gen_id() -> String {
    format!(
        "chatcmpl-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    )
}

fn parse_stop(stop: Option<serde_json::Value>) -> Vec<String> {
    match stop {
        Some(serde_json::Value::String(s)) => vec![s],
        Some(serde_json::Value::Array(arr)) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        _ => vec![],
    }
}

fn build_config(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stop: Vec<String>,
) -> GenerationConfig {
    let mut cfg = GenerationConfig::default();
    if let Some(n) = max_tokens {
        cfg.max_new_tokens = n;
    }
    if let Some(t) = temperature {
        if t > 0.0 {
            cfg.strategy = SamplingStrategy::TopP {
                temperature: t,
                p: 0.95,
            };
        }
    }
    cfg.stop_sequences = stop;
    cfg
}

fn model_err(msg: impl std::fmt::Display) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {
            "message": msg.to_string(),
            "type": "invalid_request_error",
            "code": "model_not_found"
        }})),
    )
        .into_response()
}

fn server_err(msg: impl std::fmt::Display) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": msg.to_string(), "type": "server_error"}})),
    )
        .into_response()
}

/// Resolve which model to run:
/// 1. Use the `"model"` field from the request if non-empty.
/// 2. Fall back to whichever model is currently loaded in memory.
/// 3. Return a 400 if neither is available.
async fn resolve_model(
    requested: Option<&str>,
    cache: &Mutex<ModelCache>,
) -> std::result::Result<String, Response> {
    if let Some(m) = requested.filter(|s| !s.is_empty()) {
        return Ok(m.to_string());
    }
    // Fall back to the most-recently-used resident model.
    match cache.lock().await.mru_id() {
        Some(id) => Ok(id),
        None => Err(model_err(
            "No model specified and no model is currently loaded. \
             Pass 'model' in the request, or start the server with: sapient serve <model>",
        )),
    }
}

// ── Route handlers ────────────────────────────────────────────────────────────

async fn handle_models(State(state): State<ServeState>) -> impl IntoResponse {
    // Return all locally cached registry models.
    let cached = crate::hub::list_cached_models().unwrap_or_default();
    let now = now_secs();
    let data: Vec<_> = cached
        .iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "owned_by": "sapient",
                "created": now,
            })
        })
        .collect();

    let (resident, active) = {
        let guard = state.cache.lock().await;
        (guard.ids(), guard.mru_id())
    };

    Json(json!({
        "object": "list",
        "data": data,
        "active_model": active,
        "resident_models": resident,
    }))
}

async fn handle_health(State(state): State<ServeState>) -> impl IntoResponse {
    let (resident, active) = {
        let guard = state.cache.lock().await;
        (guard.ids(), guard.mru_id())
    };
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "loaded_model": active,
        "resident_models": resident,
    }))
}

async fn handle_chat_completions(
    State(state): State<ServeState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model_id = match resolve_model(req.model.as_deref(), &state.cache).await {
        Ok(m) => m,
        Err(r) => return r,
    };

    let model = match state.get_or_load(&model_id).await {
        Ok(m) => m,
        Err(e) => return server_err(e),
    };

    // Admission control: wait for an inference slot (bounds concurrency).
    let permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return server_err("server is shutting down"),
    };

    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| match m.role.as_str() {
            "assistant" => ChatMessage::assistant(m.content.clone()),
            _ => ChatMessage::user(m.content.clone()),
        })
        .collect();
    let cfg = build_config(req.max_tokens, req.temperature, parse_stop(req.stop));

    if req.stream {
        let id = gen_id();
        let created = now_secs();
        let model_clone = model_id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<sse::Event, Infallible>>(128);

        let role_json = serde_json::to_string(&ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![DeltaChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        })
        .unwrap();
        let _ = tx.send(Ok(sse::Event::default().data(role_json))).await;

        let tx2 = tx.clone();
        tokio::task::spawn(async move {
            // `model` (Arc) is moved in — no cache lock held during streaming, so
            // other models' requests run concurrently and the cache stays free.
            // `permit` is held for the stream's lifetime, released on completion.
            let _permit = permit;
            let mut stream = model
                .pipeline
                .chat_stream_with_config(&messages, &cfg)
                .await;
            while let Some(token) = stream.next().await {
                let chunk_json = serde_json::to_string(&ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_clone.clone(),
                    choices: vec![DeltaChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(token),
                        },
                        finish_reason: None,
                    }],
                })
                .unwrap();
                if tx2
                    .send(Ok(sse::Event::default().data(chunk_json)))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            let stop_json = serde_json::to_string(&ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_clone,
                choices: vec![DeltaChoice {
                    index: 0,
                    delta: Delta::default(),
                    finish_reason: Some("stop"),
                }],
            })
            .unwrap();
            let _ = tx2.send(Ok(sse::Event::default().data(stop_json))).await;
            let _ = tx2.send(Ok(sse::Event::default().data("[DONE]"))).await;
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        match model.pipeline.chat_with_config(&messages, &cfg).await {
            Ok(reply) => Json(ChatCompletionResponse {
                id: gen_id(),
                object: "chat.completion",
                created: now_secs(),
                model: model_id,
                choices: vec![ChatChoice {
                    index: 0,
                    message: OAIMessage {
                        role: "assistant".into(),
                        content: reply,
                    },
                    finish_reason: "stop",
                }],
                usage: Usage::default(),
            })
            .into_response(),
            Err(e) => server_err(e),
        }
    }
}

async fn handle_completions(
    State(state): State<ServeState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let model_id = match resolve_model(req.model.as_deref(), &state.cache).await {
        Ok(m) => m,
        Err(r) => return r,
    };

    let model = match state.get_or_load(&model_id).await {
        Ok(m) => m,
        Err(e) => return server_err(e),
    };

    let permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return server_err("server is shutting down"),
    };

    let cfg = build_config(req.max_tokens, req.temperature, parse_stop(req.stop));
    let prompt = req.prompt;

    if req.stream {
        let id = gen_id();
        let created = now_secs();
        let model_clone = model_id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<sse::Event, Infallible>>(128);
        let tx2 = tx.clone();

        tokio::task::spawn(async move {
            let _permit = permit;
            let mut stream = model
                .pipeline
                .generate_stream_with_config(&prompt, &cfg)
                .await;
            while let Some(token) = stream.next().await {
                let data = serde_json::to_string(&json!({
                    "id": id, "object": "text_completion", "created": created,
                    "model": model_clone,
                    "choices": [{"text": token, "index": 0, "finish_reason": null}]
                }))
                .unwrap();
                if tx2
                    .send(Ok(sse::Event::default().data(data)))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            let _ = tx2.send(Ok(sse::Event::default().data("[DONE]"))).await;
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        match model.pipeline.generate_with_config(&prompt, &cfg).await {
            Ok(text) => Json(CompletionResponse {
                id: gen_id(),
                object: "text_completion",
                created: now_secs(),
                model: model_id,
                choices: vec![CompletionChoice {
                    index: 0,
                    text,
                    finish_reason: "stop",
                }],
                usage: Usage::default(),
            })
            .into_response(),
            Err(e) => server_err(e),
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn serve_llm(
    preload_model: Option<&str>,
    port: u16,
    backend: &str,
    mmap: bool,
    max_models: usize,
    cache_gb: f64,
    max_concurrency: usize,
) -> Result<()> {
    // Concurrency limit: explicit flag, else number of CPUs (capped) — inference
    // is compute-bound, so oversubscribing hurts. At least 1.
    let concurrency = if max_concurrency > 0 {
        max_concurrency
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4)
    }
    .max(1);
    // Default byte budget: ~70% of system RAM (so the resident set can't OOM the
    // box), or the explicit --cache-gb if given. 0 / unknown RAM → effectively
    // unlimited bytes, falling back to the --max-models count cap alone.
    let budget_bytes = if cache_gb > 0.0 {
        (cache_gb * 1e9) as u64
    } else {
        let ram = total_ram_bytes();
        if ram > 0 {
            (ram as f64 * 0.70) as u64
        } else {
            u64::MAX
        }
    };
    info!(
        "model cache: up to {max_models} models, ~{:.1} GB budget",
        budget_bytes as f64 / 1e9
    );

    info!("inference concurrency limit: {concurrency}");
    let state = ServeState {
        cache: Arc::new(Mutex::new(ModelCache::new(max_models, budget_bytes))),
        load_lock: Arc::new(Mutex::new(())),
        inference_sem: Arc::new(tokio::sync::Semaphore::new(concurrency)),
        backend: backend.to_string(),
        force_mmap: mmap,
    };

    if let Some(model_id) = preload_model {
        let spinner = crate::ui::spinner(format!("loading {model_id}…"));
        let entry = state.get_or_load(model_id).await?;
        spinner.finish_and_clear();
        let arch = format!("{:?}", entry.pipeline.arch());
        let mmap_label = if entry.pipeline.is_mmap() {
            " · mmap"
        } else {
            ""
        };
        print_banner(port, backend, Some((model_id, &arch, mmap_label)));
    } else {
        print_banner(port, backend, None);
    }

    let app = Router::new()
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/completions", post(handle_completions))
        .route("/v1/health", get(handle_health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("SAPIENT serve listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn print_banner(port: u16, backend: &str, loaded: Option<(&str, &str, &str)>) {
    println!();
    println!(
        "  {} {}",
        console::style("⚡ SAPIENT serve").cyan().bold(),
        console::style(format!("· {backend}")).dim()
    );
    if let Some((model_id, arch, mmap_label)) = loaded {
        println!(
            "  {} {}{}",
            console::style("model  ").dim(),
            console::style(model_id).bold(),
            console::style(format!("  ({arch}{mmap_label})")).dim()
        );
    } else {
        println!(
            "  {}",
            console::style("no model pre-loaded — models load on first API request").dim()
        );
    }
    println!(
        "  {} http://0.0.0.0:{}",
        console::style("address").dim(),
        port
    );
    println!();
    println!("  {}", console::style("Endpoints:").dim());
    println!("  {}  GET  /v1/models", console::style("·").dim());
    println!(
        "  {}  POST /v1/chat/completions  (stream=true|false)",
        console::style("·").dim()
    );
    println!(
        "  {}  POST /v1/completions       (stream=true|false)",
        console::style("·").dim()
    );
    println!();
    println!("  {}", console::style("Example:").dim());
    println!(
        "  {}",
        console::style(format!(
            "curl http://localhost:{port}/v1/chat/completions -H 'Content-Type: application/json' \\"
        ))
        .dim()
    );
    println!(
        "  {}",
        console::style(
            "    -d '{\"model\":\"openhorizon/qwen2.5-0.5b-q4\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
        )
        .dim()
    );
    println!();
}
