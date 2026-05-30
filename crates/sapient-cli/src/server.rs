//! OpenAI-compatible HTTP inference server.
//!
//! Models are loaded on demand: the first request for a model triggers download
//! and load; subsequent requests reuse the in-memory pipeline. Only one model is
//! resident at a time. Swap by naming a different model in the `"model"` field.
//!
//! Routes: GET /v1/models, POST /v1/chat/completions, POST /v1/completions,
//! GET /v1/health.

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

// ── Model cache ───────────────────────────────────────────────────────────────

struct LoadedModel {
    model_id: String,
    pipeline: Pipeline,
}

#[derive(Clone)]
struct ServeState {
    cache: Arc<Mutex<Option<LoadedModel>>>,
    backend: String,
    force_mmap: bool,
}

impl ServeState {
    /// Ensure `model_id` is loaded. Swaps out the previous model if different.
    async fn ensure_loaded(&self, model_id: &str) -> Result<()> {
        let mut guard = self.cache.lock().await;
        if guard
            .as_ref()
            .map(|m| m.model_id == model_id)
            .unwrap_or(false)
        {
            return Ok(());
        }

        info!("Loading model '{model_id}'…");
        let backend_kind = crate::parse_generation_backend(&self.backend)?;
        let opts = LoadOptions {
            backend: backend_kind,
            force_mmap: self.force_mmap,
            ..LoadOptions::default()
        };

        let pipeline = Pipeline::from_pretrained_with_opts(model_id, opts)
            .await
            .with_context(|| format!("failed to load model '{model_id}'"))?;

        *guard = Some(LoadedModel {
            model_id: model_id.to_string(),
            pipeline,
        });
        info!("Model '{model_id}' ready");
        Ok(())
    }
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
    cache: &Mutex<Option<LoadedModel>>,
) -> std::result::Result<String, Response> {
    if let Some(m) = requested.filter(|s| !s.is_empty()) {
        return Ok(m.to_string());
    }
    let guard = cache.lock().await;
    match guard.as_ref() {
        Some(loaded) => Ok(loaded.model_id.clone()),
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

    let loaded_id = {
        let guard = state.cache.lock().await;
        guard.as_ref().map(|m| m.model_id.clone())
    };

    Json(json!({
        "object": "list",
        "data": data,
        "active_model": loaded_id,
    }))
}

async fn handle_health(State(state): State<ServeState>) -> impl IntoResponse {
    let loaded = {
        let guard = state.cache.lock().await;
        guard.as_ref().map(|m| m.model_id.clone())
    };
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "loaded_model": loaded,
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

    if let Err(e) = state.ensure_loaded(&model_id).await {
        return server_err(e);
    }

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
            let guard = state.cache.lock().await;
            let Some(loaded) = guard.as_ref() else { return };
            let mut stream = loaded
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
        let guard = state.cache.lock().await;
        let Some(loaded) = guard.as_ref() else {
            return server_err("model cache unexpectedly empty");
        };
        match loaded.pipeline.chat_with_config(&messages, &cfg).await {
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

    if let Err(e) = state.ensure_loaded(&model_id).await {
        return server_err(e);
    }

    let cfg = build_config(req.max_tokens, req.temperature, parse_stop(req.stop));
    let prompt = req.prompt;

    if req.stream {
        let id = gen_id();
        let created = now_secs();
        let model_clone = model_id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<sse::Event, Infallible>>(128);
        let tx2 = tx.clone();

        tokio::task::spawn(async move {
            let guard = state.cache.lock().await;
            let Some(loaded) = guard.as_ref() else { return };
            let mut stream = loaded
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
        let guard = state.cache.lock().await;
        let Some(loaded) = guard.as_ref() else {
            return server_err("model cache unexpectedly empty");
        };
        match loaded.pipeline.generate_with_config(&prompt, &cfg).await {
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
) -> Result<()> {
    let state = ServeState {
        cache: Arc::new(Mutex::new(None)),
        backend: backend.to_string(),
        force_mmap: mmap,
    };

    if let Some(model_id) = preload_model {
        let spinner = crate::ui::spinner(format!("loading {model_id}…"));
        state.ensure_loaded(model_id).await?;
        spinner.finish_and_clear();

        let guard = state.cache.lock().await;
        if let Some(loaded) = guard.as_ref() {
            let arch = format!("{:?}", loaded.pipeline.arch());
            let mmap_label = if loaded.pipeline.is_mmap() { " · mmap" } else { "" };
            drop(guard);
            print_banner(port, backend, Some((model_id, &arch, mmap_label)));
        }
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
            console::style(
                "no model pre-loaded — models load on first API request"
            )
            .dim()
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
