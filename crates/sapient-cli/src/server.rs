//! OpenAI-compatible HTTP inference server.
//!
//! Endpoints:
//!   GET  /v1/models                — list the loaded model (OpenAI format)
//!   POST /v1/chat/completions      — streaming (SSE) + non-streaming chat
//!   POST /v1/completions           — raw text completion
//!   GET  /v1/health                — health / version check

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

// ── Server state ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct ServeState {
    pipeline: Arc<Mutex<Pipeline>>,
    model_id: String,
}

// ── OpenAI request/response types ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

fn build_config(max_tokens: Option<usize>, temperature: Option<f32>, stop: Vec<String>) -> GenerationConfig {
    let mut cfg = GenerationConfig::default();
    if let Some(n) = max_tokens {
        cfg.max_new_tokens = n;
    }
    if let Some(t) = temperature {
        if t > 0.0 {
            cfg.strategy = SamplingStrategy::TopP { temperature: t, p: 0.95 };
        }
    }
    cfg.stop_sequences = stop;
    cfg
}

fn server_err(msg: impl std::fmt::Display) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": msg.to_string(), "type": "server_error"}})),
    )
        .into_response()
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn handle_models(State(state): State<ServeState>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "owned_by": "sapient",
            "created": now_secs(),
        }]
    }))
}

async fn handle_health() -> impl IntoResponse {
    Json(json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")}))
}

async fn handle_chat_completions(
    State(state): State<ServeState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model_id = state.model_id.clone();
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
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<sse::Event, Infallible>>(128);

        // Role delta
        let role_json = serde_json::to_string(&ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![DeltaChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
        })
        .unwrap();
        let _ = tx.send(Ok(sse::Event::default().data(role_json))).await;

        // Spawn generation
        let tx2 = tx.clone();
        tokio::task::spawn(async move {
            let guard = state.pipeline.lock().await;
            let mut stream = guard.chat_stream_with_config(&messages, &cfg).await;
            while let Some(token) = stream.next().await {
                let chunk_json = serde_json::to_string(&ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_id.clone(),
                    choices: vec![DeltaChoice {
                        index: 0,
                        delta: Delta { role: None, content: Some(token) },
                        finish_reason: None,
                    }],
                })
                .unwrap();
                if tx2.send(Ok(sse::Event::default().data(chunk_json))).await.is_err() {
                    return;
                }
            }
            // Stop chunk
            let stop_json = serde_json::to_string(&ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id,
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
        let result = {
            let guard = state.pipeline.lock().await;
            guard.chat_with_config(&messages, &cfg).await
        };
        match result {
            Ok(reply) => Json(ChatCompletionResponse {
                id: gen_id(),
                object: "chat.completion",
                created: now_secs(),
                model: model_id,
                choices: vec![ChatChoice {
                    index: 0,
                    message: OAIMessage { role: "assistant".into(), content: reply },
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
    let model_id = state.model_id.clone();
    let cfg = build_config(req.max_tokens, req.temperature, parse_stop(req.stop));
    let prompt = req.prompt;

    if req.stream {
        let id = gen_id();
        let created = now_secs();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<sse::Event, Infallible>>(128);
        let tx2 = tx.clone();

        tokio::task::spawn(async move {
            let guard = state.pipeline.lock().await;
            let mut stream = guard.generate_stream_with_config(&prompt, &cfg).await;
            while let Some(token) = stream.next().await {
                let data = serde_json::to_string(&json!({
                    "id": id, "object": "text_completion", "created": created,
                    "model": model_id,
                    "choices": [{"text": token, "index": 0, "finish_reason": null}]
                }))
                .unwrap();
                if tx2.send(Ok(sse::Event::default().data(data))).await.is_err() {
                    return;
                }
            }
            let _ = tx2.send(Ok(sse::Event::default().data("[DONE]"))).await;
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        let result = {
            let guard = state.pipeline.lock().await;
            guard.generate_with_config(&prompt, &cfg).await
        };
        match result {
            Ok(text) => Json(CompletionResponse {
                id: gen_id(),
                object: "text_completion",
                created: now_secs(),
                model: model_id,
                choices: vec![CompletionChoice { index: 0, text, finish_reason: "stop" }],
                usage: Usage::default(),
            })
            .into_response(),
            Err(e) => server_err(e),
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn serve_llm(model_id: &str, port: u16, backend: &str, mmap: bool) -> Result<()> {
    let backend_kind = crate::parse_generation_backend(backend)?;
    let opts = LoadOptions {
        backend: backend_kind,
        force_mmap: mmap,
        ..LoadOptions::default()
    };

    let spinner = crate::ui::spinner(format!("loading {model_id}…"));
    let pipeline = Pipeline::from_pretrained_with_opts(model_id, opts)
        .await
        .with_context(|| format!("failed to load model '{model_id}'"))?;
    spinner.finish_and_clear();

    let arch = format!("{:?}", pipeline.arch());
    let mmap_label = if pipeline.is_mmap() { " · mmap" } else { "" };

    let state = ServeState {
        pipeline: Arc::new(Mutex::new(pipeline)),
        model_id: model_id.to_string(),
    };

    let app = Router::new()
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/completions", post(handle_completions))
        .route("/v1/health", get(handle_health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    println!();
    println!(
        "  {} {}",
        console::style("⚡ SAPIENT serve").cyan().bold(),
        console::style(format!("· {arch} · {backend}{mmap_label}")).dim()
    );
    println!("  {} {}", console::style("model  ").dim(), console::style(model_id).bold());
    println!("  {} http://0.0.0.0:{port}", console::style("address").dim());
    println!();
    println!("  {}", console::style("OpenAI-compatible endpoints:").dim());
    println!("  {}  GET  /v1/models", console::style("·").dim());
    println!("  {}  POST /v1/chat/completions  (stream=true|false)", console::style("·").dim());
    println!("  {}  POST /v1/completions       (stream=true|false)", console::style("·").dim());
    println!();
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
            "    -d '{\"model\":\"...\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
        )
        .dim()
    );
    println!();

    info!("SAPIENT serve listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
