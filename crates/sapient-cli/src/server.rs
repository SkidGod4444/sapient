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
    extract::{Multipart, State},
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

use sapient_generate::{
    GenerationConfig, KokoroTts, LoadOptions, Pipeline, SamplingStrategy, SpeculativePipeline,
    TranscribeOptions, TranscribePipeline, Tts, VlmPipeline, DEFAULT_KOKORO_VOICE,
};
use sapient_tokenizers::{ChatMessage, ToolCall};

// ── ServedModel ────────────────────────────────────────────────────────────────
//
// A resident model is either a plain `Pipeline` or a `SpeculativePipeline`
// (target+draft). Both keep their forward engines loaded and reuse them across
// requests, and both expose the same `*_with_config` inference surface — so the
// cache, admission control, and route handlers treat them uniformly.

enum ServedModel {
    Plain(Box<Pipeline>),
    Speculative(Box<SpeculativePipeline>),
}

impl ServedModel {
    async fn chat_with_config(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> Result<String> {
        match self {
            ServedModel::Plain(p) => p.chat_with_config(messages, config).await,
            ServedModel::Speculative(p) => p.chat_with_config(messages, config).await,
        }
    }

    async fn chat_stream_with_config(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> ReceiverStream<String> {
        match self {
            ServedModel::Plain(p) => p.chat_stream_with_config(messages, config).await,
            ServedModel::Speculative(p) => p.chat_stream_with_config(messages, config).await,
        }
    }

    async fn generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        match self {
            ServedModel::Plain(p) => p.generate_with_config(prompt, config).await,
            ServedModel::Speculative(p) => p.generate_with_config(prompt, config).await,
        }
    }

    async fn generate_stream_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> ReceiverStream<String> {
        match self {
            ServedModel::Plain(p) => p.generate_stream_with_config(prompt, config).await,
            ServedModel::Speculative(p) => p.generate_stream_with_config(prompt, config).await,
        }
    }

    /// Architecture label for the startup banner.
    fn arch_label(&self) -> String {
        match self {
            ServedModel::Plain(p) => format!("{:?}", p.arch()),
            ServedModel::Speculative(p) => format!("{:?} +draft", p.arch()),
        }
    }

    fn is_mmap(&self) -> bool {
        match self {
            ServedModel::Plain(p) => p.is_mmap(),
            ServedModel::Speculative(p) => p.is_mmap(),
        }
    }

    /// Token count of `text` per the model's tokenizer (for OpenAI `usage`).
    fn count_tokens(&self, text: &str) -> usize {
        let encoded = match self {
            ServedModel::Plain(p) => p.tokenizer().encode(text),
            ServedModel::Speculative(p) => p.tokenizer().encode(text),
        };
        encoded.map(|t| t.len()).unwrap_or(0)
    }

    /// Render the chat prompt string (used to count prompt tokens for `usage`).
    /// Takes `tools` because the tools preamble is part of the prompt the model
    /// actually sees — omitting it would under-report `prompt_tokens`.
    fn format_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> anyhow::Result<String> {
        match self {
            ServedModel::Plain(p) => p.format_chat_prompt_with_tools(messages, tools),
            ServedModel::Speculative(p) => p.format_chat_prompt_with_tools(messages, tools),
        }
    }
}

// ── Multi-model LRU cache ─────────────────────────────────────────────────────
//
// Unlike Ollama (one resident model, cold reload on every switch), we keep the N
// most-recently-used models resident, bounded by a memory budget. Switching back
// to a recent model is then instant — no download, no re-quantization, no reload.
// Each cached model is an `Arc<CachedModel>`, so a streaming request keeps using
// its model after the cache lock is released (and even if the model is evicted
// mid-request — the Arc keeps it alive until the stream finishes).

/// A model held resident in the cache. Generic over the payload so the LRU
/// bookkeeping can be unit-tested without constructing a real `Pipeline`.
struct CachedModel<P> {
    model_id: String,
    payload: P,
    /// Resident-size estimate (bytes) used for budget-based eviction.
    bytes: u64,
}

/// Bounded LRU cache of loaded models. Most-recently-used is at the back.
struct ModelCache<P> {
    entries: VecDeque<Arc<CachedModel<P>>>,
    max_models: usize,
    budget_bytes: u64,
    used_bytes: u64,
}

impl<P> ModelCache<P> {
    fn new(max_models: usize, budget_bytes: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            max_models: max_models.max(1),
            budget_bytes,
            used_bytes: 0,
        }
    }

    /// Promote an already-resident model to MRU and return it.
    fn touch(&mut self, id: &str) -> Option<Arc<CachedModel<P>>> {
        let pos = self.entries.iter().position(|m| m.model_id == id)?;
        let m = self.entries.remove(pos).expect("position just found");
        self.entries.push_back(m.clone());
        Some(m)
    }

    /// Insert a freshly-loaded model at MRU, evicting LRU entries until both the
    /// count and byte budgets are satisfied. Returns the evicted model IDs.
    /// The just-inserted model is never evicted (it stays at the back).
    fn insert(&mut self, entry: Arc<CachedModel<P>>) -> Vec<String> {
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
    cache: Arc<Mutex<ModelCache<ServedModel>>>,
    /// Parallel LRU cache for Whisper STT models (POST /v1/audio/transcriptions),
    /// kept separate from the text `cache` so the text inference surface stays
    /// uncluttered. Shares `load_lock` and `inference_sem` with the text path.
    audio_cache: Arc<Mutex<ModelCache<Arc<TranscribePipeline>>>>,
    /// Parallel LRU cache for vision-language models (image parts in
    /// POST /v1/chat/completions — Phase 12.3), mirroring `audio_cache`.
    /// `VlmPipeline` inference is `&mut` + blocking, so each entry sits behind
    /// its own async Mutex whose owned guard moves into `spawn_blocking`.
    vlm_cache: Arc<Mutex<ModelCache<Arc<Mutex<VlmPipeline>>>>>,
    /// Parallel LRU cache for Kokoro TTS (POST /v1/audio/speech), mirroring
    /// `audio_cache`. A single entry serves every voice — all 54 embeddings are
    /// resident in the loaded model.
    tts_cache: Arc<Mutex<ModelCache<Arc<KokoroTts>>>>,
    /// Serializes model loads so two concurrent first-requests for the same model
    /// don't both download/load it, and loads don't thrash each other.
    load_lock: Arc<Mutex<()>>,
    /// Admission control: bounds the number of inferences running at once so a
    /// burst of requests queues fairly instead of oversubscribing the CPU/GPU and
    /// exploding thread/memory use. Excess requests await a permit.
    inference_sem: Arc<tokio::sync::Semaphore>,
    backend: String,
    force_mmap: bool,
    /// Serve every model with speculative decoding (target = requested model,
    /// draft = `draft_model` or an auto-selected small model).
    speculative: bool,
    /// Explicit draft model for speculative decoding (else auto-selected).
    draft_model: Option<String>,
}

impl ServeState {
    /// Return the resident model for `model_id`, loading (and LRU-evicting) it if
    /// needed. The cache lock is NOT held during the (slow) load or during
    /// inference, so cache hits and other models' requests aren't blocked.
    async fn get_or_load(&self, model_id: &str) -> Result<Arc<CachedModel<ServedModel>>> {
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

        let backend_kind = crate::parse_generation_backend(&self.backend)?;
        let opts = LoadOptions {
            backend: backend_kind,
            force_mmap: self.force_mmap,
            ..LoadOptions::default()
        };

        let payload = if self.speculative {
            info!("loading model '{model_id}' (speculative target)…");
            let spec = match &self.draft_model {
                Some(draft) => SpeculativePipeline::new_with_opts(model_id, draft, 5, opts).await,
                None => SpeculativePipeline::with_auto_draft_with_opts(model_id, 5, opts).await,
            }
            .with_context(|| format!("failed to load speculative pipeline for '{model_id}'"))?;
            ServedModel::Speculative(Box::new(spec))
        } else {
            info!("loading model '{model_id}'…");
            let mut pipeline = Pipeline::from_pretrained_with_opts(model_id, opts)
                .await
                .with_context(|| format!("failed to load model '{model_id}'"))?;
            // Reuse the KV cache across requests sharing a prompt prefix (multi-turn
            // chat, shared system prompts) — skips re-prefilling the whole history.
            pipeline.enable_prefix_cache();
            ServedModel::Plain(Box::new(pipeline))
        };

        // Speculative residency also holds a draft model in memory; add a rough
        // draft overhead so the byte budget isn't underestimated.
        let bytes = estimate_model_bytes(model_id)
            + if self.speculative {
                self.draft_model
                    .as_deref()
                    .map(estimate_model_bytes)
                    .unwrap_or(512 * 1024 * 1024)
            } else {
                0
            };
        let entry = Arc::new(CachedModel {
            model_id: model_id.to_string(),
            payload,
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

    /// Like [`get_or_load`], but for the Whisper STT (`TranscribePipeline`) cache
    /// used by `POST /v1/audio/transcriptions`. Same fast-path / load-lock /
    /// LRU-insert structure; no speculative variant.
    async fn get_or_load_audio(
        &self,
        model_id: &str,
    ) -> Result<Arc<CachedModel<Arc<TranscribePipeline>>>> {
        if let Some(m) = self.audio_cache.lock().await.touch(model_id) {
            return Ok(m);
        }
        let _load = self.load_lock.lock().await;
        if let Some(m) = self.audio_cache.lock().await.touch(model_id) {
            return Ok(m);
        }

        let backend_kind = crate::parse_generation_backend(&self.backend)?;
        info!("loading STT model '{model_id}'…");
        let pipeline = TranscribePipeline::from_pretrained_with_backend(model_id, backend_kind)
            .await
            .with_context(|| format!("failed to load STT model '{model_id}'"))?;
        let entry = Arc::new(CachedModel {
            model_id: model_id.to_string(),
            payload: Arc::new(pipeline),
            bytes: estimate_model_bytes(model_id),
        });
        let evicted = self.audio_cache.lock().await.insert(entry.clone());
        for id in &evicted {
            info!("evicted STT '{id}' from audio cache (LRU)");
        }
        Ok(entry)
    }

    /// Like [`get_or_load_audio`](Self::get_or_load_audio), but for the Kokoro
    /// TTS engine backing `POST /v1/audio/speech`.
    async fn get_or_load_tts(&self, model_id: &str) -> Result<Arc<CachedModel<Arc<KokoroTts>>>> {
        if let Some(m) = self.tts_cache.lock().await.touch(model_id) {
            return Ok(m);
        }
        let _load = self.load_lock.lock().await;
        if let Some(m) = self.tts_cache.lock().await.touch(model_id) {
            return Ok(m);
        }

        info!("loading TTS model '{model_id}'…");
        let tts = KokoroTts::from_default()
            .await
            .with_context(|| format!("failed to load TTS model '{model_id}'"))?;
        let entry = Arc::new(CachedModel {
            model_id: model_id.to_string(),
            payload: Arc::new(tts),
            bytes: estimate_model_bytes(model_id),
        });
        let evicted = self.tts_cache.lock().await.insert(entry.clone());
        for id in &evicted {
            info!("evicted TTS '{id}' from tts cache (LRU)");
        }
        Ok(entry)
    }

    /// Like [`get_or_load`], but for vision-language models (`VlmPipeline`)
    /// serving image parts in POST /v1/chat/completions. Same fast-path /
    /// load-lock / LRU-insert structure as the audio cache.
    async fn get_or_load_vlm(
        &self,
        model_id: &str,
    ) -> Result<Arc<CachedModel<Arc<Mutex<VlmPipeline>>>>> {
        if let Some(m) = self.vlm_cache.lock().await.touch(model_id) {
            return Ok(m);
        }
        let _load = self.load_lock.lock().await;
        if let Some(m) = self.vlm_cache.lock().await.touch(model_id) {
            return Ok(m);
        }

        info!("loading VLM '{model_id}'…");
        let pipeline = VlmPipeline::from_pretrained(model_id)
            .await
            .with_context(|| format!("failed to load VLM '{model_id}'"))?;
        let entry = Arc::new(CachedModel {
            model_id: model_id.to_string(),
            payload: Arc::new(Mutex::new(pipeline)),
            bytes: estimate_model_bytes(model_id),
        });
        let evicted = self.vlm_cache.lock().await.insert(entry.clone());
        for id in &evicted {
            info!("evicted VLM '{id}' from vlm cache (LRU)");
        }
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
                    if let Some(kb) = rest.split_whitespace().next() {
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
    /// OpenAI tool definitions, passed to the chat template verbatim. Models
    /// whose template has no tools branch ignore them and answer in prose.
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    /// `"none"` suppresses the tools preamble; `"auto"` (the default) lets the
    /// model decide; `"required"` and a named-function object force a call —
    /// see [`ChatCompletionRequest::tool_prefill`].
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
}

impl ChatCompletionRequest {
    /// The tools to expose, after applying `tool_choice: "none"`.
    fn active_tools(&self) -> Option<Vec<serde_json::Value>> {
        if self.tool_choice.as_ref().and_then(|c| c.as_str()) == Some("none") {
            return None;
        }
        let tools = self.tools.as_deref().filter(|t| !t.is_empty())?;
        Some(tools.iter().map(sanitize_tool).collect())
    }

    /// The assistant-turn prefix that makes `tool_choice` binding.
    ///
    /// `"auto"` is a request; `"required"` and `{"type":"function", …}` are
    /// instructions — the model *must* call a tool, and OpenAI clients rely on
    /// that. With no constrained sampler, the way to make it true is to write
    /// the beginning of the answer ourselves: open the `<tool_call>` tag (and,
    /// for a named function, the name too) and hand the model a continuation it
    /// cannot escape into prose.
    ///
    /// This matters well beyond spec compliance. A caller that must not be
    /// answered from imagination — "look before you tell me what you see" —
    /// has no other lever, and a small model *will* otherwise describe a room
    /// it never looked at.
    fn tool_prefill(&self) -> Option<String> {
        self.active_tools()?; // nothing to force if no tools are on offer
        match self.tool_choice.as_ref()? {
            serde_json::Value::String(s) if s == "required" => {
                Some("<tool_call>\n".to_owned())
            }
            // {"type":"function","function":{"name":"look"}} — force that one.
            //
            // The trailing `{` matters: stop at `"arguments": ` and the model is
            // free to continue with any JSON value, and a small one will happily
            // write `0`. Opening the object commits it to writing arguments.
            serde_json::Value::Object(o) => {
                let name = o.get("function")?.get("name")?.as_str()?;
                Some(format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {{"))
            }
            _ => None, // "auto" | "none"
        }
    }
}

/// Strip JSON-Schema *dialect* metadata from a tool definition.
///
/// Chat templates serialize each tool verbatim (`{{- tool | tojson }}`) straight
/// into the model's tools preamble, so whatever the client sends is what the
/// model reads. Zod v4's JSON-Schema emitter — which the Vercel AI SDK uses —
/// tags every schema with `"$schema": "http://json-schema.org/draft-07/schema#"`.
///
/// That key is a declaration *about the schema language*; it says nothing about
/// the function's arguments, and tool-trained models never saw it in training.
/// Left in, a small model reads the preamble, drifts off-distribution, and
/// answers in prose instead of calling the tool — verified on Qwen2.5-1.5B,
/// where its presence alone is the difference between a `walk` call and the
/// model merely *claiming* to have walked. On a robot, that distinction is the
/// whole game.
///
/// Only dialect metadata is removed. Constraints the model should actually see
/// (`enum`, `required`, `additionalProperties`, `minimum`, …) are preserved.
fn sanitize_tool(tool: &serde_json::Value) -> serde_json::Value {
    fn strip(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(map) => serde_json::Value::Object(
                map.iter()
                    .filter(|(k, _)| k.as_str() != "$schema")
                    .map(|(k, v)| (k.clone(), strip(v)))
                    .collect(),
            ),
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(strip).collect())
            }
            other => other.clone(),
        }
    }
    strip(tool)
}

/// `POST /v1/audio/speech` body (OpenAI shape).
#[derive(Debug, Deserialize)]
struct SpeechRequest {
    #[serde(default)]
    model: Option<String>,
    /// The text to synthesize.
    input: String,
    /// One of Kokoro's 54 voices, e.g. `af_heart`. Defaults to `af_heart`.
    #[serde(default)]
    voice: Option<String>,
    /// `wav` or `pcm`. MP3/Opus/AAC/FLAC are not supported.
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
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
    /// Optional because an assistant turn that *only* calls tools carries
    /// `"content": null` — OpenAI clients send exactly that back on the next
    /// turn of a tool loop, so a non-optional field would reject every
    /// multi-step agent conversation at deserialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<OAIContent>,
    /// Tool calls the assistant asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OAIToolCall>>,
    /// On a `role: "tool"` turn: which call this result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl OAIMessage {
    /// The message's text, or empty for a content-less tool-call turn.
    fn text(&self) -> String {
        self.content.as_ref().map(OAIContent::text).unwrap_or_default()
    }

    fn image_urls(&self) -> Vec<&str> {
        self.content
            .as_ref()
            .map(OAIContent::image_urls)
            .unwrap_or_default()
    }
}

/// An OpenAI tool call. `function.arguments` is a JSON-encoded **string** on
/// the wire (not an object) — that is OpenAI's format, and clients parse it as
/// such, so it must be stringified on the way out and parsed on the way in.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OAIFunctionCall,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OAIFunctionCall {
    pub name: String,
    /// JSON-encoded arguments, e.g. `"{\"meters\":2}"`.
    pub arguments: String,
}

/// OpenAI message `content`: a plain string, or an array of typed parts —
/// text and base64 data-URI images (Phase 12.3). Untagged, so plain-string
/// clients keep working unchanged, and `Text` serializes back to a plain
/// string in responses.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum OAIContent {
    Text(String),
    Parts(Vec<OAIContentPart>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OAIContentPart {
    Text { text: String },
    ImageUrl { image_url: OAIImageUrl },
}

/// OpenAI `image_url` payload. `detail` is accepted for compatibility and
/// ignored (the tower has one fixed input resolution).
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OAIImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl OAIContent {
    /// The message's text: the whole string, or all text parts joined.
    fn text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    OAIContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// URLs of every `image_url` part (empty for plain-string content).
    fn image_urls(&self) -> Vec<&str> {
        match self {
            Self::Text(_) => Vec::new(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    OAIContentPart::ImageUrl { image_url } => Some(image_url.url.as_str()),
                    _ => None,
                })
                .collect(),
        }
    }
}

/// Decode an OpenAI image part URL. Only `data:<mime>;base64,<payload>` is
/// accepted — the server never fetches remote image URLs (no surprise egress).
fn decode_image_data_uri(url: &str) -> Result<Vec<u8>> {
    let rest = url.strip_prefix("data:").ok_or_else(|| {
        anyhow::anyhow!(
            "only base64 data URIs are supported (data:image/...;base64,...); \
             the server does not fetch remote image URLs"
        )
    })?;
    let (_mime, payload) = rest.split_once(";base64,").ok_or_else(|| {
        anyhow::anyhow!("image data URI must be base64-encoded (data:image/...;base64,...)")
    })?;
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .context("invalid base64 in image data URI")
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
    /// Token usage — OpenAI sends this only on the final chunk of a stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

/// A tool call inside a streaming delta. OpenAI streams `arguments` as partial
/// string fragments accumulated by `index`; we emit each call complete in a
/// single delta, which clients accumulate identically (one fragment that
/// happens to be the whole thing).
#[derive(Serialize)]
struct DeltaToolCall {
    index: usize,
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: OAIFunctionCall,
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

#[derive(Serialize, Default, Clone, Copy)]
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
    tools: Option<&[serde_json::Value]>,
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
    cfg.tools = tools.map(<[serde_json::Value]>::to_vec);
    cfg
}

// ── Tool calls ────────────────────────────────────────────────────────────────

/// Translate OpenAI messages into template messages.
///
/// The one subtlety is `arguments`: OpenAI carries it as a JSON-encoded
/// *string*, while chat templates pipe it through `| tojson`. Handing the
/// template the raw string would render it double-encoded, and the model
/// faithfully imitates that broken shape on its next turn — so it is parsed
/// back into a value here.
fn to_chat_messages(messages: &[OAIMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|m| match m.role.as_str() {
            "assistant" => match m.tool_calls.as_deref() {
                Some(calls) if !calls.is_empty() => ChatMessage::assistant_tool_calls(
                    m.text(),
                    calls
                        .iter()
                        .map(|c| ToolCall {
                            name: c.function.name.clone(),
                            arguments: serde_json::from_str(&c.function.arguments)
                                .unwrap_or_else(|_| json!({})),
                        })
                        .collect(),
                ),
                _ => ChatMessage::assistant(m.text()),
            },
            "tool" => ChatMessage::tool(m.text()),
            // A system prompt rendered as a user turn degrades the persona and
            // reads as if the user said it — map it to the template's system role.
            "system" | "developer" => ChatMessage::system(m.text()),
            _ => ChatMessage::user(m.text()),
        })
        .collect()
}

/// Lift `<tool_call>{"name":…,"arguments":{…}}</tool_call>` blocks out of the
/// model's text into structured OpenAI tool calls, returning the prose that sat
/// outside them alongside.
///
/// Tool-trained models emit calls this way because their chat template
/// explicitly instructs them to — the tags are the model's half of the
/// contract, and this is the other half.
///
/// A block that is unterminated (generation hit `max_tokens` mid-call) or whose
/// body will not parse is deliberately **left in the text** rather than
/// dropped, so a broken call surfaces to the caller instead of vanishing into
/// an empty assistant turn.
fn extract_tool_calls(reply: &str) -> (String, Vec<OAIToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut calls: Vec<OAIToolCall> = Vec::new();
    let mut text = String::new();
    let mut rest = reply;

    while let Some(start) = rest.find(OPEN) {
        let body_start = start + OPEN.len();

        let (body, block_end) = match rest[body_start..].find(CLOSE) {
            Some(rel_end) => (
                &rest[body_start..body_start + rel_end],
                body_start + rel_end + CLOSE.len(),
            ),
            // No closing tag: either a call cut short by `max_tokens`, or a model
            // that finished its JSON and forgot the tag — common when generation
            // was prefilled *into* the call, since the model never saw itself
            // open one. Let JSON validity decide rather than guessing: a complete
            // call parses and is honored; a truncated one does not, and stays
            // visible as text below.
            None => (&rest[body_start..], rest.len()),
        };

        match parse_tool_call(body, calls.len()) {
            Some(call) => {
                text.push_str(&rest[..start]);
                calls.push(call);
            }
            None => text.push_str(&rest[..block_end]),
        }
        rest = &rest[block_end..];
    }
    text.push_str(rest);

    (text.trim().to_string(), calls)
}

/// Re-attach a forced-tool-call prefill to the model's completion.
///
/// The `<tool_call>` opener lives in the *prompt*, so the model never emits it —
/// without this the reply looks like a bare JSON fragment and parses as prose.
fn stitch(prefill: Option<&str>, reply: &str) -> String {
    match prefill {
        Some(p) => format!("{p}{reply}"),
        None => reply.to_owned(),
    }
}

fn parse_tool_call(body: &str, index: usize) -> Option<OAIToolCall> {
    let v: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_owned();

    // Back to OpenAI's stringified form. A model that already emitted a string
    // passes through untouched rather than being double-encoded.
    //
    // A no-args call must serialize to `"{}"`, never `"null"`: clients parse
    // this string to get the tool's input, and `null` is silently treated as
    // "no call to execute" — the tool never runs and the agent loop stalls with
    // no error to explain why.
    // Anything that is not an object cannot be a valid argument set, so it
    // becomes `"{}"` rather than being passed through. A model completing a
    // forced call sometimes writes a bare scalar (`0`); stringifying that to
    // `"0"` would hand the client something it cannot parse as arguments, and
    // the tool would fail for a reason nothing in the response explains.
    let arguments = match v.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(obj @ serde_json::Value::Object(_)) => obj.to_string(),
        _ => "{}".to_owned(),
    };

    Some(OAIToolCall {
        id: format!(
            "call_{}_{index}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ),
        kind: "function".to_owned(),
        function: OAIFunctionCall { name, arguments },
    })
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
    cache: &Mutex<ModelCache<ServedModel>>,
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
    let audio_resident = state.audio_cache.lock().await.ids();
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "loaded_model": active,
        "resident_models": resident,
        "audio_models": audio_resident,
    }))
}

/// `POST /v1/audio/transcriptions` (OpenAI-compatible, `multipart/form-data`).
/// Fields: `file` (audio bytes, required), `model` (Whisper alias/repo, required),
/// optional `language`, `response_format` (`json` default | `text`), `translate`.
async fn handle_audio_transcriptions(
    State(state): State<ServeState>,
    mut multipart: Multipart,
) -> Response {
    let mut audio: Option<(Vec<u8>, String)> = None; // (bytes, extension)
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format = String::from("json");
    let mut translate = false;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return audio_err(
                    StatusCode::BAD_REQUEST,
                    &format!("malformed multipart: {e}"),
                )
            }
        };
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                // Preserve the upload's extension so symphonia can sniff the format.
                let ext = field
                    .file_name()
                    .and_then(|f| f.rsplit('.').next())
                    .filter(|e| !e.is_empty())
                    .unwrap_or("wav")
                    .to_string();
                match field.bytes().await {
                    Ok(b) => audio = Some((b.to_vec(), ext)),
                    Err(e) => {
                        return audio_err(StatusCode::BAD_REQUEST, &format!("reading file: {e}"))
                    }
                }
            }
            "model" => model = field.text().await.ok(),
            "language" => language = field.text().await.ok().filter(|s| !s.is_empty()),
            "response_format" => {
                if let Ok(v) = field.text().await {
                    response_format = v;
                }
            }
            "translate" => {
                translate = field
                    .text()
                    .await
                    .map(|v| matches!(v.as_str(), "true" | "1"))
                    .unwrap_or(false);
            }
            _ => {} // ignore unknown fields (temperature, etc.)
        }
    }

    let Some((bytes, ext)) = audio else {
        return audio_err(StatusCode::BAD_REQUEST, "missing `file` field");
    };
    let Some(model) = model else {
        return audio_err(StatusCode::BAD_REQUEST, "missing `model` field");
    };

    // Persist to a temp file (load_audio dispatches on the path's extension).
    let tmp = match tempfile::Builder::new()
        .suffix(&format!(".{ext}"))
        .tempfile()
    {
        Ok(t) => t,
        Err(e) => return audio_err(StatusCode::INTERNAL_SERVER_ERROR, &format!("tempfile: {e}")),
    };
    if let Err(e) = std::fs::write(tmp.path(), &bytes) {
        return audio_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("writing audio: {e}"),
        );
    }

    // Admission control (shared with text inference).
    let _permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return audio_err(StatusCode::SERVICE_UNAVAILABLE, "server shutting down"),
    };

    let pipeline = match state.get_or_load_audio(&model).await {
        Ok(m) => m,
        Err(e) => return audio_err(StatusCode::BAD_REQUEST, &format!("load '{model}': {e:#}")),
    };

    let opts = TranscribeOptions {
        language,
        translate,
        ..Default::default()
    };
    let text = match pipeline.payload.transcribe_with(tmp.path(), opts).await {
        Ok(t) => t,
        Err(e) => {
            return audio_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("transcribe: {e:#}"),
            )
        }
    };

    if response_format == "text" {
        text.into_response()
    } else {
        Json(json!({ "text": text })).into_response()
    }
}

/// OpenAI-style error envelope for the audio endpoint.
fn audio_err(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": { "message": msg } }))).into_response()
}

/// `POST /v1/audio/speech` — OpenAI-compatible text-to-speech.
///
/// Returns the synthesized waveform as a 16-bit PCM WAV body. OpenAI defaults
/// this endpoint to MP3; SAPIENT has no MP3 encoder (and pulling one in for a
/// local endpoint isn't worth it), so `wav`/`pcm` are the supported formats and
/// anything else is rejected loudly rather than silently returning WAV bytes
/// under an `audio/mpeg` header.
async fn handle_audio_speech(
    State(state): State<ServeState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let model_id = req.model.filter(|m| !m.is_empty()).unwrap_or_else(|| "kokoro-82m".into());

    if let Some(fmt) = req.response_format.as_deref() {
        if !matches!(fmt, "wav" | "pcm") {
            return model_err(format!(
                "response_format '{fmt}' is not supported — SAPIENT synthesizes WAV \
                 (16-bit PCM). Use \"wav\", or omit the field."
            ));
        }
    }
    if req.input.trim().is_empty() {
        return model_err("input must not be empty");
    }

    let tts = match state.get_or_load_tts(&model_id).await {
        Ok(t) => t,
        Err(e) => return server_err(format!("{e:#}")),
    };

    let permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return server_err("server is shutting down"),
    };

    let voice = req
        .voice
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_KOKORO_VOICE.to_string());
    let speed = req.speed.unwrap_or(1.0);
    let input = req.input;

    // Synthesis is CPU-bound and blocking — keep it off the async runtime.
    let engine = tts.payload.clone();
    let synth = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let samples = engine.synthesize_as(&input, &voice, speed)?;
        Ok::<_, anyhow::Error>(sapient_audio::encode_wav(&samples, engine.sample_rate()))
    })
    .await;

    match synth {
        Ok(Ok(wav)) => (
            StatusCode::OK,
            [
                (axum::http::header::CONTENT_TYPE, "audio/wav"),
                (axum::http::header::CACHE_CONTROL, "no-store"),
            ],
            wav,
        )
            .into_response(),
        // An unknown voice lands here — a client error, not a server fault.
        Ok(Err(e)) => model_err(format!("{e:#}")),
        Err(e) => server_err(format!("speech synthesis task panicked: {e}")),
    }
}

async fn handle_chat_completions(
    State(state): State<ServeState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model_id = match resolve_model(req.model.as_deref(), &state.cache).await {
        Ok(m) => m,
        Err(r) => return r,
    };

    // Image parts route to the vision pipeline (Phase 12.3).
    if req.messages.iter().any(|m| !m.image_urls().is_empty()) {
        return handle_vision_chat(state, req, model_id).await;
    }

    let model = match state.get_or_load(&model_id).await {
        Ok(m) => m,
        Err(e) => return server_err(e),
    };

    // Admission control: wait for an inference slot (bounds concurrency).
    let permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return server_err("server is shutting down"),
    };

    let messages: Vec<ChatMessage> = to_chat_messages(&req.messages);
    let tools: Option<Vec<serde_json::Value>> = req.active_tools();
    // Forced tool use: we open the assistant's `<tool_call>` ourselves, so the
    // model resumes *inside* a call. Whatever it generates must be stitched back
    // onto this prefix before parsing — the opening tag is in the prompt, not in
    // the completion.
    let prefill = req.tool_prefill();
    let mut cfg = build_config(
        req.max_tokens,
        req.temperature,
        parse_stop(req.stop),
        tools.as_deref(),
    );
    cfg.prefill = prefill.clone();

    // Prompt tokens for `usage` — the rendered chat prompt is what the model sees.
    let prompt_tokens = model
        .payload
        .format_chat_prompt(&messages, tools.as_deref())
        .map(|p| model.payload.count_tokens(&p))
        .unwrap_or(0);

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
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        })
        .unwrap();
        let _ = tx.send(Ok(sse::Event::default().data(role_json))).await;

        let tx2 = tx.clone();
        let tools_active = tools.is_some();
        let prefill = prefill.clone();
        tokio::task::spawn(async move {
            // `model` (Arc) is moved in — no cache lock held during streaming, so
            // other models' requests run concurrently and the cache stays free.
            // `permit` is held for the stream's lifetime, released on completion.
            let _permit = permit;
            let mut full = String::new();
            let mut stream = model.payload.chat_stream_with_config(&messages, &cfg).await;
            while let Some(token) = stream.next().await {
                full.push_str(&token);

                // When tools are on the table, hold the text back instead of
                // streaming it. A tool call arrives as `<tool_call>…</tool_call>`
                // markup that has to be lifted into a structured delta once
                // complete — streaming it token-by-token would spray raw tags at
                // the client as if they were prose.
                if tools_active {
                    continue;
                }

                let chunk_json = serde_json::to_string(&ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_clone.clone(),
                    choices: vec![DeltaChoice {
                        index: 0,
                        delta: Delta {
                            content: Some(token),
                            ..Default::default()
                        },
                        finish_reason: None,
                    }],
                    usage: None,
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

            let completion_tokens = model.payload.count_tokens(&full);
            let mut finish_reason = "stop";

            // The buffered tool turn: emit the whole thing as one delta.
            if tools_active {
                let (text, calls) = extract_tool_calls(&stitch(prefill.as_deref(), &full));
                let delta = if calls.is_empty() {
                    Delta {
                        content: Some(text),
                        ..Default::default()
                    }
                } else {
                    finish_reason = "tool_calls";
                    Delta {
                        content: (!text.is_empty()).then_some(text),
                        tool_calls: Some(
                            calls
                                .into_iter()
                                .enumerate()
                                .map(|(index, c)| DeltaToolCall {
                                    index,
                                    id: c.id,
                                    kind: c.kind,
                                    function: c.function,
                                })
                                .collect(),
                        ),
                        ..Default::default()
                    }
                };
                let chunk_json = serde_json::to_string(&ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_clone.clone(),
                    choices: vec![DeltaChoice {
                        index: 0,
                        delta,
                        finish_reason: None,
                    }],
                    usage: None,
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

            // Final chunk carries token usage (OpenAI convention).
            let stop_json = serde_json::to_string(&ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_clone,
                choices: vec![DeltaChoice {
                    index: 0,
                    delta: Delta::default(),
                    finish_reason: Some(finish_reason),
                }],
                usage: Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                }),
            })
            .unwrap();
            let _ = tx2.send(Ok(sse::Event::default().data(stop_json))).await;
            let _ = tx2.send(Ok(sse::Event::default().data("[DONE]"))).await;
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        match model.payload.chat_with_config(&messages, &cfg).await {
            Ok(reply) => {
                let completion_tokens = model.payload.count_tokens(&reply);

                // Only mine the reply for calls when tools were actually
                // offered — otherwise a model merely *quoting* `<tool_call>` in
                // prose would be misread as calling one.
                let (text, tool_calls) = if tools.is_some() {
                    extract_tool_calls(&stitch(prefill.as_deref(), &reply))
                } else {
                    (reply, Vec::new())
                };
                let called = !tool_calls.is_empty();

                Json(ChatCompletionResponse {
                    id: gen_id(),
                    object: "chat.completion",
                    created: now_secs(),
                    model: model_id,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: OAIMessage {
                            role: "assistant".into(),
                            // A pure tool-call turn carries no content — OpenAI
                            // puts `null` there, and clients expect that.
                            content: (!called || !text.is_empty())
                                .then_some(OAIContent::Text(text)),
                            tool_calls: called.then_some(tool_calls),
                            tool_call_id: None,
                        },
                        finish_reason: if called { "tool_calls" } else { "stop" },
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                })
                .into_response()
            }
            Err(e) => server_err(e),
        }
    }
}

/// One vision-language turn (Phase 12.3): the final message must be the user
/// turn carrying exactly one `image_url` part — a base64 data URI — and its
/// text parts are the question (single-image, single-turn, matching the
/// `sapient see` v1 scope). `stream: true` is honored as a single content
/// chunk plus the usage chunk — the VLM pipeline decodes greedily without a
/// token stream yet.
async fn handle_vision_chat(
    state: ServeState,
    req: ChatCompletionRequest,
    model_id: String,
) -> Response {
    let Some(last) = req.messages.last() else {
        return model_err("messages must not be empty");
    };
    let earlier_images = req.messages[..req.messages.len() - 1]
        .iter()
        .any(|m| !m.image_urls().is_empty());
    if earlier_images || last.role != "user" {
        return model_err(
            "image parts are only supported in the final user message (single-turn vision v1)",
        );
    }
    let urls = last.image_urls();
    if urls.len() != 1 {
        return model_err("exactly one image part per request is supported (vision v1)");
    }
    let image_bytes = match decode_image_data_uri(urls[0]) {
        Ok(b) => b,
        Err(e) => return model_err(format!("{e:#}")),
    };
    let question = last.text();

    let model = match state.get_or_load_vlm(&model_id).await {
        Ok(m) => m,
        Err(e) => return server_err(format!("{e:#}")),
    };

    // Admission control, as in the text path.
    let _permit = match state.inference_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return server_err("server is shutting down"),
    };

    let max_new = req.max_tokens.unwrap_or(512);
    // The owned guard moves into the blocking task; concurrent requests for the
    // same VLM queue on the async mutex without blocking an executor thread.
    let mut vlm = model.payload.clone().lock_owned().await;
    let result = tokio::task::spawn_blocking(move || {
        vlm.answer_bytes_with_stats(&image_bytes, &question, max_new)
    })
    .await;
    let (reply, stats) = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return server_err(format!("{e:#}")),
        Err(e) => return server_err(format!("vision inference task failed: {e}")),
    };
    let usage = Usage {
        prompt_tokens: stats.prompt_tokens,
        completion_tokens: stats.gen_tokens,
        total_tokens: stats.prompt_tokens + stats.gen_tokens,
    };
    info!(
        "vision turn: {} prompt tokens, {} generated (vision {} ms · prefill {} ms · decode {} ms)",
        stats.prompt_tokens, stats.gen_tokens, stats.vision_ms, stats.prefill_ms, stats.decode_ms
    );

    let id = gen_id();
    let created = now_secs();
    if req.stream {
        let chunk = |delta: Delta, finish: Option<&'static str>, usage: Option<Usage>| {
            serde_json::to_string(&ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id.clone(),
                choices: vec![DeltaChoice {
                    index: 0,
                    delta,
                    finish_reason: finish,
                }],
                usage,
            })
            .unwrap()
        };
        let events = vec![
            chunk(
                Delta {
                    role: Some("assistant"),
                    ..Default::default()
                },
                None,
                None,
            ),
            chunk(
                Delta {
                    content: Some(reply),
                    ..Default::default()
                },
                None,
                None,
            ),
            chunk(Delta::default(), Some("stop"), Some(usage)),
            "[DONE]".to_string(),
        ];
        let stream = futures::stream::iter(
            events
                .into_iter()
                .map(|e| Ok::<_, Infallible>(sse::Event::default().data(e))),
        );
        Sse::new(stream).into_response()
    } else {
        Json(ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model: model_id,
            choices: vec![ChatChoice {
                index: 0,
                message: OAIMessage {
                    role: "assistant".into(),
                    content: Some(OAIContent::Text(reply)),
                    // The VLM path is single-turn: it describes an image, it
                    // does not drive tools.
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop",
            }],
            usage,
        })
        .into_response()
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

    let cfg = build_config(req.max_tokens, req.temperature, parse_stop(req.stop), None);
    let prompt = req.prompt;
    let prompt_tokens = model.payload.count_tokens(&prompt);

    if req.stream {
        let id = gen_id();
        let created = now_secs();
        let model_clone = model_id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<sse::Event, Infallible>>(128);
        let tx2 = tx.clone();

        tokio::task::spawn(async move {
            let _permit = permit;
            let mut full = String::new();
            let mut stream = model
                .payload
                .generate_stream_with_config(&prompt, &cfg)
                .await;
            while let Some(token) = stream.next().await {
                full.push_str(&token);
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
            // Final chunk carries token usage (OpenAI convention).
            let completion_tokens = model.payload.count_tokens(&full);
            let usage = json!({
                "id": id, "object": "text_completion", "created": created,
                "model": model_clone,
                "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}
            });
            let _ = tx2
                .send(Ok(sse::Event::default().data(usage.to_string())))
                .await;
            let _ = tx2.send(Ok(sse::Event::default().data("[DONE]"))).await;
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        match model.payload.generate_with_config(&prompt, &cfg).await {
            Ok(text) => {
                let completion_tokens = model.payload.count_tokens(&text);
                Json(CompletionResponse {
                    id: gen_id(),
                    object: "text_completion",
                    created: now_secs(),
                    model: model_id,
                    choices: vec![CompletionChoice {
                        index: 0,
                        text,
                        finish_reason: "stop",
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                })
                .into_response()
            }
            Err(e) => server_err(e),
        }
    }
}

// ── Single-instance lock ───────────────────────────────────────────────────────
//
// Only one `sapient serve` should run per machine — two instances (even on
// different ports) double the resident-model RAM and confuse which one a tunnel
// targets. A pidfile guards this: startup refuses if a live serve already holds
// it; the file is removed on clean exit. A stale file (previous crash) is taken
// over. Override with SAPIENT_ALLOW_MULTIPLE=1.

/// RAII pidfile lock for the serve process. Removed on drop.
struct ServeLock {
    path: std::path::PathBuf,
}

impl ServeLock {
    fn acquire() -> Result<Option<Self>> {
        if std::env::var("SAPIENT_ALLOW_MULTIPLE").is_ok() {
            return Ok(None);
        }
        let path = serve_lock_path();
        if let Ok(contents) = std::fs::read_to_string(&path) {
            if let Ok(pid) = contents.trim().parse::<i32>() {
                if pid != std::process::id() as i32 && pid_alive(pid) {
                    anyhow::bail!(
                        "another `sapient serve` is already running (PID {pid}).\n\
                         Stop it first (e.g. `kill {pid}`), or set SAPIENT_ALLOW_MULTIPLE=1 \
                         to run a second instance. Lock: {}",
                        path.display()
                    );
                }
                // Stale pidfile (previous instance crashed) — take it over.
            }
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&path, std::process::id().to_string())
            .with_context(|| format!("failed to write serve lock {}", path.display()))?;
        Ok(Some(Self { path }))
    }
}

impl Drop for ServeLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn serve_lock_path() -> std::path::PathBuf {
    dirs::runtime_dir()
        .or_else(dirs::cache_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("sapient")
        .join("serve.lock")
}

/// True if a process with `pid` exists. `kill(pid, 0)` returns 0 when alive and
/// `EPERM` when it exists but we can't signal it; only `ESRCH` means truly gone.
#[cfg(unix)]
fn pid_alive(pid: i32) -> bool {
    let rc = unsafe { libc::kill(pid, 0) };
    if rc == 0 {
        true
    } else {
        // EPERM → exists but we can't signal it; ESRCH → truly gone.
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
}

#[cfg(not(unix))]
fn pid_alive(_pid: i32) -> bool {
    // No portable liveness check; assume a present pidfile means a live instance
    // (a stale one can be removed manually).
    true
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub async fn serve_llm(
    preload_model: Option<&str>,
    port: u16,
    backend: &str,
    mmap: bool,
    max_models: usize,
    cache_gb: f64,
    max_concurrency: usize,
    speculative: bool,
    draft_model: Option<&str>,
) -> Result<()> {
    // Refuse to start if another `sapient serve` is already running (single
    // instance per machine). Held for the server's lifetime; removed on exit.
    let _serve_lock = ServeLock::acquire()?;

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
    if speculative {
        match draft_model {
            Some(d) => info!("speculative decoding enabled (draft model: {d})"),
            None => info!("speculative decoding enabled (auto-selected draft model)"),
        }
    }
    let state = ServeState {
        cache: Arc::new(Mutex::new(ModelCache::<ServedModel>::new(
            max_models,
            budget_bytes,
        ))),
        audio_cache: Arc::new(Mutex::new(ModelCache::<Arc<TranscribePipeline>>::new(
            max_models,
            budget_bytes,
        ))),
        vlm_cache: Arc::new(Mutex::new(ModelCache::<Arc<Mutex<VlmPipeline>>>::new(
            max_models,
            budget_bytes,
        ))),
        tts_cache: Arc::new(Mutex::new(ModelCache::<Arc<KokoroTts>>::new(
            max_models,
            budget_bytes,
        ))),
        load_lock: Arc::new(Mutex::new(())),
        inference_sem: Arc::new(tokio::sync::Semaphore::new(concurrency)),
        backend: backend.to_string(),
        force_mmap: mmap,
        speculative,
        draft_model: draft_model.map(str::to_string),
    };

    if let Some(model_id) = preload_model {
        let spinner = crate::ui::spinner(format!("loading {model_id}…"));
        let entry = state.get_or_load(model_id).await?;
        spinner.finish_and_clear();
        let arch = entry.payload.arch_label();
        let mmap_label = if entry.payload.is_mmap() {
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
        // Speech-to-text (multipart audio upload). Higher body limit than the text
        // routes — audio files dwarf chat payloads.
        .route(
            "/v1/audio/transcriptions",
            post(handle_audio_transcriptions)
                .layer(axum::extract::DefaultBodyLimit::max(512 * 1024 * 1024)),
        )
        // Text-to-speech. Returns a WAV body, not JSON.
        .route("/v1/audio/speech", post(handle_audio_speech))
        .layer(CorsLayer::permissive())
        // Allow large prompts (long context / pasted documents) but cap to guard
        // against unbounded request bodies. 32 MiB ≫ any realistic chat payload.
        .layer(axum::extract::DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("SAPIENT serve listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    // Graceful Ctrl-C: without this the process dies without unwinding, the
    // ServeLock pidfile survives, and the next `sapient serve` refuses to start
    // (on Windows pid_alive can't verify liveness, so it would refuse forever).
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("\n  shutting down…");
        })
        .await?;
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
        "  {} http://localhost:{}",
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tool calling ────────────────────────────────────────────────────────
    //
    // The rules below are the wire contract the OpenAI SDKs actually enforce
    // (verified against `@ai-sdk/openai-compatible`), not merely what reads
    // nicely — each assertion here corresponds to a client-side hard failure.

    /// `function.arguments` must be a JSON-encoded **string**. An object there
    /// makes the AI SDK throw `Invalid JSON response` — the single easiest way
    /// to get a hand-rolled tool-calling server wrong.
    #[test]
    fn tool_call_arguments_are_a_json_string() {
        let (text, calls) =
            extract_tool_calls(r#"<tool_call>{"name":"walk","arguments":{"meters":2}}</tool_call>"#);

        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "walk");
        assert_eq!(calls[0].function.arguments, r#"{"meters":2}"#);
        assert_eq!(calls[0].kind, "function");
        assert!(!calls[0].id.is_empty(), "clients require an id when streaming");
    }

    /// A no-argument call must produce `"{}"`. `"null"` is silently treated as
    /// "nothing to execute", so the tool never runs and the agent loop stalls
    /// without surfacing an error.
    #[test]
    fn tool_call_without_arguments_is_empty_object_not_null() {
        let (_, calls) = extract_tool_calls(r#"<tool_call>{"name":"sit"}</tool_call>"#);
        assert_eq!(calls[0].function.arguments, "{}");

        let (_, calls) =
            extract_tool_calls(r#"<tool_call>{"name":"sit","arguments":null}</tool_call>"#);
        assert_eq!(calls[0].function.arguments, "{}", "null must not survive");

        // A model completing a forced call sometimes writes a bare scalar. It is
        // not a valid argument set, and `"0"` is not something a client can parse.
        let (_, calls) =
            extract_tool_calls(r#"<tool_call>{"name":"sit","arguments":0}</tool_call>"#);
        assert_eq!(calls[0].function.arguments, "{}", "scalar must not survive");
    }

    /// Models often narrate before calling. The prose becomes `content`, the
    /// call becomes structured output, and neither contaminates the other.
    #[test]
    fn prose_and_call_are_separated() {
        let (text, calls) = extract_tool_calls(
            "Let me look first.\n<tool_call>{\"name\":\"look\",\"arguments\":{}}</tool_call>",
        );
        assert_eq!(text, "Let me look first.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "look");
    }

    #[test]
    fn multiple_calls_are_indexed_in_order() {
        let (_, calls) = extract_tool_calls(
            r#"<tool_call>{"name":"stand","arguments":{}}</tool_call><tool_call>{"name":"walk","arguments":{"meters":1}}</tool_call>"#,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "stand");
        assert_eq!(calls[1].function.name, "walk");
    }

    /// A call cut off by `max_tokens` must not vanish. Silently dropping it
    /// would hand the client an empty assistant turn with no hint that the
    /// model tried to act — leave it visible as text instead.
    #[test]
    fn unterminated_call_survives_as_text() {
        let raw = r#"<tool_call>{"name":"walk","arguments":{"met"#;
        let (text, calls) = extract_tool_calls(raw);
        assert!(calls.is_empty());
        assert_eq!(text, raw);
    }

    /// A complete call that merely forgot its closing tag is still a call.
    ///
    /// Models routinely drop `</tool_call>` when generation was prefilled *into*
    /// the call — they never saw themselves open one, so they do not think to
    /// close it. The JSON is whole; only the tag is missing. Refusing it would
    /// turn a forced tool call into silent prose, which is the exact failure
    /// forcing was meant to prevent.
    #[test]
    fn complete_call_without_closing_tag_is_honored() {
        let (text, calls) =
            extract_tool_calls("<tool_call>\n{\"name\":\"look\",\"arguments\":{}}");

        assert_eq!(calls.len(), 1, "complete JSON must be honored: {text:?}");
        assert_eq!(calls[0].function.name, "look");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    /// Same for a block whose body isn't valid JSON.
    #[test]
    fn malformed_call_survives_as_text() {
        let raw = "<tool_call>not json at all</tool_call>";
        let (text, calls) = extract_tool_calls(raw);
        assert!(calls.is_empty());
        assert_eq!(text, raw);
    }

    #[test]
    fn plain_reply_has_no_calls() {
        let (text, calls) = extract_tool_calls("The room looks clear.");
        assert_eq!(text, "The room looks clear.");
        assert!(calls.is_empty());
    }

    /// The tool-loop round-trip: an assistant turn with `"content": null` plus
    /// `tool_calls`, followed by a `role: "tool"` result. OpenAI clients send
    /// exactly this on turn two, so failing to deserialize it would break every
    /// multi-step agent — while a non-optional `content` field would reject it.
    #[test]
    fn tool_loop_turn_two_deserializes() {
        let body = r#"{
            "messages": [
                {"role":"user","content":"walk 2m"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_1","type":"function",
                     "function":{"name":"walk","arguments":"{\"meters\":2}"}}
                ]},
                {"role":"tool","tool_call_id":"call_1","content":"ok, moved 2m"}
            ],
            "tools": [{"type":"function","function":{
                "name":"walk",
                "parameters":{"$schema":"http://json-schema.org/draft-07/schema#",
                              "type":"object","additionalProperties":false}
            }}],
            "tool_choice": "auto"
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert!(req.active_tools().is_some());

        let msgs = to_chat_messages(&req.messages);
        assert_eq!(msgs.len(), 3);

        // The wire carries `arguments` as a string; the template needs a value.
        let calls = msgs[1].tool_calls.as_ref().expect("tool calls preserved");
        assert_eq!(calls[0].name, "walk");
        assert_eq!(calls[0].arguments, json!({"meters": 2}));

        assert_eq!(msgs[2].role, sapient_tokenizers::ChatRole::Tool);
        assert_eq!(msgs[2].content, "ok, moved 2m");
    }

    /// `$schema` must never reach the model.
    ///
    /// Templates dump each tool verbatim into the tools preamble, and a draft-07
    /// dialect URL sitting in there is enough to knock a small tool-trained model
    /// off-distribution: Qwen2.5-1.5B stops emitting `<tool_call>` and answers in
    /// prose instead — it *claims* to have walked without walking. Zod v4 (and so
    /// the Vercel AI SDK) tags every schema with it, making this the default
    /// payload rather than an edge case.
    #[test]
    fn schema_dialect_key_is_stripped_from_tools() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[],"tools":[{"type":"function","function":{
                "name":"walk",
                "parameters":{
                    "$schema":"http://json-schema.org/draft-07/schema#",
                    "type":"object",
                    "properties":{"meters":{"type":"number","exclusiveMinimum":0}},
                    "required":["meters"],
                    "additionalProperties":false
                }}}]}"#,
        )
        .unwrap();

        let tools = req.active_tools().expect("tools active");
        let rendered = serde_json::to_string(&tools).unwrap();

        assert!(!rendered.contains("$schema"), "dialect key leaked: {rendered}");
        // Everything the model genuinely needs must survive the strip.
        assert!(rendered.contains(r#""name":"walk""#));
        assert!(rendered.contains(r#""required":["meters"]"#));
        assert!(rendered.contains(r#""exclusiveMinimum":0"#));
        assert!(rendered.contains(r#""additionalProperties":false"#));
    }

    /// `tool_choice` must be *binding*, not advisory.
    ///
    /// `auto` is a suggestion the model may decline. `required` and a named
    /// function are not — and a small model left to decide will happily answer
    /// "what do you see?" from imagination rather than calling the tool. The
    /// prefill removes the choice: the assistant turn is already open inside a
    /// `<tool_call>`, so prose is no longer a reachable continuation.
    #[test]
    fn tool_choice_required_forces_a_call() {
        let prefill_for = |choice: &str| -> Option<String> {
            let body = format!(
                r#"{{"messages":[],"tools":[{{"type":"function","function":{{"name":"look"}}}}],
                    "tool_choice":{choice}}}"#
            );
            serde_json::from_str::<ChatCompletionRequest>(&body)
                .unwrap()
                .tool_prefill()
        };

        assert_eq!(prefill_for(r#""required""#).as_deref(), Some("<tool_call>\n"));
        // A named function pins the name too, so the model cannot call another.
        assert_eq!(
            prefill_for(r#"{"type":"function","function":{"name":"look"}}"#).as_deref(),
            Some("<tool_call>\n{\"name\": \"look\", \"arguments\": {"),
        );
        // `auto` leaves the model free; `none` leaves it toolless. Neither forces.
        assert_eq!(prefill_for(r#""auto""#), None);
        assert_eq!(prefill_for(r#""none""#), None);
    }

    /// The forced-call prefill lives in the *prompt*, so the model never emits
    /// the opening tag. Un-stitched, its reply is a bare JSON fragment that
    /// parses as prose — the forced call would silently become no call at all.
    #[test]
    fn prefilled_reply_is_stitched_before_parsing() {
        let generated = r#"{"name":"look","arguments":{}}</tool_call>"#;

        let (_, unstitched) = extract_tool_calls(generated);
        assert!(unstitched.is_empty(), "a bare fragment must not parse alone");

        let (_, calls) = extract_tool_calls(&stitch(Some("<tool_call>\n"), generated));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "look");
    }

    /// `tool_choice: "none"` means "answer in prose" — the tools preamble must
    /// not be rendered at all.
    #[test]
    fn tool_choice_none_disables_tools() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[],"tools":[{"type":"function","function":{"name":"walk"}}],
                "tool_choice":"none"}"#,
        )
        .unwrap();
        assert!(req.active_tools().is_none());
    }

    /// An empty `tools: []` must behave as "no tools", not as an empty preamble.
    #[test]
    fn empty_tools_array_is_no_tools() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[],"tools":[]}"#).unwrap();
        assert!(req.active_tools().is_none());
    }

    // ── OpenAI content parsing (Phase 12.3) ─────────────────────────────────

    /// Plain-string content must keep deserializing (existing clients) and
    /// `Text` must serialize back to a plain string (response shape unchanged).
    #[test]
    fn content_plain_string_roundtrip() {
        let m: OAIMessage = serde_json::from_str(r#"{"role":"user","content":"hi"}"#).unwrap();
        assert_eq!(m.text(), "hi");
        assert!(m.image_urls().is_empty());
        let back = serde_json::to_string(&m).unwrap();
        assert_eq!(back, r#"{"role":"user","content":"hi"}"#);
    }

    /// OpenAI parts array: text + image_url (with an ignored `detail`).
    #[test]
    fn content_parts_with_image() {
        let m: OAIMessage = serde_json::from_str(
            r#"{"role":"user","content":[
                {"type":"text","text":"what is"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA","detail":"low"}},
                {"type":"text","text":"here?"}
            ]}"#,
        )
        .unwrap();
        assert_eq!(m.text(), "what is\nhere?");
        assert_eq!(m.image_urls(), vec!["data:image/png;base64,AAAA"]);
    }

    #[test]
    fn data_uri_decodes_base64() {
        // "SAPIENT" base64-encoded.
        let bytes = decode_image_data_uri("data:image/png;base64,U0FQSUVOVA==").unwrap();
        assert_eq!(bytes, b"SAPIENT");
    }

    #[test]
    fn data_uri_rejects_remote_urls_and_non_base64() {
        assert!(decode_image_data_uri("https://example.com/cat.png")
            .unwrap_err()
            .to_string()
            .contains("does not fetch remote"));
        assert!(decode_image_data_uri("data:image/png,rawpayload").is_err());
        assert!(decode_image_data_uri("data:image/png;base64,!!!not-base64!!!").is_err());
    }

    fn entry(id: &str, bytes: u64) -> Arc<CachedModel<()>> {
        Arc::new(CachedModel {
            model_id: id.to_string(),
            payload: (),
            bytes,
        })
    }

    #[test]
    fn lru_evicts_by_count() {
        let mut c = ModelCache::<()>::new(2, u64::MAX);
        c.insert(entry("a", 1));
        c.insert(entry("b", 1));
        assert_eq!(c.ids(), vec!["a", "b"]);
        c.insert(entry("c", 1)); // exceeds count → evict LRU "a"
        assert_eq!(c.ids(), vec!["b", "c"]);
        assert_eq!(c.mru_id().as_deref(), Some("c"));
    }

    #[test]
    fn touch_promotes_to_mru_and_changes_eviction_order() {
        let mut c = ModelCache::<()>::new(2, u64::MAX);
        c.insert(entry("a", 1));
        c.insert(entry("b", 1));
        assert!(c.touch("a").is_some()); // now order is [b, a]
        c.insert(entry("c", 1)); // evicts LRU "b", not "a"
        assert_eq!(c.ids(), vec!["a", "c"]);
    }

    #[test]
    fn evicts_by_byte_budget() {
        let mut c = ModelCache::<()>::new(10, 100);
        c.insert(entry("a", 60));
        c.insert(entry("b", 60)); // 120 > 100 → evict "a"
        assert_eq!(c.ids(), vec!["b"]);
        assert_eq!(c.used_bytes, 60);
    }

    #[test]
    fn never_evicts_the_just_inserted_even_if_over_budget() {
        let mut c = ModelCache::<()>::new(10, 10);
        c.insert(entry("big", 999)); // single oversized model stays resident
        assert_eq!(c.ids(), vec!["big"]);
    }

    #[test]
    fn reinserting_resident_model_is_a_touch_not_a_duplicate() {
        let mut c = ModelCache::<()>::new(3, u64::MAX);
        c.insert(entry("a", 5));
        c.insert(entry("b", 5));
        let evicted = c.insert(entry("a", 5)); // already present → touch
        assert!(evicted.is_empty());
        assert_eq!(c.ids(), vec!["b", "a"]); // a promoted to MRU
        assert_eq!(c.used_bytes, 10); // not double-counted
    }
}
