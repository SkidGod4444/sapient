//! Stable FFI surface for embedding SAPIENT in host applications.
//!
//! This crate is the Phase-11 (mobile & SDKs) boundary layer: a small,
//! blocking, object-oriented API exported through [UniFFI] so the same Rust
//! code surfaces as idiomatic Swift (iOS/macOS) and Kotlin (Android/JVM).
//! Node.js / React Native go through the first-party TypeScript SDK
//! (`sdks/typescript`), which speaks to `sapient serve` today and will bind
//! this crate natively (napi/JSI) next.
//!
//! Design rules (see `docs/MOBILE.md` for the full story):
//! - **Blocking API, internal runtime.** Model loading and generation are
//!   synchronous calls; a private multi-thread tokio runtime drives the async
//!   `Pipeline` internals. Hosts call from a background thread/queue —
//!   never the UI thread.
//! - **Futures run on runtime workers, not the caller.** Everything async is
//!   `tokio::spawn`-ed and joined (`run_async`) because `Pipeline` internals
//!   use `block_in_place`, which panics on a thread that isn't a runtime
//!   worker.
//! - **Streaming = foreign callback.** `TokenListener::on_token` returns
//!   `bool`; returning `false` drops the token receiver, which makes the
//!   engine's next `blocking_send` fail and halts generation — cancellation
//!   without any new engine API.
//! - **Sessions own the conversation.** `LlmSession` keeps the chat history
//!   and enables the prefix cache, so multi-turn chats skip re-prefilling
//!   history exactly like `sapient serve` does.
//!
//! [UniFFI]: https://mozilla.github.io/uniffi-rs/

uniffi::setup_scaffolding!();

use std::sync::{Arc, Mutex, OnceLock};

use sapient_generate::{GenerationConfig, LoadOptions, Pipeline, SamplingStrategy};
use sapient_tokenizers::chat::ChatMessage;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors crossing the FFI boundary. The `reason` strings carry the
/// underlying `anyhow` chain so host apps can log something actionable.
///
/// The field is deliberately NOT named `message`: UniFFI maps error enums to
/// Kotlin exception classes, and a `message` field collides with
/// `Throwable.message` ("hides member of supertype" — the generated Kotlin
/// doesn't compile; found by the Android sample app).
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum SapientError {
    #[error("model load failed: {reason}")]
    Load { reason: String },
    #[error("generation failed: {reason}")]
    Generation { reason: String },
    #[error("invalid argument: {reason}")]
    InvalidArgument { reason: String },
    #[error("internal error: {reason}")]
    Internal { reason: String },
}

// ── Runtime plumbing ──────────────────────────────────────────────────────────

/// Private tokio runtime driving the async `Pipeline` internals. Two workers
/// are plenty: inference itself runs on tokio's blocking pool
/// (`spawn_blocking`), not on these workers.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("sapient-ffi")
            .enable_all()
            .build()
            .expect("failed to build sapient-ffi tokio runtime")
    })
}

/// Run a future to completion on the runtime's workers and block the calling
/// (foreign) thread on the result. The future MUST run on a worker — not via
/// `block_on` on the caller — because `Pipeline` uses `block_in_place`
/// internally, which panics outside a multi-thread-runtime worker.
fn run_async<T, F>(fut: F) -> anyhow::Result<T>
where
    F: std::future::Future<Output = anyhow::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let handle = runtime().spawn(fut);
    runtime()
        .block_on(handle)
        .map_err(|e| anyhow::anyhow!("sapient-ffi runtime join error: {e}"))?
}

// ── Free functions ────────────────────────────────────────────────────────────

/// The SAPIENT engine version compiled into this library.
#[uniffi::export]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// One row of the curated model catalog.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelEntry {
    /// Canonical alias accepted by `LlmSession::load` (e.g. `qwen2.5-0.5b`).
    pub alias: String,
    /// HuggingFace repository the alias resolves to.
    pub repo_id: String,
    /// Architecture family, for display.
    pub family: String,
    /// Approximate parameter count, for display (e.g. `0.5B`).
    pub params: String,
    /// Capability bucket: `chat`, `speech-to-text`, `text-to-speech`, `vision`.
    pub category: String,
    /// Whether the repo is gated (needs an accepted license / HF token).
    pub gated: bool,
}

fn category_slug(m: &sapient_hub::registry::SupportedModel) -> &'static str {
    use sapient_hub::registry::ModelCategory as C;
    match m.category() {
        C::Chat => "chat",
        C::SpeechToText => "speech-to-text",
        C::TextToSpeech => "text-to-speech",
        C::Vision => "vision",
    }
}

/// The curated model catalog — every alias `LlmSession::load` accepts.
#[uniffi::export]
pub fn list_models() -> Vec<ModelEntry> {
    sapient_hub::registry::catalog()
        .iter()
        .map(|m| ModelEntry {
            alias: m.alias.to_string(),
            repo_id: m.repo_id.to_string(),
            family: m.family.to_string(),
            params: m.params.to_string(),
            category: category_slug(m).to_string(),
            gated: m.gated,
        })
        .collect()
}

/// Resolve a model alias (with fuzzy matching) to its HuggingFace repo id.
/// Errors with the full catalog listing for unknown names.
#[uniffi::export]
pub fn resolve_alias(name: String) -> Result<String, SapientError> {
    sapient_hub::registry::resolve_model_alias(&name).map_err(|e| SapientError::InvalidArgument {
        reason: e.to_string(),
    })
}

// ── Generation options ────────────────────────────────────────────────────────

/// Options for creating an [`LlmSession`]. All fields have defaults, so
/// foreign callers can construct this with only the fields they care about.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerationOptions {
    /// Hard cap on new tokens per reply.
    #[uniffi(default = 512)]
    pub max_tokens: u32,
    /// Sampling temperature. Leaving every sampling field unset selects
    /// greedy (deterministic) decoding.
    #[uniffi(default = None)]
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold (0–1).
    #[uniffi(default = None)]
    pub top_p: Option<f32>,
    /// Top-k cutoff. `0`/unset disables the filter.
    #[uniffi(default = None)]
    pub top_k: Option<u32>,
    /// Repetition penalty (1.0 = off).
    #[uniffi(default = None)]
    pub repetition_penalty: Option<f32>,
    /// Optional system prompt seeded at the start of the conversation.
    #[uniffi(default = None)]
    pub system_prompt: Option<String>,
    /// Backend override: `auto` (default), `cpu`, `metal`, `wgpu`. Mobile
    /// static libs are CPU-only today, so `auto` resolves to CPU there.
    #[uniffi(default = None)]
    pub backend: Option<String>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            system_prompt: None,
            backend: None,
        }
    }
}

impl GenerationOptions {
    /// Map to the engine's sampling strategy. All sampling fields unset →
    /// greedy; otherwise the combined sampler with engine-neutral defaults
    /// (`top_k 0` and `top_p 1.0` disable those filters; `rp 1.0` is a no-op).
    fn strategy(&self) -> SamplingStrategy {
        if self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.repetition_penalty.is_none()
        {
            return SamplingStrategy::Greedy;
        }
        SamplingStrategy::Combined {
            top_k: self.top_k.unwrap_or(0) as usize,
            top_p: self.top_p.unwrap_or(1.0),
            temperature: self.temperature.unwrap_or(0.7),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
        }
    }

    fn generation_config(&self) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: self.max_tokens as usize,
            strategy: self.strategy(),
            ..GenerationConfig::default()
        }
    }

    fn backend_kind(&self) -> Result<sapient_generate::GenerationBackend, SapientError> {
        use sapient_generate::GenerationBackend as B;
        match self.backend.as_deref() {
            None | Some("auto") => Ok(B::Auto),
            Some("cpu") => Ok(B::Cpu),
            Some("metal") => Ok(B::Metal),
            Some("wgpu") => Ok(B::Wgpu),
            Some(other) => Err(SapientError::InvalidArgument {
                reason: format!("unknown backend '{other}' (expected auto|cpu|metal|wgpu)"),
            }),
        }
    }
}

// ── Streaming callback ────────────────────────────────────────────────────────

/// Foreign-implemented token sink for streaming replies. Return `true` to
/// keep generating, `false` to cancel — cancellation drops the internal
/// receiver, which halts the engine at its next token emit.
#[uniffi::export(with_foreign)]
pub trait TokenListener: Send + Sync {
    fn on_token(&self, token: String) -> bool;
}

// ── Chat transcript ───────────────────────────────────────────────────────────

/// One message of the session transcript, for host-side display/persistence.
#[derive(Debug, Clone, uniffi::Record)]
pub struct Message {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    pub content: String,
}

// ── LlmSession ────────────────────────────────────────────────────────────────

/// A loaded chat model plus its conversation state. Thread-safe; generation
/// calls on the same session serialize on the engine's internal lock.
#[derive(uniffi::Object)]
pub struct LlmSession {
    pipeline: Arc<Pipeline>,
    history: Mutex<Vec<ChatMessage>>,
    system_prompt: Option<String>,
    config: GenerationConfig,
    model: String,
}

impl LlmSession {
    fn seeded_history(system_prompt: &Option<String>) -> Vec<ChatMessage> {
        match system_prompt {
            Some(s) if !s.is_empty() => vec![ChatMessage::system(s.clone())],
            _ => Vec::new(),
        }
    }

    /// History snapshot + the new user turn, without mutating state — history
    /// is only committed after a turn succeeds.
    fn messages_with(&self, user_message: &str) -> Vec<ChatMessage> {
        let mut msgs = self
            .history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        msgs.push(ChatMessage::user(user_message));
        msgs
    }

    fn commit_turn(&self, user_message: &str, reply: &str) {
        let mut h = self.history.lock().unwrap_or_else(|e| e.into_inner());
        h.push(ChatMessage::user(user_message));
        h.push(ChatMessage::assistant(reply));
    }
}

#[uniffi::export]
impl LlmSession {
    /// Download (if needed) and load a model from the curated catalog, then
    /// hold it resident. Blocking — first call downloads weights; run this on
    /// a background thread and surface progress in the host UI.
    #[uniffi::constructor]
    pub fn load(model: String, options: GenerationOptions) -> Result<Arc<Self>, SapientError> {
        let config = options.generation_config();
        let load_opts = LoadOptions {
            generation: config.clone(),
            backend: options.backend_kind()?,
            ..LoadOptions::default()
        };
        let alias = model.clone();
        let mut pipeline =
            run_async(async move { Pipeline::from_pretrained_with_opts(&alias, load_opts).await })
                .map_err(|e| SapientError::Load {
                    reason: format!("{e:#}"),
                })?;
        // Multi-turn chats re-send the whole history; the prefix cache keeps
        // the KV for the shared prefix so only the new turn is prefilled.
        pipeline.enable_prefix_cache();
        Ok(Arc::new(Self {
            pipeline: Arc::new(pipeline),
            history: Mutex::new(Self::seeded_history(&options.system_prompt)),
            system_prompt: options.system_prompt,
            config,
            model,
        }))
    }

    /// One blocking chat turn: appends the user message, generates the full
    /// reply, commits both to the session history.
    pub fn chat(&self, user_message: String) -> Result<String, SapientError> {
        let messages = self.messages_with(&user_message);
        let pipeline = Arc::clone(&self.pipeline);
        let config = self.config.clone();
        let reply = run_async(async move { pipeline.chat_with_config(&messages, &config).await })
            .map_err(|e| SapientError::Generation {
            reason: format!("{e:#}"),
        })?;
        self.commit_turn(&user_message, &reply);
        Ok(reply)
    }

    /// Streaming chat turn: `listener.on_token` receives each text fragment
    /// as it decodes; returning `false` cancels generation. Returns the full
    /// (possibly cancelled-partial) reply, which is committed to history —
    /// on cancel that is intentional, so the history matches what the user
    /// saw and the prefix cache stays aligned with the engine's KV state.
    ///
    /// Error semantics: `Err` covers failures *starting* the stream (prompt
    /// formatting, runtime). A generation failure *mid-stream* follows the
    /// engine's in-band convention — the pipeline emits a final
    /// `Error: …` text fragment and ends the stream (exactly what
    /// `sapient serve` SSE clients see) — because the token channel carries
    /// only `String`, with no sideband to distinguish error-close from
    /// normal close. Promoting that to a typed error needs a
    /// `Result`-carrying stream in `sapient-generate` (shared with
    /// serve/CLI) and is tracked as a Phase 11 follow-up rung.
    pub fn chat_stream(
        &self,
        user_message: String,
        listener: Arc<dyn TokenListener>,
    ) -> Result<String, SapientError> {
        let messages = self.messages_with(&user_message);
        let pipeline = Arc::clone(&self.pipeline);
        let config = self.config.clone();
        let stream =
            run_async(
                async move { Ok(pipeline.chat_stream_with_config(&messages, &config).await) },
            )
            .map_err(|e| SapientError::Generation {
                reason: format!("{e:#}"),
            })?;
        // Consume on the caller's thread — blocking_recv must not run on a
        // runtime worker. Dropping `rx` early is the cancellation signal.
        let mut rx = stream.into_inner();
        let mut reply = String::new();
        while let Some(token) = rx.blocking_recv() {
            reply.push_str(&token);
            if !listener.on_token(token) {
                break;
            }
        }
        drop(rx);
        self.commit_turn(&user_message, &reply);
        Ok(reply)
    }

    /// Clear the conversation (keeps the model loaded). The system prompt
    /// given at load time is re-seeded.
    pub fn reset(&self) {
        *self.history.lock().unwrap_or_else(|e| e.into_inner()) =
            Self::seeded_history(&self.system_prompt);
        self.pipeline.reset_cache();
    }

    /// The conversation so far (excluding any in-flight turn).
    pub fn transcript(&self) -> Vec<Message> {
        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|m| Message {
                role: m.role.to_string(),
                content: m.content.clone(),
            })
            .collect()
    }

    /// The model alias this session was created with.
    pub fn model(&self) -> String {
        self.model.clone()
    }

    /// Human-readable resolved backend (e.g. `CPU`, `Metal GPU`).
    pub fn backend_label(&self) -> String {
        self.pipeline.backend_display_label()
    }

    /// Whether the weights are memory-mapped (RSS ≈ working set, not file size).
    pub fn is_mmap(&self) -> bool {
        self.pipeline.is_mmap()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_crate() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn catalog_is_exposed_with_categories() {
        let models = list_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.category == "chat"));
        assert!(models.iter().any(|m| m.category == "speech-to-text"));
        assert!(models.iter().any(|m| m.category == "text-to-speech"));
        for m in &models {
            assert!(!m.alias.is_empty());
            assert!(!m.repo_id.is_empty());
        }
    }

    #[test]
    fn resolve_alias_roundtrips_catalog_and_rejects_garbage() {
        let first = &list_models()[0];
        assert_eq!(resolve_alias(first.alias.clone()).unwrap(), first.repo_id);
        assert!(matches!(
            resolve_alias("definitely-not-a-model-xyz".into()),
            Err(SapientError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn default_options_map_to_greedy() {
        let opts = GenerationOptions::default();
        assert!(matches!(opts.strategy(), SamplingStrategy::Greedy));
        let cfg = opts.generation_config();
        assert_eq!(cfg.max_new_tokens, 512);
    }

    #[test]
    fn sampling_options_map_to_combined_with_neutral_defaults() {
        let opts = GenerationOptions {
            temperature: Some(0.6),
            ..GenerationOptions::default()
        };
        match opts.strategy() {
            SamplingStrategy::Combined {
                top_k,
                top_p,
                temperature,
                repetition_penalty,
            } => {
                assert_eq!(top_k, 0); // disabled
                assert_eq!(top_p, 1.0); // disabled
                assert_eq!(temperature, 0.6);
                assert_eq!(repetition_penalty, 1.0); // no-op
            }
            other => panic!("expected Combined, got {other:?}"),
        }
    }

    #[test]
    fn backend_strings_parse() {
        use sapient_generate::GenerationBackend as B;
        let mk = |b: Option<&str>| GenerationOptions {
            backend: b.map(str::to_string),
            ..GenerationOptions::default()
        };
        assert!(matches!(mk(None).backend_kind(), Ok(B::Auto)));
        assert!(matches!(mk(Some("cpu")).backend_kind(), Ok(B::Cpu)));
        assert!(matches!(mk(Some("metal")).backend_kind(), Ok(B::Metal)));
        assert!(matches!(mk(Some("wgpu")).backend_kind(), Ok(B::Wgpu)));
        assert!(mk(Some("cuda")).backend_kind().is_err());
    }

    /// Golden end-to-end gate: downloads SmolLM2-135M (~100 MB GGUF), loads a
    /// real session, and runs a greedy turn plus a streamed turn. Run with:
    /// `cargo test -p sapient-ffi --release -- --ignored`
    #[test]
    #[ignore = "downloads a model — network + disk"]
    fn e2e_chat_and_stream_smollm2() {
        let session = LlmSession::load(
            "smollm2-135m-q4".into(),
            GenerationOptions {
                max_tokens: 32,
                ..GenerationOptions::default()
            },
        )
        .expect("load smollm2-135m-q4");
        let reply = session.chat("Reply with one short sentence: hello!".into());
        let reply = reply.expect("chat turn");
        assert!(!reply.trim().is_empty());

        struct Collect(Mutex<Vec<String>>);
        impl TokenListener for Collect {
            fn on_token(&self, token: String) -> bool {
                self.0.lock().unwrap().push(token);
                true
            }
        }
        let sink = Arc::new(Collect(Mutex::new(Vec::new())));
        let full = session
            .chat_stream("And another one?".into(), sink.clone())
            .expect("stream turn");
        let joined: String = sink.0.lock().unwrap().concat();
        assert_eq!(joined, full);
        assert_eq!(session.transcript().len(), 4);
    }
}
