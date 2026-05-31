//! Speculative decoding: a small draft model generates K candidate tokens,
//! a larger target model verifies them in one forward pass, and accepted tokens
//! are kept. Expected speedup: 3–5× on generation-heavy workloads.
//!
//! # Algorithm
//!
//! ```text
//! loop:
//!   1. Draft K=5 tokens autoregressively from the draft model.
//!   2. Target model runs ONE forward pass over [last_context_token, draft_0..draft_{k-1}]
//!      and returns logits for all k positions.
//!   3. Accept/reject each draft token by comparing draft vs target probabilities.
//!      - If accepted: keep the token.
//!      - If rejected: resample from the adjusted distribution and break.
//!   4. If all K tokens were accepted, sample one bonus token free from the
//!      last target logit vector.
//!   5. Add accepted tokens to the context, emit them, repeat.
//! ```

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use sapient_hub::model_info::ArchType;
use sapient_models::ForwardEngine;
use sapient_tokenizers::chat::ChatMessage;
use sapient_tokenizers::tokenizer::SapientTokenizer;

use crate::pipeline::{GenerationConfig, LoadOptions, Pipeline};
use crate::sampler::Sampler;

// ── SpeculativePipeline ───────────────────────────────────────────────────────

/// A pipeline that uses a small **draft** model to accelerate generation by
/// a larger **target** model via speculative decoding.
///
/// Both the target and draft `Pipeline`s keep their forward engines loaded
/// (`Arc<Mutex<ForwardEngine>>`); every generation **reuses** those engines
/// inside a `spawn_blocking` closure rather than rebuilding+re-quantizing them
/// per request. This makes the pipeline reusable across many requests (e.g. in
/// `sapient serve`) with a one-time load cost, mirroring `Pipeline`.
pub struct SpeculativePipeline {
    target: Pipeline,
    draft: Pipeline,
    /// Speculation depth K: how many draft tokens to propose each round.
    k: usize,
}

impl SpeculativePipeline {
    /// Load both a target and a draft model.
    ///
    /// `k` is the speculation depth (typically 5).
    pub async fn new(target_model: &str, draft_model: &str, k: usize) -> Result<Self> {
        Self::new_with_opts(target_model, draft_model, k, LoadOptions::default()).await
    }

    /// Load both models with explicit load options (backend, mmap, generation).
    ///
    /// The same `opts` are applied to both target and draft so they share a
    /// backend (CPU/Metal) — used by `sapient serve` to honor `--backend`.
    pub async fn new_with_opts(
        target_model: &str,
        draft_model: &str,
        k: usize,
        opts: LoadOptions,
    ) -> Result<Self> {
        debug!("Loading target model: {target_model}");
        let target = Pipeline::from_pretrained_with_opts(target_model, opts.clone())
            .await
            .with_context(|| format!("failed to load target model '{target_model}'"))?;

        debug!("Loading draft model: {draft_model}");
        let draft = Pipeline::from_pretrained_with_opts(draft_model, opts)
            .await
            .with_context(|| format!("failed to load draft model '{draft_model}'"))?;

        // Speculative decoding requires the draft and target to share a vocabulary:
        // the draft proposes token IDs that the target scores with its own logits,
        // so a vocab mismatch indexes the wrong logits and yields garbage. Reject
        // an incompatible pair up front instead of silently producing junk.
        let target_vocab = target.tokenizer().vocab_size();
        let draft_vocab = draft.tokenizer().vocab_size();
        if target_vocab != draft_vocab {
            anyhow::bail!(
                "draft model '{draft_model}' (vocab {draft_vocab}) is incompatible with target \
                 '{target_model}' (vocab {target_vocab}): speculative decoding requires a draft \
                 from the same model family / tokenizer. Pick a matching draft, e.g. a smaller \
                 model in the target's family."
            );
        }

        Ok(Self { target, draft, k })
    }

    /// Load the target model; auto-select a small draft from the **same model
    /// family** as the target (speculative decoding needs a shared vocabulary).
    ///
    /// Draft candidates: `openhorizon/qwen2.5-0.5b-q4` (Qwen family),
    /// `openhorizon/smollm2-135m-q4` (SmolLM2 family). The one matching the
    /// target's family is preferred; otherwise a locally-cached candidate.
    pub async fn with_auto_draft(target_model: &str, k: usize) -> Result<Self> {
        Self::with_auto_draft_with_opts(target_model, k, LoadOptions::default()).await
    }

    /// Auto-select a draft model and load both with explicit load options.
    pub async fn with_auto_draft_with_opts(
        target_model: &str,
        k: usize,
        opts: LoadOptions,
    ) -> Result<Self> {
        let draft_model = select_auto_draft(target_model);
        debug!("Auto-selected draft model: {draft_model}");

        Self::new_with_opts(target_model, draft_model, k, opts).await
    }

    // ── Config helpers ──────────────────────────────────────────────────────

    /// All EOS token ids: the target's defaults plus any per-request override.
    fn merged_eos(&self, config: &GenerationConfig) -> Vec<u32> {
        let mut ids = self.target.eos_token_ids_pub();
        if let Some(e) = config.eos_token_id {
            if !ids.contains(&e) {
                ids.push(e);
            }
        }
        ids
    }

    /// Stop sequences: per-request stops plus the target's configured stops.
    fn merged_stops(&self, config: &GenerationConfig) -> Vec<String> {
        let mut stop = config.stop_sequences.clone();
        for s in self.target.stop_sequences() {
            if !stop.contains(s) {
                stop.push(s.clone());
            }
        }
        stop
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// Generate a completion for `prompt` using speculative decoding.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let config = self.target.config().clone();
        self.generate_with_config(prompt, &config).await
    }

    /// Generate with a custom generation config (per-request max_tokens/temp/stop).
    pub async fn generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        let target_tok = self.target.tokenizer();
        let input_ids = target_tok.encode(prompt)?;
        let output_ids = self.speculative_generate(input_ids, config).await?;
        let text = target_tok.decode(&output_ids, true)?;
        Ok(text)
    }

    /// Chat with a message history using speculative decoding.
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = self.target.format_chat_prompt(messages)?;
        self.generate(&prompt).await
    }

    /// Chat with a custom generation config (used by `sapient serve`).
    pub async fn chat_with_config(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> Result<String> {
        let prompt = self.target.format_chat_prompt(messages)?;
        self.generate_with_config(&prompt, config).await
    }

    /// Stream tokens from speculative decoding.
    /// Tokens are emitted in bursts (one burst per speculation round).
    pub async fn generate_stream(&self, prompt: &str) -> ReceiverStream<String> {
        let config = self.target.config().clone();
        self.generate_stream_with_config(prompt, &config).await
    }

    /// Stream tokens with a custom generation config (per-request settings).
    pub async fn generate_stream_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> ReceiverStream<String> {
        let (tx, rx) = mpsc::channel(64);
        let target_tok = self.target.tokenizer_arc();
        let input_ids = self.target.tokenizer().encode(prompt).unwrap_or_default();

        // Reuse the already-loaded engines instead of rebuilding them — re-loading
        // and re-quantizing both models per request previously dominated TTFT.
        let target_engine = self.target.engine_arc();
        let draft_engine = self.draft.engine_arc();

        let eos_ids = self.merged_eos(config);
        let stop_seqs = self.merged_stops(config);
        let max_new = config.max_new_tokens;
        let strategy = config.strategy.clone();
        let k = self.k;

        tokio::task::spawn_blocking(move || {
            let mut target_engine = match target_engine.lock() {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(format!("Error: target engine lock poisoned: {e}"));
                    return;
                }
            };
            let mut draft_engine = match draft_engine.lock() {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(format!("Error: draft engine lock poisoned: {e}"));
                    return;
                }
            };

            // Start each request from a clean KV cache (engines are reused).
            target_engine.reset_cache();
            draft_engine.reset_cache();

            let mut sampler = Sampler::new(strategy);
            let mut state = match spec_prefill(&mut target_engine, &mut draft_engine, input_ids) {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.blocking_send(format!("Prefill error: {e}"));
                    return;
                }
            };
            let mut generated: Vec<u32> = Vec::new();
            let mut emitted = 0usize;
            let mut total_generated = 0usize;

            'outer: loop {
                if total_generated >= max_new {
                    break;
                }

                let round = match spec_round(
                    &mut target_engine,
                    &mut draft_engine,
                    &mut state,
                    &mut sampler,
                    k,
                    &eos_ids,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.blocking_send(format!("Speculative error: {e}"));
                        break;
                    }
                };
                let SpecRound { committed, hit_eos } = round;
                let made_progress = !committed.is_empty();

                for tok in committed {
                    generated.push(tok);
                    total_generated += 1;

                    if eos_ids.contains(&tok) {
                        // Flush held-back text and stop.
                        if let Ok(text) = target_tok.decode(&generated, true) {
                            if text.len() > emitted {
                                let _ = tx.blocking_send(text[emitted..].to_string());
                            }
                        }
                        break 'outer;
                    }

                    // Emit decoded text up to a safe (stop-marker-free) boundary.
                    if let Ok(text) = target_tok.decode(&generated, true) {
                        if let Some(idx) = earliest_stop(&text, &stop_seqs) {
                            if idx > emitted {
                                let _ = tx.blocking_send(text[emitted..idx].to_string());
                            }
                            break 'outer;
                        }
                        let safe = safe_emit_end(&text, &stop_seqs);
                        if safe > emitted {
                            if tx.blocking_send(text[emitted..safe].to_string()).is_err() {
                                break 'outer;
                            }
                            emitted = safe;
                        }
                    }

                    if total_generated >= max_new {
                        break 'outer;
                    }
                }

                if hit_eos || !made_progress {
                    break;
                }
            }

            // Flush any remaining text not yet emitted.
            if let Ok(text) = target_tok.decode(&generated, true) {
                if text.len() > emitted {
                    let _ = tx.blocking_send(text[emitted..].to_string());
                }
            }
        });

        ReceiverStream::new(rx)
    }

    /// Stream chat reply using speculative decoding.
    pub async fn chat_stream(&self, messages: &[ChatMessage]) -> ReceiverStream<String> {
        match self.target.format_chat_prompt(messages) {
            Ok(prompt) => self.generate_stream(&prompt).await,
            Err(e) => {
                let (tx, rx) = mpsc::channel(1);
                let _ = tx.try_send(format!("Error: {e}"));
                ReceiverStream::new(rx)
            }
        }
    }

    /// Stream chat reply with a custom generation config (used by `sapient serve`).
    pub async fn chat_stream_with_config(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
    ) -> ReceiverStream<String> {
        match self.target.format_chat_prompt(messages) {
            Ok(prompt) => self.generate_stream_with_config(&prompt, config).await,
            Err(e) => {
                let (tx, rx) = mpsc::channel(1);
                let _ = tx.try_send(format!("Error: {e}"));
                ReceiverStream::new(rx)
            }
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// The target model's tokenizer (used to encode/decode at the API edge).
    pub fn tokenizer(&self) -> &SapientTokenizer {
        self.target.tokenizer()
    }

    /// The target model's architecture.
    pub fn arch(&self) -> &ArchType {
        self.target.arch()
    }

    /// True when the target's weights are memory-mapped from disk.
    pub fn is_mmap(&self) -> bool {
        self.target.is_mmap()
    }

    /// The target's active generation config.
    pub fn config(&self) -> &GenerationConfig {
        self.target.config()
    }

    /// Render a chat prompt string for a message history (target template).
    pub fn format_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String> {
        self.target.format_chat_prompt(messages)
    }

    // ── Core speculative loop (blocking) ──────────────────────────────────────

    async fn speculative_generate(
        &self,
        input_ids: Vec<u32>,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        // Reuse the already-loaded engines (one-time load), like `Pipeline`.
        let target_engine = self.target.engine_arc();
        let draft_engine = self.draft.engine_arc();

        let eos_ids = self.merged_eos(config);
        let max_new = config.max_new_tokens;
        let strategy = config.strategy.clone();
        let k = self.k;

        tokio::task::spawn_blocking(move || {
            let mut target_engine = target_engine
                .lock()
                .map_err(|e| anyhow::anyhow!("target engine lock poisoned: {e}"))?;
            let mut draft_engine = draft_engine
                .lock()
                .map_err(|e| anyhow::anyhow!("draft engine lock poisoned: {e}"))?;

            // Start from a clean KV cache (engines are reused across requests).
            target_engine.reset_cache();
            draft_engine.reset_cache();

            let mut sampler = Sampler::new(strategy);
            let mut state = spec_prefill(&mut target_engine, &mut draft_engine, input_ids)?;
            let mut generated: Vec<u32> = Vec::new();
            let mut total_generated = 0usize;

            loop {
                if total_generated >= max_new {
                    break;
                }

                let SpecRound { committed, hit_eos } = spec_round(
                    &mut target_engine,
                    &mut draft_engine,
                    &mut state,
                    &mut sampler,
                    k,
                    &eos_ids,
                )?;
                if committed.is_empty() {
                    break;
                }

                let mut stop = hit_eos;
                for tok in committed {
                    generated.push(tok);
                    total_generated += 1;
                    if eos_ids.contains(&tok) {
                        stop = true;
                        break;
                    }
                    if total_generated >= max_new {
                        stop = true;
                        break;
                    }
                }

                if stop {
                    break;
                }
            }

            Ok(generated)
        })
        .await
        .context("speculative_generate task panicked")?
    }
}

// ── Core speculative loop (cache-aware) ───────────────────────────────────────

/// State carried between speculation rounds. The invariant after each round:
/// both the target and draft KV caches hold **exactly** the committed tokens
/// (`committed_len` positions), and `*_next` are each model's logits predicting
/// the next (not-yet-generated) position.
struct SpecState {
    all_tokens: Vec<u32>,
    /// Target logits predicting the token at position `committed_len`.
    target_next: Vec<f32>,
    /// Draft logits predicting the token at position `committed_len`.
    draft_next: Vec<f32>,
    committed_len: usize,
}

/// Result of one speculation round.
struct SpecRound {
    /// Tokens committed this round (1 reject-corrected, or up to `k`+1 on full accept).
    committed: Vec<u32>,
    /// Whether an EOS token was committed (generation should stop).
    hit_eos: bool,
}

/// Prefill both models on the prompt, leaving each KV cache holding the prompt
/// and capturing the next-token logits.
fn spec_prefill(
    target: &mut ForwardEngine,
    draft: &mut ForwardEngine,
    prompt: Vec<u32>,
) -> Result<SpecState> {
    let target_next = target.forward_logits(&prompt, true)?;
    let draft_next = draft.forward_logits(&prompt, true)?;
    let committed_len = prompt.len();
    Ok(SpecState {
        all_tokens: prompt,
        target_next,
        draft_next,
        committed_len,
    })
}

/// Run one speculation round: draft up to `k` tokens, verify them against the
/// target in a single cache-appending forward pass, accept/reject with rejection
/// sampling, then reconcile both KV caches to exactly the committed context.
///
/// The target's distribution for `drafts[i]` is `target_next` (for `i == 0`) or
/// the target logits *after* `drafts[i-1]` — both computed **with prompt context
/// in the KV cache**. Rejected speculative tokens are rolled back with
/// `truncate_cache`, so after this returns the caches match the committed tokens.
fn spec_round(
    target: &mut ForwardEngine,
    draft: &mut ForwardEngine,
    state: &mut SpecState,
    sampler: &mut Sampler,
    k: usize,
    eos_ids: &[u32],
) -> Result<SpecRound> {
    let l = state.committed_len;

    // ── 1. Draft up to k tokens, extending the draft KV cache. ──────────────
    let mut drafts: Vec<u32> = Vec::with_capacity(k);
    let mut dprobs: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut dcur = state.draft_next.clone();
    for _ in 0..k {
        let probs = softmax(&dcur);
        let d = sampler.sample(&dcur, &state.all_tokens).unwrap_or(0);
        drafts.push(d);
        dprobs.push(probs);
        if eos_ids.contains(&d) {
            break;
        }
        dcur = draft.forward_logits(&[d], true)?;
    }
    let m = drafts.len();
    if m == 0 {
        return Ok(SpecRound {
            committed: Vec::new(),
            hit_eos: false,
        });
    }

    // ── 2. Target verifies all m drafts in one cache-appending pass. ────────
    // t_all[i] = target prediction for the token AFTER drafts[i] (position l+i+1).
    let t_all = target.forward_all_logits_cached(&drafts)?;

    // ── 3. Accept/reject each draft via rejection sampling. ─────────────────
    let mut accepted: Vec<u32> = Vec::new();
    let mut rejected = false;
    let mut hit_eos = false;
    for i in 0..m {
        let tlogits = if i == 0 {
            &state.target_next
        } else {
            &t_all[i - 1]
        };
        let tprobs = softmax(tlogits);
        let d = drafts[i];
        let dp = dprobs[i][d as usize].max(1e-9);
        let tp = tprobs[d as usize];
        if random_f32(sampler) < (tp / dp).min(1.0) {
            accepted.push(d);
            if eos_ids.contains(&d) {
                hit_eos = true;
                rejected = true; // stop after EOS
                break;
            }
        } else {
            // Resample from the adjusted distribution max(0, target - draft).
            let corrected = sample_from_probs(&adjusted_dist(&tprobs, &dprobs[i]), sampler);
            accepted.push(corrected);
            rejected = true;
            if eos_ids.contains(&corrected) {
                hit_eos = true;
            }
            break;
        }
    }

    // ── 4. All drafts accepted → one free bonus token from the last logit. ──
    if !rejected {
        let bonus = sampler
            .sample(&t_all[m - 1], &state.all_tokens)
            .unwrap_or(0);
        accepted.push(bonus);
        if eos_ids.contains(&bonus) {
            hit_eos = true;
        }
    }

    let num_acc = accepted.len();

    // ── 5. Reconcile caches to the committed context (unless we're stopping). ─
    if !hit_eos {
        if rejected {
            // accepted = drafts[0..num_acc-1] (already in target cache) + corrected.
            // Drop speculative drafts past the accepted prefix, then commit corrected.
            target.truncate_cache(l + (num_acc - 1));
        }
        // For full-accept the m drafts are already cached; either way append the
        // final committed token (corrected or bonus) to refresh target_next.
        state.target_next = target.forward_logits(&[accepted[num_acc - 1]], true)?;

        // Re-sync the draft cache to exactly the committed tokens.
        draft.truncate_cache(l);
        let mut dl = state.draft_next.clone();
        for &tok in &accepted {
            dl = draft.forward_logits(&[tok], true)?;
        }
        state.draft_next = dl;
    }

    state.all_tokens.extend_from_slice(&accepted);
    state.committed_len = l + num_acc;
    Ok(SpecRound {
        committed: accepted,
        hit_eos,
    })
}

// ── Math helpers ──────────────────────────────────────────────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        out.iter_mut().for_each(|x| *x /= sum);
    }
    out
}

/// Adjusted distribution for rejection-sampling correction:
/// `max(0, target_p - draft_p)` re-normalised.
fn adjusted_dist(target: &[f32], draft: &[f32]) -> Vec<f32> {
    let mut adj: Vec<f32> = target
        .iter()
        .zip(draft.iter())
        .map(|(&t, &d)| (t - d).max(0.0))
        .collect();
    let sum: f32 = adj.iter().sum();
    if sum > 1e-9 {
        adj.iter_mut().for_each(|x| *x /= sum);
    } else {
        // Degenerate case: fall back to uniform over highest-target token.
        let best = target
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        adj.fill(0.0);
        if best < adj.len() {
            adj[best] = 1.0;
        }
    }
    adj
}

/// Sample a token index from a probability vector using the sampler's RNG.
fn sample_from_probs(probs: &[f32], sampler: &mut Sampler) -> u32 {
    // Re-use the sampler's internal RNG via a Temperature(1.0) logit trick:
    // convert probs back to logits and sample. Simpler: use a direct categorical.
    let r = random_f32(sampler);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i as u32;
        }
    }
    (probs.len().saturating_sub(1)) as u32
}

/// Pull one uniform [0,1) float from the sampler's RNG without going through
/// the full `sample()` path (which requires logits).
fn random_f32(sampler: &mut Sampler) -> f32 {
    // We use a dummy uniform logit vector of length 2 and check which slot
    // was selected — that gives us a 50/50 coin flip but not a uniform float.
    // Instead, rely on the sampler being Temperature-based to read r directly.
    // Since Sampler::random_f32 is private, we mimic its xorshift here.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345);
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut x = seed.wrapping_add(c.wrapping_mul(6364136223846793005));
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    // Map to [0, 1)
    (x >> 11) as f32 / (1u64 << 53) as f32
}

// ── Stop-sequence helpers (mirrors pipeline.rs) ───────────────────────────────

fn earliest_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

fn safe_emit_end(text: &str, stops: &[String]) -> usize {
    let mut hold = 0usize;
    for s in stops {
        let max_k = s.len().min(text.len());
        for k in (1..max_k).rev() {
            if !s.is_char_boundary(k) {
                continue;
            }
            if text.ends_with(&s[..k]) {
                hold = hold.max(k);
                break;
            }
        }
    }
    text.len() - hold
}

// ── Draft auto-selection ──────────────────────────────────────────────────────

/// Auto-select a draft model for `target_model`. Speculative decoding requires
/// a shared vocabulary, so prefer a draft from the **same family** as the target
/// (matched by model-id keyword); fall back to a locally-cached candidate.
fn select_auto_draft(target_model: &str) -> &'static str {
    let id = target_model.to_ascii_lowercase();
    // (family keyword, family draft). Order = preference when no family matches.
    let candidates: [(&str, &'static str); 2] = [
        ("qwen", "openhorizon/qwen2.5-0.5b-q4"),
        ("smol", "openhorizon/smollm2-135m-q4"),
    ];
    // Family match first (skip the draft if the target *is* that draft).
    for (kw, draft) in candidates {
        if id.contains(kw) && id != *draft {
            return draft;
        }
    }
    // No family match — use a locally-cached candidate, else the first.
    let ids: [&'static str; 2] = [candidates[1].1, candidates[0].1];
    select_cached_draft(&ids)
}

/// Select the first draft candidate that appears to be locally cached.
/// Falls back to the first candidate if none are found.
fn select_cached_draft(candidates: &[&'static str]) -> &'static str {
    // Check the HuggingFace Hub cache directory for each candidate.
    let hub_cache = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
            format!("{home}/.cache/huggingface/hub")
        });

    for &candidate in candidates {
        // The hub stores models under hub/models--{owner}--{name}/
        let dir_name = candidate.replace('/', "--");
        let path = format!("{hub_cache}/models--{dir_name}");
        if std::path::Path::new(&path).exists() {
            return candidate;
        }
    }
    candidates[0]
}
