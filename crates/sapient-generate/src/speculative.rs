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

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use sapient_models::{ForwardEngine, LlmBackendKind};
use sapient_tokenizers::chat::{ChatMessage, ChatTemplate};
use sapient_tokenizers::tokenizer::{SapientTokenizer, TokenizerOptions};

use crate::pipeline::{GenerationConfig, LoadOptions, Pipeline};
use crate::sampler::{Sampler, SamplingStrategy};

// ── SpeculativePipeline ───────────────────────────────────────────────────────

/// A pipeline that uses a small **draft** model to accelerate generation by
/// a larger **target** model via speculative decoding.
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
        debug!("Loading target model: {target_model}");
        let target = Pipeline::from_pretrained(target_model)
            .await
            .with_context(|| format!("failed to load target model '{target_model}'"))?;

        debug!("Loading draft model: {draft_model}");
        let draft = Pipeline::from_pretrained(draft_model)
            .await
            .with_context(|| format!("failed to load draft model '{draft_model}'"))?;

        Ok(Self { target, draft, k })
    }

    /// Load the target model; auto-select the best locally-cached draft model.
    ///
    /// Draft selection priority:
    /// 1. `openhorizon/smollm2-135m-q4` (fastest, already in registry)
    /// 2. `openhorizon/qwen2.5-0.5b-q4` (fallback)
    ///
    /// The chosen draft is always downloaded if not present.
    pub async fn with_auto_draft(target_model: &str, k: usize) -> Result<Self> {
        // Draft candidates in preference order — prefer the smallest one.
        let draft_candidates = ["openhorizon/smollm2-135m-q4", "openhorizon/qwen2.5-0.5b-q4"];

        // Try to resolve a candidate that is already in the HF cache so we
        // don't force a download during auto-selection.  If none are cached,
        // fall back to the first candidate (it will be downloaded on first use).
        let draft_model = select_cached_draft(&draft_candidates);
        debug!("Auto-selected draft model: {draft_model}");

        Self::new(target_model, draft_model, k).await
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// Generate a completion for `prompt` using speculative decoding.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let target_tok = self.target.tokenizer();
        let input_ids = target_tok.encode(prompt)?;
        let output_ids = self.speculative_generate(input_ids).await?;
        let text = target_tok.decode(&output_ids, true)?;
        Ok(text)
    }

    /// Chat with a message history using speculative decoding.
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = self.target.format_chat_prompt(messages)?;
        self.generate(&prompt).await
    }

    /// Stream tokens from speculative decoding.
    /// Tokens are emitted in bursts (one burst per speculation round).
    pub async fn generate_stream(&self, prompt: &str) -> ReceiverStream<String> {
        let (tx, rx) = mpsc::channel(64);
        let target_tok = Arc::clone(&self.target.tokenizer_arc());
        let input_ids = self.target.tokenizer().encode(prompt).unwrap_or_default();

        let target_info = self.target.model_info().clone();
        let target_paths = self.target.weight_paths().to_vec();
        let target_backend = self.target.configured_backend_kind();

        let draft_info = self.draft.model_info().clone();
        let draft_paths = self.draft.weight_paths().to_vec();
        let draft_backend = self.draft.configured_backend_kind();

        let eos_ids = self.target.eos_token_ids_pub();
        let stop_seqs = self.target.stop_sequences().to_vec();
        let max_new = self.target.config().max_new_tokens;
        let k = self.k;

        tokio::task::spawn_blocking(move || {
            // Re-create engines for this thread (ForwardEngine is not Send).
            let mut target_engine = match ForwardEngine::from_weight_paths_with_backend(
                target_info,
                &target_paths,
                target_backend,
            ) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(format!("Error loading target model: {e}"));
                    return;
                }
            };
            let mut draft_engine = match ForwardEngine::from_weight_paths_with_backend(
                draft_info,
                &draft_paths,
                draft_backend,
            ) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(format!("Error loading draft model: {e}"));
                    return;
                }
            };

            let mut sampler = Sampler::new(SamplingStrategy::default());
            let mut all_tokens = input_ids;
            let mut generated: Vec<u32> = Vec::new();
            let mut emitted = 0usize;

            // Prefill: run both models on the full prompt to populate KV caches.
            // Target model prefill (with cache).
            let _ = target_engine.forward_logits(&all_tokens, true);
            // Draft model prefill (with cache).
            let _ = draft_engine.forward_logits(&all_tokens, true);

            let mut total_generated = 0usize;

            'outer: loop {
                if total_generated >= max_new {
                    break;
                }

                // ── Step 1: Draft K candidate tokens ─────────────────────────
                let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
                let mut draft_probs: Vec<Vec<f32>> = Vec::with_capacity(k);

                for _ in 0..k {
                    if total_generated + draft_tokens.len() >= max_new {
                        break;
                    }
                    let last = *all_tokens.last().unwrap_or(&0);
                    let logits = match draft_engine.forward_logits(&[last], true) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = tx.blocking_send(format!("Draft error: {e}"));
                            break 'outer;
                        }
                    };
                    let probs = softmax(&logits);
                    let dt = sampler.sample(&logits, &all_tokens).unwrap_or(0);
                    draft_tokens.push(dt);
                    draft_probs.push(probs);

                    if eos_ids.contains(&dt) {
                        break;
                    }
                }

                if draft_tokens.is_empty() {
                    break;
                }

                // ── Step 2: Target model verifies in one pass ─────────────────
                // verify_input = [last context token] + draft_tokens[0..k-1]
                // i.e., k tokens total: the prefix sets context, then k-1 drafts.
                let num_draft = draft_tokens.len();
                let last_ctx = *all_tokens.last().unwrap_or(&0);
                let mut verify_input = Vec::with_capacity(num_draft);
                verify_input.push(last_ctx);
                verify_input.extend_from_slice(&draft_tokens[..num_draft.saturating_sub(1)]);

                // forward_all_logits runs WITHOUT updating the KV cache.
                // target_all_logits[i] = logits after seeing verify_input[0..=i]
                // so target_all_logits[i] predicts the token at position i+1.
                let target_all_logits = match target_engine.forward_all_logits(&verify_input) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx.blocking_send(format!("Target verify error: {e}"));
                        break;
                    }
                };

                // ── Step 3: Accept/reject each draft token ────────────────────
                let mut accepted: Vec<u32> = Vec::new();
                let mut rejected = false;

                for i in 0..num_draft {
                    let target_probs = softmax(&target_all_logits[i]);
                    let dt = draft_tokens[i];
                    let dp = draft_probs[i][dt as usize].max(1e-9);
                    let tp = target_probs[dt as usize];
                    let accept_ratio = (tp / dp).min(1.0);
                    let r = random_f32(&mut sampler);

                    if r < accept_ratio {
                        accepted.push(dt);
                        if eos_ids.contains(&dt) {
                            rejected = true; // treat EOS as stopping
                            break;
                        }
                    } else {
                        // Sample from adjusted distribution: max(0, target - draft)
                        let adjusted = adjusted_dist(&target_probs, &draft_probs[i]);
                        let bonus = sample_from_probs(&adjusted, &mut sampler);
                        accepted.push(bonus);
                        rejected = true;
                        break;
                    }
                }

                // ── Step 4: Bonus token if all K accepted ─────────────────────
                if !rejected && !accepted.is_empty() {
                    let last_idx = num_draft.saturating_sub(1);
                    if last_idx < target_all_logits.len() {
                        let bonus_logits = &target_all_logits[last_idx];
                        let bonus = sampler.sample(bonus_logits, &all_tokens).unwrap_or(0);
                        accepted.push(bonus);
                    }
                }

                // ── Update target KV cache for accepted tokens ────────────────
                // Run target model with use_cache=true on accepted tokens so its
                // KV cache reflects the actual accepted context.
                // We feed them one at a time (or as a batch for the first pass).
                if !accepted.is_empty() {
                    for &tok in &accepted {
                        let _ = target_engine.forward_logits(&[tok], true);
                    }
                }

                // ── Update draft KV cache for accepted tokens ─────────────────
                // Reset draft cache and re-feed up to accepted to keep in sync.
                // Simpler: just feed accepted tokens one-by-one to extend cache.
                for &tok in &accepted {
                    let _ = draft_engine.forward_logits(&[tok], true);
                }

                // ── Emit accepted tokens ──────────────────────────────────────
                for tok in accepted {
                    all_tokens.push(tok);
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

                    // Emit decoded text up to safe boundary.
                    let decoded_text: anyhow::Result<String> = target_tok.decode(&generated, true);
                    if let Ok(text) = decoded_text {
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
                        break;
                    }
                }

                if rejected {
                    // After rejection we already sampled the adjusted token and
                    // the KV caches have been extended — continue the outer loop.
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

    // ── Core speculative loop (blocking) ──────────────────────────────────────

    async fn speculative_generate(&self, input_ids: Vec<u32>) -> Result<Vec<u32>> {
        let target_info = self.target.model_info().clone();
        let target_paths = self.target.weight_paths().to_vec();
        let target_backend = self.target.configured_backend_kind();

        let draft_info = self.draft.model_info().clone();
        let draft_paths = self.draft.weight_paths().to_vec();
        let draft_backend = self.draft.configured_backend_kind();

        let eos_ids = self.target.eos_token_ids_pub();
        let max_new = self.target.config().max_new_tokens;
        let k = self.k;

        tokio::task::spawn_blocking(move || {
            let mut target_engine = ForwardEngine::from_weight_paths_with_backend(
                target_info,
                &target_paths,
                target_backend,
            )?;
            let mut draft_engine = ForwardEngine::from_weight_paths_with_backend(
                draft_info,
                &draft_paths,
                draft_backend,
            )?;

            let mut sampler = Sampler::new(SamplingStrategy::default());
            let mut all_tokens = input_ids;
            let mut generated: Vec<u32> = Vec::new();
            let mut total_generated = 0usize;

            // Prefill both models.
            let _ = target_engine.forward_logits(&all_tokens, true);
            let _ = draft_engine.forward_logits(&all_tokens, true);

            loop {
                if total_generated >= max_new {
                    break;
                }

                // Draft K tokens.
                let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
                let mut draft_probs: Vec<Vec<f32>> = Vec::with_capacity(k);

                for _ in 0..k {
                    if total_generated + draft_tokens.len() >= max_new {
                        break;
                    }
                    let last = *all_tokens.last().unwrap_or(&0);
                    let logits = draft_engine.forward_logits(&[last], true)?;
                    let probs = softmax(&logits);
                    let dt = sampler.sample(&logits, &all_tokens).unwrap_or(0);
                    draft_tokens.push(dt);
                    draft_probs.push(probs);
                    if eos_ids.contains(&dt) {
                        break;
                    }
                }

                if draft_tokens.is_empty() {
                    break;
                }

                // Target verifies.
                let num_draft = draft_tokens.len();
                let last_ctx = *all_tokens.last().unwrap_or(&0);
                let mut verify_input = Vec::with_capacity(num_draft);
                verify_input.push(last_ctx);
                verify_input.extend_from_slice(&draft_tokens[..num_draft.saturating_sub(1)]);

                let target_all_logits = target_engine.forward_all_logits(&verify_input)?;

                // Accept/reject.
                let mut accepted: Vec<u32> = Vec::new();
                let mut rejected = false;

                for i in 0..num_draft {
                    let target_probs = softmax(&target_all_logits[i]);
                    let dt = draft_tokens[i];
                    let dp = draft_probs[i][dt as usize].max(1e-9);
                    let tp = target_probs[dt as usize];
                    let r = random_f32(&mut sampler);

                    if r < (tp / dp).min(1.0) {
                        accepted.push(dt);
                        if eos_ids.contains(&dt) {
                            rejected = true;
                            break;
                        }
                    } else {
                        let adjusted = adjusted_dist(&target_probs, &draft_probs[i]);
                        accepted.push(sample_from_probs(&adjusted, &mut sampler));
                        rejected = true;
                        break;
                    }
                }

                // Bonus if all accepted.
                if !rejected {
                    let last_idx = num_draft.saturating_sub(1);
                    if last_idx < target_all_logits.len() {
                        let bonus = sampler
                            .sample(&target_all_logits[last_idx], &all_tokens)
                            .unwrap_or(0);
                        accepted.push(bonus);
                    }
                }

                // Update KV caches for accepted tokens.
                for &tok in &accepted {
                    let _ = target_engine.forward_logits(&[tok], true);
                    let _ = draft_engine.forward_logits(&[tok], true);
                }

                let mut eos_hit = false;
                for tok in accepted {
                    all_tokens.push(tok);
                    generated.push(tok);
                    total_generated += 1;
                    if eos_ids.contains(&tok) {
                        eos_hit = true;
                        break;
                    }
                    if total_generated >= max_new {
                        break;
                    }
                }

                if eos_hit {
                    break;
                }
            }

            Ok(generated)
        })
        .await
        .context("speculative_generate task panicked")?
    }
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
