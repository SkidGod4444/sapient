//! `SapientTokenizer` — wraps the HuggingFace `tokenizers` crate.

use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

// ── TokenizerOptions ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TokenizerOptions {
    /// Add BOS token at the start of every encoding (default: true for LLMs).
    pub add_bos: bool,
    /// Add EOS token at the end of every encoding (default: false — let templates control this).
    pub add_eos: bool,
    /// Truncate to this many tokens (0 = no limit).
    pub max_length: usize,
}

impl Default for TokenizerOptions {
    fn default() -> Self {
        Self {
            add_bos: true,
            add_eos: false,
            max_length: 0,
        }
    }
}

// ── SapientTokenizer ──────────────────────────────────────────────────────────

/// A tokenizer loaded from a HuggingFace `tokenizer.json`.
///
/// Supports every tokenizer type HF ships: BPE, WordPiece, Unigram (SentencePiece).
/// Known end-of-turn / end-of-sequence marker tokens across model families.
/// A model may define several (e.g. Qwen has both `<|endoftext|>` and
/// `<|im_end|>`); generation must stop on *any* of them.
const EOS_CANDIDATES: &[&str] = &[
    "</s>",
    "<eos>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|im_end|>",
    "<end_of_turn>",
    "<|redacted_EOS|>",
];

pub struct SapientTokenizer {
    inner: Tokenizer,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    /// Every EOS/turn-end token id present in this tokenizer's vocab.
    pub eos_ids: Vec<u32>,
    pub pad_id: Option<u32>,
    opts: TokenizerOptions,
}

impl SapientTokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file(path: &Path, opts: TokenizerOptions) -> Result<Self> {
        match Tokenizer::from_file(path) {
            Ok(inner) => Self::from_inner(inner, opts),
            Err(first_err) => {
                let normalized = normalize_tokenizer_json(path).with_context(|| {
                    format!("Failed to load tokenizer and could not normalize it: {first_err}")
                })?;
                let inner = Tokenizer::from_bytes(&normalized)
                    .map_err(|e| anyhow::anyhow!("Failed to load normalized tokenizer: {e}"))?;
                Self::from_inner(inner, opts)
            }
        }
    }

    /// Load from a HuggingFace model ID string (uses the HF Hub cache).
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let inner = Tokenizer::from_pretrained(model_id, None)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer for '{model_id}': {e}"))?;

        let bos_id = Self::special_token_id(&inner, &["<s>", "<bos>", "<|begin_of_text|>"]);
        let eos_ids = Self::all_special_token_ids(&inner, EOS_CANDIDATES);
        let eos_id = eos_ids.first().copied();
        let pad_id = Self::special_token_id(&inner, &["<pad>"]);

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            eos_ids,
            pad_id,
            opts: TokenizerOptions::default(),
        })
    }

    /// Encode a text string to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?;

        let mut ids = encoding.get_ids().to_vec();

        // Prepend BOS if configured.
        if self.opts.add_bos {
            if let Some(bos) = self.bos_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }

        // Append EOS if configured.
        if self.opts.add_eos {
            if let Some(eos) = self.eos_id {
                ids.push(eos);
            }
        }

        // Truncate if needed.
        if self.opts.max_length > 0 && ids.len() > self.opts.max_length {
            ids.truncate(self.opts.max_length);
        }

        Ok(ids)
    }

    /// Encode `text` to raw token IDs **without** injecting BOS/EOS.
    ///
    /// `add_special` toggles the tokenizer's own special-token logic (post-
    /// processor). Audio LMs such as Orpheus TTS build their own special-token
    /// framing around the text, so they encode the body with `add_special =
    /// false` and prepend/append the control ids themselves.
    pub fn encode_ids(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {e}"))
    }

    /// Decode a single token ID to a string (for streaming).
    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id], true)
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// True if `id` is any of this tokenizer's end-of-sequence markers.
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }

    fn special_token_id(tok: &Tokenizer, candidates: &[&str]) -> Option<u32> {
        for c in candidates {
            if let Some(id) = tok.token_to_id(c) {
                return Some(id);
            }
        }
        None
    }

    /// All ids (in candidate order, deduplicated) for tokens present in the vocab.
    fn all_special_token_ids(tok: &Tokenizer, candidates: &[&str]) -> Vec<u32> {
        let mut ids = Vec::new();
        for c in candidates {
            if let Some(id) = tok.token_to_id(c) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    fn from_inner(inner: Tokenizer, opts: TokenizerOptions) -> Result<Self> {
        let bos_id =
            Self::special_token_id(&inner, &["<s>", "<bos>", "<|begin_of_text|>", "[BOS]"]);
        let eos_ids = Self::all_special_token_ids(&inner, EOS_CANDIDATES);
        let eos_id = eos_ids.first().copied();
        let pad_id =
            Self::special_token_id(&inner, &["<pad>", "<|finetune_right_pad_id|>", "[PAD]"]);

        Ok(Self {
            inner,
            bos_id,
            eos_id,
            eos_ids,
            pad_id,
            opts,
        })
    }
}

/// Normalize newer HuggingFace tokenizer JSON into a format older `tokenizers`
/// versions can deserialize (e.g. BPE merges stored as `[a, b]` pairs).
fn normalize_tokenizer_json(path: &Path) -> Result<Vec<u8>> {
    let text = std::fs::read_to_string(path)?;
    let mut value: serde_json::Value = serde_json::from_str(&text)?;

    let Some(model) = value.get_mut("model") else {
        anyhow::bail!("tokenizer.json missing model section");
    };
    let Some(merges) = model.get_mut("merges") else {
        anyhow::bail!("tokenizer.json missing BPE merges");
    };
    let Some(arr) = merges.as_array_mut() else {
        anyhow::bail!("tokenizer merges are not an array");
    };
    if arr.is_empty() {
        return Ok(text.into_bytes());
    }
    if !arr[0].is_array() {
        anyhow::bail!("tokenizer merges already use string format");
    }

    let normalized: Vec<String> = arr
        .iter()
        .filter_map(|entry| {
            let pair = entry.as_array()?;
            if pair.len() != 2 {
                return None;
            }
            Some(format!("{} {}", pair[0].as_str()?, pair[1].as_str()?))
        })
        .collect();

    if normalized.len() != arr.len() {
        anyhow::bail!("failed to normalize all BPE merges");
    }

    *merges = serde_json::Value::Array(
        normalized
            .into_iter()
            .map(serde_json::Value::String)
            .collect(),
    );

    Ok(serde_json::to_vec(&value)?)
}

#[cfg(test)]
mod tests {
    // Integration tests require network access to download tokenizer.json.
    // Run with: cargo test -p sapient-tokenizers -- --ignored
    // (or point at a local tokenizer.json)
}
