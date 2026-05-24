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
        Self { add_bos: true, add_eos: false, max_length: 0 }
    }
}

// ── SapientTokenizer ──────────────────────────────────────────────────────────

/// A tokenizer loaded from a HuggingFace `tokenizer.json`.
///
/// Supports every tokenizer type HF ships: BPE, WordPiece, Unigram (SentencePiece).
pub struct SapientTokenizer {
    inner: Tokenizer,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    opts: TokenizerOptions,
}

impl SapientTokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file(path: &Path, opts: TokenizerOptions) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let bos_id = Self::special_token_id(&inner, &["<s>", "<bos>", "[BOS]"]);
        let eos_id = Self::special_token_id(&inner, &["</s>", "<eos>", "[EOS]", "<|endoftext|>", "<|im_end|>"]);
        let pad_id = Self::special_token_id(&inner, &["<pad>", "[PAD]"]);

        Ok(Self { inner, bos_id, eos_id, pad_id, opts })
    }

    /// Load from a HuggingFace model ID string (uses the HF Hub cache).
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let inner = Tokenizer::from_pretrained(model_id, None)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer for '{model_id}': {e}"))?;

        let bos_id = Self::special_token_id(&inner, &["<s>", "<bos>"]);
        let eos_id = Self::special_token_id(&inner, &["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"]);
        let pad_id = Self::special_token_id(&inner, &["<pad>"]);

        Ok(Self { inner, bos_id, eos_id, pad_id, opts: TokenizerOptions::default() })
    }

    /// Encode a text string to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, true)
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

    /// Decode token IDs back to a string.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner.decode(ids, skip_special)
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

    fn special_token_id(tok: &Tokenizer, candidates: &[&str]) -> Option<u32> {
        for c in candidates {
            if let Some(id) = tok.token_to_id(c) {
                return Some(id);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require network access to download tokenizer.json.
    // Run with: cargo test -p sapient-tokenizers -- --ignored
    // (or point at a local tokenizer.json)
}
