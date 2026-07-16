// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `WhisperTokenizer` — Whisper's GPT-2 BPE vocabulary plus its control-token
//! protocol (start-of-transcript, language, task, timestamps, end-of-text).
//!
//! Whisper is prompted with a *forced* decoder prefix rather than a chat
//! template, so this is kept separate from [`crate::SapientTokenizer`]. Special
//! token IDs are resolved from the tokenizer's added-token vocabulary, so the
//! same code works across multilingual checkpoints (tiny…large-v3).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

/// The 99 language codes Whisper supports (multilingual checkpoints). English-
/// only (`.en`) models simply won't have these tokens in their vocab.
const LANGUAGE_CODES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Whisper tokenizer + control-token registry.
pub struct WhisperTokenizer {
    inner: Tokenizer,
    /// `<|startoftranscript|>`
    pub sot: u32,
    /// `<|endoftext|>`
    pub eot: u32,
    /// `<|transcribe|>`
    pub transcribe: u32,
    /// `<|translate|>`
    pub translate: u32,
    /// `<|notimestamps|>`
    pub no_timestamps: u32,
    /// First timestamp token (`<|0.00|>`), one past `no_timestamps`.
    pub timestamp_begin: u32,
    /// `<|nospeech|>` / `<|nocaptions|>` (variant-dependent).
    pub no_speech: Option<u32>,
    /// Language code → token id, for those present in the vocab.
    lang_tokens: HashMap<String, u32>,
    /// Token id → language code (inverse of `lang_tokens`).
    lang_by_id: HashMap<u32, String>,
}

impl WhisperTokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("failed to load Whisper tokenizer: {e}"))?;
        Self::from_inner(inner)
    }

    /// Load from a HuggingFace model ID (uses the Hub cache).
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let inner = Tokenizer::from_pretrained(model_id, None)
            .map_err(|e| anyhow!("failed to load Whisper tokenizer for '{model_id}': {e}"))?;
        Self::from_inner(inner)
    }

    fn from_inner(inner: Tokenizer) -> Result<Self> {
        let id = |t: &str| inner.token_to_id(t);
        let req = |t: &str| {
            id(t).ok_or_else(|| anyhow!("Whisper tokenizer missing required token `{t}`"))
        };

        let sot = req("<|startoftranscript|>")?;
        let eot = id("<|endoftext|>").ok_or_else(|| anyhow!("missing `<|endoftext|>`"))?;
        let transcribe = req("<|transcribe|>")?;
        let translate = req("<|translate|>")?;
        let no_timestamps = req("<|notimestamps|>")?;
        let no_speech = id("<|nospeech|>").or_else(|| id("<|nocaptions|>"));

        let mut lang_tokens = HashMap::new();
        let mut lang_by_id = HashMap::new();
        for &code in LANGUAGE_CODES {
            if let Some(tid) = id(&format!("<|{code}|>")) {
                lang_tokens.insert(code.to_string(), tid);
                lang_by_id.insert(tid, code.to_string());
            }
        }

        Ok(Self {
            inner,
            sot,
            eot,
            transcribe,
            translate,
            no_timestamps,
            timestamp_begin: no_timestamps + 1,
            no_speech,
            lang_tokens,
            lang_by_id,
        })
    }

    /// Build the forced decoder prefix: `[<|sot|>, <|lang|>?, <|task|>,
    /// <|notimestamps|>?]`. `lang = None` omits the language token (used for the
    /// language-detection probe). Unknown language codes are ignored.
    pub fn sot_sequence(&self, lang: Option<&str>, translate: bool, timestamps: bool) -> Vec<u32> {
        let mut seq = vec![self.sot];
        if let Some(code) = lang {
            if let Some(&tid) = self.lang_tokens.get(code) {
                seq.push(tid);
            }
        }
        seq.push(if translate {
            self.translate
        } else {
            self.transcribe
        });
        if !timestamps {
            seq.push(self.no_timestamps);
        }
        seq
    }

    /// All language token ids (for restricting a language-detection argmax).
    pub fn language_token_ids(&self) -> Vec<u32> {
        self.lang_by_id.keys().copied().collect()
    }

    /// Language code for a language token id, if any.
    pub fn language_code(&self, id: u32) -> Option<&str> {
        self.lang_by_id.get(&id).map(|s| s.as_str())
    }

    /// True if `id` is the end-of-text marker.
    pub fn is_eot(&self, id: u32) -> bool {
        id == self.eot
    }

    /// True if `id` is a timestamp token (`<|t|>`).
    pub fn is_timestamp(&self, id: u32) -> bool {
        id >= self.timestamp_begin
    }

    /// True if `id` is any Whisper control token (sot/eot/lang/task/timestamp/…),
    /// i.e. anything at or above `<|endoftext|>` that should not be emitted as text.
    pub fn is_special(&self, id: u32) -> bool {
        id >= self.eot
    }

    /// Decode token ids to text. `skip_special` drops control tokens.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("Whisper decode error: {e}"))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    // Construction requires a real Whisper tokenizer.json (network/Hub), so the
    // sot_sequence ordering is validated in the ignored end-to-end test in
    // sapient-generate. Pure-unit coverage here would need an embedded fixture.
}
