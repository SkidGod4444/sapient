//! GGUF file selection helpers for HuggingFace Hub downloads.

/// Pick the best GGUF file from a sorted list of repo filenames.
///
/// Prefers Q4_K_M (the edge sweet spot — small + near-lossless) over Q8_0 and other
/// quants, falls back to Q8_0/K-quants/float, and avoids ultra-low-bit Q2_K.
///
/// **`SAPIENT_GGUF_QUANT`** overrides the preference (Phase 8.2, low-RAM boards):
/// e.g. `SAPIENT_GGUF_QUANT=Q4_K_S sapient pull llama-3.2-3b` picks the smaller
/// Q4_K_S file on a 4 GB Pi where the default Q4_K_M would squeeze RAM. Matched
/// case-insensitively against the filename; falls back to the normal scoring
/// (with a warning) when no file matches.
pub fn select_best_gguf(filenames: &[String]) -> Option<&str> {
    let ggufs: Vec<&str> = filenames
        .iter()
        .filter(|n| n.ends_with(".gguf"))
        .map(String::as_str)
        .collect();

    if ggufs.is_empty() {
        return None;
    }

    let want = std::env::var("SAPIENT_GGUF_QUANT").ok();
    select_gguf_with_override(&ggufs, want.as_deref())
}

/// Pure selection core (unit-testable without touching process env): pick the
/// override-matching file when `want` is set and matches, else the best-scoring
/// one.
fn select_gguf_with_override<'a>(ggufs: &[&'a str], want: Option<&str>) -> Option<&'a str> {
    if let Some(want) = want.map(|w| w.trim().to_ascii_uppercase()) {
        if !want.is_empty() {
            if let Some(hit) = ggufs
                .iter()
                .filter(|n| n.to_ascii_uppercase().contains(&want))
                .max_by_key(|name| gguf_preference_score(name))
            {
                return Some(hit);
            }
            tracing::warn!(
                "SAPIENT_GGUF_QUANT={want}: no matching .gguf in this repo — \
                 falling back to the default quant preference"
            );
        }
    }

    ggufs
        .iter()
        .max_by_key(|name| gguf_preference_score(name))
        .copied()
}

/// Score a GGUF filename — higher is better.
///
/// Quantized formats are strongly preferred over F16/BF16. GGUF repos exist
/// specifically for quantized deployment; BF16/F16 in a GGUF repo is a
/// full-precision fallback that costs 2–3× more RAM than Q4_K_M for no
/// benefit on CPU-only inference. Only fall back to F16/BF16 when no
/// quantized file is offered.
pub fn gguf_preference_score(name: &str) -> i32 {
    let upper = name.to_ascii_uppercase();
    let base = upper
        .rsplit('/')
        .next()
        .unwrap_or(&upper)
        .replace(".GGUF", "");

    // Edge-device sweet spot: Q4_K_M is preferred over Q8_0. It is ~40% smaller
    // (≈4.9 GB vs ≈8.5 GB for an 8B model), near-lossless, and runs through the
    // Q4_K matmul kernel — which fits a 16 GB Pi where the larger Q8_0 file would
    // force memory-mapping. Q8_0 stays a high-quality fallback when no K-quant
    // variant is published.
    if base.contains("Q4_K_M") {
        return 100;
    }
    if base.contains("Q5_K_M") {
        return 96;
    }
    if base.contains("Q8_0") {
        return 94;
    }
    if base.contains("Q4_0") {
        return 85;
    }
    if base.contains("Q5_0") {
        return 80;
    }
    if base.contains("Q6_K") {
        return 75;
    }
    if base.contains("Q4_K_S") {
        return 70;
    }
    if base.contains("Q5_K_S") {
        return 65;
    }
    if base.contains("Q3_K_M") {
        return 55;
    }
    if base.contains("Q3_K_S") {
        return 50;
    }
    if base.contains("Q4_K") {
        return 60;
    }
    if base.contains("Q5_K") {
        return 58;
    }
    if base.contains("Q3_K") {
        return 45;
    }
    if base.contains("Q2_K") {
        return 30;
    }
    // Float formats — only chosen when no quantized alternative exists.
    if base.contains("F32") {
        return 10;
    }
    if base.contains("F16") || base.contains("BF16") {
        return 5;
    }
    // Unknown quant tag — still usable if it is the only GGUF.
    20
}

/// HuggingFace model ID to use for tokenizer files when a GGUF-only repo has none.
pub fn tokenizer_fallback_model(model_id: &str) -> Option<&'static str> {
    let id = model_id.to_ascii_lowercase();

    // Orpheus TTS extends the Llama-3.2 tokenizer with SNAC audio-codec tokens
    // (vocab 156940). The base Llama-3 tokenizer (128256) would mismatch the
    // model and fail the vocab-compat check, so resolve the ungated Orpheus
    // tokenizer. Checked before the generic arch fallback ("llama" → Llama-3.1).
    if id.contains("orpheus") {
        return Some("unsloth/orpheus-3b-0.1-ft");
    }
    // SmolLM / SmolLM2
    if id.contains("smollm") {
        return Some("HuggingFaceTB/SmolLM2-360M-Instruct");
    }
    // DeepSeek R1 distills carry their own (ungated) tokenizer.
    if id.contains("deepseek") {
        return Some("deepseek-ai/DeepSeek-R1-Distill-Llama-8B");
    }
    if id.contains("llama-2") || id.contains("llama2") {
        return Some("NousResearch/Llama-2-7b-hf");
    }
    // Use the ungated unsloth mirror — meta-llama/* repos are gated (401 without a
    // HF token), which would make every Llama-3 GGUF unusable out of the box.
    if id.contains("llama-3") || id.contains("llama3") {
        return Some("unsloth/Meta-Llama-3.1-8B-Instruct");
    }
    // Plain "llama" arch from GGUF metadata (no version suffix)
    if id == "llama" {
        return Some("unsloth/Meta-Llama-3.1-8B-Instruct");
    }
    // Mistral v0.3 (vocab 32768) REORDERED the vocabulary vs v0.1/v0.2 (vocab
    // 32000) — the tokenizers are NOT interchangeable. Loading a v0.1 tokenizer for
    // a v0.3 GGUF mis-encodes the prompt and mis-decodes output into mixed-script
    // token-salad (verified: v0.1 decodes v0.3 ids as "Г str — ...らíses...レ").
    // Our GGUF catalog ships Mistral-7B-Instruct-v0.3, so default to the matching
    // (ungated) v0.3 tokenizer; only fall back to v0.1 when the id names v0.1/v0.2.
    if id.contains("mistral") || id.contains("ministral") {
        if id.contains("v0.1") || id.contains("v0.2") || id.contains("v01") || id.contains("v02") {
            return Some("mistralai/Mistral-7B-v0.1");
        }
        return Some("unsloth/mistral-7b-instruct-v0.3");
    }
    if id.contains("codellama") || id.contains("code-llama") {
        return Some("codellama/CodeLlama-7b-hf");
    }
    // Phi-4-mini uses a GPT-2/tiktoken BPE tokenizer (~200k vocab) — completely
    // different from Phi-3's 32k SentencePiece. The GGUF general.name is "Phi 4
    // Mini Instruct" (spaces), so match those too. Loading the Phi-3 tokenizer here
    // mis-encodes every prompt → repeated/garbage output.
    if id.contains("phi-4") || id.contains("phi4") || id.contains("phi 4") {
        return Some("microsoft/Phi-4-mini-instruct");
    }
    if id.contains("phi-3") || id.contains("phi3") {
        return Some("microsoft/Phi-3-mini-4k-instruct");
    }
    if id.contains("phi-2") || id.contains("phi2") {
        return Some("microsoft/phi-2");
    }
    if id.contains("gemma-2") || id.contains("gemma2") {
        return Some("google/gemma-2-2b-it");
    }
    if id.contains("gemma") {
        return Some("google/gemma-2b");
    }
    if id.contains("qwen2.5") || id.contains("qwen-2.5") {
        return Some("Qwen/Qwen2.5-7B-Instruct");
    }
    if id.contains("qwen2") || id.contains("qwen-2") {
        return Some("Qwen/Qwen2-7B-Instruct");
    }
    if id.contains("qwen") {
        return Some("Qwen/Qwen2-7B-Instruct");
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_override_picks_matching_file() {
        let files = vec!["model-Q4_K_M.gguf", "model-Q4_K_S.gguf", "model-Q8_0.gguf"];
        // Default preference: Q4_K_M wins.
        assert_eq!(
            select_gguf_with_override(&files, None),
            Some("model-Q4_K_M.gguf")
        );
        // Low-RAM override (Phase 8.2): case-insensitive substring match.
        assert_eq!(
            select_gguf_with_override(&files, Some("q4_k_s")),
            Some("model-Q4_K_S.gguf")
        );
        // Ambiguous override resolves by score among the matches.
        assert_eq!(
            select_gguf_with_override(&files, Some("Q4_K")),
            Some("model-Q4_K_M.gguf")
        );
        // Non-matching override falls back to the default preference.
        assert_eq!(
            select_gguf_with_override(&files, Some("Q2_K")),
            Some("model-Q4_K_M.gguf")
        );
        // Blank override is ignored.
        assert_eq!(
            select_gguf_with_override(&files, Some("  ")),
            Some("model-Q4_K_M.gguf")
        );
    }

    #[test]
    fn prefers_q4_k_m_over_q8_and_q2() {
        // Edge default: Q4_K_M beats Q8_0 (smaller, fits constrained devices) and
        // both beat the ultra-low-bit Q2_K.
        let files = vec![
            "llama-2-7b.Q2_K.gguf".into(),
            "llama-2-7b.Q4_K_M.gguf".into(),
            "llama-2-7b.Q8_0.gguf".into(),
        ];
        assert_eq!(select_best_gguf(&files), Some("llama-2-7b.Q4_K_M.gguf"));
    }

    #[test]
    fn prefers_q8_when_no_k_quant() {
        // Q8_0 remains the high-quality fallback when no K-quant is published.
        let files = vec!["llama-2-7b.Q2_K.gguf".into(), "llama-2-7b.Q8_0.gguf".into()];
        assert_eq!(select_best_gguf(&files), Some("llama-2-7b.Q8_0.gguf"));
    }

    #[test]
    fn prefers_q4_k_m_over_q2() {
        let files = vec![
            "llama-2-7b.Q2_K.gguf".into(),
            "llama-2-7b.Q4_K_M.gguf".into(),
        ];
        assert_eq!(select_best_gguf(&files), Some("llama-2-7b.Q4_K_M.gguf"));
    }

    #[test]
    fn llama2_tokenizer_fallback() {
        assert_eq!(
            tokenizer_fallback_model("TheBloke/Llama-2-7B-GGUF"),
            Some("NousResearch/Llama-2-7b-hf")
        );
    }

    #[test]
    fn mistral_tokenizer_fallback_defaults_to_v03() {
        // v0.3 reordered the vocab (32768) vs v0.1/v0.2 (32000); a v0.1 tokenizer on
        // a v0.3 GGUF produces mixed-script salad. Default to the matching v0.3.
        assert_eq!(
            tokenizer_fallback_model("Mistral-7B-Instruct-v0.3"),
            Some("unsloth/mistral-7b-instruct-v0.3")
        );
        assert_eq!(
            tokenizer_fallback_model("bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
            Some("unsloth/mistral-7b-instruct-v0.3")
        );
        // Explicit older versions still map to the 32000-vocab v0.1 tokenizer.
        assert_eq!(
            tokenizer_fallback_model("mistralai/Mistral-7B-v0.1"),
            Some("mistralai/Mistral-7B-v0.1")
        );
    }

    #[test]
    fn orpheus_tokenizer_fallback_is_extended_vocab() {
        // The Orpheus GGUF's general.name is "Orpheus Tts 0.1 Pretrained" — it
        // must resolve to the 156940-vocab Orpheus tokenizer, NOT the generic
        // Llama-3 fallback (128256), which would fail the vocab-compat check.
        assert_eq!(
            tokenizer_fallback_model("Orpheus Tts 0.1 Pretrained"),
            Some("unsloth/orpheus-3b-0.1-ft")
        );
        assert_eq!(
            tokenizer_fallback_model("isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"),
            Some("unsloth/orpheus-3b-0.1-ft")
        );
    }
}
