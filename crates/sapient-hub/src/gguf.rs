//! GGUF file selection helpers for HuggingFace Hub downloads.

/// Pick the best GGUF file from a sorted list of repo filenames.
///
/// Prefers formats we can dequantize natively (Q8_0, Q4_0) then common K-quants
/// (Q4_K_M, Q5_K_M), and avoids ultra-low-bit quants (Q2_K) when better options exist.
pub fn select_best_gguf(filenames: &[String]) -> Option<&str> {
    let ggufs: Vec<&str> = filenames
        .iter()
        .filter(|n| n.ends_with(".gguf"))
        .map(String::as_str)
        .collect();

    if ggufs.is_empty() {
        return None;
    }

    ggufs
        .iter()
        .max_by_key(|name| gguf_preference_score(name))
        .copied()
}

/// Score a GGUF filename — higher is better.
pub fn gguf_preference_score(name: &str) -> i32 {
    let upper = name.to_ascii_uppercase();
    let base = upper
        .rsplit('/')
        .next()
        .unwrap_or(&upper)
        .replace(".GGUF", "");

    // Full-precision / well-supported native dequant formats first.
    if base.contains("F32") {
        return 110;
    }
    if base.contains("F16") || base.contains("BF16") {
        return 105;
    }
    if base.contains("Q8_0") {
        return 100;
    }
    if base.contains("Q4_K_M") {
        return 95;
    }
    if base.contains("Q5_K_M") {
        return 90;
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
    if base.contains("Q2_K") {
        return 30;
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

    // Unknown quant tag — still usable if it is the only GGUF.
    40
}

/// HuggingFace model ID to use for tokenizer files when a GGUF-only repo has none.
pub fn tokenizer_fallback_model(model_id: &str) -> Option<&'static str> {
    let id = model_id.to_ascii_lowercase();

    if id.contains("llama-2") || id.contains("llama2") {
        return Some("NousResearch/Llama-2-7b-hf");
    }
    if id.contains("llama-3") || id.contains("llama3") {
        return Some("meta-llama/Meta-Llama-3-8B-Instruct");
    }
    if id.contains("mistral") {
        return Some("mistralai/Mistral-7B-v0.1");
    }
    if id.contains("codellama") || id.contains("code-llama") {
        return Some("codellama/CodeLlama-7b-hf");
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
    fn prefers_q8_over_q2() {
        let files = vec![
            "llama-2-7b.Q2_K.gguf".into(),
            "llama-2-7b.Q4_K_M.gguf".into(),
            "llama-2-7b.Q8_0.gguf".into(),
        ];
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
}
