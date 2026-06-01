//! Chat template rendering using Jinja2 via `minijinja`.
//!
//! Renders HuggingFace chat templates (stored in `tokenizer_config.json`)
//! exactly as the Python `transformers` library does.
//!
//! Supports: Llama 3 / ChatML / Mistral / Phi / Gemma / Qwen / Zephyr templates.

use std::path::Path;

use anyhow::{Context, Result};
use minijinja::Environment;
use serde::{Deserialize, Serialize};

// ── ChatRole ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::System => f.write_str("system"),
            ChatRole::User => f.write_str("user"),
            ChatRole::Assistant => f.write_str("assistant"),
            ChatRole::Tool => f.write_str("tool"),
        }
    }
}

// ── ChatMessage ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

// ── ChatTemplate ──────────────────────────────────────────────────────────────

/// Renders a conversation to a tokenizable string using the model's
/// Jinja2 chat template (from `tokenizer_config.json`).
pub struct ChatTemplate {
    template_src: String,
}

impl ChatTemplate {
    /// Load from a `tokenizer_config.json` file.
    pub fn from_tokenizer_config(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).context("Failed to read tokenizer_config.json")?;
        let config: serde_json::Value =
            serde_json::from_str(&text).context("Invalid tokenizer_config.json")?;

        let template_src = config["chat_template"]
            .as_str()
            .context("No chat_template found in tokenizer_config.json")?
            .to_owned();

        Ok(Self { template_src })
    }

    /// Build with a raw Jinja2 template string.
    pub fn from_template(template: impl Into<String>) -> Self {
        Self {
            template_src: template.into(),
        }
    }

    /// Render a list of chat messages to a prompt string.
    ///
    /// Set `add_generation_prompt = true` to append the assistant turn header
    /// (so the model knows to generate).
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        let mut env = Environment::new();

        // Register the template.
        env.add_template("chat", &self.template_src)
            .map_err(|e| anyhow::anyhow!("Template parse error: {e}"))?;

        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("Template load error: {e}"))?;

        // Build context — same variable names as HF Python.
        let messages_val: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role.to_string(),
                    "content": m.content,
                })
            })
            .collect();

        let ctx = serde_json::json!({
            "messages": messages_val,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": "<s>",
            "eos_token": "</s>",
        });

        tmpl.render(ctx)
            .map_err(|e| anyhow::anyhow!("Template render error: {e}"))
    }
}

/// Built-in templates for common models (fallback when tokenizer_config.json
/// doesn't contain a chat_template field).
pub mod builtin {
    /// ChatML format — used by Phi-3, Qwen, Mistral-Instruct variants.
    pub const CHATML: &str = concat!(
        "{% for message in messages %}",
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n",
        "{% endfor %}",
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
    );

    /// Phi-3 / Phi-4 instruct format: `<|role|>\n{content}<|end|>\n` per turn,
    /// then `<|assistant|>\n`. Turn terminator is `<|end|>`.
    pub const PHI3: &str = concat!(
        "{% for message in messages %}",
        "<|{{ message['role'] }}|>\n{{ message['content'] }}<|end|>\n",
        "{% endfor %}",
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}",
    );

    /// Llama 3 Instruct format.
    pub const LLAMA3: &str = concat!(
        "<|begin_of_text|>",
        "{% for message in messages %}",
        "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n",
        "{{ message['content'] }}<|eot_id|>",
        "{% endfor %}",
        "{% if add_generation_prompt %}",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "{% endif %}",
    );

    /// Llama 2 / Mistral instruct format.
    pub const LLAMA2: &str = concat!(
        "{% if messages[0]['role'] == 'system' %}",
        "{{ '[INST] <<SYS>>\n' + messages[0]['content'] + '\n<</SYS>>\n\n' }}",
        "{% set messages = messages[1:] %}",
        "{% endif %}",
        "{% for message in messages %}",
        "{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}",
        "{% elif message['role'] == 'assistant' %}{{ message['content'] + '</s>' }}",
        "{% endif %}",
        "{% endfor %}",
    );

    /// Gemma instruct format.
    pub const GEMMA: &str = concat!(
        "{% for message in messages %}",
        "{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n",
        "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n",
        "{% endif %}",
        "{% endfor %}",
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}",
    );

    /// Zephyr / TinyLlama chat format.
    pub const ZEPHYR: &str = concat!(
        "{% for message in messages %}",
        "{% if message['role'] == 'system' %}",
        "<|system|>\n{{ message['content'] }}</s>\n",
        "{% elif message['role'] == 'user' %}",
        "<|user|>\n{{ message['content'] }}</s>\n",
        "{% elif message['role'] == 'assistant' %}",
        "<|assistant|>\n{{ message['content'] }}</s>\n",
        "{% endif %}",
        "{% endfor %}",
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_render() {
        let tmpl = ChatTemplate::from_template(builtin::CHATML);
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
        ];
        let out = tmpl.render(&messages, true).unwrap();
        assert!(out.contains("<|im_start|>system"));
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("<|im_start|>assistant"));
    }

    #[test]
    fn llama3_render() {
        let tmpl = ChatTemplate::from_template(builtin::LLAMA3);
        let messages = vec![ChatMessage::user("What is 2+2?")];
        let out = tmpl.render(&messages, true).unwrap();
        assert!(out.contains("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
    }
}
