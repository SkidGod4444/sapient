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

// ── ToolCall ──────────────────────────────────────────────────────────────────

/// One tool call on an assistant turn, in the shape HF chat templates expect:
/// a flat `{name, arguments}` where `arguments` is a JSON **value**, not a
/// string.
///
/// The OpenAI wire format sends `arguments` as a JSON-*encoded string*; callers
/// must parse it before building a `ToolCall`. Templates pipe it through
/// `| tojson` (Qwen2.5, Hermes), so handing them a string renders it
/// double-encoded — `"{\"deg\": 90}"` instead of `{"deg": 90}` — and the model
/// then imitates that malformed shape on its next turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

// ── ChatMessage ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Tool calls requested by an assistant turn. Templates that support tool
    /// use render these back into the transcript so the model can see what it
    /// already called; the rest ignore the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
            tool_calls: None,
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            tool_calls: None,
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            tool_calls: None,
        }
    }
    /// An assistant turn that called tools. `content` is typically empty — the
    /// model emits the call instead of prose.
    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            tool_calls: Some(tool_calls),
        }
    }
    /// A tool-result turn, fed back after the caller executed a tool call.
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: content.into(),
            tool_calls: None,
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
        self.render_with_tools(messages, None, add_generation_prompt)
    }

    /// Render with tool definitions exposed to the template.
    ///
    /// `tools` is the OpenAI-shaped array — `[{"type":"function","function":{
    /// "name":…, "description":…, "parameters":{…}}}]` — passed through
    /// verbatim, exactly as HF's `apply_chat_template(tools=…)` does.
    ///
    /// Tool-aware templates (Qwen2.5, Hermes) branch on the `tools` variable and
    /// emit the model's tool-call preamble; templates without that branch just
    /// ignore it, so passing tools to a non-tool model degrades to plain chat
    /// rather than erroring.
    pub fn render_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<String> {
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
                let mut v = serde_json::json!({
                    "role": m.role.to_string(),
                    "content": m.content,
                });
                // Only present on assistant turns that called tools. Templates
                // gate on `message.tool_calls`, so an absent key (not an empty
                // list) is what keeps a normal turn on the plain-prose branch.
                if let Some(calls) = &m.tool_calls {
                    v["tool_calls"] = serde_json::to_value(calls)
                        .map_err(|e| anyhow::anyhow!("Failed to serialize tool calls: {e}"))?;
                }
                Ok(v)
            })
            .collect::<Result<_>>()?;

        // `tools: None` serializes to JSON null, which is falsy in Jinja — so
        // `{%- if tools %}` takes the no-tools branch, same as HF omitting it.
        let ctx = serde_json::json!({
            "messages": messages_val,
            "tools": tools,
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

    /// The verbatim `chat_template` from `Qwen/Qwen2.5-1.5B-Instruct`'s
    /// `tokenizer_config.json`. Tool-calling correctness is entirely a property
    /// of *this* template rendering right, so the tests below run the real thing
    /// rather than a hand-written approximation — it exercises the `| tojson`
    /// filter, the `is defined` test, and `messages[loop.index0 - 1]` indexing,
    /// any of which silently changing under minijinja would break tool use.
    const QWEN25: &str = r#"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

    fn walk_tool() -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "walk",
                "description": "Walk forward a distance in meters.",
                "parameters": {
                    "type": "object",
                    "properties": {"meters": {"type": "number"}},
                    "required": ["meters"],
                },
            },
        })
    }

    #[test]
    fn qwen_renders_tool_definitions() {
        let tmpl = ChatTemplate::from_template(QWEN25);
        let messages = vec![ChatMessage::user("Walk two meters.")];
        let tools = [walk_tool()];

        let out = tmpl
            .render_with_tools(&messages, Some(&tools), true)
            .unwrap();

        // The tools preamble is what teaches the model the call syntax.
        assert!(out.contains("<tools>"), "missing tools block:\n{out}");
        assert!(
            out.contains(r#""name":"walk""#),
            "tool not rendered:\n{out}"
        );
        assert!(out.contains("<tool_call>"), "missing call syntax:\n{out}");
        assert!(out.trim_end().ends_with("<|im_start|>assistant"));
    }

    /// Without tools the template must take its no-tools branch — otherwise
    /// every plain chat request would be paying for a tools preamble.
    #[test]
    fn qwen_without_tools_has_no_preamble() {
        let tmpl = ChatTemplate::from_template(QWEN25);
        let messages = vec![ChatMessage::user("Hello!")];

        let out = tmpl.render(&messages, true).unwrap();

        assert!(!out.contains("<tools>"), "leaked tools preamble:\n{out}");
        assert!(out.contains("<|im_start|>user\nHello!<|im_end|>"));
    }

    /// The full agent round-trip: the model calls a tool, we execute it, and the
    /// result is fed back. Regression guard for the `arguments` encoding — the
    /// template pipes `arguments` through `| tojson`, so passing OpenAI's
    /// JSON-*string* form would render `"{\"meters\":2}"` (double-encoded) and
    /// teach the model to emit that same broken shape on the next turn.
    #[test]
    fn qwen_renders_tool_call_and_result_round_trip() {
        let tmpl = ChatTemplate::from_template(QWEN25);
        let messages = vec![
            ChatMessage::user("Walk two meters."),
            ChatMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    name: "walk".into(),
                    arguments: serde_json::json!({"meters": 2}),
                }],
            ),
            ChatMessage::tool("walked 2m, obstacle at 0.8m"),
        ];
        let tools = [walk_tool()];

        let out = tmpl
            .render_with_tools(&messages, Some(&tools), true)
            .unwrap();

        // arguments must be a bare JSON object, NOT a quoted string.
        assert!(
            out.contains(r#"{"name": "walk", "arguments": {"meters":2}}"#),
            "tool call mis-encoded:\n{out}"
        );
        assert!(!out.contains(r#"arguments": ""#), "double-encoded:\n{out}");
        assert!(
            out.contains("<tool_response>\nwalked 2m, obstacle at 0.8m\n</tool_response>"),
            "tool result not fed back:\n{out}"
        );
    }
}
