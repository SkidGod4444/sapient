// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Live Markdown rendering for streamed chat replies.
//!
//! Chat models emit Markdown — headings, **bold**, lists, tables, and fenced
//! code blocks. Printing the raw token stream shows the markup characters
//! verbatim. [`StreamRenderer`] instead renders the stream as formatted terminal
//! output *while it streams*:
//!
//! - Prose, headings, lists, quotes, tables, inline styling → [`termimad`].
//! - Fenced code blocks → [`syntect`] 24-bit syntax highlighting.
//!
//! ## Streaming strategy — "commit & preview"
//!
//! Repainting the entire reply on every token would thrash the screen and breaks
//! once the output scrolls past the top of the viewport (the cursor can't move
//! back up to it). Instead the renderer splits the reply into Markdown *blocks*
//! (separated by blank lines, or closed by a code fence):
//!
//! - **Completed blocks** are rendered once and printed permanently — they scroll
//!   naturally and are never touched again.
//! - The **trailing incomplete block** is repainted in place on each update by
//!   moving the cursor up over it and clearing. The repaint region is therefore
//!   one block, not the whole reply.
//!
//! A viewport guard commits the preview early if a single in-progress block would
//! overflow the screen height (e.g. a very long unclosed code block), trading live
//! formatting for correctness in that rare case.
//!
//! Non-TTY output (pipes, `NO_COLOR`, `--raw`) falls back to raw passthrough so
//! piping `sapient chat` into a file or another program stays clean.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use console::measure_text_width;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};
use termimad::MadSkin;

/// Repaint at most ~30 times/second while tokens stream without a newline.
const PAINT_INTERVAL: Duration = Duration::from_millis(33);

/// Dim gray gutter drawn to the left of every highlighted code line.
const CODE_GUTTER: &str = "\x1b[38;5;240m│\x1b[0m ";

pub struct StreamRenderer {
    /// The full reply text accumulated so far (raw Markdown).
    full: String,
    /// Byte offset in `full` up to which blocks have been permanently printed.
    committed: usize,
    /// Number of terminal rows the current live preview occupies (to erase next).
    preview_rows: usize,
    /// Whether to render rich Markdown (true) or stream raw text (false).
    rich: bool,
    /// Wrap width for termimad / overflow math.
    width: usize,
    /// Terminal height, for the viewport-overflow guard.
    term_rows: usize,
    skin: MadSkin,
    syntaxes: SyntaxSet,
    theme: Theme,
    last_paint: Instant,
    /// Buffer for detecting `<think>` / `</think>` tags that may straddle tokens.
    pending: String,
    /// True while inside a `<think>…</think>` reasoning span.
    reasoning_active: bool,
    /// Whether the "💭 thinking" header has been printed for the current span.
    reasoning_header_shown: bool,
    /// True once any reasoning has been rendered (so we can separate it cleanly).
    reasoning_any: bool,
}

impl StreamRenderer {
    /// Build a renderer. `force_raw` (the `--raw` flag) disables rich rendering
    /// even on a TTY. Rich rendering is also disabled for non-terminals and when
    /// `NO_COLOR` is set.
    pub fn new(force_raw: bool) -> Self {
        let term = console::Term::stdout();
        let is_tty = term.is_term();
        let rich = is_tty && !force_raw && std::env::var_os("NO_COLOR").is_none();
        let (rows, cols) = if is_tty { term.size() } else { (24, 80) };
        let width = (cols as usize).clamp(20, 120);

        Self {
            full: String::new(),
            committed: 0,
            preview_rows: 0,
            rich,
            width,
            term_rows: rows as usize,
            skin: default_skin(),
            syntaxes: SyntaxSet::load_defaults_newlines(),
            theme: theme(),
            last_paint: Instant::now(),
            pending: String::new(),
            reasoning_active: false,
            reasoning_header_shown: false,
            reasoning_any: false,
        }
    }

    /// Call once after the assistant prompt badge is printed, before the first
    /// token. Reserves a fresh line below the badge for the rendered region.
    pub fn begin(&mut self) -> io::Result<()> {
        if self.rich {
            let mut out = io::stdout();
            writeln!(out)?;
            out.flush()?;
        }
        Ok(())
    }

    /// Feed one streamed token/chunk.
    pub fn push(&mut self, chunk: &str) -> io::Result<()> {
        if !self.rich {
            self.full.push_str(chunk);
            let mut out = io::stdout();
            out.write_all(chunk.as_bytes())?;
            return out.flush();
        }
        // Split off `<think>…</think>` reasoning spans (rendered dimmed) from the
        // answer (rendered as Markdown) before feeding the block pipeline.
        self.feed(chunk)
    }

    /// Streaming `<think>`/`</think>` splitter. Reasoning text is rendered dimmed
    /// under a "💭 thinking" header; everything else is the answer and flows into
    /// the Markdown block pipeline. Tags may straddle token boundaries, so a
    /// possible partial-tag suffix is held back in `self.pending`.
    fn feed(&mut self, chunk: &str) -> io::Result<()> {
        self.pending.push_str(chunk);
        loop {
            if self.reasoning_active {
                if let Some(p) = self.pending.find("</think>") {
                    let reason: String = self.pending[..p].into();
                    self.emit_reasoning(&reason)?;
                    self.pending.drain(..p + "</think>".len());
                    self.reasoning_active = false;
                    self.end_reasoning()?;
                } else {
                    let keep = partial_tag_suffix(&self.pending, "</think>");
                    let emit = self.pending.len() - keep;
                    let reason: String = self.pending[..emit].into();
                    self.emit_reasoning(&reason)?;
                    self.pending.drain(..emit);
                    break;
                }
            } else if let Some(p) = self.pending.find("<think>") {
                let ans: String = self.pending[..p].into();
                self.push_answer(&ans)?;
                // Reasoning prints as free-scrolling dim text below the cursor,
                // outside the block-managed region — commit the live preview
                // first, or the next repaint's move-up-N erases the reasoning
                // lines and leaves stale preview rows above them.
                self.repaint(true)?;
                self.pending.drain(..p + "<think>".len());
                self.reasoning_active = true;
                self.reasoning_header_shown = false;
            } else {
                let keep = partial_tag_suffix(&self.pending, "<think>");
                let emit = self.pending.len() - keep;
                let ans: String = self.pending[..emit].into();
                self.push_answer(&ans)?;
                self.pending.drain(..emit);
                break;
            }
        }
        Ok(())
    }

    /// Append answer text and repaint (throttled unless a newline arrived).
    fn push_answer(&mut self, text: &str) -> io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.full.push_str(text);
        let force = text.contains('\n');
        if !force && self.last_paint.elapsed() < PAINT_INTERVAL {
            return Ok(());
        }
        self.repaint(false)
    }

    /// Print reasoning text dimmed (live, scrolling — not block-managed).
    fn emit_reasoning(&mut self, text: &str) -> io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let mut out = io::stdout();
        if !self.reasoning_header_shown {
            self.reasoning_header_shown = true;
            self.reasoning_any = true;
            // Dim "thinking" header on its own line.
            writeln!(out, "\x1b[2m\x1b[3m💭 thinking…\x1b[0m")?;
        }
        // Dim per chunk, reset per chunk: an interrupt (Ctrl-C, io error) must
        // never leave the user's terminal stuck rendering dim.
        write!(out, "\x1b[2m{text}\x1b[0m")?;
        out.flush()
    }

    /// Close the reasoning span: reset styling and separate from the answer.
    fn end_reasoning(&mut self) -> io::Result<()> {
        let mut out = io::stdout();
        writeln!(out, "\x1b[0m")?; // reset dim, newline before the answer
        self.reasoning_header_shown = false;
        out.flush()
    }

    /// Flush the final state: commit the trailing block and end the line.
    pub fn finish(&mut self) -> io::Result<()> {
        if !self.rich {
            let mut out = io::stdout();
            if !self.full.is_empty() {
                writeln!(out)?;
            }
            return out.flush();
        }
        // Drain any buffered tail (e.g. a lone "<" that never became a tag).
        let tail = std::mem::take(&mut self.pending);
        if self.reasoning_active {
            self.emit_reasoning(&tail)?;
            self.end_reasoning()?;
            self.reasoning_active = false;
        } else {
            self.push_answer(&tail)?;
        }
        self.repaint(true)?;
        let mut out = io::stdout();
        writeln!(out)?;
        out.flush()
    }

    /// The accumulated answer text (reasoning is excluded — kept out of history,
    /// matching the convention for reasoning models). Used for history + tokens.
    /// In rich mode `full` never receives reasoning; in raw mode the stream is
    /// passed through verbatim (including tags), so strip the spans here — a
    /// `--raw` or piped session must not feed `<think>` text back as context.
    pub fn into_text(self) -> String {
        if self.rich {
            self.full
        } else {
            strip_think_spans(&self.full)
        }
    }

    fn repaint(&mut self, final_: bool) -> io::Result<()> {
        let mut out = io::stdout();

        // 1. Erase the previously-painted live preview region.
        if self.preview_rows > 0 {
            write!(out, "\x1b[{}A", self.preview_rows)?;
        }
        write!(out, "\r\x1b[0J")?;

        // 2. Commit any newly-completed blocks (everything, when finalizing).
        let rel = complete_prefix_len(&self.full[self.committed..], final_);
        if rel > 0 {
            let done = self.render(&self.full[self.committed..self.committed + rel]);
            out.write_all(done.as_bytes())?;
            ensure_newline(&mut out, &done)?;
            self.committed += rel;
        }

        // 3. Repaint the trailing incomplete block as a live preview, with a
        //    dim block cursor riding the stream head — the "still typing"
        //    signal every modern LLM surface uses.
        self.preview_rows = 0;
        if !final_ {
            let preview_src = &self.full[self.committed..];
            if !preview_src.is_empty() {
                let plain = self.render(preview_src);
                if cursor_down_moves(&plain, self.width) >= self.term_rows.saturating_sub(2) {
                    // Viewport guard: this block alone would overflow the screen,
                    // so commit it (let it scroll) rather than try to repaint it.
                    // Committed content must not carry the stream cursor.
                    out.write_all(plain.as_bytes())?;
                    ensure_newline(&mut out, &plain)?;
                    self.committed = self.full.len();
                } else {
                    // Attach the cursor to the last visual line (before its
                    // '\n' if present) so it rides the text, not a blank row.
                    let cursor = "\x1b[2m\x1b[36m▍\x1b[0m";
                    let preview = match plain.strip_suffix('\n') {
                        Some(stripped) => format!("{stripped}{cursor}\n"),
                        None => format!("{plain}{cursor}"),
                    };
                    out.write_all(preview.as_bytes())?;
                    self.preview_rows = cursor_down_moves(&preview, self.width);
                }
            }
        }

        self.last_paint = Instant::now();
        out.flush()
    }

    /// Render a slice of Markdown to ANSI: prose via termimad, fenced code via
    /// syntect. Code blocks are pulled out so syntect's colors aren't escaped by
    /// termimad's own code-block styling.
    fn render(&self, md: &str) -> String {
        let mut out = String::new();
        let mut text_buf = String::new();
        let mut code_buf = String::new();
        let mut in_fence = false;
        let mut fence_lang = String::new();

        for line in md.split_inclusive('\n') {
            let trimmed = line.trim_start();
            let is_fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");
            if is_fence {
                if in_fence {
                    out.push_str(&self.render_code(&fence_lang, &code_buf));
                    code_buf.clear();
                    fence_lang.clear();
                    in_fence = false;
                } else {
                    if !text_buf.is_empty() {
                        out.push_str(&self.render_text(&text_buf));
                        text_buf.clear();
                    }
                    fence_lang = trimmed.trim_start_matches(['`', '~']).trim().to_lowercase();
                    in_fence = true;
                }
            } else if in_fence {
                code_buf.push_str(line);
            } else {
                text_buf.push_str(line);
            }
        }

        // Flush whatever trailing segment remains (an unclosed code fence in a
        // live preview renders as a partial highlighted block).
        if in_fence {
            out.push_str(&self.render_code(&fence_lang, &code_buf));
        } else if !text_buf.is_empty() {
            out.push_str(&self.render_text(&text_buf));
        }
        out
    }

    fn render_text(&self, text: &str) -> String {
        self.skin.text(text, Some(self.width)).to_string()
    }

    fn render_code(&self, lang: &str, code: &str) -> String {
        let syntax = if lang.is_empty() {
            self.syntaxes.find_syntax_plain_text()
        } else {
            self.syntaxes
                .find_syntax_by_token(lang)
                .unwrap_or_else(|| self.syntaxes.find_syntax_plain_text())
        };
        let mut hl = HighlightLines::new(syntax, &self.theme);
        let mut out = String::new();
        for line in LinesWithEndings::from(code) {
            let escaped = match hl.highlight_line(line, &self.syntaxes) {
                Ok(ranges) => as_24_bit_terminal_escaped(&ranges, false),
                Err(_) => line.to_string(),
            };
            out.push_str(CODE_GUTTER);
            out.push_str(escaped.trim_end_matches('\n'));
            out.push_str("\x1b[0m\n");
        }
        out
    }
}

/// Byte length of the leading portion of `s` that forms complete Markdown blocks.
///
/// When `final_` the whole slice is complete. Otherwise the commit point is the
/// end of the last blank line (paragraph boundary) or the last closing code fence
/// — everything after is the still-in-progress trailing block.
fn complete_prefix_len(s: &str, final_: bool) -> usize {
    if final_ {
        return s.len();
    }
    let mut in_fence = false;
    let mut last_commit = 0;
    let mut pos = 0;
    for line in s.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let is_fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");
        let complete_line = line.ends_with('\n');
        if is_fence {
            if in_fence {
                in_fence = false;
                if complete_line {
                    last_commit = pos + line.len();
                }
            } else {
                in_fence = true;
            }
        } else if !in_fence && complete_line && line.trim().is_empty() {
            last_commit = pos + line.len();
        }
        pos += line.len();
    }
    last_commit
}

/// Terminal rows the cursor advances when `s` is printed starting at column 0,
/// accounting for soft-wrapping of lines wider than `width`.
fn cursor_down_moves(s: &str, width: usize) -> usize {
    let parts: Vec<&str> = s.split('\n').collect();
    let mut down = parts.len().saturating_sub(1); // one move per '\n'
    if width > 0 {
        for p in &parts {
            let w = measure_text_width(p); // strips ANSI, counts display columns
            if w > width {
                down += (w - 1) / width; // extra soft-wrapped rows
            }
        }
    }
    down
}

/// Length of the longest suffix of `s` that is a (proper or full) prefix of
/// `tag`. Used to hold back a possibly-incomplete `<think>`/`</think>` tag that
/// straddles a token boundary, so we never emit half a tag as visible text.
fn partial_tag_suffix(s: &str, tag: &str) -> usize {
    let max = tag.len().min(s.len());
    for n in (1..=max).rev() {
        if s.as_bytes()[s.len() - n..] == tag.as_bytes()[..n] {
            return n;
        }
    }
    0
}

fn ensure_newline(out: &mut impl Write, s: &str) -> io::Result<()> {
    if !s.ends_with('\n') {
        writeln!(out)?;
    }
    Ok(())
}

/// Remove `<think>…</think>` spans (and an unterminated trailing `<think>…`)
/// from a completed reply. Used by the raw/non-TTY path, where the stream is
/// printed verbatim but reasoning must still stay out of the chat history.
fn strip_think_spans(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find("<think>") {
        out.push_str(&rest[..open]);
        match rest[open..].find("</think>") {
            Some(close) => rest = &rest[open + close + "</think>".len()..],
            None => return out, // unterminated span: drop to end
        }
    }
    out.push_str(rest);
    out
}

fn default_skin() -> MadSkin {
    let mut skin = MadSkin::default();
    skin.set_headers_fg(termimad::crossterm::style::Color::Cyan);
    skin
}

fn theme() -> Theme {
    let mut themes = ThemeSet::load_defaults().themes;
    themes
        .remove("base16-ocean.dark")
        .or_else(|| themes.values().next().cloned())
        .expect("syntect ships default themes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commits_at_blank_line_not_mid_paragraph() {
        let s = "first paragraph\n\nsecond para in progress";
        let n = complete_prefix_len(s, false);
        assert_eq!(&s[..n], "first paragraph\n\n");
    }

    #[test]
    fn does_not_commit_inside_open_code_fence() {
        let s = "intro\n\n```rust\nlet x = 1;\n\nlet y = 2;";
        let n = complete_prefix_len(s, false);
        // Blank lines inside the fence must NOT be treated as block boundaries.
        assert_eq!(&s[..n], "intro\n\n");
    }

    #[test]
    fn commits_through_closed_code_fence() {
        let s = "```rust\nlet x = 1;\n```\nnext";
        let n = complete_prefix_len(s, false);
        assert_eq!(&s[..n], "```rust\nlet x = 1;\n```\n");
    }

    #[test]
    fn finalizing_commits_everything() {
        let s = "an unterminated paragraph";
        assert_eq!(complete_prefix_len(s, true), s.len());
    }

    #[test]
    fn partial_tag_suffix_holds_back_straddling_tags() {
        // A trailing partial tag must be held back so we never emit half a tag.
        assert_eq!(partial_tag_suffix("reasoning</thi", "</think>"), 5); // "</thi" is a prefix of "</think>"
        assert_eq!(partial_tag_suffix("hello <", "<think>"), 1); // "<" could start "<think>"
        assert_eq!(partial_tag_suffix("hello <think>", "<think>"), 7); // full tag held (caller finds it)
        assert_eq!(partial_tag_suffix("plain text", "<think>"), 0); // nothing to hold
        assert_eq!(partial_tag_suffix("done</think>", "</think>"), 8);
    }

    #[test]
    fn renders_code_block_with_gutter_and_color() {
        let r = StreamRenderer::new(true);
        let out = r.render("```rust\nlet x = 1;\n```\n");
        assert!(out.contains(CODE_GUTTER), "code lines get a gutter");
        assert!(out.contains("\x1b["), "syntect emits ANSI color escapes");
        assert!(out.contains('1'), "code content is preserved");
    }

    #[test]
    fn renders_prose_with_ansi_styling() {
        let r = StreamRenderer::new(true);
        let out = r.render("Hello **world**\n");
        assert!(out.contains("world"), "text content is preserved");
        assert!(out.contains("\x1b["), "termimad emits styling escapes");
    }

    #[test]
    fn strip_think_spans_removes_reasoning() {
        assert_eq!(
            strip_think_spans("<think>let me see</think>Paris."),
            "Paris."
        );
        assert_eq!(
            strip_think_spans("a<think>x</think>b<think>y</think>c"),
            "abc"
        );
        // Unterminated span drops to the end; plain text passes through.
        assert_eq!(strip_think_spans("hi<think>never closed"), "hi");
        assert_eq!(strip_think_spans("no tags at all"), "no tags at all");
    }

    #[test]
    fn wrap_aware_row_count() {
        // 3 visual lines: "a", a line 2*width wide (wraps to 2 rows), trailing "".
        let w = 10;
        let line = "x".repeat(20);
        let s = format!("a\n{line}\n");
        // "a" -> 1 part move, wide line -> 1 part move + 1 wrap, trailing "" part.
        // parts = ["a", wide, ""] => 2 newline moves + 1 wrap = 3.
        assert_eq!(cursor_down_moves(&s, w), 3);
    }
}
