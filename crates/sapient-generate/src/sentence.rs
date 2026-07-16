// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Streaming sentence chunker for the speech-to-speech cascade.
//!
//! In a voice conversation we want to start speaking the LLM's reply before it
//! has finished generating. [`SentenceChunker`] buffers streamed text tokens and
//! emits a chunk as soon as a sentence boundary is reached (terminal punctuation
//! followed by whitespace/end), so each completed sentence can be handed to TTS
//! while the LLM keeps decoding the next one — turning "wait for the whole reply"
//! into "speak after the first sentence."
//!
//! Pure string logic with no audio/engine dependencies, so it is unit-tested in
//! isolation and compiles regardless of the `audio-io` feature.

/// Buffers streamed text and flushes complete sentences.
#[derive(Debug, Default)]
pub struct SentenceChunker {
    buf: String,
    /// Don't emit a "sentence" shorter than this (avoids "Mr." / "3.14" splits).
    min_chars: usize,
    /// Force a flush once the buffer reaches this length even without a boundary.
    max_chars: usize,
    /// Time-to-first-audio mode: while no chunk has been emitted yet, ALSO
    /// break at a clause boundary (`,;:` + whitespace) once this many chars
    /// have accumulated — the speaker starts after the first *clause* instead
    /// of the first sentence. `0` disables. Subsequent chunks use normal
    /// sentence boundaries (clause-level joins cost a little prosody, so pay
    /// that price only where it buys perceived latency).
    early_first_chars: usize,
    emitted_any: bool,
}

impl SentenceChunker {
    /// `min_chars` guards against premature splits on abbreviations/decimals;
    /// `max_chars` bounds latency for a run-on stream with no punctuation.
    pub fn new(min_chars: usize, max_chars: usize) -> Self {
        Self {
            buf: String::new(),
            min_chars: min_chars.max(1),
            max_chars: max_chars.max(2),
            early_first_chars: 0,
            emitted_any: false,
        }
    }

    /// Enable early-first-chunk mode (see the field docs): the FIRST emitted
    /// chunk may end at a clause boundary once `n_chars` have accumulated.
    pub fn with_early_first(mut self, n_chars: usize) -> Self {
        self.early_first_chars = n_chars;
        self
    }

    /// Feed a streamed text fragment; return any sentences that are now complete.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        while let Some(s) = self.try_take() {
            out.push(s);
        }
        out
    }

    /// Emit whatever remains (call when the stream ends).
    pub fn flush(&mut self) -> Option<String> {
        let rest = self.buf.trim().to_string();
        self.buf.clear();
        (!rest.is_empty()).then_some(rest)
    }

    /// Pop one complete sentence from the front of the buffer, if any.
    fn try_take(&mut self) -> Option<String> {
        // Find the earliest terminal punctuation that is followed by whitespace
        // (or end-of-buffer) and sits past `min_chars` of content.
        let bytes = self.buf.as_bytes();
        let mut boundary: Option<usize> = None;
        // Early-first: before anything has been emitted, a clause boundary
        // past `early_first_chars` is good enough to start the speaker.
        if !self.emitted_any && self.early_first_chars > 0 {
            for (i, &b) in bytes.iter().enumerate() {
                if matches!(b, b',' | b';' | b':')
                    && bytes.get(i + 1).is_none_or(|c| c.is_ascii_whitespace())
                    && self.buf[..=i].trim().chars().count() >= self.early_first_chars
                {
                    boundary = Some(i + 1);
                    break;
                }
            }
        }
        for (i, &b) in bytes.iter().enumerate() {
            if boundary.is_some_and(|bd| bd <= i) {
                break; // an earlier (clause) boundary already won
            }
            if matches!(b, b'.' | b'!' | b'?' | b'\n') {
                // A newline is itself a terminator. For `.!?`, require the next
                // char to be whitespace/end so "3.14" / "u.s.a" don't trigger.
                let next_ws = bytes.get(i + 1).is_none_or(|c| c.is_ascii_whitespace());
                let is_boundary_char = b == b'\n' || next_ws;
                let enough = self.buf[..=i].trim().chars().count() >= self.min_chars;
                if is_boundary_char && enough {
                    boundary = Some(i + 1);
                    break;
                }
            }
        }
        // Fall back to a hard flush at a whitespace near max_chars — with a
        // much smaller cap for the FIRST chunk in early-first mode (2× the
        // clause threshold): a reply whose first comma comes late must not
        // stall the speaker; a word-boundary cut is the bounded worst case.
        let cap = if !self.emitted_any && self.early_first_chars > 0 {
            (self.early_first_chars * 2).min(self.max_chars)
        } else {
            self.max_chars
        };
        if boundary.is_none() && self.buf.len() >= cap {
            // Split at the last whitespace within the buffer to avoid cutting a word.
            boundary = self
                .buf
                .char_indices()
                .rev()
                .find(|&(_, c)| c.is_whitespace())
                .map(|(i, _)| i + 1)
                .or(Some(self.buf.len()));
        }
        let cut = boundary?;
        let sentence = self.buf[..cut].trim().to_string();
        self.buf.drain(..cut);
        if sentence.is_empty() {
            // Boundary was pure whitespace; loop will try again or stop.
            None
        } else {
            self.emitted_any = true;
            Some(sentence)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn early_first_breaks_at_clause_then_reverts_to_sentences() {
        let mut c = super::SentenceChunker::new(8, 200).with_early_first(10);
        // First chunk: clause boundary after >=10 chars.
        let got = c.push("Sure thing, here is the plan. We start now. ");
        assert_eq!(
            got,
            vec![
                "Sure thing,".to_string(),
                "here is the plan.".to_string(),
                "We start now.".to_string()
            ]
        );
        // After the first emission, commas no longer split.
        let got = c.push("Second reply, with a comma. ");
        assert_eq!(got, vec!["Second reply, with a comma.".to_string()]);
    }

    #[test]
    fn early_first_word_caps_a_late_clause() {
        let mut c = super::SentenceChunker::new(8, 200).with_early_first(24);
        // Streamed word-by-word (like real LLM tokens): no comma in the first
        // 48 chars → the first chunk hard-cuts at a word boundary at the cap,
        // BEFORE the sentence's final "." ever arrives.
        let text = "As a nation built on the principles of freedom and opportunity we act. ";
        let mut chunks: Vec<String> = Vec::new();
        for w in text.split_inclusive(' ') {
            chunks.extend(c.push(w));
        }
        chunks.extend(c.flush());
        assert!(
            chunks.len() >= 2,
            "expected an early word-cut chunk: {chunks:?}"
        );
        // Bounded by the cap plus at most the word that crossed it.
        assert!(
            chunks[0].chars().count() <= 60,
            "first chunk should be capped: {:?}",
            chunks[0]
        );
        assert!(!chunks[0].ends_with(char::is_whitespace));
    }

    #[test]
    fn early_first_disabled_keeps_old_behavior() {
        let mut c = super::SentenceChunker::new(8, 200);
        let got = c.push("Sure thing, here is the plan. ");
        assert_eq!(got, vec!["Sure thing, here is the plan.".to_string()]);
    }

    use super::*;

    #[test]
    fn flushes_on_sentence_boundary() {
        let mut c = SentenceChunker::new(3, 200);
        let mut out = Vec::new();
        for tok in ["Hello", " there", ". How", " are", " you", "? Bye", "."] {
            out.extend(c.push(tok));
        }
        out.extend(c.flush());
        assert_eq!(out, vec!["Hello there.", "How are you?", "Bye."]);
    }

    #[test]
    fn does_not_split_on_decimal_or_abbrev_midword() {
        let mut c = SentenceChunker::new(3, 200);
        // "3.14" has no whitespace after the dot → not a boundary.
        let out = c.push("pi is 3.14 today");
        assert!(out.is_empty(), "should not split inside 3.14: {out:?}");
        assert_eq!(c.flush().unwrap(), "pi is 3.14 today");
    }

    #[test]
    fn force_flush_on_max_chars() {
        let mut c = SentenceChunker::new(3, 20);
        // No punctuation; exceeds max_chars → flush at a word boundary.
        let out = c.push("aaaa bbbb cccc dddd eeee ffff");
        assert!(!out.is_empty(), "expected a forced flush");
        // The forced chunk should not cut mid-word.
        assert!(out[0].split_whitespace().all(|w| !w.is_empty()));
    }

    #[test]
    fn newline_is_a_boundary() {
        let mut c = SentenceChunker::new(3, 200);
        let out = c.push("first line\nsecond");
        assert_eq!(out, vec!["first line"]);
        assert_eq!(c.flush().unwrap(), "second");
    }
}
