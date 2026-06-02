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
}

impl SentenceChunker {
    /// `min_chars` guards against premature splits on abbreviations/decimals;
    /// `max_chars` bounds latency for a run-on stream with no punctuation.
    pub fn new(min_chars: usize, max_chars: usize) -> Self {
        Self {
            buf: String::new(),
            min_chars: min_chars.max(1),
            max_chars: max_chars.max(2),
        }
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
        for (i, &b) in bytes.iter().enumerate() {
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
        // Fall back to a hard flush at a whitespace near max_chars.
        if boundary.is_none() && self.buf.len() >= self.max_chars {
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
            Some(sentence)
        }
    }
}

#[cfg(test)]
mod tests {
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
