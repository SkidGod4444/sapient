//! Terminal UI helpers for interactive commands.
//!
//! Styling goes through the `console` crate, which automatically disables
//! colours when output is not a TTY or when `NO_COLOR` is set — so piped output
//! stays clean while interactive sessions look modern.

use std::io::{self, Write};
use std::time::Duration;

use console::{style, Emoji};
use indicatif::{ProgressBar, ProgressStyle};

// Emoji with plain-text fallbacks for terminals without unicode/emoji support.
static BOLT: Emoji<'_, '_> = Emoji("⚡", "*");
static CHECK: Emoji<'_, '_> = Emoji("✓", "OK");
static CROSS: Emoji<'_, '_> = Emoji("✗", "x");
static INFO: Emoji<'_, '_> = Emoji("ℹ", "i");
static ARROW: Emoji<'_, '_> = Emoji("›", ">");
static NOTE: Emoji<'_, '_> = Emoji("♪", "~");

/// A branded spinner shown on stderr while a long operation runs.
/// Shows a live elapsed count so slow devices (Pi-class) never feel hung.
/// Call [`ProgressBar::finish_and_clear`] when done, or [`spinner_success`]
/// to settle it into a green ✓ line.
pub fn spinner(message: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg} {elapsed:.dim}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]),
    );
    pb.set_message(message.into());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// Settle a spinner into a permanent one-line receipt: green ✓ + message +
/// dim elapsed time. The small "done" moment that tells the user the wait was
/// real and is over (spinners that vanish silently read as "did it work?").
/// Printed to stderr, like the spinner itself, so piped stdout stays clean.
pub fn spinner_success(pb: ProgressBar, msg: impl std::fmt::Display) {
    let took = pb.elapsed();
    pb.finish_and_clear();
    let took = if took >= Duration::from_millis(100) {
        style(format!(" ({})", fmt_duration(took)))
            .dim()
            .to_string()
    } else {
        String::new()
    };
    eprintln!("{} {}{}", style(CHECK).green().bold(), msg, took);
}

/// Compact human duration: "780ms", "1.4s", "2m 05s".
pub fn fmt_duration(d: Duration) -> String {
    let ms = d.as_millis();
    if ms < 1_000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.1}s", d.as_secs_f64())
    } else {
        format!("{}m {:02}s", ms / 60_000, (ms % 60_000) / 1000)
    }
}

/// The brand wordmark: a cyan→blue gradient on truecolor terminals, bold cyan
/// everywhere else. One subtle moment of identity, zero cost when piped.
pub fn brand(text: &str) -> String {
    let truecolor = console::colors_enabled()
        && std::env::var("COLORTERM")
            .map(|v| v.contains("truecolor") || v.contains("24bit"))
            .unwrap_or(false);
    if !truecolor {
        return style(text).bold().cyan().to_string();
    }
    // Lerp #22d3ee (cyan-400) → #3b82f6 (blue-500) across the glyphs.
    let (from, to) = ((0x22i32, 0xd3i32, 0xeei32), (0x3bi32, 0x82i32, 0xf6i32));
    let chars: Vec<char> = text.chars().collect();
    let steps = chars.len().saturating_sub(1).max(1) as i32;
    let lerp = |a: i32, b: i32, t: i32| (a + (b - a) * t / steps) as u8;
    let mut out = String::new();
    for (i, c) in chars.iter().enumerate() {
        let t = i as i32;
        let (r, g, b) = (
            lerp(from.0, to.0, t),
            lerp(from.1, to.1, t),
            lerp(from.2, to.2, t),
        );
        out.push_str(&format!("\x1b[1m\x1b[38;2;{r};{g};{b}m{c}"));
    }
    out.push_str("\x1b[0m");
    out
}

/// The SAPIENT wordmark banner, shown at the top of interactive sessions.
pub fn print_logo() {
    let bar = style("━".repeat(52)).dim();
    println!("{bar}");
    println!(
        "  {} {}   {}",
        BOLT,
        brand("SAPIENT"),
        style("edge inference engine").dim()
    );
    println!("{bar}");
}

pub fn print_chat_banner(model: &str, arch: &str, backend: &str) {
    let bar = style("━".repeat(52)).dim();
    println!("\n{bar}");
    println!(
        "  {} {} {}",
        brand("SAPIENT"),
        style("Chat").bold(),
        style(format!("· {arch} · {backend}")).dim()
    );
    println!("  {} {}", style("model").dim(), style(model).bold());
    println!(
        "  {}",
        style("type a message · /help for commands · /exit to quit").dim()
    );
    if arch.to_ascii_lowercase().contains("vision") || model.to_ascii_lowercase().contains("vlm") {
        println!(
            "  {}",
            style("note: vision models run in text-only mode for now").yellow()
        );
    }
    println!("{bar}");
}

pub fn print_chat_help() {
    println!("\n  {}", style("Commands").bold());
    for (cmd, desc) in [
        ("/help, /?", "show this help"),
        ("/clear", "clear the conversation history"),
        ("/exit, /quit, /q", "leave the chat"),
    ] {
        println!("    {:<18} {}", style(cmd).cyan(), style(desc).dim());
    }
    println!();
}

/// The styled "you" input prompt as a string (leading newline + badge + arrow).
/// Used as the prompt for the line editor so bracketed-paste / cursor math line up.
pub fn user_prompt_str() -> String {
    format!(
        "\n{} {} ",
        style(" you ").black().on_green().bold(),
        style(ARROW).green().dim()
    )
}

pub fn write_assistant_prompt() -> io::Result<()> {
    write!(
        io::stdout(),
        "\n{} {} ",
        style(" sapient ").black().on_cyan().bold(),
        style(ARROW).cyan().dim()
    )?;
    io::stdout().flush()
}

/// A dim one-line generation stat shown after a reply (tokens & speed).
/// `ttft` is the time-to-first-token — shown when available.
pub fn print_gen_stats(tokens: usize, elapsed: Duration, ttft: Option<Duration>) {
    let secs = elapsed.as_secs_f64().max(1e-6);
    let tps = tokens as f64 / secs;
    let ttft_str = ttft
        .map(|d| format!("  (first token: {}ms)", d.as_millis()))
        .unwrap_or_default();
    println!(
        "{}",
        style(format!(
            "  {BOLT} {tokens} tokens · {tps:.1} tok/s · {secs:.1}s{ttft_str}"
        ))
        .dim()
    );
}

// ── converse (live voice) UI ──────────────────────────────────────────────────

/// Banner shown when `sapient converse` starts. `backend` is the resolved
/// compute label (e.g. "metal (MLX native graph)" or a CPU label).
pub fn converse_banner(
    input_rate: u32,
    stt: &str,
    llm: &str,
    backend: &str,
    speak: bool,
    tts_engine: &str,
) {
    let bar = style("━".repeat(52)).dim();
    println!("\n{bar}");
    println!(
        "  {} {} {}",
        brand("SAPIENT"),
        style("Voice").bold(),
        style(format!("· in {input_rate}Hz · stt {stt} · llm {llm}")).dim()
    );
    // Compute backend — green when accelerated, yellow + hint when CPU-only.
    let lower = backend.to_ascii_lowercase();
    let accelerated = lower.contains("metal") || lower.contains("gpu") || lower.contains("wgpu");
    if accelerated {
        println!("  {} {}", style("compute").dim(), style(backend).green());
    } else {
        println!(
            "  {} {} {}",
            style("compute").dim(),
            style(backend).yellow(),
            style("— for GPU latency run the accelerated build: `sapient update --metal`").dim()
        );
    }
    let mode = if speak {
        let label = if tts_engine == "orpheus" {
            format!("{NOTE} voice replies on (Orpheus-3B TTS · slow on CPU)")
        } else {
            format!("{NOTE} voice replies on (Kokoro-82M TTS · real-time)")
        };
        style(label).green()
    } else {
        style(format!("{INFO} text replies · pass --speak to hear them")).dim()
    };
    println!("  {mode}");
    println!("  {}", style("speak, then pause — Ctrl-C to stop").dim());
    println!("{bar}");
}

/// Live input-level meter with a decaying peak-hold marker — the classic
/// audio-gear nano-interaction. The bar tracks the instantaneous level; the
/// lone `▎` tick rides the recent maximum and glides back down, so brief
/// speech spikes stay visible between redraws instead of flickering away.
#[derive(Default)]
pub struct MicMeter {
    peak: f32,
}

impl MicMeter {
    const WIDTH: usize = 22;
    /// Per-redraw peak decay (redraws arrive every ~50–100 ms in converse).
    const DECAY: f32 = 0.90;

    /// One row of the meter (returned so the caller controls the in-place
    /// `\r` redraw). Green = loud, cyan = speech-ish, dim = quiet.
    pub fn line(&mut self, rms: f32) -> String {
        let level = (rms * 60.0).min(1.0);
        self.peak = (self.peak * Self::DECAY).max(level);
        let bars = (level * Self::WIDTH as f32).round() as usize;
        let peak_cell = ((self.peak * Self::WIDTH as f32).round() as usize)
            .min(Self::WIDTH)
            .max(bars);
        let fill = "█".repeat(bars);
        let fill = if level > 0.5 {
            style(fill).green()
        } else if level > 0.12 {
            style(fill).cyan()
        } else {
            style(fill).dim()
        };
        // Quiet gap up to the peak marker, then the marker, then the rest —
        // always exactly WIDTH cells (the marker occupies the peak cell).
        let (gap, tick, rest) = if peak_cell > bars {
            (peak_cell - bars - 1, "▎", Self::WIDTH - peak_cell)
        } else {
            (0, "", Self::WIDTH - bars)
        };
        let gap = "·".repeat(gap);
        format!(
            "\r\x1b[2K  {} {}{}{}{}",
            style("mic").dim(),
            fill,
            style(gap).dim(),
            style(tick).cyan(),
            style("·".repeat(rest)).dim()
        )
    }
}

/// A transient dim status on the current line (e.g. "transcribing…").
pub fn converse_status(msg: &str) {
    print!("\r\x1b[2K  {}", style(format!("· {msg}")).dim());
    let _ = io::stdout().flush();
}

/// The user's transcribed line: `[ you ] › <text>`.
pub fn converse_you(transcript: &str) {
    println!(
        "\r\x1b[2K{} {} {}",
        style(" you ").black().on_green().bold(),
        style(ARROW).green().dim(),
        transcript
    );
}

/// The assistant badge + arrow (no newline) — stream reply tokens after it.
pub fn converse_assistant_prefix() {
    print!(
        "{} {} ",
        style(" sapient ").black().on_cyan().bold(),
        style(ARROW).cyan().dim()
    );
    let _ = io::stdout().flush();
}

/// Dim STT telemetry under the user line.
pub fn converse_stt_stats(audio_secs: f32, stt: Duration) {
    let rt = audio_secs / stt.as_secs_f32().max(1e-3);
    println!(
        "  {}",
        style(format!(
            "{INFO} heard {audio_secs:.1}s · STT {}ms ({rt:.1}× realtime)",
            stt.as_millis()
        ))
        .dim()
    );
}

/// Dim generation/TTS telemetry under the assistant line. `tts` is
/// `(synthesis_time, spoken_audio_secs)` when `--speak` is on.
pub fn converse_gen_stats(tokens: usize, gen: Duration, tts: Option<(Duration, f32)>) {
    let secs = gen.as_secs_f64().max(1e-6);
    let tps = tokens as f64 / secs;
    let mut line = format!("{BOLT} {tokens} tok · {tps:.1} tok/s · {secs:.1}s");
    if let Some((tts_d, audio_secs)) = tts {
        let tsecs = tts_d.as_secs_f32();
        let rt = audio_secs / tsecs.max(1e-3);
        line.push_str(&format!(" · {NOTE} TTS {tsecs:.1}s ({rt:.2}× realtime)"));
    }
    println!("  {}", style(line).dim());
}

/// A dim note on its own line (clears any in-place meter first).
pub fn converse_note(msg: &str) {
    println!("\r\x1b[2K  {}", style(format!("· {msg}")).dim());
}

/// Closing line when the session ends.
pub fn converse_bye() {
    println!("\n  {}", style("ended — bye").dim());
}

/// A yellow warning row (e.g. a silent mic).
pub fn converse_warn(msg: &str) {
    eprintln!(
        "  {} {}",
        style(" ! ").black().on_yellow().bold(),
        style(msg).yellow()
    );
}

/// Per-run result for `bench-llm`.
pub struct BenchRun {
    pub run: usize,
    pub ttft_ms: u64,
    pub tps: f64,
    pub total_tokens: usize,
}

/// Print a bench-llm results table.
pub fn print_bench_table(
    model: &str,
    backend: &str,
    load_ms: u64,
    runs: &[BenchRun],
    peak_rss_mb: u64,
) {
    let bar = style("━".repeat(52)).dim();
    println!("\n{bar}");
    println!(
        "  {} {}",
        style("SAPIENT bench-llm").bold().cyan(),
        style(format!("· {backend}")).dim()
    );
    println!("  {} {}", style("model").dim(), style(model).bold());
    println!("{bar}");
    println!(
        "  {} {:.2}s",
        style("Load time").dim(),
        load_ms as f64 / 1000.0
    );
    println!();

    let headers = &["Run", "TTFT", "Tok/s", "Tokens"];
    let rows: Vec<Vec<String>> = runs
        .iter()
        .map(|r| {
            vec![
                format!("{}", r.run),
                format!("{} ms", r.ttft_ms),
                format!("{:.1}", r.tps),
                format!("{}", r.total_tokens),
            ]
        })
        .collect();
    print_table(headers, &rows);

    if !runs.is_empty() {
        let mean_ttft = runs.iter().map(|r| r.ttft_ms).sum::<u64>() / runs.len() as u64;
        let mean_tps = runs.iter().map(|r| r.tps).sum::<f64>() / runs.len() as f64;
        println!();
        println!(
            "  {} {}ms  {}  {} {}  {}  {} {} MB",
            style("Mean TTFT:").dim(),
            style(mean_ttft).bold().cyan(),
            style("|").dim(),
            style("Mean tok/s:").dim(),
            style(format!("{mean_tps:.1}")).bold().cyan(),
            style("|").dim(),
            style("Peak RSS:").dim(),
            style(peak_rss_mb).bold(),
        );
    }
    println!("{bar}\n");
}

/// `key: value` info row used by `sapient info`.
pub fn info_row(key: &str, value: impl std::fmt::Display) {
    println!("  {:<12} {}", style(key).dim(), value);
}

pub fn success(msg: impl std::fmt::Display) {
    println!("{} {}", style(CHECK).green().bold(), msg);
}

pub fn hint(msg: impl std::fmt::Display) {
    println!("{} {}", style(INFO).cyan(), style(msg).dim());
}

/// Render a simple aligned table with a dim header rule. Column widths use
/// *display* width (ANSI stripped, wide glyphs counted) so styled cells,
/// unicode model names, and emoji don't skew the alignment.
pub fn print_table(headers: &[&str], rows: &[Vec<String>]) {
    use console::measure_text_width;

    let mut widths: Vec<usize> = headers.iter().map(|h| measure_text_width(h)).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(measure_text_width(cell));
            }
        }
    }
    // Pad by display width: format!'s {:<w} counts chars, not columns or ANSI.
    let pad = |s: &str, w: usize| {
        let fill = w.saturating_sub(measure_text_width(s));
        format!("{s}{}  ", " ".repeat(fill))
    };

    let mut header_line = String::from("  ");
    for (i, h) in headers.iter().enumerate() {
        header_line.push_str(&pad(h, widths[i]));
    }
    println!("{}", style(header_line).bold());

    let rule: usize = widths.iter().map(|w| w + 2).sum::<usize>();
    println!("  {}", style("─".repeat(rule)).dim());

    for row in rows {
        let mut line = String::from("  ");
        for (i, cell) in row.iter().enumerate() {
            let w = widths.get(i).copied().unwrap_or(0);
            line.push_str(&pad(cell, w));
        }
        println!("{line}");
    }
}

/// Print an error with its full cause chain, one dim `↳` per cause — instead
/// of anyhow's single "a: b: c" run-on line, which buries the actionable root.
pub fn failure_with_chain(err: &anyhow::Error) {
    eprintln!("{} {}", style(CROSS).red().bold(), style(err).red());
    for cause in err.chain().skip(1) {
        eprintln!("    {} {}", style("↳").dim(), style(cause).dim());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The meter line is always exactly WIDTH cells of bar content, with the
    /// peak-hold tick decaying back toward the live level over redraws.
    #[test]
    fn mic_meter_peak_holds_then_decays() {
        let mut m = MicMeter::default();
        let loud = m.line(1.0); // full-scale hit
        let quiet1 = m.line(0.0); // peak tick should survive…
        assert!(quiet1.contains('▎'), "peak marker shown after a spike");
        for _ in 0..80 {
            m.line(0.0); // …and decay away eventually
        }
        let settled = m.line(0.0);
        assert!(!settled.contains('▎'), "peak marker decays to silence");
        // Constant display width across states (strip ANSI, count columns).
        let w = |s: &str| console::measure_text_width(s.trim_start_matches('\r'));
        assert_eq!(w(&loud), w(&quiet1));
        assert_eq!(w(&loud), w(&settled));
    }

    #[test]
    fn duration_formatting_is_compact() {
        assert_eq!(fmt_duration(Duration::from_millis(780)), "780ms");
        assert_eq!(fmt_duration(Duration::from_millis(1_400)), "1.4s");
        assert_eq!(fmt_duration(Duration::from_secs(125)), "2m 05s");
    }
}
