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

/// A branded spinner shown on stderr while a long operation runs.
/// Call [`ProgressBar::finish_and_clear`] when done.
pub fn spinner(message: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]),
    );
    pb.set_message(message.into());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// The SAPIENT wordmark banner, shown at the top of interactive sessions.
pub fn print_logo() {
    let bar = style("━".repeat(52)).dim();
    println!("{bar}");
    println!(
        "  {} {}   {}",
        BOLT,
        style("SAPIENT").bold().cyan(),
        style("edge inference engine").dim()
    );
    println!("{bar}");
}

pub fn print_chat_banner(model: &str, arch: &str, backend: &str) {
    let bar = style("━".repeat(52)).dim();
    println!("\n{bar}");
    println!(
        "  {} {}",
        style("SAPIENT Chat").bold().cyan(),
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

pub fn write_user_prompt() -> io::Result<()> {
    // Modern "chip" badge for the role, then an inline prompt arrow.
    write!(
        io::stdout(),
        "\n{} {} ",
        style(" you ").black().on_green().bold(),
        style(ARROW).green().dim()
    )?;
    io::stdout().flush()
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

pub fn failure(msg: impl std::fmt::Display) {
    eprintln!("{} {}", style(CROSS).red().bold(), msg);
}

pub fn hint(msg: impl std::fmt::Display) {
    println!("{} {}", style(INFO).cyan(), style(msg).dim());
}

/// Render a simple aligned table with a dim header rule.
pub fn print_table(headers: &[&str], rows: &[Vec<String>]) {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    let mut header_line = String::from("  ");
    for (i, h) in headers.iter().enumerate() {
        header_line.push_str(&format!("{:<width$}  ", h, width = widths[i]));
    }
    println!("{}", style(header_line).bold());

    let rule: usize = widths.iter().map(|w| w + 2).sum::<usize>();
    println!("  {}", style("─".repeat(rule)).dim());

    for row in rows {
        let mut line = String::from("  ");
        for (i, cell) in row.iter().enumerate() {
            let w = widths.get(i).copied().unwrap_or(cell.len());
            line.push_str(&format!("{:<width$}  ", cell, width = w));
        }
        println!("{line}");
    }
}
