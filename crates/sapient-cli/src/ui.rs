//! Terminal UI helpers for interactive commands.

use std::io::{self, IsTerminal, Write};

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";

fn use_color() -> bool {
    std::env::var("NO_COLOR").is_err() && io::stdout().is_terminal()
}

fn dim() -> &'static str {
    if use_color() { DIM } else { "" }
}

fn bold() -> &'static str {
    if use_color() { BOLD } else { "" }
}

fn cyan() -> &'static str {
    if use_color() { CYAN } else { "" }
}

fn reset() -> &'static str {
    if use_color() { RESET } else { "" }
}

/// Clear the current stderr line (spinner / status text).
pub fn clear_status() {
    let _ = write!(io::stderr(), "\r\x1b[2K");
    let _ = io::stderr().flush();
}

/// Show a single-line loading status on stderr (no paths or log noise).
pub fn show_loading(message: &str) {
    let _ = write!(io::stderr(), "\r{}…{} ", message, reset());
    let _ = io::stderr().flush();
}

pub fn print_chat_banner(model: &str, arch: &str) {
    let line = "─".repeat(48);
    println!(
        "\n{}{}{}\n{}  {}SAPIENT Chat{}\n{}  Model: {}{}\n{}  Type /exit or /help{}",
        dim(),
        line,
        reset(),
        dim(),
        cyan(),
        reset(),
        dim(),
        model,
        reset(),
        dim(),
        reset(),
    );
    if arch.contains("vision") || model.to_ascii_lowercase().contains("vlm") {
        println!(
            "{}  Note: vision models run in text-only mode for now{}",
            dim(),
            reset()
        );
    }
    println!("{}{}{}\n", dim(), line, reset());
}

pub fn print_chat_help() {
    println!("\n{}Commands:{} /exit  /quit  /help\n", bold(), reset());
}

pub fn write_user_prompt() -> io::Result<()> {
    write!(
        io::stdout(),
        "\n{}you{} {}›{} ",
        bold(),
        reset(),
        cyan(),
        reset()
    )?;
    io::stdout().flush()
}

pub fn write_assistant_prompt() -> io::Result<()> {
    write!(
        io::stdout(),
        "\n{}sapient{} {}›{} ",
        bold(),
        reset(),
        cyan(),
        reset()
    )?;
    io::stdout().flush()
}
