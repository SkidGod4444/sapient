//! `sapient stats` — a live monitor of the machine resources SAPIENT is using.
//!
//! Aggregates every running `sapient` process (chat / serve / converse / speak /
//! transcribe), shows per-core CPU load (flagging the hottest core), the SAPIENT
//! processes' combined CPU + RSS, system memory pressure, and on-disk footprint
//! (model cache + binary). Refreshes ~1 Hz in place until Ctrl-C.

use std::io::Write;
use std::time::Duration;

use anyhow::Result;
use console::style;
use sysinfo::{ProcessesToUpdate, System, MINIMUM_CPU_UPDATE_INTERVAL};

use crate::hub::{format_bytes, hub_cache_size};

pub async fn run() -> Result<()> {
    let mut sys = System::new_all();
    let exe_size = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    // Scanning the cache dir is slow; refresh it only every ~10 ticks.
    let mut cache_size = hub_cache_size();
    let mut tick = 0u64;

    print!("\x1b[?25l"); // hide cursor
    let _ = std::io::stdout().flush();

    loop {
        // CPU/process usage needs two samples spaced by the minimum interval.
        sys.refresh_cpu_all();
        sys.refresh_processes(ProcessesToUpdate::All, true);
        tokio::time::sleep(MINIMUM_CPU_UPDATE_INTERVAL).await;
        sys.refresh_cpu_all();
        sys.refresh_memory();
        sys.refresh_processes(ProcessesToUpdate::All, true);

        if tick % 10 == 0 {
            cache_size = hub_cache_size();
        }
        tick += 1;

        render(&sys, cache_size, exe_size);

        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = tokio::time::sleep(Duration::from_millis(900)) => {}
        }
    }

    println!("\x1b[?25h"); // restore cursor
    let _ = std::io::stdout().flush();
    Ok(())
}

fn render(sys: &System, cache_size: u64, exe_size: u64) {
    let mut o = String::new();
    o.push_str("\x1b[2J\x1b[H"); // clear + cursor home

    o.push_str(&format!(
        "{}  {}\n\n",
        style(format!("SAPIENT stats · v{}", env!("CARGO_PKG_VERSION")))
            .bold()
            .cyan(),
        style("live — Ctrl-C to exit").dim()
    ));

    // ── SAPIENT processes (aggregate of every `sapient` process) ──────────────
    let me = sysinfo::get_current_pid().ok();
    let (mut pcpu, mut pmem, mut count, mut others) = (0.0f32, 0u64, 0u32, 0u32);
    for (pid, p) in sys.processes() {
        if p.name()
            .to_string_lossy()
            .to_lowercase()
            .contains("sapient")
        {
            pcpu += p.cpu_usage();
            pmem += p.memory();
            count += 1;
            if Some(*pid) != me {
                others += 1;
            }
        }
    }
    o.push_str(&format!("{}\n", style("sapient processes").bold()));
    o.push_str(&format!(
        "  {:<16}{} {}\n",
        style("running").dim(),
        count,
        style(format!("({others} besides this monitor)")).dim()
    ));
    o.push_str(&format!("  {:<16}{:.1}%\n", style("cpu (sum)").dim(), pcpu));
    o.push_str(&format!(
        "  {:<16}{}\n\n",
        style("memory (rss)").dim(),
        format_bytes(pmem)
    ));

    // ── system memory ─────────────────────────────────────────────────────────
    let (used, total) = (sys.used_memory(), sys.total_memory().max(1));
    o.push_str(&format!("{}\n", style("system memory").bold()));
    o.push_str(&format!(
        "  {} {} / {}\n\n",
        bar(used as f32 / total as f32, 28),
        format_bytes(used),
        format_bytes(total)
    ));

    // ── per-core CPU (flag the hottest) ───────────────────────────────────────
    let cpus = sys.cpus();
    o.push_str(&format!(
        "{} {}\n",
        style("cpu cores").bold(),
        style(format!(
            "({} logical · total {:.0}%)",
            cpus.len(),
            sys.global_cpu_usage()
        ))
        .dim()
    ));
    let hottest = cpus
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.cpu_usage().total_cmp(&b.1.cpu_usage()))
        .map(|(i, _)| i);
    for (i, c) in cpus.iter().enumerate() {
        let u = c.cpu_usage();
        let line = format!("  c{:<2} {} {:>5.1}%", i, bar(u / 100.0, 22), u);
        if Some(i) == hottest && u > 40.0 {
            o.push_str(&format!(
                "{} {}\n",
                style(line).yellow(),
                style("◀ hottest").yellow().bold()
            ));
        } else {
            o.push_str(&format!("{line}\n"));
        }
    }
    o.push('\n');

    // ── storage ───────────────────────────────────────────────────────────────
    o.push_str(&format!("{}\n", style("storage").bold()));
    o.push_str(&format!(
        "  {:<16}{}\n",
        style("model cache").dim(),
        format_bytes(cache_size)
    ));
    o.push_str(&format!(
        "  {:<16}{}\n",
        style("binary").dim(),
        format_bytes(exe_size)
    ));

    print!("{o}");
    let _ = std::io::stdout().flush();
}

/// A coloured `[████····]` bar for a 0..1 fraction (green→yellow→red).
fn bar(frac: f32, width: usize) -> String {
    let frac = frac.clamp(0.0, 1.0);
    let filled = (frac * width as f32).round() as usize;
    let fill = "█".repeat(filled);
    let rest = "·".repeat(width.saturating_sub(filled));
    let fill = if frac > 0.85 {
        style(fill).red()
    } else if frac > 0.6 {
        style(fill).yellow()
    } else {
        style(fill).green()
    };
    format!("[{}{}]", fill, style(rest).dim())
}
