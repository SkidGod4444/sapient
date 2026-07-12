//! Thermal-aware sustained decode (roadmap Phase 8.4).
//!
//! Passively-cooled boards (Raspberry Pi 4/5) hit their thermal limit under
//! sustained decode; the firmware then hard-throttles every core's clock and
//! throughput collapses. Backing off *before* the trip point sustains a higher
//! steady-state: this module reads the Linux thermal zones
//! (`/sys/class/thermal/thermal_zone*/temp`, millidegrees) and lowers the
//! **effective parallelism target** the GEMV chunker uses — fewer, larger rayon
//! tasks leave cores idle (rayon's global pool cannot be resized at runtime, but
//! splitting work into fewer tasks than threads idles the rest), cutting package
//! power so the clocks stay up.
//!
//! Behaviour: one temperature sample at most every [`TICK_MS`] (the per-matmul
//! `tick()` is otherwise a single atomic load). Hysteresis steps the effective
//! thread target down by one core at/above the hot threshold (default 80 °C,
//! Pi 5 firmware throttles at 85 °C) and back up at/below the cool threshold
//! (default 70 °C). The floor is half the cores — graceful degradation, never
//! collapse. On machines with no thermal zones (macOS, Windows, containers) the
//! governor is inert and `effective_threads()` is exactly
//! `rayon::current_num_threads()`.
//!
//! Env: `SAPIENT_THERMAL=off` disables; `SAPIENT_THERMAL_HOT` / `_COOL` set the
//! thresholds in °C; `SAPIENT_THERMAL_PATH` overrides the sysfs root (tests).
//!
//! **Mobile (roadmap 11.3):** iOS and Android expose no sysfs — the host app
//! observes the OS thermal signal (`ProcessInfo.thermalState` /
//! `PowerManager.addThermalStatusListener`) and feeds it in through
//! [`set_external_thermal_level`] (exported over `sapient-ffi` as
//! `set_thermal_level`). The external level caps the same effective-thread
//! target the sysfs governor steps, so every downstream consumer (GEMV
//! chunking, the spin-pool gate) reacts identically; when both sources are
//! active the stricter one wins.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Minimum interval between temperature reads.
const TICK_MS: u64 = 500;
/// Default backoff threshold, °C (Pi firmware throttles at 85).
const DEFAULT_HOT_C: i64 = 80;
/// Default recovery threshold, °C.
const DEFAULT_COOL_C: i64 = 70;

/// Hysteresis governor over a set of sysfs thermal zones. Pure state machine —
/// constructed directly in tests with a fake sysfs root; the process-wide
/// singleton (built from env) lives behind [`tick`]/[`effective_threads`].
pub struct ThermalGovernor {
    /// `…/thermal_zone*/temp` files found under the root at construction.
    zones: Vec<PathBuf>,
    hot_mc: i64,
    cool_mc: i64,
    max_threads: usize,
    min_threads: usize,
    effective: AtomicUsize,
    warned: AtomicBool,
}

impl ThermalGovernor {
    /// Scan `root` for `thermal_zone*/temp` files. `hot_c`/`cool_c` are in °C;
    /// `max_threads` is the full parallelism to restore to when cool.
    pub fn new(root: &Path, hot_c: i64, cool_c: i64, max_threads: usize) -> Self {
        let mut zones = Vec::new();
        if let Ok(entries) = std::fs::read_dir(root) {
            for e in entries.flatten() {
                let name = e.file_name();
                if name.to_string_lossy().starts_with("thermal_zone") {
                    let temp = e.path().join("temp");
                    if temp.is_file() {
                        zones.push(temp);
                    }
                }
            }
        }
        zones.sort();
        let max_threads = max_threads.max(1);
        Self {
            zones,
            hot_mc: hot_c * 1000,
            cool_mc: cool_c * 1000,
            max_threads,
            min_threads: (max_threads / 2).max(1),
            effective: AtomicUsize::new(max_threads),
            warned: AtomicBool::new(false),
        }
    }

    /// True when the machine exposes at least one thermal zone.
    pub fn is_active(&self) -> bool {
        !self.zones.is_empty()
    }

    /// Hottest zone in millidegrees, or `None` when nothing is readable.
    pub fn max_temp_mc(&self) -> Option<i64> {
        self.zones
            .iter()
            .filter_map(|p| std::fs::read_to_string(p).ok())
            .filter_map(|s| s.trim().parse::<i64>().ok())
            .max()
    }

    /// Current effective thread target.
    pub fn effective(&self) -> usize {
        self.effective.load(Ordering::Relaxed)
    }

    /// Take one temperature sample and step the target: −1 core at/above hot
    /// (floored at half the cores), +1 at/below cool (capped at full). Between
    /// the thresholds the target holds (hysteresis). Returns the new target.
    pub fn sample(&self) -> usize {
        let Some(t) = self.max_temp_mc() else {
            return self.effective();
        };
        let cur = self.effective();
        if t >= self.hot_mc && cur > self.min_threads {
            let next = cur - 1;
            self.effective.store(next, Ordering::Relaxed);
            if !self.warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "thermal: {:.1} °C ≥ {} °C — backing decode off to {next}/{} threads \
                     to sustain clocks (set SAPIENT_THERMAL=off to disable)",
                    t as f64 / 1000.0,
                    self.hot_mc / 1000,
                    self.max_threads
                );
            }
            next
        } else if t <= self.cool_mc && cur < self.max_threads {
            let next = cur + 1;
            self.effective.store(next, Ordering::Relaxed);
            next
        } else {
            cur
        }
    }
}

/// Host-OS thermal pressure fed by an embedding app (mobile FFI). Follows the
/// 4-level shape both mobile OSes expose: 0 nominal, 1 fair/moderate,
/// 2 serious/severe, 3 critical.
static EXTERNAL_LEVEL: AtomicU8 = AtomicU8::new(0);

/// Thread cap for an external thermal level over `max` cores. Nominal runs
/// full; fair sheds a quarter; serious halves (the sysfs governor's floor);
/// critical quarters — on mobile, critical means the OS is about to act
/// (forced sleep / shutdown), so dropping below the sysfs floor is deliberate.
fn external_cap(level: u8, max: usize) -> usize {
    match level {
        0 => max,
        1 => (max * 3 / 4).max(1),
        2 => (max / 2).max(1),
        _ => (max / 4).max(1),
    }
}

/// Feed the host OS's thermal state into the governor (levels > 3 clamp to
/// critical). Cheap and thread-safe — call it straight from the OS callback.
pub fn set_external_thermal_level(level: u8) {
    if thermal_disabled() {
        return;
    }
    let level = level.min(3);
    let prev = EXTERNAL_LEVEL.swap(level, Ordering::Relaxed);
    if prev != level {
        tracing::info!(
            "thermal: external level {prev} → {level}; effective decode threads now {}",
            effective_threads()
        );
    }
}

/// The last level fed to [`set_external_thermal_level`] (0 when never set).
pub fn external_thermal_level() -> u8 {
    EXTERNAL_LEVEL.load(Ordering::Relaxed)
}

/// `SAPIENT_THERMAL=off` disables BOTH the sysfs governor and the external
/// level — one escape hatch for the whole mechanism.
fn thermal_disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("SAPIENT_THERMAL").is_ok_and(|v| v.eq_ignore_ascii_case("off"))
    })
}

fn governor() -> Option<&'static ThermalGovernor> {
    static GOV: OnceLock<Option<ThermalGovernor>> = OnceLock::new();
    GOV.get_or_init(|| {
        if thermal_disabled() {
            return None;
        }
        let root = std::env::var("SAPIENT_THERMAL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/sys/class/thermal"));
        let parse_c = |var: &str, default: i64| {
            std::env::var(var)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default)
        };
        let gov = ThermalGovernor::new(
            &root,
            parse_c("SAPIENT_THERMAL_HOT", DEFAULT_HOT_C),
            parse_c("SAPIENT_THERMAL_COOL", DEFAULT_COOL_C),
            rayon::current_num_threads().max(1),
        );
        gov.is_active().then_some(gov)
    })
    .as_ref()
}

/// The parallelism target GEMV chunking should size tasks for: the full rayon
/// thread count, reduced while the sysfs governor is backing off and/or a
/// host-fed external level caps it (the stricter source wins). Cheap (two
/// atomic loads) — called per matmul.
pub fn effective_threads() -> usize {
    let max = rayon::current_num_threads().max(1);
    let base = match governor() {
        Some(g) => g.effective(),
        None => max,
    };
    base.min(external_cap(EXTERNAL_LEVEL.load(Ordering::Relaxed), max))
}

/// Rate-limited thermal sample: at most one sysfs read per [`TICK_MS`]; all
/// other calls are a single atomic compare. Call from hot-path entry points
/// (the matmul dispatcher does).
pub fn tick() {
    let Some(g) = governor() else { return };
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    let now_ms = EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now_ms.saturating_sub(last) < TICK_MS {
        return;
    }
    if LAST_MS
        .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        g.sample();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_sysfs(dir: &Path, zone: usize, millideg: i64) {
        let z = dir.join(format!("thermal_zone{zone}"));
        std::fs::create_dir_all(&z).unwrap();
        std::fs::write(z.join("temp"), format!("{millideg}\n")).unwrap();
    }

    /// Unique fake-sysfs root per test — tests run in parallel, so a shared
    /// directory would let one test's zone files leak into another's governor.
    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "sapient-thermal-test-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn no_zones_is_inert() {
        let d = tmp("inert");
        let g = ThermalGovernor::new(&d, 80, 70, 8);
        assert!(!g.is_active());
        assert_eq!(g.sample(), 8, "no zones → full threads");
    }

    #[test]
    fn hot_steps_down_to_floor_and_cool_restores() {
        let d = tmp("steps");
        fake_sysfs(&d, 0, 85_000);
        let g = ThermalGovernor::new(&d, 80, 70, 4);
        assert!(g.is_active());
        assert_eq!(g.sample(), 3);
        assert_eq!(g.sample(), 2);
        assert_eq!(g.sample(), 2, "floor at half the cores — never collapses");

        fake_sysfs(&d, 0, 60_000);
        assert_eq!(g.sample(), 3);
        assert_eq!(g.sample(), 4);
        assert_eq!(g.sample(), 4, "capped at full threads");
    }

    #[test]
    fn hysteresis_holds_between_thresholds() {
        let d = tmp("hysteresis");
        fake_sysfs(&d, 0, 85_000);
        let g = ThermalGovernor::new(&d, 80, 70, 4);
        g.sample();
        assert_eq!(g.effective(), 3);
        fake_sysfs(&d, 0, 75_000); // between cool (70) and hot (80)
        assert_eq!(g.sample(), 3, "holds inside the hysteresis band");
    }

    #[test]
    fn hottest_zone_wins() {
        let d = tmp("hottest");
        fake_sysfs(&d, 0, 50_000);
        fake_sysfs(&d, 1, 90_000);
        let g = ThermalGovernor::new(&d, 80, 70, 4);
        assert_eq!(g.max_temp_mc(), Some(90_000));
        assert_eq!(g.sample(), 3, "backs off on the hottest zone");
    }

    #[test]
    fn external_cap_mapping() {
        // 8 cores: nominal full, fair 6, serious 4, critical 2.
        assert_eq!(external_cap(0, 8), 8);
        assert_eq!(external_cap(1, 8), 6);
        assert_eq!(external_cap(2, 8), 4);
        assert_eq!(external_cap(3, 8), 2);
        // Never below one thread, even on tiny core counts.
        assert_eq!(external_cap(3, 1), 1);
        assert_eq!(external_cap(2, 1), 1);
        // Levels past critical clamp to the critical cap.
        assert_eq!(external_cap(7, 8), 2);
    }

    /// The global external level caps `effective_threads()` and releasing it
    /// restores full parallelism. Serialized within this one test (the level
    /// is process-global); no other test reads `effective_threads()`.
    #[test]
    fn external_level_caps_effective_threads() {
        let max = rayon::current_num_threads().max(1);
        set_external_thermal_level(0);
        assert_eq!(effective_threads(), max);
        set_external_thermal_level(3);
        assert_eq!(effective_threads(), (max / 4).max(1));
        assert_eq!(external_thermal_level(), 3);
        // Clamped, not wrapped.
        set_external_thermal_level(200);
        assert_eq!(external_thermal_level(), 3);
        set_external_thermal_level(0);
        assert_eq!(effective_threads(), max, "release restores full threads");
    }
}
