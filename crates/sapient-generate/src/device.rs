// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Device capability detection and backend recommendation.
//!
//! Detects CPUs, GPUs, and memory on macOS (Apple Silicon, Intel) and Windows
//! (NVIDIA/AMD discrete GPUs via DXGI/wmic). Recommends the fastest backend
//! combination for a given model size and reports estimated throughput.

use std::process::Command;

// ── Public types ──────────────────────────────────────────────────────────────

/// Full picture of the host device's compute capabilities.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub cpu: CpuInfo,
    pub gpus: Vec<GpuInfo>,
    pub ram_bytes: u64,
    /// True when CPU and GPU share the same physical memory (Apple Silicon UMA).
    pub unified_memory: bool,
}

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub name: String,
    pub logical_cores: usize,
    /// Apple Silicon performance cores (0 on non-Apple or unknown).
    pub performance_cores: usize,
    /// Apple Silicon efficiency cores (0 on non-Apple or unknown).
    pub efficiency_cores: usize,
    /// Rough memory bandwidth estimate in GB/s (0 = unknown).
    pub bandwidth_gbps: f64,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    /// Dedicated VRAM in bytes. None = shared/unified memory (Apple Silicon).
    pub vram_bytes: Option<u64>,
    pub apis: Vec<ComputeApi>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeApi {
    Metal,
    Cuda,
    Vulkan,
    DirectX12,
    OpenCL,
}

/// The backend(s) to use for a given model.
#[derive(Debug, Clone)]
pub enum BackendPlan {
    /// Run everything on the CPU using NEON/AVX2 kernels.
    Cpu,
    /// Run everything on the Metal GPU via MLX (Apple Silicon only).
    Metal,
    /// Run the first `gpu_layers` transformer layers on Metal, the rest on CPU.
    /// Only used when the model doesn't fit entirely in the Metal memory budget.
    MetalCpuSplit {
        gpu_layers: usize,
        total_layers: usize,
    },
    /// CUDA backend (future — not yet implemented).
    Cuda,
}

impl BackendPlan {
    /// Human-readable label for display.
    pub fn label(&self) -> String {
        match self {
            Self::Cpu => "CPU (NEON/AVX2)".into(),
            Self::Metal => "Metal GPU (full model)".into(),
            Self::MetalCpuSplit {
                gpu_layers,
                total_layers,
            } => {
                format!("Metal+CPU hybrid  ({gpu_layers}/{total_layers} layers on GPU)")
            }
            Self::Cuda => "CUDA GPU".into(),
        }
    }

    /// Rough tok/s estimate for a model with `model_bytes` weights.
    pub fn estimated_tps(&self, model_bytes: u64, profile: &DeviceProfile) -> f64 {
        // Very rough bandwidth-based estimate: tok/s ≈ bandwidth / bytes_per_token
        // Weight bytes per decode step ≈ model_bytes (reading all weights once).
        let bw_gbps = match self {
            Self::Cpu => profile.cpu.bandwidth_gbps,
            Self::Metal => {
                // Apple Silicon memory bandwidth (much higher via GPU fabric)
                // M1: 68, M1 Pro: 200, M1 Max: 400, M2: 100, M2 Pro: 200, M3/M4: similar
                // We use RAM as a proxy: more RAM → higher-tier chip → higher bandwidth
                let gb = profile.ram_bytes as f64 / 1e9;
                if gb >= 96.0 {
                    400.0
                } else if gb >= 36.0 {
                    300.0
                } else if gb >= 24.0 {
                    200.0
                } else {
                    100.0
                }
            }
            Self::MetalCpuSplit {
                gpu_layers,
                total_layers,
            } => {
                let gpu_frac = *gpu_layers as f64 / *total_layers as f64;
                let metal_bw = 150.0f64.min(profile.ram_bytes as f64 / 1e9 * 8.0);
                let cpu_bw = profile.cpu.bandwidth_gbps;
                gpu_frac * metal_bw + (1.0 - gpu_frac) * cpu_bw
            }
            Self::Cuda => {
                profile
                    .gpus
                    .iter()
                    .find(|g| g.apis.contains(&ComputeApi::Cuda))
                    .and_then(|g| g.vram_bytes)
                    .map(|v| (v as f64 / 1e9) * 50.0) // rough: 50 tok/s per GB VRAM
                    .unwrap_or(50.0)
            }
        };

        if bw_gbps <= 0.0 || model_bytes == 0 {
            return 0.0;
        }
        (bw_gbps * 1e9 / model_bytes as f64).max(0.1)
    }
}

// ── Detection ─────────────────────────────────────────────────────────────────

/// Detect all device capabilities on the current host.
pub fn detect() -> DeviceProfile {
    let cpu = detect_cpu();
    let gpus = detect_gpus();
    let ram_bytes = detect_ram();
    let unified_memory = is_unified_memory();
    DeviceProfile {
        cpu,
        gpus,
        ram_bytes,
        unified_memory,
    }
}

/// Recommend the best backend plan for a model of the given size.
/// `total_layers` is the transformer depth (used for the hybrid split).
pub fn recommend(profile: &DeviceProfile, model_bytes: u64, total_layers: usize) -> BackendPlan {
    // ── Metal (Apple Silicon) ─────────────────────────────────────────────────
    let has_metal = profile
        .gpus
        .iter()
        .any(|g| g.apis.contains(&ComputeApi::Metal));
    if has_metal && profile.unified_memory {
        // Reserve 2 GB for OS, require 1.5× KV-cache headroom.
        let budget = profile.ram_bytes.saturating_sub(2 * 1024 * 1024 * 1024);
        let needed = (model_bytes as f64 * 1.5) as u64;

        if needed <= budget {
            return BackendPlan::Metal;
        }

        // Partial fit: calculate how many layers we can run on Metal.
        if total_layers > 0 && model_bytes < profile.ram_bytes {
            let bytes_per_layer = model_bytes / total_layers as u64;
            let gpu_layers =
                ((budget as f64 / (bytes_per_layer as f64 * 1.5)) as usize).min(total_layers);
            if gpu_layers > total_layers / 4 {
                return BackendPlan::MetalCpuSplit {
                    gpu_layers,
                    total_layers,
                };
            }
        }
    }

    // ── CUDA (NVIDIA) ─────────────────────────────────────────────────────────
    let cuda_gpu = profile
        .gpus
        .iter()
        .find(|g| g.apis.contains(&ComputeApi::Cuda));
    if let Some(gpu) = cuda_gpu {
        if let Some(vram) = gpu.vram_bytes {
            if model_bytes < vram * 9 / 10 {
                return BackendPlan::Cuda;
            }
        }
    }

    BackendPlan::Cpu
}

// ── OS-specific detection helpers ─────────────────────────────────────────────

fn detect_cpu() -> CpuInfo {
    let name = cpu_name();
    let logical_cores = logical_cpu_count();
    let (perf_cores, eff_cores) = apple_core_split();
    let bandwidth_gbps = estimate_cpu_bandwidth_gbps(logical_cores, &name);

    CpuInfo {
        name,
        logical_cores,
        performance_cores: perf_cores,
        efficiency_cores: eff_cores,
        bandwidth_gbps,
    }
}

fn cpu_name() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Some(s) = sysctl_str("machdep.cpu.brand_string") {
            return s;
        }
        // Apple Silicon CPUs report chip via hw.model, not the cpu brand string
        if let Some(model) = sysctl_str("hw.model") {
            return model;
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(s) = wmic_query("cpu get name /value") {
            if let Some(v) = parse_wmic_value(&s, "Name") {
                return v;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in info.lines() {
                if let Some(rest) = line.strip_prefix("model name") {
                    if let Some(val) = rest.split(':').nth(1) {
                        return val.trim().to_string();
                    }
                }
            }
        }
    }
    "Unknown CPU".to_string()
}

fn logical_cpu_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = sysctl_u64("hw.logicalcpu") {
            return n as usize;
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(v) = std::env::var("NUMBER_OF_PROCESSORS") {
            if let Ok(n) = v.parse::<usize>() {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Returns (performance_cores, efficiency_cores) for Apple Silicon.
/// Returns (0, 0) for non-Apple or when detection fails.
fn apple_core_split() -> (usize, usize) {
    #[cfg(target_os = "macos")]
    {
        let p = sysctl_u64("hw.perflevel0.logicalcpu").unwrap_or(0) as usize;
        let e = sysctl_u64("hw.perflevel1.logicalcpu").unwrap_or(0) as usize;
        if p + e > 0 {
            return (p, e);
        }
    }
    (0, 0)
}

fn estimate_cpu_bandwidth_gbps(cores: usize, name: &str) -> f64 {
    let name_lower = name.to_lowercase();
    // Apple Silicon: use RAM size (already fetched separately) as a proxy.
    // Tier: base (M1/M2/M3/M4) ≈ 68-100 GB/s, Pro ≈ 200, Max ≈ 400, Ultra ≈ 800
    if name_lower.contains("apple")
        || name_lower.contains("m1")
        || name_lower.contains("m2")
        || name_lower.contains("m3")
        || name_lower.contains("m4")
    {
        return if name_lower.contains("ultra") {
            800.0
        } else if name_lower.contains("max") {
            400.0
        } else if name_lower.contains("pro") {
            200.0
        } else {
            100.0
        };
    }
    // x86: rough per-core bandwidth (typically 20-50 GB/s total for desktop)
    (cores.min(8) as f64 * 5.0).max(20.0)
}

fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // ── macOS: Metal via system_profiler + sysctl ─────────────────────────────
    #[cfg(target_os = "macos")]
    {
        let metal_available = {
            #[cfg(all(target_os = "macos", feature = "mlx"))]
            {
                true
            }
            #[cfg(not(all(target_os = "macos", feature = "mlx")))]
            {
                // Check if we're on Apple Silicon (Metal always available there)
                cfg!(target_arch = "aarch64")
            }
        };

        if metal_available {
            let name = macos_gpu_name();
            gpus.push(GpuInfo {
                name,
                vram_bytes: None, // Apple UMA — shared with CPU RAM
                apis: vec![ComputeApi::Metal],
            });
        }

        // Check for NVIDIA eGPU (rare but possible)
        if let Some(nvidia) = detect_nvidia_macos() {
            gpus.push(nvidia);
        }
    }

    // ── Windows: DXGI / wmic ──────────────────────────────────────────────────
    #[cfg(target_os = "windows")]
    {
        for gpu in detect_windows_gpus() {
            gpus.push(gpu);
        }
    }

    // ── Linux: detect NVIDIA/AMD via /proc or nvidia-smi ─────────────────────
    #[cfg(target_os = "linux")]
    {
        for gpu in detect_linux_gpus() {
            gpus.push(gpu);
        }
    }

    gpus
}

#[cfg(target_os = "macos")]
fn macos_gpu_name() -> String {
    // system_profiler SPDisplaysDataType prints GPU info; parse Chipset Model line.
    let out = Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output();
    if let Ok(out) = out {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(val) = trimmed
                .strip_prefix("Chipset Model:")
                .or_else(|| trimmed.strip_prefix("GPU:"))
            {
                return val.trim().to_string();
            }
        }
    }
    // Fallback: derive from chip model
    sysctl_str("hw.model").unwrap_or_else(|| "Apple Silicon GPU".to_string())
}

#[cfg(target_os = "macos")]
fn detect_nvidia_macos() -> Option<GpuInfo> {
    // nvidia-smi presence = eGPU connected
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next()?;
    let parts: Vec<&str> = line.splitn(2, ',').collect();
    let name = parts.first()?.trim().to_string();
    let vram = parts
        .get(1)
        .and_then(|v| v.trim().strip_suffix(" MiB"))
        .and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024);
    Some(GpuInfo {
        name,
        vram_bytes: vram,
        apis: vec![ComputeApi::Cuda],
    })
}

#[cfg(target_os = "windows")]
fn detect_windows_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // Query all display adapters via wmic
    let out = Command::new("wmic")
        .args([
            "path",
            "win32_VideoController",
            "get",
            "Name,AdapterRAM,DriverVersion",
            "/format:csv",
        ])
        .output();

    if let Ok(out) = out {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines().skip(2) {
            // skip header + blank line
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 3 {
                continue;
            }
            let name = cols.get(2).unwrap_or(&"").trim().to_string();
            if name.is_empty() {
                continue;
            }

            let vram = cols
                .get(1)
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|&v| v > 0);

            let name_lower = name.to_lowercase();
            let mut apis = Vec::new();
            if name_lower.contains("nvidia") {
                // Check for CUDA: try nvidia-smi
                if Command::new("nvidia-smi").output().is_ok() {
                    apis.push(ComputeApi::Cuda);
                }
                apis.push(ComputeApi::Vulkan);
                apis.push(ComputeApi::DirectX12);
            } else if name_lower.contains("amd") || name_lower.contains("radeon") {
                apis.push(ComputeApi::Vulkan);
                apis.push(ComputeApi::DirectX12);
            } else if name_lower.contains("intel") {
                apis.push(ComputeApi::DirectX12);
                apis.push(ComputeApi::Vulkan);
            }

            if !apis.is_empty() {
                gpus.push(GpuInfo {
                    name,
                    vram_bytes: vram,
                    apis,
                });
            }
        }
    }
    gpus
}

#[cfg(target_os = "linux")]
fn detect_linux_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // NVIDIA via nvidia-smi
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();
    if let Ok(out) = out {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let parts: Vec<&str> = line.splitn(2, ',').collect();
            let name = parts
                .first()
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let vram = parts
                .get(1)
                .and_then(|v| v.trim().parse::<u64>().ok())
                .map(|mb| mb * 1024 * 1024);
            gpus.push(GpuInfo {
                name,
                vram_bytes: vram,
                apis: vec![ComputeApi::Cuda, ComputeApi::Vulkan],
            });
        }
    }

    // AMD/Intel via lspci (fallback)
    if gpus.is_empty() {
        let out = Command::new("lspci").output();
        if let Ok(out) = out {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let low = line.to_lowercase();
                if low.contains("vga") || low.contains("3d controller") || low.contains("display") {
                    let name = line
                        .splitn(2, ':')
                        .last()
                        .unwrap_or(line)
                        .trim()
                        .to_string();
                    let mut apis = vec![ComputeApi::Vulkan];
                    if low.contains("nvidia") {
                        apis.push(ComputeApi::Cuda);
                    }
                    gpus.push(GpuInfo {
                        name,
                        vram_bytes: None,
                        apis,
                    });
                }
            }
        }
    }

    gpus
}

fn detect_ram() -> u64 {
    #[cfg(target_os = "macos")]
    if let Some(n) = sysctl_u64("hw.memsize") {
        return n;
    }
    #[cfg(target_os = "linux")]
    if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
        for line in info.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                if let Ok(kb) = rest.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    if let Some(s) = wmic_query("OS get TotalVisibleMemorySize /value") {
        if let Some(kb_str) = parse_wmic_value(&s, "TotalVisibleMemorySize") {
            if let Ok(kb) = kb_str.parse::<u64>() {
                return kb * 1024;
            }
        }
    }
    0
}

fn is_unified_memory() -> bool {
    // Apple Silicon always has unified memory.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return true;
    #[allow(unreachable_code)]
    false
}

// ── Utility helpers ───────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn sysctl_str(key: &str) -> Option<String> {
    let out = Command::new("sysctl").args(["-n", key]).output().ok()?;
    let s = std::str::from_utf8(&out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn sysctl_u64(key: &str) -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        sysctl_str(key)?.parse().ok()
    }
    #[cfg(target_os = "linux")]
    {
        let out = Command::new("sysctl").args(["-n", key]).output().ok()?;
        std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()
    }
}

#[cfg(target_os = "windows")]
fn wmic_query(args: &str) -> Option<String> {
    let parts: Vec<&str> = args.split_whitespace().collect();
    let out = Command::new("wmic").args(&parts).output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

#[cfg(target_os = "windows")]
fn parse_wmic_value(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        if let Some(val) = line.strip_prefix(&format!("{key}=")) {
            let v = val.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

// ── Formatting ────────────────────────────────────────────────────────────────

impl DeviceProfile {
    /// Render a human-readable device report.
    pub fn report(&self) -> String {
        let mut out = String::new();
        let gb = |b: u64| b as f64 / 1e9;

        // CPU
        out.push_str(&format!("  CPU   {}\n", self.cpu.name));
        if self.cpu.performance_cores > 0 {
            out.push_str(&format!(
                "        {} cores  ({} performance + {} efficiency)\n",
                self.cpu.logical_cores, self.cpu.performance_cores, self.cpu.efficiency_cores
            ));
        } else {
            out.push_str(&format!(
                "        {} logical cores\n",
                self.cpu.logical_cores
            ));
        }
        if self.ram_bytes > 0 {
            out.push_str(&format!(
                "        {:.0} GB RAM{}  ·  est. {:.0} GB/s bandwidth\n",
                gb(self.ram_bytes),
                if self.unified_memory { " unified" } else { "" },
                self.cpu.bandwidth_gbps
            ));
        }

        // GPUs
        for gpu in &self.gpus {
            let apis: Vec<&str> = gpu
                .apis
                .iter()
                .map(|a| match a {
                    ComputeApi::Metal => "Metal",
                    ComputeApi::Cuda => "CUDA",
                    ComputeApi::Vulkan => "Vulkan",
                    ComputeApi::DirectX12 => "DX12",
                    ComputeApi::OpenCL => "OpenCL",
                })
                .collect();
            let vram_str = match gpu.vram_bytes {
                Some(b) => format!("  {:.0} GB VRAM", gb(b)),
                None => "  shared memory".to_string(),
            };
            out.push_str(&format!(
                "\n  GPU   {}\n        {}  ·  [{}]\n",
                gpu.name,
                vram_str.trim(),
                apis.join(", ")
            ));
        }

        out
    }

    /// Render model-size-specific backend recommendations.
    pub fn recommendations(&self) -> String {
        let mut out = String::new();
        out.push_str("\n  Recommended backends:\n");

        let scenarios = [
            ("0.5B Q8", 0.5e9 * 1.06, 24),
            ("1.5B Q8", 1.5e9 * 1.06, 28),
            ("3B Q4", 3.0e9 * 0.5625, 32),
            ("7B Q4", 7.0e9 * 0.5625, 32),
            ("14B Q4", 14.0e9 * 0.5625, 40),
            ("32B Q4", 32.0e9 * 0.5625, 64),
        ];

        for (label, bytes, layers) in scenarios {
            let model_bytes = bytes as u64;
            let plan = recommend(self, model_bytes, layers);
            let tps = plan.estimated_tps(model_bytes, self);
            let tps_str = if tps > 0.5 {
                format!("~{:.0} tok/s", tps)
            } else {
                "slow (model too large)".to_string()
            };
            out.push_str(&format!(
                "    {label:<12}  →  {:<36}  {tps_str}\n",
                plan.label()
            ));
        }
        out
    }
}
