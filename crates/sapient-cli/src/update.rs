//! Self-update: download the latest release binary from GitHub.

use std::fs;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

const REPO: &str = "SkidGod4444/sapient";
const BINARY: &str = "sapient";

struct PlatformAsset {
    triple: &'static str,
    archive: ArchiveKind,
}

/// Which build of SAPIENT to install.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Variant {
    Cpu,
    Metal,
}

impl Variant {
    /// Filename suffix on the release archive (`-metal` for the GPU build).
    fn suffix(self) -> &'static str {
        match self {
            Variant::Cpu => "",
            Variant::Metal => "-metal",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Variant::Cpu => "CPU",
            Variant::Metal => "Metal (GPU)",
        }
    }
}

/// The variant this binary was compiled as. A binary built with the `mlx`
/// feature is the Apple Silicon Metal build; everything else is CPU.
const fn current_variant() -> Variant {
    if cfg!(feature = "mlx") {
        Variant::Metal
    } else {
        Variant::Cpu
    }
}

/// Whether this machine can run the Metal build: Apple Silicon (macOS aarch64),
/// the only platform we publish a `-metal` artifact for.
fn metal_capable() -> bool {
    std::env::consts::OS == "macos" && std::env::consts::ARCH == "aarch64"
}

#[derive(Copy, Clone)]
enum ArchiveKind {
    TarGz,
    Zip,
}

fn platform_asset() -> Result<PlatformAsset> {
    let asset = match std::env::consts::OS {
        "macos" => match std::env::consts::ARCH {
            "aarch64" => PlatformAsset {
                triple: "aarch64-apple-darwin",
                archive: ArchiveKind::TarGz,
            },
            "x86_64" => PlatformAsset {
                triple: "x86_64-apple-darwin",
                archive: ArchiveKind::TarGz,
            },
            other => bail!("unsupported macOS architecture: {other}"),
        },
        "linux" => match std::env::consts::ARCH {
            "x86_64" => PlatformAsset {
                triple: "x86_64-unknown-linux-gnu",
                archive: ArchiveKind::TarGz,
            },
            "aarch64" => PlatformAsset {
                triple: "aarch64-unknown-linux-gnu",
                archive: ArchiveKind::TarGz,
            },
            other => bail!("unsupported Linux architecture: {other}"),
        },
        "windows" => match std::env::consts::ARCH {
            "x86_64" => PlatformAsset {
                triple: "x86_64-pc-windows-msvc",
                archive: ArchiveKind::Zip,
            },
            "aarch64" => PlatformAsset {
                triple: "aarch64-pc-windows-msvc",
                archive: ArchiveKind::Zip,
            },
            other => bail!("unsupported Windows architecture: {other}"),
        },
        other => bail!("self-update is not supported on {other}"),
    };
    Ok(asset)
}

fn parse_version(tag: &str) -> Option<(u32, u32, u32)> {
    let tag = tag.strip_prefix('v').unwrap_or(tag);
    let mut parts = tag.split('.');
    Some((
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    ))
}

fn version_lt(a: &str, b: &str) -> bool {
    match (parse_version(a), parse_version(b)) {
        (Some(av), Some(bv)) => av < bv,
        _ => a != b,
    }
}

fn http_get_bytes(url: &str) -> Result<Vec<u8>> {
    let resp = ureq::get(url)
        .set("User-Agent", "sapient-cli")
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn fetch_latest_tag() -> Result<String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let body = ureq::get(&url)
        .set("User-Agent", "sapient-cli")
        .call()
        .with_context(|| format!("failed to fetch {url}"))?
        .into_string()?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    json["tag_name"]
        .as_str()
        .map(String::from)
        .context("releases/latest response missing tag_name")
}

fn verify_sha256(data: &[u8], expected_hex: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    let actual = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
    if actual != expected_hex.trim().to_ascii_lowercase() {
        bail!("checksum mismatch: expected {expected_hex}, got {actual}");
    }
    Ok(())
}

fn extract_tar_gz(data: &[u8], dest: &Path) -> Result<PathBuf> {
    let decoder = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(dest).context("failed to extract tar.gz")?;

    let binary = dest.join(BINARY);
    if binary.is_file() {
        return Ok(binary);
    }
    bail!("binary '{BINARY}' not found in archive")
}

fn extract_zip(data: &[u8], dest: &Path) -> Result<PathBuf> {
    let reader = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(reader).context("failed to read zip")?;
    archive.extract(dest).context("failed to extract zip")?;
    let binary = dest.join(format!("{BINARY}.exe"));
    if binary.is_file() {
        return Ok(binary);
    }
    bail!("binary '{BINARY}.exe' not found in archive")
}

fn download_release(asset: &PlatformAsset, tag: &str, variant: Variant) -> Result<PathBuf> {
    let suffix = variant.suffix();
    let filename = match asset.archive {
        ArchiveKind::TarGz => format!("{BINARY}-{}{suffix}.tar.gz", asset.triple),
        ArchiveKind::Zip => format!("{BINARY}-{}{suffix}.zip", asset.triple),
    };
    let url = format!("https://github.com/{REPO}/releases/download/{tag}/{filename}");
    println!("Downloading {tag} ({}, {})...", asset.triple, variant.label());

    let archive_bytes = http_get_bytes(&url)?;

    if let Ok(checksum_text) = http_get_bytes(&format!("{url}.sha256")) {
        let expected = String::from_utf8_lossy(&checksum_text)
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        if !expected.is_empty() {
            verify_sha256(&archive_bytes, &expected)?;
            println!("Checksum verified.");
        }
    }

    let tmp = tempfile::tempdir().context("failed to create temp dir")?;
    let extracted = match asset.archive {
        ArchiveKind::TarGz => extract_tar_gz(&archive_bytes, tmp.path())?,
        ArchiveKind::Zip => extract_zip(&archive_bytes, tmp.path())?,
    };

    let staged = std::env::temp_dir().join(format!("sapient-update-{tag}"));
    let _ = fs::remove_file(&staged);
    fs::copy(&extracted, &staged)
        .with_context(|| format!("failed to stage {}", staged.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&staged, fs::Permissions::from_mode(0o755))?;
    }

    Ok(staged)
}

fn replace_current_binary(new_binary: &Path) -> Result<()> {
    let current = std::env::current_exe().context("could not locate current sapient binary")?;

    #[cfg(windows)]
    {
        let backup = current.with_extension("exe.old");
        let _ = fs::remove_file(&backup);
        if current.exists() {
            fs::rename(&current, &backup).ok();
        }
        fs::copy(new_binary, &current).with_context(|| {
            format!(
                "failed to install update to {}. \
                 Try: irm https://raw.githubusercontent.com/{REPO}/main/install.ps1 | iex",
                current.display()
            )
        })?;
        return Ok(());
    }

    #[cfg(unix)]
    {
        fs::copy(new_binary, &current)
            .with_context(|| format!("failed to install update to {}", current.display()))?;
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&current, fs::Permissions::from_mode(0o755))?;
        Ok(())
    }
}

/// Decide which build variant to install.
///
/// Order of precedence: explicit `--metal`/`--cpu` flag → interactive prompt on
/// Apple Silicon → the variant this binary was built as. On machines without a
/// Metal artifact, always CPU (and reject an explicit `--metal`).
fn resolve_variant(explicit: Option<Variant>) -> Result<Variant> {
    if !metal_capable() {
        if explicit == Some(Variant::Metal) {
            bail!("the Metal build is only available on Apple Silicon (macOS arm64)");
        }
        return Ok(Variant::Cpu);
    }

    if let Some(v) = explicit {
        return Ok(v);
    }

    // Apple Silicon, no explicit choice: ask if we're attached to a terminal,
    // otherwise keep whatever this binary already is.
    if std::io::stdin().is_terminal() {
        Ok(prompt_variant(current_variant()))
    } else {
        Ok(current_variant())
    }
}

/// Ask the user which build to install, defaulting to `default` on empty input.
fn prompt_variant(default: Variant) -> Variant {
    use std::io::Write;
    println!("You're on Apple Silicon — choose a build to install:");
    println!("  1) Metal (GPU) — faster on Apple Silicon");
    println!("  2) CPU         — maximum compatibility");
    let default_num = match default {
        Variant::Metal => 1,
        Variant::Cpu => 2,
    };
    print!("Build [1/2] (default {default_num}, currently {}): ", default.label());
    let _ = std::io::stdout().flush();

    let mut buf = String::new();
    if std::io::stdin().read_line(&mut buf).is_err() {
        return default;
    }
    match buf.trim() {
        "1" => Variant::Metal,
        "2" => Variant::Cpu,
        _ => default,
    }
}

/// Check for updates and install the latest release if available.
pub fn run_update(force: bool, variant: Option<Variant>) -> Result<()> {
    let current = env!("CARGO_PKG_VERSION");
    let latest_tag = fetch_latest_tag()?;
    let latest = latest_tag.strip_prefix('v').unwrap_or(&latest_tag);

    let target_variant = resolve_variant(variant)?;
    let switching = target_variant != current_variant();

    // If already current AND not switching build variant, nothing to do.
    if !force && !switching && !version_lt(current, latest) {
        println!("sapient {current} ({}) is already up to date.", current_variant().label());
        return Ok(());
    }

    if switching {
        println!(
            "Switching build: {} → {}",
            current_variant().label(),
            target_variant.label()
        );
    }
    println!("Updating sapient {current} → {latest}...");
    let asset = platform_asset()?;
    let staged = download_release(&asset, &latest_tag, target_variant)?;
    replace_current_binary(&staged)?;
    let _ = fs::remove_file(&staged);

    println!("✓ Updated to sapient {latest} ({})", target_variant.label());
    println!("  Run: sapient --version");
    Ok(())
}
