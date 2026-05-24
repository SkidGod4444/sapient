#!/usr/bin/env sh
# SAPIENT Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/SkidGod4444/sapient/main/install.sh | sh
#
# Supported platforms:
#   macOS  — Apple Silicon (arm64) + Intel (x86_64)
#   Linux  — x86_64, aarch64 (glibc 2.17+, covers Ubuntu 18+, Debian 10+, RHEL 7+)
#   WSL    — treated as Linux

set -e

REPO="SkidGod4444/sapient"
BINARY_NAME="sapient"
INSTALL_DIR="${SAPIENT_INSTALL_DIR:-/usr/local/bin}"

# ── Colours (stderr so `BINARY=$(download)` only captures the path) ───────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

print_banner() {
  printf "\n${BOLD}${CYAN}  SAPIENT${RESET}\n" >&2
  printf "  ${BOLD}LLM & SLM Inference Engine${RESET}\n\n" >&2
}

info()    { printf "${CYAN}  →${RESET} %s\n" "$1" >&2; }
success() { printf "${GREEN}  ✓${RESET} %s\n" "$1" >&2; }
warn()    { printf "${YELLOW}  ⚠${RESET} %s\n" "$1" >&2; }
error()   { printf "${RED}  ✗ ERROR:${RESET} %s\n" "$1" >&2; exit 1; }

is_interactive() {
  [ -t 0 ] && [ -t 1 ]
}

# ── Detect OS and arch ───────────────────────────────────────────────────────
detect_platform() {
  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "$OS" in
    Darwin)
      case "$ARCH" in
        arm64)  PLATFORM="aarch64-apple-darwin" ;;
        x86_64) PLATFORM="x86_64-apple-darwin" ;;
        *)      error "Unsupported macOS architecture: $ARCH" ;;
      esac
      EXT="tar.gz"
      ;;
    Linux)
      case "$ARCH" in
        x86_64)  PLATFORM="x86_64-unknown-linux-gnu" ;;
        aarch64) PLATFORM="aarch64-unknown-linux-gnu" ;;
        armv7l)  PLATFORM="armv7-unknown-linux-gnueabihf" ;;
        *)       error "Unsupported Linux architecture: $ARCH" ;;
      esac
      EXT="tar.gz"
      ;;
    MINGW*|MSYS*|CYGWIN*)
      error "Please use the PowerShell installer on Windows:\n  irm https://raw.githubusercontent.com/SkidGod4444/sapient/main/install.ps1 | iex"
      ;;
    *)
      error "Unsupported OS: $OS"
      ;;
  esac
}

# ── Get latest release version ───────────────────────────────────────────────
get_latest_version() {
  if command -v curl > /dev/null 2>&1; then
    VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
      | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
  elif command -v wget > /dev/null 2>&1; then
    VERSION=$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" \
      | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
  else
    error "Neither curl nor wget found. Please install one and retry."
  fi

  if [ -z "$VERSION" ]; then
    error "Could not determine latest version. Check your internet connection."
  fi
}

hash_file() {
  FILE="$1"
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$FILE" | awk '{print $1}'
  elif command -v shasum > /dev/null 2>&1; then
    shasum -a 256 "$FILE" | awk '{print $1}'
  else
    echo ""
  fi
}

# ── Download ─────────────────────────────────────────────────────────────────
download() {
  FILENAME="${BINARY_NAME}-${PLATFORM}.${EXT}"
  URL="https://github.com/${REPO}/releases/download/${VERSION}/${FILENAME}"
  TMPDIR="$(mktemp -d)"
  TMPFILE="${TMPDIR}/${FILENAME}"

  info "Downloading ${BINARY_NAME} ${VERSION} for ${PLATFORM}..."

  if command -v curl > /dev/null 2>&1; then
    curl -fsSL --progress-bar "$URL" -o "$TMPFILE" || error "Download failed. URL: $URL"
  else
    wget -q --show-progress "$URL" -O "$TMPFILE" || error "Download failed. URL: $URL"
  fi

  # Verify checksum of the downloaded archive (matches release .sha256 files)
  CHECKSUM_URL="${URL}.sha256"
  if command -v curl > /dev/null 2>&1; then
    EXPECTED=$(curl -fsSL "$CHECKSUM_URL" 2>/dev/null | awk '{print $1}')
  else
    EXPECTED=$(wget -qO- "$CHECKSUM_URL" 2>/dev/null | awk '{print $1}')
  fi

  if [ -n "$EXPECTED" ]; then
    ACTUAL=$(hash_file "$TMPFILE")
    if [ -n "$ACTUAL" ] && [ "$ACTUAL" != "$EXPECTED" ]; then
      error "Checksum mismatch! Expected: $EXPECTED  Got: $ACTUAL\nDownload may be corrupted."
    fi
    success "Checksum verified"
  fi

  info "Extracting..."
  case "$EXT" in
    tar.gz) tar -xzf "$TMPFILE" -C "$TMPDIR" ;;
    zip)    unzip -q "$TMPFILE" -d "$TMPDIR" ;;
  esac

  EXTRACTED_BINARY="${TMPDIR}/${BINARY_NAME}"
  if [ ! -f "$EXTRACTED_BINARY" ]; then
    EXTRACTED_BINARY="$(find "$TMPDIR" -name "${BINARY_NAME}" -type f | head -1)"
  fi

  [ -f "$EXTRACTED_BINARY" ] || error "Binary not found after extraction"

  chmod +x "$EXTRACTED_BINARY"
  echo "$EXTRACTED_BINARY"
}

# ── Install ───────────────────────────────────────────────────────────────────
install_binary() {
  EXTRACTED_BINARY="$1"
  FINAL_PATH=""
  USED_LOCAL_BIN=0

  if [ -n "${SAPIENT_INSTALL_DIR:-}" ]; then
    mkdir -p "$INSTALL_DIR"
  fi

  if [ -w "$INSTALL_DIR" ]; then
    cp "$EXTRACTED_BINARY" "${INSTALL_DIR}/${BINARY_NAME}"
    FINAL_PATH="${INSTALL_DIR}/${BINARY_NAME}"
  elif is_interactive && command -v sudo > /dev/null 2>&1; then
    info "Requesting sudo to install to ${INSTALL_DIR}..."
    sudo cp "$EXTRACTED_BINARY" "${INSTALL_DIR}/${BINARY_NAME}"
    sudo chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
    FINAL_PATH="${INSTALL_DIR}/${BINARY_NAME}"
  else
    LOCAL_BIN="${HOME}/.local/bin"
    mkdir -p "$LOCAL_BIN"
    cp "$EXTRACTED_BINARY" "${LOCAL_BIN}/${BINARY_NAME}"
    FINAL_PATH="${LOCAL_BIN}/${BINARY_NAME}"
    USED_LOCAL_BIN=1
    if ! is_interactive; then
      warn "Non-interactive install — using ${LOCAL_BIN} (sudo skipped)"
    else
      warn "Installed to ${LOCAL_BIN} (no write access to ${INSTALL_DIR})"
    fi
    case ":${PATH}:" in
      *":${LOCAL_BIN}:"*) ;;
      *)
        warn "Add to your PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
        warn "Then restart your terminal, or run: hash -r"
        ;;
    esac
  fi

  success "Installed to ${FINAL_PATH}"
  INSTALLED_PATH="$FINAL_PATH"
}

# ── Post-install ──────────────────────────────────────────────────────────────
post_install() {
  printf "\n${BOLD}${GREEN}✓ SAPIENT ${VERSION} installed successfully!${RESET}\n\n"
  if ! command -v "${BINARY_NAME}" > /dev/null 2>&1; then
    printf "  ${YELLOW}Note:${RESET} ${BINARY_NAME} is not on your PATH yet.\n"
    printf "  Run: ${BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}\n"
    printf "  Or:  ${BOLD}${INSTALLED_PATH} --version${RESET}\n\n"
  fi
  printf "  Run your first model:\n\n"
  printf "    ${BOLD}sapient chat microsoft/phi-2${RESET}\n\n"
  printf "  Other useful commands:\n"
  printf "    ${CYAN}sapient pull TheBloke/Llama-2-7B-GGUF${RESET}   # Download a model\n"
  printf "    ${CYAN}sapient list${RESET}                             # List cached models\n"
  printf "    ${CYAN}sapient serve microsoft/phi-2 --port 8080${RESET} # Start API server\n"
  printf "    ${CYAN}sapient --help${RESET}                           # Full help\n\n"
  printf "  Docs: https://github.com/${REPO}\n\n"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  print_banner
  detect_platform
  get_latest_version
  BINARY=$(download)
  install_binary "$BINARY"
  post_install

  rm -rf "$(dirname "$BINARY")"
}

main "$@"
