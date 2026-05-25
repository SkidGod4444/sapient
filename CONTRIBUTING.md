# Contributing to SAPIENT

Thank you for your interest in contributing! SAPIENT is a Rust-native LLM inference engine, and every bug report, doc fix, and code contribution helps.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [License](#license)
- [Before you start](#before-you-start)
- [Getting set up](#getting-set-up)
- [Project structure](#project-structure)
- [Development workflow](#development-workflow)
- [Testing](#testing)
- [Formatting and linting](#formatting-and-linting)
- [Commit messages](#commit-messages)
- [Pull requests](#pull-requests)
- [Common contribution areas](#common-contribution-areas)
- [Adding a CLI command](#adding-a-cli-command)
- [Adding a model architecture](#adding-a-model-architecture)
- [Hub and download testing](#hub-and-download-testing)
- [Releases (maintainers)](#releases-maintainers)
- [Getting help](#getting-help)

---

## Code of conduct

Be respectful, constructive, and inclusive. We want this project to be welcoming to contributors at every experience level. Harassment, spam, and bad-faith arguments are not tolerated.

---

## License

SAPIENT is licensed under **[GPL-3.0-only](LICENSE)**.

By contributing code, documentation, or other materials, you agree that your contributions will be licensed under the same terms. If you modify SAPIENT and distribute binaries, those binaries must also comply with the GPL.

---

## Before you start

1. **Search existing issues** — someone may already be working on it.
2. **Open an issue first** for large changes (new backends, architecture rewrites, breaking API changes).
3. **Keep PRs focused** — one logical change per pull request is easier to review and merge.
4. **Prefer minimal diffs** — match existing style and conventions in the crate you are editing.

---

## Getting set up

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Rust | **1.75+** | Install via [rustup](https://rustup.rs/) |
| Git | any recent | |
| [just](https://github.com/casey/just) | optional | Task runner — see `justfile` |

### Clone and build

```bash
git clone https://github.com/SkidGod4444/sapient.git
cd sapient
cargo build --workspace
```

Run the CLI locally:

```bash
cargo run -p sapient-cli -- --help
cargo run -p sapient-cli -- chat <model>
```

Release binary (faster inference):

```bash
cargo build --release -p sapient-cli
./target/release/sapient --version
```

If you use `just`:

```bash
just build      # debug build
just release    # release build
just --list     # all tasks
```

---

## Project structure

SAPIENT is a Cargo workspace. Crates are layered from low-level primitives up to the user-facing API.

```
crates/
├── sapient-core/           # Tensors, dtypes, buffers
├── sapient-ir/             # Computation graph IR and passes
├── sapient-io/             # Safetensors, GGUF, ONNX loaders
├── sapient-backends/cpu/   # CPU inference kernels
├── sapient-backends/metal/ # Apple GPU backend (WIP)
├── sapient-scheduler/      # Request scheduling and batching
├── sapient-telemetry/      # Tracing and metrics
├── sapient-runtime/        # InferenceSession, Model
├── sapient-hub/            # HuggingFace Hub client
├── sapient-tokenizers/     # Tokenizers + chat templates
├── sapient-models/         # Forward engines (Llama, Phi, …)
├── sapient-generate/       # Pipeline API (from_pretrained, chat, stream)
└── sapient-cli/            # `sapient` binary

install.sh / install.ps1    # Install scripts (attached to releases)
Formula/sapient.rb          # Homebrew formula template
.github/workflows/          # CI and release automation
tests/                      # Workspace integration tests
```

**Dependency direction (simplified):**

```
sapient-cli → sapient-generate → sapient-hub, sapient-models, sapient-tokenizers
sapient-generate → sapient-runtime → sapient-backends-cpu, sapient-io, sapient-ir
```

When adding a dependency, keep layers acyclic — lower crates must not depend on higher ones.

---

## Development workflow

### 1. Create a branch

```bash
git checkout -b feat/my-feature
# or
git checkout -b fix/issue-123-download-lock
```

### 2. Make your changes

- Follow existing naming, error handling (`anyhow` in binaries, `thiserror` in libraries), and import style.
- Avoid unrelated refactors in the same PR.
- Add or update tests when behavior changes.
- Update `README.md` if user-facing CLI or install behavior changes.

### 3. Run checks locally

```bash
just fmt-check
just clippy
just test
```

Or without `just`:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -- -D warnings
cargo test --workspace
```

### 4. Push and open a PR

Push your branch and open a pull request against `main`. Fill in what changed, why, and how you tested it.

---

## Testing

### Run all tests

```bash
cargo test --workspace
```

### Run tests for one crate

```bash
cargo test -p sapient-hub
cargo test -p sapient-models
cargo test -p sapient-cli
```

### Hub integration tests (network)

Some Hub tests download public models and require network + write access to `~/.cache/huggingface/`:

```bash
cargo test -p sapient-hub --test download_parallel -- --test-threads=1
```

Run serially (`--test-threads=1`) when multiple tests touch the same cached model to avoid HF cache lock conflicts.

### Benchmarks

```bash
cargo bench -p sapient-backends-cpu
just bench
```

---

## Formatting and linting

CI enforces both. PRs that fail these checks will not merge.

```bash
cargo fmt --all              # auto-format
cargo fmt --all -- --check   # CI check (no writes)
cargo clippy --workspace --all-targets -- -D warnings
```

Rules of thumb:

- No `clippy` warnings — CI uses `-D warnings`.
- Prefer `debug!` / `tracing` for internal logs; keep CLI output user-friendly unless `-v` is passed.
- Do not commit secrets (`.env`, HF tokens, API keys).

---

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

[optional body]
```

Common types:

| Type | Use for |
|---|---|
| `feat` | New feature or CLI command |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Tooling, deps, CI |
| `ci` | CI workflow changes |

Examples:

```
feat: add sapient embed command for sentence vectors
fix: parse GitHub release tag_name correctly in install.sh
docs: document fast download env vars in README
ci(release): build Linux aarch64 on ARM runners
```

Keep the subject line under ~72 characters. Use the body for context, trade-offs, and breaking changes.

---

## Pull requests

### Checklist

Before requesting review, confirm:

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] User-facing changes are reflected in `README.md` (if applicable)
- [ ] No secrets or large binary blobs are included
- [ ] PR description explains **what**, **why**, and **how to test**

### Review expectations

- Maintainers may ask for tests, docs, or a smaller scope.
- Breaking changes need a clear migration note in the PR description.
- Installer changes (`install.sh`, `install.ps1`) should be tested manually on the target platform when possible.

### CI

Every PR runs:

- `cargo fmt --check`
- `cargo clippy` (warnings denied)
- `cargo test --workspace` on Linux and macOS (arm64 + x86_64)
- Windows release build of `sapient`
- Cross-compile check for `aarch64-unknown-linux-gnu`
- `cargo doc --workspace --no-deps`

---

## Common contribution areas

These are especially welcome:

| Area | Location | Notes |
|---|---|---|
| **Model architectures** | `crates/sapient-models/src/` | Llama, Phi, Gemma, Qwen, … |
| **CPU kernels** | `crates/sapient-backends/cpu/src/kernels/` | Attention, RoPE, matmul, … |
| **Metal / GPU backend** | `crates/sapient-backends/metal/` | Apple Silicon — WIP |
| **Hub client** | `crates/sapient-hub/` | Downloads, caching, auth |
| **CLI UX** | `crates/sapient-cli/` | Commands, terminal UI |
| **Tokenizers / chat templates** | `crates/sapient-tokenizers/` | HF tokenizer edge cases |
| **Docs & examples** | `README.md`, crate doc comments | |
| **Install scripts** | `install.sh`, `install.ps1` | Must work on fresh machines |

---

## Adding a CLI command

1. Add a variant to `Commands` in `crates/sapient-cli/src/main.rs` (use `clap` derive).
2. Wire it in the `match cli.command` block in `main()`.
3. Implement the handler function in `main.rs` or a submodule (`hub.rs`, `ui.rs`, …).
4. Add `--help` text via clap doc comments on the struct fields.
5. Update `README.md` and post-install help in `install.sh` / `install.ps1` if user-facing.

Example pattern:

```rust
/// Remove one cached model from this device.
#[command(name = "rm", visible_aliases = ["remove"])]
Rm {
    /// HuggingFace model ID to remove.
    model: String,
},
```

---

## Adding a model architecture

1. **Detect architecture** in `crates/sapient-hub/src/model_info.rs` — map HF `config.json` `model_type` to `ArchType`.
2. **Add forward engine** under `crates/sapient-models/src/forward/` if native inference is supported.
3. **Register weights loading** in `crates/sapient-models/src/weights.rs`.
4. **Wire Pipeline** in `crates/sapient-generate/src/pipeline.rs`.
5. **Add tests** in `crates/sapient-models/tests/` with a tiny fixture or mocked tensors where possible.

For text-only support of vision models, document limitations in CLI output rather than failing silently.

---

## Hub and download testing

Fast downloads are controlled by `LoadOptions` and env vars (see README **Fast Downloads** section):

```bash
SAPIENT_HUB_MAX_PARALLEL=4 sapient pull <model>
SAPIENT_FAST_DOWNLOAD=0 sapient pull <model>   # sequential mode
sapient -v pull <model>                        # verbose logs + file paths
```

When debugging Hub issues:

- Check `~/.cache/huggingface/hub/` for partial downloads (`.sync.part`, `.lock`).
- Use `sapient reset --stale` to clear incomplete downloads.
- Gated models require `sapient login` or `HF_TOKEN`.

---

## Releases (maintainers)

Releases are automated via GitHub Actions when a semver tag is pushed:

```bash
# 1. Bump version in root Cargo.toml (workspace.package.version)
#    and Formula/sapient.rb if needed

# 2. Commit, tag, push
git tag v0.1.x
git push origin main
git push origin v0.1.x
```

The [release workflow](.github/workflows/release.yml) builds binaries for:

- macOS (Apple Silicon + Intel)
- Linux (x86_64 + aarch64)
- Windows (x86_64 + ARM64)

and attaches `install.sh` / `install.ps1` to the GitHub Release.

Install URLs in docs should point to release assets:

```bash
curl -fsSL https://github.com/SkidGod4444/sapient/releases/latest/download/install.sh | sh
```

Do **not** use `raw.githubusercontent.com/.../main/install.sh` in user-facing docs — the CDN can serve stale scripts.

---

## Getting help

- **Bug reports & feature requests:** [GitHub Issues](https://github.com/SkidGod4444/sapient/issues)
- **Questions:** Open a Discussion or issue with the `question` label
- **Security issues:** Do not open public issues — contact maintainers privately

When filing a bug, include:

- OS and architecture (`uname -a` or Windows version)
- `sapient --version` output
- Full command you ran
- Expected vs actual behavior
- Relevant logs (`sapient -v …` for verbose output)

---

Thank you for helping make local LLM inference faster and more accessible.
