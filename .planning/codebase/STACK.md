# Technology Stack

**Analysis Date:** 2026-02-05

## Languages

- **Rust (Nightly)** - Core library language. Edition 2024. Toolchain managed via `rust-toolchain.toml`.
- **Python 3.14** - Available in devenv for future PyO3 bindings (not yet integrated into Cargo)
- **Nix** - Development environment via devenv (`devenv.nix`, `devenv.yaml`)

## Runtime

- Rust nightly toolchain, minimal profile
- Components: rustc, cargo, clippy, rustfmt, rust-analyzer, miri, rust-src
- Build target: `wasm32-unknown-unknown` (WebAssembly, declared but not yet used)
- Release profile: LTO enabled, codegen-units = 1

## Workspace

Cargo workspace with resolver v3. Single crate:

- `crates/core` - `redline-core` v0.1.0 (lib crate, `redline_core`)

## Dependencies

### Active

| Crate | Version | Purpose |
|-------|---------|---------|
| `candle-core` | 0.9.1 | Tensor computation engine (multi-backend) |
| `candle-nn` | 0.9.1 | Neural network layers |
| `candle-transformers` | 0.9.1 | Pre-trained transformer model support |
| `hf-hub` | 0.4.3 | HuggingFace Hub API for model downloads |
| `tokenizers` | 0.22.2 | Fast BERT-compatible tokenization |
| `unicode-normalization` | 0.1 | Unicode text normalization |
| `anyhow` | 1.0 | Error handling with context |
| `serde_json` | 1.0 | JSON serialization for model configs |
| `console` | 0.16 | Terminal output (optional feature) |

### Commented Out (Planned)

| Crate | Purpose |
|-------|---------|
| `ndarray`, `linfa`, `linfa-clustering` | Statistical analysis, threshold learning |
| `pyo3` | Python bindings via PyO3 |
| `criterion` | Benchmarking (dev dependency) |

## Feature Flags

| Flag | Effect |
|------|--------|
| `metal` | Apple Silicon GPU via candle Metal backend |
| `accelerate` | Apple Accelerate framework |
| `cuda` | NVIDIA GPU via candle CUDA backend |
| `mkl` | Intel MKL for CPU optimization |
| `default` | Empty (no features enabled by default) |

## Development Environment

Managed via **devenv** (Nix-based):

- **Packages**: git, jq, ripgrep, fd, tree, just, nixfmt, python314, uv, ruff
- **Languages**: Rust (from `rust-toolchain.toml`), Python 3.14, Nix
- **Git hooks** (pre-commit): nixfmt, rustfmt, shfmt, shellcheck, commitizen, ruff, action-validator
- **Diff tool**: difftastic enabled
- **Secrets**: secretspec with keyring provider
- **Nix inputs**: cachix/devenv-nixpkgs (rolling), nixpkgs-unstable, nixpkgs-python, rust-overlay

---

*Stack analysis: 2026-02-05*
