# Concerns & Technical Debt

**Analysis Date:** 2026-02-05

## High Priority

### Design vs Implementation Gap

The `docs/design/DESIGN.md` describes a comprehensive layered architecture (foundation, core engine, orchestration, API layers) with ~20 modules. The actual implementation is a single utility file with 4 functions. This gap needs a clear roadmap to bridge.

- **Design**: TextStore, TextProcessor, DiffComputer, MetricsEngine, AnalysisCoordinator, DiffOrchestrator, plugin system, config builder, cache manager
- **Reality**: `bert_v1.rs` with model loading + embedding computation
- **Risk**: Design may be over-engineered relative to what gets built; design decisions may need revision as implementation reveals constraints

### Duplicated Device Selection Logic

`model_all_minilm_l6_v2()` and `model_all_mpnet_base_v2()` contain identical device selection code (~10 lines each). This should be extracted into a shared function.

- **Location**: `crates/core/src/utils/bert_v1.rs:9-19` and `crates/core/src/utils/bert_v1.rs:54-64`
- **Impact**: Maintenance burden, inconsistency risk

### Dead Code Warnings

All 4 functions in `bert_v1.rs` are flagged as unused (`dead_code` warning):
- `model_all_minilm_l6_v2` (line 8)
- `model_all_mpnet_base_v2` (line 53)
- `get_embeddings` (line 100)
- `cosine_similarity` (line 166)

Functions are private and only called from `#[cfg(test)]`. They need `pub` visibility or `#[allow(dead_code)]` to suppress warnings.

### No Public API Surface

`lib.rs` only declares `mod utils` - no `pub use` re-exports. The crate has no usable public API. Consumers can't import anything meaningful.

## Medium Priority

### Empty Placeholder File

`crates/core/src/utils/bert_v2.rs` is declared in `mod.rs` but completely empty. This causes a dead module in the crate.

### Unsafe Code

`bert_v1.rs` uses `unsafe` for memory-mapped safetensors loading:

```rust
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)?
};
```

This is standard candle usage but should be documented with a safety comment explaining why it's sound.

### No Structured Logging

Using `println!` for all output. The design calls for instrumentation and observability. No `tracing` or `log` framework configured.

### No CI/CD

No GitHub Actions or CI pipeline. Relies entirely on local git hooks (rustfmt, clippy via devenv). No automated testing on push/PR.

## Low Priority

### Missing Documentation

- No rustdoc (`///`) comments on any functions
- No README.md in project root
- Design docs exist but are in `docs/design/` (previously were in `plan/` based on git status showing deleted files)

### Commented-Out Dependencies

Several dependencies are commented out in `Cargo.toml`:
- ndarray, linfa, linfa-clustering (statistical analysis)
- pyo3 (Python bindings)
- criterion (benchmarks)

These represent planned features that haven't been started. Should be tracked as roadmap items rather than commented code.

### WASM Target Declared but Unused

`rust-toolchain.toml` declares `wasm32-unknown-unknown` as a build target, but no WASM-related code or configuration exists. May cause unnecessary toolchain downloads.

## Security

- No secrets or API keys in code (HF Hub uses public endpoints)
- `unsafe` block for mmaped file loading (standard candle pattern)
- `devenv.nix` has a `protect-secrets` hook preventing edits to `.env`/`.secret` files
- secretspec configured with keyring provider in `devenv.yaml`

## Performance

- Model loading is synchronous and blocks (design calls for async)
- No caching of loaded models between calls
- Release profile has LTO + codegen-units=1 (good for production, slow builds)

---

*Concerns analysis: 2026-02-05*
