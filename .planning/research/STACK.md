# Stack Research

**Domain:** High-performance text diffing and analysis framework (Rust)
**Project:** Redline Core v1.0
**Researched:** 2026-02-05
**Confidence:** HIGH (core stack), MEDIUM (PyO3 ecosystem versions)

---

## Critical Findings: Spec Corrections Required

The technical spec (`docs/design/TECHNICAL_SPEC.md`) specifies dependency versions that are significantly outdated. The spec was written 2025-10-23. Here are the corrections that **must** be applied:

| Spec Dependency | Spec Version | Current Version | Breaking Change? |
|----------------|-------------|----------------|-----------------|
| `thiserror` | `1.0` | `2.0.18` | YES - v2.0 is a new major |
| `pyo3` | `0.20` | `0.28.0` | YES - massive API changes (Bound<T> API) |
| `pyo3-asyncio` | `0.20` | **DEPRECATED** | YES - replaced by `pyo3-async-runtimes 0.28.0` |
| `pythonize` | `0.20` | `0.27.0` (needs 0.28) | YES - must track PyO3 version |
| `hashbrown` | `0.14` | `0.16.1` | Minor - API compatible |
| `bumpalo` | `3.14` | `3.19.0` | No - semver compatible |
| `lru` | `0.12` | `0.16.0` | YES - several minor versions with API changes |
| `criterion` | `0.5` | `0.8.2` | YES - new major-ish release |
| `proptest` | `1.4` | `1.9.0` | No - semver compatible |
| `string-interner` | `0.15` | Available but consider custom instead | See analysis below |

---

## Recommended Stack

### Core Dependencies

| Crate | Version | Purpose | Why Recommended | Confidence |
|-------|---------|---------|-----------------|------------|
| **bumpalo** | `3.19` | Arena allocation for TextStore | The standard Rust bump allocator. WASM-compatible (`no_std`). Used by Servo/Dioxus for DOM allocation. The spec correctly identified this. Formally-verified `extend_from_slice` added in 3.17. | HIGH |
| **hashbrown** | `0.16` | Fast HashMap (SwissTable) for interner | SwissTable implementation backing `std::collections::HashMap`. Direct use gives access to raw entry API needed for zero-copy interning. WASM/`no_std` compatible. | HIGH |
| **smallvec** | `1.15` | Stack-allocated small vectors for Spans, MappingEntries | Avoids heap allocation for common small collections (e.g., `SmallVec<[Span; 1]>` for original_spans). Do NOT use 2.0-alpha -- still in alpha since Nov 2025. Features: `union`, `const_generics`. | HIGH |
| **thiserror** | `2.0` | Derive macro for structured error types | **Version 2.0, not 1.0 as spec says.** v2.0 is the current maintained line. Works with Rust edition 2024. Drop-in replacement for v1 usage patterns. | HIGH |
| **tokio** | `1.49` | Async runtime for parallel analyzer execution | The dominant Rust async runtime. Features needed: `rt`, `sync`, `time`, `macros`. Use `spawn_blocking` for CPU-bound diff work. LTS: 1.47.x until Sep 2026. | HIGH |
| **futures** | `0.3` | Async combinators (join, select, FutureExt) | Standard async utility crate. Needed for `join_all` on parallel analyzer futures. | HIGH |
| **lru** | `0.16` | LRU cache for MetricsEngine and CacheManager | **Version 0.16, not 0.12 as spec says.** 0.16 adds Clone for caches with custom hashers. O(1) put/get/pop. Simpler than `cached` or `moka` for this use case. | HIGH |
| **serde** | `1.0` | Serialization framework (optional feature) | Standard. Features: `derive`. Required for config serialization, cache keys, and WASM data exchange. | HIGH |
| **serde_json** | `1.0` | JSON serialization | Needed for config files, WASM bridge serialization. | HIGH |
| **tracing** | `0.1` | Structured diagnostics and instrumentation | **Missing from spec but essential.** The spec mentions observability as optional. Recommend making it a default feature. Integrates with tokio's tracing ecosystem. Critical for debugging async analyzer execution. | HIGH |
| **unicode-normalization** | `0.1` | Unicode NFC/NFD/NFKC/NFKD normalization | Already in Cargo.toml. Required for the Unicode normalizer in text processing layer. | HIGH |

### String Interning Strategy

The spec specifies `string-interner = "0.15"` and a hand-rolled `hashbrown::HashMap<&'static str, StringId>` interner in the TextStore. **Recommendation: Custom interner. Do NOT use `string-interner` or `lasso`.**

**Rationale:**
1. Strings live in the bumpalo arena -- their lifetime is tied to TextStore, not `'static`
2. StringId is a simple u32 newtype (Copy, 4 bytes) -- custom key type
3. Need tight integration with the arena allocator
4. No external interner crate cleanly supports "allocate in my arena, intern with my IDs"
5. The TextStore design in the spec IS already a custom interner -- just implement it

**Alternatives evaluated:**
- `string-interner 0.15`: Does not integrate with bumpalo. 145% memory overhead per benchmarks. Would require double allocation.
- `lasso 0.7.3`: Concurrent interner with `no_std` support. Lower memory overhead. But: last release >1 year ago, doesn't integrate with bumpalo arena. Would require awkward bridging.

### Python Bindings

| Crate | Version | Purpose | Why Recommended | Confidence |
|-------|---------|---------|-----------------|------------|
| **pyo3** | `0.28` | Rust-Python bindings | **Version 0.28, not 0.20 as spec says.** Uses `Bound<'py, T>` API (introduced 0.21), proper `__init__` support, abi3 subclassing. Supports Python 3.7-3.14. Tested against Python 3.14.0 final since 0.27. MSRV: Rust 1.83. Features: `extension-module`, `abi3-py39`, `multiple-pymethods`. | HIGH |
| **pyo3-async-runtimes** | `0.28` | Tokio <-> asyncio bridge | **Replaces `pyo3-asyncio` which is DEPRECATED and ARCHIVED.** This is the official successor maintained under PyO3 org. Features: `tokio-runtime`. Requires Python 3.9+. | HIGH |
| **pythonize** | `0.28`* | Serde <-> Python object conversion | Tracks PyO3 versions. v0.27 is latest published (Nov 2024, for PyO3 0.27). 0.28 release should land shortly. *Monitor for release -- may need to pin PyO3 to 0.27 if delayed.* | MEDIUM |
| **pyo3-stub-gen** | latest | Generate .pyi type stubs from PyO3 bindings | Actively maintained. Generates mypy/pyright-compatible stubs. Dev dependency only. | MEDIUM |
| **maturin** | `1.11` | Build system for Python wheels from Rust | Standard PyO3 build tool. Supports Python 3.8+. MSRV: Rust 1.85. Uses Rust 2024 edition. | HIGH |

**Critical PyO3 Version Decision:**

The spec says `pyo3 = "0.20"`. This is 8 major releases behind. The entire PyO3 API changed in 0.21 (GIL Refs -> Bound<T>). Every code example in the technical spec's Python bindings section uses the old 0.20 API and must be rewritten. Key changes:
- `&PyAny` -> `Bound<'py, PyAny>`
- `PyTuple::new(...)` -> `PyTuple::new_bound(...)`
- `pyo3_asyncio::tokio::future_into_py(...)` -> `pyo3_async_runtimes::tokio::future_into_py(...)`

**Python Version Floor:** The spec says "Python >= 3.8". However:
- `pyo3-async-runtimes` requires Python 3.9+
- Python 3.8 reached EOL October 2024
- Dev environment uses Python 3.14
- **Recommendation: Set minimum to Python 3.9.** Use `abi3-py39` feature flag.

### WASM Target

| Crate | Version | Purpose | Why Recommended | Confidence |
|-------|---------|---------|-----------------|------------|
| **wasm-bindgen** | `0.2` | JS interop for browser deployment | Standard Rust-WASM bridge. New maintainers (transferred from archived rustwasm org). Feature-gate behind `wasm` feature flag. | HIGH |
| **serde-wasm-bindgen** | `0.6` | Serde <-> JsValue conversion | Preferred over deprecated `wasm-bindgen` serde feature. Smaller code size, faster than JSON roundtrip. | HIGH |
| **js-sys** | `0.3` | Access to JS standard builtins | Needed for `Date.now()`, `Math.random()`. Minimal usage. | MEDIUM |
| **getrandom** | `0.3` | Random number generation | **WASM pitfall:** Do NOT enable `wasm_js` feature in the library crate. Only enable at the application/binary level. hashbrown uses deterministic hasher by default in WASM -- this is fine. | MEDIUM |

**WASM Architecture Notes:**
- `wasm-pack` was archived July 2025. Use `wasm-bindgen` CLI directly or `trunk` for builds.
- Feature-gate all WASM code behind `#[cfg(target_arch = "wasm32")]`
- Feature-gate all PyO3 code behind `#[cfg(feature = "python")]` so it compiles out for WASM
- `bumpalo`, `hashbrown`, `smallvec`, `lru`, `thiserror`, `serde` are all WASM-compatible
- `tokio` is NOT available in WASM. Async in WASM must use `wasm-bindgen-futures` instead.

### Development Dependencies

| Crate | Version | Purpose | Why Recommended | Confidence |
|-------|---------|---------|-----------------|------------|
| **proptest** | `1.9` | Property-based testing | Hypothesis-like testing for diff algorithms. MSRV 1.82. Essential for correctness. | HIGH |
| **criterion** | `0.8` | Microbenchmarking | **Version 0.8, not 0.5 as spec says.** MSRV: Rust 1.88. Statistical benchmarking for performance targets. | HIGH |
| **test-case** | `3.3` | Parameterized tests | Clean parameterized test syntax for normalizer/tokenizer variants. | HIGH |
| **tokio-test** | `0.4` | Async test utilities | Part of tokio project. `assert_ready!`, `assert_pending!` macros. | MEDIUM |

---

## Feature Flag Architecture

```toml
[features]
default = ["std"]

# Standard library support
std = []

# Async support (tokio-based, native targets only)
async = ["dep:tokio", "dep:futures"]

# Serialization support
serde = ["dep:serde", "dep:serde_json", "smallvec/serde"]

# Structured diagnostics
tracing = ["dep:tracing"]

# Python bindings (mutually exclusive with wasm at build time)
python = ["dep:pyo3", "dep:pyo3-async-runtimes", "async"]

# WASM target support
wasm = ["dep:wasm-bindgen", "dep:serde-wasm-bindgen", "dep:js-sys", "serde"]

# Full feature set (excluding wasm -- cannot coexist with python)
full = ["async", "serde", "tracing"]
```

**Key design:** `python` and `wasm` features are mutually exclusive at the build level. Enforce with:
```rust
#[cfg(all(feature = "python", target_arch = "wasm32"))]
compile_error!("The `python` feature cannot be used with WASM targets");
```

---

## Recommended Cargo.toml

```toml
[package]
name = "redline-core"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"

[dependencies]
# Foundation
bumpalo = { version = "3.19", features = ["collections"] }
hashbrown = { version = "0.16", default-features = false, features = ["default-hasher"] }
smallvec = { version = "1.15", features = ["union", "const_generics"] }

# Error handling
thiserror = "2.0"

# Unicode
unicode-normalization = "0.1"

# Caching
lru = "0.16"

# Async (optional)
tokio = { version = "1.49", optional = true, features = ["rt", "sync", "time", "macros"] }
futures = { version = "0.3", optional = true }

# Serialization (optional)
serde = { version = "1.0", optional = true, features = ["derive"] }
serde_json = { version = "1.0", optional = true }

# Diagnostics (optional)
tracing = { version = "0.1", optional = true }

# Python bindings (optional)
pyo3 = { version = "0.28", optional = true, features = ["extension-module", "abi3-py39", "multiple-pymethods"] }
pyo3-async-runtimes = { version = "0.28", optional = true, features = ["tokio-runtime"] }

# WASM (optional)
wasm-bindgen = { version = "0.2", optional = true }
serde-wasm-bindgen = { version = "0.6", optional = true }
js-sys = { version = "0.3", optional = true }

[dev-dependencies]
proptest = "1.9"
criterion = "0.8"
test-case = "3.3"
tokio = { version = "1.49", features = ["rt-multi-thread", "macros", "test-util"] }
tokio-test = "0.4"
serde_json = "1.0"

[features]
default = ["std"]
std = []
async = ["dep:tokio", "dep:futures"]
serde = ["dep:serde", "dep:serde_json", "smallvec/serde"]
tracing = ["dep:tracing"]
python = ["dep:pyo3", "dep:pyo3-async-runtimes", "async"]
wasm = ["dep:wasm-bindgen", "dep:serde-wasm-bindgen", "dep:js-sys", "serde"]
full = ["async", "serde", "tracing"]

[[bench]]
name = "diff_bench"
harness = false

[profile.release]
lto = true
codegen-units = 1
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Arena allocator | `bumpalo 3.19` | `typed-arena`, `bump-scope` | bumpalo is the ecosystem standard, WASM-proven. `bump-scope` benchmarks ~2x faster in WASM but is newer/less battle-tested. `typed-arena` is type-specific, not general. |
| Hash map | `hashbrown 0.16` | `std::collections::HashMap` | hashbrown gives raw entry API needed for zero-copy intern lookup. `std` HashMap wraps hashbrown anyway. |
| Small vec | `smallvec 1.15` | `tinyvec`, `arrayvec` | `smallvec` is the standard. `tinyvec` is `forbid(unsafe_code)` which is nice but has performance tradeoff. `arrayvec` is fixed-size only. |
| Error handling | `thiserror 2.0` | `anyhow`, `eyre`, `snafu` | `anyhow` is for applications not libraries. `thiserror` gives typed errors suitable for library API. `snafu` is more verbose. |
| Async runtime | `tokio 1.49` | `async-std`, `smol` | tokio is dominant. PyO3 async bridge is built for tokio. |
| LRU cache | `lru 0.16` | `moka`, `cached`, `quick-cache` | `lru` is simple. `moka` is concurrent-first (heavier) -- overkill with `Arc<RwLock<LruCache>>`. |
| String interning | Custom (bumpalo+hashbrown) | `string-interner`, `lasso` | Neither integrates with bumpalo arena. Custom is 50-80 lines for exactly what we need. |
| Python bridge | `pyo3 0.28` | `cpython`, `rust-cpython` | `pyo3` is the active standard. Others are unmaintained. |
| WASM serialization | `serde-wasm-bindgen 0.6` | `wasm-bindgen` serde feature | The serde feature on wasm-bindgen is deprecated. |
| Build system | `maturin 1.11` | `setuptools-rust` | `maturin` is PyO3-recommended. Zero-config for pure Rust extensions. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `anyhow` (in library code) | Application-level error handling. Erases error types. In current Cargo.toml from prototype. | `thiserror 2.0` |
| `pyo3-asyncio` | **Deprecated and archived.** No updates for PyO3 >= 0.21. | `pyo3-async-runtimes 0.28` |
| `pyo3 0.20` (as spec says) | Uses removed GIL Refs API. 8 versions behind. All spec code examples wrong. | `pyo3 0.28` |
| `wasm-pack` | **Archived July 2025.** rustwasm org sunset. | `wasm-bindgen` CLI directly |
| `smallvec 2.0-alpha` | Still alpha since Nov 2025. Breaking changes possible. | `smallvec 1.15` |
| `candle-*`, `hf-hub`, `tokenizers` | BERT prototype deps. Not in v1 scope. ML lives in Python plugins. | Remove entirely |
| `console` | Terminal formatting from prototype. Library should not do terminal output. | `tracing` |
| `string-interner` crate | Does not integrate with bumpalo arena. Memory overhead. | Custom TextStore interning |

---

## Version Compatibility Matrix

| Crate A | Version | Compatible With | Notes |
|---------|---------|-----------------|-------|
| `pyo3` | `0.28` | `pyo3-async-runtimes 0.28` | Must match major.minor |
| `pyo3` | `0.28` | `pythonize 0.28`* | *May not be released yet. Monitor. |
| `pyo3` | `0.28` | `maturin >= 1.11` | Supports all recent PyO3 |
| `pyo3` | `0.28` | Python 3.7 - 3.14 | Tested against 3.14 final |
| `pyo3-async-runtimes` | `0.28` | Python 3.9+ | Higher floor than PyO3 itself |
| `tokio` | `1.49` | `pyo3-async-runtimes 0.28` | Bridge built for tokio |
| `bumpalo` | `3.19` | WASM `wasm32-unknown-unknown` | `no_std` compatible |
| `hashbrown` | `0.16` | WASM `wasm32-unknown-unknown` | `no_std` compatible |
| `criterion` | `0.8` | Rust >= 1.88 | Dev-dependency only |
| `proptest` | `1.9` | Rust >= 1.82 | Dev-dependency only |
| `tokio` | `1.49` | NOT WASM | Must use `wasm-bindgen-futures` for WASM async |

---

## Stack Patterns by Variant

**Default build (library only):**
```bash
cargo build
# Features: std (default)
# Deps: bumpalo, hashbrown, smallvec, thiserror, unicode-normalization, lru
```

**With async support:**
```bash
cargo build --features async
# Adds: tokio, futures
```

**Python wheel build:**
```bash
maturin develop --features python
# Adds: pyo3, pyo3-async-runtimes, tokio, futures
```

**WASM build:**
```bash
cargo build --target wasm32-unknown-unknown --features wasm --no-default-features
# Adds: wasm-bindgen, serde-wasm-bindgen, js-sys, serde
# Excludes: tokio
```

---

## Prototype Cleanup Required

Remove from current `Cargo.toml` before v1 implementation:

**Remove from `[dependencies]`:**
- `candle-core`, `candle-nn`, `candle-transformers` (BERT prototype)
- `hf-hub` (model downloading)
- `tokenizers` (BERT tokenization)
- `anyhow` (application-level errors)
- `serde_json` (re-add as optional under `serde` feature)
- `console` (terminal formatting)

**Remove feature flags:**
- `metal`, `accelerate`, `cuda`, `mkl` (candle GPU features)

**Remove commented-out dependencies:**
- `ndarray`, `linfa`, `linfa-clustering` (not in v1 scope)

---

## Sources

### Verified (HIGH confidence)
- [bumpalo 3.19.0 - crates.io](https://crates.io/crates/bumpalo)
- [hashbrown 0.16.1 - crates.io](https://crates.io/crates/hashbrown)
- [smallvec 1.15.1 - crates.io](https://crates.io/crates/smallvec)
- [thiserror 2.0.18 - docs.rs](https://docs.rs/crate/thiserror/latest)
- [tokio 1.49.0 - crates.io](https://crates.io/crates/tokio)
- [lru 0.16.0 - docs.rs](https://docs.rs/crate/lru/latest/source/CHANGELOG.md)
- [PyO3 0.28.0 - GitHub Releases](https://github.com/pyo3/pyo3/releases)
- [pyo3-async-runtimes 0.28.0 - docs.rs](https://docs.rs/crate/pyo3-async-runtimes/latest)
- [pyo3-asyncio DEPRECATED - GitHub](https://github.com/davidhewitt/pyo3-asyncio)
- [wasm-bindgen 0.2.108 - GitHub Releases](https://github.com/rustwasm/wasm-bindgen/releases)
- [criterion 0.8.2 - docs.rs](https://docs.rs/crate/criterion/latest)
- [proptest 1.9.0 - docs.rs](https://docs.rs/crate/proptest/latest)
- [maturin 1.11.5 - GitHub](https://github.com/PyO3/maturin)
- [imara-diff 0.2.0 - docs.rs](https://docs.rs/crate/imara-diff/latest)

### Cross-referenced (MEDIUM confidence)
- [pythonize 0.27.0 - docs.rs](https://docs.rs/crate/pythonize/latest)
- [serde-wasm-bindgen 0.6.5 - docs.rs](https://docs.rs/crate/serde-wasm-bindgen/latest)
- [lasso 0.7.3 - crates.io](https://crates.io/crates/lasso)
- [tracing 0.1.41 - crates.io](https://crates.io/crates/tracing)
- [rustwasm org sunset - Rust Blog](https://blog.rust-lang.org/inside-rust/2025/07/21/sunsetting-the-rustwasm-github-org/)
- [PyO3 migration guide](https://pyo3.rs/main/migration.html)

---
*Stack research for: Redline Core v1.0*
*Researched: 2026-02-05*
