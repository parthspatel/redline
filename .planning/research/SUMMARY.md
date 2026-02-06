# Research Summary: Redline Core v1.0

> High-performance text diffing and analysis framework in Rust with Python bindings (PyO3) and WASM target

**Date:** 2026-02-06
**Status:** Research Complete
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md

---

## Executive Summary

Redline Core v1.0 is a greenfield build from comprehensive design documents. The existing codebase is a BERT embedding proof-of-concept that will be replaced entirely. The v1.0 effort builds a 4-layer architecture (Foundation -> Engine -> Orchestration -> API) providing text normalization with full traceability, pluggable tokenization, Myers/Histogram diff algorithms, a comprehensive metrics engine (20+ text metrics, 15+ pairwise), and an extensible analyzer plugin framework -- all with Python extensibility via PyO3 and browser deployment via WASM.

Research uncovered **critical spec bugs** that must be corrected before implementation: the Token struct cannot derive Copy with SmallVec (compile error), PyO3 version is 8 major releases behind (0.20 vs 0.28), the async bridge crate is deprecated, and WASM feature gating must be designed from Phase 1. The core architectural approach is validated, but specific data structure decisions need revision.

---

## Top Findings Across All Dimensions

### 1. The Technical Spec Has Critical Version Drift

The spec was written targeting PyO3 0.20. The current version is 0.28. The entire GIL Refs API was removed in 0.23, meaning **every Python binding code example in the spec is wrong**. Additionally:
- `pyo3-asyncio` is **deprecated and archived** -- replaced by `pyo3-async-runtimes`
- `thiserror` jumped to v2.0 (new major)
- `criterion` jumped to 0.8
- Python 3.8 reached EOL; minimum should be 3.9+
- `wasm-pack` was archived July 2025

### 2. Token Copy Semantics Are Broken in the Spec

The spec declares `Token` as `#[derive(Copy, Clone)]` but includes `original_spans: SmallVec<[Span; 1]>`. SmallVec does **not** implement Copy. This is a compile-time error that must be resolved in Phase 1.

**Recommended fix:** Replace SmallVec with a single `original_span: Span`. Most tokens map 1:1. Use CharMapping for rare multi-span lookups.

### 3. No Existing Library Combines Redline's Feature Set

The competitive analysis confirms that no existing Rust library combines:
- Diffing with normalization traceability (CharMapping through normalizer chains)
- Comprehensive metrics (20+ text, 15+ pairwise)
- Pluggable analyzer framework with dependency resolution
- Python bindings with plugin extensibility

This is Redline's genuine competitive advantage. Libraries like `similar`, `imara-diff`, `textdistance`, and `strsim` each cover at most 2-3 of these areas.

### 4. WASM and Tokio Are Fundamentally Incompatible

Tokio's `rt-multi-thread` does not compile to `wasm32-unknown-unknown`. WASM futures are `!Send`. Feature flag architecture must be designed from Phase 1 with CI verification:
- `python` and `wasm` features are mutually exclusive
- `async` feature gates all tokio usage
- `#[cfg(target_arch = "wasm32")]` for WASM-specific code paths

### 5. Custom String Interning Is Correct

Research confirms the spec's TextStore design (bumpalo arena + hashbrown HashMap) is the right approach. External crates (`string-interner`, `lasso`) don't integrate with bumpalo and would require double allocation. The custom interner is ~50-80 lines of code.

---

## Critical Spec Corrections Needed Before Implementation

| # | Issue | Correction | Severity |
|---|-------|-----------|----------|
| 1 | `Token` has SmallVec field but derives Copy | Replace `original_spans: SmallVec<[Span; 1]>` with `original_span: Span` | BLOCKING |
| 2 | PyO3 version 0.20 | Target PyO3 0.28, use `Bound<'py, T>` API | BLOCKING |
| 3 | `pyo3-asyncio` dependency | Replace with `pyo3-async-runtimes 0.28` | BLOCKING |
| 4 | Python >= 3.8 minimum | Python >= 3.9 (3.8 is EOL, pyo3-async-runtimes requires 3.9+) | HIGH |
| 5 | `thiserror 1.0` | `thiserror 2.0` | MEDIUM |
| 6 | `criterion 0.5` | `criterion 0.8` | LOW (dev-dep) |
| 7 | `wasm-pack` for WASM builds | `wasm-bindgen` CLI directly (wasm-pack archived) | HIGH |
| 8 | `TokenKind::Custom(&'static str)` | `TokenKind::Custom(StringId)` for runtime extensibility | MEDIUM |
| 9 | ProcessedText stores owned `String` fields | Use `StringId` references into TextStore | MEDIUM |
| 10 | `TextStore` with `&'static str` interning | Proper phase separation (build/read) with Arc promotion | HIGH |

---

## Recommended Build Order

Research across all dimensions converges on this phase order:

```
Phase 1: Foundation
  TextStore, StringId, Span, CharMapping, Token, Error types
  Feature flag architecture + WASM CI check
  CRITICAL: Resolve Token Copy, TextStore safety

Phase 2: Text Processing
  Normalizer trait + 6 built-in normalizers
  Tokenizer trait + 4 built-in tokenizers
  TextProcessor with pipeline traceability
  ProcessedText

Phase 3: Diff Computation
  DiffAlgorithm trait + Myers + Histogram
  DiffComputer, EditOperation, DiffResult
  Common prefix/suffix optimization

Phase 4: Metrics Engine
  Metric trait + 20+ TextMetrics + 15+ PairwiseMetrics
  MetricsEngine with lazy evaluation
  Content-addressed caching

Phase 5: Analysis Framework
  AnalyzerPlugin trait + PluginRegistry
  ExecutionPlanner (DAG-based dependency resolution)
  AnalysisCoordinator (sync first, parallel later)
  4 built-in analyzers

Phase 6: Orchestration & Config
  DiffOrchestrator, CacheManager
  ConfigBuilder + 4 presets (Fast/Syntactic/Semantic/Comprehensive)
  Public convenience API

Phase 7: Async Support
  Async APIs, tokio integration
  Parallel analyzer execution
  Timeout and cancellation

Phase 8: Python Bindings (PyO3)
  PyO3 module, type bindings, plugin adapters
  Batch Python interface (minimize GIL crossings)
  Async bridge via pyo3-async-runtimes
  .pyi stub generation

Phase 9: WASM Target
  Feature-gated compilation
  Sync-only API subset for browser
  wasm-bindgen integration
```

**Key constraints:**
- Phase 1 is blocking -- Token design and TextStore safety affect everything
- Phases 2-3 can partially overlap (diff algorithms only need Token from Phase 1)
- Phases 8-9 are independent and can run in parallel after Phase 6
- Async (Phase 7) is an optimization layer, not a foundation

---

## Key Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Token Copy + SmallVec compile error | CRITICAL | Resolve in Phase 1: use single Span, verify with `static_assertions::assert_impl_all!(Token: Copy)` |
| TextStore `&'static str` unsoundness | CRITICAL | Two-phase design: mutable build phase, then immutable Arc<TextStore>. Run Miri on all tests. |
| PyO3 0.28 API is completely different from spec examples | HIGH | Rewrite all Python binding code for Bound<T> API. Budget extra time for Phase 8. |
| WASM breakage discovered late | HIGH | Add WASM CI check in Phase 1. Test `cargo check --target wasm32-unknown-unknown --features wasm` from day 1. |
| Myers O(N^2) performance cliff | MEDIUM | Common prefix/suffix strip + D-threshold cutoff in Phase 3. Histogram as default for large inputs. |
| GIL deadlock in async Python bridge | MEDIUM | Use `py.allow_threads()` before entering tokio. Test with multi-threaded runtime from day 1. |
| RwLock write starvation in CacheManager | MEDIUM | Use `parking_lot::RwLock` instead of `std::sync::RwLock`. |
| Plugin dependency cycles | LOW | Validate on every `register()` call, not just at build time. |

---

## Stack Decisions Summary

| Category | Decision | Version | Confidence |
|----------|----------|---------|------------|
| Arena allocator | bumpalo | 3.19 | HIGH |
| Hash map | hashbrown | 0.16 | HIGH |
| Small vec | smallvec | 1.15 | HIGH |
| Error handling | thiserror | 2.0 | HIGH |
| Async runtime | tokio | 1.49 | HIGH |
| LRU cache | lru | 0.16 | HIGH |
| String interning | Custom (bumpalo + hashbrown) | N/A | HIGH |
| Python bindings | PyO3 | 0.28 | HIGH |
| Python async bridge | pyo3-async-runtimes | 0.28 | HIGH |
| Python build tool | maturin | 1.11 | HIGH |
| WASM bindings | wasm-bindgen | 0.2 | HIGH |
| WASM serialization | serde-wasm-bindgen | 0.6 | HIGH |
| Benchmarking | criterion | 0.8 | HIGH |
| Property testing | proptest | 1.9 | HIGH |
| Diagnostics | tracing | 0.1 | HIGH |
| Unicode | unicode-normalization | 0.1 | HIGH |

---

## Ready for Roadmap

All four research dimensions are complete:

- **STACK.md**: Dependency versions corrected, Cargo.toml template ready, feature flag architecture designed
- **FEATURES.md**: Table stakes, differentiators, and anti-features categorized with competitor analysis
- **ARCHITECTURE.md**: 4-layer design validated, data flow mapped, critical tensions identified with solutions
- **PITFALLS.md**: 15 pitfalls documented with prevention strategies and phase-to-pitfall mapping

Research is complete and ready for requirements definition and roadmap creation.

---
*Research synthesis for: Redline Core v1.0*
*Synthesized: 2026-02-06*
