# Redline Core

## What This Is

Redline Core is a high-performance text diffing and analysis framework in Rust that goes beyond traditional line-based diff tools. It provides a layered architecture for text normalization, tokenization, diff computation, metrics, and pluggable analysis — with Python extensibility via PyO3 and browser deployment via WASM. It's a library for developers building tools that need to understand *how* and *why* text changed, not just *what* changed.

## Core Value

Extensible, correct text diff computation with a clean plugin story — users can write analyzers in Python that plug into a fast Rust engine, and the whole thing runs in browsers via WASM.

## Requirements

### Validated

(None yet — ship to validate)

### Active

#### Foundation
- [ ] Arena-allocated TextStore with string interning (StringId, zero-copy)
- [ ] Span and position types with full API
- [ ] Bidirectional CharMapping with composition
- [ ] Structured error types via thiserror

#### Text Processing
- [ ] Normalizer trait with 5-6 built-in normalizers (lowercase, whitespace, unicode, diacritics, punctuation, digits)
- [ ] Tokenizer trait with 3-4 built-in tokenizers (char, word, sentence, n-gram)
- [ ] TextProcessor orchestrator with pipeline traceability layers
- [ ] Character mapping preserved through entire pipeline

#### Diff Computation
- [ ] DiffAlgorithm trait
- [ ] Myers O(ND) algorithm
- [ ] Histogram diff algorithm
- [ ] DiffComputer with algorithm selection
- [ ] EditOperation types (Equal, Delete, Insert, Replace) with token slices
- [ ] DiffResult with statistics and similarity ratio

#### Metrics Engine
- [ ] Metric trait and registry for custom metrics
- [ ] TextMetrics (20+ single-text metrics: counts, readability, complexity)
- [ ] PairwiseMetrics (15+ comparison metrics: similarity, distance, set-based)
- [ ] MetricsEngine with lazy evaluation
- [ ] Content-addressed caching

#### Analysis Framework
- [ ] AnalyzerPlugin trait with metadata, dependencies, cost
- [ ] Dependency resolution and execution planning
- [ ] AnalysisCoordinator with sync and async execution
- [ ] Plugin registry
- [ ] Parallel analyzer execution via tokio
- [ ] 3-4 built-in analyzers (semantic, stylistic, readability, edit classification)

#### Orchestration & Configuration
- [ ] DiffOrchestrator coordinating full workflow
- [ ] ConfigBuilder with fluent API and validation
- [ ] Preset configurations (Fast, Syntactic, Semantic, Comprehensive)
- [ ] CacheManager with RwLock for concurrent access

#### Async Support
- [ ] Async diff and analysis APIs
- [ ] Tokio integration for CPU-bound work (spawn_blocking)
- [ ] Parallel analyzer execution
- [ ] Timeout and cancellation support
- [ ] Graceful degradation (partial results on analyzer failure)

#### Python Bindings (PyO3)
- [ ] PyO3 module with maturin build system
- [ ] Core type bindings (Token, Span, DiffResult, etc.)
- [ ] Python plugin adapters (PyNormalizer, PyTokenizer, PyAnalyzerPlugin)
- [ ] Async bridge (tokio <-> asyncio via pyo3-asyncio)
- [ ] Error translation (Rust errors -> Python exceptions)
- [ ] Automated .pyi stub generation
- [ ] Python test suite

#### WASM Target
- [ ] Redline Core compiles to wasm32-unknown-unknown
- [ ] Core diffing and analysis works in browser context
- [ ] No Python/PyO3 dependency when targeting WASM (feature-gated)

#### Query Language
- [ ] Struct-based query types for filtering edit operations
- [ ] Filter by EditKind, span ranges, token properties
- [ ] Composable query predicates

### Out of Scope

- Streaming diffs for large files — complexity not justified for v1 use cases
- WASM-based plugin sandbox (plugins as WASM modules) — v2 extensibility story, Python plugins sufficient for v1
- Native Rust ML analyzers — ML lives in Python plugin layer where ecosystem is richer; existing candle/BERT prototype won't be in v1 core
- Patience diff algorithm — Myers + Histogram sufficient for v1, can add later
- Real-time collaborative diffing — not a use case for v1
- GUI or CLI tool — Redline is a library, not an end-user application

## Context

- Existing codebase is a BERT embedding proof-of-concept (bert_v1.rs, bert_v2.rs) that will be replaced — essentially a greenfield build from detailed design docs
- Design document (`docs/design/DESIGN.md`) and technical spec (`docs/design/TECHNICAL_SPEC.md`) are comprehensive and approved for implementation
- Development environment uses Nix/devenv with Rust nightly toolchain, Python 3.14, and pre-commit hooks
- Workspace structure: `crates/core` as the main library crate
- Design emphasizes zero-cost abstractions, memory efficiency (<50MB for typical workloads), and testability
- Token design uses Copy semantics (StringId-based) to enable lock-free concurrency
- The spec includes detailed API signatures for every component — implementation should follow these closely

## Constraints

- **Tech stack**: Rust (nightly, edition 2024) for core, PyO3 for Python bindings, maturin for Python packaging
- **Performance**: <1ms diff for 100 tokens, 2x faster than v0 prototype, <50MB memory for 10K word documents
- **Safety**: Zero unsafe code in public API surface, all public methods use safe Rust
- **Compatibility**: Python >= 3.8, WASM via wasm32-unknown-unknown target
- **Dependencies**: Use crates specified in technical spec (bumpalo, hashbrown, smallvec, thiserror, tokio, lru, pyo3)
- **Testing**: >80% unit test coverage, property-based tests via proptest, benchmarks via criterion

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ID-based tokens (StringId) over borrowed strings | Enables Copy semantics, no lifetime complexity, lock-free concurrency | -- Pending |
| Python for ML plugins, not Rust-native | Python ML ecosystem (HuggingFace, spaCy) is vastly richer, keeps Rust core lean | -- Pending |
| PyO3 for Python interface (not WASM plugins) | Direct Python access, native speed, full ML ecosystem integration for v1 | -- Pending |
| WASM as compilation target (not plugin sandbox) | Browser deployment for v1, WASM plugin sandbox deferred to v2 | -- Pending |
| Struct-based query language (not DSL) | Simple, type-safe, sufficient for v1 filtering needs | -- Pending |
| Myers + Histogram algorithms (skip Patience) | Two algorithms cover most use cases, Patience adds complexity without clear v1 benefit | -- Pending |

---
*Last updated: 2026-02-05 after initialization*
