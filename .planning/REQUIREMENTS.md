# Requirements: Redline Core v1.0

**Defined:** 2026-02-06
**Core Value:** Extensible, correct text diff computation with a clean plugin story -- users can write analyzers in Python that plug into a fast Rust engine, and the whole thing runs in browsers via WASM.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Foundation

- [ ] **FOUN-01**: Arena-allocated TextStore with string interning via bumpalo + hashbrown (StringId u32 newtype, Copy)
- [ ] **FOUN-02**: Span type using byte offsets with full API (contains, overlaps, merge, split)
- [ ] **FOUN-03**: Bidirectional CharMapping with composition across normalizer chains
- [ ] **FOUN-04**: Token type with Copy semantics (StringId, Span, TokenKind) -- no SmallVec
- [ ] **FOUN-05**: Structured error types via thiserror 2.0 (RedlineError hierarchy)
- [ ] **FOUN-06**: Feature flag architecture (std, async, serde, tracing, python, wasm) with WASM CI from day 1
- [ ] **FOUN-07**: Compile-time enforcement of python/wasm mutual exclusivity

### Text Processing

- [ ] **TEXT-01**: Normalizer trait with name(), cost(), and NormalizationResult (normalized text + CharMapping)
- [ ] **TEXT-02**: 6 built-in normalizers: Lowercase, Whitespace, Unicode (NFC/NFD/NFKC/NFKD), RemoveDiacritics, RemovePunctuation, RemoveDigits
- [ ] **TEXT-03**: Tokenizer trait with tokenize() producing Vec<Token> via TextStore interning
- [ ] **TEXT-04**: 4 built-in tokenizers: CharTokenizer, WordTokenizer, SentenceTokenizer, NGramTokenizer
- [ ] **TEXT-05**: TextProcessor orchestrator chaining normalizers then tokenizer with pipeline traceability
- [ ] **TEXT-06**: ProcessedText type preserving tokens, normalization layers, and composed CharMapping
- [ ] **TEXT-07**: Character mapping preserved and composable through entire normalization pipeline
- [ ] **TEXT-08**: Unicode correctness: grapheme-aware tokenization, CJK/emoji/RTL test coverage

### Diff Computation

- [ ] **DIFF-01**: DiffAlgorithm trait with compute() method operating on token slices
- [ ] **DIFF-02**: Myers O(ND) algorithm with common prefix/suffix optimization
- [ ] **DIFF-03**: Histogram diff algorithm (consistent performance, 10-100% faster than Myers on varied workloads)
- [ ] **DIFF-04**: DiffComputer with configurable algorithm selection
- [ ] **DIFF-05**: EditOperation types (Equal, Delete, Insert, Replace) with token index ranges
- [ ] **DIFF-06**: DiffResult with operations, statistics (edit distance, similarity ratio, change counts)
- [ ] **DIFF-07**: Myers D-threshold cutoff to prevent O(N^2) performance cliff on dissimilar texts

### Metrics Engine

- [ ] **METR-01**: Metric trait with id(), compute(), dependencies(), cost() for lazy evaluation
- [ ] **METR-02**: MetricRegistry for custom metric registration
- [ ] **METR-03**: TextMetrics: 20+ single-text metrics (word/char/sentence counts, lexical density, vocabulary richness, Flesch Reading Ease, Flesch-Kincaid, Gunning Fog, SMOG, average word/sentence length)
- [ ] **METR-04**: PairwiseMetrics: 15+ comparison metrics (Levenshtein, Jaro-Winkler, Jaccard, cosine similarity, Hamming, edit distance, similarity ratio, set-based metrics)
- [ ] **METR-05**: MetricsEngine with lazy evaluation (compute on first access, cache result)
- [ ] **METR-06**: Content-addressed caching via ContentHash keys and LRU eviction

### Analysis Framework

- [ ] **ANAL-01**: AnalyzerPlugin trait with metadata (id, name, version), dependencies (Dependency enum), and cost estimation
- [ ] **ANAL-02**: PluginRegistry for analyzer registration with incremental cycle detection on register()
- [ ] **ANAL-03**: ExecutionPlanner with DAG-based topological sort for dependency resolution
- [ ] **ANAL-04**: AnalysisCoordinator with synchronous execution (sequential, respecting dependency order)
- [ ] **ANAL-05**: AnalysisContext providing DiffResult, PairwiseMetrics, and TextMetrics to analyzers
- [ ] **ANAL-06**: AnalysisReport collecting results with graceful degradation (partial results on analyzer failure)
- [ ] **ANAL-07**: 4 built-in analyzers: SemanticAnalyzer, StylisticAnalyzer, ReadabilityAnalyzer, EditClassifier
- [ ] **ANAL-08**: EditClassifier with IntentCategory enum (Clarification, Expansion, Reduction, Correction, Reformulation, Stylistic)

### Orchestration & Configuration

- [ ] **ORCH-01**: DiffOrchestrator coordinating TextProcessor -> DiffComputer -> MetricsEngine -> AnalysisCoordinator
- [ ] **ORCH-02**: ConfigBuilder with fluent API, validation, and error reporting
- [ ] **ORCH-03**: 4 preset configurations: Fast (minimal processing), Syntactic (full diff, basic metrics), Semantic (full analysis), Comprehensive (everything)
- [ ] **ORCH-04**: CacheManager with parking_lot::RwLock for concurrent access and configurable max size

### Async Support

- [ ] **ASYN-01**: Async diff and analysis APIs behind `async` feature flag
- [ ] **ASYN-02**: Tokio integration with spawn_blocking for CPU-bound diff work
- [ ] **ASYN-03**: Parallel analyzer execution for independent analyzers (futures::join_all)
- [ ] **ASYN-04**: Timeout and cancellation support for long-running analyses
- [ ] **ASYN-05**: Graceful degradation: partial AnalysisReport when some analyzers fail or timeout

### Python Bindings

- [ ] **PYTH-01**: PyO3 0.28 module with maturin build system (abi3-py39)
- [ ] **PYTH-02**: Core type bindings: PyToken, PySpan, PyDiffResult, PyAnalysisReport (all owned, no Rust lifetimes)
- [ ] **PYTH-03**: Python plugin adapters: PyNormalizerAdapter, PyTokenizerAdapter, PyAnalyzerPluginAdapter (batch interface)
- [ ] **PYTH-04**: Async bridge via pyo3-async-runtimes (tokio <-> asyncio)
- [ ] **PYTH-05**: Error translation: every RedlineError variant maps to a Python exception class
- [ ] **PYTH-06**: Automated .pyi stub generation via pyo3-stub-gen
- [ ] **PYTH-07**: Python test suite covering all bindings

### WASM Target

- [ ] **WASM-01**: redline-core compiles to wasm32-unknown-unknown with `--features wasm --no-default-features`
- [ ] **WASM-02**: Core diffing (TextProcessor + DiffComputer) works in browser context (sync API)
- [ ] **WASM-03**: Metrics computation works in WASM (sync API)
- [ ] **WASM-04**: No Python/PyO3/tokio dependency when targeting WASM (feature-gated out)
- [ ] **WASM-05**: wasm-bindgen integration with serde-wasm-bindgen for type serialization

### Query Language

- [ ] **QURY-01**: Struct-based query types for filtering edit operations (EditFilter)
- [ ] **QURY-02**: Filter by EditKind, span ranges, token properties, similarity thresholds
- [ ] **QURY-03**: Composable query predicates (AND, OR, NOT)

### Quality & Performance

- [ ] **QUAL-01**: >80% unit test coverage across all modules
- [ ] **QUAL-02**: Property-based tests via proptest for diff algorithms and CharMapping composition
- [ ] **QUAL-03**: Criterion benchmarks for performance targets (<1ms diff for 100 tokens)
- [ ] **QUAL-04**: Zero unsafe code in public API surface
- [ ] **QUAL-05**: <50MB memory usage for 10K word document pair analysis
- [ ] **QUAL-06**: Miri clean on all TextStore and arena-related tests

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Algorithms

- **ALGO-01**: Patience diff algorithm
- **ALGO-02**: Streaming diff for large files (line-level only)

### Extended Plugins

- **PLUG-01**: WASM-based plugin sandbox (plugins as WASM modules)
- **PLUG-02**: Plugin marketplace / discovery mechanism

### Extended Analysis

- **XANA-01**: Native Rust ML analyzers (candle/BERT integration)
- **XANA-02**: Natural language change summarization
- **XANA-03**: Three-way merge analysis

### Extended Platform

- **PLAT-01**: Language-specific tokenizers (code-aware via tree-sitter)
- **PLAT-02**: Real-time collaborative diffing (OT/CRDT)
- **PLAT-03**: Diff visualization helpers (HTML output crate)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Streaming diffs for large files | Fundamentally changes architecture; tokens must fit in memory for diff algorithms. v2 for line-level only. |
| WASM plugin sandbox | v2 extensibility story. Python plugins via PyO3 sufficient for v1. |
| Native Rust ML analyzers | Python ML ecosystem (HuggingFace, spaCy) is vastly richer. ML lives in Python plugin layer. |
| Patience diff algorithm | Myers + Histogram cover most cases. Patience adds complexity without clear v1 benefit. |
| Real-time collaborative diffing | Different algorithm family (OT/CRDT). Not a v1 use case. |
| GUI or CLI tool | Redline is a library, not an end-user application. |
| Binary file diffing | Text-only framework. Different problem domain. |
| Syntax-aware / language-specific diff | Requires per-language parsers. Redline is text-first, not code-first. Plugin tokenizers can add this. |
| Automatic language detection | Adds model dependencies. Users know their input language. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUN-01 | Phase 1 | Pending |
| FOUN-02 | Phase 1 | Pending |
| FOUN-03 | Phase 1 | Pending |
| FOUN-04 | Phase 1 | Pending |
| FOUN-05 | Phase 1 | Pending |
| FOUN-06 | Phase 1 | Pending |
| FOUN-07 | Phase 1 | Pending |
| TEXT-01 | Phase 2 | Pending |
| TEXT-02 | Phase 2 | Pending |
| TEXT-03 | Phase 2 | Pending |
| TEXT-04 | Phase 2 | Pending |
| TEXT-05 | Phase 2 | Pending |
| TEXT-06 | Phase 2 | Pending |
| TEXT-07 | Phase 2 | Pending |
| TEXT-08 | Phase 2 | Pending |
| DIFF-01 | Phase 3 | Pending |
| DIFF-02 | Phase 3 | Pending |
| DIFF-03 | Phase 3 | Pending |
| DIFF-04 | Phase 3 | Pending |
| DIFF-05 | Phase 3 | Pending |
| DIFF-06 | Phase 3 | Pending |
| DIFF-07 | Phase 3 | Pending |
| METR-01 | Phase 4 | Pending |
| METR-02 | Phase 4 | Pending |
| METR-03 | Phase 4 | Pending |
| METR-04 | Phase 4 | Pending |
| METR-05 | Phase 4 | Pending |
| METR-06 | Phase 4 | Pending |
| ANAL-01 | Phase 5 | Pending |
| ANAL-02 | Phase 5 | Pending |
| ANAL-03 | Phase 5 | Pending |
| ANAL-04 | Phase 5 | Pending |
| ANAL-05 | Phase 5 | Pending |
| ANAL-06 | Phase 5 | Pending |
| ANAL-07 | Phase 5 | Pending |
| ANAL-08 | Phase 5 | Pending |
| ORCH-01 | Phase 6 | Pending |
| ORCH-02 | Phase 6 | Pending |
| ORCH-03 | Phase 6 | Pending |
| ORCH-04 | Phase 6 | Pending |
| ASYN-01 | Phase 7 | Pending |
| ASYN-02 | Phase 7 | Pending |
| ASYN-03 | Phase 7 | Pending |
| ASYN-04 | Phase 7 | Pending |
| ASYN-05 | Phase 7 | Pending |
| PYTH-01 | Phase 8 | Pending |
| PYTH-02 | Phase 8 | Pending |
| PYTH-03 | Phase 8 | Pending |
| PYTH-04 | Phase 8 | Pending |
| PYTH-05 | Phase 8 | Pending |
| PYTH-06 | Phase 8 | Pending |
| PYTH-07 | Phase 8 | Pending |
| WASM-01 | Phase 9 | Pending |
| WASM-02 | Phase 9 | Pending |
| WASM-03 | Phase 9 | Pending |
| WASM-04 | Phase 9 | Pending |
| WASM-05 | Phase 9 | Pending |
| QURY-01 | Phase 6 | Pending |
| QURY-02 | Phase 6 | Pending |
| QURY-03 | Phase 6 | Pending |
| QUAL-01 | All | Pending |
| QUAL-02 | Phase 3 | Pending |
| QUAL-03 | Phase 3 | Pending |
| QUAL-04 | All | Pending |
| QUAL-05 | Phase 6 | Pending |
| QUAL-06 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 62 total
- Mapped to phases: 62
- Unmapped: 0

---
*Requirements defined: 2026-02-06*
*Last updated: 2026-02-06 after research synthesis*
