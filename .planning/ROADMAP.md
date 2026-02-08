# Roadmap: Redline Core v1.0

> High-performance text diffing and analysis framework in Rust with Python bindings (PyO3) and WASM target

**Milestone:** v1.0
**Phases:** 9
**Requirements:** 62 (all mapped)
**Created:** 2026-02-06

---

## Cross-Cutting Requirements

Applied to every phase:

| ID | Requirement | Enforcement |
|----|-------------|-------------|
| QUAL-01 | >80% unit test coverage across all modules | CI coverage check per phase |
| QUAL-04 | Zero unsafe code in public API surface | `#![forbid(unsafe_code)]` on public modules |

---

## Phase 1: Foundation

**Goal:** Establish core types, memory layout, error handling, and build infrastructure that every subsequent phase depends on.

**Requirements:**

| ID | Description |
|----|-------------|
| FOUN-01 | Arena-allocated TextStore with string interning via bumpalo + hashbrown |
| FOUN-02 | Span type using byte offsets with full API |
| FOUN-03 | Bidirectional CharMapping with composition across normalizer chains |
| FOUN-04 | Token type with Copy semantics (StringId, Span, TokenKind) -- no SmallVec |
| FOUN-05 | Structured error types via thiserror 2.0 |
| FOUN-06 | Feature flag architecture (std, async, serde, tracing, python, wasm) with WASM CI |
| FOUN-07 | Compile-time enforcement of python/wasm mutual exclusivity |
| QUAL-06 | Miri clean on all TextStore and arena-related tests |

**Dependencies:** None (root phase, BLOCKING)

**Key Risks:**
- Token Copy + SmallVec is a compile error -- must resolve to single Span on day one
- TextStore `&'static str` unsoundness -- use two-phase design (build/read) with Arc promotion
- Bump allocator is `!Sync` -- separate mutable interning phase from immutable read phase

**Success Criteria:**
1. `static_assertions::assert_impl_all!(Token: Copy, Clone, Debug, PartialEq)` compiles
2. TextStore interns 10K strings, promotes to `Arc<TextStore>`, all lookups return correct values
3. CharMapping composes 3 mappings and round-trips positions correctly (property-tested)
4. `cargo check --target wasm32-unknown-unknown --no-default-features --features wasm` passes in CI
5. `cargo +nightly miri test` passes on all foundation tests

---

## Phase 2: Text Processing

**Goal:** Raw text is normalized and tokenized into token streams with full position traceability back to original text.

**Requirements:**

| ID | Description |
|----|-------------|
| TEXT-01 | Normalizer trait with name(), cost(), and NormalizationResult |
| TEXT-02 | 6 built-in normalizers (Lowercase, Whitespace, Unicode, Diacritics, Punctuation, Digits) |
| TEXT-03 | Tokenizer trait producing Vec<Token> via TextStore interning |
| TEXT-04 | 4 built-in tokenizers (Char, Word, Sentence, NGram) |
| TEXT-05 | TextProcessor orchestrator chaining normalizers then tokenizer |
| TEXT-06 | ProcessedText type with tokens, layers, and composed CharMapping |
| TEXT-07 | Character mapping preserved through entire normalization pipeline |
| TEXT-08 | Unicode correctness: grapheme-aware tokenization, CJK/emoji/RTL test coverage |

**Dependencies:** Phase 1 (TextStore, Token, Span, CharMapping)

**Key Risks:**
- Unicode CharMapping breaks for multi-codepoint graphemes (NFC: e + accent -> single codepoint)
- ProcessedText should use StringId references, not owned Strings (Pitfall #14)
- NormalizationLayer text storage is memory-heavy -- make optional via trace flag (Pitfall #15)

**Success Criteria:**
1. Normalize "Hello  WORLD" with Lowercase + Whitespace -> "hello world" with valid CharMapping
2. CharMapping maps normalized position back to original through 3-normalizer chain (property-tested with Unicode)
3. WordTokenizer handles CJK, emoji, RTL text correctly
4. TextProcessor pipeline: raw text -> 3 normalizers -> WordTokenizer -> ProcessedText with valid spans
5. All normalizer CharMappings compose correctly for accented characters and combining characters

**Plans:** 8 plans in 5 waves

| Plan | Wave | Type | Description | Depends On |
|------|------|------|-------------|------------|
| 02-01 | 1 | execute | Setup: deps, modules, traits, errors | — |
| 02-02 | 2 | tdd | Lowercase + WhitespaceNormalizer | 02-01 |
| 02-03 | 2 | tdd | UnicodeNormalizer + RemoveDiacritics | 02-01 |
| 02-04 | 2 | tdd | RemovePunctuation + RemoveDigits | 02-01 |
| 02-05 | 2 | tdd | CharTokenizer + WordTokenizer | 02-01 |
| 02-06 | 3 | tdd | SentenceTokenizer + NGram tokenizers | 02-05 |
| 02-07 | 4 | tdd | TextProcessor + ProcessedText pipeline | 02-02..06 |
| 02-08 | 5 | execute | Integration tests + Unicode edge cases | all |

---

## Phase 2.1: Test Coverage Enhancement & Deterministic Simulation Testing (INSERTED)

**Goal:** Harden Phase 1+2 code with comprehensive edge case coverage, improved integration tests, and seed-based deterministic simulation testing for reproducible fuzz-style discovery.

**Requirements:**

| ID | Description |
|----|-------------|
| TEST-01 | Review all public methods and ensure edge case coverage in unit tests |
| TEST-02 | Enhance unit tests: boundary conditions, error paths, empty/huge inputs |
| TEST-03 | Improve integration tests to be comprehensive across normalizer/tokenizer combinations |
| TEST-04 | Deterministic simulation testing framework: seed-based, reproducible, replayable |
| TEST-05 | Simulation test suites for CharMapping, normalizers, tokenizers, and full pipeline |

**Dependencies:** Phase 2 (all Phase 1+2 code exists to test)

**Key Risks:**
- Simulation framework design: proptest already provides seeded testing — determine if custom framework needed or proptest config suffices
- Test combinatorial explosion: 6 normalizers x 5 tokenizers = 30 combos; need smart selection not exhaustive

**Success Criteria:**
1. Every public method has at least one edge case test (empty input, single char, max-length, Unicode boundary)
2. Integration tests cover all normalizer-tokenizer combinations that make semantic sense
3. Deterministic simulation: run N random scenarios from seed S, replay any failure with just the seed
4. Running `cargo test` with `PROPTEST_CASES=10000` finds zero new failures
5. All simulation failures are reproducible: `SIMULATION_SEED=X cargo test` replays exact scenario

**Plans:** 5 plans in 1 wave

| Plan | Wave | Type | Description | Depends On |
|------|------|------|-------------|------------|
| 02.1-01 | 1 | execute | Core type edge case tests (Span, Token, TextStore, CharMapping, Error) | -- |
| 02.1-02 | 1 | execute | Normalizer edge case tests (all 6) | -- |
| 02.1-03 | 1 | execute | Tokenizer + process edge case tests | -- |
| 02.1-04 | 1 | execute | Normalizer x tokenizer integration matrix | -- |
| 02.1-05 | 1 | execute | Deterministic simulation testing (proptest) | -- |

---

## Phase 3: Diff Computation

**Goal:** Given two ProcessedTexts, compute a correct, performant edit script with configurable algorithm selection.

**Requirements:**

| ID | Description |
|----|-------------|
| DIFF-01 | DiffAlgorithm trait with compute() method on token slices |
| DIFF-02 | Myers O(ND) algorithm with common prefix/suffix optimization |
| DIFF-03 | Histogram diff algorithm |
| DIFF-04 | DiffComputer with configurable algorithm selection |
| DIFF-05 | EditOperation types (Equal, Delete, Insert, Replace) with token index ranges |
| DIFF-06 | DiffResult with operations, statistics, similarity ratio |
| DIFF-07 | Myers D-threshold cutoff to prevent O(N^2) cliff |
| QUAL-02 | Property-based tests: apply(diff(a,b), a) == b for all inputs |
| QUAL-03 | Criterion benchmarks: <1ms for 100 tokens, <200ms for 10K tokens |

**Dependencies:** Phase 1 (Token), Phase 2 (ProcessedText)

**Key Risks:**
- Myers O(N^2) performance cliff on dissimilar texts >5K tokens (Pitfall #8)
- EditOperation lifetime: use index ranges not borrowed slices to keep DiffResult cacheable (Pitfall #4)

**Success Criteria:**
1. Property test: `apply(diff(a,b), a) == b` holds for 1000+ generated input pairs
2. Myers diff of "the cat sat" vs "the dog sat on the mat" produces correct Equal/Replace/Insert operations
3. Histogram diff of same input produces equivalent (or better) result
4. 100-token diff completes in <1ms (criterion benchmark)
5. 10K-token dissimilar diff completes in <500ms with D-threshold cutoff (no hang)
6. DiffResult can be stored in a HashMap (no lifetime parameter)

**Plans:** 5 plans in 4 waves

| Plan | Wave | Type | Description | Depends On |
|------|------|------|-------------|------------|
| 03-01 | 1 | execute | Diff module foundation: types, trait, error, utilities, computer, result | -- |
| 03-02 | 2 | tdd | Myers O(ND) algorithm with D-threshold and common prefix/suffix | 03-01 |
| 03-03 | 3 | tdd | Histogram diff algorithm with chain limit and Myers fallback | 03-01, 03-02 |
| 03-04 | 4 | execute | Property tests and integration tests | 03-01..03 |
| 03-05 | 4 | execute | Criterion benchmarks | 03-01..03 |

---

## Phase 4: Metrics Engine

**Goal:** Quantitative measurements of text properties and pairwise comparisons, with lazy evaluation and caching.

**Requirements:**

| ID | Description |
|----|-------------|
| METR-01 | Metric trait with id(), compute(), dependencies(), cost() |
| METR-02 | MetricRegistry for custom metric registration |
| METR-03 | TextMetrics: 20+ single-text metrics (counts, readability, complexity) |
| METR-04 | PairwiseMetrics: 15+ comparison metrics (similarity, distance, set-based) |
| METR-05 | MetricsEngine with lazy evaluation |
| METR-06 | Content-addressed caching via ContentHash keys and LRU eviction |

**Dependencies:** Phase 1 (types), Phase 2 (ProcessedText), Phase 3 (DiffResult for pairwise delta metrics)

**Key Risks:**
- Eager computation of all 35+ metrics wastes time (Anti-pattern #3) -- lazy evaluation essential
- Floating point consistency across platforms (including WASM)

**Success Criteria:**
1. TextMetrics computes word count, sentence count, Flesch Reading Ease, Flesch-Kincaid Grade Level correctly
2. PairwiseMetrics computes Levenshtein, Jaro-Winkler, Jaccard, cosine similarity correctly
3. Lazy evaluation: requesting only "word_count" does not compute readability scores
4. Cache hit: same text hashed twice returns cached metrics without recomputation
5. Custom metric via MetricRegistry integrates with lazy evaluation and caching

**Plans:** 8 plans in 5 waves

| Plan | Wave | Type | Description | Depends On |
|------|------|------|-------------|------------|
| 04-01 | 1 | execute | Foundation: Metric trait, registry, cache, engine | -- |
| 04-02 | 2 | tdd | Count metrics batch 1 (word, char, byte, sentence, syllable, unique_word) | 04-01 |
| 04-03 | 2 | tdd | Count metrics batch 2 (paragraph, line, letter, digit, whitespace, punctuation) + engine integration | 04-01 |
| 04-04 | 3 | tdd | Readability batch 1 (avg_word_length, avg_sentence_length, vocabulary_richness, lexical_density, flesch_reading_ease) | 04-02, 04-03 |
| 04-05 | 3 | tdd | Readability batch 2 (flesch_kincaid_grade, gunning_fog, smog_index, coleman_liau, ari) + engine integration | 04-02, 04-03 |
| 04-06 | 4 | tdd | Pairwise set/vector (jaccard, cosine, dice, overlap, length_ratio, word_count_diff, char_count_diff) | 04-01 |
| 04-07 | 4 | tdd | Pairwise edit distance (levenshtein, damerau_levenshtein, hamming, jaro, jaro_winkler) | 04-01 |
| 04-08 | 5 | execute | Delta metrics + register_builtins + comprehensive integration tests | 04-04..07 |

---

## Phase 4.1: Realistic Test Fixtures (INSERTED)

**Goal:** Improve test coverage across all modules by replacing synthetic/minimal test data with real-world phrases, sentences, and paragraphs that exercise realistic text processing scenarios.

**Requirements:**

| ID | Description |
|----|-------------|
| TFIX-01 | Shared test fixtures module with real-world English text (prose, technical, literary) |
| TFIX-02 | Enhance normalizer tests with realistic multi-sentence paragraphs |
| TFIX-03 | Enhance tokenizer tests with real phrases, sentences, and paragraphs |
| TFIX-04 | Enhance diff computation tests with realistic before/after document pairs |
| TFIX-05 | Enhance metrics tests with real text that produces meaningful metric values |
| TFIX-06 | Multi-language fixtures (CJK, accented, mixed-script) for Unicode paths |

**Dependencies:** Phase 4 (all modules exist to test against)

**Key Risks:**
- Test fixture size: keep fixtures large enough to be realistic but small enough for fast CI
- Avoid copyrighted text: use public domain or original prose

**Success Criteria:**
1. Every module has at least one test using a multi-sentence paragraph (not just "hello world")
2. Readability metrics tested against text with known grade levels (e.g., Hemingway vs. academic)
3. Diff tests use realistic edit scenarios (typo fix, paragraph rewrite, sentence insertion)
4. All existing tests still pass (no regressions)
5. Test fixtures are shared via a common module to avoid duplication

**Plans:** 4 plans in 2 waves

| Plan | Wave | Type | Description | Depends On |
|------|------|------|-------------|------------|
| 04.1-01 | 1 | execute | Shared fixtures module (constants, helpers, diff pairs, multilang) | -- |
| 04.1-02 | 2 | execute | Enhance normalizer/tokenizer/pipeline tests with realistic fixtures | 04.1-01 |
| 04.1-03 | 2 | execute | Enhance diff tests with realistic before/after document pairs | 04.1-01 |
| 04.1-04 | 2 | execute | Enhance metrics tests with readability ordering and realistic pairwise | 04.1-01 |

---

## Phase 5: Analysis Framework

**Goal:** Pluggable analyzer system with dependency resolution that enriches diff results with semantic insights.

**Requirements:**

| ID | Description |
|----|-------------|
| ANAL-01 | AnalyzerPlugin trait with metadata, dependencies, cost |
| ANAL-02 | PluginRegistry with incremental cycle detection on register() |
| ANAL-03 | ExecutionPlanner with DAG-based topological sort |
| ANAL-04 | AnalysisCoordinator with synchronous execution |
| ANAL-05 | AnalysisContext providing DiffResult, metrics to analyzers |
| ANAL-06 | AnalysisReport with graceful degradation (partial results on failure) |
| ANAL-07 | 4 built-in analyzers (Semantic, Stylistic, Readability, EditClassifier) |
| ANAL-08 | EditClassifier with IntentCategory enum |

**Dependencies:** Phase 3 (DiffResult), Phase 4 (metrics)

**Key Risks:**
- Circular analyzer dependencies at runtime (Pitfall #12) -- validate on every register()
- Start with sequential execution, add parallel later (Anti-pattern #4)
- Classification thresholds need sensible defaults with configuration override

**Success Criteria:**
1. Register 3 analyzers with declared dependencies; ExecutionPlanner produces valid topological order
2. Register circular dependency -> returns error, never panics
3. AnalysisCoordinator runs all 4 built-in analyzers on a diff result and produces AnalysisReport
4. Analyzer failure: if ReadabilityAnalyzer panics, remaining analyzers still produce results
5. EditClassifier categorizes "fix typo" edit as Correction, "add paragraph" as Expansion

---

## Phase 6: Orchestration & Configuration

**Goal:** Single entry point coordinating the full pipeline with unified configuration, query API, and presets.

**Requirements:**

| ID | Description |
|----|-------------|
| ORCH-01 | DiffOrchestrator: text in -> analysis out, single call |
| ORCH-02 | ConfigBuilder with fluent API and validation |
| ORCH-03 | 4 presets: Fast, Syntactic, Semantic, Comprehensive |
| ORCH-04 | CacheManager with parking_lot::RwLock and configurable max size |
| QURY-01 | Struct-based query types for filtering edit operations |
| QURY-02 | Filter by EditKind, span ranges, token properties |
| QURY-03 | Composable query predicates (AND, OR, NOT) |
| QUAL-05 | <50MB memory usage for 10K word document pair |

**Dependencies:** Phase 5 (all pipeline stages exist)

**Key Risks:**
- Configuration explosion -- use builder pattern with sensible defaults
- CacheManager RwLock deadlock (Pitfall #10) -- use parking_lot, isolate lock scopes
- Query API design is the primary consumer-facing filter interface

**Success Criteria:**
1. `DiffOrchestrator::new(config).diff("old", "new")` returns complete result (diff + metrics + analysis)
2. `ConfigBuilder::preset(Fast)` produces valid config with minimal processing
3. Query: filter operations where `kind == Replace && span.len() > 10` returns correct subset
4. Memory: 10K word document pair analysis uses <50MB (measured)
5. CacheManager multi-threaded stress test passes (no deadlock)

---

## Phase 7: Async Support

**Goal:** Non-blocking pipeline and parallel analyzer execution for async applications.

**Requirements:**

| ID | Description |
|----|-------------|
| ASYN-01 | Async diff and analysis APIs behind `async` feature flag |
| ASYN-02 | Tokio integration with spawn_blocking for CPU-bound work |
| ASYN-03 | Parallel analyzer execution for independent analyzers |
| ASYN-04 | Timeout and cancellation support |
| ASYN-05 | Graceful degradation: partial results on timeout |

**Dependencies:** Phase 6 (sync pipeline to make async)

**Key Risks:**
- Async is CPU-bound, not I/O-bound -- spawn_blocking for diff, join_all for independent analyzers
- WASM is single-threaded -- async on WASM means sequential execution, not parallel
- Feature flag interactions: `async` + `wasm` must compose correctly

**Success Criteria:**
1. `orchestrator.diff_async("old", "new").await` produces same results as sync version
2. Independent analyzers run in parallel (wall clock < sum of individual times)
3. Timeout: 10ms timeout on a long diff returns partial results, not an error
4. Cancellation: cancelled operation returns within 100ms
5. `cargo build --features async` does not break `--features wasm` compilation

---

## Phase 8: Python Bindings

**Goal:** Python developers can `pip install redline` and use the full pipeline with Pythonic types and plugin extensibility.

**Requirements:**

| ID | Description |
|----|-------------|
| PYTH-01 | PyO3 0.28 module with maturin build (abi3-py39) |
| PYTH-02 | Core type bindings (PyToken, PySpan, PyDiffResult, etc.) -- all owned |
| PYTH-03 | Python plugin adapters with batch interface (minimize GIL crossings) |
| PYTH-04 | Async bridge via pyo3-async-runtimes |
| PYTH-05 | Error translation: every RedlineError -> Python exception |
| PYTH-06 | .pyi stub generation via pyo3-stub-gen |
| PYTH-07 | Python test suite covering all bindings |

**Dependencies:** Phase 6 (stable sync API). Independent of Phase 7 and Phase 9.

**Key Risks:**
- GIL deadlock in async bridge (Pitfall #6) -- use py.allow_threads() before tokio
- Per-call GIL overhead kills Python plugin performance (Pitfall #11) -- batch interface
- All Rust-to-Python types must be owned (no lifetimes crossing FFI boundary)

**Success Criteria:**
1. `from redline import Pipeline; result = Pipeline().run("old", "new")` works in Python 3.9+
2. Python analyzer plugin: implement in Python, register, execute from Rust pipeline
3. Batch plugin: tokenize 1000 texts in one call, not 1000 GIL acquisitions
4. Async: `await pipeline.run_async("old", "new")` works from asyncio
5. `.pyi` stubs provide autocomplete for all public API in VS Code

---

## Phase 9: WASM Target

**Goal:** Redline runs in the browser via a WASM module with sync API.

**Requirements:**

| ID | Description |
|----|-------------|
| WASM-01 | Compiles to wasm32-unknown-unknown with `--features wasm --no-default-features` |
| WASM-02 | Core diffing (TextProcessor + DiffComputer) works in browser |
| WASM-03 | Metrics computation works in WASM |
| WASM-04 | No Python/PyO3/tokio dependencies in WASM build |
| WASM-05 | wasm-bindgen + serde-wasm-bindgen for type serialization |

**Dependencies:** Phase 6 (stable core API). Independent of Phase 7 and Phase 8.

**Key Risks:**
- Tokio not available in WASM -- sync-only API
- Bundle size -- use `wasm-opt`, LTO, `opt-level = 'z'`
- wasm-bindgen can't handle trait objects -- use concrete types / serialization boundary

**Success Criteria:**
1. `cargo build --target wasm32-unknown-unknown --no-default-features --features wasm` succeeds
2. Diff two 1000-word documents in headless browser test
3. Metrics computation returns valid results in WASM
4. No tokio, PyO3, or std::thread references in WASM binary
5. WASM module size reasonable (target <500KB gzipped)

---

## Progress

| Phase | Name | Reqs | Status | Dependencies |
|-------|------|------|--------|-------------|
| 1 | Foundation | 8 | ✓ Complete | None (BLOCKING) |
| 2 | Text Processing | 8 | ✓ Complete | Phase 1 |
| 2.1 | Test Coverage & Simulation | 5 | Planned | Phase 2 (INSERTED) |
| 3 | Diff Computation | 9 | Planned (5 plans) | Phase 1, 2 |
| 4 | Metrics Engine | 6 | ✓ Complete | Phase 1, 2, 3 |
| 4.1 | Realistic Test Fixtures | 6 | Planned (4 plans) | Phase 4 (INSERTED) |
| 5 | Analysis Framework | 8 | Pending | Phase 3, 4 |
| 6 | Orchestration & Config | 8 | Pending | Phase 2-5 |
| 7 | Async Support | 5 | Pending | Phase 6 |
| 8 | Python Bindings | 7 | Pending | Phase 6 |
| 9 | WASM Target | 5 | Pending | Phase 6 |
| -- | Cross-cutting | 2 | Ongoing | All phases |

**Total:** 62 requirements mapped + 2 cross-cutting = 64 requirement slots. Zero orphans.

## Dependency Graph

```
Phase 1: Foundation [BLOCKING]
  |
  +---> Phase 2: Text Processing ---+
  |                                  |
  +--- (partial overlap possible) ---+
                                     |
                               Phase 3: Diff Computation
                                     |
                               Phase 4: Metrics Engine
                                     |
                               Phase 5: Analysis Framework
                                     |
                               Phase 6: Orchestration & Config
                                /     |     \
                               /      |      \
                        Phase 7   Phase 8   Phase 9
                        (Async)   (Python)   (WASM)
                           \        |        /
                            [Independent, parallelizable]
```

---
*Roadmap created: 2026-02-06*
*Last updated: 2026-02-06 after requirements definition*
