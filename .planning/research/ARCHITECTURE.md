# Architecture Research

**Domain:** Text diffing and analysis framework (Rust)
**Project:** Redline Core v1.0
**Researched:** 2026-02-05
**Confidence:** HIGH

## Architecture Layers

The design documents describe a 4-layer architecture. This research validates the layer design and identifies integration points, risks, and build order.

### Layer 1: Foundation (Types + Storage)

**Components:** TextStore, StringId, Span, CharMapping, Token, Error types

**Purpose:** Arena-allocated string storage with interning, position tracking primitives, and the core Token type that flows through all subsequent layers.

**Key characteristics:**
- TextStore owns all string data via bumpalo arena
- StringId is a u32 newtype (Copy, 4 bytes) -- zero-cost token references
- Span tracks byte ranges in text (start, end, both usize)
- CharMapping provides bidirectional position mapping (normalized <-> original)
- No dependencies on other layers

### Layer 2: Engine (Processing + Computation)

**Components:** Normalizers, Tokenizers, TextProcessor, DiffAlgorithm, DiffComputer, MetricsEngine

**Purpose:** Text processing pipeline, diff computation, and metrics calculation. The core computational layer.

**Sub-layers:**
1. Text Processing: Normalizer trait + Tokenizer trait + TextProcessor orchestrator
2. Diff Computation: DiffAlgorithm trait + Myers + Histogram + DiffComputer
3. Metrics: Metric trait + TextMetrics + PairwiseMetrics + MetricsEngine

### Layer 3: Orchestration (Coordination + Analysis)

**Components:** AnalyzerPlugin, PluginRegistry, ExecutionPlanner, AnalysisCoordinator, DiffOrchestrator, CacheManager, ConfigBuilder

**Purpose:** Coordinate the full workflow from text input to analysis output. Plugin system for extensible analysis.

### Layer 4: API (Public Surface + Bindings)

**Components:** Public convenience API, Python bindings (PyO3), WASM bindings (wasm-bindgen)

**Purpose:** Ergonomic public API, cross-language bindings, configuration presets.

## Key Architectural Patterns

### Pattern 1: Arena + Interning (StringId-based Tokens)

**What:** All strings live in a bumpalo arena. A hashbrown HashMap maps string content to StringId (u32). Tokens reference strings by StringId, not by pointer or slice. This makes Token small and Copy.

**Trade-offs:**
- PRO: Token is Copy (4-byte StringId instead of fat pointer), enables lock-free sharing
- PRO: Arena allocation is fast (bump pointer, no per-string overhead)
- PRO: String deduplication via interning reduces memory for repeated tokens
- CON: String lookup requires indirection through TextStore
- CON: Cross-store bugs are silent -- StringId from store A used with store B

**Confidence:** HIGH -- this pattern is proven in imara-diff, salsa, and rustc's own symbol interning.

### Pattern 2: Trait Objects with NormalizerClone

**What:** Extensible components (Normalizer, Tokenizer, DiffAlgorithm, AnalyzerPlugin, Metric) use trait objects (`Box<dyn Trait>`) for runtime polymorphism. A helper `NormalizerClone` trait enables `Clone for Box<dyn Normalizer>`.

**Trade-offs:**
- PRO: Full runtime extensibility -- users write a struct implementing the trait
- PRO: Python plugins wrap `Py<PyAny>` in adapter structs that implement the Rust trait
- CON: Dynamic dispatch overhead (~5ns per vtable lookup) -- negligible for text processing
- CON: Clone for trait objects requires the NormalizerClone workaround pattern

**Confidence:** HIGH -- standard Rust pattern, used by serde, tower, and many other libraries.

### Pattern 3: Arc<ProcessedText> Lifetime Bridge

**What:** `ProcessedText` contains an `Arc<TextStore>` so tokens remain valid. `DiffResult<'a>` borrows `&'a [Token]` slices from the `Arc<ProcessedText>` owned by the result.

**Trade-offs:**
- PRO: Zero-copy token access in EditOperations
- PRO: Arc enables thread-safe sharing without cloning the entire TextStore
- CON: The lifetime `'a` on DiffResult ties it to the ProcessedText
- CON: Arc has atomic reference counting overhead (~5ns increment/decrement)

**Confidence:** HIGH -- self-referential via Arc is a well-established Rust pattern.

### Pattern 4: Feature-Gated Platform Layers

**What:** PyO3 bindings and WASM compatibility are gated behind Cargo feature flags and `#[cfg]` attributes, so the core library compiles for all targets.

**Trade-offs:**
- PRO: Core crate compiles to WASM without PyO3 (which cannot target WASM)
- PRO: Users who only need Rust don't pay for Python binding compilation
- CON: Feature matrix testing is multiplicative -- must test all combinations
- CON: Conditional compilation can hide bugs

**Confidence:** HIGH -- this is the standard Rust approach, used by reqwest, wasm-bindgen, etc.

### Pattern 5: DAG-Based Dependency Resolution for Plugins

**What:** Analyzers declare dependencies on metrics and other analyzers. The ExecutionPlanner builds a DAG, performs topological sort, identifies parallelizable groups, and schedules execution.

**Trade-offs:**
- PRO: Users never need to manually order analyzer execution
- PRO: Enables parallel execution of independent analyzers
- CON: Cycle detection adds complexity
- CON: Cost-based scheduling adds further complexity

**Confidence:** MEDIUM -- the pattern is sound but implementation complexity is high. Recommend starting with simple linear execution and adding parallel scheduling as an optimization later.

## Data Flow

### Primary Diff Workflow

```
User provides (original: &str, modified: &str)
    |
    v
TextProcessor.process(text)
    |
    | original --> [Normalizer 1] --> [Normalizer 2] --> ...
    |               (+ CharMapping)    (+ CharMapping)
    |                    |
    |        CharMapping::compose() <--- accumulated
    |                    |
    |               Tokenizer (&mut TextStore)
    |                    |
    |               ProcessedText {
    |                 original: String,
    |                 normalized: String,
    |                 tokens: Vec<Token>,
    |                 layers: Vec<NormLayer>,
    |                 mapping: CharMapping,
    |                 text_store: Arc<TextStore>,
    |               }
    |
    | (both texts processed)
    v
DiffComputer.compute(original_pt, modified_pt)
    |
    | original_pt.tokens --> Myers/Histogram <-- modified_pt.tokens
    |                            |
    |              Vec<EditOperation<'a>> {
    |                kind: Equal|Delete|Insert|Replace,
    |                original_tokens: &'a [Token],
    |                modified_tokens: &'a [Token],
    |              }
    |                            |
    |              DiffResult<'a> {
    |                operations, statistics,
    |                original: Arc<ProcessedText>,
    |                modified: Arc<ProcessedText>,
    |              }
    v
MetricsEngine.compute_pairwise(original, modified)
    |
    | ContentHash(original) + ContentHash(modified) --> cache key
    | cache miss? --> compute TextMetrics x2 + PairwiseMetrics
    | cache hit?  --> return Arc<PairwiseMetrics>
    v
AnalysisCoordinator.analyze(ctx, plugin_ids)
    |
    | 1. Build dependency DAG from plugin declarations
    | 2. Topological sort --> execution levels
    | 3. For each level: execute (sync or parallel)
    | 4. Collect results, errors --> AnalysisReport
    |
    | Graceful degradation: if analyzer X fails,
    | continue with others, report partial results
```

## Critical Design Decisions and Tensions

### CRITICAL: Token Cannot Be Copy With SmallVec

**Issue:** The technical spec declares `Token` as `#[derive(Copy, Clone)]` but includes `original_spans: SmallVec<[Span; 1]>`. SmallVec does NOT implement Copy because it can heap-allocate when more than N elements are stored. This is a compile-time error.

**Resolution options (in order of recommendation):**

1. **Replace SmallVec with a single Span** (recommended): Most tokens map 1:1 to a single original span. Use `original_span: Span` (singular) and handle the rare multi-span case separately. This preserves Copy.

2. **Replace SmallVec with a fixed-size array**: Use `original_spans: [Span; 2]` with a `span_count: u8` field. Copy-able, but wastes space when count is 1 (the common case).

3. **Drop Copy, keep Clone**: Use `SmallVec<[Span; 1]>` as designed, but Token becomes Clone-only. This contradicts the design's stated goal of Copy tokens for lock-free concurrency.

4. **Use arena-allocated span slices**: Store spans in the arena and reference them by index/length pair. Token stays Copy but span access requires the arena.

**Recommendation:** Option 1. The design docs state that most normalizations produce 1:1 mappings. For the rare multi-span case, store a flag indicating "complex mapping" and look up the full mapping in CharMapping when needed.

**Confidence:** HIGH -- SmallVec not implementing Copy is a verified fact.

### TextStore Lifetime Safety

**Issue:** The spec uses `&'static str` references inside TextStore's interner, which works because bumpalo arena memory is stable (never moves/frees until the Bump is dropped). However, this requires that StringIds are NEVER used with a different TextStore than the one that created them.

**Mitigation:** The Arc<TextStore> ownership chain (ProcessedText owns Arc<TextStore>, DiffResult owns Arc<ProcessedText>) ensures the TextStore lives as long as any token that references it.

**Confidence:** HIGH -- bumpalo's memory model is well-documented.

### WASM vs Tokio Incompatibility

**Issue:** Tokio's `rt-multi-thread` feature does not compile to `wasm32-unknown-unknown`. WASM futures are `!Send`, while tokio futures require `Send`. The `spawn_blocking` API is unavailable in WASM.

**Resolution:** Use conditional compilation to provide different async implementations:

```rust
// For native targets
#[cfg(not(target_arch = "wasm32"))]
async fn run_parallel_analyzers(...) { /* tokio::spawn + join_all */ }

// For WASM targets
#[cfg(target_arch = "wasm32")]
async fn run_parallel_analyzers(...) { /* sequential execution */ }
```

For v1, WASM support means: core diffing and metrics work synchronously in the browser. Async parallel analysis is native-only.

**Confidence:** HIGH -- tokio WASM limitations are well-documented.

### PyO3 Adapter Boundary

**Issue:** Type conversion overhead between Rust and Python can be 2-50x depending on the type. Calling Python methods from Rust (for plugin evaluation) requires acquiring the GIL.

**Resolution:** Follow the 3-layer pattern:
1. **Rust Core**: Pure Rust, no PyO3 dependency
2. **Adapter Layer**: PyNormalizerAdapter wraps `Py<PyAny>`, implements Rust `Normalizer` trait, manages GIL acquisition
3. **Python-facing Layer**: `#[pyclass]` wrappers around Rust types, `#[pymethods]` for Python API

Minimize boundary crossings: batch operations where possible.

**Confidence:** HIGH -- pattern proven in Polars, cryptography, and other major PyO3 projects.

## Anti-Patterns

### Anti-Pattern 1: Passing TextStore Everywhere

**What people do:** Thread `&TextStore` through every function signature to enable Token text lookups.
**Do this instead:** Store `Arc<TextStore>` in ProcessedText and DiffResult. Functions that need text access receive the containing struct.

### Anti-Pattern 2: Making TextStore Sync via RwLock

**What people do:** Wrap TextStore in `Arc<RwLock<TextStore>>` to allow concurrent interning.
**Do this instead:** Intern all strings during TextProcessor.process() (which takes &mut self). After processing, promote to Arc<TextStore> (immutable). All subsequent phases only read.

### Anti-Pattern 3: Eager Metric Computation

**What people do:** Compute all 20+ text metrics and 15+ pairwise metrics upfront, even when only a subset is needed.
**Do this instead:** Use lazy evaluation. MetricsEngine computes metrics on first access and caches them.

### Anti-Pattern 4: Monolithic Plugin Execution

**What people do:** Run all analyzers sequentially in registration order.
**Do this instead:** Build a dependency DAG, identify independent groups, and execute groups in parallel. But START with sequential execution for correctness, then add parallel execution as an optimization.

### Anti-Pattern 5: Leaking Lifetimes Through the API

**What people do:** Expose `DiffResult<'a>` in the top-level public API, forcing users to manage lifetimes.
**Do this instead:** The API layer should return owned types or use self-referential techniques to hide the lifetime. For PyO3 bindings, always convert to fully-owned Python types.

## Build Order Implications

Based on the dependency graph, the implementation order should be:

```
Phase 1: Foundation (L1)
  TextStore, StringId, Span, CharMapping, Token, Error types
  NO dependencies on other layers.

Phase 2: Text Processing (L2, part 1)
  Normalizer trait + built-in normalizers
  Tokenizer trait + built-in tokenizers
  TextProcessor orchestrator, ProcessedText
  DEPENDS ON: Phase 1

Phase 3: Diff Computation (L2, part 2)
  DiffAlgorithm trait + Myers + Histogram
  DiffComputer, EditOperation, DiffResult
  DEPENDS ON: Phase 1 + Phase 2

Phase 4: Metrics Engine (L2, part 3)
  Metric trait + TextMetrics + PairwiseMetrics
  MetricsEngine with caching, ContentHash
  DEPENDS ON: Phase 1 + Phase 2 + Phase 3

Phase 5: Analysis Framework (L3, part 1)
  AnalyzerPlugin trait, Dependency, AnalysisContext
  PluginRegistry, ExecutionPlanner, AnalysisCoordinator
  Built-in analyzers
  DEPENDS ON: Phase 3 + Phase 4

Phase 6: Orchestration + Config (L3-L4)
  DiffOrchestrator, CacheManager
  ConfigBuilder, Presets, Public convenience API
  DEPENDS ON: All of Phase 2-5

Phase 7: Async + Optimization
  Async APIs, tokio integration
  Parallel analyzer execution, Performance tuning
  DEPENDS ON: Phase 6

Phase 8: Python Bindings (cross-cutting)
  PyO3 module, type bindings, plugin adapters
  Async bridge, error translation, stub generation
  DEPENDS ON: Phase 6+ (stable Rust API)

Phase 9: WASM Target (cross-cutting)
  Feature-gated compilation, sync-only subset
  wasm-bindgen integration
  DEPENDS ON: Phase 6 (stable core)
```

**Key ordering constraint:** Foundation MUST be complete and stable before any L2 work begins. The Token struct design (Copy vs Clone decision, SmallVec resolution) is a blocking decision for Phase 1.

**Parallelizable work:** After Phase 1, text processing (Phase 2) and diff algorithm research/prototyping can proceed in parallel, since diff algorithms only need Token slices (defined in Phase 1).

## Sources

- [bumpalo - Rust arena allocator](https://docs.rs/bumpalo/latest/bumpalo/)
- [imara-diff - Reliably performant diffing](https://docs.rs/imara-diff/latest/imara_diff/)
- [similar - High level diffing library](https://docs.rs/similar)
- [PyO3 user guide](https://pyo3.rs/)
- [Tokio WASM limitations](https://github.com/tokio-rs/tokio/issues/6178)
- [SmallVec documentation](https://docs.rs/smallvec/latest/smallvec/struct.SmallVec.html) -- confirms no Copy impl
- Redline Core DESIGN.md and TECHNICAL_SPEC.md

---
*Architecture research for: Redline Core v1.0*
*Researched: 2026-02-05*
