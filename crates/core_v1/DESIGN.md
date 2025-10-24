# Redline Core v1.0 - Design Document

**Version:** 1.0  
**Author:** Parth Patel  
**Date:** 2025-10-15  
**Status:** Proposed

---

## Executive Summary

Redline Core is a comprehensive text diffing and analysis framework that goes beyond traditional line-based diff tools. The v1 redesign addresses architectural limitations discovered in v0, introduces a cleaner separation of concerns, improves performance through better caching strategies, and provides a more extensible plugin architecture.

### Key Objectives for v1

1. **Modular Architecture**: Clean separation between diff computation, text processing, and semantic analysis
2. **Performance**: Eliminate redundant computations through intelligent caching and lazy evaluation
3. **Extensibility**: Plugin-based analyzer/classifier system with clear dependency management
4. **Type Safety**: Stronger type system with better error handling
5. **Async Support**: Enable async operations for expensive analyses (ML models, external APIs)
6. **Zero-Copy**: Reduce allocations through better lifetime management and string interning

---

## Analysis of v0 Architecture

### Strengths

1. **Layered Pipeline System**: The `TextPipeline` with `NormalizationLayer` provides excellent traceability from normalized text back to original positions
2. **Character Mapping**: The `CharacterMap` bidirectional mapping system is sophisticated and handles complex transformations well
3. **Execution Planning**: The dependency-aware execution system (`ExecutionPlan`) is a great foundation for avoiding redundant metric computation
4. **Comprehensive Metrics**: The `TextMetrics` and `PairwiseMetrics` system centralizes metric computation
5. **Tokenization Abstraction**: Clean `Tokenizer` trait with multiple implementations

### Weaknesses & Pain Points

1. **Overly Coupled Design**
   - `DiffEngine` does too much: normalization, tokenization, diff computation, analysis, classification
   - `DiffResult` mixes diff operations with analysis results and metrics
   - Hard to test individual components in isolation

2. **Unclear Ownership & Lifetimes**
   - Heavy use of `String` and `Vec` allocations throughout
   - `Token` stores both normalized and original text (duplication)
   - No string interning or arena allocation

3. **Inconsistent Abstraction Levels**
   - Some normalizers are simple (Lowercase), others complex (WhitespaceNormalizer)
   - Tokenizers mix concerns (mapping logic + text splitting)
   - Analyzers have two different traits (`SingleDiffAnalyzer`, `MultiDiffAnalyzer`)

4. **Feature Flags Complexity**
   - Optional features (`spacy`, `bert`, `sklearn`) leak through the API
   - Conditional compilation makes the codebase harder to reason about
   - Feature-gated analyzers don't compose well

5. **Limited Async Support**
   - All operations are synchronous
   - No way to run expensive ML models without blocking
   - Can't parallelize independent analyzers

6. **Error Handling**
   - Most functions return `Result` types only in algorithm implementations
   - Many operations that can fail (e.g., Python interop) panic or use unwrap
   - No structured error types

7. **Config Ergonomics**
   - `DiffConfig` has grown too large with many flags
   - Builder pattern is verbose
   - No validation of configuration combinations

8. **Testing & Observability**
   - Hard to inject test doubles for analyzers
   - No instrumentation or metrics collection
   - Limited debugging support for pipeline stages

---

## v1 Core Design Principles

### 1. **Separation of Concerns**

```
┌─────────────────────────────────────────────────────────────┐
│                      User API Layer                          │
│  (High-level convenience functions, builders, presets)       │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                   Orchestration Layer                        │
│  • DiffOrchestrator: Coordinates the full diff workflow     │
│  • AnalysisCoordinator: Manages analyzer execution          │
│  • CacheManager: Handles metric caching & invalidation      │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                    Core Engine Layer                         │
│  • TextProcessor: Normalization + Tokenization              │
│  • DiffComputer: Pure diff algorithm implementations        │
│  • MetricsEngine: Centralized metric computation            │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                   Foundation Layer                           │
│  • TextStore: Arena-allocated strings, string interning     │
│  • CharMapping: Bidirectional position mappings             │
│  • Types: Core data structures (Token, Span, Operation)     │
└─────────────────────────────────────────────────────────────┘
```

### 2. **Plugin Architecture**

Analyzers and classifiers become **plugins** that declare:
- **Dependencies**: What metrics/data they need
- **Capabilities**: What they can compute
- **Cost**: Estimated computational expense
- **Async**: Whether they support async execution

```rust
trait AnalyzerPlugin: Send + Sync {
    fn metadata(&self) -> AnalyzerMetadata;
    fn dependencies(&self) -> Vec<Dependency>;
    fn analyze(&self, ctx: &AnalysisContext) -> Result<AnalysisResult>;
    fn analyze_async(&self, ctx: &AnalysisContext) -> BoxFuture<Result<AnalysisResult>>;
}
```

### 3. **Smart Caching with Dependency Tracking**

```rust
struct CacheManager {
    // Metrics cache keyed by content hash
    metrics: HashMap<ContentHash, Arc<ComputedMetrics>>,
    
    // Dependency graph for invalidation
    dependencies: DependencyGraph,
    
    // LRU eviction policy
    eviction: LRUCache,
}
```

### 4. **Zero-Copy String Handling**

```rust
struct TextStore {
    // Arena allocator for strings
    arena: Arena<str>,
    
    // String interning for deduplication
    interner: StringInterner,
    
    // Reference-counted strings for shared ownership
    shared: HashMap<StringId, Arc<str>>,
}

// Tokens reference strings by ID, not by value
struct Token {
    text_id: StringId,
    span: Span,
    // ...
}
```

### 5. **Async-First Design**

```rust
// Sync API (default)
let diff = engine.diff(original, modified)?;

// Async API (for expensive operations)
let diff = engine.diff_async(original, modified).await?;

// Parallel analysis
let results = engine
    .analyze_parallel(diff, analyzers)
    .await?;
```

### 6. **Structured Error Handling**

```rust
#[derive(Debug, thiserror::Error)]
enum RedlineError {
    #[error("Normalization failed: {0}")]
    NormalizationError(String),
    
    #[error("Tokenization failed: {0}")]
    TokenizationError(String),
    
    #[error("Analyzer '{name}' failed: {source}")]
    AnalyzerError {
        name: String,
        source: Box<dyn Error>,
    },
    
    #[error("Dependency not satisfied: {0}")]
    DependencyError(String),
    
    #[error("Configuration invalid: {0}")]
    ConfigError(String),
}
```

---

## v1 Module Architecture

### Module Structure

```
redline-core-v1/
├── src/
│   ├── lib.rs                 # Public API, re-exports
│   │
│   ├── foundation/            # Core types and utilities
│   │   ├── mod.rs
│   │   ├── text_store.rs      # Arena allocation, string interning
│   │   ├── span.rs            # Span, Position types
│   │   ├── char_map.rs        # Character position mapping
│   │   └── types.rs           # Common types (EditOperation, etc.)
│   │
│   ├── text/                  # Text processing
│   │   ├── mod.rs
│   │   ├── processor.rs       # TextProcessor orchestrator
│   │   ├── normalizer/        # Normalization
│   │   │   ├── mod.rs
│   │   │   ├── traits.rs
│   │   │   ├── case.rs
│   │   │   ├── whitespace.rs
│   │   │   └── unicode.rs
│   │   ├── tokenizer/         # Tokenization
│   │   │   ├── mod.rs
│   │   │   ├── traits.rs
│   │   │   ├── word.rs
│   │   │   ├── char.rs
│   │   │   └── sentence.rs
│   │   └── pipeline.rs        # Pipeline orchestration
│   │
│   ├── diff/                  # Diff computation
│   │   ├── mod.rs
│   │   ├── computer.rs        # DiffComputer (algorithm dispatcher)
│   │   ├── algorithms/        # Diff algorithms
│   │   │   ├── mod.rs
│   │   │   ├── myers.rs
│   │   │   ├── histogram.rs
│   │   │   └── patience.rs
│   │   ├── result.rs          # DiffResult type
│   │   └── operations.rs      # Edit operations
│   │
│   ├── metrics/               # Metric computation
│   │   ├── mod.rs
│   │   ├── engine.rs          # MetricsEngine
│   │   ├── text_metrics.rs    # Single-text metrics
│   │   ├── pairwise.rs        # Comparison metrics
│   │   └── registry.rs        # Metric registry for plugins
│   │
│   ├── analysis/              # Analysis & classification
│   │   ├── mod.rs
│   │   ├── coordinator.rs     # AnalysisCoordinator
│   │   ├── plugin.rs          # Plugin trait & metadata
│   │   ├── context.rs         # AnalysisContext
│   │   ├── result.rs          # AnalysisResult
│   │   ├── plugins/           # Built-in plugins
│   │   │   ├── mod.rs
│   │   │   ├── semantic.rs
│   │   │   ├── stylistic.rs
│   │   │   ├── readability.rs
│   │   │   └── classification.rs
│   │   └── registry.rs        # Plugin registry
│   │
│   ├── cache/                 # Caching layer
│   │   ├── mod.rs
│   │   ├── manager.rs         # CacheManager
│   │   ├── policy.rs          # Eviction policies
│   │   └── hash.rs            # Content hashing
│   │
│   ├── orchestration/         # High-level orchestration
│   │   ├── mod.rs
│   │   ├── diff_orchestrator.rs
│   │   └── execution_plan.rs  # Dependency-aware execution
│   │
│   ├── config/                # Configuration
│   │   ├── mod.rs
│   │   ├── builder.rs         # Fluent builder
│   │   ├── presets.rs         # Common presets
│   │   └── validation.rs      # Config validation
│   │
│   ├── error.rs               # Error types
│   └── util/                  # Utilities
│       ├── mod.rs
│       └── string.rs
│
└── benches/                   # Benchmarks
    ├── diff_algorithms.rs
    └── text_processing.rs
```

---

## Key Component Designs

### 1. TextStore (Zero-Copy String Management)

**Purpose**: Eliminate string duplication and reduce allocations.

```rust
/// Manages text storage with interning and arena allocation
pub struct TextStore {
    arena: Arena<str>,
    interner: StringInterner,
}

/// ID for an interned string
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct StringId(u32);

impl TextStore {
    /// Intern a string, returning its ID
    pub fn intern(&mut self, s: &str) -> StringId;
    
    /// Get a string by ID
    pub fn get(&self, id: StringId) -> &str;
    
    /// Allocate in arena (for unique strings)
    pub fn alloc(&mut self, s: &str) -> &str;
}
```

**Benefits**:
- Tokens reference strings by ID (4 bytes) instead of copying strings
- Common strings (stopwords, punctuation) only stored once
- Faster equality checks (compare IDs instead of string content)

### 2. TextProcessor (Unified Text Processing)

**Purpose**: Single entry point for normalization + tokenization.

```rust
pub struct TextProcessor {
    normalizers: Vec<Box<dyn Normalizer>>,
    tokenizer: Box<dyn Tokenizer>,
    text_store: TextStore,
}

impl TextProcessor {
    /// Process text through pipeline, returning tokens and mappings
    pub fn process(&mut self, text: &str) -> ProcessedText;
}

pub struct ProcessedText {
    /// Interned tokens
    pub tokens: Vec<Token>,
    
    /// Normalization layers with mappings
    pub layers: Vec<Layer>,
    
    /// Reference to text store
    pub text_store: Arc<TextStore>,
}
```

### 3. DiffComputer (Pure Diff Computation)

**Purpose**: Only compute edit operations, no side effects.

```rust
pub struct DiffComputer {
    algorithm: DiffAlgorithm,
}

impl DiffComputer {
    /// Compute diff operations
    pub fn compute(
        &self,
        original: &ProcessedText,
        modified: &ProcessedText,
    ) -> Result<Vec<EditOperation>>;
}
```

### 4. MetricsEngine (Centralized Metrics)

**Purpose**: Compute and cache all text metrics.

```rust
pub struct MetricsEngine {
    cache: CacheManager,
}

impl MetricsEngine {
    /// Compute metrics for a single text
    pub fn compute_text_metrics(&self, text: &str) -> Arc<TextMetrics>;
    
    /// Compute pairwise metrics
    pub fn compute_pairwise_metrics(
        &self,
        original: &str,
        modified: &str,
    ) -> Arc<PairwiseMetrics>;
    
    /// Get or compute a specific metric
    pub fn get_metric<M: Metric>(
        &self,
        key: &MetricKey,
    ) -> Result<M::Output>;
}
```

### 5. AnalysisCoordinator (Plugin Management)

**Purpose**: Execute analyzers in dependency order, with caching.

```rust
pub struct AnalysisCoordinator {
    registry: PluginRegistry,
    cache: CacheManager,
}

impl AnalysisCoordinator {
    /// Register an analyzer plugin
    pub fn register(&mut self, plugin: Box<dyn AnalyzerPlugin>);
    
    /// Analyze a diff with selected plugins
    pub fn analyze(
        &self,
        diff: &DiffResult,
        plugins: &[PluginId],
    ) -> Result<AnalysisReport>;
    
    /// Async version with parallel execution
    pub async fn analyze_async(
        &self,
        diff: &DiffResult,
        plugins: &[PluginId],
    ) -> Result<AnalysisReport>;
}
```

### 6. DiffOrchestrator (High-Level API)

**Purpose**: Coordinate the entire diff + analysis workflow.

```rust
pub struct DiffOrchestrator {
    text_processor: TextProcessor,
    diff_computer: DiffComputer,
    metrics_engine: MetricsEngine,
    analysis_coordinator: AnalysisCoordinator,
}

impl DiffOrchestrator {
    /// Full diff with analysis
    pub fn diff_with_analysis(
        &mut self,
        original: &str,
        modified: &str,
        config: &DiffConfig,
    ) -> Result<DiffReport>;
    
    /// Async version
    pub async fn diff_with_analysis_async(
        &mut self,
        original: &str,
        modified: &str,
        config: &DiffConfig,
    ) -> Result<DiffReport>;
}
```

---

## Data Model Changes

### Core Types

```rust
/// Immutable reference to text in the store
#[derive(Copy, Clone)]
pub struct TextRef<'store> {
    id: StringId,
    store: &'store TextStore,
}

/// A token with zero-copy text reference
pub struct Token<'store> {
    pub text: TextRef<'store>,
    pub span: Span,
    pub index: usize,
    pub kind: TokenKind,
}

/// Span in text (byte positions)
#[derive(Copy, Clone, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Edit operation (now generic over token lifetime)
pub struct EditOperation<'store> {
    pub kind: EditKind,
    pub original_tokens: &'store [Token<'store>],
    pub modified_tokens: &'store [Token<'store>],
    pub original_span: Option<Span>,
    pub modified_span: Option<Span>,
}

/// Complete diff result
pub struct DiffResult<'store> {
    pub operations: Vec<EditOperation<'store>>,
    pub statistics: DiffStatistics,
    pub original: ProcessedText<'store>,
    pub modified: ProcessedText<'store>,
}

/// Analysis report
pub struct DiffReport {
    pub diff: DiffResult<'static>, // Owned version
    pub metrics: Arc<PairwiseMetrics>,
    pub analysis_results: Vec<AnalysisResult>,
}
```

---

## Configuration System

### Improved Builder Pattern

```rust
// Old (v0) - verbose and error-prone
let config = DiffConfig::new()
    .with_algorithm(DiffAlgorithm::Histogram)
    .with_pipeline(
        TextPipeline::new()
            .add_normalizer(Box::new(Lowercase))
    )
    .with_tokenizer(Box::new(WordTokenizer::new()))
    .with_semantic_similarity(true)
    .with_edit_classification(true);

// New (v1) - cleaner with presets and validation
let config = DiffConfig::builder()
    .preset(DiffPreset::Semantic)  // Includes common settings
    .algorithm(Algorithm::Histogram)
    .normalize(|n| n.lowercase().collapse_whitespace())
    .tokenize(Tokenization::Words)
    .analyze(|a| a
        .semantic()
        .readability()
        .classification()
    )
    .build()?;  // Validates configuration

// Or use presets directly
let config = DiffConfig::preset(DiffPreset::FastSyntactic);
```

### Configuration Presets

```rust
pub enum DiffPreset {
    /// Minimal diff, no analysis (fastest)
    Fast,
    
    /// Syntactic comparison (case-insensitive, whitespace normalized)
    Syntactic,
    
    /// Semantic comparison (word overlap, similarity metrics)
    Semantic,
    
    /// Full analysis (all features enabled)
    Comprehensive,
    
    /// Custom from JSON/TOML
    Custom(PathBuf),
}
```

---

## Performance Optimizations

### 1. Lazy Evaluation

```rust
pub struct LazyMetrics {
    // Metrics computed on first access
    char_similarity: OnceCell<f64>,
    word_overlap: OnceCell<f64>,
    // ...
}
```

### 2. Parallel Analysis

```rust
// Analyze with multiple plugins in parallel
let results = coordinator
    .analyze_parallel(diff, &[
        PluginId::Semantic,
        PluginId::Stylistic,
        PluginId::Readability,
    ])
    .await?;
```

### 3. Smart Caching

```rust
// Content-addressed caching
let metrics = cache.get_or_compute(
    ContentHash::of(text),
    || expensive_computation(text)
);
```

### 4. String Interning

```rust
// Common strings stored once
let stopwords = ["the", "a", "an", "is", "are"];
for word in stopwords {
    text_store.intern(word);
}
```

---

## Migration Path from v0 to v1

### Compatibility Layer

Provide a compatibility shim for v0 users:

```rust
// v0 API (deprecated but supported)
pub mod compat {
    pub use crate::DiffEngine as DiffEngineV1;
    
    /// Compatibility wrapper for v0 API
    pub struct DiffEngine {
        inner: DiffEngineV1,
    }
    
    impl DiffEngine {
        pub fn diff(&self, original: &str, modified: &str) -> DiffResult {
            // Convert v1 types to v0 types
            self.inner.diff(original, modified)
                .map(|r| r.into_v0())
                .unwrap()
        }
    }
}
```

### Migration Guide

1. **Phase 1**: Dual support (v0 + v1 APIs coexist)
2. **Phase 2**: Deprecation warnings for v0 API
3. **Phase 3**: Remove v0 API (breaking change, major version bump)

---

## Testing Strategy

### 1. Unit Tests
- Each component tested in isolation
- Mock dependencies with traits

### 2. Integration Tests
- Full pipeline tests
- Analyzer integration tests

### 3. Property-Based Tests
- Use `proptest` for fuzzing
- Invariant testing (e.g., diff reversibility)

### 4. Performance Tests
- Benchmark suite for critical paths
- Regression detection

### 5. Compatibility Tests
- Ensure v0 compatibility layer works

---

## Success Metrics

1. **Performance**: 2x faster than v0 for typical use cases
2. **Memory**: 50% reduction in allocations
3. **Extensibility**: Add new analyzer in <100 LOC
4. **Maintainability**: Reduced cyclomatic complexity by 30%
5. **Type Safety**: Zero `unwrap()` calls in library code

---

## Open Questions

1. Should we support streaming diffs for large files?
2. How to handle plugins that require external resources (models, dictionaries)?
3. Should we provide a WASM build for browser usage?
4. Do we need a query language for filtering/selecting operations?

---

## Next Steps

1. ✅ Review and approve design document
2. ⏳ Write technical specification (detailed API design)
3. ⏳ Prototype core components (TextStore, TextProcessor)
4. ⏳ Implement diff algorithms
5. ⏳ Build plugin system
6. ⏳ Write comprehensive tests
7. ⏳ Performance benchmarking
8. ⏳ Documentation
9. ⏳ Migration guide
10. ⏳ Release v1.0.0

---

## Appendix

### A. Dependency Graph Example

```
SemanticAnalyzer
    └─ WordOverlap (metric)
        ├─ WordCount (original)
        └─ WordCount (modified)

StylisticAnalyzer
    └─ ReadabilityDiff (metric)
        ├─ FleschScore (original)
        └─ FleschScore (modified)
            └─ SyllableCount
```

### B. Performance Benchmarks (Target)

| Operation | v0 | v1 (target) | Improvement |
|-----------|-----|-------------|-------------|
| Simple diff (100 words) | 1.2ms | 0.6ms | 2x |
| With normalization | 2.5ms | 1.0ms | 2.5x |
| Full analysis | 50ms | 20ms | 2.5x |
| 10K word document | 500ms | 150ms | 3.3x |

---

**Document Status**: Draft for Review  
**Review By**: [Team/Stakeholder]  
**Approval Date**: TBD
