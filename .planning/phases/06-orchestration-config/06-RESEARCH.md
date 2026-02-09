# Phase 6: Orchestration & Configuration - Research

**Researched:** 2026-02-09
**Domain:** Pipeline orchestration, configuration builders, query combinators, caching
**Confidence:** HIGH

## Summary

Phase 6 creates the public-facing `Redline` API that wires together all existing pipeline stages (TextProcessor, DiffComputer, MetricsEngine, AnalysisCoordinator) into a single entry point with unified configuration, presets, caching, and a query API.

The codebase already has all four pipeline stages fully implemented and tested (935 tests). The orchestrator's job is purely coordination -- no new algorithms, just wiring, configuration, and filtering. All existing traits (Normalizer, Tokenizer, DiffAlgorithm, Metric, AnalyzerPlugin) are `Send + Sync`, which makes the `parking_lot::RwLock`-based cache straightforward.

The key technical challenges are: (1) designing a ConfigBuilder that validates combinations of settings, (2) implementing predicate combinators with `&`/`|`/`!` operator overloads for the query API, (3) building a content-addressed pipeline cache with LRU eviction, and (4) ensuring the `Redline` struct itself is `Send + Sync` for downstream Python/WASM phases.

**Primary recommendation:** Build the phase in layers: config types first, then the cache, then the `Redline` orchestrator, then the query/filter API, then presets, then integration tests. Each layer has clean boundaries and can be tested independently.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Pipeline API Design:**
- Main entry point is `Redline` (not `DiffOrchestrator`) -- `Redline::new(config).diff("old", "new")`
- Config at construction sets defaults, per-call overrides for one-off tweaks -- both patterns supported
- Single `RedlineResult` struct returned from every call -- contains diff, metrics, and analysis report; users access what they need via fields
- Pluggable stages -- users can inject custom `impl Tokenizer`, `impl Normalizer`, etc. at construction time
- Built-in stages configured via config, custom implementations registered alongside built-ins

**Presets:**
- 2 presets only (not 4): Fast and Comprehensive
- Fast -- diff + basic metrics (counts, similarity), no analysis
- Comprehensive -- everything: full normalization, diff, all metrics, all analyzers (this is the default)
- Presets are starting points, not fixed -- `ConfigBuilder::preset(Fast).with_analysis(true)` overrides individual settings
- Users who want something in between start from either preset and override

**Query API:**
- Primary consumer: programmatic callers first, could power a UI later
- Predicate combinators with operator overloads -- `Filter::kind(Replace) & Filter::min_length(10)`
- `&` for AND, `|` for OR, `!` for NOT
- Filter dimensions: EditKind, span range, token count, text content matching, and analysis metadata (e.g., IntentCategory from EditClassifier)
- Returns an iterator -- lazy, chainable, zero allocation for early termination; users `.collect()` when they need a concrete collection

**Cache Strategy:**
- Full pipeline memoization -- metrics, diff results, and analysis reports all cached by content hash
- Per-instance cache -- each `Redline` instance owns its cache, no cross-instance sharing
- LRU eviction -- entry count-based by default, with optional memory cap as safety valve
- Enabled by default -- consistent with Comprehensive default; users disable for deterministic memory or one-shot processing

### Claude's Discretion
- ConfigBuilder fluent API shape and validation details
- Internal cache key structure and hashing strategy
- Memory measurement approach for optional memory cap
- Error types and error handling for invalid configurations
- CacheManager lock granularity and parking_lot usage patterns

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| parking_lot | 0.12 | RwLock for CacheManager | Faster, smaller than std RwLock; no poisoning; fair scheduling. Already decided in STATE.md. |
| lru | 0.16 | LRU cache eviction | Already a dependency in Cargo.toml. Used by existing MetricCache. |
| foldhash | 0.1 | Content-addressed hashing | Already a dependency. `FixedState` with seed 0 used by existing `ContentHash`. |
| thiserror | 2.0 | Error types | Already a dependency. Used throughout codebase for error enums. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashbrown | 0.16 | HashMap/HashSet internals | Already a dependency. Use for internal maps in config, cache keys. |

### Not Needed

| Library | Why Not |
|---------|---------|
| cached | We build our own LRU cache -- already have the pattern from MetricCache |
| derive_builder | Overkill for 2 presets + fluent API; hand-rolled builder is clearer |
| predicates (crate) | We only need the *pattern*, not the full predicate framework |

**Dependency addition:** Only `parking_lot = "0.12"` needs to be added to Cargo.toml. Everything else is already present.

## Architecture Patterns

### Recommended Module Structure

```
crates/core/src/
├── orchestrate/           # NEW: Phase 6 module
│   ├── mod.rs             # Redline struct, public API
│   ├── config.rs          # RedlineConfig, ConfigBuilder, Preset enum
│   ├── cache.rs           # CacheManager with parking_lot::RwLock<LruCache>
│   ├── result.rs          # RedlineResult struct
│   ├── filter.rs          # Filter predicate combinators + operator overloads
│   └── error.rs           # OrchestrateError (or extend RedlineError)
├── ... (existing modules unchanged)
```

### Pattern 1: ConfigBuilder with Validation

**What:** Fluent builder that starts from a preset and allows field-level overrides. Validation happens at `build()` time, returning `Result<RedlineConfig, ConfigError>`.

**Why not typestate:** Typestate builders enforce required fields at compile time, but our config has *no* required fields -- everything has defaults from presets. A runtime-validated builder with `build() -> Result` is simpler and sufficient.

**Recommended shape:**

```rust
pub enum Preset {
    Fast,
    Comprehensive,
}

pub struct ConfigBuilder {
    // All fields are Option<T> -- None means "use preset default"
    preset: Preset,                           // default: Comprehensive
    normalizers: Option<Vec<Box<dyn Normalizer>>>,
    tokenizer: Option<Box<dyn Tokenizer>>,
    algorithm: Option<Box<dyn DiffAlgorithm>>,
    metrics_enabled: Option<bool>,
    analysis_enabled: Option<bool>,
    custom_metrics: Vec<Box<dyn Metric>>,
    custom_analyzers: Vec<Box<dyn AnalyzerPlugin>>,
    cache_enabled: Option<bool>,
    cache_capacity: Option<usize>,
    cache_memory_limit: Option<usize>,        // bytes, optional safety valve
}

impl ConfigBuilder {
    pub fn new() -> Self { /* Comprehensive defaults */ }
    pub fn preset(preset: Preset) -> Self { /* start from preset */ }
    pub fn with_normalizers(mut self, n: Vec<Box<dyn Normalizer>>) -> Self { ... }
    pub fn with_tokenizer(mut self, t: Box<dyn Tokenizer>) -> Self { ... }
    pub fn with_algorithm(mut self, a: Box<dyn DiffAlgorithm>) -> Self { ... }
    pub fn with_analysis(mut self, enabled: bool) -> Self { ... }
    pub fn with_metrics(mut self, enabled: bool) -> Self { ... }
    pub fn with_cache(mut self, enabled: bool) -> Self { ... }
    pub fn with_cache_capacity(mut self, cap: usize) -> Self { ... }
    pub fn add_metric(mut self, m: Box<dyn Metric>) -> Self { ... }
    pub fn add_analyzer(mut self, a: Box<dyn AnalyzerPlugin>) -> Self { ... }
    pub fn build(self) -> Result<RedlineConfig, ConfigError> { ... }
}
```

**Validation at `build()` time:**
- Cache capacity must be > 0 if cache is enabled
- Custom normalizer names must be unique (delegate to TextProcessor::new)
- Custom metric IDs must not collide with built-ins (unless overriding is explicit)
- Memory limit must be >= cache capacity * estimated entry size (warn, don't fail)

### Pattern 2: Predicate Combinators with Operator Overloads

**What:** A `Filter` enum/struct that implements `BitAnd`, `BitOr`, and `Not` from `std::ops`, producing `AndFilter`, `OrFilter`, `NotFilter` wrapper types. All implement a common `FilterPredicate` trait with `fn matches(&self, op: &EditOperation, context: &FilterContext) -> bool`.

**Key insight from predicates crate:** The pattern uses concrete wrapper structs rather than trait objects. This enables zero-cost abstraction -- the compiler can inline and optimize predicate chains.

**Recommended shape:**

```rust
/// Trait for filter predicates
pub trait FilterPredicate {
    fn matches(&self, op: &EditOperation, context: &FilterContext<'_>) -> bool;
}

/// Leaf filter constructors
pub struct Filter;

impl Filter {
    pub fn kind(kind: EditKind) -> KindFilter { KindFilter(kind) }
    pub fn min_length(len: u32) -> MinLengthFilter { MinLengthFilter(len) }
    pub fn source_contains(text: &str) -> SourceContainsFilter { ... }
    pub fn intent(cat: IntentCategory) -> IntentFilter { IntentFilter(cat) }
    pub fn span_range(start: u32, end: u32) -> SpanRangeFilter { ... }
}

/// Concrete filter types
pub struct KindFilter(EditKind);
pub struct MinLengthFilter(u32);
pub struct IntentFilter(IntentCategory);
// ... more leaf types

/// Combinator wrappers
pub struct AndFilter<L, R>(L, R);
pub struct OrFilter<L, R>(L, R);
pub struct NotFilter<F>(F);

/// Operator overloads for all FilterPredicate implementors
impl<L: FilterPredicate, R: FilterPredicate> std::ops::BitAnd<R> for L {
    type Output = AndFilter<L, R>;
    fn bitand(self, rhs: R) -> AndFilter<L, R> { AndFilter(self, rhs) }
}
// Similar for BitOr and Not
```

**Important:** The `&`/`|`/`!` operators work on *values*, not references. Each combinator consumes its operands and produces a new combinator. This is the standard Rust pattern (see `predicates` crate, `combine` parser combinators).

**FilterContext** provides access to analysis metadata needed for intent filtering:

```rust
pub struct FilterContext<'a> {
    pub diff_result: &'a DiffResult,
    pub analysis_report: Option<&'a AnalysisReport>,
}
```

**Blanket impl problem:** You cannot write `impl<L: FilterPredicate, R: FilterPredicate> BitAnd<R> for L` because it conflicts with orphan rules. Instead, implement `BitAnd` individually for each concrete filter type, or use a wrapper newtype. The recommended approach: define a `Pred<T>` wrapper that implements the operators, and have all leaf filters wrapped in `Pred<T>`.

**Alternative (simpler, recommended):** Use an enum-based approach instead of generics:

```rust
pub enum Filter {
    Kind(EditKind),
    MinLength(u32),
    MaxLength(u32),
    SourceContains(String),
    TargetContains(String),
    SpanRange { start: u32, end: u32 },
    TokenCount { min: u32, max: u32 },
    Intent(IntentCategory),
    And(Box<Filter>, Box<Filter>),
    Or(Box<Filter>, Box<Filter>),
    Not(Box<Filter>),
}

impl std::ops::BitAnd for Filter {
    type Output = Filter;
    fn bitand(self, rhs: Filter) -> Filter {
        Filter::And(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::BitOr for Filter {
    type Output = Filter;
    fn bitor(self, rhs: Filter) -> Filter {
        Filter::Or(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Not for Filter {
    type Output = Filter;
    fn not(self) -> Filter {
        Filter::Not(Box::new(self))
    }
}
```

**Recommendation: Use the enum approach.** It is simpler, avoids orphan rule issues, avoids generic type explosion, and the Box allocation is negligible for filter construction (filters are built once, evaluated many times). The enum also naturally supports Debug, Clone, and serialization.

### Pattern 3: Content-Addressed Pipeline Cache

**What:** A `CacheManager` wrapping `parking_lot::RwLock<LruCache<PipelineCacheKey, CachedResult>>` where the key is derived from content hashing the input texts + configuration hash.

**Recommended key structure:**

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineCacheKey {
    /// Hash of (source_original, target_original, config_hash)
    hash: u64,
}
```

Use the same `foldhash::fast::FixedState` with seed 0 pattern already established by `ContentHash` in `metrics/cache.rs`. Hash the original texts (not normalized -- normalization is part of what we're caching) plus a configuration fingerprint.

**Configuration fingerprint:** Hash the normalizer names (in order), tokenizer name, algorithm name, metrics_enabled flag, analysis_enabled flag. This ensures config changes invalidate the cache.

**CachedResult:**

```rust
struct CachedResult {
    diff_result: Arc<DiffResult>,      // Already uses Arc<ProcessedText> internally
    metrics: HashMap<String, MetricValue>,
    analysis_report: Option<AnalysisReport>,
    // For memory cap tracking
    estimated_bytes: usize,
}
```

**Lock granularity:** Use `parking_lot::RwLock` (not `Mutex`). Cache lookups (reads) are far more frequent than inserts (writes) when the cache is warm. A single RwLock on the entire LRU is sufficient -- contention is low because `diff()` calls are not sub-millisecond; the lock is held only for the brief hash lookup/insert, not during computation.

**Memory cap approach:** Track `estimated_bytes` per entry (sum of string lengths + token counts * sizeof(Token) + fixed overhead). On insert, if total exceeds the memory limit, evict LRU entries until under the cap. This is approximate but sufficient as a safety valve. The entry-count-based LRU is the primary eviction mechanism.

### Pattern 4: Redline Orchestrator Pipeline

**What:** The `Redline` struct owns the config, cache, and constructed pipeline stages. The `diff()` method orchestrates the full pipeline.

**Pipeline flow:**

```
diff("old", "new")
  1. Hash inputs + config -> PipelineCacheKey
  2. Check cache (RwLock read)
  3. If hit -> return cached RedlineResult
  4. If miss:
     a. TextProcessor::process(old) -> Arc<ProcessedText>
     b. TextProcessor::process(new) -> Arc<ProcessedText>
     c. DiffComputer::compute(source, target) -> DiffResult
     d. If metrics_enabled:
        MetricsEngine::get_many(all_metric_ids, input) -> HashMap<String, MetricValue>
     e. If analysis_enabled:
        AnalysisCoordinator::run(context) -> AnalysisReport
     f. Build RedlineResult
     g. Insert into cache (RwLock write)
     h. Return RedlineResult
```

**Per-call overrides:** `diff_with("old", "new", overrides)` method that takes a lightweight `DiffOptions` struct for one-off tweaks without rebuilding the full pipeline. Overrides could include: different algorithm, skip analysis, filter metrics. The override only affects that single call -- it does not modify the `Redline` instance.

**RedlineResult structure:**

```rust
pub struct RedlineResult {
    pub diff: DiffResult,                     // Always present
    pub metrics: HashMap<String, MetricValue>, // Empty if metrics disabled
    pub analysis: Option<AnalysisReport>,      // None if analysis disabled
}

impl RedlineResult {
    /// Filter operations using a predicate combinator
    pub fn filter(&self, predicate: &Filter) -> impl Iterator<Item = &EditOperation> {
        let context = FilterContext {
            diff_result: &self.diff,
            analysis_report: self.analysis.as_ref(),
        };
        self.diff.operations.iter().filter(move |op| {
            predicate.matches(op, &context)
        })
    }
}
```

### Anti-Patterns to Avoid

- **Global/shared cache:** User decided per-instance cache. Do NOT create a global cache or allow cross-instance sharing. Each `Redline` instance is self-contained.
- **Typestate builder overkill:** All config fields have defaults. A runtime-validated builder with `build() -> Result` is the right choice, not compile-time typestate.
- **Generic type explosion in filters:** Trait-based generic combinators lead to complex type signatures (`AndFilter<KindFilter, OrFilter<MinLengthFilter, NotFilter<IntentFilter>>>`). The enum approach keeps types simple.
- **Holding RwLock during computation:** The cache lock must only be held for lookup/insert, never during the actual diff/metrics/analysis computation. Compute first, then insert.
- **Caching normalized text separately:** The pipeline cache should cache the *entire* result (ProcessedText + DiffResult + metrics + analysis), not individual stages. The content hash of the inputs + config is the key.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RwLock | Custom lock-free cache | `parking_lot::RwLock<LruCache>` | Proven, fast, no poisoning; lock-free would be premature optimization |
| LRU eviction | Custom linked-list LRU | `lru::LruCache` (already in deps) | O(1) operations, battle-tested, already used by MetricCache |
| Content hashing | Custom hash function | `foldhash::fast::FixedState` (already in deps) | Deterministic, fast, already proven pattern in codebase |
| Error types | Manual Display impl | `thiserror` derive macros | Already used throughout; consistent error handling |
| Built-in metric registration | Manual metric construction | `register_builtins()` (existing) | Already registers all 37 metrics with validation |
| Built-in analyzer registration | Manual analyzer construction | `register_builtin_analyzers()` (existing) | Already registers all 4 analyzers |

**Key insight:** The codebase already has `register_builtins()` for metrics (37 metrics) and `register_builtin_analyzers()` for analysis (4 analyzers). The orchestrator's config/preset logic just needs to call these existing functions, not rebuild the registration logic.

## Common Pitfalls

### Pitfall 1: MetricsEngine Uses RefCell (Not Send)

**What goes wrong:** `MetricsEngine` uses `RefCell<MetricCache>` internally for interior mutability. `RefCell` is `!Sync`, meaning `MetricsEngine` cannot be shared across threads directly.

**Why it matters:** The `Redline` struct must be `Send + Sync` for downstream Python/WASM phases. If `Redline` owns a `MetricsEngine`, it becomes `!Sync`.

**How to avoid:** The orchestrator should create a fresh `MetricsEngine` per `diff()` call (they are cheap -- just registry + optional cache). Alternatively, the MetricsEngine could be wrapped in the pipeline cache (the cached result includes computed metrics, so the engine is only needed for cache misses). The registry (`MetricRegistry`) can be shared (it only holds `Box<dyn Metric>` which are `Send + Sync`).

**Warning signs:** Compiler errors about `RefCell<_>` not implementing `Sync` when trying to store `MetricsEngine` in `Redline`.

### Pitfall 2: AnalysisReport Contains `dyn Any` (Not Clone)

**What goes wrong:** `AnalysisReport` stores `HashMap<String, Box<dyn Any + Send + Sync>>`. `dyn Any` is not `Clone`, so `AnalysisReport` cannot be `Clone`. This means cached results cannot simply be cloned out of the cache.

**Why it matters:** The pipeline cache needs to return results without moving them out. If the result contains a non-cloneable `AnalysisReport`, we need to wrap it in `Arc`.

**How to avoid:** Wrap the cached result in `Arc<CachedResult>`. The cache stores `Arc<CachedResult>` and returns clones of the `Arc` (cheap reference count increment). The `RedlineResult` returned to users can either own the data or hold `Arc` references. Since `DiffResult` already uses `Arc<ProcessedText>` internally, this pattern is consistent.

**Warning signs:** Cannot implement `Clone` for `RedlineResult` if it directly owns an `AnalysisReport`.

### Pitfall 3: Filter Context Lifetime Coupling

**What goes wrong:** The `filter()` method on `RedlineResult` needs to create a `FilterContext` that borrows `DiffResult` and `AnalysisReport`. If the filter method returns an iterator that borrows the context, lifetime management becomes complex.

**Why it matters:** Users expect `result.filter(predicate).collect::<Vec<_>>()` to work seamlessly.

**How to avoid:** The `filter()` method should create the `FilterContext` inside a closure that is captured by the iterator. Use `move` closures to transfer ownership of the context into the iterator. The `FilterContext` only borrows from `RedlineResult`, and the returned iterator borrows from `RedlineResult` too, so the lifetimes align naturally.

### Pitfall 4: Config Fingerprint Instability

**What goes wrong:** If the config fingerprint includes pointer addresses or HashMap iteration order, the same logical config produces different hashes, causing cache misses.

**Why it matters:** Cache hit rate drops to zero if config hashing is non-deterministic.

**How to avoid:** Hash only stable, order-independent properties: normalizer names (sorted), tokenizer name, algorithm name, boolean flags. Use the existing `foldhash::fast::FixedState` with seed 0 for deterministic hashing. Test that the same config always produces the same fingerprint.

### Pitfall 5: Preset Override Ordering

**What goes wrong:** If `ConfigBuilder::preset(Fast).with_analysis(true)` doesn't properly override the Fast preset's `analysis_enabled = false`, users get unexpected behavior.

**Why it matters:** Presets are "starting points, not fixed" per user decision. Overrides MUST win.

**How to avoid:** The builder stores `Option<T>` for every field. `build()` first expands the preset into a concrete config, then overlays any `Some(value)` from the builder's fields. This two-phase approach (preset expansion, then override application) ensures overrides always win. Test this explicitly.

### Pitfall 6: ProcessedText Ownership in Pipeline

**What goes wrong:** `DiffComputer::compute()` takes `Arc<ProcessedText>` for source and target. `TextProcessor::process()` returns `ProcessedText` (owned). The orchestrator must wrap in `Arc` between these steps.

**Why it matters:** Missing the `Arc::new()` wrapper causes type mismatch errors.

**How to avoid:** After `TextProcessor::process()`, immediately wrap: `let source = Arc::new(processor.process(old)?);`. This is already the pattern used in all existing tests.

## Code Examples

### Example 1: CacheManager with parking_lot::RwLock

```rust
use parking_lot::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct CacheManager {
    cache: RwLock<LruCache<PipelineCacheKey, Arc<CachedResult>>>,
    memory_limit: Option<usize>,
    total_bytes: RwLock<usize>,  // track memory usage separately
}

impl CacheManager {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("cache capacity must be > 0")
            )),
            memory_limit: None,
            total_bytes: RwLock::new(0),
        }
    }

    pub fn get(&self, key: &PipelineCacheKey) -> Option<Arc<CachedResult>> {
        // Read lock -- multiple concurrent readers allowed
        self.cache.write().get(key).cloned()
        // Note: LRU get() updates ordering, so we need write lock
    }

    pub fn insert(&self, key: PipelineCacheKey, value: Arc<CachedResult>) {
        let mut cache = self.cache.write();
        cache.put(key, value);
        // Memory cap eviction would go here
    }

    pub fn clear(&self) {
        self.cache.write().clear();
    }
}
```

**Important note about LRU:** `lru::LruCache::get()` updates the access order (promotes the entry to most-recently-used). This means even "reads" require a write lock. This is inherent to LRU caches. For the pipeline cache this is acceptable because the lock duration is minimal (just a hash lookup + pointer update).

### Example 2: Filter Enum with Operator Overloads

```rust
use std::ops::{BitAnd, BitOr, Not};

#[derive(Debug, Clone)]
pub enum Filter {
    Kind(EditKind),
    MinLength(u32),
    MaxLength(u32),
    SourceContains(String),
    TargetContains(String),
    SpanRange { start: u32, end: u32 },
    TokenCount { min: u32, max: u32 },
    Intent(IntentCategory),
    And(Box<Filter>, Box<Filter>),
    Or(Box<Filter>, Box<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    // Named constructors for ergonomic API
    pub fn kind(kind: EditKind) -> Self { Filter::Kind(kind) }
    pub fn min_length(len: u32) -> Self { Filter::MinLength(len) }
    pub fn intent(cat: IntentCategory) -> Self { Filter::Intent(cat) }

    pub fn matches(&self, op: &EditOperation, ctx: &FilterContext<'_>) -> bool {
        match self {
            Filter::Kind(k) => op.kind == *k,
            Filter::MinLength(n) => op.source_len().max(op.target_len()) >= *n,
            Filter::And(l, r) => l.matches(op, ctx) && r.matches(op, ctx),
            Filter::Or(l, r) => l.matches(op, ctx) || r.matches(op, ctx),
            Filter::Not(f) => !f.matches(op, ctx),
            // ... other variants
        }
    }
}

impl BitAnd for Filter {
    type Output = Filter;
    fn bitand(self, rhs: Filter) -> Filter {
        Filter::And(Box::new(self), Box::new(rhs))
    }
}

impl BitOr for Filter {
    type Output = Filter;
    fn bitor(self, rhs: Filter) -> Filter {
        Filter::Or(Box::new(self), Box::new(rhs))
    }
}

impl Not for Filter {
    type Output = Filter;
    fn not(self) -> Filter {
        Filter::Not(Box::new(self))
    }
}

// Usage:
// let f = Filter::kind(EditKind::Replace) & Filter::min_length(10);
// let results: Vec<_> = result.filter(&f).collect();
```

### Example 3: Redline Orchestrator Core Flow

```rust
pub struct Redline {
    config: RedlineConfig,
    processor: TextProcessor,
    diff_computer: DiffComputer,
    metric_registry: MetricRegistry,
    analyzer_registry: PluginRegistry,
    cache: Option<CacheManager>,
}

impl Redline {
    pub fn new(config: RedlineConfig) -> Result<Self, RedlineError> {
        let processor = TextProcessor::new(
            config.normalizers(),
            config.tokenizer(),
            ExecutionMode::Minimal, // or based on config
        )?;
        let diff_computer = DiffComputer::with_algorithm(config.algorithm());

        let mut metric_registry = MetricRegistry::new();
        if config.metrics_enabled {
            register_builtins(&mut metric_registry)?;
            for metric in config.custom_metrics() {
                metric_registry.register(metric)?;
            }
        }

        let mut analyzer_registry = PluginRegistry::new();
        if config.analysis_enabled {
            register_builtin_analyzers(&mut analyzer_registry)?;
            for analyzer in config.custom_analyzers() {
                analyzer_registry.register(analyzer)?;
            }
        }

        let cache = if config.cache_enabled {
            Some(CacheManager::new(config.cache_capacity))
        } else {
            None
        };

        Ok(Self { config, processor, diff_computer, metric_registry, analyzer_registry, cache })
    }

    pub fn diff(&self, old: &str, new: &str) -> Result<RedlineResult, RedlineError> {
        // 1. Check cache
        let cache_key = self.compute_cache_key(old, new);
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.into_result());
            }
        }

        // 2. Process texts
        let source = Arc::new(self.processor.process(old)?);
        let target = Arc::new(self.processor.process(new)?);

        // 3. Compute diff
        let diff_result = self.diff_computer.compute(source, target, None)?;

        // 4. Compute metrics (if enabled)
        let metrics = if self.config.metrics_enabled {
            let engine = MetricsEngine::with_default_cache(/* registry ref or clone */);
            // compute desired metrics
            HashMap::new() // placeholder
        } else {
            HashMap::new()
        };

        // 5. Run analysis (if enabled)
        let analysis = if self.config.analysis_enabled {
            let engine = MetricsEngine::new(/* ... */);
            let ctx = AnalysisContext::new(&diff_result, &engine);
            let coordinator = AnalysisCoordinator::new(/* registry */);
            Some(coordinator.run(&ctx)?)
        } else {
            None
        };

        // 6. Build result
        let result = RedlineResult { diff: diff_result, metrics, analysis };

        // 7. Cache result
        if let Some(ref cache) = self.cache {
            cache.insert(cache_key, Arc::new(result.to_cached()));
        }

        Ok(result)
    }
}
```

**Design challenge: registry ownership.** `MetricRegistry` and `PluginRegistry` hold `Box<dyn Trait>` which are not `Clone`. The orchestrator cannot clone registries to create fresh engines per call. Solutions:
1. **Store registries by reference** in engines (requires lifetime parameter).
2. **Create engines once** and reuse (MetricsEngine has RefCell -- not Sync).
3. **Move registry into Arc** and share. MetricRegistry holds `HashMap<String, Box<dyn Metric>>` where `Metric: Send + Sync`. An `Arc<MetricRegistry>` would be `Send + Sync`. The MetricsEngine would need to be refactored to take `&MetricRegistry` instead of owning it.

**Recommended approach (#3):** Wrap registries in Arc. The `MetricsEngine` already takes a `MetricRegistry` by value -- refactor it to accept `Arc<MetricRegistry>` (or `&MetricRegistry`). Then each `diff()` call creates a fresh `MetricsEngine` with a reference to the shared registry. This is cheap (just allocating the cache, not re-registering metrics).

### Example 4: Preset Expansion

```rust
impl ConfigBuilder {
    pub fn build(self) -> Result<RedlineConfig, ConfigError> {
        // Phase 1: Expand preset defaults
        let defaults = match self.preset {
            Preset::Fast => PresetDefaults {
                normalizers: vec![],  // no normalization
                tokenizer: Box::new(WordTokenizer),
                algorithm: Box::new(Myers::new()),
                metrics_enabled: true,   // basic metrics only
                analysis_enabled: false, // no analysis
                cache_enabled: true,
                cache_capacity: 64,
            },
            Preset::Comprehensive => PresetDefaults {
                normalizers: vec![
                    Box::new(UnicodeNormalizer::default()),
                    Box::new(Lowercase),
                    Box::new(WhitespaceNormalizer),
                ],
                tokenizer: Box::new(WordTokenizer),
                algorithm: Box::new(Myers::new()),
                metrics_enabled: true,
                analysis_enabled: true,
                cache_enabled: true,
                cache_capacity: 256,
            },
        };

        // Phase 2: Overlay builder overrides onto defaults
        let config = RedlineConfig {
            normalizers: self.normalizers.unwrap_or(defaults.normalizers),
            tokenizer: self.tokenizer.unwrap_or(defaults.tokenizer),
            algorithm: self.algorithm.unwrap_or(defaults.algorithm),
            metrics_enabled: self.metrics_enabled.unwrap_or(defaults.metrics_enabled),
            analysis_enabled: self.analysis_enabled.unwrap_or(defaults.analysis_enabled),
            cache_enabled: self.cache_enabled.unwrap_or(defaults.cache_enabled),
            cache_capacity: self.cache_capacity.unwrap_or(defaults.cache_capacity),
            cache_memory_limit: self.cache_memory_limit,
            custom_metrics: self.custom_metrics,
            custom_analyzers: self.custom_analyzers,
        };

        // Phase 3: Validate
        if config.cache_enabled && config.cache_capacity == 0 {
            return Err(ConfigError::Invalid(
                "cache capacity must be > 0 when cache is enabled".into()
            ));
        }

        Ok(config)
    }
}
```

## Existing Integration Points

### Types the Orchestrator Wires Together

| Component | Constructor | Input | Output | Thread Safety |
|-----------|-------------|-------|--------|---------------|
| `TextProcessor` | `::new(normalizers, tokenizer, mode)` | `&str` | `ProcessedText` | `Send + Sync` (all fields owned) |
| `DiffComputer` | `::new()` or `::with_algorithm(algo)` | `Arc<ProcessedText>` x2 | `DiffResult` | `Send + Sync` |
| `MetricsEngine` | `::with_cache(registry, cap)` | `&MetricInput` | `MetricValue` | `!Sync` (RefCell) |
| `AnalysisCoordinator` | `::new(registry)` | `&AnalysisContext` | `AnalysisReport` | `Send + Sync` |
| `AnalysisContext` | `::new(&DiffResult, &MetricsEngine)` | borrows | borrows | lifetime-bound |

### Existing Registration Functions

| Function | Registers | Count |
|----------|-----------|-------|
| `register_builtins(&mut MetricRegistry)` | All 37 built-in metrics | 12 counts + 10 readability + 15 similarity |
| `register_builtin_analyzers(&mut PluginRegistry)` | All 4 built-in analyzers | semantic, stylistic, readability, edit_classifier |

### Existing Content Hashing

`ContentHash` in `metrics/cache.rs` already implements deterministic content hashing with `foldhash::fast::FixedState` seed 0. The pipeline cache should use the same approach, extended to include config fingerprint.

### Error Landscape

The existing `RedlineError` enum already has variants for all subsystem errors: `Store`, `Tokenize`, `Normalize`, `Config`, `Process`, `Diff`, `Metric`. It needs a new variant for analysis errors (`Analysis(AnalysisError)`) and possibly orchestration-specific errors (`Orchestrate` or `Pipeline`).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 4 presets (Fast, Syntactic, Semantic, Comprehensive) | 2 presets (Fast, Comprehensive) | User decision 2026-02-09 | Simpler API surface, users compose via overrides |
| `DiffOrchestrator` entry point | `Redline` entry point | User decision 2026-02-09 | Brandable, cleaner API |
| Struct-based query types | Predicate combinators with operator overloads | User decision 2026-02-09 | More ergonomic, Rust-idiomatic |

## Open Questions

1. **MetricsEngine RefCell refactoring scope**
   - What we know: MetricsEngine uses `RefCell<MetricCache>` making it `!Sync`. The orchestrator needs `Sync`.
   - What's unclear: Should we refactor MetricsEngine to use `parking_lot::RwLock` in Phase 6, or create engines per-call?
   - Recommendation: Create fresh engines per `diff()` call. The MetricCache inside is per-computation anyway. Refactoring MetricsEngine internals is out of scope for this phase -- it works correctly for single-threaded use within a `diff()` call.

2. **Registry sharing strategy**
   - What we know: `MetricRegistry` and `PluginRegistry` own `Box<dyn Trait>` and are not Clone.
   - What's unclear: Best way to share registries across `diff()` calls without cloning.
   - Recommendation: Store registries in the `Redline` struct directly. Create `MetricsEngine` and `AnalysisCoordinator` per `diff()` call, passing `&MetricRegistry` / `&PluginRegistry` by reference. This may require adjusting engine constructors to take references instead of owned values. Alternatively, wrap in `Arc` and have engines take `Arc` references.

3. **Fast preset metric selection**
   - What we know: Fast preset = "diff + basic metrics (counts, similarity), no analysis"
   - What's unclear: Exactly which of the 37 metrics are "basic"?
   - Recommendation: Fast uses: all 12 count metrics + `similarity_ratio` (1 from pairwise). Skip readability (10) and most pairwise (14). Total: 13 basic metrics. This can be a curated list in the preset definition.

4. **Memory cap measurement granularity**
   - What we know: User wants optional memory cap as "safety valve"
   - What's unclear: How accurately to measure memory per cached entry
   - Recommendation: Use rough estimates -- sum of original text lengths + token count * 40 bytes + fixed 1KB overhead per entry. Precision is not critical for a safety valve; the entry-count LRU is the primary mechanism.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: all source files under `crates/core/src/` (TextProcessor, DiffComputer, MetricsEngine, AnalysisCoordinator, and all supporting types)
- `/websites/docs_rs-parking_lot-latest-parking_lot` (Context7) -- RwLock usage patterns
- parking_lot docs.rs -- version 0.12.4/0.12.5, RwLock API

### Secondary (MEDIUM confidence)
- `predicates` crate (docs.rs/predicates) -- predicate combinator pattern with boolean extension trait
- Rust std::ops documentation -- BitAnd, BitOr, Not trait signatures
- Rust builder pattern official style guide (doc.rust-lang.org/1.0.0/style/ownership/builders.html)
- lru-rs GitHub (jeromefroe/lru-rs) -- LruCache API (already in use in codebase)

### Tertiary (LOW confidence)
- WebSearch findings on Rust caching patterns -- general approach confirmed by codebase analysis

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in Cargo.toml except parking_lot; versions verified
- Architecture: HIGH -- patterns derived from existing codebase patterns and Rust idioms
- Integration points: HIGH -- all types and APIs read directly from source code
- Pitfalls: HIGH -- identified from actual type analysis (RefCell, dyn Any, Arc patterns)
- Query API pattern: MEDIUM -- enum vs generic tradeoff is a design judgment; enum recommended based on simplicity
- Memory cap approach: LOW -- rough estimation approach; may need refinement during implementation

**Research date:** 2026-02-09
**Valid until:** 2026-03-09 (stable domain, no external dependencies changing)
