# Phase 4: Metrics Engine - Research

**Researched:** 2026-02-07
**Domain:** Text metrics computation, lazy evaluation, dependency graphs, LRU caching
**Confidence:** HIGH

## Summary

Phase 4 builds a metrics engine on top of Phases 1-3 (Token, ProcessedText, DiffResult). The engine provides a `Metric` trait, a `MetricRegistry` for registration, a `MetricsEngine` with pull-based lazy evaluation, and content-addressed LRU caching. 35+ metrics are implemented: 20+ single-text metrics (counts, readability, complexity) and 15+ pairwise metrics (similarity, distance, set-based).

The architecture follows the established codebase pattern: trait objects for extensibility (`Box<dyn Metric>`), `Send + Sync` bounds for thread safety, and modular organization by category. The `lru` crate (v0.16) provides the LRU cache, aligning with the project's existing hashbrown 0.16 dependency. All metric algorithms are implemented directly -- no external metric libraries needed since the formulas operate on already-processed tokens and text.

The key design challenge is the dependency graph with both static and dynamic dependencies. Static dependencies are validated at registration time via DFS cycle detection. Dynamic dependencies are validated at compute time using a call-stack coloring approach (gray/black marking) to detect cycles during evaluation.

**Primary recommendation:** Implement metrics in TDD waves (counts -> readability -> similarity), with the engine and cache growing incrementally alongside each wave. Use `u64` content hashes via FoldHash for cache keys -- fast, zero additional dependencies, sufficient collision resistance for caching.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Pull-based lazy evaluation: `get("metric_id")` for single metric, `get_many(["a", "b", "c"])` for batch
- When a dependency fails, return `MetricValue::Unavailable` (sentinel) and log the failure
- `MetricValue` is a tagged enum: `Integer(i64)`, `Float(f64)`, `Unavailable`
- Both static AND dynamic dependencies: `static_dependencies()` and `dynamic_dependencies(input)`
- Engine validates static graph upfront, handles dynamic additions during compute with runtime cycle detection
- One type per metric: `WordCountMetric`, `FleschKincaidMetric`, `JaccardSimilarityMetric`, etc.
- Organized into modules by category (`counts/`, `readability/`, `similarity/`)
- TDD waves ordered foundation-up: counts -> readability -> pairwise
- Cache key: hash(ProcessedText) + metric_id
- `MetricsEngine::new()` -- no cache; `MetricsEngine::with_cache(capacity)` -- LRU cache
- LRU eviction by capacity only
- Custom metrics implement same `Metric` trait, get cached, can declare dependencies on built-ins
- Override via `registry.register_with_override(metric)` -- default `register(metric)` errors on ID collision
- No namespacing -- override is the collision mechanism
- Registration validates static dependency IDs exist in registry
- TDD is a hard requirement -- tests before implementation for every metric
- Engine integration tests added incrementally with each wave
- Property tests consistent with Phase 2/3 (proptest)

### Claude's Discretion
- LRU cache TTL policy (whether to add time-based expiry on top of LRU capacity eviction)
- Content hashing algorithm choice
- Exact module file structure within category directories
- Error logging mechanism for Unavailable metrics
- Default cache capacity value if a convenience constructor is added

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lru | 0.16 | LRU cache for metric results | O(1) get/put/pop, uses hashbrown 0.16 (matches project), supports custom hashers, 5M+ downloads |
| hashbrown | 0.16 (existing) | HashMap with FoldHash default | Already a dependency; FoldHash provides fast non-cryptographic hashing for cache keys |
| thiserror | 2.0 (existing) | Error types for MetricError | Already used for all error types in the project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tracing | 0.1 (existing optional dep) | Logging Unavailable metric failures | Behind `tracing` feature flag for debug diagnostics |
| proptest | 1.6 (existing dev-dep) | Property-based testing for metrics | Metric invariant testing (non-negative counts, bounded similarities) |

### NOT Needed (Hand-Roll Instead)
| Library | Why Not |
|---------|---------|
| textdistance | Metrics must implement `Metric` trait with dependency support; standalone functions don't fit the architecture |
| strsim | Same reason -- we need trait-based metrics, not free functions; algorithms are simple to implement directly |
| rust_readability | We need per-formula metrics with dependencies, not a monolithic readability struct |
| petgraph / topo_sort | Dependency graph is small and simple; DFS cycle detection is ~30 lines of code |
| syllarust / syllable | Syllable counting is a single heuristic function; no need for a dependency |

**Installation (only new dependency):**
```bash
cargo add lru@0.16
```

## Discretion Recommendations

### LRU Cache TTL Policy
**Recommendation: No TTL.** LRU eviction by capacity is sufficient for this use case. Metric results are deterministic -- the same input always produces the same output, so cached values never become stale. Adding TTL introduces unnecessary complexity (timer infrastructure, `Instant` dependency which complicates no_std). The TECHNICAL_SPEC's CacheManager with TTL is for Phase 6 (orchestration layer), not Phase 4.

### Content Hashing Algorithm
**Recommendation: Use `core::hash::Hash` trait + FoldHash hasher (via hashbrown's DefaultHashBuilder).** The content hash is a `u64` produced by hashing the relevant fields of `ProcessedText` (original text + normalized text). This is fast (~2ns for short strings), zero additional dependencies, and has sufficient collision resistance for a capacity-bounded cache. The TECHNICAL_SPEC's `ContentHash([u8; 32])` is overengineered for an in-process cache -- 32-byte cryptographic hashes are for distributed/persistent caches.

**Implementation:**
```rust
use core::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(u64);

impl ContentHash {
    pub fn of(processed: &ProcessedText) -> Self {
        let mut hasher = hashbrown::DefaultHashBuilder::default().build_hasher();
        processed.original.hash(&mut hasher);
        processed.normalized.hash(&mut hasher);
        ContentHash(hasher.finish())
    }
}
```

### Module File Structure
**Recommendation:**
```
src/metrics/
├── mod.rs              # Metric trait, MetricValue, MetricInput, re-exports
├── error.rs            # MetricError enum
├── registry.rs         # MetricRegistry
├── engine.rs           # MetricsEngine (lazy eval + cache)
├── cache.rs            # ContentHash, CacheKey, cache wrapper around lru::LruCache
├── counts/
│   ├── mod.rs          # re-exports
│   ├── word_count.rs
│   ├── char_count.rs
│   ├── sentence_count.rs
│   ├── paragraph_count.rs
│   ├── line_count.rs
│   ├── letter_count.rs
│   ├── digit_count.rs
│   ├── whitespace_count.rs
│   ├── punctuation_count.rs
│   ├── syllable_count.rs
│   └── unique_word_count.rs
├── readability/
│   ├── mod.rs          # re-exports + syllable helper
│   ├── flesch_reading_ease.rs
│   ├── flesch_kincaid.rs
│   ├── gunning_fog.rs
│   ├── smog.rs
│   ├── coleman_liau.rs
│   ├── ari.rs
│   ├── avg_word_length.rs
│   ├── avg_sentence_length.rs
│   ├── vocabulary_richness.rs
│   └── lexical_density.rs
├── similarity/
│   ├── mod.rs          # re-exports
│   ├── jaccard.rs
│   ├── cosine.rs
│   ├── dice.rs
│   ├── overlap.rs
│   ├── levenshtein.rs
│   ├── damerau_levenshtein.rs
│   ├── hamming.rs
│   ├── jaro.rs
│   ├── jaro_winkler.rs
│   ├── length_ratio.rs
│   ├── word_count_diff.rs
│   ├── char_count_diff.rs
│   ├── readability_delta.rs
│   └── grade_level_delta.rs
└── builtins.rs         # register_builtins() function
```

### Error Logging Mechanism
**Recommendation: Use `tracing::warn!` behind the `tracing` feature flag.** When a metric dependency fails and returns `Unavailable`, the engine should log via `tracing::warn!("metric '{}' unavailable: dependency '{}' failed", metric_id, dep_id)`. When the `tracing` feature is not enabled, the failure is silently swallowed (the `Unavailable` sentinel propagates). This aligns with the existing optional `tracing` dep in Cargo.toml.

```rust
#[cfg(feature = "tracing")]
tracing::warn!(
    metric_id = %id,
    dependency = %dep_id,
    "metric dependency unavailable, returning Unavailable"
);
```

### Default Cache Capacity
**Recommendation: 256 entries** for a convenience constructor like `MetricsEngine::with_default_cache()`. This is sufficient for typical use cases (20-30 metrics x ~8 input texts = 160-240 entries). The user can always specify a custom capacity via `with_cache(capacity)`.

## Architecture Patterns

### Recommended Project Structure
```
crates/core/src/
├── metrics/
│   ├── mod.rs          # Metric trait, MetricValue, MetricInput, DependencyKind
│   ├── error.rs        # MetricError
│   ├── registry.rs     # MetricRegistry
│   ├── engine.rs       # MetricsEngine
│   ├── cache.rs        # ContentHash, CacheKey, MetricCache
│   ├── counts/         # 11 count metrics (zero dependencies)
│   ├── readability/    # 10 readability metrics (depend on counts)
│   ├── similarity/     # 14 pairwise metrics
│   └── builtins.rs     # register_builtins()
└── lib.rs              # add `pub mod metrics;`
```

### Pattern 1: Metric Trait (Object-Safe)
**What:** A trait that all metrics implement, following the Normalizer/DiffAlgorithm pattern.
**When to use:** Every metric struct.
**Critical detail:** The trait MUST be object-safe (`Box<dyn Metric>`) to support the registry. This means no associated types -- use `MetricValue` enum instead of `type Value`.

```rust
/// Input provided to metric compute functions.
pub enum MetricInput<'a> {
    /// Single text analysis
    Single(&'a ProcessedText),
    /// Pairwise comparison
    Pairwise(&'a ProcessedText, &'a ProcessedText),
}

/// The kind of dependencies a metric declares.
pub enum DependencyKind {
    /// All dependencies known at compile time
    Static,
    /// Some dependencies discovered at compute time
    Dynamic,
}

/// Trait for all metrics (single-text and pairwise).
pub trait Metric: Send + Sync {
    /// Unique string identifier (e.g., "word_count", "flesch_kincaid")
    fn id(&self) -> &str;

    /// Compute the metric value given input and resolved dependencies.
    fn compute(
        &self,
        input: &MetricInput<'_>,
        dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError>;

    /// Static dependencies -- known at compile time, validated at registration.
    fn static_dependencies(&self) -> &[&str] {
        &[]
    }

    /// Dynamic dependencies -- discovered at compute time.
    fn dynamic_dependencies(&self, _input: &MetricInput<'_>) -> Vec<String> {
        Vec::new()
    }

    /// Whether this metric uses dynamic dependencies.
    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }

    /// Estimated computational cost (0.0 = trivial, 1.0 = expensive).
    fn cost(&self) -> f32 {
        0.5
    }
}
```

### Pattern 2: MetricValue Enum
**What:** Tagged enum preserving type semantics per user decision.
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum MetricValue {
    Integer(i64),
    Float(f64),
    Unavailable,
}

impl MetricValue {
    pub fn as_integer(&self) -> Option<i64> { ... }
    pub fn as_float(&self) -> Option<f64> { ... }
    pub fn is_unavailable(&self) -> bool { ... }
}
```

### Pattern 3: MetricRegistry with Override
**What:** Stores metrics by ID, validates static dependency graph at registration.
```rust
pub struct MetricRegistry {
    metrics: HashMap<String, Box<dyn Metric>>,
}

impl MetricRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, metric: Box<dyn Metric>) -> Result<(), MetricError>;
    pub fn register_with_override(&mut self, metric: Box<dyn Metric>);
    pub fn validate(&self) -> Result<(), MetricError>;
    pub fn get(&self, id: &str) -> Option<&dyn Metric>;
    pub fn ids(&self) -> Vec<&str>;
}
```

### Pattern 4: Pull-Based Lazy Evaluation in MetricsEngine
**What:** Engine resolves dependencies and computes on demand. Cache is opt-in.

**Evaluation loop (inside `get`):**
1. Check cache -- if hit, return cached value
2. Check cycle detection stack -- if in-progress, return `Unavailable`
3. Mark metric as in-progress
4. Resolve static dependencies recursively (call `get` for each)
5. Resolve dynamic dependencies (call `get` for each)
6. Call `metric.compute(input, &resolved_deps)`
7. If compute returns `Err`, convert to `MetricValue::Unavailable` and log
8. Cache the result, mark as complete
9. Return

### Pattern 5: Content-Addressed Cache
**What:** Cache keyed by (ContentHash, metric_id) tuple.
```rust
pub struct CacheKey {
    content_hash: ContentHash,
    metric_id: String,
}

pub struct MetricCache {
    inner: lru::LruCache<CacheKey, MetricValue>,
}
```

### Anti-Patterns to Avoid
- **Monolithic TextMetrics/PairwiseMetrics structs:** Each metric is an independent type implementing the trait. Do NOT create aggregate structs that eagerly compute all metrics.
- **Associated types on the Metric trait:** Using `type Value = f64` breaks object safety. Use the `MetricValue` enum instead.
- **Eager computation:** Never compute metrics that weren't requested.
- **String-based cache keys without content hash:** Always hash the ProcessedText content, not use the text store pointer.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LRU cache | Custom doubly-linked list + HashMap | `lru` crate v0.16 | O(1) operations, handles eviction, resize, hashbrown integration |
| Error types | Manual Display/From impls | `thiserror` 2.0 (existing) | Already the project pattern |
| Property tests | Manual edge-case enumeration | `proptest` 1.6 (existing) | Shrinking, regression files, strategy composition |

**Things that SHOULD be hand-rolled (too simple for a dependency):**
| Problem | Why Hand-Roll |
|---------|---------------|
| Syllable counting | Single heuristic function (~50 lines); no crate is reliable enough |
| Readability formulas | Each is 3-5 lines of arithmetic; trivial to implement and test |
| Levenshtein/edit distance | Classic DP algorithm (~30 lines); we need token-level (not character-level) |
| Jaccard/Dice/Cosine similarity | Set operations on token HashSets (~10-20 lines each) |
| DFS cycle detection | ~30 lines; tightly coupled with registry internals |

## Common Pitfalls

### Pitfall 1: Infinite Recursion in Dependency Resolution
**What goes wrong:** Metric A depends on B, B depends on C, C depends on A. Without cycle detection, `get("A")` recurses forever.
**Why it happens:** Static validation catches cycles at registration, but dynamic dependencies can create cycles at runtime.
**How to avoid:** Maintain a `HashSet<String>` of "currently computing" metric IDs on the call stack. Before computing any metric, check if it's already in progress. If so, return `Unavailable`.

### Pitfall 2: Mutable Borrow Conflicts in Engine
**What goes wrong:** `get()` needs `&mut self` for cache writes, but dependency resolution also calls `get()` recursively.
**Why it happens:** Rust's borrow checker doesn't allow recursive `&mut self` calls.
**How to avoid:** Use interior mutability for the cache (`RefCell` for single-threaded). The engine holds `registry: MetricRegistry` (immutable) and `cache: Option<RefCell<MetricCache>>` (interior mutability).

### Pitfall 3: Floating-Point Readability Formulas
**What goes wrong:** Readability scores differ slightly across platforms (x86 vs ARM vs WASM) due to floating-point precision.
**How to avoid:** Use `f64` throughout. In tests, compare with epsilon tolerance (e.g., `(actual - expected).abs() < 0.01`).

### Pitfall 4: Token-Level vs Character-Level Metrics
**What goes wrong:** Implementing Levenshtein at character level when the system has token-level data.
**How to avoid:** Pairwise metrics operate on tokens (from `ProcessedText.tokens`), NOT on raw character strings.

### Pitfall 5: Syllable Count Accuracy
**What goes wrong:** Heuristic syllable counting is wrong for ~5-10% of English words.
**How to avoid:** Accept ~90-95% accuracy. Test against known word lists. Document the limitation.

### Pitfall 6: Cache Key Collision
**What goes wrong:** Two different `ProcessedText` instances hash to the same `u64`.
**How to avoid:** Include both `original` and `normalized` text in the hash. For pairwise, hash both texts. Collision probability is ~1/2^64 per pair, sufficient for capacity-bounded cache.

## Metric Catalog

### Single-Text Metrics (20+ total, METR-03)

**Wave 1: Counts (zero dependencies)**
| Metric ID | Type | Formula/Description |
|-----------|------|---------------------|
| `word_count` | Integer | `tokens.len()` |
| `char_count` | Integer | `original.chars().count()` |
| `sentence_count` | Integer | Count sentence-ending punctuation (`.`, `!`, `?`) |
| `paragraph_count` | Integer | Count `\n\n` sequences + 1 |
| `line_count` | Integer | Count `\n` + 1 |
| `letter_count` | Integer | Count `char::is_alphabetic()` |
| `digit_count` | Integer | Count `char::is_ascii_digit()` |
| `whitespace_count` | Integer | Count `char::is_whitespace()` |
| `punctuation_count` | Integer | Count `char::is_ascii_punctuation()` |
| `syllable_count` | Integer | Sum of `count_syllables(word)` for each token |
| `unique_word_count` | Integer | `HashSet` of `text_id` values, `.len()` |

**Wave 2: Readability/Complexity (depend on counts)**
| Metric ID | Type | Formula | Dependencies |
|-----------|------|---------|--------------|
| `avg_word_length` | Float | `char_count / word_count` | `char_count`, `word_count` |
| `avg_sentence_length` | Float | `word_count / sentence_count` | `word_count`, `sentence_count` |
| `vocabulary_richness` | Float | `unique_word_count / word_count` (TTR) | `unique_word_count`, `word_count` |
| `lexical_density` | Float | content_words / word_count | `word_count` |
| `flesch_reading_ease` | Float | `206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)` | `word_count`, `sentence_count`, `syllable_count` |
| `flesch_kincaid_grade` | Float | `0.39*(words/sentences) + 11.8*(syllables/words) - 15.59` | `word_count`, `sentence_count`, `syllable_count` |
| `gunning_fog` | Float | `0.4*(words/sentences + 100*(complex_words/words))` | `word_count`, `sentence_count`, `syllable_count` |
| `smog_index` | Float | `1.0430*sqrt(polysyllables * 30/sentences) + 3.1291` | `sentence_count`, `syllable_count` |
| `coleman_liau` | Float | `0.0588*L - 0.296*S - 15.8` | `letter_count`, `word_count`, `sentence_count` |
| `ari` | Float | `4.71*(chars/words) + 0.5*(words/sentences) - 21.43` | `char_count`, `word_count`, `sentence_count` |

### Pairwise Metrics (15+ total, METR-04)

**Wave 3: Similarity/Distance**
| Metric ID | Type | Formula/Description |
|-----------|------|---------------------|
| `jaccard_similarity` | Float | `\|A ∩ B\| / \|A ∪ B\|` on token sets |
| `cosine_similarity` | Float | `(A · B) / (\|A\| * \|B\|)` on token frequency vectors |
| `dice_coefficient` | Float | `2*\|A ∩ B\| / (\|A\| + \|B\|)` on token sets |
| `overlap_coefficient` | Float | `\|A ∩ B\| / min(\|A\|, \|B\|)` on token sets |
| `levenshtein_distance` | Integer | Token-level DP edit distance |
| `damerau_levenshtein` | Integer | Token-level DP with transpositions |
| `hamming_distance` | Integer | Count of positions where tokens differ (equal-length only) |
| `jaro_similarity` | Float | Jaro algorithm on token sequences |
| `jaro_winkler_similarity` | Float | Jaro-Winkler with prefix bonus |
| `length_ratio` | Float | `target.tokens.len() / source.tokens.len()` |
| `word_count_diff` | Integer | `target.tokens.len() - source.tokens.len()` |
| `char_count_diff` | Integer | `target.original.len() - source.original.len()` |
| `readability_delta` | Float | delta of flesch_reading_ease between source and target |
| `grade_level_delta` | Float | delta of flesch_kincaid_grade between source and target |
| `similarity_ratio` | Float | From DiffResult statistics or compute from LCS |

## Code Examples

### Example 1: Simple Count Metric (WordCountMetric)
```rust
pub struct WordCountMetric;

impl Metric for WordCountMetric {
    fn id(&self) -> &str { "word_count" }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                Ok(MetricValue::Integer(text.tokens.len() as i64))
            }
            MetricInput::Pairwise(_, _) => {
                Err(MetricError::InvalidInput("word_count requires single text input".into()))
            }
        }
    }

    fn cost(&self) -> f32 { 0.01 }
}
```

### Example 2: Readability Metric with Dependencies
```rust
pub struct FleschKincaidMetric;

impl Metric for FleschKincaidMetric {
    fn id(&self) -> &str { "flesch_kincaid_grade" }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        let words = deps.get("word_count").and_then(|v| v.as_integer())
            .ok_or(MetricError::DependencyUnavailable("word_count".into()))?;
        let sentences = deps.get("sentence_count").and_then(|v| v.as_integer())
            .ok_or(MetricError::DependencyUnavailable("sentence_count".into()))?;
        let syllables = deps.get("syllable_count").and_then(|v| v.as_integer())
            .ok_or(MetricError::DependencyUnavailable("syllable_count".into()))?;

        if words == 0 || sentences == 0 {
            return Ok(MetricValue::Float(0.0));
        }

        let grade = 0.39 * (words as f64 / sentences as f64)
                  + 11.8 * (syllables as f64 / words as f64)
                  - 15.59;

        Ok(MetricValue::Float(grade))
    }

    fn static_dependencies(&self) -> &[&str] {
        &["word_count", "sentence_count", "syllable_count"]
    }

    fn cost(&self) -> f32 { 0.1 }
}
```

### Example 3: Content Hash Implementation
```rust
use core::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(u64);

impl ContentHash {
    pub fn of_single(text: &ProcessedText) -> Self {
        let state = hashbrown::DefaultHashBuilder::default();
        let mut hasher = state.build_hasher();
        text.original.hash(&mut hasher);
        text.normalized.hash(&mut hasher);
        text.tokens.len().hash(&mut hasher);
        ContentHash(hasher.finish())
    }

    pub fn of_pair(source: &ProcessedText, target: &ProcessedText) -> Self {
        let state = hashbrown::DefaultHashBuilder::default();
        let mut hasher = state.build_hasher();
        source.original.hash(&mut hasher);
        source.normalized.hash(&mut hasher);
        target.original.hash(&mut hasher);
        target.normalized.hash(&mut hasher);
        ContentHash(hasher.finish())
    }
}
```

### Example 4: LRU Cache Wrapper
```rust
use std::num::NonZeroUsize;
use lru::LruCache;

pub struct MetricCache {
    inner: LruCache<CacheKey, MetricValue>,
}

impl MetricCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: LruCache::new(
                NonZeroUsize::new(capacity).expect("cache capacity must be > 0")
            ),
        }
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<&MetricValue> {
        self.inner.get(key)
    }

    pub fn put(&mut self, key: CacheKey, value: MetricValue) {
        self.inner.put(key, value);
    }
}
```

## Open Questions

1. **Pairwise metrics that need single-text metrics on both inputs** -- `readability_delta` needs `flesch_reading_ease` on both source and target. Recommendation: engine handles transparently, computing single-text deps on each text independently. Cache keys include the content hash of the specific text.

2. **`lexical_density` implementation** -- requires content vs function word distinction. Recommendation: Use a curated ~150-word English function word list rather than POS tagging.

3. **`sentence_count` accuracy** -- whether to reuse `SentenceTokenizer` from Phase 2 or implement a simpler counter. Recommendation: simple punctuation-based counter in the metric itself to avoid coupling.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `crates/core/src/` -- all trait patterns, module organization, existing types
- `docs/design/TECHNICAL_SPEC.md` -- original design (adapted per user decisions)
- `.planning/REQUIREMENTS.md` -- METR-01 through METR-06
- lru crate docs (docs.rs/lru) -- API, version 0.16, hashbrown 0.16 integration

### Secondary (MEDIUM confidence)
- textdistance.rs (github.com/life4/textdistance.rs) -- reference implementations for similarity algorithms
- Readability formula references verified from multiple sources (readable.com, readabilityformulas.com, Wikipedia)

### Tertiary (LOW confidence)
- Syllable counting heuristic accuracy (~90-95%) -- based on multiple sources, no definitive benchmark

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- lru 0.16 verified, hashbrown already in project
- Architecture: HIGH -- follows established codebase patterns, all design decisions locked by user
- Metric algorithms: HIGH -- formulas verified from multiple authoritative sources
- Pitfalls: HIGH -- derived from codebase analysis and domain knowledge
- Syllable counting: MEDIUM -- heuristic approach is standard but inherently limited

**Research date:** 2026-02-07
**Valid until:** 2026-03-07 (stable domain, readability formulas don't change)
