# Phase 5: Analysis Framework - Research

**Researched:** 2026-02-07
**Domain:** Plugin-based analysis system with dependency resolution, edit classification, NLP heuristics
**Confidence:** HIGH

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Per-operation annotations + optional region-level summaries (both granularities)
- Typed built-ins + dynamic plugins for result storage: `report.semantic()` -> `Option<&SemanticResult>`, plugin analyzers return common `AnalysisResult` struct
- Annotations carry `confidence: f64` (0.0-1.0) for consumer-side filtering
- Coarse taxonomy (7 categories): Correction, Expansion, Deletion, Rewrite, Refinement, Formatting, Other
- Each edit gets ranked list with confidence: `Vec<(IntentCategory, f64)>` sorted by confidence descending
- Always return all ranked categories -- consumer decides what to filter
- Classify at both levels: per-operation + optional group-level for adjacent related operations
- Explicit error entry + fallback: failed analyzer returns partial results + `AnalyzerStatus::Failed(reason)`
- Dependents of failed analyzers are skipped: `AnalyzerStatus::Skipped(reason)`
- Catch panics via `std::panic::catch_unwind` around each analyzer
- Full execution metadata per analyzer: wall-clock duration and execution order index
- SemanticAnalyzer: heuristic-based v1 (token overlap, shared n-grams, Jaccard on word sets), BERT-ready interface
- StylisticAnalyzer: voice shifts, tone shifts, structural style changes
- ReadabilityAnalyzer: document-level readability delta + per-region attribution
- EditClassifier: 7-category taxonomy with ranked confidence, configurable heuristic thresholds

### Claude's Discretion
- Exact heuristic algorithms for each built-in analyzer
- AnalyzerPlugin trait design details (metadata fields, cost model)
- ExecutionPlanner DAG implementation approach
- AnalysisContext internal structure
- Grouping algorithm for adjacent related operations

### Deferred Ideas (OUT OF SCOPE)
- BERT/ML-based semantic similarity scoring -- Phase 8 or future ONNX
- Parallel analyzer execution -- Phase 7

## Summary

Phase 5 builds the analysis framework: a pluggable analyzer system where analyzers declare dependencies, register with a `PluginRegistry` that validates the dependency graph incrementally, and execute in topological order via an `ExecutionPlanner`. The system produces an `AnalysisReport` from a `DiffResult` and metrics. Four built-in analyzers (Semantic, Stylistic, Readability, EditClassifier) ship with this phase.

The architecture closely mirrors the existing `MetricRegistry` + `MetricsEngine` pattern from Phase 4, but operates at a higher level: where metrics compute scalar values from text, analyzers produce structured annotations per edit operation. The key architectural additions are: (1) DAG-based topological sort for execution ordering, (2) `catch_unwind` for panic isolation per analyzer, (3) a typed + dynamic result storage system, and (4) four substantial NLP heuristic analyzers.

No new crate dependencies are needed. The project already contains all necessary infrastructure: `HashMap` (hashbrown), `thiserror` for errors, `std::time::Instant` for timing, `std::panic::catch_unwind` for panic isolation, and `std::any::Any`/`TypeId` for typed downcasting. The existing `MetricRegistry` DFS cycle detection pattern provides a proven template. Topological sort should use Kahn's algorithm (BFS-based) since it naturally produces execution order and detects cycles in one pass.

**Primary recommendation:** Build the analyzer framework as a new `analysis/` module in `crates/core/src/`, following the same module organization pattern as `metrics/`. Use Kahn's algorithm for topological sort. Store built-in results via `HashMap<String, Box<dyn Any + Send + Sync>>` with typed convenience accessors on `AnalysisReport`. Each built-in analyzer should define its own result struct (e.g., `SemanticResult`, `StylisticResult`).

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| std::collections::HashMap | stdlib | Analyzer registry, result storage | Already used throughout codebase |
| std::any::{Any, TypeId} | stdlib | Typed result storage + downcasting for built-in analyzer results | Zero-cost abstraction for heterogeneous typed storage |
| std::panic::{catch_unwind, AssertUnwindSafe} | stdlib | Panic isolation per analyzer | Required by CONTEXT.md decision; stdlib native |
| std::time::Instant | stdlib | Wall-clock duration per analyzer | ~80ns overhead per call; already used in project |
| thiserror | 2.0 | Error types for analysis module | Existing project pattern |
| hashbrown | 0.16 | Internal HashMap for registry if needed | Existing project dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| serde | 1.x (optional) | Serialization of result structs | Behind `serde` feature flag, for AnalysisResult and built-in result structs |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled Kahn's toposort | petgraph crate | petgraph is 50KB+ dependency for ~40 lines of Kahn's algorithm; analyzer DAG is tiny (<20 nodes). Phase 4 research already rejected petgraph for same reason. |
| HashMap<TypeId, Box<dyn Any>> | typedmap crate | External dependency for a pattern that is ~15 lines of code with stdlib |
| Hand-rolled cycle detection | incremental-topo crate | Analyzer count is small; incremental is overkill. Batch validation on register() is sufficient. |

**Installation:** No new dependencies required.

## Architecture Patterns

### Recommended Project Structure
```
crates/core/src/
├── analysis/
│   ├── mod.rs                   # Public re-exports, AnalyzerPlugin trait, AnalysisResult
│   ├── error.rs                 # AnalysisError enum
│   ├── context.rs               # AnalysisContext (DiffResult + metrics access)
│   ├── registry.rs              # PluginRegistry with incremental cycle detection
│   ├── planner.rs               # ExecutionPlanner with Kahn's toposort
│   ├── coordinator.rs           # AnalysisCoordinator (synchronous execution with catch_unwind)
│   ├── report.rs                # AnalysisReport with typed + dynamic result access
│   ├── annotation.rs            # Annotation, OperationAnnotation, RegionSummary structs
│   └── builtins/
│       ├── mod.rs               # register_builtin_analyzers()
│       ├── semantic.rs          # SemanticAnalyzer + SemanticResult + ScoringBackend trait
│       ├── stylistic.rs         # StylisticAnalyzer + StylisticResult
│       ├── readability.rs       # ReadabilityAnalyzer + ReadabilityResult
│       └── edit_classifier.rs   # EditClassifier + EditClassifierResult + IntentCategory
```

### Pattern 1: AnalyzerPlugin Trait Design
The core trait that all analyzers implement:

```rust
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalyzerMeta {
    pub id: String,
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnalyzerDependency {
    Required(String),
    Optional(String),
}

pub trait AnalyzerPlugin: Send + Sync {
    fn id(&self) -> &str;
    fn meta(&self) -> AnalyzerMeta;
    fn dependencies(&self) -> Vec<AnalyzerDependency>;
    fn cost(&self) -> f32 { 0.5 }
    fn analyze(
        &self,
        context: &AnalysisContext,
        prior_results: &AnalysisReport,
    ) -> Result<Box<dyn Any + Send + Sync>, AnalysisError>;
}
```

**Key design choice:** The `analyze` method receives both the `AnalysisContext` (DiffResult + metrics) and the `AnalysisReport` built so far (for accessing prior analyzer results). This enables analyzers to depend on other analyzers' outputs.

### Pattern 2: Typed + Dynamic Result Storage
`AnalysisReport` stores results in a `HashMap<String, Box<dyn Any + Send + Sync>>` keyed by analyzer ID. Built-in results have typed accessor methods that downcast internally.

```rust
pub struct AnalysisReport {
    results: HashMap<String, Box<dyn Any + Send + Sync>>,
    pub execution_metadata: Vec<AnalyzerExecutionMeta>,
}

impl AnalysisReport {
    pub fn semantic(&self) -> Option<&SemanticResult> {
        self.results.get("semantic")?.downcast_ref::<SemanticResult>()
    }
    pub fn stylistic(&self) -> Option<&StylisticResult> {
        self.results.get("stylistic")?.downcast_ref::<StylisticResult>()
    }
    pub fn readability(&self) -> Option<&ReadabilityResult> {
        self.results.get("readability")?.downcast_ref::<ReadabilityResult>()
    }
    pub fn edit_classifier(&self) -> Option<&EditClassifierResult> {
        self.results.get("edit_classifier")?.downcast_ref::<EditClassifierResult>()
    }
    pub fn plugin_result(&self, analyzer_id: &str) -> Option<&AnalysisResult> {
        self.results.get(analyzer_id)?.downcast_ref::<AnalysisResult>()
    }
    pub(crate) fn insert(&mut self, analyzer_id: String, result: Box<dyn Any + Send + Sync>) {
        self.results.insert(analyzer_id, result);
    }
}
```

### Pattern 3: Kahn's Algorithm for Topological Sort
BFS-based topological sort that produces execution order and detects cycles in one pass. Kahn's naturally produces forward order (not reverse like DFS), gives deterministic output when the queue is sorted, and cycle detection is a natural by-product (if not all nodes are visited, there is a cycle). ~40 lines of code.

### Pattern 4: Panic-Isolated Execution with catch_unwind
Each analyzer runs inside `catch_unwind(AssertUnwindSafe(...))` so a panic in one analyzer does not crash the entire analysis pipeline. Extract panic message immediately via downcast to `&str` or `String`; store only the string in `AnalyzerStatus::Failed`.

### Pattern 5: AnalysisContext Design
Read-only context providing DiffResult and MetricsEngine access to analyzers:

```rust
pub struct AnalysisContext<'a> {
    pub diff: &'a DiffResult,
    pub metrics_engine: &'a MetricsEngine,
}
```

Provides helper methods: `source_text_for_op()`, `target_text_for_op()`, `source_tokens_for_op()`, `target_tokens_for_op()`, `change_operations()`.

### Pattern 6: Common AnalysisResult for Plugins
The common dynamic struct for custom analyzers, designed to serialize cleanly to JSON:

```rust
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalysisResult {
    pub annotations: Vec<Annotation>,
    pub summaries: Vec<RegionSummary>,
    pub scores: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}
```

### Anti-Patterns to Avoid
- **Trait objects across FFI:** Do NOT put `dyn AnalyzerPlugin` or `dyn Any` in the Python/WASM boundary
- **Mutable context:** Do NOT make `AnalysisContext` mutable -- essential for future parallel execution
- **Lazy cycle detection:** Do NOT defer cycle detection to execution time -- validate on every `register()`
- **Exposing Any in public API:** The `HashMap<String, Box<dyn Any>>` is internal; public API uses typed accessors

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Error types | Manual Display/Error impls | `thiserror` 2.0 | Existing project pattern |
| Readability scores | Re-implement Flesch/Kincaid | Phase 4 `MetricsEngine` | Already computed; ReadabilityAnalyzer should call existing metrics |
| Similarity metrics | Re-implement Jaccard/Cosine | Phase 4 similarity metrics | SemanticAnalyzer reuses existing from metrics |
| Token text extraction | Custom span-to-text | `token_texts()` helper from `metrics::similarity` | Already exists |
| Graph library | petgraph | Kahn's algorithm (~40 lines) | Analyzer DAG is tiny |
| Typed heterogeneous map | typedmap crate | `HashMap<String, Box<dyn Any>>` + typed accessors | ~15 lines |
| Syllable counting | Re-implement | Phase 4 `count_syllables()` | Already exists |

## Built-in Result Structs

### SemanticResult
```rust
pub struct SemanticResult {
    pub operation_scores: Vec<OperationSimilarity>,
    pub overall_similarity: f64,
}
pub struct OperationSimilarity {
    pub operation_index: usize,
    pub similarity: f64,
    pub method: String,
}
```

### StylisticResult
```rust
pub struct StylisticResult {
    pub voice_shifts: Vec<VoiceShift>,
    pub tone_shifts: Vec<ToneShift>,
    pub structural_changes: StructuralChanges,
}
pub struct VoiceShift { pub operation_index: usize, pub from: Voice, pub to: Voice, pub confidence: f64 }
pub enum Voice { Active, Passive, Unknown }
pub struct ToneShift { pub operation_index: usize, pub dimension: ToneDimension, pub from_score: f64, pub to_score: f64, pub confidence: f64 }
pub enum ToneDimension { FormalInformal, HedgingAssertive }
pub struct StructuralChanges { source/target avg_sentence_length, delta, repetition_change }
```

### ReadabilityResult
```rust
pub struct ReadabilityResult {
    pub source_scores: ReadabilityScores,
    pub target_scores: ReadabilityScores,
    pub delta: ReadabilityScores,
    pub attributions: Vec<ReadabilityAttribution>,
}
pub struct ReadabilityScores { flesch_reading_ease, flesch_kincaid_grade, gunning_fog, coleman_liau, ari, smog_index }
pub struct ReadabilityAttribution { pub operation_indices: Vec<usize>, pub estimated_impact: f64, pub description: String }
```

### EditClassifierResult + IntentCategory
```rust
pub struct EditClassifierResult {
    pub operations: Vec<OperationClassification>,
    pub groups: Vec<GroupClassification>,
}
pub struct OperationClassification {
    pub operation_index: usize,
    pub categories: Vec<(IntentCategory, f64)>,
}
pub struct GroupClassification {
    pub operation_indices: Vec<usize>,
    pub categories: Vec<(IntentCategory, f64)>,
}
pub enum IntentCategory { Correction, Expansion, Deletion, Rewrite, Refinement, Formatting, Other }
```

### AnalyzerStatus and Execution Metadata
```rust
pub enum AnalyzerStatus { Success, Failed(String), Skipped(String) }
pub struct AnalyzerExecutionMeta { analyzer_id, status, duration, order_index }
```

### ScoringBackend Trait (BERT-Ready)
```rust
pub trait ScoringBackend: Send + Sync {
    fn score(&self, source_tokens: &[&str], target_tokens: &[&str]) -> f64;
    fn name(&self) -> &str;
}
pub struct HeuristicScoring; // Jaccard similarity on word sets
```

## Heuristic Algorithms

### Passive Voice Detection
"to be" forms (am, is, are, was, were, be, been, being) + past participle patterns (-ed, -en, -wn, -ne).

### Hedging/Assertive Tone Detection
Hedging words: might, could, perhaps, possibly, somewhat, apparently, seemingly, arguably, presumably, likely, etc.
Assertive words: must, will, always, never, clearly, obviously, definitely, certainly, undoubtedly, etc.
Score: (hedging_count - assertive_count) / total_words. Range -1.0 (assertive) to 1.0 (hedging).

### EditClassifier Heuristics
- Insert → Expansion (0.8)
- Delete → Deletion (0.8)
- Replace: check whitespace-only (→ Formatting), typo-fix (→ Correction), length ratio > 1.5 (→ Expansion), < 0.5 (→ Deletion/Refinement), else (→ Refinement/Rewrite)

### Adjacent Operation Grouping
Group non-Equal operations separated by small Equal gaps (configurable gap_threshold in token count).

## Common Pitfalls

1. **UnwindSafe boundary:** Wrap `catch_unwind` closure in `AssertUnwindSafe`. Safe because context is immutable, report is read-only to analyzers.
2. **Incremental vs batch cycle detection:** Batch validation on each `register()` is O(V+E) per call. Fine for <20 nodes.
3. **EditOperation index mismatch:** Annotations reference original operations vector indices. Helpers return `(index, &EditOperation)` tuples.
4. **Confidence score calibration:** All scores mean "probability this annotation is correct" (0.0-1.0). Document semantics per analyzer.
5. **Panic payload drop:** Extract message immediately via downcast; don't store raw payload.
6. **SemanticAnalyzer ScoringBackend over-engineering:** Keep trait to 2 methods (`score` + `name`).
7. **ReadabilityAnalyzer recomputing metrics:** Use Phase 4 MetricsEngine, don't re-implement formulas.

## State of the Art

- The CONTEXT.md 7-category IntentCategory taxonomy **supersedes** the REQUIREMENTS.md 6-category version
- Rust 1.86+ trait upcasting is available but not needed (analyzers return `Box<dyn Any>` directly)

## Open Questions

1. **token_texts visibility:** Recommend duplicating the 8-line utility in analysis module
2. **ReadabilityAnalyzer attribution:** Recommend per-group, not per-operation (too expensive)
3. **AnalysisContext lifetime:** Use `AnalysisContext<'a>`, consistent with `MetricInput<'a>` from Phase 4
4. **Word lists:** Ship with const arrays, document as tunable

## Key Findings

1. **No new dependencies required.** Entire framework uses stdlib + existing project deps.
2. **Architecture mirrors Phase 4 MetricsEngine pattern.** PluginRegistry ≈ MetricRegistry, AnalysisCoordinator ≈ MetricsEngine, AnalyzerPlugin ≈ Metric.
3. **Kahn's algorithm is the right choice for topological sort.** ~40 lines, produces forward order, detects cycles.
4. **Typed + dynamic result storage** via `HashMap<String, Box<dyn Any + Send + Sync>>` with typed accessors.
5. **7-category IntentCategory from CONTEXT.md supersedes 6-category REQUIREMENTS.md version.**

---
*Research completed: 2026-02-07*
