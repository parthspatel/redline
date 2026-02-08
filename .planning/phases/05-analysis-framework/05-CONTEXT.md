# Phase 5: Analysis Framework - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Pluggable analyzer system with dependency resolution that enriches diff results with semantic insights. Analyzers register with declared dependencies, execute in topological order via an ExecutionPlanner, and produce an AnalysisReport from DiffResult + metrics. Includes 4 built-in analyzers (Semantic, Stylistic, Readability, EditClassifier). Async/parallel execution is NOT in scope (Phase 7).

</domain>

<decisions>
## Implementation Decisions

### Analyzer Output Design
- Per-operation annotations + optional region-level summaries (both granularities)
- Typed built-ins + dynamic plugins for result storage:
  - Built-in analyzers return specific structs: `report.semantic()` -> `Option<&SemanticResult>`, `report.readability()` -> `Option<&ReadabilityResult>`, etc.
  - Plugin (custom) analyzers return a common dynamic struct (`AnalysisResult` with annotations, summary, scores, metadata)
- Annotations carry a `confidence: f64` (0.0–1.0) for consumer-side filtering

### Edit Classification Taxonomy
- Coarse taxonomy (7 categories): Correction, Expansion, Deletion, Rewrite, Refinement, Formatting, Other
- Each edit gets a ranked list with confidence: `Vec<(IntentCategory, f64)>` sorted by confidence descending
- Always return all ranked categories — consumer decides what to filter
- Classify at both levels: per-operation classifications + optional group-level classification for adjacent related operations

### Failure & Degradation Behavior
- Explicit error entry + fallback: failed analyzer returns partial results if possible, plus `AnalyzerStatus::Failed(reason)`
- Dependents of failed analyzers are skipped: `AnalyzerStatus::Skipped(reason: "dependency X failed")`
- Catch panics via `std::panic::catch_unwind` around each analyzer — panic becomes a Failed entry, remaining analyzers continue
- Full execution metadata per analyzer: wall-clock duration and execution order index

### Built-in Analyzer Depth
- **SemanticAnalyzer:** Similarity-scored per edit region (token overlap, shared n-grams, Jaccard on word sets). Heuristic-based for v1. Interface designed to be BERT-ready so a Python plugin (Phase 8) or ONNX runtime can drop in later without redesign.
- **StylisticAnalyzer:** Detects voice shifts (active/passive), tone shifts (formal/informal, hedging/assertive via word list heuristics), and structural style changes (sentence length, paragraph density, repetition patterns).
- **ReadabilityAnalyzer:** Document-level readability delta (before vs. after Flesch/Kincaid/etc. scores from Phase 4 metrics) + per-region attribution identifying which edit operations caused readability to change.
- **EditClassifier:** Coarse 7-category taxonomy with ranked confidence scores. Configurable heuristic thresholds with sensible defaults, overridable per-category.

### Claude's Discretion
- Exact heuristic algorithms for each built-in analyzer
- AnalyzerPlugin trait design details (metadata fields, cost model)
- ExecutionPlanner DAG implementation approach
- AnalysisContext internal structure
- Grouping algorithm for adjacent related operations

</decisions>

<specifics>
## Specific Ideas

- SemanticAnalyzer interface should accept a scoring backend so BERT/ONNX can replace heuristics without changing the analyzer registration or report shape
- The common dynamic struct for plugins should serialize cleanly to JSON for Python bindings (Phase 8) and WASM (Phase 9) — no trait objects across FFI
- EditClassifier group-level classification mirrors the per-operation + region-level pattern used throughout

</specifics>

<deferred>
## Deferred Ideas

- BERT/ML-based semantic similarity scoring — Phase 8 (Python analyzer plugin) or future ONNX runtime integration
- Parallel analyzer execution — Phase 7 (Async Support)

</deferred>

---

*Phase: 05-analysis-framework*
*Context gathered: 2026-02-07*
