# Phase 6: Orchestration & Configuration - Context

**Gathered:** 2026-02-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Single entry point (`Redline`) coordinating the full pipeline (text processing → diff → metrics → analysis) with unified configuration, presets, content-addressed caching, and a query API for filtering edit operations. This is the public-facing API that downstream phases (Python bindings, WASM) will consume.

</domain>

<decisions>
## Implementation Decisions

### Pipeline API Design
- Main entry point is `Redline` (not `DiffOrchestrator`) — `Redline::new(config).diff("old", "new")`
- Config at construction sets defaults, per-call overrides for one-off tweaks — both patterns supported
- Single `RedlineResult` struct returned from every call — contains diff, metrics, and analysis report; users access what they need via fields
- Pluggable stages — users can inject custom `impl Tokenizer`, `impl Normalizer`, etc. at construction time
- Built-in stages configured via config, custom implementations registered alongside built-ins

### Presets
- **2 presets only** (not 4): Fast and Comprehensive
- **Fast** — diff + basic metrics (counts, similarity), no analysis
- **Comprehensive** — everything: full normalization, diff, all metrics, all analyzers (this is the **default**)
- Presets are **starting points**, not fixed — `ConfigBuilder::preset(Fast).with_analysis(true)` overrides individual settings
- Users who want something in between start from either preset and override

### Query API
- Primary consumer: programmatic callers first, could power a UI later
- **Predicate combinators** with operator overloads — `Filter::kind(Replace) & Filter::min_length(10)`
- `&` for AND, `|` for OR, `!` for NOT
- Filter dimensions: EditKind, span range, token count, text content matching, and analysis metadata (e.g., IntentCategory from EditClassifier)
- Returns an **iterator** — lazy, chainable, zero allocation for early termination; users `.collect()` when they need a concrete collection

### Cache Strategy
- **Full pipeline memoization** — metrics, diff results, and analysis reports all cached by content hash
- **Per-instance cache** — each `Redline` instance owns its cache, no cross-instance sharing
- **LRU eviction** — entry count-based by default, with optional memory cap as safety valve
- **Enabled by default** — consistent with Comprehensive default; users disable for deterministic memory or one-shot processing

### Claude's Discretion
- ConfigBuilder fluent API shape and validation details
- Internal cache key structure and hashing strategy
- Memory measurement approach for optional memory cap
- Error types and error handling for invalid configurations
- CacheManager lock granularity and parking_lot usage patterns

</decisions>

<specifics>
## Specific Ideas

- `Redline::new(config).diff("old", "new")` — clean, brandable entry point
- Two presets keep the API surface small; users compose what they need via overrides
- Iterator-based query API aligns with Rust idioms — no forced allocation
- Full caching by default matches "comprehensive by default" philosophy

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-orchestration-config*
*Context gathered: 2026-02-09*
