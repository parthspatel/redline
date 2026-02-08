# Phase 4: Metrics Engine - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Quantitative measurements of text properties and pairwise comparisons. Delivers: Metric trait, MetricRegistry, 20+ single-text metrics, 15+ pairwise metrics, MetricsEngine with lazy evaluation, and content-addressed caching with LRU eviction. Does NOT include analyzer plugins, orchestration, or async execution — those are later phases.

</domain>

<decisions>
## Implementation Decisions

### Metric Computation Model
- Pull-based lazy evaluation: request a metric by ID, engine resolves dependencies and computes on demand
- `get("metric_id")` for single metric, `get_many(["a", "b", "c"])` for batch (internally loops over `get`, benefits from caching shared dependencies)
- When a dependency fails, return `MetricValue::Unavailable` (sentinel) and log the failure — don't propagate errors to downstream metrics
- `MetricValue` is a tagged enum: `Integer(i64)`, `Float(f64)`, `Unavailable` — preserves type semantics (word count is an integer, not 42.0)

### Dependency Graph
- Both static AND dynamic dependencies
- `static_dependencies()` — fixed list known at compile time, validated at registration
- `dynamic_dependencies(input)` — additional dependencies discovered at compute time, validated on the fly with runtime cycle detection
- Full dependency set = union of static + dynamic
- Engine validates static graph upfront (catches cycles and missing IDs early), handles dynamic additions during compute

### Metric Organization & TDD Strategy
- One type per metric: `WordCountMetric`, `FleschKincaidMetric`, `JaccardSimilarityMetric`, etc. — each is its own struct implementing the `Metric` trait
- Organized into modules by category (`counts/`, `readability/`, `similarity/`) but each metric is independently testable
- TDD waves ordered foundation-up: zero-dependency metrics first (counts), then metrics that depend on those (readability), then pairwise metrics
- Each wave unlocks the next — no mocking dependencies that don't exist yet
- Test granularity: known-value tests (hand-calculated expected values) + property tests (word_count >= 0, similarity in [0.0, 1.0], Levenshtein(a, a) == 0)
- MetricsEngine integration tested incrementally with each wave — engine grows alongside the metrics, no big-bang integration at the end

### Caching & Content Addressing
- Cache key: hash(ProcessedText) + metric_id — each (input, metric) pair is a unique entry
- Shared dependencies benefit automatically: `word_count` cached once, reused by every metric that depends on it
- Caching is opt-in at the engine level:
  - `MetricsEngine::new()` — no cache, direct computation
  - `MetricsEngine::with_cache(capacity)` — LRU cache enabled with user-specified capacity
- LRU eviction by capacity only

### Custom Metric Extensibility
- Full integration: custom metrics implement the same `Metric` trait, get cached, can declare dependencies on built-ins, built-ins can depend on custom ones — no distinction at runtime
- Override allowed: custom metrics can replace built-in metrics, but requires explicit opt-in via `registry.register(metric, override=true)` — default `register(metric)` errors on ID collision
- No namespacing — override is the mechanism for collision handling
- Registration validates static dependency IDs exist in registry (catches typos early)
- Dynamic dependencies validated at compute time (runtime cycle detection)

### Claude's Discretion
- LRU cache TTL policy (whether to add time-based expiry on top of LRU capacity eviction)
- Content hashing algorithm choice
- Exact module file structure within category directories
- Error logging mechanism for Unavailable metrics
- Default cache capacity value if a convenience constructor is added

</decisions>

<specifics>
## Specific Ideas

- TDD is a hard requirement for this phase — tests written before implementation for every metric
- Engine integration tests added incrementally with each wave, not deferred to the end
- Property tests consistent with Phase 2 and 3 approach (proptest)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-metrics-engine*
*Context gathered: 2026-02-07*
