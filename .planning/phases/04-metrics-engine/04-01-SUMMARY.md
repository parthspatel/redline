---
phase: 04-metrics-engine
plan: 01
status: complete
started: 2026-02-07
completed: 2026-02-07
commits:
  - hash: 60f1e99
    message: "feat(04-01): metrics engine infrastructure"
key-files:
  created:
    - crates/core/src/metrics/mod.rs
    - crates/core/src/metrics/error.rs
    - crates/core/src/metrics/registry.rs
    - crates/core/src/metrics/cache.rs
    - crates/core/src/metrics/engine.rs
    - crates/core/src/metrics/builtins.rs
    - crates/core/src/metrics/counts/mod.rs
    - crates/core/src/metrics/readability/mod.rs
    - crates/core/src/metrics/similarity/mod.rs
  modified:
    - crates/core/Cargo.toml
    - crates/core/src/lib.rs
    - crates/core/src/error.rs
---

## Summary

Built the complete metrics engine infrastructure: Metric trait, MetricValue, MetricInput, MetricError, MetricRegistry with DFS cycle detection, ContentHash with deterministic FoldHash, MetricCache with LRU eviction, and MetricsEngine with pull-based lazy evaluation and runtime cycle detection.

## Deliverables

- **Metric trait**: Object-safe with id(), compute(), static_dependencies(), dynamic_dependencies(), dependency_kind(), cost()
- **MetricValue**: Tagged enum with Integer(i64), Float(f64), Unavailable variants
- **MetricInput**: Single and Pairwise variants referencing ProcessedText
- **MetricError**: 7 error variants with thiserror, integrated into RedlineError
- **MetricRegistry**: register(), register_with_override(), validate() with DFS cycle detection (White/Gray/Black coloring)
- **ContentHash**: Deterministic u64 via foldhash::fast::FixedState (seed 0) — same content always hashes identically
- **MetricCache**: LRU cache wrapping lru::LruCache with capacity-only eviction
- **MetricsEngine**: Lazy evaluation with RefCell-based cycle detection, optional caching, get() and get_many()
- **Stub modules**: counts, readability, similarity, builtins ready for later plans

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| foldhash::fast::FixedState for ContentHash | hashbrown::DefaultHashBuilder uses per-instance random seeding (DoS protection), causing same content to hash differently across calls — FixedState with seed 0 gives deterministic hashes needed for caching |
| RefCell<MetricCache> for interior mutability | Engine.get() takes &self for ergonomics; cache needs mutation on reads |
| RefCell<HashSet<String>> for cycle detection | Fresh per get() call, shared across get_many() for correct cross-metric cycle detection |

## Test Coverage

45 unit tests covering:
- MetricValue accessors and variants
- MetricRegistry: register, duplicate errors, override, DFS cycle detection (2-node, 3-node, self-referential, diamond)
- MetricCache: put/get, LRU eviction, clear, zero-capacity panic
- MetricsEngine: simple metric, missing metric, cache hit, dependency resolution, cycle detection, get_many, failed metric

## Self-Check: PASSED

All 652 tests pass (545 lib + 107 integration/doc). No regressions.
