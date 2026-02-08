# Plan 04-08 Summary: Delta Metrics + register_builtins + Integration Tests

## Status: ✓ Complete

## What Was Built

### Delta/Ratio Metrics (3)
| Metric | ID | Formula |
|--------|----|---------|
| ReadabilityDeltaMetric | readability_delta | target_FRE - source_FRE (self-contained) |
| GradeLevelDeltaMetric | grade_level_delta | target_FK - source_FK (self-contained) |
| SimilarityRatioMetric | similarity_ratio | 2*LCS/total (two-row DP) |

### register_builtins()
- Registers all 37 metrics in correct dependency order
- 12 count (no deps) -> 10 readability (depend on counts) -> 15 pairwise (jaro before jaro_winkler)
- Validates dependency graph after registration
- Re-exported from `redline_core::register_builtins`

### Comprehensive Integration Tests (13 tests)
1. Full pipeline single-text (all 22 single metrics)
2. Full pipeline pairwise (all 15 pairwise metrics)
3. Identical texts identity checks
4. Lazy evaluation without cache
5. Dependency chain with cache verification
6. Custom metric integration alongside builtins
7. Builtin override via register_with_override
8. Nonexistent metric -> Unavailable
9. Missing dependency -> validation error
10-13. Property tests (proptest): counts non-negative, similarity ranges, self-identity, no panics

## Key Decisions
- Delta metrics (readability_delta, grade_level_delta) are self-contained: they compute FRE/FK independently on each text rather than depending on single-text metrics (which can't be auto-applied to pairwise inputs)
- SimilarityRatio uses LCS (longest common subsequence) on token text sequences
- register_builtins returns Result for error propagation
- Total: 37 built-in metrics (12 count + 10 readability + 15 pairwise)

## Files Created/Modified
- `crates/core/src/metrics/similarity/readability_delta.rs`
- `crates/core/src/metrics/similarity/grade_level_delta.rs`
- `crates/core/src/metrics/similarity/similarity_ratio.rs`
- `crates/core/src/metrics/similarity/mod.rs` (updated)
- `crates/core/src/metrics/builtins.rs` (populated)
- `crates/core/src/metrics/mod.rs` (re-export)
- `crates/core/src/lib.rs` (re-export)
- `crates/core/tests/metrics_integration.rs` (new)

## Tests
- 831 total project tests, 0 failures, 0 warnings
- 13 integration tests + 4 proptest property tests

## Commit
`feat(04-08): delta metrics, register_builtins, integration tests`
