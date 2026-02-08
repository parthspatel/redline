# Plan 04-07 Summary: Pairwise Edit-Distance Metrics

## Status: ✓ Complete

## What Was Built
5 edit-distance and sequence-matching metrics:

| Metric | ID | Type | Algorithm |
|--------|----|------|-----------|
| LevenshteinDistanceMetric | levenshtein_distance | Integer | Two-row DP, O(min(n,m)) space |
| DamerauLevenshteinDistanceMetric | damerau_levenshtein | Integer | OSA with transpositions (three-row DP) |
| HammingDistanceMetric | hamming_distance | Integer | Position-wise comparison (equal-length only) |
| JaroSimilarityMetric | jaro_similarity | Float | Matching window + transposition counting |
| JaroWinklerSimilarityMetric | jaro_winkler_similarity | Float | Jaro + prefix bonus (depends on jaro_similarity) |

## Key Decisions
- All operate on token text strings (not StringId) for cross-TextStore compatibility
- Jaro-Winkler depends on jaro_similarity via the engine dependency system
- Hamming returns InvalidInput error for unequal-length sequences
- Damerau-Levenshtein correctly gives distance 1 for adjacent transpositions (vs Levenshtein's 2)
- Jaro exposed as `pub fn jaro_similarity()` for use by JaroWinkler tests

## Files Created/Modified
- 5 metric files in `crates/core/src/metrics/similarity/`
- `crates/core/src/metrics/similarity/mod.rs` (updated with all 12 declarations)

## Tests
- 55 similarity tests total, 806 tests project-wide, 0 warnings

## Commit
`feat(04-07): pairwise edit-distance metrics`
