# Plan 04-06 Summary: Pairwise Set-Based Metrics

## Status: ✓ Complete

## What Was Built
7 pairwise set/vector metrics operating on token text:

| Metric | ID | Type | Formula |
|--------|----|------|---------|
| JaccardSimilarityMetric | jaccard_similarity | Float | \|A∩B\|/\|A∪B\| |
| CosineSimilarityMetric | cosine_similarity | Float | dot/(mag_a*mag_b) |
| DiceCoefficientMetric | dice_coefficient | Float | 2\|A∩B\|/(\|A\|+\|B\|) |
| OverlapCoefficientMetric | overlap_coefficient | Float | \|A∩B\|/min(\|A\|,\|B\|) |
| LengthRatioMetric | length_ratio | Float | target.tokens/source.tokens |
| WordCountDiffMetric | word_count_diff | Integer | target-source token count |
| CharCountDiffMetric | char_count_diff | Integer | target-source char count |

## Key Decisions
- **Token text comparison, not StringId**: StringId is only valid within one TextStore. Pairwise metrics compare texts from different TextStores, so we extract actual word strings from normalized text via spans.
- Added `token_texts()` helper in `similarity/mod.rs`
- Set-based metrics use `HashSet<&str>`, cosine uses `HashMap<&str, usize>` frequency vectors
- Empty-vs-empty = 1.0 for similarity, 0 for diffs
- All reject Single input with InvalidInput error

## Files Created/Modified
- `crates/core/src/metrics/similarity/mod.rs` (token_texts helper + declarations)
- 7 metric files in `crates/core/src/metrics/similarity/`

## Tests
- 55 similarity tests total (includes plan 04-07 metrics)

## Commit
`feat(04-06): pairwise set-based metrics`
