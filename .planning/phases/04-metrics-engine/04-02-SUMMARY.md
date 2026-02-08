---
phase: 04-metrics-engine
plan: 02
status: complete
started: 2026-02-07
completed: 2026-02-07
commits:
  - hash: 2dbf558
    message: "feat(04-02): count metrics batch 1"
key-files:
  created:
    - crates/core/src/metrics/counts/word_count.rs
    - crates/core/src/metrics/counts/char_count.rs
    - crates/core/src/metrics/counts/byte_count.rs
    - crates/core/src/metrics/counts/sentence_count.rs
    - crates/core/src/metrics/counts/syllable_count.rs
    - crates/core/src/metrics/counts/unique_word_count.rs
  modified:
    - crates/core/src/metrics/counts/mod.rs
---

## Summary

Implemented 6 foundational count metrics via TDD: word_count, char_count, byte_count, sentence_count, syllable_count, unique_word_count. All are zero-dependency metrics that form the base layer for readability metrics.

## Deliverables

- **WordCountMetric**: tokens.len() as Integer
- **CharCountMetric**: original.chars().count() (Unicode codepoints)
- **ByteCountMetric**: original.len() (UTF-8 bytes)
- **SentenceCountMetric**: sentence-ender groups (.!?) with consecutive-ender deduplication
- **SyllableCountMetric**: vowel-group heuristic with silent-e, -ed, -le rules
- **UniqueWordCountMetric**: distinct StringId values via HashSet
- **count_syllables()** helper function exported for readability metrics

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Silent-e and -ed rules are mutually exclusive | Prevents double-counting (e.g., "created" was off by 1) |
| "ea" digraph treated as one vowel group | Heuristic limitation; "created" counts as 2 not 3, acceptable for readability formulas |

## Self-Check: PASSED
