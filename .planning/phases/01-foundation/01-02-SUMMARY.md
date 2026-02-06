---
phase: 01-foundation
plan: 02
subsystem: types
tags: [span, char-mapping, u32, copy, binary-search, composition, tdd, no_std, wasm]

requires: [01-01]
provides:
  - Span type (8-byte Copy struct with u32 start/end)
  - CharMapping type with bidirectional lookup and composition
  - NormalizeError integration for invalid positions and composition failures
affects: [01-03, 01-04, all-future-phases]

tech-stack:
  added: []
  patterns: [binary-search lookup, linear-scan reverse lookup, mapping composition, proptest property testing]

key-files:
  created:
    - crates/core/src/span.rs
    - crates/core/src/char_mapping.rs
  modified:
    - crates/core/src/lib.rs
    - crates/core/src/text_store.rs

key-decisions:
  - "Span uses u32 fields (not usize) — keeps struct at exactly 8 bytes, Copy-friendly"
  - "Span::overlaps uses strict < comparison — adjacent spans [0,5) and [5,10) do NOT overlap"
  - "Span::contains treats empty spans at boundaries as contained (inclusive end check)"
  - "CharMapping::new sorts alignments by original position for binary search in to_normalized"
  - "CharMapping::to_original uses linear scan (normalized positions may not be sorted after composition)"
  - "CharMapping::compose chains through to_normalized on the other mapping per pair"
  - "text_store.rs stubs expanded with method signatures to allow crate-wide compilation"

patterns-established:
  - "Half-open range semantics [start, end) consistently across Span API"
  - "Result-based error handling for CharMapping lookups (no panics on invalid positions)"
  - "Composition pattern: for each (orig, mid) in self, lookup mid in other to get final"
  - "proptest for property-based invariants (identity round-trip, compose-with-identity noop)"
  - "#[inline] on small accessor methods for performance"
  - "serde derives behind cfg_attr feature gate"

duration: 8 min
completed: 2026-02-06
---

# Phase 1 Plan 02: Span and CharMapping Types Summary

**TDD-implemented Span (8-byte Copy type) and CharMapping (position mapping with composition) for text diffing infrastructure**

## Performance

- **Duration:** ~8 min
- **Tasks:** 2/2 (4 commits: 2 RED + 2 GREEN)
- **Tests:** 38 new (25 Span + 11 CharMapping unit + 2 CharMapping proptest)
- **Files:** 4 (2 created, 2 modified)

## Accomplishments

- **Span type:** 8-byte `Copy` struct with `u32` start/end fields
  - Constructors: `new(start, end)` with panic guard, `empty(pos)`
  - Queries: `len()`, `is_empty()`, `contains(other)`, `contains_position(pos)`, `overlaps(other)`
  - Operations: `merge(other)`, `split_at(offset)`
  - Half-open range semantics: adjacent spans do not overlap, empty spans contain no positions
  - Verified: `Copy + Clone + Debug + PartialEq + Eq + Hash + Send + Sync` via `static_assertions`
  - Size verified: `core::mem::size_of::<Span>() == 8`

- **CharMapping type:** Position mapping between original and normalized text
  - Constructors: `new(alignments)` with sort, `identity(len)`
  - Bidirectional lookup: `to_normalized(pos)` via binary search, `to_original(pos)` via linear scan
  - Composition: `compose(other)` chains through intermediate positions
  - Three-mapping composition verified with full round-trip (SUCCESS CRITERION met)
  - Error handling: returns `NormalizeError::InvalidPosition` / `CompositionFailed`
  - Property tests: identity always round-trips, compose with identity is noop

- **All types compile for WASM** (`--no-default-features --features wasm`)
- **Full crate test suite:** 61 tests pass (including error + text_store tests)

## Task Commits

1. **Task 1 RED:** `764a562` — `test(01-02): add failing tests for Span type (RED)` (25 tests, all fail)
2. **Task 1 GREEN:** `eab05e5` — `feat(01-02): implement Span type (GREEN)` (25 tests pass)
3. **Task 2 RED:** `4494530` — `test(01-02): add failing tests for CharMapping (RED)` (13 tests, all fail)
4. **Task 2 GREEN:** `718f167` — `feat(01-02): implement CharMapping with composition (GREEN)` (13 tests pass)

## Decisions Made

- **Span u32 fields:** Supports texts up to ~4 GiB while keeping struct at exactly 8 bytes. No `SmallVec` — `Span` is a single value type, `Copy`-friendly as specified in project decisions.
- **Half-open range overlap:** `self.start < other.end && other.start < self.end` naturally excludes adjacent and empty spans without special cases.
- **CharMapping sort on construction:** `new()` sorts by original position, enabling O(log n) binary search in `to_normalized()`.
- **Linear scan for `to_original`:** Normalized positions may not maintain sort order after composition (e.g., reordering normalizations), so binary search on the second component is not safe. Linear scan is correct for all cases.
- **text_store.rs stubs:** Expanded stub struct fields and method signatures (still `todo!()` bodies) to allow the crate to compile for Span/CharMapping development. These stubs were already present from Plan 01-03 concurrent work.

## Deviations from Plan

- **text_store.rs required stub expansion:** The existing `text_store.rs` had bare struct stubs without method signatures, causing compilation failures. Added method stubs to unblock crate-wide compilation. This was necessary because the plan assumed text_store would already have compilable stubs.

## Issues Encountered

- **hashbrown 0.16 DefaultHashBuilder:** The `default-hasher` feature was already present in Cargo.toml (added by Plan 01-03), resolving `HashMap::new()` compilation. No action needed.
- **rustfmt pre-commit hook:** Reformatted code on first GREEN commit attempt. Re-staged and committed successfully on second try.

## Next Phase Readiness

- Span is ready for use in tokenization (Plan 01-04) and all diffing operations
- CharMapping is ready for normalization pipeline composition
- Both types are no_std compatible and WASM-verified
- No blockers for Plan 01-03 (TextStore) or Plan 01-04 (Token/Tokenizer)

---
*Phase: 01-foundation*
*Completed: 2026-02-06*
