---
phase: 01-foundation
plan: 03
subsystem: text-store
tags: [text-store, string-interning, deduplication, two-phase, no_std, wasm, arc]

requires: [01-01]
provides:
  - StringId newtype (Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Send, Sync)
  - TextStoreBuilder (mutable, deduplicating string interning)
  - TextStore (immutable, shared via Arc)
  - Two-phase design: builder -> build() -> Arc<TextStore>
affects: [01-04, future-tokenization, future-diffing]

tech-stack:
  added: [foldhash 0.2 (via hashbrown default-hasher)]
  patterns: [two-phase builder, string interning with dedup, hashbrown HashMap, no_std conditional compilation]

key-files:
  created:
    - crates/core/src/text_store.rs
  modified:
    - crates/core/src/lib.rs
    - crates/core/Cargo.toml
    - Cargo.lock

key-decisions:
  - "Vec<String> for storage instead of bumpalo arena — correctness first, optimize later"
  - "hashbrown default-hasher feature added for FoldHash-based HashMap::new()"
  - "alloc::borrow::ToOwned imported for no_std WASM compatibility"
  - "StringId is a newtype over u32 with Copy — fundamental architectural decision"

patterns-established:
  - "Two-phase pattern: mutable builder for interning, immutable store for lookups"
  - "StringId(u32) as lightweight Copy handle into the store"
  - "StoreError::IdNotFound(u32) for invalid ID lookups"
  - "hashbrown HashMap with FoldHash default hasher for no_std deduplication"

duration: 5 min
completed: 2026-02-06
---

# Phase 1 Plan 03: TextStore Two-Phase Design Summary

**String interning system with deduplication: TextStoreBuilder (mutable) promotes to Arc<TextStore> (immutable, shared)**

## Performance

- **Duration:** ~5 min
- **Tasks:** 2/2 (RED + GREEN)
- **Tests:** 13 unit tests + 1 doc-test, all passing
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments
- StringId newtype over u32 with Copy semantics and full trait coverage
- TextStoreBuilder: intern(), resolve(), len(), is_empty(), build()
- TextStore: resolve(), len(), is_empty() (immutable after build)
- Deduplication: same string always returns same StringId
- 10,000 strings interned and all lookups succeed (success criterion met)
- WASM target compiles with no_std + alloc
- Doc-test demonstrates the full workflow

## Task Commits

1. **RED: Failing tests for TextStore** — included in `764a562` (test, shared commit with 01-02)
2. **GREEN: TextStore implementation** — `513b7ce` (feat)

## Types Implemented

### StringId
- `pub struct StringId(u32)` with Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord
- `as_u32(self) -> u32` accessor
- Send + Sync (verified via static_assertions)

### TextStoreBuilder
- `strings: Vec<String>` indexed by StringId
- `dedup: HashMap<String, StringId>` for O(1) dedup lookup (hashbrown + FoldHash)
- `intern(&mut self, s: &str) -> StringId` — dedup or insert
- `resolve(&self, id: StringId) -> Result<&str, StoreError>` — bounds-checked
- `build(self) -> Arc<TextStore>` — consume builder, produce shared store

### TextStore
- `strings: Vec<String>` (immutable after build)
- `resolve(&self, id: StringId) -> Result<&str, StoreError>` — bounds-checked
- Shared via Arc for thread-safe concurrent reads

## Test Coverage

| Test | Description |
|------|-------------|
| string_id_as_u32 | StringId accessor returns raw u32 |
| builder_intern_and_resolve | Basic intern -> resolve round-trip |
| builder_deduplication | Same string yields same StringId |
| builder_different_strings_different_ids | Different strings yield different IDs |
| builder_is_empty_and_len | len/is_empty track unique strings |
| promote_and_resolve | build() -> Arc<TextStore>, all IDs resolve |
| store_len_matches_unique_strings | Duplicates not double-counted |
| arc_is_shared | Arc::clone resolves same data |
| builder_id_not_found | Invalid ID returns StoreError::IdNotFound |
| store_id_not_found | Invalid ID on store returns StoreError::IdNotFound |
| intern_empty_string | Empty string interning works |
| intern_unicode | Emoji, CJK, Arabic all round-trip |
| ten_thousand_strings_interned_and_resolved | 10K strings scale test |

## Deviations from Plan

- **hashbrown default-hasher feature:** The plan did not mention needing this, but hashbrown 0.16 with `default-features = false` does not provide `HashMap::new()`. Added `default-hasher` feature to enable FoldHash-based default hasher. This aligns with the project decision to use FoldHash.
- **RED commit shared:** The RED phase tests were committed together with 01-02 Span tests due to parallel agent execution and pre-commit hook interaction. The content is correctly separated in text_store.rs.
- **Vec<String> instead of bumpalo arena:** Plan mentioned bumpalo but the task instructions correctly noted Vec<String> is the right first approach for correctness. Arena optimization deferred to profiling.

## Issues Encountered

- hashbrown 0.16 without `default-hasher` feature does not support `HashMap::new()` — resolved by adding the feature.
- WASM (no_std) build required explicit `alloc::borrow::ToOwned` import for `&str.to_owned()`.

## Next Phase Readiness
- TextStore system ready for use by tokenization (Plan 04) and diffing modules
- StringId is Copy and can be stored in token/span structs
- Arc<TextStore> can be shared across threads for parallel diffing

---
*Phase: 01-foundation*
*Completed: 2026-02-06*
