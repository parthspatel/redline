---
phase: 02-text-processing
plan: 01
subsystem: core-types
tags: [traits, normalizer, tokenizer, process, errors]

requires:
  - phase: 01-foundation
    provides: "TextStore, Token, Span, CharMapping, error types"
provides:
  - "Normalizer trait (Send + Sync)"
  - "Tokenizer trait (Send + Sync)"
  - "ProcessedText with layer access and mapping composition"
  - "NormalizationLayer and ExecutionMode"
  - "ProcessError integrated into RedlineError"
  - "NormalizeError::EmptyInput, InvalidMapping"
  - "TokenizeError::EmptyInput, Failed"
affects: [02-02, 02-03, 02-04, 02-05, 02-06, 02-07, 02-08]

tech-stack:
  added: [unicode-normalization 0.1, unicode-segmentation 1.12, test-case 3.3, insta 1.46]
  patterns: [trait-based-normalization, trait-based-tokenization, pipeline-orchestration]

key-files:
  created:
    - crates/core/src/normalize/mod.rs
    - crates/core/src/tokenize/mod.rs
    - crates/core/src/process/mod.rs
    - crates/core/src/process/processed_text.rs
    - crates/core/src/process/layer.rs
  modified:
    - crates/core/Cargo.toml
    - crates/core/src/lib.rs
    - crates/core/src/error.rs

key-decisions:
  - "Manual Debug impl for ProcessedText (TextStore lacks Debug derive)"
  - "alloc imports in process/mod.rs for no_std Box/Vec/ToString"

patterns-established:
  - "Normalizer trait: normalize(&str) -> Result<NormalizationResult, NormalizeError>"
  - "Tokenizer trait: tokenize(&str, &CharMapping, &mut TextStoreBuilder) -> Result<Vec<Token>, TokenizeError>"

duration: 3min
completed: 2026-02-06
---

# Plan 02-01: Module Setup and Trait Definitions Summary

**Normalizer/Tokenizer traits with Send+Sync bounds, ProcessedText pipeline type, ProcessError enum integrated into RedlineError**

## Performance

- **Tasks:** 3/3 complete
- **Files created:** 5
- **Files modified:** 3

## Accomplishments
- Normalizer trait with normalize(), name(), cost() and Send + Sync bounds
- Tokenizer trait with tokenize(), name() and Send + Sync bounds
- ProcessedText struct with layer() and mapping_to_layer() accessors
- NormalizationLayer + ExecutionMode (defaults to Minimal)
- ProcessError enum with 4 variants integrated into RedlineError
- Extended NormalizeError (+EmptyInput, +InvalidMapping) and TokenizeError (+EmptyInput, +Failed)

## Task Commits

1. **Task 1: Add dependencies and module declarations** - `dcba9d9` (chore)
2. **Task 2: Create trait definitions and type structures** - `4ab8cac` (feat)
3. **Task 3: Extend error types with new variants** - `7f51691` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Manual Debug impl for ProcessedText**
- **Found during:** Task 2
- **Issue:** TextStore does not implement Debug, preventing derive(Debug) on ProcessedText
- **Fix:** Implemented fmt::Debug manually with summary representation for TextStore field
- **Files modified:** crates/core/src/process/processed_text.rs

**2. [Rule 3 - Blocking] alloc imports for process/mod.rs**
- **Found during:** Task 2
- **Issue:** TextProcessor uses Box, Vec, ToString needing explicit alloc imports in no_std
- **Fix:** Added `#[cfg(not(feature = "std"))] use alloc::{boxed::Box, string::ToString, vec::Vec};`
- **Files modified:** crates/core/src/process/mod.rs

**Total deviations:** 2 auto-fixed (both blocking)
**Impact on plan:** Both fixes required for compilation. No scope creep.

## Issues Encountered
None

## Next Phase Readiness
Plans 02-02 through 02-05 (Wave 2) can now write failing tests against:
- Normalizer trait for concrete normalizer implementations
- Tokenizer trait for concrete tokenizer implementations
- New error variants for error case tests

No blockers for subsequent plans.

---
*Plan: 02-01*
*Completed: 2026-02-06*

## Self-Check: PASSED
