# Project State: Redline Core v1.0

## Current Status

**Milestone:** v1.0
**Current Phase:** Phase 3 (Diff Computation) -- In Progress (3/5 plans complete)
**Last Action:** Plan 03-03 executed (Histogram diff algorithm)
**Updated:** 2026-02-07

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-02-05)

**Core value:** Extensible, correct text diff computation with a clean plugin story
**Current focus:** Phase 3 -- Diff Computation (next)

## Progress

| Phase | Status | Plans |
|-------|--------|-------|
| 1 - Foundation | ✓ Complete | 4/4 |
| 2 - Text Processing | ✓ Complete | 8/8 |
| 2.1 - Test Coverage & Simulation | ✓ Complete | 5/5 |
| 3 - Diff Computation | ◆ In Progress | 3/5 |
| 4 - Metrics Engine | Pending | 0/? |
| 5 - Analysis Framework | Pending | 0/? |
| 6 - Orchestration & Config | Pending | 0/? |
| 7 - Async Support | Pending | 0/? |
| 8 - Python Bindings | Pending | 0/? |
| 9 - WASM Target | Pending | 0/? |

Progress: ███░░░░░░░ 30%

## Artifacts

| Artifact | Status | Location |
|----------|--------|----------|
| PROJECT.md | Complete | `.planning/PROJECT.md` |
| config.json | Complete | `.planning/config.json` |
| Research (STACK) | Complete | `.planning/research/STACK.md` |
| Research (FEATURES) | Complete | `.planning/research/FEATURES.md` |
| Research (ARCHITECTURE) | Complete | `.planning/research/ARCHITECTURE.md` |
| Research (PITFALLS) | Complete | `.planning/research/PITFALLS.md` |
| Research (SUMMARY) | Complete | `.planning/research/SUMMARY.md` |
| REQUIREMENTS.md | Complete | `.planning/REQUIREMENTS.md` |
| ROADMAP.md | Complete | `.planning/ROADMAP.md` |
| Codebase Map | Complete | `.planning/codebase/` |
| Phase 1 Research | Complete | `.planning/phases/01-foundation/01-RESEARCH.md` |
| Phase 1 Plan 01 | ✓ Executed | `.planning/phases/01-foundation/01-01-PLAN.md` |
| Phase 1 Plan 02 | ✓ Executed | `.planning/phases/01-foundation/01-02-PLAN.md` |
| Phase 1 Plan 03 | ✓ Executed | `.planning/phases/01-foundation/01-03-PLAN.md` |
| Phase 1 Plan 04 | ✓ Executed | `.planning/phases/01-foundation/01-04-PLAN.md` |
| Phase 1 Verification | ✓ Passed | `.planning/phases/01-foundation/VERIFICATION.md` |
| Phase 2 Context | Complete | `.planning/phases/02-text-processing/02-CONTEXT.md` |
| Phase 2 Research | Complete | `.planning/phases/02-text-processing/02-RESEARCH.md` |
| Phase 2 Plan 01 | ✓ Executed | `.planning/phases/02-text-processing/02-01-PLAN.md` |
| Phase 2 Plan 02 | ✓ Executed | `.planning/phases/02-text-processing/02-02-PLAN.md` |
| Phase 2 Plan 03 | ✓ Executed | `.planning/phases/02-text-processing/02-03-PLAN.md` |
| Phase 2 Plan 04 | ✓ Executed | `.planning/phases/02-text-processing/02-04-PLAN.md` |
| Phase 2 Plan 05 | ✓ Executed | `.planning/phases/02-text-processing/02-05-PLAN.md` |
| Phase 2 Plan 06 | ✓ Executed | `.planning/phases/02-text-processing/02-06-PLAN.md` |
| Phase 2 Plan 07 | ✓ Executed | `.planning/phases/02-text-processing/02-07-PLAN.md` |
| Phase 2 Plan 08 | ✓ Executed | `.planning/phases/02-text-processing/02-08-PLAN.md` |
| Phase 2 Verification | ✓ Passed | `.planning/phases/02-text-processing/VERIFICATION.md` |
| Phase 2.1 Research | Complete | `.planning/phases/02.1-test-coverage-simulation/02.1-RESEARCH.md` |
| Phase 2.1 Plan 01 | ✓ Executed | `.planning/phases/02.1-test-coverage-simulation/02.1-01-PLAN.md` |
| Phase 2.1 Plan 02 | ✓ Executed | `.planning/phases/02.1-test-coverage-simulation/02.1-02-PLAN.md` |
| Phase 2.1 Plan 03 | ✓ Executed | `.planning/phases/02.1-test-coverage-simulation/02.1-03-PLAN.md` |
| Phase 2.1 Plan 04 | ✓ Executed | `.planning/phases/02.1-test-coverage-simulation/02.1-04-PLAN.md` |
| Phase 2.1 Plan 05 | ✓ Executed | `.planning/phases/02.1-test-coverage-simulation/02.1-05-PLAN.md` |
| Phase 2.1 Verification | ✓ Passed | `.planning/phases/02.1-test-coverage-simulation/02.1-VERIFICATION.md` |
| Phase 3 Context | Complete | `.planning/phases/03-diff-computation/03-CONTEXT.md` |
| Phase 3 Research | Complete | `.planning/phases/03-diff-computation/03-RESEARCH.md` |
| Phase 3 Plan 01 | ✓ Executed | `.planning/phases/03-diff-computation/03-01-PLAN.md` |
| Phase 3 Plan 02 | ✓ Executed | `.planning/phases/03-diff-computation/03-02-PLAN.md` |
| Phase 3 Plan 03 | ✓ Executed | `.planning/phases/03-diff-computation/03-03-PLAN.md` |
| Phase 3 Plan 04 | ○ Ready | `.planning/phases/03-diff-computation/03-04-PLAN.md` |
| Phase 3 Plan 05 | ○ Ready | `.planning/phases/03-diff-computation/03-05-PLAN.md` |

## Key Decisions Log

| Decision | Phase | Rationale |
|----------|-------|-----------|
| Token uses single Span, not SmallVec | Phase 1 | SmallVec doesn't implement Copy; single Span covers 99% case |
| Token is 16 bytes (not 24 max) | Phase 1 | StringId(u32) + Span(u32,u32) + TokenKind(u8) + padding = 16 bytes |
| StringId field is pub(crate) | Phase 1 | Allows crate-internal construction while hiding from external users |
| TextStore uses Vec<String> not bumpalo arena | Phase 1 | Simpler, Miri-clean; arena optimization deferred if needed |
| hashbrown needs default-hasher feature | Phase 1 | HashMap::new() requires FoldHash default hasher |
| proptest 1.6, not 1.10 (doesn't exist) | Phase 1 | Plan specified 1.10 but latest is 1.6 |
| CharMapping stores original_len | Phase 2 | Unicode case folding changes byte lengths; tokenizers need original text bounds |
| CharMapping::compose uses nearest-match | Phase 2 | Normalizers that remove chars create gaps in intermediate positions |
| SentenceTokenizer ignores CharMapping | Phase 2 | Sentence spans reference normalized positions; acceptable tradeoff |
| WordTokenizer: punctuation stays with words | Phase 2 | "hello," = 1 token, not 2; locked design decision |
| NGram tokenizers use decorator pattern | Phase 2 | Wrap Box<dyn Tokenizer>, composable with any base tokenizer |
| proptest seed infrastructure over custom sim | Phase 2.1 | proptest provides seed-based replay out of the box; no need for custom framework |
| PyO3 0.28, not 0.20 as spec says | Pre-impl | 0.20 API (GIL Refs) removed in 0.23; all spec examples must be rewritten |
| pyo3-async-runtimes replaces pyo3-asyncio | Pre-impl | pyo3-asyncio deprecated and archived |
| Python >= 3.9, not 3.8 | Pre-impl | 3.8 EOL Oct 2024; pyo3-async-runtimes requires 3.9+ |
| Custom interner, not string-interner crate | Pre-impl | External crates don't integrate with bumpalo arena |
| parking_lot::RwLock for CacheManager | Pre-impl | std::sync::RwLock has write starvation and deadlock risks |
| wasm-bindgen CLI, not wasm-pack | Pre-impl | wasm-pack archived July 2025 |

## Roadmap Evolution

- Phase 2.1 inserted after Phase 2: Test Coverage Enhancement & Deterministic Simulation Testing (URGENT)
  - Reason: Harden Phase 1+2 code before building diff computation on top
  - Status: ✓ Complete (2026-02-07)

## Context for Next Session

To continue work on this project:

1. Run `/gsd:execute-phase 3` to execute Phase 3 (Diff Computation)
2. Phase 3 has 5 plans in 4 waves — foundation, Myers, Histogram, tests+benchmarks
3. Foundation (Phase 1) and text processing (Phase 2) are solid and well-tested (462 tests)
4. After Phase 3 execution + verification, proceed to Phase 4 (Metrics Engine)

---
*State initialized: 2026-02-06*
*Last updated: 2026-02-07 after Phase 3 planning + verification*
