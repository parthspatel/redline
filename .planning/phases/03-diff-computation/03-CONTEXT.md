# Phase 3: Diff Computation - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Given two ProcessedTexts, compute a correct, performant edit script with configurable algorithm selection. Delivers DiffAlgorithm trait, Myers and Histogram implementations, DiffComputer coordinator, EditOperation types, and DiffResult with statistics. Metrics engine (Phase 4) and analysis framework (Phase 5) are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Edit operation granularity
- Replace is a first-class operation — emitted directly by the algorithm, not a post-processing merge of Delete+Insert
- Operations carry start/end token indices only; consumers map back to spans via the token list themselves
- Operations are complete and ordered — every token is covered by exactly one operation, in sequence, no gaps
- DiffResult stores a flat `Vec<EditOperation>` as primary representation
- `.hunks(context_lines)` method returns slices into the operation list for grouped views (zero-copy)

### Algorithm selection & fallback
- `DiffComputer` takes anything implementing the `DiffAlgorithm` trait, with Myers as the default
- Users pass in an initialized algorithm instance (or one that impls Default)
- Default with override: `DiffComputer::new()` uses Myers, `.with_algorithm(algo)` to override
- When Myers hits the D-threshold: return partial result with an approximation flag (not an error)
- D-threshold is configurable on the algorithm instance with a sensible default: `Myers::new()` vs `Myers::with_threshold(500)`
- Each algorithm handles its own limits — `DiffComputer` is a thin coordinator, just calls `compute()` and returns whatever it gets
- Custom user-provided algorithm implementations manage their own cutoff/fallback logic

### DiffResult shape & statistics
- Rich statistics computed eagerly during `compute()`: similarity ratio, operation counts (equal/delete/insert/replace), edit distance, longest common subsequence length, compression ratio, change density per region
- DiffResult holds `Arc<ProcessedText>` for both source and target — cloneable, storable, Send+Sync
- DiffResult carries a metadata struct with: algorithm name (user-provided `String`), approximation flag, and threshold value used
- Hunks method returns slices (zero-copy views) into the operation list; hunks can't outlive the DiffResult

### Performance boundaries
- No input size limit — always attempt the diff. D-threshold is the safety valve
- Common prefix/suffix optimization is configurable on the algorithm instance, on by default. Disableable for debugging/benchmarking
- `compute()` accepts an optional progress callback called periodically with estimated completion; consumer can return "cancel" from the callback
- Cancellation returns `DiffError::Cancelled` — distinct from threshold partial result (approximation flag) and successful completion

### Claude's Discretion
- Internal representation of EditOperation enum variants (field layout, packing)
- Exact default D-threshold value for Myers
- Progress callback call frequency and estimation accuracy
- Change density region boundaries and calculation method
- Compression ratio formula
- Common prefix/suffix stripping implementation details

</decisions>

<specifics>
## Specific Ideas

- DiffResult pattern mirrors Phase 1's two-phase Arc pattern — `Arc<ProcessedText>` keeps the same ownership story
- Algorithm name is a user-provided String on the trait, not a hardcoded enum — future algorithms don't need enum variants
- Three distinct outcomes are clearly distinguishable: success (complete result), threshold (partial result + approximation flag), cancellation (`DiffError::Cancelled`)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-diff-computation*
*Context gathered: 2026-02-07*
