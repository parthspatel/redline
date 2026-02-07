# Phase 3: Diff Computation - Research

**Researched:** 2026-02-07
**Domain:** Diff algorithms (Myers O(ND), Histogram), edit script computation on token sequences
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Edit operation granularity:**
- Replace is a first-class operation -- emitted directly by the algorithm, not a post-processing merge of Delete+Insert
- Operations carry start/end token indices only; consumers map back to spans via the token list themselves
- Operations are complete and ordered -- every token is covered by exactly one operation, in sequence, no gaps
- DiffResult stores a flat `Vec<EditOperation>` as primary representation
- `.hunks(context_lines)` method returns slices into the operation list for grouped views (zero-copy)

**Algorithm selection & fallback:**
- `DiffComputer` takes anything implementing the `DiffAlgorithm` trait, with Myers as the default
- Users pass in an initialized algorithm instance (or one that impls Default)
- Default with override: `DiffComputer::new()` uses Myers, `.with_algorithm(algo)` to override
- When Myers hits the D-threshold: return partial result with an approximation flag (not an error)
- D-threshold is configurable on the algorithm instance with a sensible default: `Myers::new()` vs `Myers::with_threshold(500)`
- Each algorithm handles its own limits -- `DiffComputer` is a thin coordinator, just calls `compute()` and returns whatever it gets
- Custom user-provided algorithm implementations manage their own cutoff/fallback logic

**DiffResult shape & statistics:**
- Rich statistics computed eagerly during `compute()`: similarity ratio, operation counts (equal/delete/insert/replace), edit distance, longest common subsequence length, compression ratio, change density per region
- DiffResult holds `Arc<ProcessedText>` for both source and target -- cloneable, storable, Send+Sync
- DiffResult carries a metadata struct with: algorithm name (user-provided `String`), approximation flag, and threshold value used
- Hunks method returns slices (zero-copy views) into the operation list; hunks can't outlive the DiffResult

**Performance boundaries:**
- No input size limit -- always attempt the diff. D-threshold is the safety valve
- Common prefix/suffix optimization is configurable on the algorithm instance, on by default. Disableable for debugging/benchmarking
- `compute()` accepts an optional progress callback called periodically with estimated completion; consumer can return "cancel" from the callback
- Cancellation returns `DiffError::Cancelled` -- distinct from threshold partial result (approximation flag) and successful completion

### Claude's Discretion
- Internal representation of EditOperation enum variants (field layout, packing)
- Exact default D-threshold value for Myers
- Progress callback call frequency and estimation accuracy
- Change density region boundaries and calculation method
- Compression ratio formula
- Common prefix/suffix stripping implementation details

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Summary

Phase 3 implements the core diff computation engine: given two `ProcessedText` instances, produce a correct edit script as a sequence of `EditOperation`s (Equal, Delete, Insert, Replace). The phase delivers the `DiffAlgorithm` trait, Myers and Histogram implementations, `DiffComputer` coordinator, `EditOperation` and `DiffResult` types, and rich statistics.

The standard approach is to implement Myers O(ND) with linear-space divide-and-conquer (bidirectional meet-in-the-middle) plus common prefix/suffix stripping, and Histogram diff as a patience-like alternative that finds LCS through occurrence counting. Both operate on token index sequences derived from `ProcessedText.tokens`. A critical architectural insight is that source and target `ProcessedText` instances have **different TextStores**, so token comparison must resolve `StringId` to actual strings via each text's own store, or use a shared interning layer within `DiffComputer`.

**Primary recommendation:** Build a custom implementation of both algorithms operating on `&[u32]` slices (pre-interned token IDs) with a shared interning step in `DiffComputer`. Use index-based `EditOperation` (no lifetime parameter) per the locked decisions. Implement a Replace decorator that merges adjacent Delete+Insert into Replace as an integral part of the algorithm output pipeline.

## Standard Stack

### Core (No external diff library -- custom implementation required)

The locked decisions (Replace as first-class, index-based operations, D-threshold returning partial results, progress callbacks) mean we cannot use `similar` or `imara-diff` directly. Their APIs do not support:
- First-class Replace emission from the algorithm itself (similar's Replace is a post-processing decorator)
- Partial result return on D-threshold (similar uses a deadline/timeout, not D-threshold)
- Progress callbacks with cancellation

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| **None (custom Myers)** | N/A | Primary diff algorithm | Locked decisions require D-threshold partial results, progress callback, configurable prefix/suffix -- no existing crate exposes these controls |
| **None (custom Histogram)** | N/A | Alternative diff algorithm | Must integrate with same trait and callback system |
| criterion | 0.8.2 | Benchmarking | QUAL-03 requires Criterion benchmarks for <1ms/100 tokens, <200ms/10K tokens |
| proptest | 1.6 (already in dev-deps) | Property testing | QUAL-02 requires `apply(diff(a,b), a) == b` for 1000+ input pairs |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| proptest | 1.6 | Generating random token sequences for property tests | QUAL-02: diff correctness |
| criterion | 0.8.2 | Statistical benchmarking | QUAL-03: performance targets |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom Myers | `similar` crate (2.7) | similar lacks D-threshold partial results, progress callbacks. Would need significant wrapper code that defeats the purpose. Its DiffHook pattern is good reference material though. |
| Custom Histogram | `imara-diff` (0.2.0) | imara-diff has excellent Histogram impl but uses its own InternedInput/Token/Sink system. Integration would require adapting our Token to their Token, losing our interning. |
| Custom both | Wrapping either crate | Both crates would need extensive wrapping to meet all locked decisions. Custom implementation is cleaner and gives full control. |

**Installation:**
```toml
[dev-dependencies]
criterion = { version = "0.8", features = ["html_reports"] }
# proptest 1.6 already present

[[bench]]
name = "diff_benchmarks"
harness = false
```

## Architecture Patterns

### Recommended Project Structure
```
crates/core/src/
├── diff/
│   ├── mod.rs              # DiffAlgorithm trait, DiffComputer, re-exports
│   ├── algorithm.rs        # DiffAlgorithm trait definition
│   ├── computer.rs         # DiffComputer coordinator
│   ├── edit_operation.rs   # EditOperation struct, EditKind enum
│   ├── result.rs           # DiffResult, DiffStatistics, DiffMetadata
│   ├── error.rs            # DiffError enum
│   ├── myers.rs            # Myers O(ND) implementation
│   ├── histogram.rs        # Histogram diff implementation
│   └── common.rs           # Common prefix/suffix stripping, token comparison utilities
├── ...existing modules...
```

### Pattern 1: Token Comparison via TextStore Resolution

**What:** Tokens from different `ProcessedText` instances have different `TextStore`s. StringId(5) in source != StringId(5) in target. Must resolve through respective stores.

**When to use:** Every token comparison in diff algorithms.

**Critical insight:** Each `ProcessedText.process()` call creates a fresh `TextStoreBuilder`, so the two texts have independent string interning. Two approaches:

**Approach A (Recommended): Resolve-and-intern into a shared index space**
```rust
/// Build a shared token-to-index mapping for O(1) comparison during diff
fn build_token_indices(
    source_tokens: &[Token],
    source_store: &TextStore,
    target_tokens: &[Token],
    target_store: &TextStore,
) -> (Vec<u32>, Vec<u32>) {
    let mut intern: HashMap<&str, u32> = HashMap::new();
    let mut next_id = 0u32;

    let source_ids: Vec<u32> = source_tokens.iter().map(|t| {
        let text = source_store.resolve(t.text_id).unwrap();
        *intern.entry(text).or_insert_with(|| { let id = next_id; next_id += 1; id })
    }).collect();

    let target_ids: Vec<u32> = target_tokens.iter().map(|t| {
        let text = target_store.resolve(t.text_id).unwrap();
        *intern.entry(text).or_insert_with(|| { let id = next_id; next_id += 1; id })
    }).collect();

    (source_ids, target_ids)
}
// Now diff algorithms operate on &[u32] slices with direct == comparison
```

Approach A is recommended because it makes the inner loop of Myers/Histogram operate on `u32 == u32` comparisons (1 CPU instruction) rather than string comparisons (multiple memory accesses per comparison). This is how both `similar` and `imara-diff` work internally -- they pre-intern tokens into integer IDs.

**Confidence:** HIGH -- this is how all performant diff libraries work.

### Pattern 2: DiffAlgorithm Trait with Rich Return

**What:** The trait returns a structured result, not just a Vec of operations.

```rust
pub trait DiffAlgorithm: Send + Sync {
    /// Algorithm name for metadata
    fn name(&self) -> &str;

    /// Compute diff between source and target token sequences.
    /// `source` and `target` are interned integer IDs (same ID = same content).
    /// Returns edit operations as (source_range, target_range) index pairs.
    fn compute(
        &self,
        source: &[u32],
        target: &[u32],
        progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
    ) -> Result<AlgorithmOutput, DiffError>;
}

pub struct AlgorithmOutput {
    pub operations: Vec<EditOperation>,
    pub is_approximate: bool,
    pub threshold_used: Option<usize>,
}
```

**Confidence:** HIGH -- follows locked decisions directly.

### Pattern 3: Myers with Bidirectional Linear-Space

**What:** The linear-space variant of Myers uses divide-and-conquer with forward/backward scanning to achieve O(N+M) space instead of O((N+M)D).

**Key data structures:**
- Two V arrays: `vf` (forward) and `vb` (backward), each of size N+M+1
- Diagonal `k = x - y` for forward, `k = x - y` offset by delta for backward
- Middle snake detection: when forward x >= backward x on same diagonal, paths crossed

**D-threshold implementation:**
```rust
pub struct Myers {
    max_d: Option<usize>,  // None = unlimited
    strip_common: bool,     // prefix/suffix optimization
}

impl Myers {
    pub fn new() -> Self {
        Self { max_d: Some(4096), strip_common: true }
    }
    pub fn with_threshold(max_d: usize) -> Self {
        Self { max_d: Some(max_d), strip_common: true }
    }
    pub fn unlimited() -> Self {
        Self { max_d: None, strip_common: true }
    }
    pub fn without_common_stripping(mut self) -> Self {
        self.strip_common = false;
        self
    }
}
```

When D exceeds `max_d`, return partial result with `is_approximate = true` rather than an error.

**Confidence:** HIGH -- algorithm well-documented (Myers 1986 paper), implementations verified in similar/imara-diff.

### Pattern 4: Histogram Diff via Occurrence Counting

**What:** Histogram diff finds LCS by preferring elements with the lowest occurrence count, recursively splitting around them.

**Key data structures:**
- Occurrence count HashMap for source sequence
- Chain length limit (e.g., 64) to prevent O(N^2) on pathological inputs
- Recursive region stack (avoid deep recursion with explicit stack)
- Fallback to Myers when all elements exceed chain limit

**Algorithm steps:**
1. Count occurrences of each token in source
2. Scan target; for each token also in source with count <= chain_limit, consider as LCS candidate
3. Select candidate with lowest occurrence count as split point
4. Recursively process regions before and after the split
5. If no candidate found (all counts > chain_limit), fall back to Myers for that region

**Confidence:** HIGH -- verified from JGit implementation docs and imara-diff source.

### Pattern 5: Progress Callback with ControlFlow

**What:** Use `std::ops::ControlFlow` for the progress callback return type.

```rust
use std::ops::ControlFlow;

// In DiffComputer::compute:
pub fn compute(
    &self,
    source: Arc<ProcessedText>,
    target: Arc<ProcessedText>,
    progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
) -> Result<DiffResult, DiffError> { ... }
```

**Progress estimation for Myers:** Report progress as `d / max_d` where d is the current D iteration. Call the callback every N iterations (e.g., every 16 or 32 D-steps) to avoid overhead.

**Confidence:** HIGH -- ControlFlow is idiomatic Rust for this pattern, stable since Rust 1.55.

### Pattern 6: Index-Based EditOperation (No Lifetime)

**What:** EditOperation uses token index ranges, not borrowed slices. Solves Pitfall #4.

**Recommended struct layout (uniform shape):**
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditOperation {
    pub kind: EditKind,
    pub source_range: (u32, u32),  // (start, end) token indices
    pub target_range: (u32, u32),  // (start, end) token indices
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditKind {
    Equal,
    Delete,
    Insert,
    Replace,
}
```

This struct layout is 17 bytes (1 + 8 + 8), padded to 20 bytes. Uniform layout simplifies iteration. For Equal: source_range and target_range both populated. For Delete: target_range is empty (start == end). For Insert: source_range is empty. For Replace: both populated.

**Recommendation:** Use the struct layout (EditOperation with kind field). It's simpler to iterate, serialize, and the uniform shape avoids match arms for every consumer.

**Confidence:** HIGH -- index-based approach is the established pattern (similar's DiffOp, Pitfall #4 resolution).

### Anti-Patterns to Avoid

- **Borrowing token slices in EditOperation:** Creates lifetime parameter on DiffResult, prevents caching/HashMap storage (Pitfall #4)
- **Comparing tokens by full Token equality:** Token's PartialEq compares text_id AND span AND kind. For diffing, only text content matters. Tokens at different positions with the same text are "equal" for diff purposes.
- **Using StringId directly across different TextStores:** StringId(5) in source is NOT the same string as StringId(5) in target. Must resolve through respective stores or re-intern.
- **Storing the entire edit graph:** Myers needs only O(N+M) space with the linear-space variant. Storing the full O(N*M) edit graph wastes memory.
- **Deep recursion in Histogram:** Use an explicit stack to avoid stack overflow on deeply nested recursive splits.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Benchmarking framework | Custom timing loops | criterion 0.8.2 | Statistical rigor, regression detection, HTML reports |
| Property-based testing | Manual random test generation | proptest 1.6 (already in deps) | Automatic shrinking, reproducible failures |
| Replace merging | N/A | Similar crate's Replace decorator pattern (implement our own) | The pattern of buffering pending delete/insert and emitting Replace when adjacent is well-established |

**Key insight:** The diff algorithms themselves MUST be hand-rolled because no existing crate supports all locked decisions (D-threshold partial results, progress callbacks, first-class Replace). But the surrounding infrastructure (benchmarks, property tests) should use established tools.

## Common Pitfalls

### Pitfall 1: Cross-TextStore StringId Comparison
**What goes wrong:** Source and target ProcessedTexts have independent TextStores. StringId(0) in source may be "hello" while StringId(0) in target is "world". Comparing StringIds directly gives wrong results.
**Why it happens:** TextProcessor.process() creates a new TextStoreBuilder per call.
**How to avoid:** Pre-intern both token sequences into a shared integer space before running diff algorithms. Map Token -> u32 via text resolution + shared HashMap.
**Warning signs:** Diff produces wildly incorrect results. "hello" vs "hello" shows as a Replace.

### Pitfall 2: Myers O(N^2) Performance Cliff
**What goes wrong:** Myers is O(ND). When D approaches N (completely dissimilar texts), time degrades to O(N^2). A 10K token diff of random texts takes >10 seconds.
**Why it happens:** No D-threshold cutoff, or threshold set too high.
**How to avoid:** Default D-threshold of min(N+M, 4096). When D exceeds threshold, emit all remaining as Delete+Insert and set approximation flag.
**Warning signs:** Benchmarks hang on dissimilar input. Users report "freezing" on large diffs.

### Pitfall 3: Common Prefix/Suffix Not Adjusting Indices
**What goes wrong:** After stripping common prefix of length P and suffix of length S, the algorithm operates on a reduced slice. But EditOperations must reference indices in the ORIGINAL full token arrays.
**Why it happens:** Forgetting to add P back to all source/target indices from the sub-diff.
**How to avoid:** Emit Equal operations for the prefix and suffix explicitly, then offset all indices from the core algorithm by P.
**Warning signs:** Property test `apply(diff(a,b), a) == b` fails on inputs with common prefix/suffix.

### Pitfall 4: Replace Emission During Algorithm
**What goes wrong:** Pure Myers/Histogram emit only Equal/Delete/Insert. Replace must be detected as adjacent Delete+Insert on the same position.
**Why it happens:** The algorithms natively find insertions and deletions, not replacements.
**How to avoid:** Use a post-processing pass that buffers Delete/Insert and merges adjacent ones into Replace. Even though the decision says "emitted directly by the algorithm", the implementation achieves this by having the algorithm's output pipeline include this merge step as an integral part.
**Warning signs:** No Replace operations ever emitted. Or Replace operations have wrong ranges.

### Pitfall 5: Off-by-One in Index Ranges
**What goes wrong:** EditOperation source_range/target_range off by one, especially at boundaries between operations.
**Why it happens:** Confusion between inclusive/exclusive ranges, or errors in index tracking during forward/backward scanning.
**How to avoid:** Strict invariant: operations are contiguous and cover all tokens. Sum of all source ranges must equal source.len(). Sum of all target ranges must equal target.len(). Verify with assertion in debug builds.
**Warning signs:** Property test `apply(diff(a,b), a) == b` fails. Gaps or overlaps in operations.

### Pitfall 6: Histogram Infinite Recursion
**What goes wrong:** Histogram diff recurses on sub-regions. If the "split point" doesn't actually reduce the problem, infinite recursion occurs.
**Why it happens:** Edge case where the lowest-occurrence element is at the boundary.
**How to avoid:** Use explicit stack instead of recursion. Check that sub-regions are strictly smaller than parent. Fall back to Myers for regions that don't reduce.
**Warning signs:** Stack overflow on certain inputs.

## Code Examples

### DiffAlgorithm Trait (Recommended Design)

```rust
use std::ops::ControlFlow;

/// Error type for diff computation
#[derive(Debug, thiserror::Error)]
pub enum DiffError {
    #[error("diff computation cancelled by caller")]
    Cancelled,

    #[error("algorithm error: {0}")]
    Algorithm(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Output from a diff algorithm
pub struct AlgorithmOutput {
    pub operations: Vec<EditOperation>,
    pub is_approximate: bool,
    pub threshold_used: Option<usize>,
}

/// Trait for pluggable diff algorithms
pub trait DiffAlgorithm: Send + Sync {
    fn name(&self) -> &str;

    fn compute(
        &self,
        source: &[u32],
        target: &[u32],
        progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
    ) -> Result<AlgorithmOutput, DiffError>;
}
```

### DiffComputer (Thin Coordinator)

```rust
use std::sync::Arc;

pub struct DiffComputer {
    algorithm: Box<dyn DiffAlgorithm>,
}

impl DiffComputer {
    pub fn new() -> Self {
        Self { algorithm: Box::new(Myers::new()) }
    }

    pub fn with_algorithm(algorithm: Box<dyn DiffAlgorithm>) -> Self {
        Self { algorithm }
    }

    pub fn compute(
        &self,
        source: Arc<ProcessedText>,
        target: Arc<ProcessedText>,
        progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
    ) -> Result<DiffResult, DiffError> {
        // 1. Build shared token index space
        let (src_ids, tgt_ids) = build_token_indices(
            &source.tokens, &source.text_store,
            &target.tokens, &target.text_store,
        );

        // 2. Run algorithm
        let output = self.algorithm.compute(&src_ids, &tgt_ids, progress)?;

        // 3. Compute statistics eagerly
        let statistics = DiffStatistics::from_operations(
            &output.operations,
            source.tokens.len(),
            target.tokens.len(),
        );

        // 4. Build metadata
        let metadata = DiffMetadata {
            algorithm_name: self.algorithm.name().to_string(),
            is_approximate: output.is_approximate,
            threshold_used: output.threshold_used,
        };

        Ok(DiffResult {
            operations: output.operations,
            statistics,
            metadata,
            source,
            target,
        })
    }
}
```

### DiffResult Structure

```rust
#[derive(Debug, Clone)]
pub struct DiffResult {
    pub operations: Vec<EditOperation>,
    pub statistics: DiffStatistics,
    pub metadata: DiffMetadata,
    pub source: Arc<ProcessedText>,
    pub target: Arc<ProcessedText>,
}

#[derive(Debug, Clone)]
pub struct DiffStatistics {
    pub equal_count: usize,
    pub delete_count: usize,
    pub insert_count: usize,
    pub replace_count: usize,
    pub edit_distance: usize,
    pub similarity_ratio: f64,
    pub lcs_length: usize,
    pub compression_ratio: f64,
    pub change_density: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DiffMetadata {
    pub algorithm_name: String,
    pub is_approximate: bool,
    pub threshold_used: Option<usize>,
}

impl DiffResult {
    /// Return hunks (groups of changes with context).
    /// Returns slices into self.operations -- zero-copy.
    pub fn hunks(&self, context: usize) -> Vec<&[EditOperation]> {
        // Group operations: sequences of non-Equal ops, padded by
        // `context` Equal ops on each side
        todo!()
    }
}
```

### Property Test Pattern

```rust
use proptest::prelude::*;

fn arb_token_pair(max_len: usize) -> impl Strategy<Value = (Vec<u32>, Vec<u32>)> {
    (1..=20usize).prop_flat_map(move |vocab_size| {
        let source = prop::collection::vec(0..vocab_size as u32, 0..max_len);
        let target = prop::collection::vec(0..vocab_size as u32, 0..max_len);
        (source, target)
    })
}

proptest! {
    #[test]
    fn diff_roundtrip(
        (source, target) in arb_token_pair(200)
    ) {
        let myers = Myers::new();
        let output = myers.compute(&source, &target, None).unwrap();
        let reconstructed = apply_diff_with_target(&source, &target, &output.operations);
        prop_assert_eq!(reconstructed, target);
    }
}
```

### Criterion Benchmark Setup

```rust
// benches/diff_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_myers_100_tokens(c: &mut Criterion) {
    let (source, target) = generate_similar_tokens(100, 0.8);
    c.bench_function("myers_100_tokens", |b| {
        b.iter(|| {
            let myers = Myers::new();
            myers.compute(black_box(&source), black_box(&target), None)
        })
    });
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_scaling");
    for size in [10, 50, 100, 500, 1000, 5000, 10000] {
        let (source, target) = generate_similar_tokens(size, 0.9);
        group.bench_with_input(BenchmarkId::new("myers", size), &size, |b, _| {
            b.iter(|| {
                let myers = Myers::new();
                myers.compute(black_box(&source), black_box(&target), None)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_myers_100_tokens, bench_scaling);
criterion_main!(benches);
```

### Common Prefix/Suffix Stripping

```rust
fn strip_common_affixes<'a>(
    source: &'a [u32],
    target: &'a [u32],
) -> (usize, usize, &'a [u32], &'a [u32]) {
    let prefix_len = source.iter()
        .zip(target.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let source_rest = &source[prefix_len..];
    let target_rest = &target[prefix_len..];

    let suffix_len = source_rest.iter().rev()
        .zip(target_rest.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();

    let inner_source = &source_rest[..source_rest.len() - suffix_len];
    let inner_target = &target_rest[..target_rest.len() - suffix_len];

    (prefix_len, suffix_len, inner_source, inner_target)
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| EditOperation borrows &[Token] | EditOperation uses index ranges (u32) | Pitfall #4 resolution | DiffResult is 'static, cacheable, HashMap-storable |
| Single Myers without fallback | Myers with D-threshold + approximation flag | Standard practice in Git, imara-diff | Prevents O(N^2) hang on dissimilar inputs |
| Line-only diff | Token-level diff with pluggable tokenizer | Redline's differentiator | Enables word-level, sentence-level, char-level diff |
| similar 2.x (deadline-based cutoff) | imara-diff 0.2 (heuristic-based) | 2023-2024 | imara-diff consistently faster, better heuristics |
| Patience diff | Histogram diff | Git transition | Histogram is patience + fallback for non-unique elements |

**Deprecated/outdated:**
- `DiffResult<'a>` with lifetime parameter (spec's original design) -- replaced by index-based approach per locked decisions
- PyO3 0.20's diff bindings pattern -- PyO3 0.24+ uses Bound API

## Open Questions

1. **Default D-threshold value**
   - What we know: imara-diff uses heuristics (not a fixed threshold); similar uses a time deadline; Git's Myers uses heuristics after certain D
   - What's unclear: The optimal fixed default. sqrt(N+M) * 4? Fixed 1024? Proportion of N?
   - Recommendation: Default to `min(N + M, 4096)` as the D-threshold. This is large enough for most real diffs but prevents pathological hangs. Users can override with `Myers::with_threshold()`.

2. **Histogram chain length limit**
   - What we know: JGit uses 64 as the limit. imara-diff uses 64 and falls back to Myers with heuristics.
   - What's unclear: Whether 64 is optimal for token-level (vs line-level) diffs
   - Recommendation: Use 64, matching established implementations. Make it configurable on the Histogram struct.

3. **Change density calculation**
   - What we know: The decision says "change density per region" but doesn't define regions
   - What's unclear: How to partition tokens into regions. Fixed-size windows? Sentence boundaries?
   - Recommendation: Use fixed-size windows (e.g., 10% of total tokens each = 10 regions). Each region's density = (non-equal ops in region) / (total ops in region).

4. **Should DiffComputer accept `&ProcessedText` or `Arc<ProcessedText>`?**
   - What we know: DiffResult holds `Arc<ProcessedText>`. DiffComputer needs to pass ownership to DiffResult.
   - What's unclear: Whether DiffComputer.compute() should accept Arc (caller wraps) or &ProcessedText (DiffComputer can't store it)
   - Recommendation: Accept `Arc<ProcessedText>` in compute(). Caller wraps in Arc before calling. This is consistent with the locked decision that DiffResult holds Arc.

## Sources

### Primary (HIGH confidence)
- Myers 1986 paper "An O(ND) Difference Algorithm and Its Variations" -- algorithm description
- imara-diff docs.rs -- Algorithm enum, InternedInput pattern, Histogram/Myers implementation notes
- similar docs.rs -- DiffHook trait, DiffOp enum, Replace decorator pattern
- Redline codebase: Token (Copy, 16 bytes), ProcessedText (independent TextStore per instance), existing proptest patterns

### Secondary (MEDIUM confidence)
- Neil Fraser: Diff Strategies -- common prefix/suffix optimization, two-edit detection
- JGit HistogramDiff -- Histogram algorithm specification
- Myers diff in linear space (blog.jcoglan.com) -- bidirectional meet-in-middle
- ControlFlow in std::ops -- progress callback pattern
- criterion 0.8.2 docs.rs -- benchmark setup

### Tertiary (LOW confidence)
- Default D-threshold value recommendations -- based on general practice, not empirically validated for token-level diffs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- verified against codebase constraints, locked decisions eliminate external crate options
- Architecture: HIGH -- patterns derived from similar/imara-diff source and locked decisions
- Token comparison (cross-TextStore): HIGH -- verified by reading TextProcessor.process() source
- Myers algorithm: HIGH -- well-documented with multiple verified implementations
- Histogram algorithm: HIGH -- documented in JGit, imara-diff
- Pitfalls: HIGH -- cross-TextStore comparison verified in codebase, O(N^2) cliff in project research
- Default threshold values: LOW -- no empirical data for token-level diffs specifically

**Research date:** 2026-02-07
**Valid until:** 2026-03-07 (algorithms are stable; crate versions may update)
