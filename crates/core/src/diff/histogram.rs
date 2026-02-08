//! Histogram diff algorithm with configurable chain limit and Myers fallback.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::ops::ControlFlow;

use hashbrown::HashMap;

use super::algorithm::{AlgorithmOutput, DiffAlgorithm};
use super::common::{coalesce, fuse_replaces, strip_common_affixes};
use super::edit_operation::EditOperation;
use super::error::DiffError;
use super::myers::Myers;

/// Histogram diff algorithm.
///
/// Finds the lowest-occurrence shared token between source and target as an LCS
/// split point, then iteratively (explicit stack) processes the before/after regions.
/// Falls back to Myers for degenerate regions where all shared tokens exceed
/// the chain limit.
///
/// Produces better results than Myers on structured/repetitive text where unique
/// tokens (function names, headings) provide natural split points.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Maximum occurrence count for LCS candidates. Tokens appearing more than
    /// this many times in the source are skipped. Default: 64.
    chain_limit: usize,
    /// Whether to strip common prefix/suffix before running the algorithm.
    /// Default: `true`.
    use_common_affixes: bool,
    /// D-threshold for internal Myers fallback. Default: `Some(4096)`.
    myers_threshold: Option<usize>,
}

impl Histogram {
    /// Create a new Histogram instance with default settings.
    ///
    /// - chain_limit: 64
    /// - use_common_affixes: true
    /// - myers_threshold: Some(4096)
    #[inline]
    pub fn new() -> Self {
        Self {
            chain_limit: 64,
            use_common_affixes: true,
            myers_threshold: Some(4096),
        }
    }

    /// Create a Histogram instance with a specific chain limit.
    #[inline]
    pub fn with_chain_limit(chain_limit: usize) -> Self {
        Self {
            chain_limit,
            use_common_affixes: true,
            myers_threshold: Some(4096),
        }
    }

    /// Disable common prefix/suffix optimization (builder pattern).
    #[inline]
    pub fn without_common_affixes(mut self) -> Self {
        self.use_common_affixes = false;
        self
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// A sub-problem region for the iterative stack.
#[derive(Debug, Clone, Copy)]
struct Region {
    src_start: usize,
    src_end: usize,
    tgt_start: usize,
    tgt_end: usize,
}

/// Count occurrences of each token value in a source slice.
fn count_occurrences(source: &[u32]) -> HashMap<u32, usize> {
    let mut counts: HashMap<u32, usize> = HashMap::with_capacity(source.len());
    for &token in source {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

/// Find the best split point: the shared token with the lowest occurrence count
/// in the source, appearing in both source and target within the given region.
///
/// Returns `Some((source_idx, target_idx))` in absolute coordinates,
/// or `None` if no candidate is found within the chain limit.
///
/// When multiple candidates tie on occurrence count, prefers the one closest
/// to the middle of the target (better balanced splits).
fn find_best_split(
    source: &[u32],
    target: &[u32],
    counts: &HashMap<u32, usize>,
    chain_limit: usize,
) -> Option<(usize, usize)> {
    let target_mid = target.len() / 2;
    let mut best: Option<(usize, usize, usize)> = None; // (src_idx, tgt_idx, count)

    for (tgt_idx, &token) in target.iter().enumerate() {
        if let Some(&count) = counts.get(&token) {
            if count > chain_limit {
                continue;
            }
            let is_better = match best {
                None => true,
                Some((_, _, best_count)) => {
                    if count < best_count {
                        true
                    } else if count == best_count {
                        // Tie-break: prefer closer to middle of target
                        let dist = if tgt_idx >= target_mid {
                            tgt_idx - target_mid
                        } else {
                            target_mid - tgt_idx
                        };
                        let best_tgt = best.unwrap().1;
                        let best_dist = if best_tgt >= target_mid {
                            best_tgt - target_mid
                        } else {
                            target_mid - best_tgt
                        };
                        dist < best_dist
                    } else {
                        false
                    }
                }
            };

            if is_better {
                // Find the first occurrence in source
                if let Some(src_idx) = source.iter().position(|&s| s == token) {
                    best = Some((src_idx, tgt_idx, count));
                }
            }
        }
    }

    best.map(|(s, t, _)| (s, t))
}

/// Run Myers on a sub-region, returning operations in local (0-based) coordinates.
fn myers_fallback_region(
    source: &[u32],
    target: &[u32],
    threshold: Option<usize>,
) -> Result<(Vec<EditOperation>, bool), DiffError> {
    let myers = match threshold {
        Some(t) => Myers::with_threshold(t),
        None => Myers::without_threshold(),
    };
    let output = myers.compute(source, target, None)?;
    Ok((output.operations, output.is_approximate))
}

/// Adjust all source and target indices in operations by the given offsets.
fn offset_operations(ops: &mut [EditOperation], source_offset: u32, target_offset: u32) {
    for op in ops.iter_mut() {
        op.source_start += source_offset;
        op.source_end += source_offset;
        op.target_start += target_offset;
        op.target_end += target_offset;
    }
}

impl DiffAlgorithm for Histogram {
    fn name(&self) -> &str {
        "histogram"
    }

    fn compute(
        &self,
        source: &[u32],
        target: &[u32],
        mut progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
    ) -> Result<AlgorithmOutput, DiffError> {
        let n = source.len();
        let m = target.len();

        // Step 1: Strip common affixes
        let (prefix_len, suffix_len, inner_source, inner_target) = if self.use_common_affixes {
            strip_common_affixes(source, target)
        } else {
            (0, 0, source, target)
        };

        let mut result_ops: Vec<EditOperation> = Vec::new();
        let mut is_approximate = false;

        // Emit Equal for common prefix
        if prefix_len > 0 {
            result_ops.push(EditOperation::equal(
                0,
                prefix_len as u32,
                0,
                prefix_len as u32,
            ));
        }

        // If both inner slices are empty, just prefix + suffix
        if inner_source.is_empty() && inner_target.is_empty() {
            if suffix_len > 0 {
                result_ops.push(EditOperation::equal(
                    (n - suffix_len) as u32,
                    n as u32,
                    (m - suffix_len) as u32,
                    m as u32,
                ));
            }
            return Ok(AlgorithmOutput {
                operations: result_ops,
                is_approximate: false,
                threshold_used: None,
            });
        }

        // Handle edge cases where one side is empty
        if inner_source.is_empty() {
            result_ops.push(EditOperation::insert(
                prefix_len as u32,
                prefix_len as u32,
                (prefix_len + inner_target.len()) as u32,
            ));
            if suffix_len > 0 {
                result_ops.push(EditOperation::equal(
                    (n - suffix_len) as u32,
                    n as u32,
                    (m - suffix_len) as u32,
                    m as u32,
                ));
            }
            return Ok(AlgorithmOutput {
                operations: result_ops,
                is_approximate: false,
                threshold_used: None,
            });
        }

        if inner_target.is_empty() {
            result_ops.push(EditOperation::delete(
                prefix_len as u32,
                (prefix_len + inner_source.len()) as u32,
                prefix_len as u32,
            ));
            if suffix_len > 0 {
                result_ops.push(EditOperation::equal(
                    (n - suffix_len) as u32,
                    n as u32,
                    (m - suffix_len) as u32,
                    m as u32,
                ));
            }
            return Ok(AlgorithmOutput {
                operations: result_ops,
                is_approximate: false,
                threshold_used: None,
            });
        }

        // Step 2: Iterative histogram algorithm with explicit stack
        let mut stack: Vec<Region> = Vec::new();
        let mut inner_ops: Vec<EditOperation> = Vec::new();

        let in_n = inner_source.len();
        let in_m = inner_target.len();

        stack.push(Region {
            src_start: 0,
            src_end: in_n,
            tgt_start: 0,
            tgt_end: in_m,
        });

        let mut regions_processed: usize = 0;
        let mut estimated_total: usize = 1;

        while let Some(region) = stack.pop() {
            regions_processed += 1;

            // Progress callback every 32 region pops
            if regions_processed % 32 == 0 {
                if let Some(cb) = progress.as_mut() {
                    let ratio = (regions_processed as f64 / estimated_total as f64).min(0.99);
                    if let ControlFlow::Break(()) = cb(ratio) {
                        return Err(DiffError::Cancelled);
                    }
                }
            }

            let src_slice = &inner_source[region.src_start..region.src_end];
            let tgt_slice = &inner_target[region.tgt_start..region.tgt_end];

            // Empty region
            if src_slice.is_empty() && tgt_slice.is_empty() {
                continue;
            }

            // Only source is non-empty -> Delete
            if tgt_slice.is_empty() {
                inner_ops.push(EditOperation::delete(
                    region.src_start as u32,
                    region.src_end as u32,
                    region.tgt_start as u32,
                ));
                continue;
            }

            // Only target is non-empty -> Insert
            if src_slice.is_empty() {
                inner_ops.push(EditOperation::insert(
                    region.src_start as u32,
                    region.tgt_start as u32,
                    region.tgt_end as u32,
                ));
                continue;
            }

            // Count occurrences and find best split
            let counts = count_occurrences(src_slice);
            let split = find_best_split(src_slice, tgt_slice, &counts, self.chain_limit);

            match split {
                Some((local_src, local_tgt)) => {
                    let abs_src = region.src_start + local_src;
                    let abs_tgt = region.tgt_start + local_tgt;

                    // Emit Equal for the split token
                    inner_ops.push(EditOperation::equal(
                        abs_src as u32,
                        (abs_src + 1) as u32,
                        abs_tgt as u32,
                        (abs_tgt + 1) as u32,
                    ));

                    // Push AFTER region first (LIFO -> processed second)
                    if abs_src + 1 < region.src_end || abs_tgt + 1 < region.tgt_end {
                        stack.push(Region {
                            src_start: abs_src + 1,
                            src_end: region.src_end,
                            tgt_start: abs_tgt + 1,
                            tgt_end: region.tgt_end,
                        });
                        estimated_total += 1;
                    }

                    // Push BEFORE region second (LIFO -> processed first)
                    if region.src_start < abs_src || region.tgt_start < abs_tgt {
                        stack.push(Region {
                            src_start: region.src_start,
                            src_end: abs_src,
                            tgt_start: region.tgt_start,
                            tgt_end: abs_tgt,
                        });
                        estimated_total += 1;
                    }
                }
                None => {
                    // No candidate within chain limit -> Myers fallback
                    let (mut fallback_ops, approx) =
                        myers_fallback_region(src_slice, tgt_slice, self.myers_threshold)?;

                    if approx {
                        is_approximate = true;
                    }

                    // Offset to absolute inner coordinates
                    offset_operations(
                        &mut fallback_ops,
                        region.src_start as u32,
                        region.tgt_start as u32,
                    );

                    inner_ops.extend(fallback_ops);
                }
            }
        }

        // Step 3: Sort operations by position
        inner_ops.sort_by(|a, b| {
            a.source_start
                .cmp(&b.source_start)
                .then(a.target_start.cmp(&b.target_start))
        });

        // Step 4: Coalesce consecutive same-kind operations
        let inner_ops = coalesce(inner_ops);

        // Step 5: Fuse adjacent Delete+Insert into Replace
        let mut inner_ops = fuse_replaces(inner_ops);

        // Step 6: Offset by prefix_len
        offset_operations(&mut inner_ops, prefix_len as u32, prefix_len as u32);

        // Step 7: Assemble final result
        result_ops.extend(inner_ops);

        if suffix_len > 0 {
            result_ops.push(EditOperation::equal(
                (n - suffix_len) as u32,
                n as u32,
                (m - suffix_len) as u32,
                m as u32,
            ));
        }

        Ok(AlgorithmOutput {
            operations: result_ops,
            is_approximate,
            threshold_used: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::common::apply_operations;

    /// Helper to compute and return operations.
    fn compute_ops(source: &[u32], target: &[u32], hist: &Histogram) -> AlgorithmOutput {
        hist.compute(source, target, None).unwrap()
    }

    /// Verify roundtrip: apply_operations(source, target, ops) == target.
    fn assert_roundtrip(source: &[u32], target: &[u32], ops: &[EditOperation]) {
        let reconstructed = apply_operations(source, target, ops);
        assert_eq!(
            reconstructed, target,
            "Roundtrip failed: source={:?}, target={:?}, ops={:?}",
            source, target, ops
        );
    }

    // ---- Constructor tests ----

    #[test]
    fn new_defaults() {
        let h = Histogram::new();
        assert_eq!(h.chain_limit, 64);
        assert!(h.use_common_affixes);
        assert_eq!(h.myers_threshold, Some(4096));
    }

    #[test]
    fn with_chain_limit_constructor() {
        let h = Histogram::with_chain_limit(10);
        assert_eq!(h.chain_limit, 10);
        assert!(h.use_common_affixes);
    }

    #[test]
    fn without_common_affixes_builder() {
        let h = Histogram::new().without_common_affixes();
        assert!(!h.use_common_affixes);
        assert_eq!(h.chain_limit, 64);
    }

    #[test]
    fn default_same_as_new() {
        let d = Histogram::default();
        let n = Histogram::new();
        assert_eq!(d.chain_limit, n.chain_limit);
        assert_eq!(d.use_common_affixes, n.use_common_affixes);
        assert_eq!(d.myers_threshold, n.myers_threshold);
    }

    #[test]
    fn name_is_histogram() {
        assert_eq!(Histogram::new().name(), "histogram");
    }

    // ---- Core algorithm tests ----

    #[test]
    fn empty_vs_empty() {
        let h = Histogram::new();
        let output = compute_ops(&[], &[], &h);
        assert!(output.operations.is_empty());
        assert!(!output.is_approximate);
        assert_roundtrip(&[], &[], &output.operations);
    }

    #[test]
    fn single_equal() {
        let h = Histogram::new();
        let output = compute_ops(&[0], &[0], &h);
        assert_eq!(output.operations, vec![EditOperation::equal(0, 1, 0, 1)]);
        assert!(!output.is_approximate);
        assert_roundtrip(&[0], &[0], &output.operations);
    }

    #[test]
    fn single_replace() {
        let h = Histogram::new();
        let output = compute_ops(&[0], &[1], &h);
        // No shared tokens -> Myers fallback -> Replace
        assert_roundtrip(&[0], &[1], &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn identical_three_elements() {
        let h = Histogram::new();
        let source = [0, 1, 2];
        let target = [0, 1, 2];
        let output = compute_ops(&source, &target, &h);
        assert_eq!(output.operations, vec![EditOperation::equal(0, 3, 0, 3)]);
        assert!(!output.is_approximate);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn middle_replace() {
        let h = Histogram::new();
        let source = [0, 1, 2];
        let target = [0, 3, 2];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
        // Should have Equal(0..1), Replace(1..2, 1..2), Equal(2..3)
        assert_eq!(
            output.operations,
            vec![
                EditOperation::equal(0, 1, 0, 1),
                EditOperation::replace(1, 2, 1, 2),
                EditOperation::equal(2, 3, 2, 3),
            ]
        );
    }

    #[test]
    fn append_elements() {
        let h = Histogram::new();
        let source = [0, 1, 2];
        let target = [0, 1, 2, 3, 4];
        let output = compute_ops(&source, &target, &h);
        assert_eq!(
            output.operations,
            vec![
                EditOperation::equal(0, 3, 0, 3),
                EditOperation::insert(3, 3, 5),
            ]
        );
        assert!(!output.is_approximate);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn remove_trailing_elements() {
        let h = Histogram::new();
        let source = [0, 1, 2, 3, 4];
        let target = [0, 1, 2];
        let output = compute_ops(&source, &target, &h);
        assert_eq!(
            output.operations,
            vec![
                EditOperation::equal(0, 3, 0, 3),
                EditOperation::delete(3, 5, 3),
            ]
        );
        assert!(!output.is_approximate);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn cat_sat_vs_dog_sat_on_the_mat() {
        // "the cat sat" [0,1,2] vs "the dog sat on the mat" [0,3,2,4,0,5]
        let h = Histogram::new();
        let source = [0, 1, 2];
        let target = [0, 3, 2, 4, 0, 5];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn repetitive_text_unique_split() {
        // Unique elements 1 and 2 are best split candidates (occurrence=1)
        let h = Histogram::new();
        let source = [0, 0, 0, 1, 0, 0, 0];
        let target = [0, 0, 0, 2, 0, 0, 0];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
        // Should produce Equal(0..3), Replace(3..4), Equal(4..7)
        assert_eq!(
            output.operations,
            vec![
                EditOperation::equal(0, 3, 0, 3),
                EditOperation::replace(3, 4, 3, 4),
                EditOperation::equal(4, 7, 4, 7),
            ]
        );
    }

    #[test]
    fn chain_limit_exceeded_falls_back_to_myers() {
        // chain_limit=1, but token 0 appears 3 times -> all exceed limit -> Myers fallback
        let h = Histogram::with_chain_limit(1);
        let source = [0, 0, 0];
        let target = [0, 0, 1];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn cancellation_via_callback() {
        let h = Histogram::new();
        // Large enough to trigger callback (needs >32 region pops)
        // With chain_limit=1 and unique tokens, each split creates 2 regions
        let source: Vec<u32> = (0..100).collect();
        let target: Vec<u32> = (0..100).collect();

        // Cancel immediately on first callback
        let result = h.compute(&source, &target, Some(&mut |_: f64| ControlFlow::Break(())));

        // May or may not trigger callback depending on affixes; test with different input
        let source2: Vec<u32> = (0..100).collect();
        let mut target2: Vec<u32> = (0..100).collect();
        for t in target2.iter_mut() {
            *t += 200; // all different
        }

        let h2 = Histogram::with_chain_limit(1000);
        let result2 = h2.compute(
            &source2,
            &target2,
            Some(&mut |_: f64| ControlFlow::Break(())),
        );

        // At least one of these should cancel or produce a valid result
        match (result, result2) {
            (Err(DiffError::Cancelled), _) | (_, Err(DiffError::Cancelled)) => {} // expected
            (Ok(out1), Ok(out2)) => {
                // If neither cancelled (small inputs processed before 32 pops), verify roundtrip
                assert_roundtrip(&source, &target, &out1.operations);
                assert_roundtrip(&source2, &target2, &out2.operations);
            }
            (Err(e), _) | (_, Err(e)) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn empty_source_insert_all() {
        let h = Histogram::new();
        let source: [u32; 0] = [];
        let target = [0, 1, 2];
        let output = compute_ops(&source, &target, &h);
        assert_eq!(output.operations, vec![EditOperation::insert(0, 0, 3)]);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn empty_target_delete_all() {
        let h = Histogram::new();
        let source = [0, 1, 2];
        let target: [u32; 0] = [];
        let output = compute_ops(&source, &target, &h);
        assert_eq!(output.operations, vec![EditOperation::delete(0, 3, 0)]);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn single_insert_at_beginning() {
        let h = Histogram::new();
        let source = [1, 2, 3];
        let target = [0, 1, 2, 3];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn single_delete_at_beginning() {
        let h = Histogram::new();
        let source = [0, 1, 2, 3];
        let target = [1, 2, 3];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn multiple_edits() {
        let h = Histogram::new();
        let source = [0, 1, 2, 3, 4, 5];
        let target = [0, 9, 2, 3, 8, 5];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn completely_different_two_elements() {
        let h = Histogram::new();
        let source = [0, 1];
        let target = [2, 3];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn common_affixes_disabled_same_roundtrip() {
        let source = [0, 1, 2, 3, 2];
        let target = [0, 3, 2, 4, 2];

        let with_affixes = Histogram::new();
        let without_affixes = Histogram::new().without_common_affixes();

        let output_with = compute_ops(&source, &target, &with_affixes);
        let output_without = compute_ops(&source, &target, &without_affixes);

        assert_roundtrip(&source, &target, &output_with.operations);
        assert_roundtrip(&source, &target, &output_without.operations);

        assert!(!output_with.is_approximate);
        assert!(!output_without.is_approximate);
    }

    #[test]
    fn longer_sequence_roundtrip() {
        let h = Histogram::new();
        let source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let target = [0, 1, 20, 21, 4, 5, 6, 30, 8, 9];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn prepend_and_append() {
        let h = Histogram::new();
        let source = [1, 2, 3];
        let target = [0, 1, 2, 3, 4];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn interleaved_differences() {
        let h = Histogram::new();
        let source = [0, 1, 0, 1, 0];
        let target = [1, 0, 1, 0, 1];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn histogram_vs_myers_roundtrip_equivalence() {
        // Both algorithms should produce valid roundtrips for any input
        let cases: Vec<(Vec<u32>, Vec<u32>)> = vec![
            (vec![], vec![]),
            (vec![0], vec![0]),
            (vec![0], vec![1]),
            (vec![0, 1, 2], vec![0, 1, 2]),
            (vec![0, 1, 2], vec![0, 3, 2]),
            (vec![0, 1, 2], vec![0, 1, 2, 3, 4]),
            (vec![0, 1, 2, 3, 4], vec![0, 1, 2]),
            (vec![0, 1, 2], vec![0, 3, 2, 4, 0, 5]),
            (vec![0, 0, 0, 1, 0, 0, 0], vec![0, 0, 0, 2, 0, 0, 0]),
        ];

        let h = Histogram::new();
        let m = Myers::new();

        for (source, target) in &cases {
            let h_out = h.compute(source, target, None).unwrap();
            let m_out = m.compute(source, target, None).unwrap();
            assert_roundtrip(source, target, &h_out.operations);
            assert_roundtrip(source, target, &m_out.operations);
        }
    }

    #[test]
    fn send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Histogram>();
    }

    #[test]
    fn debug_impl() {
        let h = Histogram::new();
        let debug = format!("{:?}", h);
        assert!(debug.contains("Histogram"));
    }

    #[test]
    fn clone_impl() {
        let h = Histogram::with_chain_limit(10).without_common_affixes();
        let h2 = h.clone();
        assert_eq!(h2.chain_limit, 10);
        assert!(!h2.use_common_affixes);
    }

    #[test]
    fn all_tokens_same() {
        // All tokens are identical — histogram can't find unique split, falls back to Myers
        let h = Histogram::new();
        let source = [0, 0, 0, 0];
        let target = [0, 0, 0, 0, 0];
        let output = compute_ops(&source, &target, &h);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn without_affixes_produces_valid_roundtrip() {
        let h = Histogram::new().without_common_affixes();

        let cases: Vec<(Vec<u32>, Vec<u32>)> = vec![
            (vec![], vec![]),
            (vec![0], vec![0]),
            (vec![0], vec![1]),
            (vec![0, 1, 2], vec![0, 1, 2]),
            (vec![0, 1, 2], vec![0, 3, 2]),
            (vec![0, 1, 2], vec![0, 1, 2, 3, 4]),
            (vec![0, 1, 2, 3, 4], vec![0, 1, 2]),
            (vec![0, 1, 2], vec![0, 3, 2, 4, 0, 5]),
        ];

        for (source, target) in &cases {
            let output = h.compute(source, target, None).unwrap();
            assert_roundtrip(source, target, &output.operations);
        }
    }
}
