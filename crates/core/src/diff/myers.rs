//! Myers O(ND) diff algorithm with D-threshold cutoff and common affix optimization.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::ops::ControlFlow;

use super::algorithm::{AlgorithmOutput, DiffAlgorithm};
use super::common::strip_common_affixes;
use super::edit_operation::EditOperation;
use super::error::DiffError;

/// Myers O(ND) diff algorithm.
///
/// Implements the classic forward Myers algorithm with trace-based backtrace,
/// optional D-threshold cutoff for approximate results on very different inputs,
/// and common prefix/suffix optimization to reduce the search space.
///
/// Replace operations are emitted directly during backtrace when adjacent
/// Delete+Insert steps are detected -- no separate post-processing merge pass.
#[derive(Debug, Clone)]
pub struct Myers {
    /// D-threshold: maximum edit distance to search before giving up.
    /// `None` means no limit (search up to n + m).
    /// Default: `Some(4096)`.
    threshold: Option<usize>,
    /// Whether to strip common prefix/suffix before running the algorithm.
    /// Default: `true`.
    use_common_affixes: bool,
}

impl Myers {
    /// Create a new Myers instance with default settings.
    ///
    /// - threshold: `Some(4096)`
    /// - use_common_affixes: `true`
    #[inline]
    pub fn new() -> Self {
        Self {
            threshold: Some(4096),
            use_common_affixes: true,
        }
    }

    /// Create a Myers instance with a specific D-threshold.
    #[inline]
    pub fn with_threshold(threshold: usize) -> Self {
        Self {
            threshold: Some(threshold),
            use_common_affixes: true,
        }
    }

    /// Create a Myers instance with no D-threshold (unlimited search).
    #[inline]
    pub fn without_threshold() -> Self {
        Self {
            threshold: None,
            use_common_affixes: true,
        }
    }

    /// Disable common prefix/suffix optimization (builder pattern).
    #[inline]
    pub fn without_common_affixes(mut self) -> Self {
        self.use_common_affixes = false;
        self
    }
}

impl Default for Myers {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of the forward Myers path search.
enum PathResult {
    /// Found an edit path with the V-array trace.
    Found(Vec<Vec<usize>>),
    /// D-threshold was exceeded without finding a complete path.
    ThresholdHit,
    /// Cancelled by progress callback.
    Cancelled,
}

/// Run the forward Myers O(ND) algorithm, returning the trace.
///
/// `source` and `target` are the inner slices (after stripping common affixes).
/// `max_d` is the maximum edit distance to search.
/// `progress` is an optional callback called every 32 d-iterations.
fn find_path(
    source: &[u32],
    target: &[u32],
    max_d: usize,
    progress: &mut Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
) -> PathResult {
    let n = source.len();
    let m = target.len();

    let offset = max_d as i64;
    let v_size = 2 * max_d + 3;
    let mut v: Vec<usize> = vec![0; v_size];
    let mut trace: Vec<Vec<usize>> = Vec::with_capacity(max_d + 1);

    for d in 0..=max_d {
        let di = d as i64;
        let mut k = -di;
        while k <= di {
            let idx = (k + offset) as usize;
            let idx_minus = ((k - 1) + offset) as usize;
            let idx_plus = ((k + 1) + offset) as usize;

            let mut x = if k == -di || (k != di && v[idx_minus] < v[idx_plus]) {
                v[idx_plus]
            } else {
                v[idx_minus] + 1
            };

            let mut y = (x as i64 - k) as usize;

            // Extend snake
            while x < n && y < m && source[x] == target[y] {
                x += 1;
                y += 1;
            }

            v[idx] = x;

            // Check if we've reached the end
            if x >= n && y >= m {
                trace.push(v.clone());
                return PathResult::Found(trace);
            }

            k += 2;
        }

        // Save V snapshot for this d
        trace.push(v.clone());

        // Progress callback every 32 iterations
        if d > 0 && d % 32 == 0 {
            if let Some(cb) = progress {
                let ratio = d as f64 / max_d as f64;
                if let ControlFlow::Break(()) = cb(ratio) {
                    return PathResult::Cancelled;
                }
            }
        }
    }

    PathResult::ThresholdHit
}

/// A raw edit step produced during backtrace (in forward order after reversal).
#[derive(Debug, Clone, Copy)]
enum EditStep {
    /// A diagonal run of equal elements: source[x_start..x_end] == target[y_start..y_end]
    Snake {
        x_start: usize,
        x_end: usize,
        y_start: usize,
        y_end: usize,
    },
    /// Delete source[x] (move right on the edit graph).
    /// `y` is the target position at this point.
    Delete { x: usize, y: usize },
    /// Insert target[y] (move down on the edit graph).
    /// `x` is the source position at this point.
    Insert { x: usize, y: usize },
}

/// Reconstruct edit operations from the Myers trace via backtrace.
///
/// Operates in inner-slice coordinates (0-based). Replace operations are emitted
/// directly when the backtrace encounters adjacent Delete+Insert (or Insert+Delete)
/// steps -- fused during operation construction, not as a separate pass.
fn backtrace(trace: &[Vec<usize>], n: usize, m: usize, offset: i64) -> Vec<EditOperation> {
    if trace.is_empty() {
        return Vec::new();
    }

    let d_final = trace.len() - 1;
    let mut x = n;
    let mut y = m;

    // Collect edit steps in reverse order
    let mut steps: Vec<EditStep> = Vec::new();

    for d in (0..=d_final).rev() {
        let di = d as i64;
        let k = x as i64 - y as i64;

        if d == 0 {
            // At d=0, there's no edit move. Just a snake from (0,0) to (x,y).
            if x > 0 {
                steps.push(EditStep::Snake {
                    x_start: 0,
                    x_end: x,
                    y_start: 0,
                    y_end: y,
                });
            }
            break;
        }

        let prev_v = &trace[d - 1];

        // Determine which diagonal we came from
        let came_down = k == -(di)
            || (k != di
                && prev_v[((k - 1) + offset) as usize] < prev_v[((k + 1) + offset) as usize]);

        let prev_k = if came_down { k + 1 } else { k - 1 };
        let prev_x = prev_v[(prev_k + offset) as usize];
        let prev_y = (prev_x as i64 - prev_k) as usize;

        // After the edit move, the snake starts at:
        let (snake_x, snake_y) = if came_down {
            // Insert (move down): x stays, y increments
            (prev_x, prev_y + 1)
        } else {
            // Delete (move right): x increments, y stays
            (prev_x + 1, prev_y)
        };

        // Record snake (from snake start to current position)
        if x > snake_x {
            debug_assert_eq!(x - snake_x, y - snake_y);
            steps.push(EditStep::Snake {
                x_start: snake_x,
                x_end: x,
                y_start: snake_y,
                y_end: y,
            });
        }

        // Record the edit move
        if came_down {
            steps.push(EditStep::Insert {
                x: prev_x,
                y: prev_y,
            });
        } else {
            steps.push(EditStep::Delete {
                x: prev_x,
                y: prev_y,
            });
        }

        x = prev_x;
        y = prev_y;
    }

    // Reverse to get forward order
    steps.reverse();

    // Convert steps to EditOperations, fusing adjacent Delete+Insert into Replace
    let mut ops: Vec<EditOperation> = Vec::new();
    let mut i = 0;
    while i < steps.len() {
        match steps[i] {
            EditStep::Snake {
                x_start,
                x_end,
                y_start,
                y_end,
            } => {
                ops.push(EditOperation::equal(
                    x_start as u32,
                    x_end as u32,
                    y_start as u32,
                    y_end as u32,
                ));
                i += 1;
            }
            EditStep::Delete { x: del_x, y: del_y } => {
                // Check if next non-snake step is Insert -> fuse into Replace
                if i + 1 < steps.len() {
                    if let EditStep::Insert { x: ins_x, y: ins_y } = steps[i + 1] {
                        let _ = ins_x;
                        ops.push(EditOperation::replace(
                            del_x as u32,
                            (del_x + 1) as u32,
                            ins_y as u32,
                            (ins_y + 1) as u32,
                        ));
                        i += 2;
                        continue;
                    }
                }
                ops.push(EditOperation::delete(
                    del_x as u32,
                    (del_x + 1) as u32,
                    del_y as u32,
                ));
                i += 1;
            }
            EditStep::Insert { x: ins_x, y: ins_y } => {
                // Check if next non-snake step is Delete -> fuse into Replace
                if i + 1 < steps.len() {
                    if let EditStep::Delete { x: del_x, y: del_y } = steps[i + 1] {
                        let _ = del_y;
                        ops.push(EditOperation::replace(
                            del_x as u32,
                            (del_x + 1) as u32,
                            ins_y as u32,
                            (ins_y + 1) as u32,
                        ));
                        i += 2;
                        continue;
                    }
                }
                ops.push(EditOperation::insert(
                    ins_x as u32,
                    ins_y as u32,
                    (ins_y + 1) as u32,
                ));
                i += 1;
            }
        }
    }

    coalesce(ops)
}

/// Merge consecutive operations of the same kind with contiguous indices.
///
/// For example, three adjacent `Insert(3,3..4), Insert(3,4..5), Insert(3,5..6)` become
/// a single `Insert(3,3..6)`. Similarly for consecutive Deletes, Equals, or Replaces.
fn coalesce(ops: Vec<EditOperation>) -> Vec<EditOperation> {
    if ops.is_empty() {
        return ops;
    }
    let mut result: Vec<EditOperation> = Vec::with_capacity(ops.len());
    result.push(ops[0]);
    for op in &ops[1..] {
        let last = result.last_mut().unwrap();
        if last.kind == op.kind
            && last.source_end == op.source_start
            && last.target_end == op.target_start
        {
            last.source_end = op.source_end;
            last.target_end = op.target_end;
        } else {
            result.push(*op);
        }
    }
    result
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

impl DiffAlgorithm for Myers {
    fn name(&self) -> &str {
        "myers"
    }

    fn compute(
        &self,
        source: &[u32],
        target: &[u32],
        mut progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
    ) -> Result<AlgorithmOutput, DiffError> {
        let threshold_used = self.threshold;
        let n = source.len();
        let m = target.len();

        // Step 1: Strip common affixes if enabled
        let (prefix_len, suffix_len, inner_source, inner_target) = if self.use_common_affixes {
            strip_common_affixes(source, target)
        } else {
            (0, 0, source, target)
        };

        let mut result_ops: Vec<EditOperation> = Vec::new();

        // Emit Equal for common prefix
        if prefix_len > 0 {
            result_ops.push(EditOperation::equal(
                0,
                prefix_len as u32,
                0,
                prefix_len as u32,
            ));
        }

        // Step 2: If both inner slices are empty, just prefix+suffix
        if inner_source.is_empty() && inner_target.is_empty() {
            // Only suffix Equal if there is one
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
                threshold_used,
            });
        }

        // Step 3: Handle edge cases where one side is empty
        if inner_source.is_empty() {
            // All inner_target elements are insertions
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
                threshold_used,
            });
        }

        if inner_target.is_empty() {
            // All inner_source elements are deletions
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
                threshold_used,
            });
        }

        // Step 3 continued: Run forward Myers
        let in_n = inner_source.len();
        let in_m = inner_target.len();
        let max_d = self.threshold.unwrap_or(in_n + in_m).min(in_n + in_m);

        let is_approximate;
        let mut inner_ops;

        match find_path(inner_source, inner_target, max_d, &mut progress) {
            PathResult::Found(trace) => {
                is_approximate = false;
                let offset = max_d as i64;
                inner_ops = backtrace(&trace, in_n, in_m, offset);
            }
            PathResult::ThresholdHit => {
                is_approximate = true;
                // Emit remaining as a single Replace (both non-empty at this point)
                inner_ops = vec![EditOperation::replace(0, in_n as u32, 0, in_m as u32)];
            }
            PathResult::Cancelled => {
                return Err(DiffError::Cancelled);
            }
        }

        // Step 5: Offset adjustment
        offset_operations(&mut inner_ops, prefix_len as u32, prefix_len as u32);

        // Step 6: Concatenate prefix + inner + suffix
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
            threshold_used,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::common::apply_operations;

    /// Helper to compute and return operations.
    fn compute_ops(source: &[u32], target: &[u32], myers: &Myers) -> AlgorithmOutput {
        myers.compute(source, target, None).unwrap()
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
        let m = Myers::new();
        assert_eq!(m.threshold, Some(4096));
        assert!(m.use_common_affixes);
    }

    #[test]
    fn with_threshold_constructor() {
        let m = Myers::with_threshold(100);
        assert_eq!(m.threshold, Some(100));
        assert!(m.use_common_affixes);
    }

    #[test]
    fn without_threshold_constructor() {
        let m = Myers::without_threshold();
        assert_eq!(m.threshold, None);
        assert!(m.use_common_affixes);
    }

    #[test]
    fn without_common_affixes_builder() {
        let m = Myers::new().without_common_affixes();
        assert!(!m.use_common_affixes);
        assert_eq!(m.threshold, Some(4096));
    }

    #[test]
    fn default_same_as_new() {
        let d = Myers::default();
        let n = Myers::new();
        assert_eq!(d.threshold, n.threshold);
        assert_eq!(d.use_common_affixes, n.use_common_affixes);
    }

    #[test]
    fn name_is_myers() {
        assert_eq!(Myers::new().name(), "myers");
    }

    // ---- Core algorithm tests ----

    #[test]
    fn empty_vs_empty() {
        let m = Myers::new();
        let output = compute_ops(&[], &[], &m);
        assert!(output.operations.is_empty());
        assert!(!output.is_approximate);
        assert_roundtrip(&[], &[], &output.operations);
    }

    #[test]
    fn single_equal() {
        let m = Myers::new();
        let output = compute_ops(&[0], &[0], &m);
        assert_eq!(output.operations, vec![EditOperation::equal(0, 1, 0, 1)]);
        assert!(!output.is_approximate);
        assert_roundtrip(&[0], &[0], &output.operations);
    }

    #[test]
    fn single_replace() {
        let m = Myers::new();
        let output = compute_ops(&[0], &[1], &m);
        assert_eq!(output.operations, vec![EditOperation::replace(0, 1, 0, 1)]);
        assert!(!output.is_approximate);
        assert_roundtrip(&[0], &[1], &output.operations);
    }

    #[test]
    fn identical_three_elements() {
        let m = Myers::new();
        let source = [0, 1, 2];
        let target = [0, 1, 2];
        let output = compute_ops(&source, &target, &m);
        // Common affixes strip everything -> single Equal
        assert_eq!(output.operations, vec![EditOperation::equal(0, 3, 0, 3)]);
        assert!(!output.is_approximate);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn middle_replace() {
        let m = Myers::new();
        let source = [0, 1, 2];
        let target = [0, 3, 2];
        let output = compute_ops(&source, &target, &m);
        assert_eq!(
            output.operations,
            vec![
                EditOperation::equal(0, 1, 0, 1),
                EditOperation::replace(1, 2, 1, 2),
                EditOperation::equal(2, 3, 2, 3),
            ]
        );
        assert!(!output.is_approximate);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn append_elements() {
        let m = Myers::new();
        let source = [0, 1, 2];
        let target = [0, 1, 2, 3, 4];
        let output = compute_ops(&source, &target, &m);
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
        let m = Myers::new();
        let source = [0, 1, 2, 3, 4];
        let target = [0, 1, 2];
        let output = compute_ops(&source, &target, &m);
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
        let m = Myers::new();
        let source = [0, 1, 2];
        let target = [0, 3, 2, 4, 0, 5];
        let output = compute_ops(&source, &target, &m);
        assert_eq!(
            output.operations,
            vec![
                EditOperation::equal(0, 1, 0, 1),   // "the"
                EditOperation::replace(1, 2, 1, 2), // "cat" -> "dog"
                EditOperation::equal(2, 3, 2, 3),   // "sat"
                EditOperation::insert(3, 3, 6),     // "on the mat"
            ]
        );
        assert!(!output.is_approximate);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn d_threshold_hit() {
        let m = Myers::with_threshold(5);
        // 100 completely different tokens
        let source: Vec<u32> = (0..100).collect();
        let target: Vec<u32> = (100..200).collect();
        let output = compute_ops(&source, &target, &m);
        assert!(output.is_approximate);
        assert_eq!(output.threshold_used, Some(5));
        // Should still produce a valid roundtrip
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn progress_callback_called() {
        let m = Myers::without_threshold();
        // Need enough difference that d > 32 to trigger callback
        let source: Vec<u32> = (0..50).collect();
        let target: Vec<u32> = (50..100).collect();

        let mut call_count = 0u32;
        let mut last_progress = 0.0f64;
        let output = m
            .compute(
                &source,
                &target,
                Some(&mut |p: f64| {
                    call_count += 1;
                    assert!(p >= last_progress, "Progress should be non-decreasing");
                    assert!(p >= 0.0 && p <= 1.0, "Progress should be in [0, 1]");
                    last_progress = p;
                    ControlFlow::Continue(())
                }),
            )
            .unwrap();
        assert!(
            call_count > 0,
            "Progress callback should have been called at least once"
        );
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn cancellation_via_callback() {
        let m = Myers::without_threshold();
        let source: Vec<u32> = (0..50).collect();
        let target: Vec<u32> = (50..100).collect();

        let result = m.compute(&source, &target, Some(&mut |_: f64| ControlFlow::Break(())));

        match result {
            Err(DiffError::Cancelled) => {} // expected
            other => panic!("Expected DiffError::Cancelled, got {:?}", other),
        }
    }

    #[test]
    fn common_affixes_disabled_same_results() {
        let source = [0, 1, 2, 3, 2];
        let target = [0, 3, 2, 4, 2];

        let with_affixes = Myers::new();
        let without_affixes = Myers::new().without_common_affixes();

        let output_with = compute_ops(&source, &target, &with_affixes);
        let output_without = compute_ops(&source, &target, &without_affixes);

        // Both should produce valid roundtrips
        assert_roundtrip(&source, &target, &output_with.operations);
        assert_roundtrip(&source, &target, &output_without.operations);

        // Neither should be approximate
        assert!(!output_with.is_approximate);
        assert!(!output_without.is_approximate);
    }

    #[test]
    fn empty_source_insert_all() {
        let m = Myers::new();
        let source: [u32; 0] = [];
        let target = [0, 1, 2];
        let output = compute_ops(&source, &target, &m);
        assert_eq!(output.operations, vec![EditOperation::insert(0, 0, 3)]);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn empty_target_delete_all() {
        let m = Myers::new();
        let source = [0, 1, 2];
        let target: [u32; 0] = [];
        let output = compute_ops(&source, &target, &m);
        assert_eq!(output.operations, vec![EditOperation::delete(0, 3, 0)]);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn single_insert_at_beginning() {
        let m = Myers::new();
        let source = [1, 2, 3];
        let target = [0, 1, 2, 3];
        let output = compute_ops(&source, &target, &m);
        // Insert [0] at beginning, then equal [1,2,3]
        assert_roundtrip(&source, &target, &output.operations);
        // Verify structure
        assert!(!output.is_approximate);
    }

    #[test]
    fn single_delete_at_beginning() {
        let m = Myers::new();
        let source = [0, 1, 2, 3];
        let target = [1, 2, 3];
        let output = compute_ops(&source, &target, &m);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn multiple_edits() {
        let m = Myers::new();
        let source = [0, 1, 2, 3, 4, 5];
        let target = [0, 9, 2, 3, 8, 5];
        let output = compute_ops(&source, &target, &m);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn completely_different_two_elements() {
        let m = Myers::new();
        let source = [0, 1];
        let target = [2, 3];
        let output = compute_ops(&source, &target, &m);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn threshold_used_in_output() {
        let m = Myers::with_threshold(42);
        let output = compute_ops(&[0], &[0], &m);
        assert_eq!(output.threshold_used, Some(42));
    }

    #[test]
    fn no_threshold_in_output() {
        let m = Myers::without_threshold();
        let output = compute_ops(&[0], &[0], &m);
        assert_eq!(output.threshold_used, None);
    }

    #[test]
    fn replace_emitted_directly_not_delete_insert() {
        // Verify that a simple replacement produces Replace, not Delete+Insert
        let m = Myers::new().without_common_affixes();
        let source = [0];
        let target = [1];
        let output = compute_ops(&source, &target, &m);
        assert_eq!(output.operations.len(), 1);
        assert_eq!(
            output.operations[0].kind,
            super::super::edit_operation::EditKind::Replace
        );
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn longer_sequence_roundtrip() {
        let m = Myers::new();
        let source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let target = [0, 1, 20, 21, 4, 5, 6, 30, 8, 9];
        let output = compute_ops(&source, &target, &m);
        assert_roundtrip(&source, &target, &output.operations);
        assert!(!output.is_approximate);
    }

    #[test]
    fn prepend_and_append() {
        let m = Myers::new();
        let source = [1, 2, 3];
        let target = [0, 1, 2, 3, 4];
        let output = compute_ops(&source, &target, &m);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn all_inserted() {
        let m = Myers::new();
        let source = [0, 1];
        let target = [0, 2, 3, 4, 1];
        let output = compute_ops(&source, &target, &m);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn threshold_zero() {
        // threshold=0 means only identical sequences succeed
        let m = Myers::with_threshold(0);
        let source = [0, 1];
        let target = [0, 2];
        let output = compute_ops(&source, &target, &m);
        // After stripping prefix [0], inner is [1] vs [2], which needs d=1
        // With threshold=0 (max_d=0), this can't find a path
        assert!(output.is_approximate);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Myers>();
    }

    #[test]
    fn debug_impl() {
        let m = Myers::new();
        let debug = format!("{:?}", m);
        assert!(debug.contains("Myers"));
    }

    #[test]
    fn clone_impl() {
        let m = Myers::with_threshold(100).without_common_affixes();
        let m2 = m.clone();
        assert_eq!(m2.threshold, Some(100));
        assert!(!m2.use_common_affixes);
    }

    #[test]
    fn interleaved_differences() {
        let m = Myers::new();
        let source = [0, 1, 0, 1, 0];
        let target = [1, 0, 1, 0, 1];
        let output = compute_ops(&source, &target, &m);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn d_threshold_with_partial_match() {
        // Source and target share a prefix but differ enough to exceed threshold
        let m = Myers::with_threshold(2);
        let source: Vec<u32> = (0..10).collect();
        let mut target: Vec<u32> = (0..5).collect();
        target.extend(50..60);
        let output = compute_ops(&source, &target, &m);
        // Common prefix [0..5] is stripped, then inner is [5..10] vs [50..60]
        // which needs d > 2, so threshold hit
        assert!(output.is_approximate);
        assert_roundtrip(&source, &target, &output.operations);
    }

    #[test]
    fn without_affixes_produces_valid_roundtrip_for_all_cases() {
        let m = Myers::new().without_common_affixes();

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
            let output = m.compute(source, target, None).unwrap();
            assert_roundtrip(source, target, &output.operations);
        }
    }
}
