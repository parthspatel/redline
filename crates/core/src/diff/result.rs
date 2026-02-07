//! DiffResult, DiffStatistics, and DiffMetadata types.

#[cfg(not(feature = "std"))]
use alloc::{string::String, sync::Arc, vec, vec::Vec};
#[cfg(feature = "std")]
use std::sync::Arc;

use crate::process::ProcessedText;

use super::edit_operation::{EditKind, EditOperation};

/// Statistics computed from a diff operation list.
#[derive(Debug, Clone)]
pub struct DiffStatistics {
    pub equal_count: usize,
    pub delete_count: usize,
    pub insert_count: usize,
    pub replace_count: usize,
    /// Edit distance: delete_count + insert_count + replace_count.
    pub edit_distance: usize,
    /// Similarity ratio: `2 * lcs_length / (source_len + target_len)`.
    pub similarity_ratio: f64,
    /// Length of the longest common subsequence (sum of Equal source_len values).
    pub lcs_length: usize,
    /// Compression ratio: `target_len / source_len`.
    pub compression_ratio: f64,
    /// Change density per window (up to 10 windows).
    pub change_density: Vec<f64>,
}

impl DiffStatistics {
    pub fn from_operations(ops: &[EditOperation], source_len: usize, target_len: usize) -> Self {
        let mut equal_count = 0usize;
        let mut delete_count = 0usize;
        let mut insert_count = 0usize;
        let mut replace_count = 0usize;
        let mut lcs_length = 0usize;

        for op in ops {
            match op.kind {
                EditKind::Equal => {
                    equal_count += 1;
                    lcs_length += op.source_len() as usize;
                }
                EditKind::Delete => delete_count += 1,
                EditKind::Insert => insert_count += 1,
                EditKind::Replace => replace_count += 1,
            }
        }

        let edit_distance = delete_count + insert_count + replace_count;

        let similarity_ratio = if source_len + target_len == 0 {
            1.0
        } else {
            (2.0 * lcs_length as f64) / (source_len + target_len) as f64
        };

        let compression_ratio = if source_len == 0 {
            if target_len == 0 { 1.0 } else { f64::INFINITY }
        } else {
            target_len as f64 / source_len as f64
        };

        let change_density = if ops.is_empty() {
            vec![]
        } else {
            let window_size = (ops.len() + 9) / 10;
            ops.chunks(window_size.max(1))
                .map(|chunk| {
                    let non_equal = chunk.iter().filter(|op| op.kind != EditKind::Equal).count();
                    non_equal as f64 / chunk.len() as f64
                })
                .collect()
        };

        Self {
            equal_count,
            delete_count,
            insert_count,
            replace_count,
            edit_distance,
            similarity_ratio,
            lcs_length,
            compression_ratio,
            change_density,
        }
    }
}

/// Metadata about the diff computation.
#[derive(Debug, Clone)]
pub struct DiffMetadata {
    pub algorithm_name: String,
    pub is_approximate: bool,
    pub threshold_used: Option<usize>,
}

/// The result of a diff computation.
#[derive(Debug, Clone)]
pub struct DiffResult {
    pub operations: Vec<EditOperation>,
    pub statistics: DiffStatistics,
    pub metadata: DiffMetadata,
    pub source: Arc<ProcessedText>,
    pub target: Arc<ProcessedText>,
}

impl DiffResult {
    /// Returns grouped hunks of operations with `context` Equal ops around changes.
    ///
    /// Each hunk is a slice of consecutive operations containing at least one non-Equal
    /// operation, expanded by up to `context` Equal operations on each side.
    /// Overlapping expanded regions are merged.
    pub fn hunks(&self, context: usize) -> Vec<&[EditOperation]> {
        if self.operations.is_empty() {
            return vec![];
        }

        // Find ranges of non-Equal operations
        let mut change_ranges: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < self.operations.len() {
            if self.operations[i].kind != EditKind::Equal {
                let start = i;
                while i < self.operations.len() && self.operations[i].kind != EditKind::Equal {
                    i += 1;
                }
                change_ranges.push((start, i));
            } else {
                i += 1;
            }
        }

        if change_ranges.is_empty() {
            return vec![];
        }

        // Expand each range by context and merge overlapping
        let mut expanded: Vec<(usize, usize)> = Vec::new();
        for &(start, end) in &change_ranges {
            let exp_start = start.saturating_sub(context);
            let exp_end = (end + context).min(self.operations.len());

            if let Some(last) = expanded.last_mut() {
                if exp_start <= last.1 {
                    last.1 = last.1.max(exp_end);
                    continue;
                }
            }
            expanded.push((exp_start, exp_end));
        }

        expanded
            .iter()
            .map(|&(start, end)| &self.operations[start..end])
            .collect()
    }

    #[inline]
    pub fn is_approximate(&self) -> bool {
        self.metadata.is_approximate
    }

    #[inline]
    pub fn similarity(&self) -> f64 {
        self.statistics.similarity_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_store::TextStoreBuilder;
    use static_assertions::assert_impl_all;

    assert_impl_all!(DiffResult: Clone, Send, Sync, core::fmt::Debug);

    fn _assert_static() {
        fn check<T: 'static>() {}
        check::<DiffResult>();
    }

    fn make_processed(text: &str, words: &[&str]) -> Arc<ProcessedText> {
        use crate::span::Span;
        use crate::token::{Token, TokenKind};

        let mut builder = TextStoreBuilder::new();
        let mut tokens = Vec::new();
        let mut offset = 0u32;
        for &w in words {
            let id = builder.intern(w);
            let end = offset + w.len() as u32;
            tokens.push(Token::new(id, Span::new(offset, end), TokenKind::Regular));
            offset = end + 1;
        }
        let store = builder.build();

        Arc::new(ProcessedText {
            original: text.to_string(),
            layers: vec![],
            normalized: text.to_string(),
            tokens,
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        })
    }

    fn make_diff_result(ops: Vec<EditOperation>, src_len: usize, tgt_len: usize) -> DiffResult {
        let source = make_processed("source", &["source"]);
        let target = make_processed("target", &["target"]);
        DiffResult {
            statistics: DiffStatistics::from_operations(&ops, src_len, tgt_len),
            metadata: DiffMetadata {
                algorithm_name: "test".into(),
                is_approximate: false,
                threshold_used: None,
            },
            operations: ops,
            source,
            target,
        }
    }

    #[test]
    fn stats_all_equal() {
        let stats = DiffStatistics::from_operations(&[EditOperation::equal(0, 3, 0, 3)], 3, 3);
        assert_eq!(stats.equal_count, 1);
        assert_eq!(stats.lcs_length, 3);
        assert_eq!(stats.edit_distance, 0);
        assert!((stats.similarity_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_all_replace() {
        let stats = DiffStatistics::from_operations(&[EditOperation::replace(0, 3, 0, 3)], 3, 3);
        assert_eq!(stats.replace_count, 1);
        assert_eq!(stats.lcs_length, 0);
        assert_eq!(stats.edit_distance, 1);
        assert!(stats.similarity_ratio.abs() < f64::EPSILON);
    }

    #[test]
    fn stats_mixed() {
        let ops = vec![
            EditOperation::equal(0, 2, 0, 2),
            EditOperation::delete(2, 3, 2),
            EditOperation::insert(3, 2, 4),
            EditOperation::replace(3, 5, 4, 6),
        ];
        let stats = DiffStatistics::from_operations(&ops, 5, 6);
        assert_eq!(stats.equal_count, 1);
        assert_eq!(stats.delete_count, 1);
        assert_eq!(stats.insert_count, 1);
        assert_eq!(stats.replace_count, 1);
        assert_eq!(stats.lcs_length, 2);
        assert_eq!(stats.edit_distance, 3);
        let expected = (2.0 * 2.0) / (5.0 + 6.0);
        assert!((stats.similarity_ratio - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_empty() {
        let stats = DiffStatistics::from_operations(&[], 0, 0);
        assert!((stats.similarity_ratio - 1.0).abs() < f64::EPSILON);
        assert!((stats.compression_ratio - 1.0).abs() < f64::EPSILON);
        assert!(stats.change_density.is_empty());
    }

    #[test]
    fn stats_change_density() {
        let ops: Vec<EditOperation> = (0..10u32)
            .map(|i| {
                if i % 2 == 0 {
                    EditOperation::equal(i, i + 1, i, i + 1)
                } else {
                    EditOperation::replace(i, i + 1, i, i + 1)
                }
            })
            .collect();
        let stats = DiffStatistics::from_operations(&ops, 10, 10);
        assert_eq!(stats.change_density.len(), 10);
        for (i, d) in stats.change_density.iter().enumerate() {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            assert!((d - expected).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn stats_compression_ratio_infinity() {
        let stats = DiffStatistics::from_operations(&[], 0, 5);
        assert!(stats.compression_ratio.is_infinite());
    }

    #[test]
    fn hunks_all_equal() {
        let r = make_diff_result(
            vec![
                EditOperation::equal(0, 1, 0, 1),
                EditOperation::equal(1, 2, 1, 2),
            ],
            2,
            2,
        );
        assert!(r.hunks(1).is_empty());
    }

    #[test]
    fn hunks_one_change_middle() {
        let r = make_diff_result(
            vec![
                EditOperation::equal(0, 1, 0, 1),
                EditOperation::equal(1, 2, 1, 2),
                EditOperation::equal(2, 3, 2, 3),
                EditOperation::replace(3, 4, 3, 4),
                EditOperation::equal(4, 5, 4, 5),
                EditOperation::equal(5, 6, 5, 6),
                EditOperation::equal(6, 7, 6, 7),
            ],
            7,
            7,
        );
        let hunks = r.hunks(1);
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].len(), 3);
        assert_eq!(hunks[0][1].kind, EditKind::Replace);
    }

    #[test]
    fn hunks_two_far_apart() {
        let r = make_diff_result(
            vec![
                EditOperation::replace(0, 1, 0, 1),
                EditOperation::equal(1, 2, 1, 2),
                EditOperation::equal(2, 3, 2, 3),
                EditOperation::equal(3, 4, 3, 4),
                EditOperation::equal(4, 5, 4, 5),
                EditOperation::replace(5, 6, 5, 6),
            ],
            6,
            6,
        );
        assert_eq!(r.hunks(1).len(), 2);
    }

    #[test]
    fn hunks_merged() {
        let r = make_diff_result(
            vec![
                EditOperation::replace(0, 1, 0, 1),
                EditOperation::equal(1, 2, 1, 2),
                EditOperation::replace(2, 3, 2, 3),
            ],
            3,
            3,
        );
        assert_eq!(r.hunks(2).len(), 1);
    }

    #[test]
    fn hunks_context_zero() {
        let r = make_diff_result(
            vec![
                EditOperation::equal(0, 1, 0, 1),
                EditOperation::replace(1, 2, 1, 2),
                EditOperation::equal(2, 3, 2, 3),
            ],
            3,
            3,
        );
        let hunks = r.hunks(0);
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].len(), 1);
        assert_eq!(hunks[0][0].kind, EditKind::Replace);
    }

    #[test]
    fn hunks_empty() {
        let r = make_diff_result(vec![], 0, 0);
        assert!(r.hunks(3).is_empty());
    }

    #[test]
    fn convenience_methods() {
        let r = make_diff_result(vec![EditOperation::equal(0, 3, 0, 3)], 3, 3);
        assert!((r.similarity() - 1.0).abs() < f64::EPSILON);
        assert!(!r.is_approximate());
    }

    #[test]
    fn diff_result_in_hashmap() {
        use std::collections::HashMap;
        let mut map: HashMap<String, DiffResult> = HashMap::new();
        map.insert(
            "k".into(),
            make_diff_result(vec![EditOperation::equal(0, 1, 0, 1)], 1, 1),
        );
        assert!(map.get("k").is_some());
    }
}
