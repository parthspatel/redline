//! Filter predicate combinator system for edit operations.
//!
//! Provides composable, type-safe predicates for filtering edit operations from
//! diff results. Supports operator overloads (`&`, `|`, `!`) for building
//! complex filter expressions.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String};

use crate::analysis::report::AnalysisReport;
use crate::analysis::{EditClassifierResult, IntentCategory};
use crate::diff::edit_operation::{EditKind, EditOperation};
use crate::diff::result::DiffResult;

/// Read-only context available to filter predicates during evaluation.
pub struct FilterContext<'a> {
    /// The full diff result for cross-referencing token text.
    pub diff_result: &'a DiffResult,
    /// Analysis report, if analysis was performed. Required for intent filters.
    pub analysis_report: Option<&'a AnalysisReport>,
}

impl<'a> FilterContext<'a> {
    /// Create a new filter context.
    pub fn new(diff_result: &'a DiffResult, analysis_report: Option<&'a AnalysisReport>) -> Self {
        Self {
            diff_result,
            analysis_report,
        }
    }
}

/// A composable predicate for filtering edit operations.
///
/// Leaf variants test individual properties of an [`EditOperation`].
/// Combinator variants compose filters using boolean logic.
/// Use `&` (AND), `|` (OR), and `!` (NOT) operators for ergonomic construction.
#[derive(Debug, Clone)]
pub enum Filter {
    // Leaf predicates
    /// Matches operations of the given kind.
    Kind(EditKind),
    /// Matches operations where `max(source_len, target_len) >= n`.
    MinLength(u32),
    /// Matches operations where `max(source_len, target_len) <= n`.
    MaxLength(u32),
    /// Matches operations whose source text contains the given substring.
    SourceContains(String),
    /// Matches operations whose target text contains the given substring.
    TargetContains(String),
    /// Matches operations whose source range overlaps `[start, end)`.
    SpanRange {
        /// Inclusive start of the range.
        start: u32,
        /// Exclusive end of the range.
        end: u32,
    },
    /// Matches operations where `max(source_len, target_len) >= n` (token count).
    MinTokenCount(u32),
    /// Matches operations where `max(source_len, target_len) <= n` (token count).
    MaxTokenCount(u32),
    /// Matches operations whose primary intent category matches.
    Intent(IntentCategory),

    // Combinators
    /// Logical AND: both filters must match.
    And(Box<Filter>, Box<Filter>),
    /// Logical OR: at least one filter must match.
    Or(Box<Filter>, Box<Filter>),
    /// Logical NOT: the inner filter must NOT match.
    Not(Box<Filter>),
}

// Named constructors
impl Filter {
    /// Create a filter matching operations of the given kind.
    #[inline]
    pub fn kind(kind: EditKind) -> Self {
        Filter::Kind(kind)
    }

    /// Create a filter matching operations with length >= `len`.
    #[inline]
    pub fn min_length(len: u32) -> Self {
        Filter::MinLength(len)
    }

    /// Create a filter matching operations with length <= `len`.
    #[inline]
    pub fn max_length(len: u32) -> Self {
        Filter::MaxLength(len)
    }

    /// Create a filter matching operations whose source text contains `text`.
    #[inline]
    pub fn source_contains(text: impl Into<String>) -> Self {
        Filter::SourceContains(text.into())
    }

    /// Create a filter matching operations whose target text contains `text`.
    #[inline]
    pub fn target_contains(text: impl Into<String>) -> Self {
        Filter::TargetContains(text.into())
    }

    /// Create a filter matching operations overlapping the byte range `[start, end)`.
    #[inline]
    pub fn span_range(start: u32, end: u32) -> Self {
        Filter::SpanRange { start, end }
    }

    /// Create a filter matching operations with token count >= `count`.
    #[inline]
    pub fn min_token_count(count: u32) -> Self {
        Filter::MinTokenCount(count)
    }

    /// Create a filter matching operations with token count <= `count`.
    #[inline]
    pub fn max_token_count(count: u32) -> Self {
        Filter::MaxTokenCount(count)
    }

    /// Create a filter matching operations classified with the given intent category.
    ///
    /// Returns `false` if no analysis report is available in the context.
    #[inline]
    pub fn intent(category: IntentCategory) -> Self {
        Filter::Intent(category)
    }
}

// Evaluation
impl Filter {
    /// Evaluate this filter against an edit operation within the given context.
    pub fn matches(&self, op: &EditOperation, ctx: &FilterContext<'_>) -> bool {
        match self {
            Filter::Kind(k) => op.kind == *k,

            Filter::MinLength(n) => op.source_len().max(op.target_len()) >= *n,
            Filter::MaxLength(n) => op.source_len().max(op.target_len()) <= *n,

            Filter::SourceContains(text) => Self::text_contains(
                &ctx.diff_result.source,
                op.source_start,
                op.source_end,
                text,
            ),
            Filter::TargetContains(text) => Self::text_contains(
                &ctx.diff_result.target,
                op.target_start,
                op.target_end,
                text,
            ),

            Filter::SpanRange { start, end } => op.source_start < *end && op.source_end > *start,

            Filter::MinTokenCount(n) => op.source_len().max(op.target_len()) >= *n,
            Filter::MaxTokenCount(n) => op.source_len().max(op.target_len()) <= *n,

            Filter::Intent(category) => Self::intent_matches(op, ctx, category),

            Filter::And(l, r) => l.matches(op, ctx) && r.matches(op, ctx),
            Filter::Or(l, r) => l.matches(op, ctx) || r.matches(op, ctx),
            Filter::Not(f) => !f.matches(op, ctx),
        }
    }

    /// Check if the text covered by tokens `[token_start..token_end)` contains `needle`.
    fn text_contains(
        processed: &crate::process::ProcessedText,
        token_start: u32,
        token_end: u32,
        needle: &str,
    ) -> bool {
        let start = token_start as usize;
        let end = token_end as usize;

        if start >= end || end > processed.tokens.len() {
            return false;
        }

        let tokens = &processed.tokens[start..end];
        let byte_start = tokens[0].span.start() as usize;
        let byte_end = tokens[tokens.len() - 1].span.end() as usize;

        if byte_end > processed.normalized.len() {
            return false;
        }

        processed.normalized[byte_start..byte_end].contains(needle)
    }

    /// Check if the operation's primary intent classification matches `category`.
    fn intent_matches(
        op: &EditOperation,
        ctx: &FilterContext<'_>,
        category: &IntentCategory,
    ) -> bool {
        let report = match ctx.analysis_report {
            Some(r) => r,
            None => return false,
        };

        let classifier_result = match report.builtin::<EditClassifierResult>("edit_classifier") {
            Some(r) => r,
            None => return false,
        };

        // Find this operation in the diff result by position matching.
        let op_index = match ctx.diff_result.operations.iter().position(|o| {
            o.kind == op.kind
                && o.source_start == op.source_start
                && o.source_end == op.source_end
                && o.target_start == op.target_start
                && o.target_end == op.target_end
        }) {
            Some(i) => i,
            None => return false,
        };

        classifier_result.operations.iter().any(|c| {
            c.operation_index == op_index
                && c.categories
                    .first()
                    .is_some_and(|(cat, _)| *cat == *category)
        })
    }
}

// Operator overloads
impl core::ops::BitAnd for Filter {
    type Output = Filter;

    fn bitand(self, rhs: Filter) -> Filter {
        Filter::And(Box::new(self), Box::new(rhs))
    }
}

impl core::ops::BitOr for Filter {
    type Output = Filter;

    fn bitor(self, rhs: Filter) -> Filter {
        Filter::Or(Box::new(self), Box::new(rhs))
    }
}

impl core::ops::Not for Filter {
    type Output = Filter;

    fn not(self) -> Filter {
        Filter::Not(Box::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::result::{DiffMetadata, DiffStatistics};
    use crate::span::Span;
    use crate::text_store::TextStoreBuilder;
    use crate::token::{Token, TokenKind};
    use std::sync::Arc;

    /// Build a ProcessedText from a list of words.
    fn make_processed(words: &[&str]) -> Arc<crate::process::ProcessedText> {
        let joined = words.join(" ");
        let mut builder = TextStoreBuilder::new();
        let mut tokens = Vec::new();
        let mut offset = 0u32;

        for &w in words {
            let id = builder.intern(w);
            let end = offset + w.len() as u32;
            tokens.push(Token::new(id, Span::new(offset, end), TokenKind::Regular));
            offset = end + 1; // +1 for space separator
        }

        let store = builder.build();
        Arc::new(crate::process::ProcessedText {
            original: joined.clone(),
            layers: vec![],
            normalized: joined,
            tokens,
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        })
    }

    /// Build a DiffResult with the given operations, source words, and target words.
    fn make_diff(
        ops: Vec<EditOperation>,
        source_words: &[&str],
        target_words: &[&str],
    ) -> DiffResult {
        let source = make_processed(source_words);
        let target = make_processed(target_words);
        let src_len = source.tokens.len();
        let tgt_len = target.tokens.len();
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

    fn ctx(diff: &DiffResult) -> FilterContext<'_> {
        FilterContext::new(diff, None)
    }

    #[test]
    fn kind_filter_matches() {
        let diff = make_diff(
            vec![EditOperation::replace(0, 2, 0, 2)],
            &["hello", "world"],
            &["foo", "bar"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        assert!(Filter::kind(EditKind::Replace).matches(op, &c));
        assert!(!Filter::kind(EditKind::Equal).matches(op, &c));
        assert!(!Filter::kind(EditKind::Insert).matches(op, &c));
        assert!(!Filter::kind(EditKind::Delete).matches(op, &c));
    }

    #[test]
    fn min_length_filter() {
        let diff = make_diff(
            vec![EditOperation::replace(0, 5, 0, 3)],
            &["a", "b", "c", "d", "e"],
            &["x", "y", "z"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        // max(5, 3) = 5
        assert!(Filter::min_length(3).matches(op, &c));
        assert!(Filter::min_length(5).matches(op, &c));
        assert!(!Filter::min_length(6).matches(op, &c));
    }

    #[test]
    fn max_length_filter() {
        let diff = make_diff(
            vec![EditOperation::replace(0, 3, 0, 2)],
            &["a", "b", "c"],
            &["x", "y"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        // max(3, 2) = 3
        assert!(Filter::max_length(5).matches(op, &c));
        assert!(Filter::max_length(3).matches(op, &c));
        assert!(!Filter::max_length(2).matches(op, &c));
    }

    #[test]
    fn and_combinator() {
        let diff = make_diff(
            vec![
                EditOperation::replace(0, 3, 0, 2),
                EditOperation::equal(3, 4, 2, 3),
                EditOperation::replace(4, 5, 3, 4),
            ],
            &["a", "b", "c", "d", "e"],
            &["x", "y", "d", "z"],
        );
        let c = ctx(&diff);

        let f = Filter::kind(EditKind::Replace) & Filter::min_length(2);

        // Replace(0..3, 0..2): max(3,2)=3 >= 2 -> true
        assert!(f.matches(&diff.operations[0], &c));
        // Equal: not Replace -> false
        assert!(!f.matches(&diff.operations[1], &c));
        // Replace(4..5, 3..4): max(1,1)=1 < 2 -> false
        assert!(!f.matches(&diff.operations[2], &c));
    }

    #[test]
    fn or_combinator() {
        let diff = make_diff(
            vec![
                EditOperation::insert(0, 0, 1),
                EditOperation::delete(0, 1, 1),
                EditOperation::equal(1, 2, 1, 2),
            ],
            &["a", "b"],
            &["x", "a", "b"],
        );
        let c = ctx(&diff);

        let f = Filter::kind(EditKind::Insert) | Filter::kind(EditKind::Delete);

        assert!(f.matches(&diff.operations[0], &c));
        assert!(f.matches(&diff.operations[1], &c));
        assert!(!f.matches(&diff.operations[2], &c));
    }

    #[test]
    fn not_combinator() {
        let diff = make_diff(
            vec![
                EditOperation::equal(0, 1, 0, 1),
                EditOperation::replace(1, 2, 1, 2),
                EditOperation::insert(2, 2, 3),
                EditOperation::delete(2, 3, 3),
            ],
            &["a", "b", "c"],
            &["a", "x", "y", "z"],
        );
        let c = ctx(&diff);

        let f = !Filter::kind(EditKind::Equal);

        assert!(!f.matches(&diff.operations[0], &c)); // Equal -> NOT -> false
        assert!(f.matches(&diff.operations[1], &c)); // Replace -> true
        assert!(f.matches(&diff.operations[2], &c)); // Insert -> true
        assert!(f.matches(&diff.operations[3], &c)); // Delete -> true
    }

    #[test]
    fn nested_combinators() {
        let diff = make_diff(
            vec![
                EditOperation::replace(0, 3, 0, 2), // Replace, len=3
                EditOperation::insert(3, 2, 3),     // Insert, len=1
                EditOperation::replace(3, 4, 3, 4), // Replace, len=1
                EditOperation::equal(4, 5, 4, 5),   // Equal, len=1
            ],
            &["a", "b", "c", "d", "e"],
            &["x", "y", "z", "d", "e"],
        );
        let c = ctx(&diff);

        // (Replace & min_length(2)) | Insert
        let f = (Filter::kind(EditKind::Replace) & Filter::min_length(2))
            | Filter::kind(EditKind::Insert);

        assert!(f.matches(&diff.operations[0], &c)); // Replace len=3 -> AND true
        assert!(f.matches(&diff.operations[1], &c)); // Insert -> OR true
        assert!(!f.matches(&diff.operations[2], &c)); // Replace len=1 -> AND false, not Insert
        assert!(!f.matches(&diff.operations[3], &c)); // Equal -> both false
    }

    #[test]
    fn span_range_filter() {
        let diff = make_diff(
            vec![
                EditOperation::replace(5, 10, 0, 5),
                EditOperation::replace(11, 15, 5, 9),
                EditOperation::replace(0, 3, 9, 12),
            ],
            &[
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
            ],
            &["x", "y", "z", "w", "v", "u", "t", "s", "r", "q", "p", "o"],
        );
        let c = ctx(&diff);

        let f = Filter::span_range(5, 10);

        // source [5,10) overlaps [5,10) -> true
        assert!(f.matches(&diff.operations[0], &c));
        // source [11,15) does NOT overlap [5,10) -> false
        assert!(!f.matches(&diff.operations[1], &c));
        // source [0,3) does NOT overlap [5,10) -> false
        assert!(!f.matches(&diff.operations[2], &c));
    }

    #[test]
    fn source_contains_filter() {
        let diff = make_diff(
            vec![EditOperation::replace(0, 2, 0, 1)],
            &["hello", "world"],
            &["goodbye"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        assert!(Filter::source_contains("hello").matches(op, &c));
        assert!(Filter::source_contains("world").matches(op, &c));
        // Text between first and last token spans includes the space
        assert!(Filter::source_contains("lo wo").matches(op, &c));
        assert!(!Filter::source_contains("goodbye").matches(op, &c));
    }

    #[test]
    fn target_contains_filter() {
        let diff = make_diff(
            vec![EditOperation::replace(0, 1, 0, 2)],
            &["old"],
            &["new", "text"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        assert!(Filter::target_contains("new").matches(op, &c));
        assert!(Filter::target_contains("text").matches(op, &c));
        assert!(!Filter::target_contains("old").matches(op, &c));
    }

    #[test]
    fn token_count_filters() {
        let diff = make_diff(
            vec![EditOperation::replace(0, 4, 0, 2)],
            &["a", "b", "c", "d"],
            &["x", "y"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        // max(4, 2) = 4
        assert!(Filter::min_token_count(4).matches(op, &c));
        assert!(!Filter::min_token_count(5).matches(op, &c));
        assert!(Filter::max_token_count(4).matches(op, &c));
        assert!(!Filter::max_token_count(3).matches(op, &c));
    }

    #[test]
    fn filter_is_clone() {
        let f = Filter::kind(EditKind::Replace) & Filter::min_length(5);
        let f2 = f.clone();
        assert_eq!(format!("{f:?}"), format!("{f2:?}"));
    }

    #[test]
    fn double_negation() {
        let diff = make_diff(
            vec![
                EditOperation::equal(0, 1, 0, 1),
                EditOperation::replace(1, 2, 1, 2),
            ],
            &["a", "b"],
            &["a", "x"],
        );
        let c = ctx(&diff);

        let f = !!Filter::kind(EditKind::Equal);

        assert!(f.matches(&diff.operations[0], &c));
        assert!(!f.matches(&diff.operations[1], &c));
    }

    #[test]
    fn complex_expression() {
        let diff = make_diff(
            vec![EditOperation::replace(0, 12, 0, 8)],
            &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
            &["x", "y", "z", "w", "v", "u", "t", "s"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        let f = Filter::kind(EditKind::Replace) & Filter::min_length(10);
        assert!(f.matches(op, &c)); // max(12, 8) = 12 >= 10

        let f2 = Filter::kind(EditKind::Replace) & Filter::min_length(15);
        assert!(!f2.matches(op, &c)); // 12 < 15
    }

    #[test]
    fn intent_no_analysis_report_returns_false() {
        let diff = make_diff(
            vec![EditOperation::replace(0, 1, 0, 1)],
            &["hello"],
            &["world"],
        );
        let c = ctx(&diff); // No analysis report

        let f = Filter::intent(IntentCategory::Correction);
        assert!(!f.matches(&diff.operations[0], &c));
    }

    #[test]
    fn source_contains_empty_range() {
        let diff = make_diff(vec![EditOperation::insert(0, 0, 1)], &["a"], &["x", "a"]);
        let c = ctx(&diff);
        let op = &diff.operations[0];

        // source_start == source_end (0 == 0), so no source text
        assert!(!Filter::source_contains("anything").matches(op, &c));
    }

    #[test]
    fn target_contains_empty_range() {
        let diff = make_diff(
            vec![EditOperation::delete(0, 1, 0)],
            &["removed"],
            &["kept"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        // target_start == target_end (0 == 0), so no target text
        assert!(!Filter::target_contains("anything").matches(op, &c));
    }

    #[test]
    fn span_range_partial_overlap() {
        let diff = make_diff(
            vec![EditOperation::replace(3, 8, 0, 5)],
            &["a", "b", "c", "d", "e", "f", "g", "h"],
            &["x", "y", "z", "w", "v"],
        );
        let c = ctx(&diff);
        let op = &diff.operations[0];

        // source [3,8) overlaps [5,10) at [5,8)
        assert!(Filter::span_range(5, 10).matches(op, &c));
        // source [3,8) overlaps [0,4) at [3,4)
        assert!(Filter::span_range(0, 4).matches(op, &c));
        // source [3,8) does NOT overlap [8,12)
        assert!(!Filter::span_range(8, 12).matches(op, &c));
        // source [3,8) does NOT overlap [0,3)
        assert!(!Filter::span_range(0, 3).matches(op, &c));
    }
}
