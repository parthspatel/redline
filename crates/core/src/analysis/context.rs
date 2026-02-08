//! AnalysisContext: read-only access to DiffResult and MetricsEngine for analyzers.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::diff::DiffResult;
use crate::diff::edit_operation::EditKind;
use crate::metrics::{MetricInput, MetricValue, MetricsEngine};
use crate::token::Token;

/// Read-only context provided to analyzers during execution.
///
/// Provides access to the `DiffResult` being analyzed and the `MetricsEngine`
/// for computing metrics on demand. Helper methods extract text and tokens
/// for specific edit operations.
pub struct AnalysisContext<'a> {
    /// The diff result being analyzed.
    pub diff: &'a DiffResult,
    /// The metrics engine for computing metrics on demand.
    pub metrics_engine: &'a MetricsEngine,
}

impl<'a> AnalysisContext<'a> {
    /// Create a new analysis context.
    pub fn new(diff: &'a DiffResult, metrics_engine: &'a MetricsEngine) -> Self {
        Self {
            diff,
            metrics_engine,
        }
    }

    /// Returns `(index, &EditOperation)` pairs for all non-Equal operations.
    ///
    /// The index is the position in the original `operations` vector, which is
    /// critical for annotation correctness — annotations reference this index.
    pub fn change_operations(&self) -> Vec<(usize, &crate::diff::EditOperation)> {
        self.diff
            .operations
            .iter()
            .enumerate()
            .filter(|(_, op)| op.kind != EditKind::Equal)
            .collect()
    }

    /// Extract source text for a specific operation by index.
    ///
    /// The operation's `source_start`/`source_end` are **token indices**.
    /// This method maps those to byte offsets via token spans to extract
    /// the corresponding substring from the normalized source text.
    pub fn source_text_for_op(&self, op_index: usize) -> Option<&str> {
        let op = self.diff.operations.get(op_index)?;
        let tokens = &self.diff.source.tokens;
        let start = op.source_start as usize;
        let end = op.source_end as usize;
        if start >= end || start >= tokens.len() {
            return Some("");
        }
        let end = end.min(tokens.len());
        let byte_start = tokens[start].span.start() as usize;
        let byte_end = tokens[end - 1].span.end() as usize;
        let text = &self.diff.source.normalized;
        if byte_end <= text.len() {
            Some(&text[byte_start..byte_end])
        } else {
            None
        }
    }

    /// Extract target text for a specific operation by index.
    ///
    /// Same as `source_text_for_op` but for the target side.
    pub fn target_text_for_op(&self, op_index: usize) -> Option<&str> {
        let op = self.diff.operations.get(op_index)?;
        let tokens = &self.diff.target.tokens;
        let start = op.target_start as usize;
        let end = op.target_end as usize;
        if start >= end || start >= tokens.len() {
            return Some("");
        }
        let end = end.min(tokens.len());
        let byte_start = tokens[start].span.start() as usize;
        let byte_end = tokens[end - 1].span.end() as usize;
        let text = &self.diff.target.normalized;
        if byte_end <= text.len() {
            Some(&text[byte_start..byte_end])
        } else {
            None
        }
    }

    /// Get source tokens for a specific operation's token range.
    pub fn source_tokens_for_op(&self, op_index: usize) -> Option<&[Token]> {
        let op = self.diff.operations.get(op_index)?;
        let tokens = &self.diff.source.tokens;
        let start = op.source_start as usize;
        let end = (op.source_end as usize).min(tokens.len());
        if start >= end {
            return Some(&[]);
        }
        Some(&tokens[start..end])
    }

    /// Get target tokens for a specific operation's token range.
    pub fn target_tokens_for_op(&self, op_index: usize) -> Option<&[Token]> {
        let op = self.diff.operations.get(op_index)?;
        let tokens = &self.diff.target.tokens;
        let start = op.target_start as usize;
        let end = (op.target_end as usize).min(tokens.len());
        if start >= end {
            return Some(&[]);
        }
        Some(&tokens[start..end])
    }

    /// Compute a metric on the source text.
    pub fn compute_source_metric(&self, metric_id: &str) -> MetricValue {
        let input = MetricInput::Single(&self.diff.source);
        self.metrics_engine.get(metric_id, &input)
    }

    /// Compute a metric on the target text.
    pub fn compute_target_metric(&self, metric_id: &str) -> MetricValue {
        let input = MetricInput::Single(&self.diff.target);
        self.metrics_engine.get(metric_id, &input)
    }

    /// Compute a pairwise metric comparing source and target.
    pub fn compute_pairwise_metric(&self, metric_id: &str) -> MetricValue {
        let input = MetricInput::Pairwise(&self.diff.source, &self.diff.target);
        self.metrics_engine.get(metric_id, &input)
    }
}
