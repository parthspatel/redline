//! RedlineResult: the output of a complete orchestration pipeline run.

use std::sync::Arc;

use hashbrown::HashMap;

use crate::analysis::AnalysisReport;
use crate::diff::DiffResult;
use crate::diff::edit_operation::EditOperation;
use crate::metrics::MetricValue;
use crate::orchestrate::filter::{Filter, FilterContext};

/// The combined result of a full Redline pipeline run.
///
/// Contains the diff result, computed metrics, and an optional analysis report.
/// Uses `Arc<AnalysisReport>` because `AnalysisReport` is `!Clone` (stores `Box<dyn Any>`),
/// and results may be shared with the pipeline cache.
pub struct RedlineResult {
    /// The diff computation result.
    pub diff: DiffResult,
    /// Computed metric values, keyed by metric ID.
    pub metrics: HashMap<String, MetricValue>,
    /// Optional analysis report (only present if analysis was enabled).
    pub analysis: Option<Arc<AnalysisReport>>,
}

impl RedlineResult {
    /// Filter operations using a predicate combinator.
    ///
    /// Returns an iterator over matching `EditOperation`s from the diff result.
    pub fn filter<'a>(
        &'a self,
        predicate: &'a Filter,
    ) -> impl Iterator<Item = &'a EditOperation> + 'a {
        let context = FilterContext::new(&self.diff, self.analysis.as_deref());
        self.diff
            .operations
            .iter()
            .filter(move |op| predicate.matches(op, &context))
    }
}

impl core::fmt::Debug for RedlineResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RedlineResult")
            .field("diff", &self.diff)
            .field("metrics_count", &self.metrics.len())
            .field("analysis", &self.analysis.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redline_result_debug_output() {
        // Just verify the Debug impl compiles
        fn assert_debug<T: core::fmt::Debug>() {}
        assert_debug::<RedlineResult>();
    }
}
