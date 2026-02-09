//! RedlineResult: the output of a complete orchestration pipeline run.

use hashbrown::HashMap;

use crate::analysis::AnalysisReport;
use crate::diff::DiffResult;
use crate::metrics::MetricValue;

/// The combined result of a full Redline pipeline run.
///
/// Contains the diff result, computed metrics, and an optional analysis report.
/// Not `Clone` because `AnalysisReport` stores `Box<dyn Any>`.
pub struct RedlineResult {
    /// The diff computation result.
    pub diff: DiffResult,
    /// Computed metric values, keyed by metric ID.
    pub metrics: HashMap<String, MetricValue>,
    /// Optional analysis report (only present if analysis was enabled).
    pub analysis: Option<AnalysisReport>,
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
