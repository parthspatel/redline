//! AnalysisReport: typed + dynamic result storage for analyzer outputs.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

use core::any::Any;
use core::time::Duration;

use hashbrown::HashMap;

use super::AnalysisResult;

/// Status of an analyzer's execution.
#[derive(Debug, Clone)]
pub enum AnalyzerStatus {
    /// Analyzer completed successfully.
    Success,
    /// Analyzer failed with a reason.
    Failed(String),
    /// Analyzer was skipped (e.g., because a dependency failed).
    Skipped(String),
}

/// Execution metadata for a single analyzer run.
#[derive(Debug, Clone)]
pub struct AnalyzerExecutionMeta {
    /// ID of the analyzer.
    pub analyzer_id: String,
    /// Execution status.
    pub status: AnalyzerStatus,
    /// Wall-clock duration of execution.
    pub duration: Duration,
    /// Order in which this analyzer was executed (0-based).
    pub order_index: usize,
}

/// Collected results from running all analyzers on a DiffResult.
///
/// Results are stored type-erased, keyed by analyzer ID. Built-in analyzers
/// have typed accessor methods; plugins use `plugin_result()` for `AnalysisResult`
/// or `raw_result()` for custom downcasting.
pub struct AnalysisReport {
    results: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Execution metadata for each analyzer run.
    pub execution_metadata: Vec<AnalyzerExecutionMeta>,
}

impl AnalysisReport {
    /// Create a new empty report.
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            execution_metadata: Vec::new(),
        }
    }

    // ── Typed Built-in Accessors ──────────────────────────────────

    /// Access a built-in result by analyzer ID with a concrete type.
    ///
    /// Usage: `report.builtin::<SemanticResult>("semantic")`
    pub fn builtin<T: Any>(&self, analyzer_id: &str) -> Option<&T> {
        self.results.get(analyzer_id)?.downcast_ref::<T>()
    }

    // ── Dynamic Plugin Access ─────────────────────────────────────

    /// Access a plugin's result as an `AnalysisResult`.
    pub fn plugin_result(&self, analyzer_id: &str) -> Option<&AnalysisResult> {
        self.results
            .get(analyzer_id)?
            .downcast_ref::<AnalysisResult>()
    }

    /// Access the raw stored result for an analyzer.
    pub fn raw_result(&self, analyzer_id: &str) -> Option<&(dyn Any + Send + Sync)> {
        self.results.get(analyzer_id).map(|b| b.as_ref())
    }

    // ── Query Methods ─────────────────────────────────────────────

    /// Returns all analyzer IDs that have results stored.
    pub fn analyzer_ids(&self) -> Vec<&str> {
        self.results.keys().map(|s| s.as_str()).collect()
    }

    /// Returns metadata for analyzers that succeeded.
    pub fn succeeded(&self) -> Vec<&AnalyzerExecutionMeta> {
        self.execution_metadata
            .iter()
            .filter(|m| matches!(m.status, AnalyzerStatus::Success))
            .collect()
    }

    /// Returns metadata for analyzers that failed.
    pub fn failed(&self) -> Vec<&AnalyzerExecutionMeta> {
        self.execution_metadata
            .iter()
            .filter(|m| matches!(m.status, AnalyzerStatus::Failed(_)))
            .collect()
    }

    /// Returns metadata for analyzers that were skipped.
    pub fn skipped(&self) -> Vec<&AnalyzerExecutionMeta> {
        self.execution_metadata
            .iter()
            .filter(|m| matches!(m.status, AnalyzerStatus::Skipped(_)))
            .collect()
    }

    /// Returns true if an analyzer's result is stored.
    pub fn has_result(&self, analyzer_id: &str) -> bool {
        self.results.contains_key(analyzer_id)
    }

    /// Total number of stored results.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    // ── Internal Mutation ─────────────────────────────────────────

    /// Insert a result for the given analyzer. Called by the coordinator.
    pub(crate) fn insert(&mut self, analyzer_id: String, result: Box<dyn Any + Send + Sync>) {
        self.results.insert(analyzer_id, result);
    }
}

impl Default for AnalysisReport {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for AnalysisReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AnalysisReport")
            .field("result_count", &self.results.len())
            .field("execution_metadata", &self.execution_metadata)
            .finish()
    }
}
