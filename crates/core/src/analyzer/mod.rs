//! Diff analyzers for extracting insights and metrics
//!
//! Provides two types of analyzers:
//! - Single diff analyzers: Analyze individual diff results
//! - Multi-diff analyzers: Analyze collections of diffs to find patterns

pub mod classifiers;
pub mod multi;
pub mod selectors;

use crate::diff::DiffResult;
use std::collections::HashMap;

/// Result of a single analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Name of the analyzer that produced this result
    pub analyzer_name: String,

    /// Metrics computed by the analyzer
    pub metrics: HashMap<String, f64>,

    /// Textual insights or findings
    pub insights: Vec<String>,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Metadata about the analysis
    pub metadata: HashMap<String, String>,
}

impl AnalysisResult {
    pub fn new(analyzer_name: impl Into<String>) -> Self {
        Self {
            analyzer_name: analyzer_name.into(),
            metrics: HashMap::new(),
            insights: Vec::new(),
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.insert(name.into(), value);
    }

    pub fn add_insight(&mut self, insight: impl Into<String>) {
        self.insights.push(insight.into());
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Trait for analyzers that work on a single diff result
pub trait SingleDiffAnalyzer: Send + Sync {
    /// Analyze a single diff result
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult;

    /// Get the name of this analyzer
    fn name(&self) -> &str;

    /// Get a description of what this analyzer does
    fn description(&self) -> &str {
        ""
    }

    /// Declare the metric dependencies this analyzer requires
    /// Returns a list of NodeDependencies that will be added to the execution plan
    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        Vec::new()
    }

    /// Clone into a box
    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer>;
}

impl Clone for Box<dyn SingleDiffAnalyzer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Trait for analyzers that work on multiple diffs
pub trait MultiDiffAnalyzer: Send + Sync {
    /// Analyze a collection of diffs
    fn analyze(&self, diffs: &[&DiffResult]) -> AnalysisResult;

    /// Get the name of this analyzer
    fn name(&self) -> &str;

    /// Get a description of what this analyzer does
    fn description(&self) -> &str {
        ""
    }

    /// Declare the metric dependencies this analyzer requires
    /// Returns a list of NodeDependencies that will be added to the execution plan
    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        Vec::new()
    }

    /// Clone into a box
    fn clone_box(&self) -> Box<dyn MultiDiffAnalyzer>;
}

impl Clone for Box<dyn MultiDiffAnalyzer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Combined analysis report from multiple analyzers
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// All analysis results
    pub results: Vec<AnalysisResult>,

    /// Summary statistics
    pub summary: AnalysisSummary,
}

impl AnalysisReport {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            summary: AnalysisSummary::default(),
        }
    }

    pub fn add_result(&mut self, result: AnalysisResult) {
        self.results.push(result);
    }

    pub fn compute_summary(&mut self) {
        let mut total_metrics = HashMap::new();
        let mut metric_counts = HashMap::new();

        for result in &self.results {
            for (metric, value) in &result.metrics {
                *total_metrics.entry(metric.clone()).or_insert(0.0) += value;
                *metric_counts.entry(metric.clone()).or_insert(0) += 1;
            }
        }

        self.summary.average_metrics = total_metrics
            .into_iter()
            .map(|(k, v)| (k.clone(), v / metric_counts[&k] as f64))
            .collect();

        self.summary.total_analyses = self.results.len();
    }

    /// Get all insights as a flat list
    pub fn all_insights(&self) -> Vec<&str> {
        self.results
            .iter()
            .flat_map(|r| r.insights.iter().map(|s| s.as_str()))
            .collect()
    }

    /// Get results by analyzer name
    pub fn get_by_analyzer(&self, name: &str) -> Vec<&AnalysisResult> {
        self.results
            .iter()
            .filter(|r| r.analyzer_name == name)
            .collect()
    }
}

impl Default for AnalysisReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics from an analysis report
#[derive(Debug, Clone, Default)]
pub struct AnalysisSummary {
    /// Average metrics across all analyses
    pub average_metrics: HashMap<String, f64>,

    /// Total number of analyses performed
    pub total_analyses: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_result() {
        let mut result = AnalysisResult::new("test_analyzer");
        result.add_metric("score", 0.85);
        result.add_insight("Test insight");

        assert_eq!(result.metrics.get("score"), Some(&0.85));
        assert_eq!(result.insights.len(), 1);
    }

    #[test]
    fn test_analysis_report() {
        let mut report = AnalysisReport::new();

        let mut result1 = AnalysisResult::new("analyzer1");
        result1.add_metric("metric1", 0.5);

        let mut result2 = AnalysisResult::new("analyzer2");
        result2.add_metric("metric1", 1.0);

        report.add_result(result1);
        report.add_result(result2);
        report.compute_summary();

        assert_eq!(report.summary.total_analyses, 2);
        assert_eq!(report.summary.average_metrics.get("metric1"), Some(&0.75));
    }
}
