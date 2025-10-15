//! Aggregate statistics analysis across multiple diffs

use crate::analyzer::{AnalysisResult, MultiDiffAnalyzer};
use crate::diff::DiffResult;

/// Computes aggregate statistics across multiple diffs
#[derive(Clone)]
pub struct AggregateStatisticsAnalyzer;

impl AggregateStatisticsAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AggregateStatisticsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDiffAnalyzer for AggregateStatisticsAnalyzer {
    fn analyze(&self, diffs: &[&DiffResult]) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        if diffs.is_empty() {
            return result;
        }

        // Aggregate basic statistics
        let total_diffs = diffs.len();
        let total_insertions: usize = diffs.iter().map(|d| d.statistics.insertions).sum();
        let total_deletions: usize = diffs.iter().map(|d| d.statistics.deletions).sum();
        let total_modifications: usize = diffs.iter().map(|d| d.statistics.modifications).sum();

        let avg_insertions = total_insertions as f64 / total_diffs as f64;
        let avg_deletions = total_deletions as f64 / total_diffs as f64;
        let avg_modifications = total_modifications as f64 / total_diffs as f64;

        result.add_metric("total_diffs", total_diffs as f64);
        result.add_metric("avg_insertions", avg_insertions);
        result.add_metric("avg_deletions", avg_deletions);
        result.add_metric("avg_modifications", avg_modifications);
        result.add_metric(
            "total_changes",
            (total_insertions + total_deletions + total_modifications) as f64,
        );

        // Semantic similarity statistics
        let similarities: Vec<f64> = diffs.iter().map(|d| d.semantic_similarity).collect();
        let avg_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let max_similarity = similarities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_similarity = similarities.iter().cloned().fold(f64::INFINITY, f64::min);

        result.add_metric("avg_semantic_similarity", avg_similarity);
        result.add_metric("max_semantic_similarity", max_similarity);
        result.add_metric("min_semantic_similarity", min_similarity);

        // Change percentage statistics
        let change_pcts: Vec<f64> = diffs
            .iter()
            .map(|d| d.statistics.change_percentage)
            .collect();
        let avg_change_pct = change_pcts.iter().sum::<f64>() / change_pcts.len() as f64;

        result.add_metric("avg_change_percentage", avg_change_pct * 100.0);

        // Generate insights
        result.add_insight(format!(
            "Analyzed {} diffs with average of {:.1} insertions, {:.1} deletions, {:.1} modifications per diff",
            total_diffs, avg_insertions, avg_deletions, avg_modifications
        ));

        result.add_insight(format!(
            "Average semantic similarity: {:.1}% (range: {:.1}% to {:.1}%)",
            avg_similarity * 100.0,
            min_similarity * 100.0,
            max_similarity * 100.0
        ));

        if avg_similarity < 0.5 {
            result.add_insight("Low average similarity indicates substantial rewrites");
        } else if avg_similarity > 0.8 {
            result.add_insight("High average similarity indicates mostly refinements");
        }

        result
    }

    fn name(&self) -> &str {
        "aggregate_statistics"
    }

    fn description(&self) -> &str {
        "Computes aggregate statistics across multiple diffs"
    }

    fn clone_box(&self) -> Box<dyn MultiDiffAnalyzer> {
        Box::new(self.clone())
    }
}
