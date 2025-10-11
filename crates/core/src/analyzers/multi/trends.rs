//! Temporal trend analysis for diff sequences

use crate::analyzers::{AnalysisResult, MultiDiffAnalyzer};
use crate::diff::DiffResult;

/// Analyzes trends over time (requires diffs to be in chronological order)
#[derive(Clone)]
pub struct TemporalTrendAnalyzer;

impl TemporalTrendAnalyzer {
    pub fn new() -> Self {
        Self
    }

    fn moving_average(&self, values: &[f64], window: usize) -> Vec<f64> {
        let mut result = Vec::new();

        for i in 0..values.len() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2).min(values.len());
            let window_values = &values[start..end];
            let avg = window_values.iter().sum::<f64>() / window_values.len() as f64;
            result.push(avg);
        }

        result
    }

    fn detect_trend(&self, values: &[f64]) -> &'static str {
        if values.len() < 3 {
            return "insufficient_data";
        }

        // Simple linear regression
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2));

        if slope > 0.05 {
            "increasing"
        } else if slope < -0.05 {
            "decreasing"
        } else {
            "stable"
        }
    }
}

impl Default for TemporalTrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDiffAnalyzer for TemporalTrendAnalyzer {
    fn analyze(&self, diffs: &[&DiffResult]) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        if diffs.len() < 3 {
            result.add_insight("Need at least 3 diffs for trend analysis");
            return result;
        }

        // Analyze trends in key metrics
        let similarities: Vec<f64> = diffs.iter()
            .map(|d| d.semantic_similarity)
            .collect();
        let similarity_trend = self.detect_trend(&similarities);

        let changes: Vec<f64> = diffs.iter()
            .map(|d| d.statistics.change_percentage)
            .collect();
        let change_trend = self.detect_trend(&changes);

        result.add_metadata("similarity_trend", similarity_trend);
        result.add_metadata("change_magnitude_trend", change_trend);

        result.add_insight(format!(
            "Semantic similarity trend: {}",
            similarity_trend
        ));
        result.add_insight(format!(
            "Change magnitude trend: {}",
            change_trend
        ));

        // Compute moving averages
        let ma_window = (diffs.len() / 3).max(3);
        let similarity_ma = self.moving_average(&similarities, ma_window);
        let change_ma = self.moving_average(&changes, ma_window);

        result.add_metric("final_similarity_ma", *similarity_ma.last().unwrap_or(&0.0));
        result.add_metric("final_change_ma", *change_ma.last().unwrap_or(&0.0));

        // Detect significant changes
        if similarity_trend == "decreasing" {
            result.add_insight(
                "User edits are becoming increasingly substantial over time"
            );
        } else if similarity_trend == "increasing" {
            result.add_insight(
                "User edits are becoming more conservative over time"
            );
        }

        result
    }

    fn name(&self) -> &str {
        "temporal_trends"
    }

    fn description(&self) -> &str {
        "Analyzes trends in editing behavior over time"
    }

    fn clone_box(&self) -> Box<dyn MultiDiffAnalyzer> {
        Box::new(self.clone())
    }
}
