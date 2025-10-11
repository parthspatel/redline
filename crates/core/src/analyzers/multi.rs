//! Multi-diff analyzers for aggregate analysis
//!
//! Analyzers that operate on collections of diff results to find patterns and trends

use super::{AnalysisResult, MultiDiffAnalyzer};
use crate::diff::{DiffResult, EditType};
use std::collections::HashMap;

// ============================================================================
// Aggregate Statistics Analyzer
// ============================================================================

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
        result.add_metric("total_changes", (total_insertions + total_deletions + total_modifications) as f64);

        // Semantic similarity statistics
        let similarities: Vec<f64> = diffs.iter().map(|d| d.semantic_similarity).collect();
        let avg_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let max_similarity = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_similarity = similarities.iter().cloned().fold(f64::INFINITY, f64::min);
        
        result.add_metric("avg_semantic_similarity", avg_similarity);
        result.add_metric("max_semantic_similarity", max_similarity);
        result.add_metric("min_semantic_similarity", min_similarity);

        // Change percentage statistics
        let change_pcts: Vec<f64> = diffs.iter().map(|d| d.statistics.change_percentage).collect();
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

// ============================================================================
// Pattern Detection Analyzer
// ============================================================================

/// Detects common patterns across multiple diffs
#[derive(Clone)]
pub struct PatternDetectionAnalyzer {
    /// Minimum frequency for a pattern to be reported
    pub min_frequency: f64,
}

impl PatternDetectionAnalyzer {
    pub fn new() -> Self {
        Self {
            min_frequency: 0.3, // 30%
        }
    }

    pub fn with_min_frequency(mut self, frequency: f64) -> Self {
        self.min_frequency = frequency;
        self
    }
}

impl Default for PatternDetectionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDiffAnalyzer for PatternDetectionAnalyzer {
    fn analyze(&self, diffs: &[&DiffResult]) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        
        if diffs.is_empty() {
            return result;
        }

        let total = diffs.len() as f64;

        // Pattern 1: Dominant edit type
        let mut edit_type_counts: HashMap<EditType, usize> = HashMap::new();
        for diff in diffs {
            for op in &diff.operations {
                *edit_type_counts.entry(op.edit_type).or_insert(0) += 1;
            }
        }

        if let Some((dominant_type, count)) = edit_type_counts.iter()
            .filter(|(t, _)| **t != EditType::Equal)
            .max_by_key(|(_, count)| *count) {
            
            let frequency = *count as f64 / total;
            if frequency >= self.min_frequency {
                result.add_insight(format!(
                    "Pattern: {:?} edits are common ({:.1}% of operations)",
                    dominant_type, frequency * 100.0
                ));
            }
        }

        // Pattern 2: Dominant category
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        for diff in diffs {
            for op in &diff.operations {
                if op.edit_type != EditType::Equal {
                    let cat = format!("{:?}", op.category);
                    *category_counts.entry(cat).or_insert(0) += 1;
                }
            }
        }

        if let Some((dominant_cat, count)) = category_counts.iter()
            .max_by_key(|(_, count)| *count) {
            
            let frequency = *count as f64 / total;
            if frequency >= self.min_frequency {
                result.add_insight(format!(
                    "Pattern: {} changes are dominant ({:.1}% frequency)",
                    dominant_cat, frequency * 100.0
                ));
                result.add_metadata("dominant_category", dominant_cat);
            }
        }

        // Pattern 3: Consistent expansion or reduction
        let expansions = diffs.iter()
            .filter(|d| d.statistics.insertions > d.statistics.deletions)
            .count();
        let reductions = diffs.iter()
            .filter(|d| d.statistics.deletions > d.statistics.insertions)
            .count();

        let expansion_freq = expansions as f64 / total;
        let reduction_freq = reductions as f64 / total;

        if expansion_freq >= self.min_frequency {
            result.add_insight(format!(
                "Pattern: Consistent text expansion ({:.1}% of diffs)",
                expansion_freq * 100.0
            ));
            result.add_metric("expansion_pattern_frequency", expansion_freq);
        }

        if reduction_freq >= self.min_frequency {
            result.add_insight(format!(
                "Pattern: Consistent text reduction ({:.1}% of diffs)",
                reduction_freq * 100.0
            ));
            result.add_metric("reduction_pattern_frequency", reduction_freq);
        }

        // Pattern 4: Low vs High similarity clustering
        let low_similarity = diffs.iter()
            .filter(|d| d.semantic_similarity < 0.5)
            .count();
        let high_similarity = diffs.iter()
            .filter(|d| d.semantic_similarity > 0.8)
            .count();

        if low_similarity as f64 / total >= self.min_frequency {
            result.add_insight(format!(
                "Pattern: Frequent substantial rewrites ({:.1}% with <50% similarity)",
                (low_similarity as f64 / total) * 100.0
            ));
        }

        if high_similarity as f64 / total >= self.min_frequency {
            result.add_insight(format!(
                "Pattern: Frequent minor refinements ({:.1}% with >80% similarity)",
                (high_similarity as f64 / total) * 100.0
            ));
        }

        result
    }

    fn name(&self) -> &str {
        "pattern_detection"
    }

    fn description(&self) -> &str {
        "Detects common editing patterns across multiple diffs"
    }

    fn clone_box(&self) -> Box<dyn MultiDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Edit Behavior Clustering Analyzer
// ============================================================================

/// Clusters diffs by editing behavior patterns
#[derive(Clone)]
pub struct BehaviorClusteringAnalyzer {
    /// Number of clusters to create
    pub n_clusters: usize,
}

impl BehaviorClusteringAnalyzer {
    pub fn new(n_clusters: usize) -> Self {
        Self { n_clusters }
    }

    /// Extract feature vector from a diff for clustering
    fn extract_features(&self, diff: &DiffResult) -> Vec<f64> {
        vec![
            diff.semantic_similarity,
            diff.statistics.change_percentage,
            diff.statistics.insertions as f64 / diff.statistics.original_length.max(1) as f64,
            diff.statistics.deletions as f64 / diff.statistics.original_length.max(1) as f64,
            diff.statistics.modifications as f64 / diff.statistics.original_length.max(1) as f64,
            diff.analysis.stylistic_change,
            diff.analysis.readability_change,
        ]
    }

    /// Simple k-means clustering
    fn kmeans_cluster(&self, features: &[Vec<f64>]) -> Vec<usize> {
        if features.is_empty() {
            return vec![];
        }

        let n = features.len();
        let k = self.n_clusters.min(n);
        
        // Initialize centroids randomly
        let mut centroids: Vec<Vec<f64>> = features.iter()
            .take(k)
            .cloned()
            .collect();

        let mut assignments = vec![0; n];
        let max_iterations = 100;

        for _ in 0..max_iterations {
            // Assign points to nearest centroid
            let mut changed = false;
            for (i, point) in features.iter().enumerate() {
                let distances: Vec<f64> = centroids.iter()
                    .map(|centroid| self.euclidean_distance(point, centroid))
                    .collect();
                
                let new_assignment = distances.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if new_assignment != assignments[i] {
                    changed = true;
                    assignments[i] = new_assignment;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            for cluster_idx in 0..k {
                let cluster_points: Vec<&Vec<f64>> = features.iter()
                    .zip(&assignments)
                    .filter(|&(_, &a)| a == cluster_idx)
                    .map(|(p, _)| p)
                    .collect();

                if !cluster_points.is_empty() {
                    let dim = cluster_points[0].len();
                    centroids[cluster_idx] = (0..dim)
                        .map(|d| {
                            cluster_points.iter().map(|p| p[d]).sum::<f64>() 
                                / cluster_points.len() as f64
                        })
                        .collect();
                }
            }
        }

        assignments
    }

    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn describe_cluster(&self, diffs: &[&DiffResult], cluster_diffs: &[&DiffResult]) -> String {
        if cluster_diffs.is_empty() {
            return "Empty cluster".to_string();
        }

        let avg_similarity = cluster_diffs.iter()
            .map(|d| d.semantic_similarity)
            .sum::<f64>() / cluster_diffs.len() as f64;

        let avg_change = cluster_diffs.iter()
            .map(|d| d.statistics.change_percentage)
            .sum::<f64>() / cluster_diffs.len() as f64;

        // Determine dominant characteristics
        if avg_similarity > 0.8 && avg_change < 0.2 {
            "Minor refinements".to_string()
        } else if avg_similarity > 0.6 {
            "Moderate edits".to_string()
        } else if avg_similarity < 0.5 {
            "Substantial rewrites".to_string()
        } else {
            "Mixed editing pattern".to_string()
        }
    }
}

impl Default for BehaviorClusteringAnalyzer {
    fn default() -> Self {
        Self::new(3)
    }
}

impl MultiDiffAnalyzer for BehaviorClusteringAnalyzer {
    fn analyze(&self, diffs: &[&DiffResult]) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        
        if diffs.len() < self.n_clusters {
            result.add_insight("Too few diffs for meaningful clustering");
            return result;
        }

        // Extract features
        let features: Vec<Vec<f64>> = diffs.iter()
            .map(|d| self.extract_features(d))
            .collect();

        // Cluster
        let assignments = self.kmeans_cluster(&features);

        // Analyze clusters
        for cluster_id in 0..self.n_clusters {
            let cluster_diffs: Vec<&DiffResult> = diffs.iter()
                .zip(&assignments)
                .filter(|&(_, &a)| a == cluster_id)
                .map(|(d, _)| *d)
                .collect();

            let cluster_size = cluster_diffs.len();
            let cluster_pct = cluster_size as f64 / diffs.len() as f64;

            result.add_metric(
                format!("cluster_{}_size", cluster_id),
                cluster_size as f64
            );
            result.add_metric(
                format!("cluster_{}_percentage", cluster_id),
                cluster_pct * 100.0
            );

            let description = self.describe_cluster(diffs, &cluster_diffs);
            result.add_insight(format!(
                "Cluster {}: {} ({:.1}% of diffs) - {}",
                cluster_id, cluster_size, cluster_pct * 100.0, description
            ));
        }

        result
    }

    fn name(&self) -> &str {
        "behavior_clustering"
    }

    fn description(&self) -> &str {
        "Clusters diffs by editing behavior using k-means"
    }

    fn clone_box(&self) -> Box<dyn MultiDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Temporal Trend Analyzer
// ============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiffEngine;

    #[test]
    fn test_aggregate_statistics() {
        let engine = DiffEngine::default();
        let diff1 = engine.diff("hello world", "hello rust");
        let diff2 = engine.diff("foo bar", "foo baz");
        
        let diffs = vec![&diff1, &diff2];
        
        let analyzer = AggregateStatisticsAnalyzer::new();
        let result = analyzer.analyze(&diffs);
        
        assert!(result.metrics.contains_key("total_diffs"));
        assert_eq!(result.metrics["total_diffs"], 2.0);
    }

    #[test]
    fn test_pattern_detection() {
        let engine = DiffEngine::default();
        let diff1 = engine.diff("a", "abc");  // Expansion
        let diff2 = engine.diff("x", "xyz");  // Expansion
        let diff3 = engine.diff("m", "mno");  // Expansion
        
        let diffs = vec![&diff1, &diff2, &diff3];
        
        let analyzer = PatternDetectionAnalyzer::new();
        let result = analyzer.analyze(&diffs);
        
        // Should detect expansion pattern
        assert!(result.insights.iter().any(|i| i.contains("expansion")));
    }
}
