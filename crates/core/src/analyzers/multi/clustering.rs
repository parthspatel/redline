//! Behavior-based clustering of diffs

use crate::analyzers::{AnalysisResult, MultiDiffAnalyzer};
use crate::diff::DiffResult;

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

    fn describe_cluster(&self, _diffs: &[&DiffResult], cluster_diffs: &[&DiffResult]) -> String {
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
