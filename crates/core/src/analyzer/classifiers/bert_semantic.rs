//! BERT-based semantic similarity analyzer with unsupervised threshold learning
//!
//! This module provides advanced semantic analysis using BERT embeddings and
//! automatic threshold determination through unsupervised learning.

#[cfg(feature = "bert")]
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use std::sync::Arc;

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

/// BERT-based semantic similarity analyzer with adaptive thresholds
#[derive(Clone)]
pub struct BertSemanticAnalyzer {
    /// The BERT model to use
    #[cfg(feature = "bert")]
    model_type: Arc<SentenceEmbeddingsModelType>,

    /// Learned threshold for similarity classification
    threshold: Option<f64>,

    /// Historical similarity scores for threshold learning
    similarity_history: Vec<f64>,

    /// Maximum history size for threshold learning
    max_history_size: usize,

    /// Whether to enable automatic threshold learning
    auto_threshold: bool,

    /// Confidence percentile for threshold (e.g., 0.5 = median)
    threshold_percentile: f64,
}

impl BertSemanticAnalyzer {
    /// Create a new BERT semantic analyzer with default settings
    #[cfg(feature = "bert")]
    pub fn new() -> Self {
        Self {
            model_type: Arc::new(SentenceEmbeddingsModelType::AllMiniLmL6V2),
            threshold: None,
            similarity_history: Vec::new(),
            max_history_size: 1000,
            auto_threshold: true,
            threshold_percentile: 0.5,
        }
    }

    /// Create without BERT feature (fallback to basic similarity)
    #[cfg(not(feature = "bert"))]
    pub fn new() -> Self {
        Self {
            threshold: None,
            similarity_history: Vec::new(),
            max_history_size: 1000,
            auto_threshold: true,
            threshold_percentile: 0.5,
        }
    }

    /// Set the BERT model type
    #[cfg(feature = "bert")]
    pub fn with_model(mut self, model_type: SentenceEmbeddingsModelType) -> Self {
        self.model_type = Arc::from(model_type);
        self
    }

    /// Set a manual threshold (disables auto-learning)
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self.auto_threshold = false;
        self
    }

    /// Enable automatic threshold learning
    pub fn with_auto_threshold(mut self, enable: bool) -> Self {
        self.auto_threshold = enable;
        if !enable && self.threshold.is_none() {
            self.threshold = Some(0.7); // Default fallback
        }
        self
    }

    /// Set the percentile for threshold learning (0.0 to 1.0)
    pub fn with_threshold_percentile(mut self, percentile: f64) -> Self {
        self.threshold_percentile = percentile.clamp(0.0, 1.0);
        self
    }

    /// Set maximum history size for threshold learning
    pub fn with_max_history(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Compute BERT embeddings and cosine similarity
    #[cfg(feature = "bert")]
    fn compute_bert_similarity(&self, text1: &str, text2: &str) -> Result<f64, String> {
        // Initialize the model
        let model = SentenceEmbeddingsBuilder::remote(*self.model_type.clone())
            .create_model()
            .map_err(|e| format!("Failed to create BERT model: {}", e))?;

        // Generate embeddings
        let embeddings = model
            .encode(&[text1, text2])
            .map_err(|e| format!("Failed to encode texts: {}", e))?;

        if embeddings.len() < 2 {
            return Err("Failed to generate embeddings".to_string());
        }

        // Compute cosine similarity
        let similarity = Self::cosine_similarity(&embeddings[0], &embeddings[1]);
        Ok(similarity)
    }

    /// Fallback similarity when BERT is not available
    #[cfg(not(feature = "bert"))]
    fn compute_bert_similarity(&self, text1: &str, text2: &str) -> Result<f64, String> {
        // Fallback to word-level Jaccard similarity
        Ok(self.word_jaccard_similarity(text1, text2))
    }

    /// Compute cosine similarity between two vectors
    #[cfg(feature = "bert")]
    fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f64 {
        let dot_product: f32 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
        let magnitude1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            return 0.0;
        }

        (dot_product / (magnitude1 * magnitude2)) as f64
    }

    /// Fallback word-level similarity
    fn word_jaccard_similarity(&self, text1: &str, text2: &str) -> f64 {
        let words1: std::collections::HashSet<_> =
            text1.split_whitespace().map(|w| w.to_lowercase()).collect();

        let words2: std::collections::HashSet<_> =
            text2.split_whitespace().map(|w| w.to_lowercase()).collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Add a similarity score to history for threshold learning
    fn add_to_history(&mut self, similarity: f64) {
        if !self.auto_threshold {
            return;
        }

        self.similarity_history.push(similarity);

        // Maintain max history size
        if self.similarity_history.len() > self.max_history_size {
            self.similarity_history.remove(0);
        }
    }

    /// Learn threshold from similarity history using unsupervised methods
    pub fn learn_threshold(&mut self) -> Option<f64> {
        if self.similarity_history.len() < 10 {
            // Need at least 10 samples for meaningful threshold
            return self.threshold;
        }

        // Method 1: Percentile-based threshold
        let threshold = self.compute_percentile_threshold();

        // Method 2: Gaussian Mixture Model (if enough samples)
        let gmm_threshold = if self.similarity_history.len() >= 50 {
            self.compute_gmm_threshold()
        } else {
            None
        };

        // Method 3: Elbow method on sorted similarities
        let elbow_threshold = if self.similarity_history.len() >= 30 {
            self.compute_elbow_threshold()
        } else {
            None
        };

        // Combine methods (weighted average if multiple available)
        let learned_threshold = match (gmm_threshold, elbow_threshold) {
            (Some(gmm), Some(elbow)) => {
                // Weighted average: GMM 40%, Elbow 40%, Percentile 20%
                gmm * 0.4 + elbow * 0.4 + threshold * 0.2
            }
            (Some(gmm), None) => {
                // GMM 60%, Percentile 40%
                gmm * 0.6 + threshold * 0.4
            }
            (None, Some(elbow)) => {
                // Elbow 60%, Percentile 40%
                elbow * 0.6 + threshold * 0.4
            }
            (None, None) => threshold,
        };

        self.threshold = Some(learned_threshold);
        self.threshold
    }

    /// Compute threshold using percentile method
    fn compute_percentile_threshold(&self) -> f64 {
        let mut sorted = self.similarity_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (sorted.len() as f64 * self.threshold_percentile) as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Compute threshold using Gaussian Mixture Model
    /// Assumes bimodal distribution: similar (high scores) vs dissimilar (low scores)
    fn compute_gmm_threshold(&self) -> Option<f64> {
        // Simple 2-component GMM approximation
        let mean =
            self.similarity_history.iter().sum::<f64>() / self.similarity_history.len() as f64;

        // Split into two groups around mean
        let low_scores: Vec<f64> = self
            .similarity_history
            .iter()
            .filter(|&&x| x < mean)
            .copied()
            .collect();

        let high_scores: Vec<f64> = self
            .similarity_history
            .iter()
            .filter(|&&x| x >= mean)
            .copied()
            .collect();

        if low_scores.is_empty() || high_scores.is_empty() {
            return None;
        }

        // Compute means of each group
        let low_mean = low_scores.iter().sum::<f64>() / low_scores.len() as f64;
        let high_mean = high_scores.iter().sum::<f64>() / high_scores.len() as f64;

        // Compute standard deviations
        let low_var = low_scores
            .iter()
            .map(|x| (x - low_mean).powi(2))
            .sum::<f64>()
            / low_scores.len() as f64;
        let low_std = low_var.sqrt();

        let high_var = high_scores
            .iter()
            .map(|x| (x - high_mean).powi(2))
            .sum::<f64>()
            / high_scores.len() as f64;
        let high_std = high_var.sqrt();

        // Threshold is where the two distributions are most likely to intersect
        // For Gaussian distributions: (mean1 + mean2) / 2 is a reasonable approximation
        // Adjust based on relative sizes and spread
        let weight_low = low_scores.len() as f64 / self.similarity_history.len() as f64;
        let weight_high = high_scores.len() as f64 / self.similarity_history.len() as f64;

        // Weighted midpoint, adjusted by standard deviations
        let threshold =
            (low_mean * weight_high + high_mean * weight_low) + (high_std - low_std) * 0.1; // Small adjustment based on spread

        Some(threshold.clamp(0.0, 1.0))
    }

    /// Compute threshold using elbow method on sorted similarities
    fn compute_elbow_threshold(&self) -> Option<f64> {
        let mut sorted = self.similarity_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute second derivative (curvature) to find elbow
        if sorted.len() < 3 {
            return None;
        }

        let mut max_curvature = 0.0;
        let mut elbow_idx = sorted.len() / 2;

        for i in 1..sorted.len() - 1 {
            // Second derivative approximation
            let d2 = sorted[i + 1] - 2.0 * sorted[i] + sorted[i - 1];
            let curvature = d2.abs();

            if curvature > max_curvature {
                max_curvature = curvature;
                elbow_idx = i;
            }
        }

        Some(sorted[elbow_idx])
    }

    /// Get the current threshold (learned or manual)
    pub fn get_threshold(&self) -> f64 {
        self.threshold.unwrap_or(0.7)
    }

    /// Get statistics about threshold learning
    pub fn get_learning_stats(&self) -> ThresholdLearningStats {
        let mut sorted = self.similarity_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = if sorted.is_empty() {
            0.0
        } else {
            sorted.iter().sum::<f64>() / sorted.len() as f64
        };

        let median = if sorted.is_empty() {
            0.0
        } else {
            sorted[sorted.len() / 2]
        };

        let variance = if sorted.is_empty() {
            0.0
        } else {
            sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sorted.len() as f64
        };

        let std_dev = variance.sqrt();

        ThresholdLearningStats {
            sample_count: self.similarity_history.len(),
            mean,
            median,
            std_dev,
            min: sorted.first().copied().unwrap_or(0.0),
            max: sorted.last().copied().unwrap_or(0.0),
            current_threshold: self.threshold,
            auto_learning_enabled: self.auto_threshold,
        }
    }

    /// Reset learning history
    pub fn reset_learning(&mut self) {
        self.similarity_history.clear();
        if self.auto_threshold {
            self.threshold = None;
        }
    }

    /// Export similarity history for external analysis
    pub fn export_history(&self) -> Vec<f64> {
        self.similarity_history.clone()
    }

    /// Import similarity history (e.g., from previous runs)
    pub fn import_history(&mut self, history: Vec<f64>) {
        self.similarity_history = history;
        if self.similarity_history.len() > self.max_history_size {
            self.similarity_history = self
                .similarity_history
                .iter()
                .rev()
                .take(self.max_history_size)
                .rev()
                .copied()
                .collect();
        }

        if self.auto_threshold {
            self.learn_threshold();
        }
    }
}

impl Default for BertSemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for BertSemanticAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Compute BERT-based similarity
        let similarity =
            match self.compute_bert_similarity(&diff.original_text, &diff.modified_text) {
                Ok(sim) => sim,
                Err(e) => {
                    result.add_insight(format!("Error computing similarity: {}", e));
                    // Fall back to basic similarity from diff
                    diff.semantic_similarity
                }
            };

        // Add to learning history (mutable operation needs special handling)
        // Note: In real usage, you'd need to make the analyzer mutable or use interior mutability
        // For this demo, we'll just work with the computed similarity

        result.add_metric("bert_semantic_similarity", similarity);
        result.add_metric("bert_semantic_distance", 1.0 - similarity);

        // Get threshold for classification
        let threshold = self.get_threshold();
        result.add_metric("learned_threshold", threshold);

        // Classify similarity level
        let level = if similarity >= threshold + 0.15 {
            "very_high"
        } else if similarity >= threshold {
            "high"
        } else if similarity >= threshold - 0.15 {
            "moderate"
        } else if similarity >= threshold - 0.30 {
            "low"
        } else {
            "very_low"
        };

        result.add_metadata("similarity_level", level);

        // Generate insights
        #[cfg(feature = "bert")]
        result.add_insight("Using BERT embeddings for semantic similarity".to_string());

        #[cfg(not(feature = "bert"))]
        result.add_insight(
            "Using fallback similarity (enable 'bert' feature for BERT embeddings)".to_string(),
        );

        if similarity >= threshold {
            result.add_insight(format!(
                "High semantic similarity ({:.1}% ≥ {:.1}% threshold). Meaning preserved.",
                similarity * 100.0,
                threshold * 100.0
            ));
        } else {
            result.add_insight(format!(
                "Low semantic similarity ({:.1}% < {:.1}% threshold). Substantial semantic changes.",
                similarity * 100.0, threshold * 100.0
            ));
        }

        // Add learning stats if available
        if self.similarity_history.len() >= 10 {
            result.add_metadata(
                "learning_samples",
                self.similarity_history.len().to_string(),
            );
            result.add_insight(format!(
                "Threshold learned from {} samples",
                self.similarity_history.len()
            ));
        }

        result
    }

    fn name(&self) -> &str {
        "bert_semantic"
    }

    fn description(&self) -> &str {
        "BERT-based semantic similarity with unsupervised threshold learning"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

/// Statistics about threshold learning
#[derive(Debug, Clone)]
pub struct ThresholdLearningStats {
    pub sample_count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub current_threshold: Option<f64>,
    pub auto_learning_enabled: bool,
}

impl ThresholdLearningStats {
    pub fn summary(&self) -> String {
        format!(
            "Learning Stats: {} samples, mean={:.3}, median={:.3}, std={:.3}, threshold={:.3}",
            self.sample_count,
            self.mean,
            self.median,
            self.std_dev,
            self.current_threshold.unwrap_or(0.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiffEngine;

    #[test]
    fn test_basic_similarity() {
        let analyzer = BertSemanticAnalyzer::new();
        let engine = DiffEngine::default();
        let diff = engine.diff("hello world", "hello rust");

        let result = analyzer.analyze(&diff);
        assert!(result.metrics.contains_key("bert_semantic_similarity"));
    }

    #[test]
    fn test_threshold_learning() {
        let mut analyzer = BertSemanticAnalyzer::new()
            .with_auto_threshold(true)
            .with_threshold_percentile(0.5);

        // Add some sample similarities
        for _ in 0..20 {
            analyzer.add_to_history(0.3);
        }
        for _ in 0..20 {
            analyzer.add_to_history(0.8);
        }

        // Learn threshold
        let threshold = analyzer.learn_threshold();
        assert!(threshold.is_some());
        assert!(threshold.unwrap() > 0.3);
        assert!(threshold.unwrap() < 0.8);
    }

    #[test]
    fn test_learning_stats() {
        let mut analyzer = BertSemanticAnalyzer::new();

        analyzer.add_to_history(0.5);
        analyzer.add_to_history(0.7);
        analyzer.add_to_history(0.6);

        let stats = analyzer.get_learning_stats();
        assert_eq!(stats.sample_count, 3);
        assert!(stats.mean > 0.5 && stats.mean < 0.7);
    }

    #[test]
    fn test_history_export_import() {
        let mut analyzer1 = BertSemanticAnalyzer::new();
        analyzer1.add_to_history(0.5);
        analyzer1.add_to_history(0.8);

        let history = analyzer1.export_history();

        let mut analyzer2 = BertSemanticAnalyzer::new();
        analyzer2.import_history(history);

        assert_eq!(analyzer2.similarity_history.len(), 2);
    }
}
