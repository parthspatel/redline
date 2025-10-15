//! BERT-based semantic similarity analyzer with unsupervised threshold learning
//!
//! This module provides advanced semantic analysis using BERT embeddings and
//! automatic threshold determination through unsupervised learning.
//!
//! # Architecture
//!
//! The analyzer is split into two phases:
//! 1. **Offline Learning**: Train threshold and statistical parameters on historical data
//! 2. **Online Inference**: Use pre-trained parameters for fast runtime analysis

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;
#[cfg(feature = "bert")]
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};

/// Learned parameters from offline training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedParameters {
    /// Learned threshold for similarity classification
    pub threshold: f64,

    /// Statistical parameters from training data
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,

    /// Number of samples used for learning
    pub sample_count: usize,
}

impl LearnedParameters {
    /// Create default parameters (useful when no training data available)
    pub fn default_params() -> Self {
        Self {
            threshold: 0.7,
            mean: 0.7,
            std_dev: 0.15,
            median: 0.7,
            min: 0.0,
            max: 1.0,
            sample_count: 0,
        }
    }

    /// Serialize to JSON string
    #[cfg(feature = "bert")]
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("Failed to serialize parameters: {}", e))
    }

    /// Deserialize from JSON string
    #[cfg(feature = "bert")]
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to deserialize parameters: {}", e))
    }
}

/// Configuration for inference-time behavior
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Z-score threshold for outlier detection (default: 2.5)
    pub outlier_z_threshold: f64,

    /// Standard deviation multiplier for drift detection (default: 2.0)
    pub drift_std_multiplier: f64,

    /// Number of histogram bins for entropy calculation (default: 10)
    pub entropy_bins: usize,

    /// Whether to include all metrics in analysis (default: true)
    pub include_all_metrics: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            outlier_z_threshold: 2.5,
            drift_std_multiplier: 2.0,
            entropy_bins: 10,
            include_all_metrics: true,
        }
    }
}

/// Offline trainer for learning parameters from historical similarity data
pub struct BertSemanticTrainer {
    /// Historical similarity scores
    similarity_history: Vec<f64>,

    /// Maximum history size
    max_history_size: usize,

    /// Percentile for threshold (e.g., 0.5 = median)
    threshold_percentile: f64,
}

impl BertSemanticTrainer {
    /// Create a new trainer
    pub fn new() -> Self {
        Self {
            similarity_history: Vec::new(),
            max_history_size: 10000,
            threshold_percentile: 0.5,
        }
    }

    /// Set the percentile for threshold learning
    pub fn with_threshold_percentile(mut self, percentile: f64) -> Self {
        self.threshold_percentile = percentile.clamp(0.0, 1.0);
        self
    }

    /// Set maximum history size
    pub fn with_max_history(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Add a similarity score to the training data
    pub fn add_sample(&mut self, similarity: f64) {
        self.similarity_history.push(similarity);

        // Maintain max history size
        if self.similarity_history.len() > self.max_history_size {
            self.similarity_history.remove(0);
        }
    }

    /// Add multiple samples at once
    pub fn add_samples(&mut self, similarities: &[f64]) {
        for &sim in similarities {
            self.add_sample(sim);
        }
    }

    /// Train and produce learned parameters
    pub fn train(&self) -> Result<LearnedParameters, String> {
        if self.similarity_history.len() < 10 {
            return Err(format!(
                "Need at least 10 samples for training, got {}",
                self.similarity_history.len()
            ));
        }

        let mut sorted = self.similarity_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute statistics
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
        let std_dev = variance.sqrt();

        // Learn threshold using multiple methods
        let threshold = self.learn_threshold_multi_method(&sorted, mean, std_dev)?;

        Ok(LearnedParameters {
            threshold,
            mean,
            std_dev,
            median,
            min,
            max,
            sample_count: self.similarity_history.len(),
        })
    }

    /// Learn threshold using multiple methods and combine them
    fn learn_threshold_multi_method(
        &self,
        sorted: &[f64],
        mean: f64,
        std_dev: f64,
    ) -> Result<f64, String> {
        // Method 1: Percentile-based
        let percentile_threshold = self.compute_percentile_threshold(sorted);

        // Method 2: GMM-based (if enough samples)
        let gmm_threshold = if sorted.len() >= 50 {
            self.compute_gmm_threshold(sorted, mean, std_dev)
        } else {
            None
        };

        // Method 3: Elbow method (if enough samples)
        let elbow_threshold = if sorted.len() >= 30 {
            self.compute_elbow_threshold(sorted)
        } else {
            None
        };

        // Combine methods with weighted average
        let threshold = match (gmm_threshold, elbow_threshold) {
            (Some(gmm), Some(elbow)) => gmm * 0.4 + elbow * 0.4 + percentile_threshold * 0.2,
            (Some(gmm), None) => gmm * 0.6 + percentile_threshold * 0.4,
            (None, Some(elbow)) => elbow * 0.6 + percentile_threshold * 0.4,
            (None, None) => percentile_threshold,
        };

        Ok(threshold)
    }

    /// Compute percentile-based threshold
    fn compute_percentile_threshold(&self, sorted: &[f64]) -> f64 {
        let index = (sorted.len() as f64 * self.threshold_percentile) as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Compute GMM-based threshold
    fn compute_gmm_threshold(&self, sorted: &[f64], mean: f64, _std_dev: f64) -> Option<f64> {
        let low_scores: Vec<f64> = sorted.iter().filter(|&&x| x < mean).copied().collect();
        let high_scores: Vec<f64> = sorted.iter().filter(|&&x| x >= mean).copied().collect();

        if low_scores.is_empty() || high_scores.is_empty() {
            return None;
        }

        let low_mean = low_scores.iter().sum::<f64>() / low_scores.len() as f64;
        let high_mean = high_scores.iter().sum::<f64>() / high_scores.len() as f64;

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

        let weight_low = low_scores.len() as f64 / sorted.len() as f64;
        let weight_high = high_scores.len() as f64 / sorted.len() as f64;

        let threshold =
            (low_mean * weight_high + high_mean * weight_low) + (high_std - low_std) * 0.1;

        Some(threshold.clamp(0.0, 1.0))
    }

    /// Compute elbow-based threshold
    fn compute_elbow_threshold(&self, sorted: &[f64]) -> Option<f64> {
        if sorted.len() < 3 {
            return None;
        }

        let mut max_curvature = 0.0;
        let mut elbow_idx = sorted.len() / 2;

        for i in 1..sorted.len() - 1 {
            let d2 = sorted[i + 1] - 2.0 * sorted[i] + sorted[i - 1];
            let curvature = d2.abs();

            if curvature > max_curvature {
                max_curvature = curvature;
                elbow_idx = i;
            }
        }

        Some(sorted[elbow_idx])
    }

    /// Get current sample count
    pub fn sample_count(&self) -> usize {
        self.similarity_history.len()
    }

    /// Clear all training data
    pub fn clear(&mut self) {
        self.similarity_history.clear();
    }
}

impl Default for BertSemanticTrainer {
    fn default() -> Self {
        Self::new()
    }
}

/// BERT-based semantic similarity analyzer for runtime inference
pub struct BertSemanticAnalyzer {
    /// The BERT model to use
    #[cfg(feature = "bert")]
    model_type: SentenceEmbeddingsModelType,

    /// Pre-learned parameters from offline training
    learned_params: LearnedParameters,

    /// Configuration for inference-time behavior
    config: InferenceConfig,
}

#[cfg(feature = "bert")]
fn copy_model_type(model_type: &SentenceEmbeddingsModelType) -> SentenceEmbeddingsModelType {
    match model_type {
        SentenceEmbeddingsModelType::DistiluseBaseMultilingualCased => {
            SentenceEmbeddingsModelType::DistiluseBaseMultilingualCased
        }
        SentenceEmbeddingsModelType::BertBaseNliMeanTokens => {
            SentenceEmbeddingsModelType::BertBaseNliMeanTokens
        }
        SentenceEmbeddingsModelType::AllMiniLmL12V2 => SentenceEmbeddingsModelType::AllMiniLmL12V2,
        SentenceEmbeddingsModelType::AllMiniLmL6V2 => SentenceEmbeddingsModelType::AllMiniLmL6V2,
        SentenceEmbeddingsModelType::AllDistilrobertaV1 => {
            SentenceEmbeddingsModelType::AllDistilrobertaV1
        }
        SentenceEmbeddingsModelType::ParaphraseAlbertSmallV2 => {
            SentenceEmbeddingsModelType::ParaphraseAlbertSmallV2
        }
        SentenceEmbeddingsModelType::SentenceT5Base => SentenceEmbeddingsModelType::SentenceT5Base,
    }
}

impl Clone for BertSemanticAnalyzer {
    fn clone(&self) -> Self {
        Self {
            #[cfg(feature = "bert")]
            model_type: copy_model_type(&self.model_type),
            learned_params: self.learned_params.clone(),
            config: self.config.clone(),
        }
    }
}

impl BertSemanticAnalyzer {
    /// Create a new analyzer with pre-learned parameters
    #[cfg(feature = "bert")]
    pub fn new(learned_params: LearnedParameters) -> Self {
        Self {
            model_type: SentenceEmbeddingsModelType::AllMiniLmL6V2,
            learned_params,
            config: InferenceConfig::default(),
        }
    }

    /// Create without BERT feature (fallback to basic similarity)
    #[cfg(not(feature = "bert"))]
    pub fn new(learned_params: LearnedParameters) -> Self {
        Self {
            learned_params,
            config: InferenceConfig::default(),
        }
    }

    /// Create with default parameters (no prior training)
    #[cfg(feature = "bert")]
    pub fn with_defaults() -> Self {
        Self::new(LearnedParameters::default_params())
    }

    /// Create with default parameters (no prior training)
    #[cfg(not(feature = "bert"))]
    pub fn with_defaults() -> Self {
        Self::new(LearnedParameters::default_params())
    }

    /// Set custom inference configuration
    pub fn with_config(mut self, config: InferenceConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the BERT model type
    #[cfg(feature = "bert")]
    pub fn with_model(mut self, model_type: SentenceEmbeddingsModelType) -> Self {
        self.model_type = model_type;
        self
    }

    // OLD BUILDER METHODS - DEPRECATED
    // These are kept for backwards compatibility but should not be used
    // Use BertSemanticTrainer for offline learning instead

    /// Compute BERT embeddings and cosine similarity
    #[cfg(feature = "bert")]
    fn compute_bert_similarity(&self, text1: &str, text2: &str) -> Result<f64, String> {
        // Initialize the model
        let model = SentenceEmbeddingsBuilder::remote(copy_model_type(&self.model_type))
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

    /// Get the learned threshold
    pub fn get_threshold(&self) -> f64 {
        self.learned_params.threshold
    }

    /// Get learned parameters
    pub fn get_learned_params(&self) -> &LearnedParameters {
        &self.learned_params
    }

    /// Compute confidence score using learned distribution parameters
    pub fn compute_confidence(&self, similarity: f64) -> f64 {
        let distance_from_mean = (similarity - self.learned_params.mean).abs();
        let normalized_distance = if self.learned_params.std_dev > 0.0 {
            distance_from_mean / self.learned_params.std_dev
        } else {
            0.0
        };

        let confidence = 1.0 / (1.0 + normalized_distance);
        confidence.clamp(0.0, 1.0)
    }

    /// Detect semantic drift using learned parameters
    pub fn detect_semantic_drift(&self, similarity: f64) -> (bool, f64) {
        let drift_threshold = self.learned_params.mean
            - (self.config.drift_std_multiplier * self.learned_params.std_dev);
        let is_drift = similarity < drift_threshold;

        let drift_magnitude = if is_drift {
            ((drift_threshold - similarity) / self.learned_params.std_dev).abs()
        } else {
            0.0
        };

        (is_drift, drift_magnitude)
    }

    /// Detect outliers using learned parameters
    pub fn detect_outlier(&self, similarity: f64) -> (bool, f64) {
        let z_score = if self.learned_params.std_dev > 0.0 {
            (similarity - self.learned_params.mean) / self.learned_params.std_dev
        } else {
            0.0
        };

        let is_outlier = z_score.abs() > self.config.outlier_z_threshold;

        (is_outlier, z_score)
    }

    /// Get comprehensive metrics for a similarity value
    pub fn get_similarity_metrics(&self, similarity: f64) -> SimilarityMetrics {
        let confidence = self.compute_confidence(similarity);
        let (is_drift, drift_magnitude) = self.detect_semantic_drift(similarity);
        let (is_outlier, z_score) = self.detect_outlier(similarity);

        SimilarityMetrics {
            similarity,
            confidence,
            is_drift,
            drift_magnitude,
            is_outlier,
            z_score,
        }
    }
}

impl Default for BertSemanticAnalyzer {
    fn default() -> Self {
        Self::with_defaults()
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

        // Add comprehensive metrics
        let metrics = self.get_similarity_metrics(similarity);
        result.add_metric("confidence_score", metrics.confidence);
        result.add_metric("z_score", metrics.z_score);
        result.add_metric("drift_magnitude", metrics.drift_magnitude);

        // Add learned parameters info
        result.add_metric("learned_mean", self.learned_params.mean);
        result.add_metric("learned_std_dev", self.learned_params.std_dev);

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

        // Add flags for outliers and drift
        if metrics.is_outlier {
            result.add_metadata("outlier_detected", "true");
        }
        if metrics.is_drift {
            result.add_metadata("semantic_drift_detected", "true");
        }

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
        if self.learned_params.sample_count >= 10 {
            result.add_metadata(
                "learning_samples",
                self.learned_params.sample_count.to_string(),
            );
            result.add_insight(format!(
                "Threshold learned from {} samples",
                self.learned_params.sample_count
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

/// Comprehensive similarity metrics
#[derive(Debug, Clone)]
pub struct SimilarityMetrics {
    pub similarity: f64,
    pub confidence: f64,
    pub is_drift: bool,
    pub drift_magnitude: f64,
    pub is_outlier: bool,
    pub z_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiffEngine;

    #[test]
    fn test_basic_similarity() {
        let analyzer = BertSemanticAnalyzer::with_defaults();
        let engine = DiffEngine::default();
        let diff = engine.diff("hello world", "hello rust");

        let result = analyzer.analyze(&diff);
        assert!(result.metrics.contains_key("bert_semantic_similarity"));
        assert!(result.metrics.contains_key("bert_semantic_distance"));
        assert!(result.metrics.contains_key("confidence_score"));
    }

    #[test]
    fn test_threshold_learning() {
        // Use trainer for offline learning
        let mut trainer = BertSemanticTrainer::new().with_threshold_percentile(0.5);

        // Add some sample similarities - bimodal distribution
        for _ in 0..20 {
            trainer.add_sample(0.3);
        }
        for _ in 0..20 {
            trainer.add_sample(0.8);
        }

        // Learn threshold
        let learned = trainer.train().unwrap();
        assert!(learned.threshold > 0.3);
        assert!(learned.threshold < 0.8);
    }

    #[test]
    fn test_learned_parameters() {
        let mut trainer = BertSemanticTrainer::new();

        trainer.add_sample(0.5);
        trainer.add_sample(0.7);
        trainer.add_sample(0.6);

        // Not enough samples yet
        assert!(trainer.train().is_err());

        // Add more samples
        for i in 0..10 {
            trainer.add_sample(0.5 + (i as f64 * 0.02));
        }

        let learned = trainer.train().unwrap();
        assert_eq!(learned.sample_count, 13);
        assert!(learned.mean > 0.5 && learned.mean < 0.7);
        assert!(learned.median >= 0.5 && learned.median <= 0.7);
    }

    #[test]
    fn test_confidence_score() {
        // Train with normal distribution around 0.6
        let mut trainer = BertSemanticTrainer::new();
        for i in 0..100 {
            let value = 0.6 + (i as f64 - 50.0) * 0.004; // Range: 0.4 to 0.8
            trainer.add_sample(value.clamp(0.0, 1.0));
        }

        let learned = trainer.train().unwrap();
        let analyzer = BertSemanticAnalyzer::new(learned);

        // Test confidence at mean - should be very high (close to 1.0)
        let conf_at_mean = analyzer.compute_confidence(0.6);
        assert!(
            conf_at_mean > 0.9,
            "Confidence at mean should be > 0.9, got {}",
            conf_at_mean
        );

        // Test confidence 1 std dev away - should be moderate
        let conf_1std = analyzer.compute_confidence(0.7);
        assert!(
            conf_1std > 0.4 && conf_1std < 0.6,
            "Confidence 1 std dev from mean should be moderate (0.4-0.6), got {}",
            conf_1std
        );

        // Test confidence far from mean (3+ std devs) - should be low
        let conf_far = analyzer.compute_confidence(0.2);
        assert!(
            conf_far < 0.3,
            "Confidence far from mean should be < 0.3, got {}",
            conf_far
        );

        // Verify ordering
        assert!(
            conf_at_mean > conf_1std && conf_1std > conf_far,
            "Confidence should decrease with distance from mean"
        );
    }

    #[test]
    fn test_confidence_with_default_params() {
        let analyzer = BertSemanticAnalyzer::with_defaults();
        let confidence = analyzer.compute_confidence(0.7);
        // With default params (mean=0.7, std_dev=0.15), confidence at mean should be high
        assert!(confidence > 0.9, "Confidence at mean should be high");
    }

    #[test]
    fn test_semantic_drift_detection() {
        // Train with high similarities
        let mut trainer = BertSemanticTrainer::new();
        for i in 0..20 {
            trainer.add_sample(0.75 + (i as f64 - 10.0) * 0.01);
        }

        let learned = trainer.train().unwrap();
        let analyzer = BertSemanticAnalyzer::new(learned);

        // Test normal value - no drift
        let (is_drift, _magnitude) = analyzer.detect_semantic_drift(0.75);
        assert!(!is_drift, "Normal value should not be drift");

        // Test significantly lower value - should detect drift
        let (is_drift, magnitude) = analyzer.detect_semantic_drift(0.3);
        assert!(is_drift, "Significantly lower value should be drift");
        assert!(magnitude > 0.0, "Drift magnitude should be positive");
    }

    #[test]
    fn test_drift_with_default_params() {
        // With default params, drift threshold = mean - 2*std_dev = 0.7 - 0.3 = 0.4
        let analyzer = BertSemanticAnalyzer::with_defaults();

        let (is_drift, _) = analyzer.detect_semantic_drift(0.7);
        assert!(!is_drift, "Value at mean should not be drift");

        let (is_drift, magnitude) = analyzer.detect_semantic_drift(0.3);
        assert!(is_drift, "Value below threshold should be drift");
        assert!(magnitude > 0.0, "Drift magnitude should be positive");
    }

    #[test]
    fn test_outlier_detection() {
        // Train with normal distribution
        let mut trainer = BertSemanticTrainer::new();
        for i in 0..50 {
            let value = 0.6 + (i as f64 - 25.0) * 0.01;
            trainer.add_sample(value);
        }

        let learned = trainer.train().unwrap();
        let analyzer = BertSemanticAnalyzer::new(learned);

        // Test normal value - not an outlier
        let (is_outlier, _z_score) = analyzer.detect_outlier(0.6);
        assert!(!is_outlier, "Normal value should not be outlier");

        // Test extreme value - should be outlier
        let (is_outlier, z_score) = analyzer.detect_outlier(0.1);
        assert!(is_outlier, "Extreme value should be outlier");
        assert!(z_score.abs() > 2.5, "Z-score should be greater than 2.5");
    }

    #[test]
    fn test_outlier_with_default_params() {
        // With default params (mean=0.7, std_dev=0.15), outlier z-threshold = 2.5
        let analyzer = BertSemanticAnalyzer::with_defaults();

        let (is_outlier, _) = analyzer.detect_outlier(0.7);
        assert!(!is_outlier, "Value at mean should not be outlier");

        // Value far from mean should be outlier
        let (is_outlier, z_score) = analyzer.detect_outlier(0.2);
        assert!(is_outlier, "Extreme value should be outlier");
        assert!(z_score.abs() > 2.5, "Z-score should be greater than 2.5");
    }

    #[test]
    fn test_similarity_metrics_comprehensive() {
        // Train with normal distribution
        let mut trainer = BertSemanticTrainer::new();
        for i in 0..50 {
            let value = 0.6 + (i as f64 - 25.0) * 0.01;
            trainer.add_sample(value);
        }

        let learned = trainer.train().unwrap();
        let analyzer = BertSemanticAnalyzer::new(learned);

        // Get metrics for a typical value
        let metrics = analyzer.get_similarity_metrics(0.6);
        assert_eq!(metrics.similarity, 0.6);
        assert!(metrics.confidence > 0.0 && metrics.confidence <= 1.0);
        assert!(!metrics.is_drift);
        assert!(!metrics.is_outlier);
    }

    #[test]
    fn test_gmm_threshold_learning() {
        let mut trainer = BertSemanticTrainer::new();

        // Create clear bimodal distribution
        for _ in 0..30 {
            trainer.add_sample(0.2);
        }
        for _ in 0..30 {
            trainer.add_sample(0.8);
        }

        let learned = trainer.train().unwrap();
        assert!(
            learned.threshold > 0.2 && learned.threshold < 0.8,
            "GMM threshold should be between modes"
        );
    }

    #[test]
    fn test_percentile_threshold() {
        let mut trainer = BertSemanticTrainer::new().with_threshold_percentile(0.75);

        for i in 0..100 {
            trainer.add_sample(i as f64 / 100.0);
        }

        let learned = trainer.train().unwrap();
        // Threshold is combined from multiple methods (GMM, elbow, percentile)
        // For uniform distribution, expect it to be influenced by all methods
        assert!(
            learned.threshold >= 0.4 && learned.threshold <= 0.85,
            "Threshold should be in reasonable range, got {}",
            learned.threshold
        );

        // Verify threshold increases with percentile
        let mut trainer_low = BertSemanticTrainer::new().with_threshold_percentile(0.25);
        for i in 0..100 {
            trainer_low.add_sample(i as f64 / 100.0);
        }
        let learned_low = trainer_low.train().unwrap();

        assert!(
            learned.threshold > learned_low.threshold,
            "Higher percentile should yield higher threshold"
        );
    }

    #[test]
    fn test_max_history_size() {
        // Test with trainer instead of analyzer
        let mut trainer = BertSemanticTrainer::new().with_max_history(10);

        for i in 0..20 {
            trainer.add_sample(i as f64 / 20.0);
        }

        assert_eq!(trainer.sample_count(), 10);
    }

    #[test]
    fn test_manual_threshold() {
        // Test with custom learned params
        let mut params = LearnedParameters::default_params();
        params.threshold = 0.65;
        let analyzer = BertSemanticAnalyzer::new(params);

        assert_eq!(analyzer.get_threshold(), 0.65);
    }

    #[test]
    fn test_word_jaccard_similarity() {
        let analyzer = BertSemanticAnalyzer::with_defaults();

        let sim1 = analyzer.word_jaccard_similarity("hello world", "hello world");
        assert_eq!(sim1, 1.0, "Identical texts should have 1.0 similarity");

        let sim2 = analyzer.word_jaccard_similarity("hello world", "goodbye world");
        assert!(
            sim2 > 0.0 && sim2 < 1.0,
            "Partial overlap should have partial similarity"
        );

        let sim3 = analyzer.word_jaccard_similarity("hello", "goodbye");
        assert_eq!(sim3, 0.0, "No overlap should have 0.0 similarity");

        let sim4 = analyzer.word_jaccard_similarity("", "");
        assert_eq!(sim4, 1.0, "Empty strings should have 1.0 similarity");
    }

    #[test]
    fn test_analyze_with_all_metrics() {
        // Train with sample data
        let mut trainer = BertSemanticTrainer::new();
        for i in 0..30 {
            trainer.add_sample(0.6 + (i as f64 - 15.0) * 0.01);
        }

        let learned = trainer.train().unwrap();
        let analyzer = BertSemanticAnalyzer::new(learned);

        let engine = DiffEngine::default();
        let diff = engine.diff("the quick brown fox", "the fast brown dog");

        let result = analyzer.analyze(&diff);

        // Check all metrics are present
        assert!(result.metrics.contains_key("bert_semantic_similarity"));
        assert!(result.metrics.contains_key("bert_semantic_distance"));
        assert!(result.metrics.contains_key("confidence_score"));
        assert!(result.metrics.contains_key("z_score"));
        assert!(result.metrics.contains_key("drift_magnitude"));
        assert!(result.metrics.contains_key("learned_mean"));
        assert!(result.metrics.contains_key("learned_std_dev"));
        assert!(result.metadata.contains_key("similarity_level"));
    }

    #[test]
    fn test_import_history_truncation() {
        // Test with trainer instead of analyzer
        let mut trainer = BertSemanticTrainer::new().with_max_history(5);

        let large_history: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        trainer.add_samples(&large_history);

        assert_eq!(trainer.sample_count(), 5);
    }
}
