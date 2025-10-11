//! Machine learning-based classification framework
//!
//! Provides infrastructure for training and using ML classifiers

use super::{ChangeCategory, ChangeClassifier, ClassificationResult};
use crate::diff::DiffOperation;
use std::collections::HashMap;

/// Feature vector for ML classification
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
}

impl FeatureVector {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            feature_names: Vec::new(),
        }
    }

    pub fn add_feature(&mut self, name: impl Into<String>, value: f64) {
        self.feature_names.push(name.into());
        self.features.push(value);
    }

    pub fn to_vec(&self) -> &[f64] {
        &self.features
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature extractor for diff operations
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from an operation
    fn extract(&self, operation: &DiffOperation) -> FeatureVector;

    /// Get feature names
    fn feature_names(&self) -> Vec<String>;
}

/// Standard feature extractor
#[derive(Clone)]
pub struct StandardFeatureExtractor;

impl StandardFeatureExtractor {
    pub fn new() -> Self {
        Self
    }

    fn char_similarity(&self, s1: &str, s2: &str) -> f64 {
        let len1 = s1.len();
        let len2 = s2.len();
        let max_len = len1.max(len2);
        
        if max_len == 0 {
            return 1.0;
        }

        let distance = self.levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for (i, c1) in s1.chars().enumerate() {
            for (j, c2) in s2.chars().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                    .min(matrix[i + 1][j] + 1)
                    .min(matrix[i][j] + cost);
            }
        }

        matrix[len1][len2]
    }

    fn word_overlap(&self, s1: &str, s2: &str) -> f64 {
        let words1: std::collections::HashSet<_> = s1
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        
        let words2: std::collections::HashSet<_> = s2
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

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

    fn length_ratio(&self, s1: &str, s2: &str) -> f64 {
        let len1 = s1.len() as f64;
        let len2 = s2.len() as f64;
        
        if len1 == 0.0 && len2 == 0.0 {
            return 1.0;
        }

        let max_len = len1.max(len2);
        let min_len = len1.min(len2);

        if max_len == 0.0 {
            0.0
        } else {
            min_len / max_len
        }
    }
}

impl Default for StandardFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor for StandardFeatureExtractor {
    fn extract(&self, operation: &DiffOperation) -> FeatureVector {
        let mut features = FeatureVector::new();
        
        let orig_text = operation.original_text.as_deref().unwrap_or("");
        let mod_text = operation.modified_text.as_deref().unwrap_or("");

        // Text similarity features
        features.add_feature("char_similarity", self.char_similarity(orig_text, mod_text));
        features.add_feature("word_overlap", self.word_overlap(orig_text, mod_text));
        features.add_feature("length_ratio", self.length_ratio(orig_text, mod_text));

        // Length features
        features.add_feature("orig_length", orig_text.len() as f64);
        features.add_feature("mod_length", mod_text.len() as f64);
        features.add_feature("length_diff", (mod_text.len() as f64 - orig_text.len() as f64).abs());

        // Edit type features (one-hot encoding)
        let is_insert = if matches!(operation.edit_type, crate::diff::EditType::Insert) { 1.0 } else { 0.0 };
        let is_delete = if matches!(operation.edit_type, crate::diff::EditType::Delete) { 1.0 } else { 0.0 };
        let is_modify = if matches!(operation.edit_type, crate::diff::EditType::Modify) { 1.0 } else { 0.0 };
        
        features.add_feature("is_insert", is_insert);
        features.add_feature("is_delete", is_delete);
        features.add_feature("is_modify", is_modify);

        // Case change feature
        let case_changed = if orig_text.to_lowercase() == mod_text.to_lowercase() { 1.0 } else { 0.0 };
        features.add_feature("case_only_change", case_changed);

        // Punctuation features
        let orig_punct = orig_text.chars().filter(|c| c.is_ascii_punctuation()).count() as f64;
        let mod_punct = mod_text.chars().filter(|c| c.is_ascii_punctuation()).count() as f64;
        features.add_feature("punct_diff", (mod_punct - orig_punct).abs());

        features
    }

    fn feature_names(&self) -> Vec<String> {
        vec![
            "char_similarity".to_string(),
            "word_overlap".to_string(),
            "length_ratio".to_string(),
            "orig_length".to_string(),
            "mod_length".to_string(),
            "length_diff".to_string(),
            "is_insert".to_string(),
            "is_delete".to_string(),
            "is_modify".to_string(),
            "case_only_change".to_string(),
            "punct_diff".to_string(),
        ]
    }
}

/// Training sample for ML classifier
#[derive(Clone)]
pub struct TrainingSample {
    pub features: FeatureVector,
    pub label: ChangeCategory,
}

impl TrainingSample {
    pub fn new(features: FeatureVector, label: ChangeCategory) -> Self {
        Self { features, label }
    }
}

/// Naive Bayes classifier for change categorization
#[derive(Clone)]
pub struct NaiveBayesClassifier {
    feature_extractor: StandardFeatureExtractor,
    /// Prior probabilities for each category
    priors: HashMap<String, f64>,
    /// Feature statistics per category: category -> feature_idx -> (mean, std_dev)
    feature_stats: HashMap<String, Vec<(f64, f64)>>,
    /// Whether the classifier has been trained
    trained: bool,
}

impl NaiveBayesClassifier {
    pub fn new() -> Self {
        Self {
            feature_extractor: StandardFeatureExtractor::new(),
            priors: HashMap::new(),
            feature_stats: HashMap::new(),
            trained: false,
        }
    }

    /// Train the classifier on labeled samples
    pub fn train(&mut self, samples: &[TrainingSample]) {
        if samples.is_empty() {
            return;
        }

        // Count samples per category
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        let mut category_features: HashMap<String, Vec<Vec<f64>>> = HashMap::new();

        for sample in samples {
            let cat_key = format!("{:?}", sample.label);
            *category_counts.entry(cat_key.clone()).or_insert(0) += 1;
            category_features.entry(cat_key)
                .or_default()
                .push(sample.features.features.clone());
        }

        // Compute priors
        let total_samples = samples.len() as f64;
        for (category, count) in &category_counts {
            self.priors.insert(category.clone(), *count as f64 / total_samples);
        }

        // Compute feature statistics (mean and std dev) per category
        for (category, features_list) in category_features {
            let n_features = features_list[0].len();
            let mut stats = Vec::new();

            for feature_idx in 0..n_features {
                let values: Vec<f64> = features_list.iter()
                    .map(|features| features[feature_idx])
                    .collect();

                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                let std_dev = variance.sqrt().max(1e-6); // Avoid division by zero

                stats.push((mean, std_dev));
            }

            self.feature_stats.insert(category, stats);
        }

        self.trained = true;
    }

    /// Compute Gaussian probability
    fn gaussian_prob(&self, x: f64, mean: f64, std_dev: f64) -> f64 {
        let exponent = -((x - mean).powi(2)) / (2.0 * std_dev.powi(2));
        (1.0 / (std_dev * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
    }

    /// Predict category for a feature vector
    fn predict(&self, features: &FeatureVector) -> (ChangeCategory, f64) {
        if !self.trained {
            return (ChangeCategory::Unknown, 0.0);
        }

        let mut max_posterior = f64::NEG_INFINITY;
        let mut best_category = ChangeCategory::Unknown;

        for (category, prior) in &self.priors {
            let mut log_posterior = prior.ln();

            if let Some(stats) = self.feature_stats.get(category) {
                for (i, &value) in features.features.iter().enumerate() {
                    if let Some(&(mean, std_dev)) = stats.get(i) {
                        let prob = self.gaussian_prob(value, mean, std_dev);
                        log_posterior += prob.ln().max(-100.0); // Prevent underflow
                    }
                }
            }

            if log_posterior > max_posterior {
                max_posterior = log_posterior;
                best_category = match category.as_str() {
                    "Semantic" => ChangeCategory::Semantic,
                    "Stylistic" => ChangeCategory::Stylistic,
                    "Formatting" => ChangeCategory::Formatting,
                    "Syntactic" => ChangeCategory::Syntactic,
                    "Organizational" => ChangeCategory::Organizational,
                    _ => ChangeCategory::Unknown,
                };
            }
        }

        // Convert log probability to confidence (normalized)
        let confidence = if max_posterior > f64::NEG_INFINITY {
            (max_posterior / 10.0).exp().min(1.0).max(0.0)
        } else {
            0.0
        };

        (best_category, confidence)
    }
}

impl Default for NaiveBayesClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangeClassifier for NaiveBayesClassifier {
    fn classify_operation(&self, operation: &DiffOperation) -> ClassificationResult {
        let features = self.feature_extractor.extract(operation);
        let (category, confidence) = self.predict(&features);

        ClassificationResult::new(category, confidence)
            .with_explanation("Naive Bayes classification based on trained model")
    }

    fn name(&self) -> &str {
        "naive_bayes"
    }

    fn clone_box(&self) -> Box<dyn ChangeClassifier> {
        Box::new(self.clone())
    }
}

/// Helper function to create labeled training samples from annotated diffs
pub fn create_training_samples(
    operations: &[(DiffOperation, ChangeCategory)],
    extractor: &StandardFeatureExtractor,
) -> Vec<TrainingSample> {
    operations
        .iter()
        .map(|(op, label)| {
            let features = extractor.extract(op);
            TrainingSample::new(features, label.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::CharSpan;

    #[test]
    fn test_feature_extraction() {
        let extractor = StandardFeatureExtractor::new();
        
        let op = DiffOperation::new(crate::diff::EditType::Modify)
            .with_original("hello".to_string(), CharSpan::new(0, 5))
            .with_modified("world".to_string(), CharSpan::new(0, 5));
        
        let features = extractor.extract(&op);
        
        assert!(features.features.len() > 0);
        assert_eq!(features.features.len(), features.feature_names.len());
    }

    #[test]
    fn test_naive_bayes_training() {
        let extractor = StandardFeatureExtractor::new();
        let mut classifier = NaiveBayesClassifier::new();
        
        // Create some training samples
        let op1 = DiffOperation::new(crate::diff::EditType::Modify)
            .with_original("Hello".to_string(), CharSpan::new(0, 5))
            .with_modified("hello".to_string(), CharSpan::new(0, 5));
        
        let samples = vec![
            TrainingSample::new(
                extractor.extract(&op1),
                ChangeCategory::Formatting
            ),
        ];
        
        classifier.train(&samples);
        assert!(classifier.trained);
    }
}
