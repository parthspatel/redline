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

    /// Extract features using cached metrics (more efficient)
    fn extract_with_metrics(&self, operation: &DiffOperation, metrics: Option<&crate::metrics::PairwiseMetrics>) -> FeatureVector {
        // Default implementation falls back to regular extract
        self.extract(operation)
    }

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

    fn count_words(&self, s: &str) -> usize {
        s.split_whitespace().count()
    }

    fn flesch_reading_ease(&self, s: &str) -> f64 {
        let words: Vec<&str> = s.split_whitespace().collect();
        let word_count = words.len();

        if word_count == 0 {
            return 0.0;
        }

        // Count sentences (simple approximation: . ! ?)
        let sentence_count = s.chars()
            .filter(|c| *c == '.' || *c == '!' || *c == '?')
            .count()
            .max(1);

        // Count syllables (simple approximation: count vowel groups)
        let syllable_count: usize = words.iter()
            .map(|word| self.count_syllables(word))
            .sum();

        let avg_syllables_per_word = syllable_count as f64 / word_count as f64;
        let avg_words_per_sentence = word_count as f64 / sentence_count as f64;

        // Flesch Reading Ease formula
        206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
    }

    fn count_syllables(&self, word: &str) -> usize {
        let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
        let word_lower = word.to_lowercase();
        let chars: Vec<char> = word_lower.chars().collect();

        if chars.is_empty() {
            return 0;
        }

        let mut syllable_count = 0;
        let mut prev_was_vowel = false;

        for ch in chars.iter() {
            let is_vowel = vowels.contains(ch);
            if is_vowel && !prev_was_vowel {
                syllable_count += 1;
            }
            prev_was_vowel = is_vowel;
        }

        // Adjust for silent 'e' at end
        if word_lower.ends_with('e') && syllable_count > 1 {
            syllable_count -= 1;
        }

        syllable_count.max(1)
    }

    fn stopword_ratio(&self, s: &str) -> f64 {
        const STOPWORDS: &[&str] = &[
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
            "for", "from", "has", "have", "he", "in", "is", "it", "its",
            "of", "on", "or", "that", "the", "to", "was", "were", "will", "with"
        ];

        let words: Vec<&str> = s.split_whitespace().collect();
        let word_count = words.len();

        if word_count == 0 {
            return 0.0;
        }

        let stopword_count = words.iter()
            .filter(|w| STOPWORDS.contains(&w.to_lowercase().as_str()))
            .count();

        stopword_count as f64 / word_count as f64
    }

    fn whitespace_ratio(&self, s: &str) -> f64 {
        if s.is_empty() {
            return 0.0;
        }

        let whitespace_count = s.chars().filter(|c| c.is_whitespace()).count();
        whitespace_count as f64 / s.len() as f64
    }

    fn contains_negation(&self, s: &str) -> bool {
        const NEGATION_WORDS: &[&str] = &["not", "no", "never", "neither", "none", "nobody", "nothing", "nowhere"];

        let lower = s.to_lowercase();

        // Check for negation words
        for negation in NEGATION_WORDS {
            if lower.split_whitespace().any(|w| w == *negation) {
                return true;
            }
        }

        // Check for n't contractions
        lower.contains("n't")
    }
}

impl Default for StandardFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor for StandardFeatureExtractor {
    fn extract(&self, operation: &DiffOperation) -> FeatureVector {
        // Fall back to computing metrics on-the-fly
        self.extract_with_metrics(operation, None)
    }

    fn extract_with_metrics(&self, operation: &DiffOperation, metrics: Option<&crate::metrics::PairwiseMetrics>) -> FeatureVector {
        let mut features = FeatureVector::new();

        let orig_text = operation.original_text.as_deref().unwrap_or("");
        let mod_text = operation.modified_text.as_deref().unwrap_or("");

        // If we have cached metrics, use them (MUCH faster!)
        if let Some(m) = metrics {
            // Text similarity features (from cache)
            features.add_feature("char_similarity", m.char_similarity);
            features.add_feature("word_overlap", m.word_overlap);
            features.add_feature("length_ratio", m.length_ratio);

            // Length features (from cache)
            features.add_feature("orig_length", m.original.char_count as f64);
            features.add_feature("mod_length", m.modified.char_count as f64);
            features.add_feature("length_diff", (m.modified.char_count as f64 - m.original.char_count as f64).abs());

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

            // Punctuation features (from cache)
            features.add_feature("punct_diff", (m.modified.punctuation_count as f64 - m.original.punctuation_count as f64).abs());

            // NEW FEATURES (from cache!):
            features.add_feature("readability_score_diff", m.readability_diff);
            features.add_feature("word_count_diff", m.word_count_diff);

            let avg_stopword_ratio = (m.original.stopword_ratio + m.modified.stopword_ratio) / 2.0;
            features.add_feature("stopword_ratio", avg_stopword_ratio);

            features.add_feature("whitespace_ratio_change", m.whitespace_ratio_diff);
            features.add_feature("negation_changed", if m.negation_changed { 1.0 } else { 0.0 });
        } else {
            // Fallback: compute on-the-fly (slower)
            features.add_feature("char_similarity", self.char_similarity(orig_text, mod_text));
            features.add_feature("word_overlap", self.word_overlap(orig_text, mod_text));
            features.add_feature("length_ratio", self.length_ratio(orig_text, mod_text));

            features.add_feature("orig_length", orig_text.len() as f64);
            features.add_feature("mod_length", mod_text.len() as f64);
            features.add_feature("length_diff", (mod_text.len() as f64 - orig_text.len() as f64).abs());

            let is_insert = if matches!(operation.edit_type, crate::diff::EditType::Insert) { 1.0 } else { 0.0 };
            let is_delete = if matches!(operation.edit_type, crate::diff::EditType::Delete) { 1.0 } else { 0.0 };
            let is_modify = if matches!(operation.edit_type, crate::diff::EditType::Modify) { 1.0 } else { 0.0 };

            features.add_feature("is_insert", is_insert);
            features.add_feature("is_delete", is_delete);
            features.add_feature("is_modify", is_modify);

            let case_changed = if orig_text.to_lowercase() == mod_text.to_lowercase() { 1.0 } else { 0.0 };
            features.add_feature("case_only_change", case_changed);

            let orig_punct = orig_text.chars().filter(|c| c.is_ascii_punctuation()).count() as f64;
            let mod_punct = mod_text.chars().filter(|c| c.is_ascii_punctuation()).count() as f64;
            features.add_feature("punct_diff", (mod_punct - orig_punct).abs());

            let orig_readability = self.flesch_reading_ease(orig_text);
            let mod_readability = self.flesch_reading_ease(mod_text);
            features.add_feature("readability_score_diff", (mod_readability - orig_readability).abs());

            let orig_word_count = self.count_words(orig_text) as f64;
            let mod_word_count = self.count_words(mod_text) as f64;
            features.add_feature("word_count_diff", (mod_word_count - orig_word_count).abs());

            let orig_stopword_ratio = self.stopword_ratio(orig_text);
            let mod_stopword_ratio = self.stopword_ratio(mod_text);
            let avg_stopword_ratio = (orig_stopword_ratio + mod_stopword_ratio) / 2.0;
            features.add_feature("stopword_ratio", avg_stopword_ratio);

            let orig_ws_ratio = self.whitespace_ratio(orig_text);
            let mod_ws_ratio = self.whitespace_ratio(mod_text);
            features.add_feature("whitespace_ratio_change", (mod_ws_ratio - orig_ws_ratio).abs());

            let orig_has_negation = self.contains_negation(orig_text);
            let mod_has_negation = self.contains_negation(mod_text);
            let negation_changed = if orig_has_negation != mod_has_negation { 1.0 } else { 0.0 };
            features.add_feature("negation_changed", negation_changed);
        }

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
            "readability_score_diff".to_string(),
            "word_count_diff".to_string(),
            "stopword_ratio".to_string(),
            "whitespace_ratio_change".to_string(),
            "negation_changed".to_string(),
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
        // Fall back to non-cached version
        self.classify_operation_with_metrics(operation, None)
    }

    fn classify_operation_with_metrics(&self, operation: &DiffOperation, metrics: Option<&crate::metrics::PairwiseMetrics>) -> ClassificationResult {
        // Use cached metrics if available (much faster!)
        let features = self.feature_extractor.extract_with_metrics(operation, metrics);
        let (category, confidence) = self.predict(&features);

        ClassificationResult::new(category, confidence)
            .with_explanation("Naive Bayes classification based on trained model")
    }

    fn name(&self) -> &str {
        "naive_bayes"
    }

    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        use crate::execution::{ExecutionNode, MetricType, NodeDependencies};

        // NaiveBayesClassifier uses StandardFeatureExtractor which needs:
        // - char_similarity (depends on levenshtein_distance)
        // - word_overlap (depends on word_count)
        // - readability_score_diff (depends on flesch_reading_ease)
        // - word_count_diff (depends on word_count)
        // - whitespace_ratio_change (depends on whitespace_ratio)
        // - negation_changed (depends on has_negation)

        vec![
            NodeDependencies::new(ExecutionNode::Metric(MetricType::CharSimilarity))
                .with_dependency(ExecutionNode::Metric(MetricType::LevenshteinDistance)),
            NodeDependencies::new(ExecutionNode::Metric(MetricType::WordOverlap))
                .with_dependency(ExecutionNode::Metric(MetricType::WordCount)),
            NodeDependencies::new(ExecutionNode::Metric(MetricType::ReadabilityDiff))
                .with_dependency(ExecutionNode::Metric(MetricType::FleschReadingEase)),
            NodeDependencies::new(ExecutionNode::Metric(MetricType::WordCountDiff))
                .with_dependency(ExecutionNode::Metric(MetricType::WordCount)),
            NodeDependencies::new(ExecutionNode::Metric(MetricType::WhitespaceRatioDiff))
                .with_dependency(ExecutionNode::Metric(MetricType::WhitespaceRatio)),
            NodeDependencies::new(ExecutionNode::Metric(MetricType::NegationChanged))
                .with_dependency(ExecutionNode::Metric(MetricType::HasNegation)),
        ]
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

    #[test]
    fn test_new_features() {
        let extractor = StandardFeatureExtractor::new();

        // Test readability_score_diff
        let op1 = DiffOperation::new(crate::diff::EditType::Modify)
            .with_original("The cat sat on the mat.".to_string(), CharSpan::new(0, 23))
            .with_modified("The feline reclined upon the textile floor covering.".to_string(), CharSpan::new(0, 53));

        let features1 = extractor.extract(&op1);
        assert_eq!(features1.features.len(), 16); // 11 old + 5 new features

        // Test word_count_diff
        let op2 = DiffOperation::new(crate::diff::EditType::Modify)
            .with_original("hello".to_string(), CharSpan::new(0, 5))
            .with_modified("hello world".to_string(), CharSpan::new(0, 11));

        let features2 = extractor.extract(&op2);
        let word_count_diff_idx = features2.feature_names.iter().position(|n| n == "word_count_diff").unwrap();
        assert_eq!(features2.features[word_count_diff_idx], 1.0);

        // Test negation_changed
        let op3 = DiffOperation::new(crate::diff::EditType::Modify)
            .with_original("I like this.".to_string(), CharSpan::new(0, 12))
            .with_modified("I don't like this.".to_string(), CharSpan::new(0, 18));

        let features3 = extractor.extract(&op3);
        let negation_idx = features3.feature_names.iter().position(|n| n == "negation_changed").unwrap();
        assert_eq!(features3.features[negation_idx], 1.0);

        // Test whitespace_ratio_change
        let op4 = DiffOperation::new(crate::diff::EditType::Modify)
            .with_original("hello".to_string(), CharSpan::new(0, 5))
            .with_modified("h e l l o".to_string(), CharSpan::new(0, 9));

        let features4 = extractor.extract(&op4);
        let ws_idx = features4.feature_names.iter().position(|n| n == "whitespace_ratio_change").unwrap();
        assert!(features4.features[ws_idx] > 0.0);
    }
}
