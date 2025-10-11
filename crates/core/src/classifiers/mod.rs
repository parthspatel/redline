//! Classifiers for categorizing text changes
//!
//! Implements various classification techniques from the research document

pub mod semantic;
pub mod readability;
pub mod stylistic;
pub mod syntactic;
pub mod naive_bayes;
pub mod bert_semantic;

use crate::diff::{ChangeCategory, DiffOperation, DiffResult};

/// Result of a classification
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// The assigned category
    pub category: ChangeCategory,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    
    /// Alternative categories with their scores
    pub alternatives: Vec<(ChangeCategory, f64)>,
    
    /// Explanation or reasoning
    pub explanation: String,
}

impl ClassificationResult {
    pub fn new(category: ChangeCategory, confidence: f64) -> Self {
        Self {
            category,
            confidence,
            alternatives: Vec::new(),
            explanation: String::new(),
        }
    }

    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }

    pub fn add_alternative(&mut self, category: ChangeCategory, confidence: f64) {
        self.alternatives.push((category, confidence));
    }
}

/// Trait for change classifiers
pub trait ChangeClassifier: Send + Sync {
    /// Classify a single operation
    fn classify_operation(&self, operation: &DiffOperation) -> ClassificationResult;

    /// Classify a single operation with cached metrics (more efficient)
    fn classify_operation_with_metrics(&self, operation: &DiffOperation, metrics: Option<&crate::metrics::PairwiseMetrics>) -> ClassificationResult {
        // Default implementation ignores metrics
        self.classify_operation(operation)
    }

    /// Classify an entire diff (default: aggregate individual operations)
    fn classify_diff(&self, diff: &DiffResult) -> ClassificationResult {
        let mut category_scores: std::collections::HashMap<String, f64> = 
            std::collections::HashMap::new();
        
        for op in &diff.operations {
            let classification = self.classify_operation(op);
            let key = format!("{:?}", classification.category);
            *category_scores.entry(key).or_insert(0.0) += classification.confidence;
        }

        // Find dominant category
        let (dominant_cat, score) = category_scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or_else(|| ("Unknown".to_string(), 0.0));

        let category = match dominant_cat.as_str() {
            "Semantic" => ChangeCategory::Semantic,
            "Stylistic" => ChangeCategory::Stylistic,
            "Formatting" => ChangeCategory::Formatting,
            "Syntactic" => ChangeCategory::Syntactic,
            "Organizational" => ChangeCategory::Organizational,
            _ => ChangeCategory::Unknown,
        };

        let total_score: f64 = category_scores.values().sum();
        let confidence = if total_score > 0.0 {
            score / total_score
        } else {
            0.0
        };

        ClassificationResult::new(category, confidence)
            .with_explanation(format!("Dominant category based on operation analysis"))
    }

    /// Get the name of this classifier
    fn name(&self) -> &str;

    /// Declare the metric dependencies this classifier requires
    /// Returns a list of NodeDependencies that will be added to the execution plan
    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        Vec::new()
    }

    /// Clone into a box
    fn clone_box(&self) -> Box<dyn ChangeClassifier>;
}

impl Clone for Box<dyn ChangeClassifier> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ============================================================================
// Rule-Based Classifier (Combined Rules)
// ============================================================================

/// A comprehensive rule-based classifier
#[derive(Clone)]
pub struct RuleBasedClassifier {
    /// Weight for each classification component
    pub semantic_weight: f64,
    pub syntactic_weight: f64,
    pub stylistic_weight: f64,
    pub formatting_weight: f64,
}

impl RuleBasedClassifier {
    pub fn new() -> Self {
        Self {
            semantic_weight: 1.0,
            syntactic_weight: 1.0,
            stylistic_weight: 1.0,
            formatting_weight: 1.0,
        }
    }

    /// Check if change is primarily case-based
    fn is_case_change(&self, orig: &str, modified: &str) -> bool {
        orig.to_lowercase() == modified.to_lowercase()
    }

    /// Check if change is primarily whitespace
    fn is_whitespace_change(&self, orig: &str, modified: &str) -> bool {
        let orig_no_ws: String = orig.chars().filter(|c| !c.is_whitespace()).collect();
        let mod_no_ws: String = modified.chars().filter(|c| !c.is_whitespace()).collect();
        orig_no_ws == mod_no_ws
    }

    /// Check if change is primarily punctuation
    fn is_punctuation_change(&self, orig: &str, modified: &str) -> bool {
        let orig_no_punct: String = orig.chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect();
        let mod_no_punct: String = modified.chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect();
        
        orig_no_punct.to_lowercase() == mod_no_punct.to_lowercase()
    }

    /// Compute Levenshtein distance
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

    /// Compute word-level similarity
    fn word_similarity(&self, orig: &str, modified: &str) -> f64 {
        let orig_words: std::collections::HashSet<_> = orig
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        
        let mod_words: std::collections::HashSet<_> = modified
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        if orig_words.is_empty() && mod_words.is_empty() {
            return 1.0;
        }

        let intersection = orig_words.intersection(&mod_words).count();
        let union = orig_words.union(&mod_words).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

impl Default for RuleBasedClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangeClassifier for RuleBasedClassifier {
    fn classify_operation(&self, operation: &DiffOperation) -> ClassificationResult {
        use crate::diff::EditType;

        // If already classified, return that
        if !matches!(operation.category, ChangeCategory::Unknown) {
            return ClassificationResult::new(operation.category.clone(), operation.confidence);
        }

        // For Equal operations, no classification needed
        if operation.edit_type == EditType::Equal {
            return ClassificationResult::new(ChangeCategory::Unknown, 1.0);
        }

        // Get text for analysis
        let orig_text = operation.original_text.as_deref().unwrap_or("");
        let mod_text = operation.modified_text.as_deref().unwrap_or("");

        // Check formatting changes first (highest priority for specific cases)
        if self.is_case_change(orig_text, mod_text) {
            return ClassificationResult::new(ChangeCategory::Formatting, 0.95)
                .with_explanation("Case-only change detected");
        }

        if self.is_whitespace_change(orig_text, mod_text) {
            return ClassificationResult::new(ChangeCategory::Formatting, 0.95)
                .with_explanation("Whitespace-only change detected");
        }

        if self.is_punctuation_change(orig_text, mod_text) {
            return ClassificationResult::new(ChangeCategory::Syntactic, 0.85)
                .with_explanation("Punctuation change detected");
        }

        // Compute similarity metrics
        let max_len = orig_text.len().max(mod_text.len());
        let edit_distance = self.levenshtein_distance(orig_text, mod_text);
        let char_similarity = if max_len > 0 {
            1.0 - (edit_distance as f64 / max_len as f64)
        } else {
            0.0
        };

        let word_sim = self.word_similarity(orig_text, mod_text);

        // Decision tree based on similarity
        let (category, confidence, explanation) = if char_similarity >= 0.9 {
            (
                ChangeCategory::Syntactic,
                0.8,
                "Very high character similarity suggests minor corrections"
            )
        } else if word_sim >= 0.8 {
            (
                ChangeCategory::Stylistic,
                0.75,
                "High word-level similarity suggests stylistic refinement"
            )
        } else if word_sim >= 0.5 {
            (
                ChangeCategory::Stylistic,
                0.6,
                "Moderate similarity suggests stylistic change"
            )
        } else if word_sim >= 0.3 {
            (
                ChangeCategory::Semantic,
                0.7,
                "Low-moderate similarity suggests semantic change"
            )
        } else {
            (
                ChangeCategory::Semantic,
                0.85,
                "Low similarity suggests substantial semantic change"
            )
        };

        let mut result = ClassificationResult::new(category, confidence)
            .with_explanation(explanation);

        // Add alternatives if confidence is not very high
        if confidence < 0.9 {
            if char_similarity > 0.7 {
                result.add_alternative(ChangeCategory::Syntactic, 0.3);
            }
            if word_sim > 0.6 && word_sim < 0.9 {
                result.add_alternative(ChangeCategory::Stylistic, 0.25);
            }
        }

        result
    }

    fn name(&self) -> &str {
        "rule_based"
    }

    fn clone_box(&self) -> Box<dyn ChangeClassifier> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Ensemble Classifier
// ============================================================================

/// Combines multiple classifiers using voting or averaging
#[derive(Clone)]
pub struct EnsembleClassifier {
    classifiers: Vec<Box<dyn ChangeClassifier>>,
    /// Strategy: "vote" or "average"
    strategy: String,
}

impl EnsembleClassifier {
    pub fn new(classifiers: Vec<Box<dyn ChangeClassifier>>) -> Self {
        Self {
            classifiers,
            strategy: "vote".to_string(),
        }
    }

    pub fn with_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.strategy = strategy.into();
        self
    }
}

impl ChangeClassifier for EnsembleClassifier {
    fn classify_operation(&self, operation: &DiffOperation) -> ClassificationResult {
        if self.classifiers.is_empty() {
            return ClassificationResult::new(ChangeCategory::Unknown, 0.0);
        }

        // Get classifications from all classifiers
        let classifications: Vec<ClassificationResult> = self.classifiers
            .iter()
            .map(|c| c.classify_operation(operation))
            .collect();

        match self.strategy.as_str() {
            "vote" => {
                // Count votes for each category
                let mut votes: std::collections::HashMap<String, usize> = 
                    std::collections::HashMap::new();
                
                for class in &classifications {
                    let key = format!("{:?}", class.category);
                    *votes.entry(key).or_insert(0) += 1;
                }

                // Find winner
                let (winner_key, winner_count) = votes.iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(k, v)| (k.clone(), *v))
                    .unwrap();

                let category = match winner_key.as_str() {
                    "Semantic" => ChangeCategory::Semantic,
                    "Stylistic" => ChangeCategory::Stylistic,
                    "Formatting" => ChangeCategory::Formatting,
                    "Syntactic" => ChangeCategory::Syntactic,
                    "Organizational" => ChangeCategory::Organizational,
                    _ => ChangeCategory::Unknown,
                };

                let confidence = winner_count as f64 / self.classifiers.len() as f64;

                ClassificationResult::new(category, confidence)
                    .with_explanation(format!("Ensemble vote: {} out of {}", winner_count, self.classifiers.len()))
            }
            "average" => {
                // Average confidence scores per category
                let mut category_scores: std::collections::HashMap<String, Vec<f64>> = 
                    std::collections::HashMap::new();

                for class in &classifications {
                    let key = format!("{:?}", class.category);
                    category_scores.entry(key).or_default().push(class.confidence);
                }

                // Compute average for each category
                let (winner_key, avg_confidence) = category_scores.iter()
                    .map(|(k, scores)| {
                        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                        (k.clone(), avg)
                    })
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                let category = match winner_key.as_str() {
                    "Semantic" => ChangeCategory::Semantic,
                    "Stylistic" => ChangeCategory::Stylistic,
                    "Formatting" => ChangeCategory::Formatting,
                    "Syntactic" => ChangeCategory::Syntactic,
                    "Organizational" => ChangeCategory::Organizational,
                    _ => ChangeCategory::Unknown,
                };

                ClassificationResult::new(category, avg_confidence)
                    .with_explanation("Ensemble average of classifier confidences")
            }
            _ => ClassificationResult::new(ChangeCategory::Unknown, 0.0)
        }
    }

    fn name(&self) -> &str {
        "ensemble"
    }

    fn clone_box(&self) -> Box<dyn ChangeClassifier> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::CharSpan;

    #[test]
    fn test_rule_based_classifier() {
        let classifier = RuleBasedClassifier::new();
        
        // Test case change
        let op = DiffOperation::new(crate::diff::EditType::Modify)
            .with_original("Hello".to_string(), CharSpan::new(0, 5))
            .with_modified("hello".to_string(), CharSpan::new(0, 5));
        
        let result = classifier.classify_operation(&op);
        assert!(matches!(result.category, ChangeCategory::Formatting));
    }

    #[test]
    fn test_semantic_classification() {
        let classifier = RuleBasedClassifier::new();
        
        let op = DiffOperation::new(crate::diff::EditType::Modify)
            .with_original("cat".to_string(), CharSpan::new(0, 3))
            .with_modified("dog".to_string(), CharSpan::new(0, 3));
        
        let result = classifier.classify_operation(&op);
        assert!(matches!(result.category, ChangeCategory::Semantic));
    }
}
