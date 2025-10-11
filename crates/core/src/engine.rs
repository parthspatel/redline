//! Main diff engine that orchestrates the entire diff process

use crate::algorithms::{
    DiffAlgorithm as DiffAlgoTrait, HistogramAlgorithm, MyersAlgorithm, PatienceAlgorithm,
};
use crate::config::{DiffAlgorithm, DiffConfig};
use crate::diff::{ChangeCategory, DiffResult, EditType};
use crate::normalizers::Lowercase;
use crate::pipeline::TextPipeline;
use crate::tokenizers::{Token, WordTokenizer};

/// The main diff engine
pub struct DiffEngine {
    config: DiffConfig,
}

impl DiffEngine {
    /// Create a new diff engine with the given configuration
    pub fn new(config: DiffConfig) -> Self {
        Self { config }
    }

    /// Create a diff engine with the default configuration
    pub fn default_config() -> Self {
        Self::new(DiffConfig::default())
    }

    /// Compute the diff between two strings
    ///
    /// This is the main entry point that orchestrates the entire diff process:
    /// 1. Normalize text through the pipeline
    /// 2. Tokenize the normalized text
    /// 3. Run the diff algorithm
    /// 4. Analyze the results
    /// 5. Return a comprehensive DiffResult
    pub fn diff(&self, original: &str, modified: &str) -> DiffResult {
        // Initialize result
        let mut result = DiffResult::new(original.to_string(), modified.to_string());

        // Step 0: Build execution plan based on configured features and execute metrics in dependency order
        let plan = self.config.build_execution_plan();
        self.execute_plan(&plan, &mut result);

        // Step 1: Apply normalization pipeline
        let (original_tokens, modified_tokens) = self.normalize_and_tokenize(original, modified);

        // Step 2: Run diff algorithm
        let operations = self.compute_diff_operations(&original_tokens, &modified_tokens);

        // Step 3: Add operations to result
        for op in operations {
            result.add_operation(op);
        }

        // Step 4: Perform analysis if enabled (now using cached metrics)
        if self.config.compute_semantic_similarity {
           result.analysis.semantic_similarity =
                self.compute_semantic_similarity(&result);
        }

        if self.config.analyze_style {
            result.analysis.stylistic_change = self.analyze_style_change(&result);
            result.analysis.readability_change = self.analyze_readability_change(&result);
        }

        if self.config.classify_edits {
            self.classify_edit_operations(&mut result);
        }

        // Step 5: Finalize result
        result.finalize();

        result
    }

    /// Execute metrics in dependency order based on execution plan
    fn execute_plan(&self, plan: &crate::execution::ExecutionPlan, result: &mut DiffResult) {
        use crate::execution::ExecutionNode;
        use crate::metrics::{TextMetrics, PairwiseMetrics};

        // Initialize metrics structure
        let original_metrics = TextMetrics::compute(&result.original_text);
        let modified_metrics = TextMetrics::compute(&result.modified_text);
        let mut pairwise = PairwiseMetrics {
            original: original_metrics.clone(),
            modified: modified_metrics.clone(),
            char_similarity: 0.0,
            word_overlap: 0.0,
            levenshtein_distance: 0,
            length_ratio: 0.0,
            readability_diff: 0.0,
            word_count_diff: 0.0,
            whitespace_ratio_diff: 0.0,
            negation_changed: false,
        };

        // Execute in dependency order
        for node in plan.execution_order() {
            match node {
                ExecutionNode::Metric(metric_type) => {
                    self.compute_metric(*metric_type, &mut pairwise, &original_metrics, &modified_metrics);
                }
                ExecutionNode::Analyzer(_name) => {
                    // Analyzers will be run later, just ensure metrics are ready
                }
                ExecutionNode::Classifier(_name) => {
                    // Classifiers will be run later, just ensure metrics are ready
                }
            }
        }

        // Store the computed metrics
        result.metrics = Some(pairwise);
    }

    /// Compute a specific metric type
    fn compute_metric(
        &self,
        metric_type: crate::execution::MetricType,
        pairwise: &mut crate::metrics::PairwiseMetrics,
        original: &crate::metrics::TextMetrics,
        modified: &crate::metrics::TextMetrics,
    ) {
        use crate::execution::MetricType;

        match metric_type {
            // Pairwise comparison metrics
            MetricType::CharSimilarity => {
                let lev_dist = pairwise.levenshtein_distance;
                let max_len = original.char_count.max(modified.char_count);
                pairwise.char_similarity = if max_len > 0 {
                    1.0 - (lev_dist as f64 / max_len as f64)
                } else {
                    1.0
                };
            }
            MetricType::WordOverlap => {
                // Compute word overlap (Jaccard similarity)
                use std::collections::HashSet;
                let words1: HashSet<_> = original.text()
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
                    .collect();
                let words2: HashSet<_> = modified.text()
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
                    .collect();

                let intersection = words1.intersection(&words2).count();
                let union = words1.union(&words2).count();
                pairwise.word_overlap = if union > 0 {
                    intersection as f64 / union as f64
                } else if words1.is_empty() && words2.is_empty() {
                    1.0
                } else {
                    0.0
                };
            }
            MetricType::LevenshteinDistance => {
                pairwise.levenshtein_distance = levenshtein_distance(original.text(), modified.text());
            }
            MetricType::LengthRatio => {
                let len1 = original.char_count as f64;
                let len2 = modified.char_count as f64;
                pairwise.length_ratio = if len1 == 0.0 && len2 == 0.0 {
                    1.0
                } else {
                    let max_len = len1.max(len2);
                    let min_len = len1.min(len2);
                    if max_len == 0.0 { 0.0 } else { min_len / max_len }
                };
            }
            MetricType::ReadabilityDiff => {
                pairwise.readability_diff = (modified.flesch_reading_ease - original.flesch_reading_ease).abs();
            }
            MetricType::WordCountDiff => {
                pairwise.word_count_diff = (modified.word_count as f64 - original.word_count as f64).abs();
            }
            MetricType::WhitespaceRatioDiff => {
                pairwise.whitespace_ratio_diff = (modified.whitespace_ratio - original.whitespace_ratio).abs();
            }
            MetricType::NegationChanged => {
                pairwise.negation_changed = original.has_negation != modified.has_negation;
            }
            MetricType::SemanticSimilarity => {
                // Semantic similarity uses word overlap
                // Already computed if WordOverlap was in the dependency graph
            }
            // Text-level metrics are already computed in TextMetrics::compute()
            _ => {
                // Other metrics are handled by TextMetrics::compute() already
            }
        }
    }

    /// Normalize and tokenize both input strings
    fn normalize_and_tokenize(&self, original: &str, modified: &str) -> (Vec<Token>, Vec<Token>) {
        // Get or create pipeline
        let pipeline = self
            .config
            .pipeline
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.default_pipeline());

        // Process both texts through pipeline
        let original_layers = pipeline.process(original);
        let modified_layers = pipeline.process(modified);

        // Get or create tokenizer
        let tokenizer = self
            .config
            .tokenizer
            .as_ref()
            .map(|t| t.clone_box())
            .unwrap_or_else(|| Box::new(self.default_tokenizer()));

        // Tokenize
        let original_tokens = tokenizer.tokenize(&original_layers);
        let modified_tokens = tokenizer.tokenize(&modified_layers);

        (original_tokens, modified_tokens)
    }

    /// Compute diff operations using the configured algorithm
    fn compute_diff_operations(&self, original: &[Token], modified: &[Token]) -> Vec<crate::diff::DiffOperation> {
        let algorithm: Box<dyn DiffAlgoTrait> = match self.config.algorithm {
            DiffAlgorithm::Myers => Box::new(MyersAlgorithm::new()),
            DiffAlgorithm::Patience => Box::new(PatienceAlgorithm::new()),
            DiffAlgorithm::Histogram => Box::new(HistogramAlgorithm::new()),
            DiffAlgorithm::LCS => Box::new(MyersAlgorithm::new()), // Use Myers for LCS
        };

        algorithm.compute(original, modified)
    }

    /// Compute semantic similarity using cached metrics
    fn compute_semantic_similarity(&self, result: &DiffResult) -> f64 {
        // Use cached/computed metrics
        result.get_metrics_ref().word_overlap
    }

    /// Analyze stylistic changes using cached metrics
    fn analyze_style_change(&self, result: &DiffResult) -> f64 {
        let metrics = result.get_metrics_ref();
        let length_change = (metrics.original.char_count as f64 - metrics.modified.char_count as f64).abs()
            / metrics.original.char_count.max(metrics.modified.char_count).max(1) as f64;

        let punct_change = (metrics.original.punctuation_count as f64 - metrics.modified.punctuation_count as f64).abs()
            / metrics.original.punctuation_count.max(metrics.modified.punctuation_count).max(1) as f64;

        (length_change + punct_change) / 2.0
    }

    /// Analyze readability changes using cached metrics
    fn analyze_readability_change(&self, result: &DiffResult) -> f64 {
        let metrics = result.get_metrics_ref();
        metrics.modified.avg_word_length - metrics.original.avg_word_length
    }

    /// Classify edit operations into categories
    fn classify_edit_operations(&self, result: &mut DiffResult) {
        // First check if the overall change is just case-only
        let is_case_only_change = result.original_text.to_lowercase() == result.modified_text.to_lowercase()
            && result.original_text != result.modified_text;

        for op in &mut result.operations {
            // Simple rule-based classification
            let category = match op.edit_type {
                EditType::Equal => ChangeCategory::Unknown,
                EditType::Insert | EditType::Delete => {
                    // If the entire diff is case-only, classify as formatting
                    if is_case_only_change {
                        ChangeCategory::Formatting
                    } else {
                        ChangeCategory::Semantic
                    }
                }
                EditType::Modify => {
                    // Check for semantic vs stylistic vs syntactic
                    if let (Some(orig), Some(modified)) = (&op.original_text, &op.modified_text) {
                        classify_change(orig, modified)
                    } else {
                        ChangeCategory::Semantic
                    }
                }
            };

            op.category = category;
            op.confidence = 0.8; // Placeholder confidence
        }
    }

    /// Get default pipeline
    fn default_pipeline(&self) -> TextPipeline {
        let mut pipeline = TextPipeline::new();
        
        if self.config.ignore_case {
            pipeline = pipeline.add_normalizer(Box::new(Lowercase));
        }
        
        pipeline
    }

    /// Get default tokenizer
    fn default_tokenizer(&self) -> WordTokenizer {
        WordTokenizer::new()
    }
}

impl Default for DiffEngine {
    fn default() -> Self {
        Self::default_config()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn classify_change(original: &str, modified: &str) -> ChangeCategory {
    let orig_lower = original.to_lowercase();
    let mod_lower = modified.to_lowercase();
    
    // If only case changed, it's formatting
    if orig_lower == mod_lower {
        return ChangeCategory::Formatting;
    }
    
    // If words are similar (Levenshtein distance is small), likely syntactic
    let edit_distance = levenshtein_distance(&orig_lower, &mod_lower);
    let max_len = original.len().max(modified.len());
    
    if max_len > 0 {
        let similarity = 1.0 - (edit_distance as f64 / max_len as f64);
        
        if similarity > 0.8 {
            ChangeCategory::Syntactic
        } else if similarity > 0.5 {
            ChangeCategory::Stylistic
        } else {
            ChangeCategory::Semantic
        }
    } else {
        ChangeCategory::Semantic
    }
}

fn levenshtein_distance(s1: &str, s2: &str) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_diff() {
        let engine = DiffEngine::default();
        let result = engine.diff("hello world", "hello rust");
        
        assert!(!result.is_empty());
        assert!(result.statistics.insertions > 0 || result.statistics.deletions > 0);
    }

    #[test]
    fn test_identical_text() {
        let engine = DiffEngine::default();
        let result = engine.diff("hello world", "hello world");
        
        assert_eq!(result.statistics.insertions, 0);
        assert_eq!(result.statistics.deletions, 0);
        assert_eq!(result.semantic_similarity, 1.0);
    }

    #[test]
    fn test_semantic_similarity() {
        let engine = DiffEngine::default();
        let result = engine.diff("The cat sat on the mat", "The dog sat on the mat");

        // Should be high similarity since only one word changed (5 out of 6 unique words match)
        assert!(result.semantic_similarity > 0.6);
    }

    #[test]
    fn test_with_normalization() {
        let config = DiffConfig::default()
            .with_pipeline(TextPipeline::new().add_normalizer(Box::new(Lowercase)))
            .with_ignore_case(true);
        
        let engine = DiffEngine::new(config);
        let result = engine.diff("HELLO WORLD", "hello world");
        
        // Should recognize as same after normalization
        assert!(result.semantic_similarity > 0.9);
    }

    #[test]
    fn test_change_classification() {
        let engine = DiffEngine::default();
        let result = engine.diff("Hello World", "hello world");

        // Should classify case change as formatting
        let changed_ops = result.changed_operations();
        if !changed_ops.is_empty() {
            // At least one operation should be classified
            assert!(changed_ops.iter().any(|op| matches!(
                op.category,
                ChangeCategory::Formatting | ChangeCategory::Syntactic
            )));
        }
    }

    #[test]
    fn test_automatic_execution_plan() {
        // Test that execution plan is automatically built based on config
        let config = DiffConfig::default()
            .with_semantic_similarity(true)
            .with_edit_classification(true)
            .with_style_analysis(true);

        let engine = DiffEngine::new(config);
        let result = engine.diff("hello world", "hello rust");

        // Verify metrics were computed automatically
        assert!(result.metrics.is_some());
        let metrics = result.metrics.unwrap();

        // Check that the metrics were computed in dependency order
        assert!(metrics.levenshtein_distance > 0); // Should have computed Levenshtein
        assert!(metrics.char_similarity > 0.0); // Should have computed similarity using Levenshtein
        assert!(metrics.word_overlap > 0.0); // Should have computed word overlap
        assert!(metrics.readability_diff >= 0.0); // Should have computed readability diff
    }

    #[test]
    fn test_minimal_execution_plan() {
        // Test that minimal config only computes necessary metrics
        let config = DiffConfig::minimal();

        let engine = DiffEngine::new(config);
        let result = engine.diff("hello world", "hello rust");

        // Should still compute base metrics
        assert!(result.metrics.is_some());
        let metrics = result.metrics.unwrap();

        assert!(metrics.word_overlap >= 0.0);
        assert!(metrics.char_similarity > 0.0);
    }
}
