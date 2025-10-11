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

    /// Create a diff engine with default configuration
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

        // Step 1: Apply normalization pipeline
        let (original_tokens, modified_tokens) = self.normalize_and_tokenize(original, modified);

        // Step 2: Run diff algorithm
        let operations = self.compute_diff_operations(&original_tokens, &modified_tokens);

        // Step 3: Add operations to result
        for op in operations {
            result.add_operation(op);
        }

        // Step 4: Perform analysis if enabled
        if self.config.compute_semantic_similarity {
           result.analysis.semantic_similarity =
                self.compute_semantic_similarity(original, modified);
        }

        if self.config.analyze_style {
            result.analysis.stylistic_change = self.analyze_style_change(original, modified);
            result.analysis.readability_change = self.analyze_readability_change(original, modified);
        }

        if self.config.classify_edits {
            self.classify_edit_operations(&mut result);
        }

        // Step 5: Finalize result
        result.finalize();

        result
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

    /// Compute semantic similarity between two texts
    fn compute_semantic_similarity(&self, original: &str, modified: &str) -> f64 {
        // Simple word-based similarity for now
        // In a real implementation, you'd use sentence embeddings
        
        let orig_words: std::collections::HashSet<_> = original
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        let mod_words: std::collections::HashSet<_> = modified
            .split_whitespace()
            .map(|s| s.to_lowercase())
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

    /// Analyze stylistic changes
    fn analyze_style_change(&self, original: &str, modified: &str) -> f64 {
        // Simple analysis based on length and punctuation
        let orig_len = original.len() as f64;
        let mod_len = modified.len() as f64;
        
        let length_change = (orig_len - mod_len).abs() / orig_len.max(mod_len).max(1.0);
        
        let orig_punct = original.chars().filter(|c| c.is_ascii_punctuation()).count() as f64;
        let mod_punct = modified.chars().filter(|c| c.is_ascii_punctuation()).count() as f64;
        
        let punct_change = (orig_punct - mod_punct).abs() / orig_punct.max(mod_punct).max(1.0);
        
        (length_change + punct_change) / 2.0
    }

    /// Analyze readability changes
    fn analyze_readability_change(&self, original: &str, modified: &str) -> f64 {
        // Simple readability based on average word length
        let orig_avg_word_len = average_word_length(original);
        let mod_avg_word_len = average_word_length(modified);
        
        mod_avg_word_len - orig_avg_word_len
    }

    /// Classify edit operations into categories
    fn classify_edit_operations(&self, result: &mut DiffResult) {
        for op in &mut result.operations {
            // Simple rule-based classification
            let category = match op.edit_type {
                EditType::Equal => ChangeCategory::Unknown,
                EditType::Insert | EditType::Delete | EditType::Modify => {
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

fn average_word_length(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    
    if words.is_empty() {
        return 0.0;
    }
    
    let total_len: usize = words.iter().map(|w| w.len()).sum();
    total_len as f64 / words.len() as f64
}

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
        
        // Should be high similarity since only one word changed
        assert!(result.semantic_similarity > 0.7);
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
}
