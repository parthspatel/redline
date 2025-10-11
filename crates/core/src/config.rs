//! Configuration for the diff engine

use crate::pipeline::TextPipeline;
use crate::tokenizers::Tokenizer;

/// Diff algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffAlgorithm {
    /// Myers O(ND) algorithm (default, fast for small differences)
    Myers,
    /// Patience diff (better for structured text, more human-readable)
    Patience,
    /// Histogram diff (good balance of speed and quality)
    Histogram,
    /// Simple LCS-based algorithm
    LCS,
}

impl Default for DiffAlgorithm {
    fn default() -> Self {
        Self::Histogram
    }
}

/// Configuration for diff computation
#[derive(Default)]
pub struct DiffConfig {
    /// Algorithm to use for computing diffs
    pub algorithm: DiffAlgorithm,
    
    /// Text normalization pipeline
    pub pipeline: Option<TextPipeline>,
    
    /// Tokenizer to use
    pub tokenizer: Option<Box<dyn Tokenizer>>,
    
    /// Whether to compute semantic similarity
    pub compute_semantic_similarity: bool,
    
    /// Whether to classify edit types
    pub classify_edits: bool,
    
    /// Whether to analyze stylistic changes
    pub analyze_style: bool,
    
    /// Context lines around changes (for display)
    pub context_lines: usize,
    
    /// Ignore whitespace-only changes
    pub ignore_whitespace: bool,
    
    /// Ignore case when comparing
    pub ignore_case: bool,
}

impl DiffConfig {
    /// Create a new default configuration
    pub fn new() -> Self {
        Self {
            algorithm: DiffAlgorithm::default(),
            pipeline: None,
            tokenizer: None,
            compute_semantic_similarity: true,
            classify_edits: true,
            analyze_style: true,
            context_lines: 3,
            ignore_whitespace: false,
            ignore_case: false,
        }
    }

    /// Set the diff algorithm
    pub fn with_algorithm(mut self, algorithm: DiffAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the text normalization pipeline
    pub fn with_pipeline(mut self, pipeline: TextPipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    /// Set the tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Enable or disable semantic similarity computation
    pub fn with_semantic_similarity(mut self, enable: bool) -> Self {
        self.compute_semantic_similarity = enable;
        self
    }

    /// Enable or disable edit classification
    pub fn with_edit_classification(mut self, enable: bool) -> Self {
        self.classify_edits = enable;
        self
    }

    /// Enable or disable style analysis
    pub fn with_style_analysis(mut self, enable: bool) -> Self {
        self.analyze_style = enable;
        self
    }

    /// Set the number of context lines
    pub fn with_context_lines(mut self, lines: usize) -> Self {
        self.context_lines = lines;
        self
    }

    /// Set whether to ignore whitespace
    pub fn with_ignore_whitespace(mut self, ignore: bool) -> Self {
        self.ignore_whitespace = ignore;
        self
    }

    /// Set whether to ignore case
    pub fn with_ignore_case(mut self, ignore: bool) -> Self {
        self.ignore_case = ignore;
        self
    }

    /// Create a minimal configuration (fast, no analysis)
    pub fn minimal() -> Self {
        Self {
            algorithm: DiffAlgorithm::Myers,
            pipeline: None,
            tokenizer: None,
            compute_semantic_similarity: false,
            classify_edits: false,
            analyze_style: false,
            context_lines: 0,
            ignore_whitespace: false,
            ignore_case: false,
        }
    }

    /// Create a comprehensive configuration (all analysis enabled)
    pub fn comprehensive() -> Self {
        Self {
            algorithm: DiffAlgorithm::Histogram,
            pipeline: None,
            tokenizer: None,
            compute_semantic_similarity: true,
            classify_edits: true,
            analyze_style: true,
            context_lines: 5,
            ignore_whitespace: false,
            ignore_case: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DiffConfig::default();
        assert_eq!(config.algorithm, DiffAlgorithm::Histogram);
        assert!(config.compute_semantic_similarity);
    }

    #[test]
    fn test_minimal_config() {
        let config = DiffConfig::minimal();
        assert_eq!(config.algorithm, DiffAlgorithm::Myers);
        assert!(!config.compute_semantic_similarity);
        assert!(!config.classify_edits);
    }

    #[test]
    fn test_builder_pattern() {
        let config = DiffConfig::new()
            .with_algorithm(DiffAlgorithm::Patience)
            .with_context_lines(5)
            .with_ignore_case(true);
        
        assert_eq!(config.algorithm, DiffAlgorithm::Patience);
        assert_eq!(config.context_lines, 5);
        assert!(config.ignore_case);
    }
}
