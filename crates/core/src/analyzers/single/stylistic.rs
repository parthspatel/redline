//! Stylistic analysis - word length, sentence structure, vocabulary

use crate::analyzers::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::{ChangeCategory, DiffResult};

/// Analyzes stylistic changes in the text
#[derive(Clone)]
pub struct StylisticAnalyzer;

impl StylisticAnalyzer {
    pub fn new() -> Self {
        Self
    }

    fn average_word_length(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        let total: usize = words.iter().map(|w| w.len()).sum();
        total as f64 / words.len() as f64
    }

    fn average_sentence_length(&self, text: &str) -> f64 {
        let sentences = text.split(|c| matches!(c, '.' | '!' | '?')).count();
        let words = text.split_whitespace().count();

        if sentences == 0 {
            return 0.0;
        }

        words as f64 / sentences as f64
    }

    fn punctuation_density(&self, text: &str) -> f64 {
        let punct_count = text.chars().filter(|c| c.is_ascii_punctuation()).count();
        let total_chars = text.len();

        if total_chars == 0 {
            return 0.0;
        }

        punct_count as f64 / total_chars as f64
    }

    fn lexical_diversity(&self, text: &str) -> f64 {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        if words.is_empty() {
            return 0.0;
        }

        let unique_words: std::collections::HashSet<_> = words.iter().collect();
        unique_words.len() as f64 / words.len() as f64
    }
}

impl Default for StylisticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for StylisticAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Compute stylistic metrics
        let orig_awl = self.average_word_length(&diff.original_text);
        let mod_awl = self.average_word_length(&diff.modified_text);

        let orig_asl = self.average_sentence_length(&diff.original_text);
        let mod_asl = self.average_sentence_length(&diff.modified_text);

        let orig_pd = self.punctuation_density(&diff.original_text);
        let mod_pd = self.punctuation_density(&diff.modified_text);

        let orig_ld = self.lexical_diversity(&diff.original_text);
        let mod_ld = self.lexical_diversity(&diff.modified_text);

        // Add metrics
        result.add_metric("original_avg_word_length", orig_awl);
        result.add_metric("modified_avg_word_length", mod_awl);
        result.add_metric("avg_word_length_change", mod_awl - orig_awl);

        result.add_metric("original_avg_sentence_length", orig_asl);
        result.add_metric("modified_avg_sentence_length", mod_asl);
        result.add_metric("avg_sentence_length_change", mod_asl - orig_asl);

        result.add_metric("original_punctuation_density", orig_pd);
        result.add_metric("modified_punctuation_density", mod_pd);
        result.add_metric("punctuation_density_change", mod_pd - orig_pd);

        result.add_metric("original_lexical_diversity", orig_ld);
        result.add_metric("modified_lexical_diversity", mod_ld);
        result.add_metric("lexical_diversity_change", mod_ld - orig_ld);

        // Count stylistic changes
        let stylistic_changes = diff.operations.iter()
            .filter(|op| matches!(op.category, ChangeCategory::Stylistic))
            .count();

        result.add_metric("stylistic_operations", stylistic_changes as f64);

        // Generate insights
        if (mod_awl - orig_awl).abs() > 0.5 {
            if mod_awl > orig_awl {
                result.add_insight("Vocabulary became more sophisticated (longer words)");
            } else {
                result.add_insight("Vocabulary became simpler (shorter words)");
            }
        }

        if (mod_asl - orig_asl).abs() > 2.0 {
            if mod_asl > orig_asl {
                result.add_insight("Sentences became longer and more complex");
            } else {
                result.add_insight("Sentences became shorter and more concise");
            }
        }

        if (mod_ld - orig_ld).abs() > 0.1 {
            if mod_ld > orig_ld {
                result.add_insight("Vocabulary diversity increased (more varied word choice)");
            } else {
                result.add_insight("Vocabulary diversity decreased (more repetitive)");
            }
        }

        result
    }

    fn name(&self) -> &str {
        "stylistic"
    }

    fn description(&self) -> &str {
        "Analyzes stylistic properties like word length, sentence structure, and vocabulary"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

/// Analyzer that computes and caches whitespace ratio difference
#[derive(Clone)]
pub struct WhitespaceRatioDiffAnalyzer;

impl WhitespaceRatioDiffAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WhitespaceRatioDiffAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for WhitespaceRatioDiffAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Use cached metrics if available
        let metrics = diff.get_metrics_ref();
        result.add_metric("whitespace_ratio_diff", metrics.whitespace_ratio_diff);
        result.add_metric("original_whitespace_ratio", metrics.original.whitespace_ratio);
        result.add_metric("modified_whitespace_ratio", metrics.modified.whitespace_ratio);

        result
    }

    fn name(&self) -> &str {
        "whitespace_ratio_diff"
    }

    fn description(&self) -> &str {
        "Computes difference in whitespace ratios"
    }

    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        use crate::execution::{ExecutionNode, MetricType, NodeDependencies};

        vec![
            NodeDependencies::new(ExecutionNode::Metric(MetricType::WhitespaceRatioDiff))
                .with_dependency(ExecutionNode::Metric(MetricType::WhitespaceRatio)),
        ]
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
