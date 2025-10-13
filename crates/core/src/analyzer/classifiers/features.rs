//! Feature analyzers that cache metrics from StandardFeatureExtractor

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

/// Analyzer that computes and caches character similarity metric
#[derive(Clone)]
pub struct CharSimilarityAnalyzer;

impl CharSimilarityAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CharSimilarityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for CharSimilarityAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Use cached metrics if available
        let char_similarity = diff.get_metrics_ref().char_similarity;
        result.add_metric("char_similarity", char_similarity);

        result
    }

    fn name(&self) -> &str {
        "char_similarity"
    }

    fn description(&self) -> &str {
        "Computes character-level similarity using Levenshtein distance"
    }

    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        use crate::execution::{ExecutionNode, MetricType, NodeDependencies};

        vec![
            NodeDependencies::new(ExecutionNode::Metric(MetricType::CharSimilarity))
                .with_dependency(ExecutionNode::Metric(MetricType::LevenshteinDistance)),
        ]
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

/// Analyzer that computes and caches word count difference
#[derive(Clone)]
pub struct WordCountDiffAnalyzer;

impl WordCountDiffAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WordCountDiffAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for WordCountDiffAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Use cached metrics if available
        let metrics = diff.get_metrics_ref();
        result.add_metric("word_count_diff", metrics.word_count_diff);
        result.add_metric("original_word_count", metrics.original.word_count as f64);
        result.add_metric("modified_word_count", metrics.modified.word_count as f64);

        result
    }

    fn name(&self) -> &str {
        "word_count_diff"
    }

    fn description(&self) -> &str {
        "Computes absolute difference in word counts"
    }

    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        use crate::execution::{ExecutionNode, MetricType, NodeDependencies};

        vec![NodeDependencies::new(ExecutionNode::Metric(
            MetricType::WordCountDiff,
        ))]
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

/// Analyzer that computes and caches negation change detection
#[derive(Clone)]
pub struct NegationChangedAnalyzer;

impl NegationChangedAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NegationChangedAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for NegationChangedAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Use cached metrics if available
        let metrics = diff.get_metrics_ref();
        result.add_metric(
            "negation_changed",
            if metrics.negation_changed { 1.0 } else { 0.0 },
        );
        result.add_metric(
            "original_has_negation",
            if metrics.original.has_negation {
                1.0
            } else {
                0.0
            },
        );
        result.add_metric(
            "modified_has_negation",
            if metrics.modified.has_negation {
                1.0
            } else {
                0.0
            },
        );

        if metrics.negation_changed {
            result.add_insight("Negation state changed between texts");
        }

        result
    }

    fn name(&self) -> &str {
        "negation_changed"
    }

    fn description(&self) -> &str {
        "Detects if negation was added or removed"
    }

    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        use crate::execution::{ExecutionNode, MetricType, NodeDependencies};

        vec![
            NodeDependencies::new(ExecutionNode::Metric(MetricType::NegationChanged))
                .with_dependency(ExecutionNode::Metric(MetricType::HasNegation)),
        ]
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
