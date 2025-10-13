//! Semantic analysis - similarity and word overlap

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::{ChangeCategory, DiffResult};

/// Analyzes semantic similarity between original and modified text
#[derive(Clone)]
pub struct SemanticSimilarityAnalyzer {
    /// Threshold for considering texts semantically similar
    pub similarity_threshold: f64,
}

impl SemanticSimilarityAnalyzer {
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.7,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }
}

impl Default for SemanticSimilarityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for SemanticSimilarityAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        let similarity = diff.semantic_similarity;
        result.add_metric("semantic_similarity", similarity);
        result.add_metric("semantic_distance", 1.0 - similarity);

        // Classify similarity level
        let level = if similarity >= 0.9 {
            "very_high"
        } else if similarity >= self.similarity_threshold {
            "high"
        } else if similarity >= 0.5 {
            "moderate"
        } else if similarity >= 0.3 {
            "low"
        } else {
            "very_low"
        };

        result.add_metadata("similarity_level", level);

        // Add insights
        if similarity >= self.similarity_threshold {
            result.add_insight(format!(
                "Texts are semantically similar ({:.1}% similarity). Core meaning preserved.",
                similarity * 100.0
            ));
        } else {
            result.add_insight(format!(
                "Significant semantic changes detected ({:.1}% similarity). Meaning substantially altered.",
                similarity * 100.0
            ));
        }

        // Analyze by edit type
        let semantic_changes = diff
            .operations
            .iter()
            .filter(|op| matches!(op.category, ChangeCategory::Semantic))
            .count();

        if semantic_changes > 0 {
            result.add_insight(format!(
                "{} semantic change operations detected",
                semantic_changes
            ));
        }

        result
    }

    fn name(&self) -> &str {
        "semantic_similarity"
    }

    fn description(&self) -> &str {
        "Analyzes semantic similarity and meaning preservation"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

/// Analyzer that computes and caches word overlap metric
#[derive(Clone)]
pub struct WordOverlapAnalyzer;

impl WordOverlapAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WordOverlapAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for WordOverlapAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Use cached metrics if available
        let word_overlap = diff.get_metrics_ref().word_overlap;
        result.add_metric("word_overlap", word_overlap);

        result
    }

    fn name(&self) -> &str {
        "word_overlap"
    }

    fn description(&self) -> &str {
        "Computes word overlap (Jaccard similarity) between texts"
    }

    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        use crate::execution::{ExecutionNode, MetricType, NodeDependencies};

        vec![NodeDependencies::new(ExecutionNode::Metric(
            MetricType::WordOverlap,
        ))]
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
