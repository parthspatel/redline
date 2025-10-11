//! Configuration for the diff engine

use crate::pipeline::TextPipeline;
use crate::tokenizers::Tokenizer;
use crate::execution::ExecutionPlan;

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

    /// Single diff analyzers to run
    pub(crate) analyzers: Vec<Box<dyn crate::analyzers::SingleDiffAnalyzer>>,

    /// Change classifiers to run
    pub(crate) classifiers: Vec<Box<dyn crate::classifiers::ChangeClassifier>>,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self::new()
    }
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
            analyzers: Vec::new(),
            classifiers: Vec::new(),
        }
    }

    /// Add an analyzer to the configuration
    pub fn add_analyzer(mut self, analyzer: Box<dyn crate::analyzers::SingleDiffAnalyzer>) -> Self {
        self.analyzers.push(analyzer);
        self
    }

    /// Add a classifier to the configuration
    pub fn add_classifier(mut self, classifier: Box<dyn crate::classifiers::ChangeClassifier>) -> Self {
        self.classifiers.push(classifier);
        self
    }

    /// Build an execution plan based on configured analyzers and classifiers
    /// This automatically determines which metrics need to be computed and in what order
    pub fn build_execution_plan(&self) -> ExecutionPlan {
        use crate::execution::{ExecutionPlanBuilder, ExecutionNode, MetricType, NodeDependencies};

        let mut builder = ExecutionPlanBuilder::new();

        // Always compute base text metrics (these have no dependencies)
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::CharCount)));
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::WordCount)));
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::SentenceCount)));
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::SyllableCount)));

        // Derived metrics that depend on base metrics
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::FleschReadingEase))
            .with_dependencies(vec![
                ExecutionNode::Metric(MetricType::WordCount),
                ExecutionNode::Metric(MetricType::SentenceCount),
                ExecutionNode::Metric(MetricType::SyllableCount),
            ]));

        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::AvgWordLength))
            .with_dependency(ExecutionNode::Metric(MetricType::WordCount)));

        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::AvgSentenceLength))
            .with_dependencies(vec![
                ExecutionNode::Metric(MetricType::WordCount),
                ExecutionNode::Metric(MetricType::SentenceCount),
            ]));

        // Linguistic features
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::StopwordRatio))
            .with_dependency(ExecutionNode::Metric(MetricType::WordCount)));
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::WhitespaceRatio))
            .with_dependency(ExecutionNode::Metric(MetricType::CharCount)));
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::HasNegation)));

        // Pairwise comparison metrics
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::LevenshteinDistance)));

        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::CharSimilarity))
            .with_dependency(ExecutionNode::Metric(MetricType::LevenshteinDistance)));

        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::WordOverlap))
            .with_dependency(ExecutionNode::Metric(MetricType::WordCount)));

        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::LengthRatio))
            .with_dependency(ExecutionNode::Metric(MetricType::CharCount)));

        // Add dependencies from configured analyzers
        for analyzer in &self.analyzers {
            for dep in analyzer.dependencies() {
                builder.add_node(dep);
            }
        }

        // Add dependencies from configured classifiers
        for classifier in &self.classifiers {
            for dep in classifier.dependencies() {
                builder.add_node(dep);
            }
        }

        // Legacy flag-based dependencies (for backward compatibility)
        if self.analyze_style || self.classify_edits {
            builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::ReadabilityDiff))
                .with_dependency(ExecutionNode::Metric(MetricType::FleschReadingEase)));

            builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::WhitespaceRatioDiff))
                .with_dependency(ExecutionNode::Metric(MetricType::WhitespaceRatio)));
        }

        if self.classify_edits {
            builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::WordCountDiff))
                .with_dependency(ExecutionNode::Metric(MetricType::WordCount)));

            builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::NegationChanged))
                .with_dependency(ExecutionNode::Metric(MetricType::HasNegation)));
        }

        // Semantic similarity depends on word overlap
        if self.compute_semantic_similarity {
            builder.add_node(NodeDependencies::new(ExecutionNode::Metric(MetricType::SemanticSimilarity))
                .with_dependency(ExecutionNode::Metric(MetricType::WordOverlap)));
        }

        // Build and return the plan
        builder.build().expect("Failed to build execution plan - cyclic dependency detected")
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
            analyzers: Vec::new(),
            classifiers: Vec::new(),
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
            analyzers: Vec::new(),
            classifiers: Vec::new(),
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

    #[test]
    fn test_execution_plan_adapts_to_config() {
        use crate::execution::{ExecutionNode, MetricType};

        // Test that execution plan includes semantic similarity metrics when enabled
        let config_with_semantic = DiffConfig::new()
            .with_semantic_similarity(true);
        let plan = config_with_semantic.build_execution_plan();
        let metrics = plan.get_required_metrics();

        assert!(metrics.contains(&MetricType::WordOverlap));
        assert!(metrics.contains(&MetricType::SemanticSimilarity));

        // Test that execution plan excludes optional metrics when disabled
        let config_minimal = DiffConfig::minimal();
        let plan_minimal = config_minimal.build_execution_plan();
        let metrics_minimal = plan_minimal.get_required_metrics();

        // Should not include semantic similarity
        assert!(!metrics_minimal.contains(&MetricType::SemanticSimilarity));

        // But should still include base metrics
        assert!(metrics_minimal.contains(&MetricType::WordCount));
        assert!(metrics_minimal.contains(&MetricType::CharCount));
    }

    #[test]
    fn test_execution_plan_includes_style_metrics() {
        use crate::execution::MetricType;

        // When style analysis is enabled, should include readability metrics
        let config = DiffConfig::new()
            .with_style_analysis(true);
        let plan = config.build_execution_plan();
        let metrics = plan.get_required_metrics();

        assert!(metrics.contains(&MetricType::ReadabilityDiff));
        assert!(metrics.contains(&MetricType::FleschReadingEase));
    }

    #[test]
    fn test_execution_plan_from_classifiers() {
        use crate::execution::MetricType;
        use crate::classifiers::naive_bayes::NaiveBayesClassifier;

        // When a classifier is added, its dependencies should be included in the plan
        let config = DiffConfig::new()
            .add_classifier(Box::new(NaiveBayesClassifier::new()));

        let plan = config.build_execution_plan();
        let metrics = plan.get_required_metrics();

        // NaiveBayesClassifier requires these metrics
        assert!(metrics.contains(&MetricType::CharSimilarity));
        assert!(metrics.contains(&MetricType::WordOverlap));
        assert!(metrics.contains(&MetricType::ReadabilityDiff));
        assert!(metrics.contains(&MetricType::WordCountDiff));
        assert!(metrics.contains(&MetricType::WhitespaceRatioDiff));
        assert!(metrics.contains(&MetricType::NegationChanged));
    }
}
