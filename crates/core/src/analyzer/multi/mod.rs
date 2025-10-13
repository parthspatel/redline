//! Multi-diff analyzers for aggregate analysis
//!
//! Analyzers that operate on collections of diff results to find patterns and trends

pub mod clustering;
pub mod patterns;
pub mod statistics;
pub mod trends;

// Re-export all analyzers
pub use clustering::*;
pub use patterns::*;
pub use statistics::*;
pub use trends::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::MultiDiffAnalyzer;
    use crate::DiffEngine;

    #[test]
    fn test_aggregate_statistics() {
        let engine = DiffEngine::default();
        let diff1 = engine.diff("hello world", "hello rust");
        let diff2 = engine.diff("foo bar", "foo baz");

        let diffs = vec![&diff1, &diff2];

        let analyzer = AggregateStatisticsAnalyzer::new();
        let result = analyzer.analyze(&diffs);

        assert!(result.metrics.contains_key("total_diffs"));
        assert_eq!(result.metrics["total_diffs"], 2.0);
    }

    #[test]
    fn test_pattern_detection() {
        let engine = DiffEngine::default();
        // Use examples that clearly show expansion with word tokenization
        let diff1 = engine.diff("hello", "hello world"); // Expansion (adds word)
        let diff2 = engine.diff("foo", "foo bar"); // Expansion (adds word)
        let diff3 = engine.diff("test", "test case"); // Expansion (adds word)

        let diffs = vec![&diff1, &diff2, &diff3];

        let analyzer = PatternDetectionAnalyzer::new();
        let result = analyzer.analyze(&diffs);

        // Should detect expansion pattern
        assert!(result.insights.iter().any(|i| i.contains("expansion")));
    }
}
