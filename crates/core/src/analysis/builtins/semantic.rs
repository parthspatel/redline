//! SemanticAnalyzer: heuristic-based similarity scoring per edit operation.

use core::any::Any;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, string::ToString, vec::Vec};

use hashbrown::HashSet;

use crate::analysis::{
    AnalysisContext, AnalysisError, AnalysisReport, AnalyzerDependency, AnalyzerMeta,
    AnalyzerPlugin,
};
use crate::diff::edit_operation::EditKind;

/// Per-operation similarity score.
#[derive(Debug, Clone)]
pub struct OperationSimilarity {
    pub operation_index: usize,
    pub similarity: f64,
    pub method: String,
}

/// Result of semantic analysis.
#[derive(Debug, Clone)]
pub struct SemanticResult {
    pub operation_scores: Vec<OperationSimilarity>,
    pub overall_similarity: f64,
}

/// Pluggable scoring backend (BERT-ready: exactly 2 methods).
pub trait ScoringBackend: Send + Sync {
    fn score(&self, source_tokens: &[&str], target_tokens: &[&str]) -> f64;
    fn name(&self) -> &str;
}

/// Default heuristic backend using Jaccard similarity on token sets.
pub struct HeuristicScoring;

impl ScoringBackend for HeuristicScoring {
    fn name(&self) -> &str {
        "heuristic_jaccard"
    }

    fn score(&self, source_tokens: &[&str], target_tokens: &[&str]) -> f64 {
        if source_tokens.is_empty() && target_tokens.is_empty() {
            return 1.0;
        }
        if source_tokens.is_empty() || target_tokens.is_empty() {
            return 0.0;
        }
        let a: HashSet<&str> = source_tokens.iter().copied().collect();
        let b: HashSet<&str> = target_tokens.iter().copied().collect();
        let intersection = a.intersection(&b).count();
        let union = a.union(&b).count();
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

/// Semantic analyzer scoring per-operation similarity.
pub struct SemanticAnalyzer {
    backend: Box<dyn ScoringBackend>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            backend: Box::new(HeuristicScoring),
        }
    }

    pub fn with_backend(backend: Box<dyn ScoringBackend>) -> Self {
        Self { backend }
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyzerPlugin for SemanticAnalyzer {
    fn id(&self) -> &str {
        "semantic"
    }

    fn meta(&self) -> AnalyzerMeta {
        AnalyzerMeta {
            id: "semantic".into(),
            name: "Semantic Analyzer".into(),
            version: "1.0.0".into(),
        }
    }

    fn dependencies(&self) -> Vec<AnalyzerDependency> {
        Vec::new()
    }

    fn cost(&self) -> f32 {
        0.3
    }

    fn analyze(
        &self,
        context: &AnalysisContext,
        _prior_results: &AnalysisReport,
    ) -> Result<Box<dyn Any + Send + Sync>, AnalysisError> {
        let changes = context.change_operations();
        let mut operation_scores = Vec::with_capacity(changes.len());

        for (idx, op) in &changes {
            let similarity = match op.kind {
                EditKind::Replace => {
                    let source_tokens = context.source_tokens_for_op(*idx).unwrap_or(&[]);
                    let target_tokens = context.target_tokens_for_op(*idx).unwrap_or(&[]);

                    // Extract text slices for each token
                    let src_text = &context.diff.source.normalized;
                    let tgt_text = &context.diff.target.normalized;

                    let src_strs: Vec<&str> = source_tokens
                        .iter()
                        .filter_map(|t| {
                            let s = t.span.start() as usize;
                            let e = (t.span.end() as usize).min(src_text.len());
                            if s <= e { Some(&src_text[s..e]) } else { None }
                        })
                        .collect();
                    let tgt_strs: Vec<&str> = target_tokens
                        .iter()
                        .filter_map(|t| {
                            let s = t.span.start() as usize;
                            let e = (t.span.end() as usize).min(tgt_text.len());
                            if s <= e { Some(&tgt_text[s..e]) } else { None }
                        })
                        .collect();

                    self.backend.score(&src_strs, &tgt_strs)
                }
                EditKind::Insert | EditKind::Delete => 0.0,
                EditKind::Equal => 1.0,
            };

            operation_scores.push(OperationSimilarity {
                operation_index: *idx,
                similarity,
                method: self.backend.name().to_string(),
            });
        }

        let overall_similarity = if operation_scores.is_empty() {
            1.0
        } else {
            operation_scores.iter().map(|s| s.similarity).sum::<f64>()
                / operation_scores.len() as f64
        };

        Ok(Box::new(SemanticResult {
            operation_scores,
            overall_similarity,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_identical_tokens() {
        let score = HeuristicScoring.score(&["a", "b", "c"], &["a", "b", "c"]);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn heuristic_disjoint_tokens() {
        let score = HeuristicScoring.score(&["a", "b"], &["x", "y"]);
        assert!(score.abs() < f64::EPSILON);
    }

    #[test]
    fn heuristic_partial_overlap() {
        let score = HeuristicScoring.score(&["a", "b", "c"], &["b", "c", "d"]);
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn heuristic_both_empty() {
        let score = HeuristicScoring.score(&[], &[]);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn heuristic_one_empty() {
        assert!(HeuristicScoring.score(&["a"], &[]).abs() < f64::EPSILON);
        assert!(HeuristicScoring.score(&[], &["a"]).abs() < f64::EPSILON);
    }

    #[test]
    fn semantic_analyzer_id() {
        assert_eq!(SemanticAnalyzer::new().id(), "semantic");
    }

    #[test]
    fn semantic_analyzer_no_dependencies() {
        assert!(SemanticAnalyzer::new().dependencies().is_empty());
    }

    #[test]
    fn custom_backend() {
        struct ConstantBackend(f64);
        impl ScoringBackend for ConstantBackend {
            fn score(&self, _: &[&str], _: &[&str]) -> f64 {
                self.0
            }
            fn name(&self) -> &str {
                "constant"
            }
        }
        let analyzer = SemanticAnalyzer::with_backend(Box::new(ConstantBackend(0.42)));
        assert_eq!(analyzer.id(), "semantic");
    }
}
