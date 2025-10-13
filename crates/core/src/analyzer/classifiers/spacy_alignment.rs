//! Token alignment analyzer using SpaCy
//!
//! This analyzer computes and caches token alignments between original and modified text.
//! Other SpaCy analyzers can then reuse this cached alignment instead of recomputing it.

#[cfg(feature = "spacy")]
use crate::nlp::SpacyClient;

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

#[cfg(feature = "spacy")]
use crate::analyzer::classifiers::SyntacticToken;
#[cfg(feature = "spacy")]
use crate::token_alignment;

#[cfg(feature = "spacy")]
/// SpaCy-based token alignment analyzer
///
/// This analyzer should run FIRST before other SpaCy analyzers to compute
/// and cache the token alignment in PairwiseMetrics.
///
/// ## Metrics Provided
/// - **total_alignments**: Total number of alignment operations
/// - **matches**: Number of exact token matches
/// - **insertions**: Number of inserted tokens
/// - **deletions**: Number of deleted tokens
/// - **replacements**: Number of replaced tokens
/// - **alignment_ratio**: Ratio of matches to total tokens
pub struct SpacyAlignmentAnalyzer {
    spacy_client: SpacyClient,
}

#[cfg(feature = "spacy")]
impl SpacyAlignmentAnalyzer {
    /// Create a new alignment analyzer with the specified SpaCy model
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            spacy_client: SpacyClient::new(model_name),
        }
    }

    /// Compute alignment statistics
    fn compute_alignment_stats(
        &self,
        alignments: &[token_alignment::TokenAlignment],
    ) -> (usize, usize, usize, usize) {
        let mut matches = 0;
        let mut insertions = 0;
        let mut deletions = 0;
        let mut replacements = 0;

        for alignment in alignments {
            match alignment {
                token_alignment::TokenAlignment::Match { .. } => matches += 1,
                token_alignment::TokenAlignment::Insertion { .. } => insertions += 1,
                token_alignment::TokenAlignment::Deletion { .. } => deletions += 1,
                token_alignment::TokenAlignment::Replacement { .. } => replacements += 1,
            }
        }

        (matches, insertions, deletions, replacements)
    }
}

#[cfg(feature = "spacy")]
impl Clone for SpacyAlignmentAnalyzer {
    fn clone(&self) -> Self {
        Self {
            spacy_client: self.spacy_client.clone(),
        }
    }
}

#[cfg(feature = "spacy")]
impl SingleDiffAnalyzer for SpacyAlignmentAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Get or compute tokens
        match (
            self.spacy_client.analyze(&diff.original_text),
            self.spacy_client.analyze(&diff.modified_text),
        ) {
            (Ok(orig_tokens), Ok(mod_tokens)) => {
                // Compute alignment
                let alignments = token_alignment::align_tokens(&orig_tokens, &mod_tokens);

                // Compute statistics
                let (matches, insertions, deletions, replacements) =
                    self.compute_alignment_stats(&alignments);
                let total = alignments.len();
                let alignment_ratio = if total > 0 {
                    matches as f64 / total as f64
                } else {
                    1.0
                };

                // Add metrics
                result.add_metric("total_alignments", total as f64);
                result.add_metric("matches", matches as f64);
                result.add_metric("insertions", insertions as f64);
                result.add_metric("deletions", deletions as f64);
                result.add_metric("replacements", replacements as f64);
                result.add_metric("alignment_ratio", alignment_ratio);

                // Generate insights
                if matches == total {
                    result.add_insight("All tokens aligned perfectly");
                } else {
                    result.add_insight(format!(
                        "Token alignment: {} matches, {} insertions, {} deletions, {} replacements",
                        matches, insertions, deletions, replacements
                    ));
                }

                if alignment_ratio >= 0.9 {
                    result.add_insight("Very high token alignment (>90%)");
                } else if alignment_ratio >= 0.7 {
                    result.add_insight("High token alignment (70-90%)");
                } else if alignment_ratio >= 0.5 {
                    result.add_insight("Moderate token alignment (50-70%)");
                } else {
                    result.add_insight("Low token alignment (<50%)");
                }

                // Store alignment in diff metrics for reuse by other analyzers
                // Note: We can't mutate diff here since analyze() takes &DiffResult
                // The alignment will be stored by the execution system or manually
                result.add_metadata("has_alignment", "true");
                result.add_metadata("status", "success");

                // Note: The actual caching needs to happen at a higher level
                // where we have mutable access to DiffResult.metrics
                result.add_insight(
                    "Note: Alignment computed but not cached (requires mutable access to metrics)"
                        .to_string(),
                );
            }
            (Err(e), _) | (_, Err(e)) => {
                result.add_insight(format!("Token alignment failed: {}", e));
                result.add_metadata("status", "error");
                result.add_metadata("error", &e);
                result = result.with_confidence(0.0);
            }
        }

        result
    }

    fn name(&self) -> &str {
        "spacy_alignment"
    }

    fn description(&self) -> &str {
        "Computes token alignment between original and modified text using SpaCy"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Non-SpaCy fallback implementation
// ============================================================================

#[cfg(not(feature = "spacy"))]
pub struct SpacyAlignmentAnalyzer;

#[cfg(not(feature = "spacy"))]
impl SpacyAlignmentAnalyzer {
    pub fn new(_model_name: impl Into<String>) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl Clone for SpacyAlignmentAnalyzer {
    fn clone(&self) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl SingleDiffAnalyzer for SpacyAlignmentAnalyzer {
    fn analyze(&self, _diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        result.add_insight("SpaCy feature not enabled. Compile with --features spacy".to_string());
        result.add_metadata("status", "disabled");
        result.with_confidence(0.0)
    }

    fn name(&self) -> &str {
        "spacy_alignment"
    }

    fn description(&self) -> &str {
        "SpaCy-based alignment analyzer (requires 'spacy' feature)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
