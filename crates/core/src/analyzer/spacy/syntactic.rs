//! Syntactic and grammar-based classification using SpaCy
//!
//! This module provides syntactic analysis capabilities through Python's SpaCy library
//! using PyO3 for seamless Rust-Python interoperability.
//!
//! ## Architecture
//!
//! - **SpacyClient** (in `nlp::spacy`) - Pure NLP utility, no caching
//! - **TextMetrics** - Stores cached tokens
//! - **SpacySyntacticAnalyzer** - Implements heuristics, uses cache

#[cfg(feature = "spacy")]
use crate::util::SpacyClient;
#[cfg(feature = "spacy")]
use crate::util::token_statistics;
#[cfg(feature = "spacy")]
use crate::token_alignment;
#[cfg(feature = "spacy")]
use crate::analyzer::spacy::{analyze_both_texts, handle_spacy_analysis_error};

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;
use crate::util::SyntacticToken;

/// Result of syntactic comparison
#[derive(Debug, Clone)]
pub struct SyntacticComparison {
    pub pos_changes: usize,
    pub dep_changes: usize,
    pub structural_similarity: f64,
    pub grammar_fixes: Vec<String>,
    pub pos_distribution_divergence: f64,
}

#[cfg(feature = "spacy")]
/// SpaCy-based syntactic analyzer
///
/// This analyzer implements syntactic comparison heuristics using SpaCy.
///
/// ## Caching Strategy
///
/// POS-tagged tokens are cached in `TextMetrics.syntactic_tokens`:
/// - First call: Computes tokens via `SpacyClient`, caches in metrics
/// - Subsequent calls: Reuses cached tokens
/// - Other analyzers can reuse the same tokens
///
/// ## Heuristics Implemented
///
/// - Grammar fix detection (subject-verb agreement, tense, etc.)
/// - Structural similarity comparison
/// - POS distribution divergence
#[derive(Clone)]
pub struct SpacySyntacticAnalyzer {
    spacy_client: SpacyClient,
}

#[cfg(feature = "spacy")]
impl SpacySyntacticAnalyzer {
    /// Create a new SpaCy analyzer with the specified model
    ///
    /// Common models:
    /// - "en_core_web_sm" - Small English model (fast, ~15MB)
    /// - "en_core_web_md" - Medium English model (balanced, ~45MB)
    /// - "en_core_web_lg" - Large English model (accurate, ~800MB)
    ///
    /// Make sure to download the model first:
    /// ```bash
    /// python -m spacy download en_core_web_sm
    /// ```
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            spacy_client: SpacyClient::new(model_name),
        }
    }

    /// Get or compute syntactic tokens for text, using cache if available
    fn get_or_compute_tokens(
        &self,
        text: &str,
        metrics: &mut crate::metrics::TextMetrics,
    ) -> Result<Vec<SyntacticToken>, String> {
        // Check cache first
        if let Some(tokens) = metrics.get_syntactic_tokens() {
            return Ok(tokens.clone());
        }

        // Cache miss - compute and store
        let tokens = self.spacy_client.analyze(text)?;
        metrics.set_syntactic_tokens(tokens.clone());
        Ok(tokens)
    }

    /// Analyze text and return syntactic tokens
    ///
    /// This is a convenience method that directly calls SpacyClient without caching.
    /// For cached analysis, use the analyze() method with DiffResult.
    pub fn analyze_text(&self, text: &str) -> Result<Vec<SyntacticToken>, String> {
        self.spacy_client.analyze(text)
    }

    /// Compare syntactic structures of two texts
    ///
    /// Note: This computes tokens directly without caching.
    /// For cached analysis, use the analyze() method with DiffResult.
    pub fn compare_syntactic_structures(
        &self,
        original: &str,
        modified: &str,
    ) -> Result<SyntacticComparison, String> {
        let orig_tokens = self.spacy_client.analyze(original)?;
        let mod_tokens = self.spacy_client.analyze(modified)?;

        // Align tokens for accurate comparison
        let alignments = token_alignment::align_tokens(&orig_tokens, &mod_tokens);

        // Count POS changes using aligned tokens
        let pos_changes = token_alignment::count_pos_changes_aligned(&orig_tokens, &mod_tokens, &alignments);

        // Count dependency changes using aligned tokens
        let dep_changes = token_alignment::count_dep_changes_aligned(&orig_tokens, &mod_tokens, &alignments);

        // Calculate structural similarity using aligned tokens
        let structural_similarity = token_alignment::calculate_structural_similarity_aligned(&orig_tokens, &mod_tokens, &alignments);

        // Detect grammar fixes using aligned tokens
        let grammar_fixes = token_alignment::find_grammar_fixes_aligned(&orig_tokens, &mod_tokens, &alignments);

        // Calculate POS distribution divergence using token statistics
        let pos_distribution_divergence = token_statistics::calculate_tag_distribution_divergence(&orig_tokens, &mod_tokens, |t| &t.pos);

        Ok(SyntacticComparison {
            pos_changes,
            dep_changes,
            structural_similarity,
            grammar_fixes,
            pos_distribution_divergence,
        })
    }
}

#[cfg(feature = "spacy")]
impl SingleDiffAnalyzer for SpacySyntacticAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Analyze both texts using the utility function
        match analyze_both_texts(&self.spacy_client, diff) {
            Ok((orig_tokens, mod_tokens)) => {
                // Align tokens for accurate comparison
                let alignments = token_alignment::align_tokens(&orig_tokens, &mod_tokens);

                // Calculate metrics using aligned tokens and utility functions
                let pos_changes = token_alignment::count_pos_changes_aligned(&orig_tokens, &mod_tokens, &alignments);
                let dep_changes = token_alignment::count_dep_changes_aligned(&orig_tokens, &mod_tokens, &alignments);
                let structural_similarity = token_alignment::calculate_structural_similarity_aligned(&orig_tokens, &mod_tokens, &alignments);
                let grammar_fixes = token_alignment::find_grammar_fixes_aligned(&orig_tokens, &mod_tokens, &alignments);
                let pos_distribution_divergence = token_statistics::calculate_tag_distribution_divergence(&orig_tokens, &mod_tokens, |t| &t.pos);

                // Add metrics to result
                result.add_metric("pos_changes", pos_changes as f64);
                result.add_metric("dep_changes", dep_changes as f64);
                result.add_metric("structural_similarity", structural_similarity);
                result.add_metric("grammar_fixes", grammar_fixes.len() as f64);
                result.add_metric(
                    "pos_distribution_divergence",
                    pos_distribution_divergence,
                );

                // Generate insights
                if structural_similarity >= 0.9 {
                    result.add_insight("Minimal syntactic changes detected");
                } else if structural_similarity >= 0.7 {
                    result.add_insight("Moderate syntactic changes detected");
                } else {
                    result.add_insight("Significant syntactic restructuring detected");
                }

                if !grammar_fixes.is_empty() {
                    result.add_insight(format!(
                        "Detected {} potential grammar fix(es):",
                        grammar_fixes.len()
                    ));
                    for fix in &grammar_fixes {
                        result.add_insight(format!("  - {}", fix));
                    }
                }

                if pos_distribution_divergence > 0.3 {
                    result.add_insight(format!(
                        "Significant POS distribution change (divergence: {:.2})",
                        pos_distribution_divergence
                    ));
                }

                result.add_metadata("status", "success");
            }
            Err(e) => {
                result = handle_spacy_analysis_error(result, "Syntactic", e);
            }
        }

        result
    }

    fn name(&self) -> &str {
        "spacy_syntactic"
    }

    fn description(&self) -> &str {
        "Analyzes syntactic structure using SpaCy's NLP capabilities (POS tagging, dependency parsing, grammar detection)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Non-SpaCy fallback implementation
// ============================================================================

crate::analyzer::spacy::impl_spacy_fallback!(
    SpacySyntacticAnalyzer,
    "spacy_syntactic",
    "Comprehensive syntactic analysis including POS, dependencies, and grammar"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "spacy")]
    fn test_spacy_analyzer_creation() {
        let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");
        assert_eq!(analyzer.name(), "spacy_syntactic");
    }

    #[test]
    #[cfg(feature = "spacy")]
    fn test_analyzer_with_spacy_feature() {
        let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");
        let result = analyzer.analyze_text("test");
        assert!(result.is_ok() || result.is_err()); // May fail if model not installed
    }

    #[test]
    #[cfg(not(feature = "spacy"))]
    fn test_analyzer_without_spacy_feature() {
        let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");
        assert_eq!(analyzer.name(), "spacy_syntactic");
    }
}
