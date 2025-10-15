//! Part-of-speech (POS) tag analysis using SpaCy
//!
//! This analyzer focuses on POS tag changes and distributions.

#[cfg(feature = "spacy")]
use crate::util::SpacyClient;

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

use crate::util::SyntacticToken;
#[cfg(feature = "spacy")]
use crate::token_alignment;
#[cfg(feature = "spacy")]
use crate::util::token_statistics;
#[cfg(feature = "spacy")]
use crate::analyzer::spacy::{analyze_both_texts, handle_spacy_analysis_error};

#[cfg(feature = "spacy")]
/// SpaCy-based POS tag analyzer
///
/// Analyzes changes in part-of-speech tags between original and modified text.
///
/// ## Metrics Provided
/// - **pos_changes**: Number of POS tag changes
/// - **pos_distribution_divergence**: Divergence between POS distributions (0.0 to 1.0)
/// - **pos_unchanged_ratio**: Ratio of tokens with unchanged POS tags
#[derive(Clone)]
pub struct SpacyPOSAnalyzer {
    spacy_client: SpacyClient,
}

#[cfg(feature = "spacy")]
impl SpacyPOSAnalyzer {
    /// Create a new POS analyzer with the specified SpaCy model
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            spacy_client: SpacyClient::new(model_name),
        }
    }

    /// Analyze POS tags for text
    pub fn analyze_pos(&self, text: &str) -> Result<Vec<SyntacticToken>, String> {
        self.spacy_client.analyze(text)
    }

    /// Count part-of-speech tag changes between two token sequences
    ///
    /// Uses proper token alignment to match tokens by content, not position
    pub fn count_pos_changes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> usize {
        let alignments = token_alignment::align_tokens(orig, modified);
        token_alignment::count_pos_changes_aligned(orig, modified, &alignments)
    }

    /// Calculate POS distribution divergence using symmetric divergence measure
    ///
    /// Returns a value between 0.0 (identical distributions) and 1.0 (completely different)
    pub fn calculate_pos_divergence(
        &self,
        orig: &[SyntacticToken],
        modified: &[SyntacticToken],
    ) -> f64 {
        token_statistics::calculate_tag_distribution_divergence(orig, modified, |t| &t.pos)
    }

    /// Get POS distribution for tokens
    pub fn get_pos_distribution(
        &self,
        tokens: &[SyntacticToken],
    ) -> std::collections::HashMap<String, usize> {
        token_statistics::get_token_tag_distribution(tokens, |t| &t.pos)
    }
}

#[cfg(feature = "spacy")]
impl SingleDiffAnalyzer for SpacyPOSAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Analyze POS tags using helper function
        match analyze_both_texts(&self.spacy_client, diff) {
            Ok((orig_tokens, mod_tokens)) => {
                let pos_changes = self.count_pos_changes(&orig_tokens, &mod_tokens);
                let pos_divergence = self.calculate_pos_divergence(&orig_tokens, &mod_tokens);

                // Calculate unchanged ratio using utility
                let unchanged_ratio = token_statistics::calculate_unchanged_ratio(
                    &orig_tokens,
                    &mod_tokens,
                    |a, b| a.pos == b.pos,
                );

                result.add_metric("pos_changes", pos_changes as f64);
                result.add_metric("pos_distribution_divergence", pos_divergence);
                result.add_metric("pos_unchanged_ratio", unchanged_ratio);

                // Generate insights using utility
                result.add_insight(token_statistics::categorize_change_magnitude(
                    pos_changes,
                    "POS tag",
                ));

                if pos_divergence > 0.3 {
                    result.add_insight(format!(
                        "Significant POS distribution shift (divergence: {:.2})",
                        pos_divergence
                    ));

                    // Show distribution details
                    let orig_dist = self.get_pos_distribution(&orig_tokens);
                    let mod_dist = self.get_pos_distribution(&mod_tokens);

                    result.add_insight("Original POS distribution:".to_string());
                    for (pos, count) in &orig_dist {
                        result.add_insight(format!("  - {}: {}", pos, count));
                    }

                    result.add_insight("Modified POS distribution:".to_string());
                    for (pos, count) in &mod_dist {
                        result.add_insight(format!("  - {}: {}", pos, count));
                    }
                }

                result.add_metadata("status", "success");
            }
            Err(e) => {
                result = handle_spacy_analysis_error(result, "POS", e);
            }
        }

        result
    }

    fn name(&self) -> &str {
        "spacy_pos"
    }

    fn description(&self) -> &str {
        "Analyzes part-of-speech tag changes and distributions using SpaCy"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Non-SpaCy fallback implementation
// ============================================================================

crate::analyzer::spacy::impl_spacy_fallback!(
    SpacyPOSAnalyzer,
    "spacy_pos",
    "Analyzes part-of-speech tag changes and distributions using SpaCy"
);
