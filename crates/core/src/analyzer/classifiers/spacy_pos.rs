//! Part-of-speech (POS) tag analysis using SpaCy
//!
//! This analyzer focuses on POS tag changes and distributions.

#[cfg(feature = "spacy")]
use crate::nlp::SpacyClient;

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

#[cfg(feature = "spacy")]
use crate::analyzer::classifiers::SyntacticToken;
#[cfg(feature = "spacy")]
use crate::token_alignment;

#[cfg(feature = "spacy")]
/// SpaCy-based POS tag analyzer
///
/// Analyzes changes in part-of-speech tags between original and modified text.
///
/// ## Metrics Provided
/// - **pos_changes**: Number of POS tag changes
/// - **pos_distribution_divergence**: Divergence between POS distributions (0.0 to 1.0)
/// - **pos_unchanged_ratio**: Ratio of tokens with unchanged POS tags
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
        use std::collections::HashMap;

        if orig.is_empty() && modified.is_empty() {
            return 0.0;
        }

        if orig.is_empty() || modified.is_empty() {
            return 1.0;
        }

        // Count POS tags
        let mut orig_counts: HashMap<String, usize> = HashMap::new();
        let mut mod_counts: HashMap<String, usize> = HashMap::new();

        for token in orig {
            *orig_counts.entry(token.pos.clone()).or_insert(0) += 1;
        }

        for token in modified {
            *mod_counts.entry(token.pos.clone()).or_insert(0) += 1;
        }

        // Normalize to probabilities
        let orig_total = orig.len() as f64;
        let mod_total = modified.len() as f64;

        // Calculate symmetric divergence (similar to JS divergence)
        let all_pos: std::collections::HashSet<String> = orig_counts
            .keys()
            .chain(mod_counts.keys())
            .cloned()
            .collect();

        let mut divergence = 0.0;

        for pos in all_pos {
            let p_orig = *orig_counts.get(&pos).unwrap_or(&0) as f64 / orig_total;
            let p_mod = *mod_counts.get(&pos).unwrap_or(&0) as f64 / mod_total;

            // Avoid log(0)
            if p_orig > 0.0 && p_mod > 0.0 {
                divergence += (p_orig - p_mod).abs();
            } else {
                divergence += p_orig.max(p_mod);
            }
        }

        // Normalize to [0, 1]
        divergence / 2.0
    }

    /// Get POS distribution for tokens
    pub fn get_pos_distribution(
        &self,
        tokens: &[SyntacticToken],
    ) -> std::collections::HashMap<String, usize> {
        let mut distribution = std::collections::HashMap::new();
        for token in tokens {
            if !token.is_punct {
                *distribution.entry(token.pos.clone()).or_insert(0) += 1;
            }
        }
        distribution
    }
}

#[cfg(feature = "spacy")]
impl Clone for SpacyPOSAnalyzer {
    fn clone(&self) -> Self {
        Self {
            spacy_client: self.spacy_client.clone(),
        }
    }
}

#[cfg(feature = "spacy")]
impl SingleDiffAnalyzer for SpacyPOSAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Analyze POS tags
        match (
            self.spacy_client.analyze(&diff.original_text),
            self.spacy_client.analyze(&diff.modified_text),
        ) {
            (Ok(orig_tokens), Ok(mod_tokens)) => {
                let pos_changes = self.count_pos_changes(&orig_tokens, &mod_tokens);
                let pos_divergence = self.calculate_pos_divergence(&orig_tokens, &mod_tokens);

                // Calculate unchanged ratio
                let min_len = orig_tokens.len().min(mod_tokens.len());
                let unchanged = (0..min_len)
                    .filter(|&i| orig_tokens[i].pos == mod_tokens[i].pos)
                    .count();
                let unchanged_ratio = if min_len > 0 {
                    unchanged as f64 / min_len as f64
                } else {
                    1.0
                };

                result.add_metric("pos_changes", pos_changes as f64);
                result.add_metric("pos_distribution_divergence", pos_divergence);
                result.add_metric("pos_unchanged_ratio", unchanged_ratio);

                // Generate insights
                if pos_changes == 0 {
                    result.add_insight("No POS tag changes detected");
                } else if pos_changes <= 2 {
                    result.add_insight(format!(
                        "Minimal POS changes: {} tag(s) changed",
                        pos_changes
                    ));
                } else if pos_changes <= 5 {
                    result.add_insight(format!(
                        "Moderate POS changes: {} tag(s) changed",
                        pos_changes
                    ));
                } else {
                    result.add_insight(format!(
                        "Significant POS changes: {} tag(s) changed",
                        pos_changes
                    ));
                }

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
            (Err(e), _) | (_, Err(e)) => {
                result.add_insight(format!("POS analysis failed: {}", e));
                result.add_metadata("status", "error");
                result.add_metadata("error", &e);
                result = result.with_confidence(0.0);
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

#[cfg(not(feature = "spacy"))]
pub struct SpacyPOSAnalyzer;

#[cfg(not(feature = "spacy"))]
impl SpacyPOSAnalyzer {
    pub fn new(_model_name: impl Into<String>) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl Clone for SpacyPOSAnalyzer {
    fn clone(&self) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl SingleDiffAnalyzer for SpacyPOSAnalyzer {
    fn analyze(&self, _diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        result.add_insight("SpaCy feature not enabled. Compile with --features spacy".to_string());
        result.add_metadata("status", "disabled");
        result.with_confidence(0.0)
    }

    fn name(&self) -> &str {
        "spacy_pos"
    }

    fn description(&self) -> &str {
        "SpaCy-based POS analyzer (requires 'spacy' feature)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
