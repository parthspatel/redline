//! Structural similarity analysis using SpaCy
//!
//! This analyzer measures syntactic structural similarity based on dependency trees.

#[cfg(feature = "spacy")]
use crate::nlp::SpacyClient;

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

#[cfg(feature = "spacy")]
use crate::analyzer::classifiers::SyntacticToken;
#[cfg(feature = "spacy")]
use crate::token_alignment;

#[cfg(feature = "spacy")]
/// SpaCy-based structural similarity analyzer
///
/// Measures how similar two texts are in their syntactic structure.
///
/// ## Metrics Provided
/// - **structural_similarity**: Overall syntactic similarity (0.0 to 1.0)
/// - **pos_dep_match_ratio**: Ratio of tokens with matching POS and dependency
/// - **sentence_structure_change**: Whether sentence structure was significantly altered
pub struct SpacyStructuralAnalyzer {
    spacy_client: SpacyClient,
}

#[cfg(feature = "spacy")]
impl SpacyStructuralAnalyzer {
    /// Create a new structural analyzer with the specified SpaCy model
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            spacy_client: SpacyClient::new(model_name),
        }
    }

    /// Analyze structure of text
    pub fn analyze_structure(&self, text: &str) -> Result<Vec<SyntacticToken>, String> {
        self.spacy_client.analyze(text)
    }

    /// Calculate structural similarity based on dependency trees
    ///
    /// This measures how similar the syntactic structures are by comparing
    /// both POS tags and dependency relations using proper token alignment.
    ///
    /// Returns a value between 0.0 (completely different) and 1.0 (identical)
    pub fn calculate_structural_similarity(
        &self,
        orig: &[SyntacticToken],
        modified: &[SyntacticToken],
    ) -> f64 {
        let alignments = token_alignment::align_tokens(orig, modified);
        token_alignment::calculate_structural_similarity_aligned(orig, modified, &alignments)
    }

    /// Calculate the ratio of tokens with matching POS and dependency
    ///
    /// Uses proper token alignment to match tokens by content, not position
    pub fn pos_dep_match_ratio(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> f64 {
        if orig.is_empty() || modified.is_empty() {
            return 0.0;
        }

        let alignments = token_alignment::align_tokens(orig, modified);
        let mut matches = 0;

        for alignment in &alignments {
            if let token_alignment::TokenAlignment::Match { orig_idx, mod_idx } = alignment {
                if orig[*orig_idx].pos == modified[*mod_idx].pos
                    && orig[*orig_idx].dep == modified[*mod_idx].dep
                {
                    matches += 1;
                }
            }
        }

        let total = alignments.len();
        if total > 0 {
            matches as f64 / total as f64
        } else {
            1.0
        }
    }

    /// Detect if sentence structure was significantly changed
    pub fn has_significant_structural_change(
        &self,
        orig: &[SyntacticToken],
        modified: &[SyntacticToken],
    ) -> bool {
        let similarity = self.calculate_structural_similarity(orig, modified);
        similarity < 0.7
    }

    /// Get clause structure (main verbs and their subjects)
    pub fn get_clause_structure(&self, tokens: &[SyntacticToken]) -> Vec<String> {
        let mut clauses = Vec::new();

        for token in tokens {
            if token.dep == "ROOT" {
                // Find the subject of this root verb
                let subject = tokens.iter().find(|t| {
                    t.dep == "nsubj"
                        && t.head == tokens.iter().position(|x| x.text == token.text).unwrap()
                });

                if let Some(subj) = subject {
                    clauses.push(format!("{} [{}]", subj.text, token.text));
                } else {
                    clauses.push(format!("[{}]", token.text));
                }
            }
        }

        clauses
    }
}

#[cfg(feature = "spacy")]
impl Clone for SpacyStructuralAnalyzer {
    fn clone(&self) -> Self {
        Self {
            spacy_client: self.spacy_client.clone(),
        }
    }
}

#[cfg(feature = "spacy")]
impl SingleDiffAnalyzer for SpacyStructuralAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Analyze structure
        match (
            self.spacy_client.analyze(&diff.original_text),
            self.spacy_client.analyze(&diff.modified_text),
        ) {
            (Ok(orig_tokens), Ok(mod_tokens)) => {
                let structural_similarity =
                    self.calculate_structural_similarity(&orig_tokens, &mod_tokens);
                let pos_dep_match = self.pos_dep_match_ratio(&orig_tokens, &mod_tokens);
                let significant_change =
                    self.has_significant_structural_change(&orig_tokens, &mod_tokens);

                result.add_metric("structural_similarity", structural_similarity);
                result.add_metric("pos_dep_match_ratio", pos_dep_match);
                result.add_metric(
                    "sentence_structure_change",
                    if significant_change { 1.0 } else { 0.0 },
                );

                // Generate insights
                if structural_similarity >= 0.9 {
                    result.add_insight("Minimal syntactic changes detected");
                } else if structural_similarity >= 0.7 {
                    result.add_insight("Moderate syntactic changes detected");
                } else if structural_similarity >= 0.5 {
                    result.add_insight("Significant syntactic changes detected");
                } else {
                    result.add_insight("Major syntactic restructuring detected");
                }

                if significant_change {
                    result.add_insight("Sentence structure was significantly altered");

                    // Show clause structure if available
                    let orig_clauses = self.get_clause_structure(&orig_tokens);
                    let mod_clauses = self.get_clause_structure(&mod_tokens);

                    if !orig_clauses.is_empty() || !mod_clauses.is_empty() {
                        result.add_insight(format!("Original clauses: {:?}", orig_clauses));
                        result.add_insight(format!("Modified clauses: {:?}", mod_clauses));
                    }
                }

                // Provide detailed breakdown
                result.add_insight(format!(
                    "Structural match: {:.1}% of tokens have identical POS+dependency",
                    pos_dep_match * 100.0
                ));

                result.add_metadata("status", "success");
            }
            (Err(e), _) | (_, Err(e)) => {
                result.add_insight(format!("Structural analysis failed: {}", e));
                result.add_metadata("status", "error");
                result.add_metadata("error", &e);
                result = result.with_confidence(0.0);
            }
        }

        result
    }

    fn name(&self) -> &str {
        "spacy_structural"
    }

    fn description(&self) -> &str {
        "Analyzes structural similarity using SpaCy's dependency parsing"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Non-SpaCy fallback implementation
// ============================================================================

#[cfg(not(feature = "spacy"))]
pub struct SpacyStructuralAnalyzer;

#[cfg(not(feature = "spacy"))]
impl SpacyStructuralAnalyzer {
    pub fn new(_model_name: impl Into<String>) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl Clone for SpacyStructuralAnalyzer {
    fn clone(&self) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl SingleDiffAnalyzer for SpacyStructuralAnalyzer {
    fn analyze(&self, _diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        result.add_insight("SpaCy feature not enabled. Compile with --features spacy".to_string());
        result.add_metadata("status", "disabled");
        result.with_confidence(0.0)
    }

    fn name(&self) -> &str {
        "spacy_structural"
    }

    fn description(&self) -> &str {
        "SpaCy-based structural analyzer (requires 'spacy' feature)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
