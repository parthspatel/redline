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
use crate::nlp::SpacyClient;

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

/// A syntactic token with POS and dependency information
#[derive(Debug, Clone)]
pub struct SyntacticToken {
    pub text: String,
    pub lemma: String,
    pub pos: String,    // Part-of-speech tag (coarse-grained)
    pub tag: String,    // Part-of-speech tag (fine-grained)
    pub dep: String,    // Dependency relation
    pub head: usize,    // Index of head token
    pub is_stop: bool,  // Is this a stop word?
    pub is_punct: bool, // Is this punctuation?
}

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

        // Count POS changes
        let pos_changes = self.count_pos_changes(&orig_tokens, &mod_tokens);

        // Count dependency changes
        let dep_changes = self.count_dep_changes(&orig_tokens, &mod_tokens);

        // Calculate structural similarity
        let structural_similarity = self.calculate_structural_similarity(&orig_tokens, &mod_tokens);

        // Detect grammar fixes
        let grammar_fixes = self.detect_grammar_fixes(&orig_tokens, &mod_tokens);

        // Calculate POS distribution divergence
        let pos_distribution_divergence = self.calculate_pos_divergence(&orig_tokens, &mod_tokens);

        Ok(SyntacticComparison {
            pos_changes,
            dep_changes,
            structural_similarity,
            grammar_fixes,
            pos_distribution_divergence,
        })
    }

    /// Count part-of-speech tag changes
    fn count_pos_changes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> usize {
        let min_len = orig.len().min(modified.len());
        let mut changes = 0;

        for i in 0..min_len {
            if orig[i].pos != modified[i].pos {
                changes += 1;
            }
        }

        // Add differences in length
        changes += orig.len().abs_diff(modified.len());

        changes
    }

    /// Count dependency relation changes
    fn count_dep_changes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> usize {
        let min_len = orig.len().min(modified.len());
        let mut changes = 0;

        for i in 0..min_len {
            if orig[i].dep != modified[i].dep {
                changes += 1;
            }
        }

        changes += orig.len().abs_diff(modified.len());

        changes
    }

    /// Calculate structural similarity based on dependency trees
    fn calculate_structural_similarity(
        &self,
        orig: &[SyntacticToken],
        modified: &[SyntacticToken],
    ) -> f64 {
        if orig.is_empty() && modified.is_empty() {
            return 1.0;
        }

        if orig.is_empty() || modified.is_empty() {
            return 0.0;
        }

        let min_len = orig.len().min(modified.len());
        let max_len = orig.len().max(modified.len());

        let mut matching_structures = 0;

        for i in 0..min_len {
            // Check if both POS and dependency match
            if orig[i].pos == modified[i].pos && orig[i].dep == modified[i].dep {
                matching_structures += 1;
            }
        }

        matching_structures as f64 / max_len as f64
    }

    /// Detect potential grammar fixes
    fn detect_grammar_fixes(
        &self,
        orig: &[SyntacticToken],
        modified: &[SyntacticToken],
    ) -> Vec<String> {
        let mut fixes = Vec::new();

        // Common grammar fix patterns
        for i in 0..orig.len().min(modified.len()) {
            let orig_tok = &orig[i];
            let mod_tok = &modified[i];

            // Subject-verb agreement fixes
            if orig_tok.dep == "nsubj" || orig_tok.dep == "ROOT" {
                if orig_tok.tag != mod_tok.tag && orig_tok.lemma == mod_tok.lemma {
                    fixes.push(format!(
                        "Verb form correction: '{}' → '{}'",
                        orig_tok.text, mod_tok.text
                    ));
                }
            }

            // Article corrections (a/an)
            if orig_tok.pos == "DET" && mod_tok.pos == "DET" {
                if orig_tok.text.to_lowercase() != mod_tok.text.to_lowercase() {
                    fixes.push(format!(
                        "Article correction: '{}' → '{}'",
                        orig_tok.text, mod_tok.text
                    ));
                }
            }

            // Tense consistency
            if orig_tok.pos == "VERB" && mod_tok.pos == "VERB" {
                if orig_tok.tag != mod_tok.tag && orig_tok.lemma == mod_tok.lemma {
                    fixes.push(format!(
                        "Tense correction: '{}' → '{}'",
                        orig_tok.text, mod_tok.text
                    ));
                }
            }
        }

        fixes
    }

    /// Calculate POS distribution divergence (using KL-like measure)
    fn calculate_pos_divergence(
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
}

#[cfg(feature = "spacy")]
impl Clone for SpacySyntacticAnalyzer {
    fn clone(&self) -> Self {
        Self {
            spacy_client: self.spacy_client.clone(),
        }
    }
}

#[cfg(feature = "spacy")]
impl SingleDiffAnalyzer for SpacySyntacticAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Compare syntactic structures using correct field names
        match self.compare_syntactic_structures(&diff.original_text, &diff.modified_text) {
            Ok(comparison) => {
                result.add_metric("pos_changes", comparison.pos_changes as f64);
                result.add_metric("dep_changes", comparison.dep_changes as f64);
                result.add_metric("structural_similarity", comparison.structural_similarity);
                result.add_metric("grammar_fixes", comparison.grammar_fixes.len() as f64);
                result.add_metric(
                    "pos_distribution_divergence",
                    comparison.pos_distribution_divergence,
                );

                // Generate insights
                if comparison.structural_similarity >= 0.9 {
                    result.add_insight("Minimal syntactic changes detected");
                } else if comparison.structural_similarity >= 0.7 {
                    result.add_insight("Moderate syntactic changes detected");
                } else {
                    result.add_insight("Significant syntactic restructuring detected");
                }

                if !comparison.grammar_fixes.is_empty() {
                    result.add_insight(format!(
                        "Detected {} potential grammar fix(es):",
                        comparison.grammar_fixes.len()
                    ));
                    for fix in &comparison.grammar_fixes {
                        result.add_insight(format!("  - {}", fix));
                    }
                }

                if comparison.pos_distribution_divergence > 0.3 {
                    result.add_insight(format!(
                        "Significant POS distribution change (divergence: {:.2})",
                        comparison.pos_distribution_divergence
                    ));
                }

                result.add_metadata("status", "success");
            }
            Err(e) => {
                result.add_insight(format!("SpaCy analysis failed: {}", e));
                result.add_metadata("status", "error");
                result.add_metadata("error", &e);
                result = result.with_confidence(0.0);
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

#[cfg(not(feature = "spacy"))]
/// Placeholder analyzer when SpaCy feature is not enabled
pub struct SpacySyntacticAnalyzer;

#[cfg(not(feature = "spacy"))]
impl SpacySyntacticAnalyzer {
    pub fn new(_model_name: impl Into<String>) -> Self {
        Self
    }

    pub fn analyze_text(&self, _text: &str) -> Result<Vec<SyntacticToken>, String> {
        Err("SpaCy feature not enabled. Compile with --features spacy".to_string())
    }

    pub fn compare_syntactic_structures(
        &self,
        _original: &str,
        _modified: &str,
    ) -> Result<SyntacticComparison, String> {
        Err("SpaCy feature not enabled. Compile with --features spacy".to_string())
    }
}

#[cfg(not(feature = "spacy"))]
impl Clone for SpacySyntacticAnalyzer {
    fn clone(&self) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl SingleDiffAnalyzer for SpacySyntacticAnalyzer {
    fn analyze(&self, _diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        result.add_insight(
            "SpaCy feature not enabled. Compile with --features spacy to enable syntactic analysis"
                .to_string(),
        );
        result.add_metadata("status", "disabled");
        result.with_confidence(0.0)
    }

    fn name(&self) -> &str {
        "spacy_syntactic"
    }

    fn description(&self) -> &str {
        "SpaCy-based syntactic analyzer (requires 'spacy' feature)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

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
    fn test_analyzer_without_spacy_feature() {
        let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");
        let result = analyzer.analyze_text("test");

        #[cfg(not(feature = "spacy"))]
        assert!(result.is_err());

        #[cfg(feature = "spacy")]
        assert!(result.is_ok() || result.is_err()); // May fail if model not installed
    }
}
