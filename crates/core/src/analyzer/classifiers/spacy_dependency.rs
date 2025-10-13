//! Dependency relation analysis using SpaCy
//!
//! This analyzer focuses on dependency parsing and syntactic relations.

#[cfg(feature = "spacy")]
use crate::nlp::SpacyClient;

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

#[cfg(feature = "spacy")]
use crate::analyzer::classifiers::SyntacticToken;
#[cfg(feature = "spacy")]
use crate::token_alignment;

#[cfg(feature = "spacy")]
/// SpaCy-based dependency relation analyzer
///
/// Analyzes changes in dependency relations between original and modified text.
///
/// ## Metrics Provided
/// - **dep_changes**: Number of dependency relation changes
/// - **dep_unchanged_ratio**: Ratio of tokens with unchanged dependencies
/// - **root_changes**: Number of changes to ROOT dependencies
pub struct SpacyDependencyAnalyzer {
    spacy_client: SpacyClient,
}

#[cfg(feature = "spacy")]
impl SpacyDependencyAnalyzer {
    /// Create a new dependency analyzer with the specified SpaCy model
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            spacy_client: SpacyClient::new(model_name),
        }
    }

    /// Analyze dependencies for text
    pub fn analyze_dependencies(&self, text: &str) -> Result<Vec<SyntacticToken>, String> {
        self.spacy_client.analyze(text)
    }

    /// Count dependency relation changes between two token sequences
    ///
    /// Uses proper token alignment to match tokens by content, not position
    pub fn count_dep_changes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> usize {
        let alignments = token_alignment::align_tokens(orig, modified);
        token_alignment::count_dep_changes_aligned(orig, modified, &alignments)
    }

    /// Count ROOT dependency changes (main verb changes)
    pub fn count_root_changes(
        &self,
        orig: &[SyntacticToken],
        modified: &[SyntacticToken],
    ) -> usize {
        let orig_roots: Vec<_> = orig.iter().filter(|t| t.dep == "ROOT").collect();
        let mod_roots: Vec<_> = modified.iter().filter(|t| t.dep == "ROOT").collect();

        if orig_roots.len() != mod_roots.len() {
            return orig_roots.len().max(mod_roots.len());
        }

        orig_roots
            .iter()
            .zip(mod_roots.iter())
            .filter(|(o, m)| o.lemma != m.lemma)
            .count()
    }

    /// Get dependency distribution for tokens
    pub fn get_dep_distribution(
        &self,
        tokens: &[SyntacticToken],
    ) -> std::collections::HashMap<String, usize> {
        let mut distribution = std::collections::HashMap::new();
        for token in tokens {
            if !token.is_punct {
                *distribution.entry(token.dep.clone()).or_insert(0) += 1;
            }
        }
        distribution
    }

    /// Find dependency relation changes with detailed information
    pub fn find_dep_changes(
        &self,
        orig: &[SyntacticToken],
        modified: &[SyntacticToken],
    ) -> Vec<String> {
        let min_len = orig.len().min(modified.len());
        let mut changes = Vec::new();

        for i in 0..min_len {
            if orig[i].dep != modified[i].dep {
                changes.push(format!(
                    "'{}': {} → {}",
                    orig[i].text, orig[i].dep, modified[i].dep
                ));
            }
        }

        changes
    }
}

#[cfg(feature = "spacy")]
impl Clone for SpacyDependencyAnalyzer {
    fn clone(&self) -> Self {
        Self {
            spacy_client: self.spacy_client.clone(),
        }
    }
}

#[cfg(feature = "spacy")]
impl SingleDiffAnalyzer for SpacyDependencyAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Analyze dependencies
        match (
            self.spacy_client.analyze(&diff.original_text),
            self.spacy_client.analyze(&diff.modified_text),
        ) {
            (Ok(orig_tokens), Ok(mod_tokens)) => {
                let dep_changes = self.count_dep_changes(&orig_tokens, &mod_tokens);
                let root_changes = self.count_root_changes(&orig_tokens, &mod_tokens);

                // Calculate unchanged ratio
                let min_len = orig_tokens.len().min(mod_tokens.len());
                let unchanged = (0..min_len)
                    .filter(|&i| orig_tokens[i].dep == mod_tokens[i].dep)
                    .count();
                let unchanged_ratio = if min_len > 0 {
                    unchanged as f64 / min_len as f64
                } else {
                    1.0
                };

                result.add_metric("dep_changes", dep_changes as f64);
                result.add_metric("root_changes", root_changes as f64);
                result.add_metric("dep_unchanged_ratio", unchanged_ratio);

                // Generate insights
                if dep_changes == 0 {
                    result.add_insight("No dependency relation changes detected");
                } else if dep_changes <= 2 {
                    result.add_insight(format!(
                        "Minimal dependency changes: {} relation(s) changed",
                        dep_changes
                    ));
                } else if dep_changes <= 5 {
                    result.add_insight(format!(
                        "Moderate dependency changes: {} relation(s) changed",
                        dep_changes
                    ));
                } else {
                    result.add_insight(format!(
                        "Significant dependency changes: {} relation(s) changed",
                        dep_changes
                    ));
                }

                if root_changes > 0 {
                    result.add_insight(format!(
                        "Main clause structure changed ({} ROOT change(s))",
                        root_changes
                    ));
                }

                // Show detailed changes if not too many
                if dep_changes > 0 && dep_changes <= 10 {
                    let changes = self.find_dep_changes(&orig_tokens, &mod_tokens);
                    if !changes.is_empty() {
                        result.add_insight("Dependency changes:".to_string());
                        for change in changes {
                            result.add_insight(format!("  - {}", change));
                        }
                    }
                }

                // Show dependency distributions if significantly different
                if unchanged_ratio < 0.7 {
                    let orig_dist = self.get_dep_distribution(&orig_tokens);
                    let mod_dist = self.get_dep_distribution(&mod_tokens);

                    result.add_insight("Original dependency distribution:".to_string());
                    for (dep, count) in &orig_dist {
                        result.add_insight(format!("  - {}: {}", dep, count));
                    }

                    result.add_insight("Modified dependency distribution:".to_string());
                    for (dep, count) in &mod_dist {
                        result.add_insight(format!("  - {}: {}", dep, count));
                    }
                }

                result.add_metadata("status", "success");
            }
            (Err(e), _) | (_, Err(e)) => {
                result.add_insight(format!("Dependency analysis failed: {}", e));
                result.add_metadata("status", "error");
                result.add_metadata("error", &e);
                result = result.with_confidence(0.0);
            }
        }

        result
    }

    fn name(&self) -> &str {
        "spacy_dependency"
    }

    fn description(&self) -> &str {
        "Analyzes dependency relation changes using SpaCy"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Non-SpaCy fallback implementation
// ============================================================================

#[cfg(not(feature = "spacy"))]
pub struct SpacyDependencyAnalyzer;

#[cfg(not(feature = "spacy"))]
impl SpacyDependencyAnalyzer {
    pub fn new(_model_name: impl Into<String>) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl Clone for SpacyDependencyAnalyzer {
    fn clone(&self) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl SingleDiffAnalyzer for SpacyDependencyAnalyzer {
    fn analyze(&self, _diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        result.add_insight("SpaCy feature not enabled. Compile with --features spacy".to_string());
        result.add_metadata("status", "disabled");
        result.with_confidence(0.0)
    }

    fn name(&self) -> &str {
        "spacy_dependency"
    }

    fn description(&self) -> &str {
        "SpaCy-based dependency analyzer (requires 'spacy' feature)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
