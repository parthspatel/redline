//! Grammar fix detection using SpaCy
//!
//! This analyzer detects potential grammar corrections based on syntactic patterns.

#[cfg(feature = "spacy")]
use crate::nlp::SpacyClient;

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

#[cfg(feature = "spacy")]
use crate::analyzer::classifiers::SyntacticToken;
#[cfg(feature = "spacy")]
use crate::token_alignment;

#[cfg(feature = "spacy")]
/// SpaCy-based grammar fix detector
///
/// Detects common grammar fixes based on POS tags and dependency relations.
///
/// ## Grammar Patterns Detected
/// - Subject-verb agreement fixes
/// - Verb tense corrections
/// - Article corrections (a/an)
/// - Pronoun corrections
///
/// ## Metrics Provided
/// - **grammar_fixes**: Total number of grammar fixes detected
/// - **verb_fixes**: Number of verb-related fixes
/// - **article_fixes**: Number of article corrections
/// - **tense_fixes**: Number of tense corrections
pub struct SpacyGrammarAnalyzer {
    spacy_client: SpacyClient,
}

#[cfg(feature = "spacy")]
impl SpacyGrammarAnalyzer {
    /// Create a new grammar analyzer with the specified SpaCy model
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            spacy_client: SpacyClient::new(model_name),
        }
    }

    /// Analyze grammar in text
    pub fn analyze_grammar(&self, text: &str) -> Result<Vec<SyntacticToken>, String> {
        self.spacy_client.analyze(text)
    }

    /// Detect potential grammar fixes between original and modified text
    ///
    /// Uses proper token alignment to match tokens by content, not position
    pub fn detect_grammar_fixes(
        &self,
        orig: &[SyntacticToken],
        modified: &[SyntacticToken],
    ) -> Vec<String> {
        let alignments = token_alignment::align_tokens(orig, modified);
        token_alignment::find_grammar_fixes_aligned(orig, modified, &alignments)
    }

    /// Categorize grammar fixes by type
    pub fn categorize_fixes(&self, fixes: &[String]) -> (usize, usize, usize) {
        let mut verb_fixes = 0;
        let mut article_fixes = 0;
        let mut tense_fixes = 0;

        for fix in fixes {
            if fix.contains("Verb form") || fix.contains("Auxiliary verb") {
                verb_fixes += 1;
            } else if fix.contains("Article") {
                article_fixes += 1;
            } else if fix.contains("Tense") {
                tense_fixes += 1;
            }
        }

        (verb_fixes, article_fixes, tense_fixes)
    }
}

#[cfg(feature = "spacy")]
impl Clone for SpacyGrammarAnalyzer {
    fn clone(&self) -> Self {
        Self {
            spacy_client: self.spacy_client.clone(),
        }
    }
}

#[cfg(feature = "spacy")]
impl SingleDiffAnalyzer for SpacyGrammarAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Analyze grammar
        match (
            self.spacy_client.analyze(&diff.original_text),
            self.spacy_client.analyze(&diff.modified_text),
        ) {
            (Ok(orig_tokens), Ok(mod_tokens)) => {
                let grammar_fixes = self.detect_grammar_fixes(&orig_tokens, &mod_tokens);
                let (verb_fixes, article_fixes, tense_fixes) =
                    self.categorize_fixes(&grammar_fixes);

                result.add_metric("grammar_fixes", grammar_fixes.len() as f64);
                result.add_metric("verb_fixes", verb_fixes as f64);
                result.add_metric("article_fixes", article_fixes as f64);
                result.add_metric("tense_fixes", tense_fixes as f64);

                // Generate insights
                if grammar_fixes.is_empty() {
                    result.add_insight("No grammar fixes detected");
                } else {
                    result.add_insight(format!(
                        "Detected {} potential grammar fix(es):",
                        grammar_fixes.len()
                    ));
                    for fix in &grammar_fixes {
                        result.add_insight(format!("  - {}", fix));
                    }

                    // Summary by category
                    if verb_fixes > 0 || article_fixes > 0 || tense_fixes > 0 {
                        result.add_insight(format!(
                            "Fix categories: {} verb, {} article, {} tense",
                            verb_fixes, article_fixes, tense_fixes
                        ));
                    }
                }

                result.add_metadata("status", "success");
            }
            (Err(e), _) | (_, Err(e)) => {
                result.add_insight(format!("Grammar analysis failed: {}", e));
                result.add_metadata("status", "error");
                result.add_metadata("error", &e);
                result = result.with_confidence(0.0);
            }
        }

        result
    }

    fn name(&self) -> &str {
        "spacy_grammar"
    }

    fn description(&self) -> &str {
        "Detects grammar fixes using SpaCy's syntactic analysis"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Non-SpaCy fallback implementation
// ============================================================================

#[cfg(not(feature = "spacy"))]
pub struct SpacyGrammarAnalyzer;

#[cfg(not(feature = "spacy"))]
impl SpacyGrammarAnalyzer {
    pub fn new(_model_name: impl Into<String>) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl Clone for SpacyGrammarAnalyzer {
    fn clone(&self) -> Self {
        Self
    }
}

#[cfg(not(feature = "spacy"))]
impl SingleDiffAnalyzer for SpacyGrammarAnalyzer {
    fn analyze(&self, _diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        result.add_insight("SpaCy feature not enabled. Compile with --features spacy".to_string());
        result.add_metadata("status", "disabled");
        result.with_confidence(0.0)
    }

    fn name(&self) -> &str {
        "spacy_grammar"
    }

    fn description(&self) -> &str {
        "SpaCy-based grammar analyzer (requires 'spacy' feature)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
