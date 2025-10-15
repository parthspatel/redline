// SpaCy-based analyzers (modular)
#[cfg(feature = "spacy")]
pub mod alignment;
#[cfg(feature = "spacy")]
pub mod dependency_parse;
#[cfg(feature = "spacy")]
pub mod grammar;
#[cfg(feature = "spacy")]
pub mod part_of_speech;
#[cfg(feature = "spacy")]
pub mod structural;
pub mod syntactic;

// Re-export modular SpaCy analyzers
#[cfg(feature = "spacy")]
pub use dependency_parse::SpacyDependencyAnalyzer;
#[cfg(feature = "spacy")]
pub use grammar::SpacyGrammarAnalyzer;
#[cfg(feature = "spacy")]
pub use part_of_speech::SpacyPOSAnalyzer;
#[cfg(feature = "spacy")]
pub use structural::SpacyStructuralAnalyzer;

pub use crate::util::SyntacticToken;
// Re-export SpaCy types when feature is enabled
#[cfg(feature = "spacy")]
pub use syntactic::{SpacySyntacticAnalyzer, SyntacticComparison};

// ============================================================================
// Utility macros and functions for SpaCy analyzers
// ============================================================================

#[cfg(feature = "spacy")]
use crate::util::SpacyClient;
#[cfg(feature = "spacy")]
use crate::analyzer::AnalysisResult;
#[cfg(feature = "spacy")]
use crate::diff::DiffResult;

/// Helper function to handle SpaCy analysis errors consistently
#[cfg(feature = "spacy")]
pub fn handle_spacy_analysis_error(
    mut result: AnalysisResult,
    analyzer_name: &str,
    error: String,
) -> AnalysisResult {
    result.add_insight(format!("{} analysis failed: {}", analyzer_name, error));
    result.add_metadata("status", "error");
    result.add_metadata("error", &error);
    result.with_confidence(0.0)
}

/// Helper function to analyze both original and modified texts
#[cfg(feature = "spacy")]
pub fn analyze_both_texts(
    client: &SpacyClient,
    diff: &DiffResult,
) -> Result<(Vec<SyntacticToken>, Vec<SyntacticToken>), String> {
    match (
        client.analyze(&diff.original_text),
        client.analyze(&diff.modified_text),
    ) {
        (Ok(orig), Ok(mod_tokens)) => Ok((orig, mod_tokens)),
        (Err(e), _) | (_, Err(e)) => Err(e),
    }
}

/// Macro to implement the standard SpaCy fallback for analyzers when spacy feature is disabled
macro_rules! impl_spacy_fallback {
    ($analyzer:ident, $name:expr, $description:expr) => {
        #[cfg(not(feature = "spacy"))]
        pub struct $analyzer;

        #[cfg(not(feature = "spacy"))]
        impl $analyzer {
            pub fn new(_model_name: impl Into<String>) -> Self {
                Self
            }
        }

        #[cfg(not(feature = "spacy"))]
        impl Clone for $analyzer {
            fn clone(&self) -> Self {
                Self
            }
        }

        #[cfg(not(feature = "spacy"))]
        impl crate::analyzer::SingleDiffAnalyzer for $analyzer {
            fn analyze(&self, _diff: &crate::diff::DiffResult) -> crate::analyzer::AnalysisResult {
                let mut result = crate::analyzer::AnalysisResult::new(self.name());
                result.add_insight(
                    "SpaCy feature not enabled. Compile with --features spacy".to_string(),
                );
                result.add_metadata("status", "disabled");
                result.with_confidence(0.0)
            }

            fn name(&self) -> &str {
                $name
            }

            fn description(&self) -> &str {
                $description
            }

            fn clone_box(&self) -> Box<dyn crate::analyzer::SingleDiffAnalyzer> {
                Box::new(self.clone())
            }
        }
    };
}

pub(crate) use impl_spacy_fallback;
