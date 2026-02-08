//! Built-in analyzers: semantic, stylistic, readability, edit_classifier.

pub mod edit_classifier;
pub mod readability;
pub mod semantic;
pub mod stylistic;

// Re-export key types
pub use edit_classifier::{
    EditClassifier, EditClassifierConfig, EditClassifierResult, GroupClassification,
    IntentCategory, OperationClassification,
};
pub use readability::{
    ReadabilityAnalyzer, ReadabilityAttribution, ReadabilityResult, ReadabilityScores,
};
pub use semantic::{
    HeuristicScoring, OperationSimilarity, ScoringBackend, SemanticAnalyzer, SemanticResult,
};
pub use stylistic::{
    StructuralChanges, StylisticAnalyzer, StylisticResult, ToneDimension, ToneShift, Voice,
    VoiceShift,
};

use super::AnalysisError;
use super::registry::PluginRegistry;

/// Register all 4 built-in analyzers with the given registry.
///
/// Registers: `semantic`, `stylistic`, `readability`, `edit_classifier`.
/// None have inter-analyzer dependencies, so registration order doesn't matter.
pub fn register_builtin_analyzers(registry: &mut PluginRegistry) -> Result<(), AnalysisError> {
    registry.register(Box::new(SemanticAnalyzer::new()))?;
    registry.register(Box::new(StylisticAnalyzer))?;
    registry.register(Box::new(ReadabilityAnalyzer))?;
    registry.register(Box::new(EditClassifier::new()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_builtin_analyzers_success() {
        let mut reg = PluginRegistry::new();
        register_builtin_analyzers(&mut reg).unwrap();
        assert_eq!(reg.len(), 4);
        assert!(reg.contains("semantic"));
        assert!(reg.contains("stylistic"));
        assert!(reg.contains("readability"));
        assert!(reg.contains("edit_classifier"));
    }

    #[test]
    fn register_builtin_analyzers_duplicate_error() {
        let mut reg = PluginRegistry::new();
        register_builtin_analyzers(&mut reg).unwrap();
        let err = register_builtin_analyzers(&mut reg).unwrap_err();
        assert!(matches!(err, AnalysisError::AlreadyRegistered(_)));
    }
}
