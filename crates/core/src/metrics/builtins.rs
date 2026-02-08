//! Built-in metric registration.

use super::error::MetricError;
use super::registry::MetricRegistry;

use super::counts::{
    ByteCountMetric, CharCountMetric, DigitCountMetric, LetterCountMetric, LineCountMetric,
    ParagraphCountMetric, PunctuationCountMetric, SentenceCountMetric, SyllableCountMetric,
    UniqueWordCountMetric, WhitespaceCountMetric, WordCountMetric,
};

use super::readability::{
    AriMetric, AvgSentenceLengthMetric, AvgWordLengthMetric, ColemanLiauMetric,
    FleschKincaidGradeMetric, FleschReadingEaseMetric, GunningFogMetric, LexicalDensityMetric,
    SmogIndexMetric, VocabularyRichnessMetric,
};

use super::similarity::{
    CharCountDiffMetric, CosineSimilarityMetric, DamerauLevenshteinDistanceMetric,
    DiceCoefficientMetric, GradeLevelDeltaMetric, HammingDistanceMetric, JaccardSimilarityMetric,
    JaroSimilarityMetric, JaroWinklerSimilarityMetric, LengthRatioMetric,
    LevenshteinDistanceMetric, OverlapCoefficientMetric, ReadabilityDeltaMetric,
    SimilarityRatioMetric, WordCountDiffMetric,
};

/// Register all built-in metrics with the given registry.
///
/// Registration order: counts (no deps) -> readability (depend on counts) ->
/// pairwise (jaro_winkler depends on jaro_similarity).
///
/// Validates the dependency graph after registration.
pub fn register_builtins(registry: &mut MetricRegistry) -> Result<(), MetricError> {
    // Count metrics (12) — no dependencies
    registry.register(Box::new(WordCountMetric))?;
    registry.register(Box::new(CharCountMetric))?;
    registry.register(Box::new(ByteCountMetric))?;
    registry.register(Box::new(SentenceCountMetric))?;
    registry.register(Box::new(SyllableCountMetric))?;
    registry.register(Box::new(UniqueWordCountMetric))?;
    registry.register(Box::new(ParagraphCountMetric))?;
    registry.register(Box::new(LineCountMetric))?;
    registry.register(Box::new(LetterCountMetric))?;
    registry.register(Box::new(DigitCountMetric))?;
    registry.register(Box::new(WhitespaceCountMetric))?;
    registry.register(Box::new(PunctuationCountMetric))?;

    // Readability metrics (10) — depend on count metrics
    registry.register(Box::new(AvgWordLengthMetric))?;
    registry.register(Box::new(AvgSentenceLengthMetric))?;
    registry.register(Box::new(VocabularyRichnessMetric))?;
    registry.register(Box::new(LexicalDensityMetric))?;
    registry.register(Box::new(FleschReadingEaseMetric))?;
    registry.register(Box::new(FleschKincaidGradeMetric))?;
    registry.register(Box::new(GunningFogMetric))?;
    registry.register(Box::new(SmogIndexMetric))?;
    registry.register(Box::new(ColemanLiauMetric))?;
    registry.register(Box::new(AriMetric))?;

    // Pairwise similarity metrics (15) — jaro before jaro_winkler
    registry.register(Box::new(JaccardSimilarityMetric))?;
    registry.register(Box::new(CosineSimilarityMetric))?;
    registry.register(Box::new(DiceCoefficientMetric))?;
    registry.register(Box::new(OverlapCoefficientMetric))?;
    registry.register(Box::new(LengthRatioMetric))?;
    registry.register(Box::new(WordCountDiffMetric))?;
    registry.register(Box::new(CharCountDiffMetric))?;
    registry.register(Box::new(LevenshteinDistanceMetric))?;
    registry.register(Box::new(DamerauLevenshteinDistanceMetric))?;
    registry.register(Box::new(HammingDistanceMetric))?;
    registry.register(Box::new(JaroSimilarityMetric))?;
    registry.register(Box::new(JaroWinklerSimilarityMetric))?;
    registry.register(Box::new(ReadabilityDeltaMetric))?;
    registry.register(Box::new(GradeLevelDeltaMetric))?;
    registry.register(Box::new(SimilarityRatioMetric))?;

    // Validate dependency graph
    registry.validate()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_all_builtins() {
        let mut reg = MetricRegistry::new();
        register_builtins(&mut reg).unwrap();
        assert_eq!(reg.len(), 37); // 12 + 10 + 15
    }

    #[test]
    fn validates_successfully() {
        let mut reg = MetricRegistry::new();
        let result = register_builtins(&mut reg);
        assert!(result.is_ok());
    }
}
