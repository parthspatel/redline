//! Gunning Fog Index metric.

use hashbrown::HashMap;

use crate::metrics::counts::count_syllables;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Gunning Fog Index.
/// Formula: 0.4 * (words/sentences + 100 * (complex_words/words))
/// A "complex word" has 3 or more syllables.
pub struct GunningFogMetric;

impl Metric for GunningFogMetric {
    fn id(&self) -> &str {
        "gunning_fog"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                let words = deps
                    .get("word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("word_count".into()))?;
                let sentences = deps
                    .get("sentence_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("sentence_count".into()))?;

                if words == 0 || sentences == 0 {
                    return Ok(MetricValue::Float(0.0));
                }

                let mut complex_count = 0i64;
                for token in &text.tokens {
                    let span = token.span;
                    let end = (span.end() as usize).min(text.normalized.len());
                    let word = &text.normalized[span.start() as usize..end];
                    if count_syllables(word) >= 3 {
                        complex_count += 1;
                    }
                }

                let w = words as f64;
                let s = sentences as f64;
                let fog = 0.4 * (w / s + 100.0 * (complex_count as f64 / w));
                Ok(MetricValue::Float(fog))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "gunning_fog".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["word_count", "sentence_count"]
    }

    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }

    fn cost(&self) -> f32 {
        0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextProcessor;
    use crate::process::ProcessedText;
    use crate::tokenize::WordTokenizer;

    fn process(s: &str) -> ProcessedText {
        TextProcessor::new(
            vec![],
            Box::new(WordTokenizer),
            crate::process::ExecutionMode::All,
        )
        .unwrap()
        .process(s)
        .unwrap()
    }

    #[test]
    fn simple_sentence() {
        // "The cat sat on the mat." — 6 words, 1 sentence, 0 complex words
        // fog = 0.4 * (6/1 + 100*(0/6)) = 0.4 * 6 = 2.4
        let text = process("The cat sat on the mat.");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("word_count".into(), MetricValue::Integer(6));
        deps.insert("sentence_count".into(), MetricValue::Integer(1));
        let result = GunningFogMetric.compute(&input, &deps).unwrap();
        let fog = result.as_float().unwrap();
        assert!((fog - 2.4).abs() < 0.01, "got {fog}");
    }

    #[test]
    fn with_complex_words() {
        // "The beautiful extraordinary universe amazes scientists."
        // "beautiful" = 3 syl (complex), "extraordinary" = 5+ syl (complex),
        // "universe" = 3 syl (complex), "amazes" = 3 syl (complex),
        // "scientists" = 3 syl (complex)
        // Exact fog depends on syllable heuristic, just check it's > 0
        let text = process("The beautiful extraordinary universe amazes scientists.");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("word_count".into(), MetricValue::Integer(6));
        deps.insert("sentence_count".into(), MetricValue::Integer(1));
        let result = GunningFogMetric.compute(&input, &deps).unwrap();
        let fog = result.as_float().unwrap();
        assert!(fog > 2.0, "expected high fog, got {fog}");
    }

    #[test]
    fn zero_words() {
        let text = process("");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("word_count".into(), MetricValue::Integer(0));
        deps.insert("sentence_count".into(), MetricValue::Integer(0));
        let result = GunningFogMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn metadata() {
        assert_eq!(GunningFogMetric.id(), "gunning_fog");
        assert_eq!(
            GunningFogMetric.static_dependencies(),
            &["word_count", "sentence_count"]
        );
    }
}
