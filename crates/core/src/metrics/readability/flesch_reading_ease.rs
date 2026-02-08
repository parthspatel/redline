//! Flesch Reading Ease metric.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Flesch Reading Ease score.
/// Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
pub struct FleschReadingEaseMetric;

impl Metric for FleschReadingEaseMetric {
    fn id(&self) -> &str {
        "flesch_reading_ease"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(_) => {
                let words = deps
                    .get("word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("word_count".into()))?;
                let sentences = deps
                    .get("sentence_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("sentence_count".into()))?;
                let syllables = deps
                    .get("syllable_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("syllable_count".into()))?;

                if words == 0 || sentences == 0 {
                    return Ok(MetricValue::Float(0.0));
                }

                let w = words as f64;
                let s = sentences as f64;
                let syl = syllables as f64;

                let score = 206.835 - 1.015 * (w / s) - 84.6 * (syl / w);
                Ok(MetricValue::Float(score))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "flesch_reading_ease".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["word_count", "sentence_count", "syllable_count"]
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

    fn make_deps(words: i64, sentences: i64, syllables: i64) -> HashMap<String, MetricValue> {
        let mut deps = HashMap::new();
        deps.insert("word_count".into(), MetricValue::Integer(words));
        deps.insert("sentence_count".into(), MetricValue::Integer(sentences));
        deps.insert("syllable_count".into(), MetricValue::Integer(syllables));
        deps
    }

    fn dummy_text() -> crate::process::ProcessedText {
        let store = crate::text_store::TextStoreBuilder::new().build();
        crate::process::ProcessedText {
            original: "".into(),
            layers: vec![],
            normalized: "".into(),
            tokens: vec![],
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        }
    }

    #[test]
    fn known_value() {
        // 10 words, 1 sentence, 14 syllables
        // 206.835 - 1.015*(10/1) - 84.6*(14/10)
        // = 206.835 - 10.15 - 118.44 = 78.245
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(10, 1, 14);
        let result = FleschReadingEaseMetric.compute(&input, &deps).unwrap();
        let score = result.as_float().unwrap();
        assert!((score - 78.245).abs() < 0.01, "got {score}");
    }

    #[test]
    fn zero_words() {
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(0, 0, 0);
        let result = FleschReadingEaseMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn zero_sentences() {
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(10, 0, 14);
        let result = FleschReadingEaseMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn metadata() {
        assert_eq!(FleschReadingEaseMetric.id(), "flesch_reading_ease");
        assert_eq!(
            FleschReadingEaseMetric.static_dependencies(),
            &["word_count", "sentence_count", "syllable_count"]
        );
    }
}
