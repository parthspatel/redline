//! Flesch-Kincaid Grade Level metric.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Flesch-Kincaid Grade Level score.
/// Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
pub struct FleschKincaidGradeMetric;

impl Metric for FleschKincaidGradeMetric {
    fn id(&self) -> &str {
        "flesch_kincaid_grade"
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

                let grade = 0.39 * (w / s) + 11.8 * (syl / w) - 15.59;
                Ok(MetricValue::Float(grade))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "flesch_kincaid_grade".into(),
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
        // 0.39*(10/1) + 11.8*(14/10) - 15.59
        // = 3.9 + 16.52 - 15.59 = 4.83
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(10, 1, 14);
        let result = FleschKincaidGradeMetric.compute(&input, &deps).unwrap();
        let grade = result.as_float().unwrap();
        assert!((grade - 4.83).abs() < 0.01, "got {grade}");
    }

    #[test]
    fn zero_words() {
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(0, 1, 0);
        let result = FleschKincaidGradeMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn zero_sentences() {
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(10, 0, 14);
        let result = FleschKincaidGradeMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn metadata() {
        assert_eq!(FleschKincaidGradeMetric.id(), "flesch_kincaid_grade");
        assert_eq!(
            FleschKincaidGradeMetric.static_dependencies(),
            &["word_count", "sentence_count", "syllable_count"]
        );
    }
}
