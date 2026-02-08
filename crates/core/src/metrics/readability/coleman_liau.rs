//! Coleman-Liau Index metric.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Coleman-Liau Index.
/// Formula: 0.0588 * L - 0.296 * S - 15.8
/// where L = letters per 100 words, S = sentences per 100 words.
pub struct ColemanLiauMetric;

impl Metric for ColemanLiauMetric {
    fn id(&self) -> &str {
        "coleman_liau"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(_) => {
                let letters = deps
                    .get("letter_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("letter_count".into()))?;
                let words = deps
                    .get("word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("word_count".into()))?;
                let sentences = deps
                    .get("sentence_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("sentence_count".into()))?;

                if words == 0 {
                    return Ok(MetricValue::Float(0.0));
                }

                let w = words as f64;
                let l = letters as f64 * 100.0 / w; // letters per 100 words
                let s = sentences as f64 * 100.0 / w; // sentences per 100 words

                let index = 0.0588 * l - 0.296 * s - 15.8;
                Ok(MetricValue::Float(index))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "coleman_liau".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["letter_count", "word_count", "sentence_count"]
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

    fn make_deps(letters: i64, words: i64, sentences: i64) -> HashMap<String, MetricValue> {
        let mut deps = HashMap::new();
        deps.insert("letter_count".into(), MetricValue::Integer(letters));
        deps.insert("word_count".into(), MetricValue::Integer(words));
        deps.insert("sentence_count".into(), MetricValue::Integer(sentences));
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
        // 50 letters, 10 words, 1 sentence
        // L = 50*100/10 = 500, S = 1*100/10 = 10
        // 0.0588*500 - 0.296*10 - 15.8 = 29.4 - 2.96 - 15.8 = 10.64
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(50, 10, 1);
        let result = ColemanLiauMetric.compute(&input, &deps).unwrap();
        let index = result.as_float().unwrap();
        assert!((index - 10.64).abs() < 0.01, "got {index}");
    }

    #[test]
    fn zero_words() {
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(0, 0, 0);
        let result = ColemanLiauMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn metadata() {
        assert_eq!(ColemanLiauMetric.id(), "coleman_liau");
        assert_eq!(
            ColemanLiauMetric.static_dependencies(),
            &["letter_count", "word_count", "sentence_count"]
        );
    }
}
