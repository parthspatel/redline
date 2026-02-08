//! Average sentence length metric: word_count / sentence_count.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Average number of words per sentence.
pub struct AvgSentenceLengthMetric;

impl Metric for AvgSentenceLengthMetric {
    fn id(&self) -> &str {
        "avg_sentence_length"
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
                if sentences == 0 {
                    return Ok(MetricValue::Float(0.0));
                }
                Ok(MetricValue::Float(words as f64 / sentences as f64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "avg_sentence_length".into(),
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

    fn make_deps(word_count: i64, sentence_count: i64) -> HashMap<String, MetricValue> {
        let mut deps = HashMap::new();
        deps.insert("word_count".into(), MetricValue::Integer(word_count));
        deps.insert(
            "sentence_count".into(),
            MetricValue::Integer(sentence_count),
        );
        deps
    }

    #[test]
    fn normal_case() {
        let store = crate::text_store::TextStoreBuilder::new().build();
        let text = crate::process::ProcessedText {
            original: "".into(),
            layers: vec![],
            normalized: "".into(),
            tokens: vec![],
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        };
        let input = MetricInput::Single(&text);
        let deps = make_deps(20, 2);
        let result = AvgSentenceLengthMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(10.0));
    }

    #[test]
    fn zero_sentences() {
        let store = crate::text_store::TextStoreBuilder::new().build();
        let text = crate::process::ProcessedText {
            original: "".into(),
            layers: vec![],
            normalized: "".into(),
            tokens: vec![],
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        };
        let input = MetricInput::Single(&text);
        let deps = make_deps(10, 0);
        let result = AvgSentenceLengthMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }
}
