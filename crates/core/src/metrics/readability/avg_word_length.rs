//! Average word length metric: char_count / word_count.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Average number of characters per word.
pub struct AvgWordLengthMetric;

impl Metric for AvgWordLengthMetric {
    fn id(&self) -> &str {
        "avg_word_length"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(_) => {
                let chars = deps
                    .get("char_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("char_count".into()))?;
                let words = deps
                    .get("word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("word_count".into()))?;
                if words == 0 {
                    return Ok(MetricValue::Float(0.0));
                }
                Ok(MetricValue::Float(chars as f64 / words as f64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "avg_word_length".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["char_count", "word_count"]
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

    fn make_deps(char_count: i64, word_count: i64) -> HashMap<String, MetricValue> {
        let mut deps = HashMap::new();
        deps.insert("char_count".into(), MetricValue::Integer(char_count));
        deps.insert("word_count".into(), MetricValue::Integer(word_count));
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
        let deps = make_deps(20, 4);
        let result = AvgWordLengthMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(5.0));
    }

    #[test]
    fn zero_words() {
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
        let deps = make_deps(0, 0);
        let result = AvgWordLengthMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn metadata() {
        assert_eq!(AvgWordLengthMetric.id(), "avg_word_length");
        assert_eq!(
            AvgWordLengthMetric.static_dependencies(),
            &["char_count", "word_count"]
        );
    }
}
