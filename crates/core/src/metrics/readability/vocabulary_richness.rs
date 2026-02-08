//! Vocabulary richness metric: unique_word_count / word_count (Type-Token Ratio).

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Type-Token Ratio (TTR): unique words divided by total words.
pub struct VocabularyRichnessMetric;

impl Metric for VocabularyRichnessMetric {
    fn id(&self) -> &str {
        "vocabulary_richness"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(_) => {
                let unique = deps
                    .get("unique_word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| {
                        MetricError::DependencyUnavailable("unique_word_count".into())
                    })?;
                let words = deps
                    .get("word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("word_count".into()))?;
                if words == 0 {
                    return Ok(MetricValue::Float(0.0));
                }
                Ok(MetricValue::Float(unique as f64 / words as f64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "vocabulary_richness".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["unique_word_count", "word_count"]
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

    #[test]
    fn all_unique() {
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
        let mut deps = HashMap::new();
        deps.insert("unique_word_count".into(), MetricValue::Integer(5));
        deps.insert("word_count".into(), MetricValue::Integer(5));
        let result = VocabularyRichnessMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn half_unique() {
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
        let mut deps = HashMap::new();
        deps.insert("unique_word_count".into(), MetricValue::Integer(3));
        deps.insert("word_count".into(), MetricValue::Integer(6));
        let result = VocabularyRichnessMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.5));
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
        let mut deps = HashMap::new();
        deps.insert("unique_word_count".into(), MetricValue::Integer(0));
        deps.insert("word_count".into(), MetricValue::Integer(0));
        let result = VocabularyRichnessMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }
}
