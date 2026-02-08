//! Whitespace count metric: counts whitespace characters in original text.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts characters where `char::is_whitespace()` is true.
pub struct WhitespaceCountMetric;

impl Metric for WhitespaceCountMetric {
    fn id(&self) -> &str {
        "whitespace_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                let count = text.original.chars().filter(|c| c.is_whitespace()).count();
                Ok(MetricValue::Integer(count as i64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "whitespace_count".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &[]
    }

    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }

    fn cost(&self) -> f32 {
        0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::ProcessedText;
    use crate::text_store::TextStoreBuilder;

    fn make_text(s: &str) -> ProcessedText {
        let store = TextStoreBuilder::new().build();
        ProcessedText {
            original: s.to_string(),
            layers: vec![],
            normalized: s.to_string(),
            tokens: vec![],
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        }
    }

    #[test]
    fn single_space() {
        let text = make_text("Hello World");
        let input = MetricInput::Single(&text);
        assert_eq!(
            WhitespaceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(1)
        );
    }

    #[test]
    fn tabs_and_newlines() {
        let text = make_text("A\tB\nC");
        let input = MetricInput::Single(&text);
        assert_eq!(
            WhitespaceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(2)
        );
    }

    #[test]
    fn empty_text() {
        let text = make_text("");
        let input = MetricInput::Single(&text);
        assert_eq!(
            WhitespaceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(0)
        );
    }

    #[test]
    fn no_whitespace() {
        let text = make_text("abc123");
        let input = MetricInput::Single(&text);
        assert_eq!(
            WhitespaceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(0)
        );
    }
}
