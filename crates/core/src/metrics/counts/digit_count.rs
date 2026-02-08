//! Digit count metric: counts ASCII digits in original text.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts characters where `char::is_ascii_digit()` is true.
pub struct DigitCountMetric;

impl Metric for DigitCountMetric {
    fn id(&self) -> &str {
        "digit_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                let count = text.original.chars().filter(|c| c.is_ascii_digit()).count();
                Ok(MetricValue::Integer(count as i64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "digit_count".into(),
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
    fn mixed_text() {
        let text = make_text("Hello 123!");
        let input = MetricInput::Single(&text);
        assert_eq!(
            DigitCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(3)
        );
    }

    #[test]
    fn no_digits() {
        let text = make_text("Hello world!");
        let input = MetricInput::Single(&text);
        assert_eq!(
            DigitCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(0)
        );
    }

    #[test]
    fn empty_text() {
        let text = make_text("");
        let input = MetricInput::Single(&text);
        assert_eq!(
            DigitCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(0)
        );
    }
}
