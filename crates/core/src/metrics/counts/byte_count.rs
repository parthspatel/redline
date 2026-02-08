//! Byte count metric: counts UTF-8 bytes in original text.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts the number of UTF-8 bytes in the original text.
pub struct ByteCountMetric;

impl Metric for ByteCountMetric {
    fn id(&self) -> &str {
        "byte_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => Ok(MetricValue::Integer(text.original.len() as i64)),
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "byte_count".into(),
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
    fn empty_text() {
        let text = make_text("");
        let input = MetricInput::Single(&text);
        assert_eq!(
            ByteCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(0)
        );
    }

    #[test]
    fn ascii_text() {
        let text = make_text("hello");
        let input = MetricInput::Single(&text);
        assert_eq!(
            ByteCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(5)
        );
    }

    #[test]
    fn unicode_multibyte() {
        // cafe + combining accent: 'c'=1, 'a'=1, 'f'=1, 'e'=1, '\u{0301}'=2 = 6 bytes
        let text = make_text("cafe\u{0301}");
        let input = MetricInput::Single(&text);
        assert_eq!(
            ByteCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(6)
        );
    }

    #[test]
    fn rejects_pairwise() {
        let t1 = make_text("a");
        let t2 = make_text("b");
        let input = MetricInput::Pairwise(&t1, &t2);
        assert!(ByteCountMetric.compute(&input, &HashMap::new()).is_err());
    }
}
