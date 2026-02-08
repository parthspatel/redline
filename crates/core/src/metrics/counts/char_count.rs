//! Character count metric: counts Unicode characters in original text.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts the number of Unicode characters in the original text.
pub struct CharCountMetric;

impl Metric for CharCountMetric {
    fn id(&self) -> &str {
        "char_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                Ok(MetricValue::Integer(text.original.chars().count() as i64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "char_count".into(),
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
            CharCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(0)
        );
    }

    #[test]
    fn ascii_text() {
        let text = make_text("hello");
        let input = MetricInput::Single(&text);
        assert_eq!(
            CharCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(5)
        );
    }

    #[test]
    fn unicode_combining() {
        // cafe + combining accent = 5 chars (not 4 graphemes)
        let text = make_text("cafe\u{0301}");
        let input = MetricInput::Single(&text);
        assert_eq!(
            CharCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(5)
        );
    }

    #[test]
    fn emoji() {
        let text = make_text("Hi 👋🏽");
        let input = MetricInput::Single(&text);
        // "Hi " = 3 chars, 👋🏽 = 2 chars (wave + skin tone modifier)
        assert_eq!(
            CharCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(5)
        );
    }

    #[test]
    fn rejects_pairwise() {
        let t1 = make_text("a");
        let t2 = make_text("b");
        let input = MetricInput::Pairwise(&t1, &t2);
        assert!(CharCountMetric.compute(&input, &HashMap::new()).is_err());
    }
}
