//! Paragraph count metric: counts paragraphs separated by double newlines.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts paragraphs by detecting `\n\n` sequences, returning count + 1.
pub struct ParagraphCountMetric;

impl Metric for ParagraphCountMetric {
    fn id(&self) -> &str {
        "paragraph_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                if text.original.is_empty() {
                    return Ok(MetricValue::Integer(1));
                }
                let breaks = text.original.matches("\n\n").count();
                Ok(MetricValue::Integer(breaks as i64 + 1))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "paragraph_count".into(),
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
            ParagraphCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(1)
        );
    }

    #[test]
    fn single_paragraph() {
        let text = make_text("No paragraphs here");
        let input = MetricInput::Single(&text);
        assert_eq!(
            ParagraphCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(1)
        );
    }

    #[test]
    fn two_paragraphs() {
        let text = make_text("Hello\n\nWorld");
        let input = MetricInput::Single(&text);
        assert_eq!(
            ParagraphCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(2)
        );
    }

    #[test]
    fn three_paragraphs() {
        let text = make_text("A\n\nB\n\nC");
        let input = MetricInput::Single(&text);
        assert_eq!(
            ParagraphCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(3)
        );
    }

    #[test]
    fn rejects_pairwise() {
        let t1 = make_text("a");
        let t2 = make_text("b");
        let input = MetricInput::Pairwise(&t1, &t2);
        assert!(
            ParagraphCountMetric
                .compute(&input, &HashMap::new())
                .is_err()
        );
    }
}
