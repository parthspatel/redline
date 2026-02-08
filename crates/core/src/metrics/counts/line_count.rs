//! Line count metric: counts lines in original text.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts lines by counting `\n` characters + 1.
pub struct LineCountMetric;

impl Metric for LineCountMetric {
    fn id(&self) -> &str {
        "line_count"
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
                let newlines = text.original.chars().filter(|&c| c == '\n').count();
                Ok(MetricValue::Integer(newlines as i64 + 1))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "line_count".into(),
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
    fn single_line() {
        let text = make_text("Single line");
        let input = MetricInput::Single(&text);
        assert_eq!(
            LineCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(1)
        );
    }

    #[test]
    fn three_lines() {
        let text = make_text("Hello\nWorld\nFoo");
        let input = MetricInput::Single(&text);
        assert_eq!(
            LineCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(3)
        );
    }

    #[test]
    fn empty_text() {
        let text = make_text("");
        let input = MetricInput::Single(&text);
        assert_eq!(
            LineCountMetric.compute(&input, &HashMap::new()).unwrap(),
            MetricValue::Integer(1)
        );
    }

    #[test]
    fn rejects_pairwise() {
        let t1 = make_text("a");
        let t2 = make_text("b");
        let input = MetricInput::Pairwise(&t1, &t2);
        assert!(LineCountMetric.compute(&input, &HashMap::new()).is_err());
    }
}
