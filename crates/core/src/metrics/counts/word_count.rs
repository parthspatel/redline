//! Word count metric: counts tokens in processed text.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts the number of tokens (words) in the processed text.
pub struct WordCountMetric;

impl Metric for WordCountMetric {
    fn id(&self) -> &str {
        "word_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => Ok(MetricValue::Integer(text.tokens.len() as i64)),
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "word_count".into(),
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

    fn make_text_with_tokens(s: &str, n_tokens: usize) -> ProcessedText {
        use crate::span::Span;
        use crate::text_store::StringId;
        use crate::token::{Token, TokenKind};

        let store = TextStoreBuilder::new().build();
        let tokens: Vec<Token> = (0..n_tokens)
            .map(|i| Token::new(StringId(i as u32), Span::new(0, 1), TokenKind::Regular))
            .collect();
        ProcessedText {
            original: s.to_string(),
            layers: vec![],
            normalized: s.to_string(),
            tokens,
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        }
    }

    #[test]
    fn empty_text() {
        let text = make_text("");
        let input = MetricInput::Single(&text);
        let result = WordCountMetric.compute(&input, &HashMap::new()).unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }

    #[test]
    fn counts_tokens() {
        let text = make_text_with_tokens("hello world", 2);
        let input = MetricInput::Single(&text);
        let result = WordCountMetric.compute(&input, &HashMap::new()).unwrap();
        assert_eq!(result, MetricValue::Integer(2));
    }

    #[test]
    fn rejects_pairwise() {
        let t1 = make_text("a");
        let t2 = make_text("b");
        let input = MetricInput::Pairwise(&t1, &t2);
        assert!(WordCountMetric.compute(&input, &HashMap::new()).is_err());
    }

    #[test]
    fn metadata() {
        assert_eq!(WordCountMetric.id(), "word_count");
        assert_eq!(WordCountMetric.static_dependencies(), &[] as &[&str]);
        assert_eq!(WordCountMetric.dependency_kind(), DependencyKind::Static);
        assert!(WordCountMetric.cost() < 0.1);
    }
}
