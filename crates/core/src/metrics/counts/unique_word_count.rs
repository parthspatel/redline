//! Unique word count metric: counts distinct StringId values in tokens.

use hashbrown::{HashMap, HashSet};

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts the number of distinct words (by StringId) in the token list.
pub struct UniqueWordCountMetric;

impl Metric for UniqueWordCountMetric {
    fn id(&self) -> &str {
        "unique_word_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                let unique: HashSet<u32> = text.tokens.iter().map(|t| t.text_id.0).collect();
                Ok(MetricValue::Integer(unique.len() as i64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "unique_word_count".into(),
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
    use crate::span::Span;
    use crate::text_store::{StringId, TextStoreBuilder};
    use crate::token::{Token, TokenKind};

    fn make_text_with_ids(ids: &[u32]) -> ProcessedText {
        let store = TextStoreBuilder::new().build();
        let tokens: Vec<Token> = ids
            .iter()
            .map(|&id| Token::new(StringId(id), Span::new(0, 1), TokenKind::Regular))
            .collect();
        ProcessedText {
            original: "test".into(),
            layers: vec![],
            normalized: "test".into(),
            tokens,
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        }
    }

    #[test]
    fn empty_tokens() {
        let text = make_text_with_ids(&[]);
        let input = MetricInput::Single(&text);
        assert_eq!(
            UniqueWordCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(0)
        );
    }

    #[test]
    fn all_unique() {
        let text = make_text_with_ids(&[1, 2, 3]);
        let input = MetricInput::Single(&text);
        assert_eq!(
            UniqueWordCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(3)
        );
    }

    #[test]
    fn duplicates() {
        // "the cat the dog" -> IDs: 1, 2, 1, 3 -> 3 unique
        let text = make_text_with_ids(&[1, 2, 1, 3]);
        let input = MetricInput::Single(&text);
        assert_eq!(
            UniqueWordCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(3)
        );
    }

    #[test]
    fn all_same() {
        let text = make_text_with_ids(&[5, 5, 5, 5]);
        let input = MetricInput::Single(&text);
        assert_eq!(
            UniqueWordCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(1)
        );
    }

    #[test]
    fn rejects_pairwise() {
        let t1 = make_text_with_ids(&[1]);
        let t2 = make_text_with_ids(&[2]);
        let input = MetricInput::Pairwise(&t1, &t2);
        assert!(
            UniqueWordCountMetric
                .compute(&input, &HashMap::new())
                .is_err()
        );
    }
}
