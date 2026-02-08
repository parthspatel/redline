//! Sentence count metric: counts sentence-ending punctuation in original text.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts sentences by detecting sentence-ending punctuation (`.`, `!`, `?`).
/// Consecutive sentence-enders (e.g., `...`, `?!`) count as one.
pub struct SentenceCountMetric;

impl SentenceCountMetric {
    fn count_sentences(text: &str) -> i64 {
        if text.is_empty() {
            return 0;
        }
        let mut count = 0i64;
        let mut in_ender = false;
        for ch in text.chars() {
            if ch == '.' || ch == '!' || ch == '?' {
                if !in_ender {
                    count += 1;
                    in_ender = true;
                }
            } else {
                in_ender = false;
            }
        }
        count
    }
}

impl Metric for SentenceCountMetric {
    fn id(&self) -> &str {
        "sentence_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                Ok(MetricValue::Integer(Self::count_sentences(&text.original)))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "sentence_count".into(),
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
            SentenceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(0)
        );
    }

    #[test]
    fn single_sentence() {
        let text = make_text("Hello world.");
        let input = MetricInput::Single(&text);
        assert_eq!(
            SentenceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(1)
        );
    }

    #[test]
    fn multiple_sentences() {
        let text = make_text("Hello! How are you? Fine.");
        let input = MetricInput::Single(&text);
        assert_eq!(
            SentenceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(3)
        );
    }

    #[test]
    fn no_sentence_enders() {
        let text = make_text("Hello world");
        let input = MetricInput::Single(&text);
        assert_eq!(
            SentenceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(0)
        );
    }

    #[test]
    fn ellipsis_counts_as_one() {
        let text = make_text("Wait... really?");
        let input = MetricInput::Single(&text);
        assert_eq!(
            SentenceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(2)
        );
    }

    #[test]
    fn mixed_enders() {
        let text = make_text("What?! No way...");
        let input = MetricInput::Single(&text);
        assert_eq!(
            SentenceCountMetric
                .compute(&input, &HashMap::new())
                .unwrap(),
            MetricValue::Integer(2)
        );
    }

    #[test]
    fn rejects_pairwise() {
        let t1 = make_text("a");
        let t2 = make_text("b");
        let input = MetricInput::Pairwise(&t1, &t2);
        assert!(
            SentenceCountMetric
                .compute(&input, &HashMap::new())
                .is_err()
        );
    }
}
