//! Word count difference metric: target - source token counts.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Difference in word (token) count between target and source.
pub struct WordCountDiffMetric;

impl Metric for WordCountDiffMetric {
    fn id(&self) -> &str {
        "word_count_diff"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                let diff = target.tokens.len() as i64 - source.tokens.len() as i64;
                Ok(MetricValue::Integer(diff))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "word_count_diff".into(),
                "expected Pairwise input".into(),
            )),
        }
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
    use crate::TextProcessor;
    use crate::process::ProcessedText;
    use crate::tokenize::WordTokenizer;

    fn process(s: &str) -> ProcessedText {
        TextProcessor::new(
            vec![],
            Box::new(WordTokenizer),
            crate::process::ExecutionMode::All,
        )
        .unwrap()
        .process(s)
        .unwrap()
    }

    #[test]
    fn same_length() {
        let a = process("cat dog");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = WordCountDiffMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }

    #[test]
    fn target_longer() {
        let a = process("cat");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = WordCountDiffMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(1));
    }

    #[test]
    fn target_shorter() {
        let a = process("cat dog bird");
        let b = process("fish");
        let input = MetricInput::Pairwise(&a, &b);
        let result = WordCountDiffMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(-2));
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = WordCountDiffMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }
}
