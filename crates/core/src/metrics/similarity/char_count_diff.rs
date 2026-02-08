//! Character count difference metric: target - source original lengths.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Difference in character count between target and source original text.
pub struct CharCountDiffMetric;

impl Metric for CharCountDiffMetric {
    fn id(&self) -> &str {
        "char_count_diff"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                let diff =
                    target.original.chars().count() as i64 - source.original.chars().count() as i64;
                Ok(MetricValue::Integer(diff))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "char_count_diff".into(),
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
        let a = process("cat");
        let b = process("dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = CharCountDiffMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }

    #[test]
    fn target_longer() {
        let a = process("hi");
        let b = process("hello");
        let input = MetricInput::Pairwise(&a, &b);
        let result = CharCountDiffMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(3));
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = CharCountDiffMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }
}
