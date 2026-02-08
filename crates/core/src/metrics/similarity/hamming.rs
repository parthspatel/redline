//! Hamming distance metric on token sequences.

use hashbrown::HashMap;

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Token-level Hamming distance: count of positions where tokens differ.
/// Only defined for equal-length sequences.
pub struct HammingDistanceMetric;

impl Metric for HammingDistanceMetric {
    fn id(&self) -> &str {
        "hamming_distance"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                let a = token_texts(source);
                let b = token_texts(target);

                if a.len() != b.len() {
                    return Err(MetricError::InvalidInput(
                        "hamming_distance".into(),
                        "requires equal-length token sequences".into(),
                    ));
                }

                let distance = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();

                Ok(MetricValue::Integer(distance as i64))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "hamming_distance".into(),
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
    fn identical() {
        let a = process("the cat sat");
        let b = process("the cat sat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = HammingDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }

    #[test]
    fn one_different() {
        let a = process("the cat sat");
        let b = process("the dog sat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = HammingDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(1));
    }

    #[test]
    fn all_different() {
        let a = process("cat dog");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = HammingDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(2));
    }

    #[test]
    fn unequal_lengths_error() {
        let a = process("cat");
        let b = process("cat dog");
        let input = MetricInput::Pairwise(&a, &b);
        assert!(
            HammingDistanceMetric
                .compute(&input, &HashMap::new())
                .is_err()
        );
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = HammingDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }
}
