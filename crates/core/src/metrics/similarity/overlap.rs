//! Overlap coefficient metric on token sets.

use hashbrown::{HashMap, HashSet};

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Overlap coefficient: |A ∩ B| / min(|A|, |B|) on token text sets.
pub struct OverlapCoefficientMetric;

impl Metric for OverlapCoefficientMetric {
    fn id(&self) -> &str {
        "overlap_coefficient"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                let a: HashSet<&str> = token_texts(source).into_iter().collect();
                let b: HashSet<&str> = token_texts(target).into_iter().collect();

                if a.is_empty() && b.is_empty() {
                    return Ok(MetricValue::Float(1.0));
                }

                let min_size = a.len().min(b.len());
                if min_size == 0 {
                    return Ok(MetricValue::Float(0.0));
                }

                let intersection = a.intersection(&b).count();
                Ok(MetricValue::Float(intersection as f64 / min_size as f64))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "overlap_coefficient".into(),
                "expected Pairwise input".into(),
            )),
        }
    }

    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }

    fn cost(&self) -> f32 {
        0.1
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
    fn subset_overlap() {
        let a = process("the cat");
        let b = process("the cat dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = OverlapCoefficientMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn no_overlap() {
        let a = process("cat");
        let b = process("dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = OverlapCoefficientMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn one_empty() {
        let a = process("cat");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = OverlapCoefficientMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = OverlapCoefficientMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }
}
