//! Length ratio metric: target token count / source token count.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Ratio of target token count to source token count.
pub struct LengthRatioMetric;

impl Metric for LengthRatioMetric {
    fn id(&self) -> &str {
        "length_ratio"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                if source.tokens.is_empty() {
                    return Ok(MetricValue::Float(0.0));
                }
                Ok(MetricValue::Float(
                    target.tokens.len() as f64 / source.tokens.len() as f64,
                ))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "length_ratio".into(),
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
    fn equal_lengths() {
        let a = process("cat dog");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = LengthRatioMetric.compute(&input, &HashMap::new()).unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn double_length() {
        let a = process("cat");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = LengthRatioMetric.compute(&input, &HashMap::new()).unwrap();
        assert_eq!(result, MetricValue::Float(2.0));
    }

    #[test]
    fn source_empty() {
        let a = process("");
        let b = process("cat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = LengthRatioMetric.compute(&input, &HashMap::new()).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }
}
