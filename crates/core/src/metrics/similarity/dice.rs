//! Dice coefficient metric on token sets.

use hashbrown::{HashMap, HashSet};

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Dice coefficient: 2*|A ∩ B| / (|A| + |B|) on token text sets.
pub struct DiceCoefficientMetric;

impl Metric for DiceCoefficientMetric {
    fn id(&self) -> &str {
        "dice_coefficient"
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

                let intersection = a.intersection(&b).count();
                let denominator = a.len() + b.len();

                Ok(MetricValue::Float(
                    2.0 * intersection as f64 / denominator as f64,
                ))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "dice_coefficient".into(),
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
    fn identical_texts() {
        let a = process("the cat sat");
        let b = process("the cat sat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = DiceCoefficientMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn completely_different() {
        let a = process("cat dog");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = DiceCoefficientMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn partial_overlap() {
        let a = process("the cat");
        let b = process("the dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = DiceCoefficientMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let dice = result.as_float().unwrap();
        assert!((dice - 0.5).abs() < 0.001, "got {dice}");
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = DiceCoefficientMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }
}
