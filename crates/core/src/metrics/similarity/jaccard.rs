//! Jaccard similarity metric on token sets.

use hashbrown::{HashMap, HashSet};

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Jaccard similarity: |A ∩ B| / |A ∪ B| on token text sets.
pub struct JaccardSimilarityMetric;

impl Metric for JaccardSimilarityMetric {
    fn id(&self) -> &str {
        "jaccard_similarity"
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
                let union = a.union(&b).count();

                Ok(MetricValue::Float(intersection as f64 / union as f64))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "jaccard_similarity".into(),
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
        let result = JaccardSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn completely_different() {
        let a = process("cat dog");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = JaccardSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn partial_overlap() {
        let a = process("the cat");
        let b = process("the dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = JaccardSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let sim = result.as_float().unwrap();
        assert!((sim - 1.0 / 3.0).abs() < 0.001, "got {sim}");
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = JaccardSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn rejects_single() {
        let a = process("test");
        let input = MetricInput::Single(&a);
        assert!(
            JaccardSimilarityMetric
                .compute(&input, &HashMap::new())
                .is_err()
        );
    }
}
