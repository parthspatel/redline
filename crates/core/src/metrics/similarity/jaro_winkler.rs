//! Jaro-Winkler similarity metric on token sequences.

use hashbrown::HashMap;

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Token-level Jaro-Winkler similarity.
/// Adds a prefix bonus to the Jaro score: jw = jaro + l*p*(1-jaro)
/// where l = common prefix length (max 4), p = 0.1.
pub struct JaroWinklerSimilarityMetric;

impl Metric for JaroWinklerSimilarityMetric {
    fn id(&self) -> &str {
        "jaro_winkler_similarity"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                let jaro = deps
                    .get("jaro_similarity")
                    .and_then(|v| v.as_float())
                    .ok_or_else(|| MetricError::DependencyUnavailable("jaro_similarity".into()))?;

                let a = token_texts(source);
                let b = token_texts(target);

                // Common prefix length (up to 4)
                let prefix_len = a
                    .iter()
                    .zip(b.iter())
                    .take(4)
                    .take_while(|(x, y)| x == y)
                    .count();

                let p = 0.1;
                let jw = jaro + (prefix_len as f64) * p * (1.0 - jaro);
                Ok(MetricValue::Float(jw))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "jaro_winkler_similarity".into(),
                "expected Pairwise input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["jaro_similarity"]
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
    fn identical() {
        let a = process("the cat sat");
        let b = process("the cat sat");
        let input = MetricInput::Pairwise(&a, &b);
        let mut deps = HashMap::new();
        deps.insert("jaro_similarity".into(), MetricValue::Float(1.0));
        let result = JaroWinklerSimilarityMetric.compute(&input, &deps).unwrap();
        let jw = result.as_float().unwrap();
        assert!((jw - 1.0).abs() < 0.001, "got {jw}");
    }

    #[test]
    fn prefix_bonus() {
        // "the cat sat" vs "the cat dog" — prefix = 2 ("the", "cat")
        let a = process("the cat sat");
        let b = process("the cat dog");
        let input = MetricInput::Pairwise(&a, &b);
        let mut deps = HashMap::new();
        deps.insert("jaro_similarity".into(), MetricValue::Float(0.777));
        let result = JaroWinklerSimilarityMetric.compute(&input, &deps).unwrap();
        let jw = result.as_float().unwrap();
        assert!(jw > 0.777, "jw should be >= jaro, got {jw}");
    }

    #[test]
    fn no_prefix() {
        // "cat" vs "dog" — prefix = 0, jaro=0.0
        // jw = 0.0 + 0*0.1*(1-0.0) = 0.0
        let a = process("cat");
        let b = process("dog");
        let input = MetricInput::Pairwise(&a, &b);
        let mut deps = HashMap::new();
        deps.insert("jaro_similarity".into(), MetricValue::Float(0.0));
        let result = JaroWinklerSimilarityMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn jw_geq_jaro() {
        let a = process("the big cat sat");
        let b = process("the big dog lay");
        let input = MetricInput::Pairwise(&a, &b);

        let a_texts = token_texts(&a);
        let b_texts = token_texts(&b);
        let jaro_val = super::super::jaro::jaro_similarity(&a_texts, &b_texts);

        let mut deps = HashMap::new();
        deps.insert("jaro_similarity".into(), MetricValue::Float(jaro_val));
        let result = JaroWinklerSimilarityMetric.compute(&input, &deps).unwrap();
        let jw = result.as_float().unwrap();
        assert!(jw >= jaro_val, "jw={jw} should be >= jaro={jaro_val}");
    }

    #[test]
    fn metadata() {
        assert_eq!(JaroWinklerSimilarityMetric.id(), "jaro_winkler_similarity");
        assert_eq!(
            JaroWinklerSimilarityMetric.static_dependencies(),
            &["jaro_similarity"]
        );
    }
}
