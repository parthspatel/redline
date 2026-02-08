//! Levenshtein edit distance metric on token sequences.

use hashbrown::HashMap;

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Token-level Levenshtein edit distance (insert, delete, substitute).
/// Uses two-row DP for O(min(n,m)) space.
pub struct LevenshteinDistanceMetric;

fn levenshtein(a: &[&str], b: &[&str]) -> usize {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let n = short.len();
    let m = long.len();

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for j in 1..=m {
        curr[0] = j;
        for i in 1..=n {
            let cost = if short[i - 1] == long[j - 1] { 0 } else { 1 };
            curr[i] = (prev[i] + 1).min(curr[i - 1] + 1).min(prev[i - 1] + cost);
        }
        core::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

impl Metric for LevenshteinDistanceMetric {
    fn id(&self) -> &str {
        "levenshtein_distance"
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
                Ok(MetricValue::Integer(levenshtein(&a, &b) as i64))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "levenshtein_distance".into(),
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
    fn identical() {
        let a = process("the cat sat");
        let b = process("the cat sat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = LevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }

    #[test]
    fn one_substitution() {
        let a = process("the cat");
        let b = process("the dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = LevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(1));
    }

    #[test]
    fn one_insertion() {
        let a = process("the cat");
        let b = process("the big cat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = LevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(1));
    }

    #[test]
    fn symmetric() {
        let a = process("cat dog");
        let b = process("dog fish bird");
        let input_ab = MetricInput::Pairwise(&a, &b);
        let input_ba = MetricInput::Pairwise(&b, &a);
        let r1 = LevenshteinDistanceMetric
            .compute(&input_ab, &HashMap::new())
            .unwrap();
        let r2 = LevenshteinDistanceMetric
            .compute(&input_ba, &HashMap::new())
            .unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = LevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }

    #[test]
    fn one_empty() {
        let a = process("");
        let b = process("cat dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = LevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(2));
    }
}
