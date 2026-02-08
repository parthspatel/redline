//! Similarity ratio metric via LCS (Longest Common Subsequence) on token sequences.

use hashbrown::HashMap;

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Similarity ratio: 2 * LCS_length / (|source| + |target|).
/// Uses two-row DP for LCS computation.
pub struct SimilarityRatioMetric;

fn lcs_length(a: &[&str], b: &[&str]) -> usize {
    let n = a.len();
    let m = b.len();

    let mut prev = vec![0usize; m + 1];
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n {
        for j in 1..=m {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        core::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

impl Metric for SimilarityRatioMetric {
    fn id(&self) -> &str {
        "similarity_ratio"
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

                if a.is_empty() && b.is_empty() {
                    return Ok(MetricValue::Float(1.0));
                }

                let lcs = lcs_length(&a, &b);
                let total = a.len() + b.len();
                Ok(MetricValue::Float(2.0 * lcs as f64 / total as f64))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "similarity_ratio".into(),
                "expected Pairwise input".into(),
            )),
        }
    }

    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }

    fn cost(&self) -> f32 {
        0.5
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
        let result = SimilarityRatioMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let ratio = result.as_float().unwrap();
        assert!((ratio - 1.0).abs() < 0.001, "got {ratio}");
    }

    #[test]
    fn completely_different() {
        let a = process("cat dog");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = SimilarityRatioMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn partial_match() {
        // "the cat sat" vs "the dog sat" — LCS = ["the", "sat"], length 2
        // ratio = 2*2 / (3+3) = 4/6 ≈ 0.667
        let a = process("the cat sat");
        let b = process("the dog sat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = SimilarityRatioMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let ratio = result.as_float().unwrap();
        assert!((ratio - 0.667).abs() < 0.01, "got {ratio}");
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = SimilarityRatioMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn in_range() {
        let a = process("the big cat sat on mat");
        let b = process("a small dog lay on rug");
        let input = MetricInput::Pairwise(&a, &b);
        let result = SimilarityRatioMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let ratio = result.as_float().unwrap();
        assert!((0.0..=1.0).contains(&ratio), "got {ratio}");
    }
}
