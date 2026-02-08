//! Damerau-Levenshtein edit distance metric on token sequences.

use hashbrown::HashMap;

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Token-level Damerau-Levenshtein distance (insert, delete, substitute, transpose).
/// Uses the optimal string alignment (restricted edit distance) variant.
pub struct DamerauLevenshteinDistanceMetric;

fn damerau_levenshtein(a: &[&str], b: &[&str]) -> usize {
    let n = a.len();
    let m = b.len();

    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }

    // Need three rows for transposition lookback
    let mut prev2 = vec![0usize; m + 1];
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);

            // Transposition
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                curr[j] = curr[j].min(prev2[j - 2] + cost);
            }
        }
        core::mem::swap(&mut prev2, &mut prev);
        core::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

impl Metric for DamerauLevenshteinDistanceMetric {
    fn id(&self) -> &str {
        "damerau_levenshtein"
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
                Ok(MetricValue::Integer(damerau_levenshtein(&a, &b) as i64))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "damerau_levenshtein".into(),
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
        let result = DamerauLevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }

    #[test]
    fn transposition() {
        let a = process("cat dog");
        let b = process("dog cat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = DamerauLevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(1));
    }

    #[test]
    fn transposition_vs_levenshtein() {
        let a = process("cat dog");
        let b = process("dog cat");
        let input = MetricInput::Pairwise(&a, &b);
        let dl = DamerauLevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();

        let lev = super::super::levenshtein::LevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();

        assert_eq!(dl, MetricValue::Integer(1));
        assert_eq!(lev, MetricValue::Integer(2));
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = DamerauLevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(0));
    }

    #[test]
    fn one_empty() {
        let a = process("");
        let b = process("cat dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = DamerauLevenshteinDistanceMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Integer(2));
    }
}
