//! Jaro similarity metric on token sequences.

use hashbrown::HashMap;

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Token-level Jaro similarity in [0.0, 1.0].
pub struct JaroSimilarityMetric;

/// Compute Jaro similarity on two string slices.
pub fn jaro_similarity(a: &[&str], b: &[&str]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let match_window = (a.len().max(b.len()) / 2).saturating_sub(1);

    let mut a_matched = vec![false; a.len()];
    let mut b_matched = vec![false; b.len()];
    let mut matches = 0usize;

    for i in 0..a.len() {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(b.len());
        for j in start..end {
            if !b_matched[j] && a[i] == b[j] {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut transpositions = 0usize;
    let mut k = 0;
    for i in 0..a.len() {
        if !a_matched[i] {
            continue;
        }
        while !b_matched[k] {
            k += 1;
        }
        if a[i] != b[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let t = transpositions as f64 / 2.0;
    (m / a.len() as f64 + m / b.len() as f64 + (m - t) / m) / 3.0
}

impl Metric for JaroSimilarityMetric {
    fn id(&self) -> &str {
        "jaro_similarity"
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
                Ok(MetricValue::Float(jaro_similarity(&a, &b)))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "jaro_similarity".into(),
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
        let result = JaroSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let sim = result.as_float().unwrap();
        assert!((sim - 1.0).abs() < 0.001, "got {sim}");
    }

    #[test]
    fn completely_different() {
        let a = process("cat dog");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = JaroSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn partial_match() {
        let a = process("the cat sat");
        let b = process("the dog sat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = JaroSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let sim = result.as_float().unwrap();
        assert!(sim > 0.0 && sim < 1.0, "got {sim}");
    }

    #[test]
    fn both_empty() {
        let a = process("");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = JaroSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn one_empty() {
        let a = process("cat");
        let b = process("");
        let input = MetricInput::Pairwise(&a, &b);
        let result = JaroSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn in_range() {
        let a = process("the big cat sat on mat");
        let b = process("a small dog lay on rug");
        let input = MetricInput::Pairwise(&a, &b);
        let result = JaroSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let sim = result.as_float().unwrap();
        assert!((0.0..=1.0).contains(&sim), "expected [0,1], got {sim}");
    }
}
