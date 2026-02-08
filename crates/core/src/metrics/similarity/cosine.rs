//! Cosine similarity metric on token frequency vectors.

use hashbrown::HashMap;

use super::token_texts;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Cosine similarity using token frequency vectors.
pub struct CosineSimilarityMetric;

impl Metric for CosineSimilarityMetric {
    fn id(&self) -> &str {
        "cosine_similarity"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                let a_words = token_texts(source);
                let b_words = token_texts(target);

                if a_words.is_empty() && b_words.is_empty() {
                    return Ok(MetricValue::Float(1.0));
                }

                let mut a_freq: HashMap<&str, usize> = HashMap::new();
                for w in &a_words {
                    *a_freq.entry(w).or_insert(0) += 1;
                }
                let mut b_freq: HashMap<&str, usize> = HashMap::new();
                for w in &b_words {
                    *b_freq.entry(w).or_insert(0) += 1;
                }

                let mut dot = 0.0f64;
                for (word, &count_a) in &a_freq {
                    if let Some(&count_b) = b_freq.get(word) {
                        dot += count_a as f64 * count_b as f64;
                    }
                }

                let mag_a: f64 = a_freq
                    .values()
                    .map(|&c| (c as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let mag_b: f64 = b_freq
                    .values()
                    .map(|&c| (c as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if mag_a == 0.0 || mag_b == 0.0 {
                    return Ok(MetricValue::Float(0.0));
                }

                Ok(MetricValue::Float(dot / (mag_a * mag_b)))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "cosine_similarity".into(),
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
        let result = CosineSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let sim = result.as_float().unwrap();
        assert!((sim - 1.0).abs() < 0.001, "got {sim}");
    }

    #[test]
    fn orthogonal() {
        let a = process("cat dog");
        let b = process("fish bird");
        let input = MetricInput::Pairwise(&a, &b);
        let result = CosineSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn partial_overlap() {
        let a = process("the cat");
        let b = process("the dog");
        let input = MetricInput::Pairwise(&a, &b);
        let result = CosineSimilarityMetric
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
        let result = CosineSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn frequency_matters() {
        // "cat cat cat" vs "cat" — same direction = 1.0
        let a = process("cat cat cat");
        let b = process("cat");
        let input = MetricInput::Pairwise(&a, &b);
        let result = CosineSimilarityMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let sim = result.as_float().unwrap();
        assert!((sim - 1.0).abs() < 0.001, "got {sim}");
    }
}
