//! Readability delta metric: difference in Flesch Reading Ease between two texts.

use hashbrown::HashMap;

use crate::metrics::counts::count_syllables;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};
use crate::process::ProcessedText;

/// Difference in Flesch Reading Ease score between target and source.
pub struct ReadabilityDeltaMetric;

fn flesch_reading_ease(text: &ProcessedText) -> f64 {
    let words = text.tokens.len();
    if words == 0 {
        return 0.0;
    }

    let mut sentences = 0usize;
    let mut prev_ender = false;
    for ch in text.original.chars() {
        let is_ender = matches!(ch, '.' | '!' | '?');
        if is_ender && !prev_ender {
            sentences += 1;
        }
        prev_ender = is_ender;
    }
    if sentences == 0 {
        return 0.0;
    }

    let syllables: usize = text
        .tokens
        .iter()
        .map(|t| {
            let start = t.span.start() as usize;
            let end = (t.span.end() as usize).min(text.normalized.len());
            count_syllables(&text.normalized[start..end])
        })
        .sum();

    let w = words as f64;
    let s = sentences as f64;
    let syl = syllables as f64;
    206.835 - 1.015 * (w / s) - 84.6 * (syl / w)
}

impl Metric for ReadabilityDeltaMetric {
    fn id(&self) -> &str {
        "readability_delta"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                let source_fre = flesch_reading_ease(source);
                let target_fre = flesch_reading_ease(target);
                Ok(MetricValue::Float(target_fre - source_fre))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "readability_delta".into(),
                "expected Pairwise input".into(),
            )),
        }
    }

    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }

    fn cost(&self) -> f32 {
        0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextProcessor;
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
    fn identical_texts_zero_delta() {
        let a = process("The cat sat on the mat.");
        let b = process("The cat sat on the mat.");
        let input = MetricInput::Pairwise(&a, &b);
        let result = ReadabilityDeltaMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn different_texts_nonzero() {
        let simple = process("The cat sat.");
        let complex =
            process("The extraordinarily sophisticated algorithmic computation proceeded.");
        let input = MetricInput::Pairwise(&simple, &complex);
        let result = ReadabilityDeltaMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        // Complex text should have lower FRE, so delta should be negative
        let delta = result.as_float().unwrap();
        assert!(delta != 0.0, "expected nonzero delta");
    }

    #[test]
    fn rejects_single() {
        let a = process("test");
        let input = MetricInput::Single(&a);
        assert!(
            ReadabilityDeltaMetric
                .compute(&input, &HashMap::new())
                .is_err()
        );
    }
}
