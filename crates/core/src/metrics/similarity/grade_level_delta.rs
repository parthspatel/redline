//! Grade level delta metric: difference in Flesch-Kincaid Grade between two texts.

use hashbrown::HashMap;

use crate::metrics::counts::count_syllables;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};
use crate::process::ProcessedText;

/// Difference in Flesch-Kincaid Grade Level between target and source.
pub struct GradeLevelDeltaMetric;

fn flesch_kincaid_grade(text: &ProcessedText) -> f64 {
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
    0.39 * (w / s) + 11.8 * (syl / w) - 15.59
}

impl Metric for GradeLevelDeltaMetric {
    fn id(&self) -> &str {
        "grade_level_delta"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Pairwise(source, target) => {
                let source_grade = flesch_kincaid_grade(source);
                let target_grade = flesch_kincaid_grade(target);
                Ok(MetricValue::Float(target_grade - source_grade))
            }
            MetricInput::Single(_) => Err(MetricError::InvalidInput(
                "grade_level_delta".into(),
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
        let result = GradeLevelDeltaMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn different_complexity() {
        let simple = process("I see a dog.");
        let complex = process("The extraordinarily sophisticated computation proceeded.");
        let input = MetricInput::Pairwise(&simple, &complex);
        let result = GradeLevelDeltaMetric
            .compute(&input, &HashMap::new())
            .unwrap();
        let delta = result.as_float().unwrap();
        // Complex text should have higher grade level
        assert!(
            delta > 0.0,
            "expected positive delta for complex target, got {delta}"
        );
    }
}
