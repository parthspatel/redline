//! SMOG Index metric.

use hashbrown::HashMap;

use crate::metrics::counts::count_syllables;
use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// SMOG Index (Simple Measure of Gobbledygook).
/// Formula: 1.0430 * sqrt(polysyllables * 30 / sentences) + 3.1291
/// A polysyllabic word has 3 or more syllables.
pub struct SmogIndexMetric;

impl Metric for SmogIndexMetric {
    fn id(&self) -> &str {
        "smog_index"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                let sentences = deps
                    .get("sentence_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("sentence_count".into()))?;

                if sentences == 0 {
                    return Ok(MetricValue::Float(0.0));
                }

                let mut polysyllables = 0i64;
                for token in &text.tokens {
                    let span = token.span;
                    let end = (span.end() as usize).min(text.normalized.len());
                    let word = &text.normalized[span.start() as usize..end];
                    if count_syllables(word) >= 3 {
                        polysyllables += 1;
                    }
                }

                let s = sentences as f64;
                let smog = 1.0430 * (polysyllables as f64 * 30.0 / s).sqrt() + 3.1291;
                Ok(MetricValue::Float(smog))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "smog_index".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["sentence_count"]
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
    fn no_polysyllables() {
        // "The cat sat." — 0 polysyllabic words, 1 sentence
        // smog = 1.0430 * sqrt(0*30/1) + 3.1291 = 3.1291
        let text = process("The cat sat.");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("sentence_count".into(), MetricValue::Integer(1));
        let result = SmogIndexMetric.compute(&input, &deps).unwrap();
        let smog = result.as_float().unwrap();
        assert!((smog - 3.1291).abs() < 0.01, "got {smog}");
    }

    #[test]
    fn with_polysyllables() {
        // "The beautiful universe." — "beautiful" (3 syl), "universe" (3 syl)
        // = 2 polysyllabic words, 1 sentence
        // smog = 1.0430 * sqrt(2*30/1) + 3.1291 = 1.0430 * sqrt(60) + 3.1291
        //      = 1.0430 * 7.746 + 3.1291 ≈ 8.08 + 3.13 ≈ 11.21
        let text = process("The beautiful universe.");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("sentence_count".into(), MetricValue::Integer(1));
        let result = SmogIndexMetric.compute(&input, &deps).unwrap();
        let smog = result.as_float().unwrap();
        assert!(smog > 5.0, "expected high smog, got {smog}");
    }

    #[test]
    fn zero_sentences() {
        let text = process("no ending punctuation");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("sentence_count".into(), MetricValue::Integer(0));
        let result = SmogIndexMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn metadata() {
        assert_eq!(SmogIndexMetric.id(), "smog_index");
        assert_eq!(SmogIndexMetric.static_dependencies(), &["sentence_count"]);
    }
}
