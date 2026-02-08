//! Automated Readability Index (ARI) metric.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Automated Readability Index.
/// Formula: 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
pub struct AriMetric;

impl Metric for AriMetric {
    fn id(&self) -> &str {
        "ari"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(_) => {
                let chars = deps
                    .get("char_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("char_count".into()))?;
                let words = deps
                    .get("word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("word_count".into()))?;
                let sentences = deps
                    .get("sentence_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("sentence_count".into()))?;

                if words == 0 || sentences == 0 {
                    return Ok(MetricValue::Float(0.0));
                }

                let c = chars as f64;
                let w = words as f64;
                let s = sentences as f64;

                let ari = 4.71 * (c / w) + 0.5 * (w / s) - 21.43;
                Ok(MetricValue::Float(ari))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "ari".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["char_count", "word_count", "sentence_count"]
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

    fn make_deps(chars: i64, words: i64, sentences: i64) -> HashMap<String, MetricValue> {
        let mut deps = HashMap::new();
        deps.insert("char_count".into(), MetricValue::Integer(chars));
        deps.insert("word_count".into(), MetricValue::Integer(words));
        deps.insert("sentence_count".into(), MetricValue::Integer(sentences));
        deps
    }

    fn dummy_text() -> crate::process::ProcessedText {
        let store = crate::text_store::TextStoreBuilder::new().build();
        crate::process::ProcessedText {
            original: "".into(),
            layers: vec![],
            normalized: "".into(),
            tokens: vec![],
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        }
    }

    #[test]
    fn known_value() {
        // 50 chars, 10 words, 1 sentence
        // 4.71*(50/10) + 0.5*(10/1) - 21.43
        // = 23.55 + 5.0 - 21.43 = 7.12
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(50, 10, 1);
        let result = AriMetric.compute(&input, &deps).unwrap();
        let ari = result.as_float().unwrap();
        assert!((ari - 7.12).abs() < 0.01, "got {ari}");
    }

    #[test]
    fn zero_words() {
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(0, 0, 0);
        let result = AriMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn zero_sentences() {
        let text = dummy_text();
        let input = MetricInput::Single(&text);
        let deps = make_deps(50, 10, 0);
        let result = AriMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }

    #[test]
    fn metadata() {
        assert_eq!(AriMetric.id(), "ari");
        assert_eq!(
            AriMetric.static_dependencies(),
            &["char_count", "word_count", "sentence_count"]
        );
    }
}
