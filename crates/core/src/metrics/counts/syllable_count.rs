//! Syllable count metric: sums syllable estimates for each word token.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Counts total syllables across all word tokens using a heuristic algorithm.
pub struct SyllableCountMetric;

/// Heuristic syllable counter for English words (~90-95% accuracy).
///
/// Algorithm:
/// 1. Lowercase and strip non-alpha characters
/// 2. Count vowel groups (a, e, i, o, u; y when not first char)
/// 3. Subtract 1 for silent-e (trailing 'e' unless word ends in "le")
/// 4. Handle -ed suffix (not a syllable unless preceded by t or d)
/// 5. Minimum 1 syllable per word
pub fn count_syllables(word: &str) -> usize {
    let lower: String = word
        .chars()
        .filter(|c| c.is_alphabetic())
        .collect::<String>()
        .to_lowercase();
    if lower.is_empty() {
        return 0;
    }
    if lower.len() <= 2 {
        return 1;
    }

    let chars: Vec<char> = lower.chars().collect();
    let mut count = 0usize;
    let mut prev_vowel = false;

    for (i, &ch) in chars.iter().enumerate() {
        let is_vowel = matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u') || (ch == 'y' && i > 0);
        if is_vowel && !prev_vowel {
            count += 1;
        }
        prev_vowel = is_vowel;
    }

    // -ed suffix: usually not a syllable unless preceded by t or d
    let ends_with_ed = chars.len() >= 3 && chars.ends_with(&['e', 'd']);
    if ends_with_ed {
        let before_ed = chars[chars.len() - 3];
        if before_ed != 't' && before_ed != 'd' && count > 1 {
            count -= 1;
        }
    } else if chars.last() == Some(&'e') && count > 1 {
        // Silent-e: if word ends in 'e' (but not "le" or "-ed"), subtract 1
        let len = chars.len();
        if len < 2 || chars[len - 2] != 'l' {
            count -= 1;
        }
    }

    count.max(1)
}

impl Metric for SyllableCountMetric {
    fn id(&self) -> &str {
        "syllable_count"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        _dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                let total: usize = text
                    .tokens
                    .iter()
                    .map(|token| {
                        let span = token.span;
                        let end = (span.end() as usize).min(text.normalized.len());
                        let word = &text.normalized[span.start() as usize..end];
                        count_syllables(word)
                    })
                    .sum();
                Ok(MetricValue::Integer(total as i64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "syllable_count".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &[]
    }

    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }

    fn cost(&self) -> f32 {
        0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syllable_hello() {
        assert_eq!(count_syllables("hello"), 2);
    }

    #[test]
    fn syllable_the() {
        assert_eq!(count_syllables("the"), 1);
    }

    #[test]
    fn syllable_beautiful() {
        assert_eq!(count_syllables("beautiful"), 3);
    }

    #[test]
    fn syllable_a() {
        assert_eq!(count_syllables("a"), 1);
    }

    #[test]
    fn syllable_empty() {
        assert_eq!(count_syllables(""), 0);
    }

    #[test]
    fn syllable_simple() {
        assert_eq!(count_syllables("simple"), 2);
    }

    #[test]
    fn syllable_jumped() {
        // jumped: 1 syllable (ed not separate)
        assert_eq!(count_syllables("jumped"), 1);
    }

    #[test]
    fn syllable_created() {
        // Heuristic: "ea" counted as one vowel group, so cre(a)t-ed = 2
        // (true value is 3, but "ea" digraph fools simple vowel-group counting)
        assert_eq!(count_syllables("created"), 2);
    }

    #[test]
    fn syllable_i() {
        assert_eq!(count_syllables("I"), 1);
    }

    #[test]
    fn syllable_people() {
        // peo-ple: 2 syllables (ends in "le")
        assert_eq!(count_syllables("people"), 2);
    }

    #[test]
    fn rejects_pairwise() {
        use crate::process::ProcessedText;
        use crate::text_store::TextStoreBuilder;

        let store = TextStoreBuilder::new().build();
        let t1 = ProcessedText {
            original: "a".into(),
            layers: vec![],
            normalized: "a".into(),
            tokens: vec![],
            composed_mapping: None,
            text_store: store.clone(),
            skipped_normalizers: vec![],
        };
        let t2 = ProcessedText {
            original: "b".into(),
            layers: vec![],
            normalized: "b".into(),
            tokens: vec![],
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        };
        let input = MetricInput::Pairwise(&t1, &t2);
        assert!(
            SyllableCountMetric
                .compute(&input, &HashMap::new())
                .is_err()
        );
    }
}
