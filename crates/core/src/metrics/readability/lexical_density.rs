//! Lexical density metric: content words / total words.

use hashbrown::HashMap;

use crate::metrics::{DependencyKind, Metric, MetricError, MetricInput, MetricValue};

/// Common English function words (articles, prepositions, conjunctions, pronouns,
/// determiners, auxiliary verbs). Words NOT in this list are "content words".
static FUNCTION_WORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "nor",
    "for",
    "yet",
    "so",
    "if",
    "then",
    "else",
    "when",
    "while",
    "as",
    "at",
    "by",
    "in",
    "of",
    "on",
    "to",
    "up",
    "off",
    "out",
    "from",
    "into",
    "onto",
    "with",
    "about",
    "above",
    "after",
    "against",
    "along",
    "among",
    "around",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "beyond",
    "during",
    "except",
    "inside",
    "near",
    "outside",
    "over",
    "past",
    "since",
    "through",
    "toward",
    "towards",
    "under",
    "until",
    "upon",
    "within",
    "without",
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "can",
    "could",
    "must",
    "not",
    "no",
    "very",
    "too",
    "also",
    "just",
    "more",
    "most",
    "much",
    "many",
    "some",
    "any",
    "all",
    "each",
    "every",
    "both",
    "few",
    "several",
    "such",
    "than",
    "how",
    "here",
    "there",
    "where",
];

/// Ratio of content words to total words.
pub struct LexicalDensityMetric;

impl Metric for LexicalDensityMetric {
    fn id(&self) -> &str {
        "lexical_density"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(text) => {
                let words = deps
                    .get("word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("word_count".into()))?;
                if words == 0 {
                    return Ok(MetricValue::Float(0.0));
                }

                let mut content_count = 0i64;
                for token in &text.tokens {
                    let span = token.span;
                    let end = (span.end() as usize).min(text.normalized.len());
                    let word = &text.normalized[span.start() as usize..end];
                    let lower = word.to_lowercase();
                    // Strip trailing punctuation for matching
                    let clean: &str = lower.trim_end_matches(|c: char| c.is_ascii_punctuation());
                    if !clean.is_empty() && !FUNCTION_WORDS.contains(&clean) {
                        content_count += 1;
                    }
                }

                Ok(MetricValue::Float(content_count as f64 / words as f64))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "lexical_density".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["word_count"]
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
    fn all_content_words() {
        let text = process("beautiful sunset mountains");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("word_count".into(), MetricValue::Integer(3));
        let result = LexicalDensityMetric.compute(&input, &deps).unwrap();
        // All 3 are content words
        assert_eq!(result, MetricValue::Float(1.0));
    }

    #[test]
    fn mixed_content_function() {
        let text = process("the big red dog");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("word_count".into(), MetricValue::Integer(4));
        let result = LexicalDensityMetric.compute(&input, &deps).unwrap();
        // "the" is function, "big", "red", "dog" are content = 3/4 = 0.75
        assert_eq!(result, MetricValue::Float(0.75));
    }

    #[test]
    fn zero_words() {
        let text = process("");
        let input = MetricInput::Single(&text);
        let mut deps = HashMap::new();
        deps.insert("word_count".into(), MetricValue::Integer(0));
        let result = LexicalDensityMetric.compute(&input, &deps).unwrap();
        assert_eq!(result, MetricValue::Float(0.0));
    }
}
