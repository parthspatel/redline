//! Centralized text metrics computation and caching
//!
//! This module provides efficient, one-time computation of expensive text metrics
//! to avoid redundant calculations across analyzers, classifiers, and engines.

use std::collections::HashSet;

/// Metrics computed for a single text
#[derive(Debug, Clone)]
pub struct TextMetrics {
    // Basic counts
    pub char_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub syllable_count: usize,
    pub whitespace_count: usize,
    pub punctuation_count: usize,

    // Derived metrics
    pub flesch_reading_ease: f64,
    pub flesch_kincaid_grade: f64,
    pub avg_word_length: f64,
    pub avg_sentence_length: f64,

    // Linguistic features
    pub stopword_ratio: f64,
    pub whitespace_ratio: f64,
    pub has_negation: bool,

    // Raw text (for comparison operations)
    text: String,
}

impl TextMetrics {
    /// Compute all metrics for a given text
    pub fn compute(text: &str) -> Self {
        let char_count = text.len();
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();

        let sentence_count = text.chars()
            .filter(|c| matches!(c, '.' | '!' | '?'))
            .count()
            .max(1);

        let syllable_count: usize = words.iter()
            .map(|w| count_syllables(w))
            .sum();

        let whitespace_count = text.chars().filter(|c| c.is_whitespace()).count();
        let punctuation_count = text.chars().filter(|c| c.is_ascii_punctuation()).count();

        // Derived metrics
        let avg_word_length = if word_count > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64
        } else {
            0.0
        };

        let avg_sentence_length = if sentence_count > 0 {
            word_count as f64 / sentence_count as f64
        } else {
            0.0
        };

        let avg_syllables_per_word = if word_count > 0 {
            syllable_count as f64 / word_count as f64
        } else {
            0.0
        };

        let flesch_reading_ease = if word_count > 0 {
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        } else {
            0.0
        };

        let flesch_kincaid_grade = if word_count > 0 {
            (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        } else {
            0.0
        };

        // Linguistic features
        let stopword_ratio = compute_stopword_ratio(&words);
        let whitespace_ratio = if char_count > 0 {
            whitespace_count as f64 / char_count as f64
        } else {
            0.0
        };
        let has_negation = contains_negation(text);

        Self {
            char_count,
            word_count,
            sentence_count,
            syllable_count,
            whitespace_count,
            punctuation_count,
            flesch_reading_ease,
            flesch_kincaid_grade,
            avg_word_length,
            avg_sentence_length,
            stopword_ratio,
            whitespace_ratio,
            has_negation,
            text: text.to_string(),
        }
    }

    /// Get the raw text
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Metrics computed for a pair of texts (comparison metrics)
#[derive(Debug, Clone)]
pub struct PairwiseMetrics {
    pub original: TextMetrics,
    pub modified: TextMetrics,

    // Comparison metrics
    pub char_similarity: f64,
    pub word_overlap: f64,
    pub levenshtein_distance: usize,
    pub length_ratio: f64,

    // Diff metrics
    pub readability_diff: f64,
    pub word_count_diff: f64,
    pub whitespace_ratio_diff: f64,
    pub negation_changed: bool,
}

impl PairwiseMetrics {
    /// Compute pairwise metrics for two texts
    pub fn compute(original: &str, modified: &str) -> Self {
        let original_metrics = TextMetrics::compute(original);
        let modified_metrics = TextMetrics::compute(modified);

        // Compute comparison metrics
        let levenshtein_distance = levenshtein_distance(original, modified);
        let max_len = original.len().max(modified.len());
        let char_similarity = if max_len > 0 {
            1.0 - (levenshtein_distance as f64 / max_len as f64)
        } else {
            1.0
        };

        let word_overlap = compute_word_overlap(original, modified);

        let length_ratio = {
            let len1 = original.len() as f64;
            let len2 = modified.len() as f64;
            if len1 == 0.0 && len2 == 0.0 {
                1.0
            } else {
                let max_len = len1.max(len2);
                let min_len = len1.min(len2);
                if max_len == 0.0 {
                    0.0
                } else {
                    min_len / max_len
                }
            }
        };

        // Diff metrics
        let readability_diff = (modified_metrics.flesch_reading_ease - original_metrics.flesch_reading_ease).abs();
        let word_count_diff = (modified_metrics.word_count as f64 - original_metrics.word_count as f64).abs();
        let whitespace_ratio_diff = (modified_metrics.whitespace_ratio - original_metrics.whitespace_ratio).abs();
        let negation_changed = original_metrics.has_negation != modified_metrics.has_negation;

        Self {
            original: original_metrics,
            modified: modified_metrics,
            char_similarity,
            word_overlap,
            levenshtein_distance,
            length_ratio,
            readability_diff,
            word_count_diff,
            whitespace_ratio_diff,
            negation_changed,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn count_syllables(word: &str) -> usize {
    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let word_lower = word.to_lowercase();
    let chars: Vec<char> = word_lower.chars().collect();

    if chars.is_empty() {
        return 0;
    }

    let mut syllable_count = 0;
    let mut prev_was_vowel = false;

    for ch in chars.iter() {
        let is_vowel = vowels.contains(ch);
        if is_vowel && !prev_was_vowel {
            syllable_count += 1;
        }
        prev_was_vowel = is_vowel;
    }

    // Adjust for silent 'e' at end
    if word_lower.ends_with('e') && syllable_count > 1 {
        syllable_count -= 1;
    }

    syllable_count.max(1)
}

fn compute_stopword_ratio(words: &[&str]) -> f64 {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "for", "from", "has", "have", "he", "in", "is", "it", "its",
        "of", "on", "or", "that", "the", "to", "was", "were", "will", "with"
    ];

    if words.is_empty() {
        return 0.0;
    }

    let stopword_count = words.iter()
        .filter(|w| STOPWORDS.contains(&w.to_lowercase().as_str()))
        .count();

    stopword_count as f64 / words.len() as f64
}

fn contains_negation(text: &str) -> bool {
    const NEGATION_WORDS: &[&str] = &["not", "no", "never", "neither", "none", "nobody", "nothing", "nowhere"];

    let lower = text.to_lowercase();

    // Check for negation words
    for negation in NEGATION_WORDS {
        if lower.split_whitespace().any(|w| w == *negation) {
            return true;
        }
    }

    // Check for n't contractions
    lower.contains("n't")
}

fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                .min(matrix[i + 1][j] + 1)
                .min(matrix[i][j] + cost);
        }
    }

    matrix[len1][len2]
}

fn compute_word_overlap(s1: &str, s2: &str) -> f64 {
    let words1: HashSet<_> = s1
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    let words2: HashSet<_> = s2
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if words1.is_empty() && words2.is_empty() {
        return 1.0;
    }

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_metrics() {
        let text = "The cat sat on the mat.";
        let metrics = TextMetrics::compute(text);

        assert_eq!(metrics.word_count, 6);
        assert_eq!(metrics.sentence_count, 1);
        assert!(metrics.flesch_reading_ease > 0.0);
    }

    #[test]
    fn test_pairwise_metrics() {
        let metrics = PairwiseMetrics::compute("hello world", "hello rust");

        assert!(metrics.word_overlap > 0.0);
        assert!(metrics.char_similarity > 0.0);
        assert_eq!(metrics.word_count_diff, 0.0);
    }

    #[test]
    fn test_negation_detection() {
        assert!(contains_negation("I don't like this"));
        assert!(contains_negation("This is not good"));
        assert!(!contains_negation("This is good"));
    }
}
