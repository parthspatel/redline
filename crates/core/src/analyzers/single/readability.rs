//! Readability analysis - Flesch, SMOG, ARI metrics

use crate::analyzers::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::DiffResult;

/// Analyzes readability changes using multiple metrics
#[derive(Clone)]
pub struct ReadabilityAnalyzer;

impl ReadabilityAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Flesch Reading Ease Score
    /// Higher score = easier to read
    /// 90-100: Very easy (5th grade)
    /// 60-70: Standard (8th-9th grade)
    /// 0-30: Very difficult (college graduate)
    fn flesch_reading_ease(&self, text: &str) -> f64 {
        let sentences = self.count_sentences(text);
        let words = self.count_words(text);
        let syllables = self.count_syllables(text);

        if sentences == 0 || words == 0 {
            return 0.0;
        }

        let avg_sentence_length = words as f64 / sentences as f64;
        let avg_syllables_per_word = syllables as f64 / words as f64;

        206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    }

    /// Flesch-Kincaid Grade Level
    /// Returns U.S. grade level needed to understand the text
    fn flesch_kincaid_grade(&self, text: &str) -> f64 {
        let sentences = self.count_sentences(text);
        let words = self.count_words(text);
        let syllables = self.count_syllables(text);

        if sentences == 0 || words == 0 {
            return 0.0;
        }

        let avg_sentence_length = words as f64 / sentences as f64;
        let avg_syllables_per_word = syllables as f64 / words as f64;

        (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    }

    /// SMOG (Simple Measure of Gobbledygook) Index
    /// Estimates years of education needed
    fn smog_index(&self, text: &str) -> f64 {
        let sentences = self.count_sentences(text);
        let polysyllables = self.count_polysyllables(text);

        if sentences == 0 {
            return 0.0;
        }

        let polysyllables_per_sentence = polysyllables as f64 / sentences as f64;
        1.0430 * (polysyllables_per_sentence * 30.0).sqrt() + 3.1291
    }

    /// Automated Readability Index
    fn automated_readability_index(&self, text: &str) -> f64 {
        let sentences = self.count_sentences(text);
        let words = self.count_words(text);
        let characters = text.chars().filter(|c| !c.is_whitespace()).count();

        if sentences == 0 || words == 0 {
            return 0.0;
        }

        let chars_per_word = characters as f64 / words as f64;
        let words_per_sentence = words as f64 / sentences as f64;

        (4.71 * chars_per_word) + (0.5 * words_per_sentence) - 21.43
    }

    fn count_sentences(&self, text: &str) -> usize {
        text.chars()
            .filter(|c| matches!(c, '.' | '!' | '?'))
            .count()
            .max(1)
    }

    fn count_words(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn count_syllables(&self, text: &str) -> usize {
        text.split_whitespace()
            .map(|word| self.syllables_in_word(word))
            .sum()
    }

    fn syllables_in_word(&self, word: &str) -> usize {
        let word = word.to_lowercase();
        let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];

        let mut count = 0;
        let mut last_was_vowel = false;

        for ch in word.chars() {
            let is_vowel = vowels.contains(&ch);
            if is_vowel && !last_was_vowel {
                count += 1;
            }
            last_was_vowel = is_vowel;
        }

        // Adjust for silent 'e'
        if word.ends_with('e') && count > 1 {
            count -= 1;
        }

        count.max(1)
    }

    fn count_polysyllables(&self, text: &str) -> usize {
        text.split_whitespace()
            .filter(|word| self.syllables_in_word(word) >= 3)
            .count()
    }
}

impl Default for ReadabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for ReadabilityAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Compute metrics for both texts
        let orig_fre = self.flesch_reading_ease(&diff.original_text);
        let mod_fre = self.flesch_reading_ease(&diff.modified_text);

        let orig_fk = self.flesch_kincaid_grade(&diff.original_text);
        let mod_fk = self.flesch_kincaid_grade(&diff.modified_text);

        let orig_smog = self.smog_index(&diff.original_text);
        let mod_smog = self.smog_index(&diff.modified_text);

        let orig_ari = self.automated_readability_index(&diff.original_text);
        let mod_ari = self.automated_readability_index(&diff.modified_text);

        // Add metrics
        result.add_metric("original_flesch_reading_ease", orig_fre);
        result.add_metric("modified_flesch_reading_ease", mod_fre);
        result.add_metric("flesch_reading_ease_change", mod_fre - orig_fre);

        result.add_metric("original_flesch_kincaid_grade", orig_fk);
        result.add_metric("modified_flesch_kincaid_grade", mod_fk);
        result.add_metric("flesch_kincaid_grade_change", mod_fk - orig_fk);

        result.add_metric("original_smog_index", orig_smog);
        result.add_metric("modified_smog_index", mod_smog);
        result.add_metric("smog_index_change", mod_smog - orig_smog);

        result.add_metric("original_ari", orig_ari);
        result.add_metric("modified_ari", mod_ari);
        result.add_metric("ari_change", mod_ari - orig_ari);

        // Generate insights
        let fre_change = mod_fre - orig_fre;
        if fre_change.abs() > 5.0 {
            if fre_change > 0.0 {
                result.add_insight(format!(
                    "Readability improved: Flesch score increased by {:.1} points (easier to read)",
                    fre_change
                ));
            } else {
                result.add_insight(format!(
                    "Readability decreased: Flesch score decreased by {:.1} points (harder to read)",
                    fre_change.abs()
                ));
            }
        }

        let grade_change = mod_fk - orig_fk;
        if grade_change.abs() > 1.0 {
            if grade_change > 0.0 {
                result.add_insight(format!(
                    "Grade level increased by {:.1} grades (more complex)",
                    grade_change
                ));
            } else {
                result.add_insight(format!(
                    "Grade level decreased by {:.1} grades (simpler)",
                    grade_change.abs()
                ));
            }
        }

        // Classify readability level
        let level = if mod_fre >= 80.0 {
            "very_easy"
        } else if mod_fre >= 60.0 {
            "easy"
        } else if mod_fre >= 50.0 {
            "moderate"
        } else if mod_fre >= 30.0 {
            "difficult"
        } else {
            "very_difficult"
        };

        result.add_metadata("readability_level", level);

        result
    }

    fn name(&self) -> &str {
        "readability"
    }

    fn description(&self) -> &str {
        "Analyzes readability using Flesch, Flesch-Kincaid, SMOG, and ARI metrics"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

/// Analyzer that computes and caches readability difference
#[derive(Clone)]
pub struct ReadabilityDiffAnalyzer;

impl ReadabilityDiffAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadabilityDiffAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for ReadabilityDiffAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        // Use cached metrics if available
        let metrics = diff.get_metrics_ref();
        result.add_metric("readability_diff", metrics.readability_diff);
        result.add_metric("original_flesch", metrics.original.flesch_reading_ease);
        result.add_metric("modified_flesch", metrics.modified.flesch_reading_ease);

        result
    }

    fn name(&self) -> &str {
        "readability_diff"
    }

    fn description(&self) -> &str {
        "Computes difference in Flesch Reading Ease scores"
    }

    fn dependencies(&self) -> Vec<crate::execution::NodeDependencies> {
        use crate::execution::{ExecutionNode, MetricType, NodeDependencies};

        vec![
            NodeDependencies::new(ExecutionNode::Metric(MetricType::ReadabilityDiff))
                .with_dependency(ExecutionNode::Metric(MetricType::FleschReadingEase)),
        ]
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
