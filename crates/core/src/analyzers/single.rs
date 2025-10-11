//! Single diff analyzers
//!
//! Analyzers that operate on individual diff results

use super::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::{ChangeCategory, DiffResult, EditType};
use std::collections::HashMap;

// ============================================================================
// Semantic Similarity Analyzer
// ============================================================================

/// Analyzes semantic similarity between original and modified text
#[derive(Clone)]
pub struct SemanticSimilarityAnalyzer {
    /// Threshold for considering texts semantically similar
    pub similarity_threshold: f64,
}

impl SemanticSimilarityAnalyzer {
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.7,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }
}

impl Default for SemanticSimilarityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for SemanticSimilarityAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        
        let similarity = diff.semantic_similarity;
        result.add_metric("semantic_similarity", similarity);
        result.add_metric("semantic_distance", 1.0 - similarity);
        
        // Classify similarity level
        let level = if similarity >= 0.9 {
            "very_high"
        } else if similarity >= self.similarity_threshold {
            "high"
        } else if similarity >= 0.5 {
            "moderate"
        } else if similarity >= 0.3 {
            "low"
        } else {
            "very_low"
        };
        
        result.add_metadata("similarity_level", level);
        
        // Add insights
        if similarity >= self.similarity_threshold {
            result.add_insight(format!(
                "Texts are semantically similar ({:.1}% similarity). Core meaning preserved.",
                similarity * 100.0
            ));
        } else {
            result.add_insight(format!(
                "Significant semantic changes detected ({:.1}% similarity). Meaning substantially altered.",
                similarity * 100.0
            ));
        }
        
        // Analyze by edit type
        let semantic_changes = diff.operations.iter()
            .filter(|op| matches!(op.category, ChangeCategory::Semantic))
            .count();
        
        if semantic_changes > 0 {
            result.add_insight(format!(
                "{} semantic change operations detected",
                semantic_changes
            ));
        }
        
        result
    }

    fn name(&self) -> &str {
        "semantic_similarity"
    }

    fn description(&self) -> &str {
        "Analyzes semantic similarity and meaning preservation"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Readability Analyzer
// ============================================================================

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

// ============================================================================
// Stylistic Change Analyzer
// ============================================================================

/// Analyzes stylistic changes in the text
#[derive(Clone)]
pub struct StylisticAnalyzer;

impl StylisticAnalyzer {
    pub fn new() -> Self {
        Self
    }

    fn average_word_length(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let total: usize = words.iter().map(|w| w.len()).sum();
        total as f64 / words.len() as f64
    }

    fn average_sentence_length(&self, text: &str) -> f64 {
        let sentences = text.split(|c| matches!(c, '.' | '!' | '?')).count();
        let words = text.split_whitespace().count();
        
        if sentences == 0 {
            return 0.0;
        }
        
        words as f64 / sentences as f64
    }

    fn punctuation_density(&self, text: &str) -> f64 {
        let punct_count = text.chars().filter(|c| c.is_ascii_punctuation()).count();
        let total_chars = text.len();
        
        if total_chars == 0 {
            return 0.0;
        }
        
        punct_count as f64 / total_chars as f64
    }

    fn lexical_diversity(&self, text: &str) -> f64 {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        
        if words.is_empty() {
            return 0.0;
        }
        
        let unique_words: std::collections::HashSet<_> = words.iter().collect();
        unique_words.len() as f64 / words.len() as f64
    }
}

impl Default for StylisticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for StylisticAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        
        // Compute stylistic metrics
        let orig_awl = self.average_word_length(&diff.original_text);
        let mod_awl = self.average_word_length(&diff.modified_text);
        
        let orig_asl = self.average_sentence_length(&diff.original_text);
        let mod_asl = self.average_sentence_length(&diff.modified_text);
        
        let orig_pd = self.punctuation_density(&diff.original_text);
        let mod_pd = self.punctuation_density(&diff.modified_text);
        
        let orig_ld = self.lexical_diversity(&diff.original_text);
        let mod_ld = self.lexical_diversity(&diff.modified_text);
        
        // Add metrics
        result.add_metric("original_avg_word_length", orig_awl);
        result.add_metric("modified_avg_word_length", mod_awl);
        result.add_metric("avg_word_length_change", mod_awl - orig_awl);
        
        result.add_metric("original_avg_sentence_length", orig_asl);
        result.add_metric("modified_avg_sentence_length", mod_asl);
        result.add_metric("avg_sentence_length_change", mod_asl - orig_asl);
        
        result.add_metric("original_punctuation_density", orig_pd);
        result.add_metric("modified_punctuation_density", mod_pd);
        result.add_metric("punctuation_density_change", mod_pd - orig_pd);
        
        result.add_metric("original_lexical_diversity", orig_ld);
        result.add_metric("modified_lexical_diversity", mod_ld);
        result.add_metric("lexical_diversity_change", mod_ld - orig_ld);
        
        // Count stylistic changes
        let stylistic_changes = diff.operations.iter()
            .filter(|op| matches!(op.category, ChangeCategory::Stylistic))
            .count();
        
        result.add_metric("stylistic_operations", stylistic_changes as f64);
        
        // Generate insights
        if (mod_awl - orig_awl).abs() > 0.5 {
            if mod_awl > orig_awl {
                result.add_insight("Vocabulary became more sophisticated (longer words)");
            } else {
                result.add_insight("Vocabulary became simpler (shorter words)");
            }
        }
        
        if (mod_asl - orig_asl).abs() > 2.0 {
            if mod_asl > orig_asl {
                result.add_insight("Sentences became longer and more complex");
            } else {
                result.add_insight("Sentences became shorter and more concise");
            }
        }
        
        if (mod_ld - orig_ld).abs() > 0.1 {
            if mod_ld > orig_ld {
                result.add_insight("Vocabulary diversity increased (more varied word choice)");
            } else {
                result.add_insight("Vocabulary diversity decreased (more repetitive)");
            }
        }
        
        result
    }

    fn name(&self) -> &str {
        "stylistic"
    }

    fn description(&self) -> &str {
        "Analyzes stylistic properties like word length, sentence structure, and vocabulary"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Edit Category Distribution Analyzer
// ============================================================================

/// Analyzes the distribution of edit categories
#[derive(Clone)]
pub struct CategoryDistributionAnalyzer;

impl CategoryDistributionAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CategoryDistributionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for CategoryDistributionAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        let mut total_changes = 0;
        
        for op in &diff.operations {
            if op.edit_type != EditType::Equal {
                let category = format!("{:?}", op.category);
                *category_counts.entry(category).or_insert(0) += 1;
                total_changes += 1;
            }
        }
        
        // Add metrics as percentages
        for (category, count) in &category_counts {
            let percentage = if total_changes > 0 {
                (*count as f64 / total_changes as f64) * 100.0
            } else {
                0.0
            };
            
            result.add_metric(
                format!("{}_percentage", category.to_lowercase()),
                percentage
            );
            result.add_metric(
                format!("{}_count", category.to_lowercase()),
                *count as f64
            );
        }
        
        result.add_metric("total_changes", total_changes as f64);
        
        // Find dominant category
        if let Some((dominant_cat, dominant_count)) = category_counts.iter()
            .max_by_key(|(_, count)| *count) {
            
            let percentage = (*dominant_count as f64 / total_changes as f64) * 100.0;
            result.add_metadata("dominant_category", dominant_cat);
            result.add_insight(format!(
                "Primary change type: {} ({:.1}% of changes)",
                dominant_cat, percentage
            ));
        }
        
        // Generate insights based on distribution
        let semantic_pct = category_counts.get("Semantic").unwrap_or(&0);
        let stylistic_pct = category_counts.get("Stylistic").unwrap_or(&0);
        let formatting_pct = category_counts.get("Formatting").unwrap_or(&0);
        
        if *semantic_pct as f64 / total_changes.max(1) as f64 > 0.5 {
            result.add_insight("Majority of changes are semantic (content/meaning)");
        }
        
        if *stylistic_pct as f64 / total_changes.max(1) as f64 > 0.3 {
            result.add_insight("Significant stylistic modifications detected");
        }
        
        if *formatting_pct as f64 / total_changes.max(1) as f64 > 0.5 {
            result.add_insight("Primarily formatting changes (no content modification)");
        }
        
        result
    }

    fn name(&self) -> &str {
        "category_distribution"
    }

    fn description(&self) -> &str {
        "Analyzes the distribution of edit categories (semantic, stylistic, etc.)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Edit Intent Classifier
// ============================================================================

/// Classifies the intent behind edits
#[derive(Clone)]
pub struct EditIntentClassifier;

impl EditIntentClassifier {
    pub fn new() -> Self {
        Self
    }

    fn classify_intent(&self, diff: &DiffResult) -> Vec<String> {
        let mut intents = Vec::new();
        
        // Check for clarification intent
        if diff.statistics.modifications > diff.statistics.insertions 
            && diff.semantic_similarity > 0.8 {
            intents.push("clarification".to_string());
        }
        
        // Check for expansion intent
        if diff.statistics.insertions > diff.statistics.deletions * 2 {
            intents.push("expansion".to_string());
        }
        
        // Check for condensation intent
        if diff.statistics.deletions > diff.statistics.insertions * 2 {
            intents.push("condensation".to_string());
        }
        
        // Check for correction intent
        let syntactic_changes = diff.operations.iter()
            .filter(|op| matches!(op.category, ChangeCategory::Syntactic))
            .count();
        
        if syntactic_changes > diff.operations.len() / 4 {
            intents.push("correction".to_string());
        }
        
        // Check for reformatting intent
        let formatting_changes = diff.operations.iter()
            .filter(|op| matches!(op.category, ChangeCategory::Formatting))
            .count();
        
        if formatting_changes > diff.operations.len() / 2 {
            intents.push("reformatting".to_string());
        }
        
        // Check for stylistic refinement
        let stylistic_changes = diff.operations.iter()
            .filter(|op| matches!(op.category, ChangeCategory::Stylistic))
            .count();
        
        if stylistic_changes > 0 && diff.semantic_similarity > 0.7 {
            intents.push("stylistic_refinement".to_string());
        }
        
        // Check for rewriting intent
        if diff.semantic_similarity < 0.5 {
            intents.push("rewrite".to_string());
        }
        
        intents
    }
}

impl Default for EditIntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleDiffAnalyzer for EditIntentClassifier {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());
        
        let intents = self.classify_intent(diff);
        
        for intent in &intents {
            result.add_metric(&format!("intent_{}", intent), 1.0);
            result.add_insight(format!("Detected intent: {}", intent));
        }
        
        result.add_metadata("primary_intent", 
            intents.first().unwrap_or(&"unknown".to_string()));
        
        if intents.is_empty() {
            result.add_insight("No clear edit intent detected");
        }
        
        result
    }

    fn name(&self) -> &str {
        "edit_intent"
    }

    fn description(&self) -> &str {
        "Classifies the intent behind edits (clarification, expansion, correction, etc.)"
    }

    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiffEngine;

    #[test]
    fn test_semantic_analyzer() {
        let engine = DiffEngine::default();
        let diff = engine.diff("Hello world", "Hello rust");
        
        let analyzer = SemanticSimilarityAnalyzer::new();
        let result = analyzer.analyze(&diff);
        
        assert!(result.metrics.contains_key("semantic_similarity"));
    }

    #[test]
    fn test_readability_analyzer() {
        let engine = DiffEngine::default();
        let diff = engine.diff(
            "This is simple.",
            "This is a significantly more complicated sentence with multiple clauses."
        );
        
        let analyzer = ReadabilityAnalyzer::new();
        let result = analyzer.analyze(&diff);
        
        assert!(result.metrics.contains_key("flesch_reading_ease_change"));
    }
}
