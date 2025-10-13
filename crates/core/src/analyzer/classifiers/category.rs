//! Category distribution and edit intent analysis

use crate::analyzer::{AnalysisResult, SingleDiffAnalyzer};
use crate::diff::{ChangeCategory, DiffResult, EditType};
use std::collections::HashMap;

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
                percentage,
            );
            result.add_metric(format!("{}_count", category.to_lowercase()), *count as f64);
        }

        result.add_metric("total_changes", total_changes as f64);

        // Find dominant category
        if let Some((dominant_cat, dominant_count)) =
            category_counts.iter().max_by_key(|(_, count)| *count)
        {
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
            && diff.semantic_similarity > 0.8
        {
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
        let syntactic_changes = diff
            .operations
            .iter()
            .filter(|op| matches!(op.category, ChangeCategory::Syntactic))
            .count();

        if syntactic_changes > diff.operations.len() / 4 {
            intents.push("correction".to_string());
        }

        // Check for reformatting intent
        let formatting_changes = diff
            .operations
            .iter()
            .filter(|op| matches!(op.category, ChangeCategory::Formatting))
            .count();

        if formatting_changes > diff.operations.len() / 2 {
            intents.push("reformatting".to_string());
        }

        // Check for stylistic refinement
        let stylistic_changes = diff
            .operations
            .iter()
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

        result.add_metadata(
            "primary_intent",
            intents.first().unwrap_or(&"unknown".to_string()),
        );

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
