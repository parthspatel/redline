//! Pattern detection across multiple diffs

use crate::analyzer::{AnalysisResult, MultiDiffAnalyzer};
use crate::diff::{DiffResult, EditType};
use std::collections::HashMap;

/// Detects common patterns across multiple diffs
#[derive(Clone)]
pub struct PatternDetectionAnalyzer {
    /// Minimum frequency for a pattern to be reported
    pub min_frequency: f64,
}

impl PatternDetectionAnalyzer {
    pub fn new() -> Self {
        Self {
            min_frequency: 0.3, // 30%
        }
    }

    pub fn with_min_frequency(mut self, frequency: f64) -> Self {
        self.min_frequency = frequency;
        self
    }
}

impl Default for PatternDetectionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDiffAnalyzer for PatternDetectionAnalyzer {
    fn analyze(&self, diffs: &[&DiffResult]) -> AnalysisResult {
        let mut result = AnalysisResult::new(self.name());

        if diffs.is_empty() {
            return result;
        }

        let total = diffs.len() as f64;

        // Pattern 1: Dominant edit type
        let mut edit_type_counts: HashMap<EditType, usize> = HashMap::new();
        for diff in diffs {
            for op in &diff.operations {
                *edit_type_counts.entry(op.edit_type).or_insert(0) += 1;
            }
        }

        if let Some((dominant_type, count)) = edit_type_counts
            .iter()
            .filter(|(t, _)| **t != EditType::Equal)
            .max_by_key(|(_, count)| *count)
        {
            let frequency = *count as f64 / total;
            if frequency >= self.min_frequency {
                result.add_insight(format!(
                    "Pattern: {:?} edits are common ({:.1}% of operations)",
                    dominant_type,
                    frequency * 100.0
                ));
            }
        }

        // Pattern 2: Dominant category
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        for diff in diffs {
            for op in &diff.operations {
                if op.edit_type != EditType::Equal {
                    let cat = format!("{:?}", op.category);
                    *category_counts.entry(cat).or_insert(0) += 1;
                }
            }
        }

        if let Some((dominant_cat, count)) = category_counts.iter().max_by_key(|(_, count)| *count)
        {
            let frequency = *count as f64 / total;
            if frequency >= self.min_frequency {
                result.add_insight(format!(
                    "Pattern: {} changes are dominant ({:.1}% frequency)",
                    dominant_cat,
                    frequency * 100.0
                ));
                result.add_metadata("dominant_category", dominant_cat);
            }
        }

        // Pattern 3: Consistent expansion or reduction
        let expansions = diffs
            .iter()
            .filter(|d| d.statistics.insertions > d.statistics.deletions)
            .count();
        let reductions = diffs
            .iter()
            .filter(|d| d.statistics.deletions > d.statistics.insertions)
            .count();

        let expansion_freq = expansions as f64 / total;
        let reduction_freq = reductions as f64 / total;

        if expansion_freq >= self.min_frequency {
            result.add_insight(format!(
                "Pattern: Consistent text expansion ({:.1}% of diffs)",
                expansion_freq * 100.0
            ));
            result.add_metric("expansion_pattern_frequency", expansion_freq);
        }

        if reduction_freq >= self.min_frequency {
            result.add_insight(format!(
                "Pattern: Consistent text reduction ({:.1}% of diffs)",
                reduction_freq * 100.0
            ));
            result.add_metric("reduction_pattern_frequency", reduction_freq);
        }

        // Pattern 4: Low vs High similarity clustering
        let low_similarity = diffs.iter().filter(|d| d.semantic_similarity < 0.5).count();
        let high_similarity = diffs.iter().filter(|d| d.semantic_similarity > 0.8).count();

        if low_similarity as f64 / total >= self.min_frequency {
            result.add_insight(format!(
                "Pattern: Frequent substantial rewrites ({:.1}% with <50% similarity)",
                (low_similarity as f64 / total) * 100.0
            ));
        }

        if high_similarity as f64 / total >= self.min_frequency {
            result.add_insight(format!(
                "Pattern: Frequent minor refinements ({:.1}% with >80% similarity)",
                (high_similarity as f64 / total) * 100.0
            ));
        }

        result
    }

    fn name(&self) -> &str {
        "pattern_detection"
    }

    fn description(&self) -> &str {
        "Detects common editing patterns across multiple diffs"
    }

    fn clone_box(&self) -> Box<dyn MultiDiffAnalyzer> {
        Box::new(self.clone())
    }
}
