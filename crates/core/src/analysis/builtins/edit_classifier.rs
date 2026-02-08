//! EditClassifier: 7-category intent taxonomy with ranked confidence scores.
//!
//! Each operation gets all 7 categories with confidence, sorted descending.
//! Adjacent operations are grouped for group-level classification.

use core::any::Any;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, string::ToString, vec, vec::Vec};

use crate::analysis::{
    AnalysisContext, AnalysisError, AnalysisReport, AnalyzerDependency, AnalyzerMeta,
    AnalyzerPlugin,
};
use crate::diff::edit_operation::EditKind;

/// The 7 intent categories for classifying edit operations.
///
/// This enum is a locked design decision per 05-CONTEXT.md.
/// Later plans will expand `EditClassifierResult` but must NOT change these variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntentCategory {
    /// Fixing a typo, grammar, or factual error.
    Correction,
    /// Adding new content (sentences, paragraphs).
    Expansion,
    /// Removing content without replacement.
    Deletion,
    /// Substantially rewriting existing content.
    Rewrite,
    /// Minor improvements to wording or clarity.
    Refinement,
    /// Whitespace, punctuation, or structural changes only.
    Formatting,
    /// Does not fit other categories.
    Other,
}

impl IntentCategory {
    /// All 7 intent categories.
    pub fn all() -> &'static [IntentCategory] {
        &[
            Self::Correction,
            Self::Expansion,
            Self::Deletion,
            Self::Rewrite,
            Self::Refinement,
            Self::Formatting,
            Self::Other,
        ]
    }
}

/// Classification of a single operation with all 7 categories ranked.
#[derive(Debug, Clone)]
pub struct OperationClassification {
    /// Index of the operation in the diff.
    pub operation_index: usize,
    /// All 7 categories sorted by confidence descending.
    pub categories: Vec<(IntentCategory, f64)>,
}

/// Classification of a group of adjacent operations.
#[derive(Debug, Clone)]
pub struct GroupClassification {
    /// Indices of operations in this group.
    pub operation_indices: Vec<usize>,
    /// Aggregated categories sorted by confidence descending.
    pub categories: Vec<(IntentCategory, f64)>,
}

/// Result of edit classification analysis.
#[derive(Debug, Clone)]
pub struct EditClassifierResult {
    /// Per-operation classifications.
    pub operations: Vec<OperationClassification>,
    /// Group-level classifications for adjacent operations.
    pub groups: Vec<GroupClassification>,
}

/// Configurable thresholds for the edit classifier.
#[derive(Debug, Clone)]
pub struct EditClassifierConfig {
    /// Maximum edit distance to classify as typo correction.
    pub typo_max_edit_distance: usize,
    /// Maximum length ratio (target/source) to classify as typo correction.
    pub typo_max_length_ratio: f64,
    /// Minimum length ratio (target/source) to classify as expansion.
    pub expansion_length_ratio: f64,
    /// Maximum length ratio (target/source) to classify as deletion-like.
    pub deletion_length_ratio: f64,
    /// Whether formatting detection only considers whitespace changes.
    pub formatting_only_whitespace: bool,
    /// Maximum Equal-operation gap to bridge when grouping adjacent changes.
    pub group_gap_threshold: usize,
}

impl Default for EditClassifierConfig {
    fn default() -> Self {
        Self {
            typo_max_edit_distance: 2,
            typo_max_length_ratio: 1.3,
            expansion_length_ratio: 1.5,
            deletion_length_ratio: 0.5,
            formatting_only_whitespace: true,
            group_gap_threshold: 2,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────

/// Simple Levenshtein edit distance with early bail-out.
fn simple_edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    // Early bail-out for large length differences
    if m.abs_diff(n) > m.max(n) / 2 + 1 {
        return m.max(n);
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        core::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Returns true if the change is whitespace/punctuation only.
fn is_formatting_only(source: &str, target: &str) -> bool {
    let strip = |s: &str| -> String {
        s.chars()
            .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
            .collect()
    };
    strip(source) == strip(target)
}

/// Classify a single operation into all 7 categories with confidence.
///
/// Returns all 7 categories with baseline minimum 0.01, normalized to sum ~1.0,
/// sorted by confidence descending.
fn classify_operation(
    kind: EditKind,
    source_text: &str,
    target_text: &str,
    config: &EditClassifierConfig,
) -> Vec<(IntentCategory, f64)> {
    let mut scores = [0.0f64; 7]; // indexed by category order in all()

    match kind {
        EditKind::Insert => {
            scores[1] = 0.80; // Expansion
            scores[6] = 0.05; // Other
            scores[5] = 0.03; // Formatting
            scores[4] = 0.02; // Refinement
        }
        EditKind::Delete => {
            scores[2] = 0.80; // Deletion
            scores[6] = 0.05; // Other
            scores[4] = 0.03; // Refinement
            scores[5] = 0.02; // Formatting
        }
        EditKind::Replace => {
            if is_formatting_only(source_text, target_text) {
                scores[5] = 0.90; // Formatting
            } else {
                let source_len = source_text.len();
                let target_len = target_text.len();
                let edit_dist = simple_edit_distance(source_text, target_text);

                let len_ratio = if source_len > 0 {
                    target_len as f64 / source_len as f64
                } else {
                    f64::MAX
                };

                // Typo/correction detection
                if edit_dist <= config.typo_max_edit_distance
                    && len_ratio <= config.typo_max_length_ratio
                {
                    scores[0] = 0.85; // Correction
                    scores[4] = 0.05; // Refinement
                }
                // Expansion detection
                else if len_ratio > config.expansion_length_ratio {
                    scores[1] = 0.60; // Expansion
                    scores[3] = 0.15; // Rewrite
                    scores[4] = 0.10; // Refinement
                }
                // Deletion-like shrink
                else if len_ratio < config.deletion_length_ratio {
                    scores[2] = 0.40; // Deletion
                    scores[4] = 0.30; // Refinement
                    scores[3] = 0.10; // Rewrite
                }
                // Default: refinement or rewrite
                else {
                    let similarity = 1.0
                        - (edit_dist as f64 / source_len.max(target_len).max(1) as f64).min(1.0);
                    if similarity > 0.6 {
                        scores[4] = 0.55; // Refinement
                        scores[3] = 0.20; // Rewrite
                        scores[0] = 0.10; // Correction
                    } else {
                        scores[3] = 0.55; // Rewrite
                        scores[4] = 0.20; // Refinement
                        scores[1] = 0.05; // Expansion
                    }
                }
            }
        }
        EditKind::Equal => {
            scores[6] = 0.90; // Other (shouldn't be classified)
        }
    }

    // Apply baseline minimum and normalize
    let categories = IntentCategory::all();
    let mut result: Vec<(IntentCategory, f64)> = categories
        .iter()
        .zip(scores.iter())
        .map(|(&cat, &score)| (cat, score.max(0.01)))
        .collect();

    let total: f64 = result.iter().map(|(_, s)| s).sum();
    if total > 0.0 {
        for item in &mut result {
            item.1 /= total;
        }
    }

    // Sort descending by confidence
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    result
}

/// Group adjacent non-Equal operations, bridging small Equal gaps.
fn group_adjacent_operations(
    operations: &[crate::diff::EditOperation],
    gap_threshold: usize,
) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group: Vec<usize> = Vec::new();
    let mut gap = 0usize;

    for (i, op) in operations.iter().enumerate() {
        if op.kind != EditKind::Equal {
            if !current_group.is_empty() && gap > gap_threshold {
                groups.push(core::mem::take(&mut current_group));
            }
            current_group.push(i);
            gap = 0;
        } else if !current_group.is_empty() {
            gap += 1;
        }
    }

    if !current_group.is_empty() {
        groups.push(current_group);
    }

    groups
}

/// Classify a group by averaging per-operation classifications.
fn classify_group(
    group_indices: &[usize],
    op_classifications: &[OperationClassification],
) -> Vec<(IntentCategory, f64)> {
    let categories = IntentCategory::all();
    let mut sums = [0.0f64; 7];
    let mut count = 0usize;

    for &idx in group_indices {
        if let Some(op_class) = op_classifications.iter().find(|c| c.operation_index == idx) {
            for &(cat, conf) in &op_class.categories {
                let pos = categories.iter().position(|&c| c == cat).unwrap_or(6);
                sums[pos] += conf;
            }
            count += 1;
        }
    }

    if count > 0 {
        for s in &mut sums {
            *s /= count as f64;
        }
    }

    let mut result: Vec<(IntentCategory, f64)> = categories
        .iter()
        .zip(sums.iter())
        .map(|(&cat, &avg)| (cat, avg))
        .collect();

    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    result
}

// ── Analyzer ──────────────────────────────────────────────────

/// Edit classifier with configurable thresholds.
pub struct EditClassifier {
    config: EditClassifierConfig,
}

impl EditClassifier {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: EditClassifierConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: EditClassifierConfig) -> Self {
        Self { config }
    }
}

impl Default for EditClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyzerPlugin for EditClassifier {
    fn id(&self) -> &str {
        "edit_classifier"
    }

    fn meta(&self) -> AnalyzerMeta {
        AnalyzerMeta {
            id: "edit_classifier".into(),
            name: "Edit Classifier".into(),
            version: "1.0.0".into(),
        }
    }

    fn dependencies(&self) -> Vec<AnalyzerDependency> {
        Vec::new()
    }

    fn cost(&self) -> f32 {
        0.3
    }

    fn analyze(
        &self,
        context: &AnalysisContext,
        _prior_results: &AnalysisReport,
    ) -> Result<Box<dyn Any + Send + Sync>, AnalysisError> {
        let changes = context.change_operations();

        let operations: Vec<OperationClassification> = changes
            .iter()
            .map(|&(idx, op)| {
                let source_text = context.source_text_for_op(idx).unwrap_or("");
                let target_text = context.target_text_for_op(idx).unwrap_or("");
                let categories =
                    classify_operation(op.kind, source_text, target_text, &self.config);
                OperationClassification {
                    operation_index: idx,
                    categories,
                }
            })
            .collect();

        let groups_indices =
            group_adjacent_operations(&context.diff.operations, self.config.group_gap_threshold);

        let groups: Vec<GroupClassification> = groups_indices
            .into_iter()
            .map(|group_indices| {
                let categories = classify_group(&group_indices, &operations);
                GroupClassification {
                    operation_indices: group_indices,
                    categories,
                }
            })
            .collect();

        Ok(Box::new(EditClassifierResult { operations, groups }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_identical() {
        assert_eq!(simple_edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn edit_distance_one_char() {
        assert_eq!(simple_edit_distance("cat", "bat"), 1);
    }

    #[test]
    fn edit_distance_substitution() {
        assert_eq!(simple_edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn edit_distance_empty() {
        assert_eq!(simple_edit_distance("", "abc"), 3);
        assert_eq!(simple_edit_distance("abc", ""), 3);
        assert_eq!(simple_edit_distance("", ""), 0);
    }

    #[test]
    fn classify_insert_as_expansion() {
        let config = EditClassifierConfig::default();
        let cats = classify_operation(EditKind::Insert, "", "new content", &config);
        assert_eq!(cats[0].0, IntentCategory::Expansion);
    }

    #[test]
    fn classify_delete_as_deletion() {
        let config = EditClassifierConfig::default();
        let cats = classify_operation(EditKind::Delete, "removed content", "", &config);
        assert_eq!(cats[0].0, IntentCategory::Deletion);
    }

    #[test]
    fn classify_whitespace_replace_as_formatting() {
        let config = EditClassifierConfig::default();
        let cats = classify_operation(EditKind::Replace, "hello  world", "hello world", &config);
        assert_eq!(cats[0].0, IntentCategory::Formatting);
    }

    #[test]
    fn classify_typo_replace_as_correction() {
        let config = EditClassifierConfig::default();
        let cats = classify_operation(EditKind::Replace, "teh", "the", &config);
        assert_eq!(cats[0].0, IntentCategory::Correction);
    }

    #[test]
    fn classify_long_expansion_replace() {
        let config = EditClassifierConfig::default();
        let cats = classify_operation(
            EditKind::Replace,
            "short",
            "a much longer replacement text here",
            &config,
        );
        assert_eq!(cats[0].0, IntentCategory::Expansion);
    }

    #[test]
    fn all_seven_categories_present() {
        let config = EditClassifierConfig::default();
        let cats = classify_operation(EditKind::Insert, "", "new text", &config);
        assert_eq!(cats.len(), 7);
        let unique: hashbrown::HashSet<_> = cats.iter().map(|(c, _)| *c).collect();
        assert_eq!(unique.len(), 7);
    }

    #[test]
    fn categories_sorted_descending() {
        let config = EditClassifierConfig::default();
        let cats = classify_operation(
            EditKind::Replace,
            "old text here",
            "new text there",
            &config,
        );
        for w in cats.windows(2) {
            assert!(w[0].1 >= w[1].1, "not sorted: {:?} < {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn categories_sum_to_one() {
        let config = EditClassifierConfig::default();
        let cats = classify_operation(EditKind::Replace, "hello", "world", &config);
        let sum: f64 = cats.iter().map(|(_, c)| c).sum();
        assert!((sum - 1.0).abs() < 0.01, "sum should be ~1.0, got {}", sum);
    }

    #[test]
    fn intent_category_all_has_seven() {
        assert_eq!(IntentCategory::all().len(), 7);
    }

    #[test]
    fn group_adjacent_operations_basic() {
        use crate::diff::EditOperation;
        let ops = vec![
            EditOperation::equal(0, 2, 0, 2),
            EditOperation::replace(2, 4, 2, 5),
            EditOperation::equal(4, 5, 5, 6),
            EditOperation::delete(5, 7, 6),
            EditOperation::equal(7, 10, 6, 9),
        ];
        let groups = group_adjacent_operations(&ops, 1);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].contains(&1));
        assert!(groups[0].contains(&3));
    }

    #[test]
    fn classify_group_averages() {
        let op_classes = vec![
            OperationClassification {
                operation_index: 1,
                categories: IntentCategory::all()
                    .iter()
                    .map(|&c| {
                        if c == IntentCategory::Correction {
                            (c, 0.80)
                        } else {
                            (c, 0.033)
                        }
                    })
                    .collect(),
            },
            OperationClassification {
                operation_index: 3,
                categories: IntentCategory::all()
                    .iter()
                    .map(|&c| {
                        if c == IntentCategory::Expansion {
                            (c, 0.80)
                        } else {
                            (c, 0.033)
                        }
                    })
                    .collect(),
            },
        ];
        let result = classify_group(&[1, 3], &op_classes);
        // Both Correction and Expansion should have averaged to ~0.4
        let correction = result
            .iter()
            .find(|(c, _)| *c == IntentCategory::Correction)
            .unwrap()
            .1;
        let expansion = result
            .iter()
            .find(|(c, _)| *c == IntentCategory::Expansion)
            .unwrap()
            .1;
        assert!((correction - expansion).abs() < 0.01);
        assert!(correction > 0.3); // should be ~0.416
    }

    #[test]
    fn edit_classifier_id() {
        assert_eq!(EditClassifier::new().id(), "edit_classifier");
    }

    #[test]
    fn edit_classifier_no_deps() {
        assert!(EditClassifier::new().dependencies().is_empty());
    }

    #[test]
    fn edit_classifier_default_config() {
        let config = EditClassifierConfig::default();
        assert_eq!(config.typo_max_edit_distance, 2);
        assert!((config.typo_max_length_ratio - 1.3).abs() < f64::EPSILON);
        assert!((config.expansion_length_ratio - 1.5).abs() < f64::EPSILON);
        assert!((config.deletion_length_ratio - 0.5).abs() < f64::EPSILON);
    }
}
