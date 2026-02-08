//! ReadabilityAnalyzer: document-level readability delta + per-region attribution.
//!
//! Delegates to MetricsEngine for all score computation (does NOT re-implement formulas).

use core::any::Any;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, string::ToString, vec, vec::Vec};

use crate::analysis::{
    AnalysisContext, AnalysisError, AnalysisReport, AnalyzerDependency, AnalyzerMeta,
    AnalyzerPlugin,
};
use crate::diff::edit_operation::EditKind;

/// Readability scores from 6 standard metrics.
#[derive(Debug, Clone)]
pub struct ReadabilityScores {
    pub flesch_reading_ease: f64,
    pub flesch_kincaid_grade: f64,
    pub gunning_fog: f64,
    pub coleman_liau: f64,
    pub ari: f64,
    pub smog_index: f64,
}

/// Attribution of readability changes to specific edit regions.
#[derive(Debug, Clone)]
pub struct ReadabilityAttribution {
    /// Operation indices in this region.
    pub operation_indices: Vec<usize>,
    /// Estimated impact on readability (proportional).
    pub estimated_impact: f64,
    /// Human-readable description.
    pub description: String,
}

/// Result of readability analysis.
#[derive(Debug, Clone)]
pub struct ReadabilityResult {
    pub source_scores: ReadabilityScores,
    pub target_scores: ReadabilityScores,
    pub delta: ReadabilityScores,
    pub attributions: Vec<ReadabilityAttribution>,
}

// ── Helpers ───────────────────────────────────────────────────

fn compute_scores(context: &AnalysisContext) -> (ReadabilityScores, ReadabilityScores) {
    let get_source =
        |id: &str| -> f64 { context.compute_source_metric(id).as_float().unwrap_or(0.0) };
    let get_target =
        |id: &str| -> f64 { context.compute_target_metric(id).as_float().unwrap_or(0.0) };

    let source = ReadabilityScores {
        flesch_reading_ease: get_source("flesch_reading_ease"),
        flesch_kincaid_grade: get_source("flesch_kincaid_grade"),
        gunning_fog: get_source("gunning_fog"),
        coleman_liau: get_source("coleman_liau"),
        ari: get_source("ari"),
        smog_index: get_source("smog_index"),
    };
    let target = ReadabilityScores {
        flesch_reading_ease: get_target("flesch_reading_ease"),
        flesch_kincaid_grade: get_target("flesch_kincaid_grade"),
        gunning_fog: get_target("gunning_fog"),
        coleman_liau: get_target("coleman_liau"),
        ari: get_target("ari"),
        smog_index: get_target("smog_index"),
    };
    (source, target)
}

fn delta_scores(source: &ReadabilityScores, target: &ReadabilityScores) -> ReadabilityScores {
    ReadabilityScores {
        flesch_reading_ease: target.flesch_reading_ease - source.flesch_reading_ease,
        flesch_kincaid_grade: target.flesch_kincaid_grade - source.flesch_kincaid_grade,
        gunning_fog: target.gunning_fog - source.gunning_fog,
        coleman_liau: target.coleman_liau - source.coleman_liau,
        ari: target.ari - source.ari,
        smog_index: target.smog_index - source.smog_index,
    }
}

/// Group adjacent non-Equal operations, bridging small Equal gaps.
///
/// `gap_threshold` is the maximum number of Equal tokens between two
/// non-Equal operations that should be merged into the same group.
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

// ── Analyzer ──────────────────────────────────────────────────

/// Readability analyzer computing before/after readability scores via MetricsEngine.
pub struct ReadabilityAnalyzer;

impl AnalyzerPlugin for ReadabilityAnalyzer {
    fn id(&self) -> &str {
        "readability"
    }

    fn meta(&self) -> AnalyzerMeta {
        AnalyzerMeta {
            id: "readability".into(),
            name: "Readability Analyzer".into(),
            version: "1.0.0".into(),
        }
    }

    fn dependencies(&self) -> Vec<AnalyzerDependency> {
        Vec::new()
    }

    fn cost(&self) -> f32 {
        0.5
    }

    fn analyze(
        &self,
        context: &AnalysisContext,
        _prior_results: &AnalysisReport,
    ) -> Result<Box<dyn Any + Send + Sync>, AnalysisError> {
        let (source_scores, target_scores) = compute_scores(context);
        let delta = delta_scores(&source_scores, &target_scores);

        // Attribution: group adjacent operations and estimate impact
        let groups = group_adjacent_operations(&context.diff.operations, 2);

        // Total number of changed tokens for proportional impact
        let total_changed_tokens: usize = groups
            .iter()
            .flat_map(|g| g.iter())
            .map(|&i| {
                let op = &context.diff.operations[i];
                (op.source_len() + op.target_len()) as usize
            })
            .sum();

        let abs_fre_delta = delta.flesch_reading_ease.abs();

        let attributions = groups
            .into_iter()
            .map(|group_indices| {
                let group_tokens: usize = group_indices
                    .iter()
                    .map(|&i| {
                        let op = &context.diff.operations[i];
                        (op.source_len() + op.target_len()) as usize
                    })
                    .sum();

                let proportion = if total_changed_tokens > 0 {
                    group_tokens as f64 / total_changed_tokens as f64
                } else {
                    0.0
                };

                let estimated_impact = proportion * abs_fre_delta;

                ReadabilityAttribution {
                    operation_indices: group_indices,
                    estimated_impact,
                    description: format!(
                        "{} changed tokens ({:.0}% of changes)",
                        group_tokens,
                        proportion * 100.0
                    ),
                }
            })
            .collect();

        Ok(Box::new(ReadabilityResult {
            source_scores,
            target_scores,
            delta,
            attributions,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_scores_computation() {
        let source = ReadabilityScores {
            flesch_reading_ease: 60.0,
            flesch_kincaid_grade: 8.0,
            gunning_fog: 10.0,
            coleman_liau: 9.0,
            ari: 7.0,
            smog_index: 8.5,
        };
        let target = ReadabilityScores {
            flesch_reading_ease: 70.0,
            flesch_kincaid_grade: 6.0,
            gunning_fog: 8.0,
            coleman_liau: 7.5,
            ari: 5.5,
            smog_index: 7.0,
        };
        let d = delta_scores(&source, &target);
        assert!((d.flesch_reading_ease - 10.0).abs() < f64::EPSILON);
        assert!((d.flesch_kincaid_grade - (-2.0)).abs() < f64::EPSILON);
        assert!((d.gunning_fog - (-2.0)).abs() < f64::EPSILON);
        assert!((d.coleman_liau - (-1.5)).abs() < f64::EPSILON);
        assert!((d.ari - (-1.5)).abs() < f64::EPSILON);
        assert!((d.smog_index - (-1.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn delta_scores_negative() {
        let source = ReadabilityScores {
            flesch_reading_ease: 80.0,
            flesch_kincaid_grade: 5.0,
            gunning_fog: 7.0,
            coleman_liau: 6.0,
            ari: 4.0,
            smog_index: 5.5,
        };
        let target = ReadabilityScores {
            flesch_reading_ease: 50.0,
            flesch_kincaid_grade: 10.0,
            gunning_fog: 12.0,
            coleman_liau: 11.0,
            ari: 9.0,
            smog_index: 10.5,
        };
        let d = delta_scores(&source, &target);
        assert!(d.flesch_reading_ease < 0.0);
        assert!(d.flesch_kincaid_grade > 0.0);
    }

    #[test]
    fn readability_analyzer_id() {
        assert_eq!(ReadabilityAnalyzer.id(), "readability");
    }

    #[test]
    fn readability_analyzer_no_analyzer_deps() {
        assert!(ReadabilityAnalyzer.dependencies().is_empty());
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
        // gap_threshold=1: bridge the single Equal between Replace and Delete
        let groups = group_adjacent_operations(&ops, 1);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].contains(&1)); // Replace
        assert!(groups[0].contains(&3)); // Delete
    }

    #[test]
    fn group_adjacent_no_changes() {
        use crate::diff::EditOperation;
        let ops = vec![
            EditOperation::equal(0, 5, 0, 5),
            EditOperation::equal(5, 10, 5, 10),
        ];
        let groups = group_adjacent_operations(&ops, 2);
        assert!(groups.is_empty());
    }

    #[test]
    fn group_adjacent_single_change() {
        use crate::diff::EditOperation;
        let ops = vec![
            EditOperation::equal(0, 3, 0, 3),
            EditOperation::insert(3, 3, 5),
            EditOperation::equal(3, 8, 5, 10),
        ];
        let groups = group_adjacent_operations(&ops, 2);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![1]);
    }
}
