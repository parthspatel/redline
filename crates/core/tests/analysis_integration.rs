//! Integration tests for the analysis framework covering all 5 Phase 5 success criteria.

mod common;

use redline_core::analysis::builtins::{
    EditClassifierResult, IntentCategory, ReadabilityResult, SemanticResult, StylisticResult,
};
use redline_core::analysis::report::AnalyzerStatus;
use redline_core::analysis::{
    AnalysisContext, AnalysisCoordinator, AnalysisError, AnalysisReport, AnalyzerDependency,
    AnalyzerMeta, AnalyzerPlugin, ExecutionPlanner, PluginRegistry, register_builtin_analyzers,
};
use redline_core::diff::DiffComputer;
use redline_core::metrics::{MetricRegistry, MetricsEngine};
use redline_core::register_builtins;

// ── Test helpers ──────────────────────────────────────────────

fn make_diff_and_engine(
    source: &str,
    target: &str,
) -> (redline_core::diff::DiffResult, MetricsEngine) {
    let src = common::process_arc(source);
    let tgt = common::process_arc(target);
    let diff = DiffComputer::new().compute(src, tgt, None).unwrap();

    let mut registry = MetricRegistry::new();
    let _ = register_builtins(&mut registry);
    let engine = MetricsEngine::new(registry);

    (diff, engine)
}

fn make_coordinator() -> AnalysisCoordinator {
    let mut registry = PluginRegistry::new();
    register_builtin_analyzers(&mut registry).unwrap();
    AnalysisCoordinator::new(registry)
}

struct MockAnalyzer {
    id: String,
    deps: Vec<AnalyzerDependency>,
}

impl MockAnalyzer {
    fn new(id: &str, deps: Vec<AnalyzerDependency>) -> Self {
        Self {
            id: id.to_string(),
            deps,
        }
    }
}

impl AnalyzerPlugin for MockAnalyzer {
    fn id(&self) -> &str {
        &self.id
    }
    fn meta(&self) -> AnalyzerMeta {
        AnalyzerMeta {
            id: self.id.clone(),
            name: self.id.clone(),
            version: "1.0.0".into(),
        }
    }
    fn dependencies(&self) -> Vec<AnalyzerDependency> {
        self.deps.clone()
    }
    fn analyze(
        &self,
        _ctx: &AnalysisContext,
        _report: &AnalysisReport,
    ) -> Result<Box<dyn core::any::Any + Send + Sync>, AnalysisError> {
        Ok(Box::new(42i32))
    }
}

struct PanicAnalyzer;

impl AnalyzerPlugin for PanicAnalyzer {
    fn id(&self) -> &str {
        "panicker"
    }
    fn meta(&self) -> AnalyzerMeta {
        AnalyzerMeta {
            id: "panicker".into(),
            name: "Panic Analyzer".into(),
            version: "1.0.0".into(),
        }
    }
    fn analyze(
        &self,
        _ctx: &AnalysisContext,
        _report: &AnalysisReport,
    ) -> Result<Box<dyn core::any::Any + Send + Sync>, AnalysisError> {
        panic!("intentional panic for testing");
    }
}

// ── Success Criterion 1: Topological ordering ─────────────────

#[test]
fn sc1_topological_ordering() {
    let mut reg = PluginRegistry::new();
    reg.register(Box::new(MockAnalyzer::new("c", vec![])))
        .unwrap();
    reg.register(Box::new(MockAnalyzer::new(
        "b",
        vec![AnalyzerDependency::Required("c".into())],
    )))
    .unwrap();
    reg.register(Box::new(MockAnalyzer::new(
        "a",
        vec![AnalyzerDependency::Required("b".into())],
    )))
    .unwrap();

    let order = ExecutionPlanner::plan(&reg).unwrap();
    assert_eq!(order, vec!["c", "b", "a"]);
}

// ── Success Criterion 2: Circular dependency rejection ────────

#[test]
fn sc2_circular_dependency_rejected() {
    let mut reg = PluginRegistry::new();
    reg.register(Box::new(MockAnalyzer::new("x", vec![])))
        .unwrap();
    reg.register(Box::new(MockAnalyzer::new(
        "y",
        vec![AnalyzerDependency::Required("x".into())],
    )))
    .unwrap();

    // Remove x and re-register with dep on y -> cycle
    reg.unregister("x");
    let result = reg.register(Box::new(MockAnalyzer::new(
        "x",
        vec![AnalyzerDependency::Required("y".into())],
    )));

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        AnalysisError::CycleDetected(_)
    ));
}

// ── Success Criterion 3: Full pipeline with 4 builtins ────────

#[test]
fn sc3_full_pipeline_four_builtins() {
    let coord = make_coordinator();
    let (diff, engine) = make_diff_and_engine(
        common::diff_pairs::TYPO_BEFORE,
        common::diff_pairs::TYPO_AFTER,
    );
    let ctx = AnalysisContext::new(&diff, &engine);
    let report = coord.run(&ctx).unwrap();

    // All 4 analyzers should succeed
    assert_eq!(report.succeeded().len(), 4);
    assert_eq!(report.failed().len(), 0);
    assert_eq!(report.skipped().len(), 0);

    // Typed accessors work
    let semantic = report.builtin::<SemanticResult>("semantic");
    assert!(semantic.is_some(), "semantic result missing");

    let stylistic = report.builtin::<StylisticResult>("stylistic");
    assert!(stylistic.is_some(), "stylistic result missing");

    let readability = report.builtin::<ReadabilityResult>("readability");
    assert!(readability.is_some(), "readability result missing");

    let classifier = report.builtin::<EditClassifierResult>("edit_classifier");
    assert!(classifier.is_some(), "edit_classifier result missing");
}

// ── Success Criterion 4: Panic isolation ──────────────────────

#[test]
fn sc4_panic_isolation() {
    let mut registry = PluginRegistry::new();
    // Register panicker first (no deps)
    registry.register(Box::new(PanicAnalyzer)).unwrap();
    // Then register 4 builtins (no deps, so order doesn't matter)
    register_builtin_analyzers(&mut registry).unwrap();

    let coord = AnalysisCoordinator::new(registry);
    let (diff, engine) = make_diff_and_engine(
        common::diff_pairs::TYPO_BEFORE,
        common::diff_pairs::TYPO_AFTER,
    );
    let ctx = AnalysisContext::new(&diff, &engine);

    // This must NOT panic — coordinator catches it
    let report = coord.run(&ctx).unwrap();

    // Panicker failed with panic message
    assert_eq!(report.failed().len(), 1);
    let failed_meta = &report.failed()[0];
    assert_eq!(failed_meta.analyzer_id, "panicker");
    assert!(matches!(
        &failed_meta.status,
        AnalyzerStatus::Failed(msg) if msg.contains("panic:")
    ));

    // All 4 builtins still succeeded
    assert_eq!(report.succeeded().len(), 4);
    assert_eq!(report.result_count(), 4);
}

// ── Success Criterion 5: EditClassifier categorization ────────

#[test]
fn sc5_typo_classified_as_correction() {
    let coord = make_coordinator();
    let (diff, engine) = make_diff_and_engine(
        common::diff_pairs::TYPO_BEFORE,
        common::diff_pairs::TYPO_AFTER,
    );
    let ctx = AnalysisContext::new(&diff, &engine);
    let report = coord.run(&ctx).unwrap();

    let result = report
        .builtin::<EditClassifierResult>("edit_classifier")
        .expect("edit_classifier result missing");

    // Find the non-equal operation classification
    assert!(!result.operations.is_empty(), "no operations classified");

    // At least one operation should have Correction as top category
    let has_correction_top = result
        .operations
        .iter()
        .any(|op| op.categories.first().map(|(cat, _)| *cat) == Some(IntentCategory::Correction));
    assert!(
        has_correction_top,
        "typo fix should have Correction as top category, got: {:?}",
        result
            .operations
            .iter()
            .map(|o| &o.categories[0])
            .collect::<Vec<_>>()
    );
}

#[test]
fn sc5_insertion_classified_as_expansion() {
    let coord = make_coordinator();
    let (diff, engine) = make_diff_and_engine(
        common::diff_pairs::INSERT_BEFORE,
        common::diff_pairs::INSERT_AFTER,
    );
    let ctx = AnalysisContext::new(&diff, &engine);
    let report = coord.run(&ctx).unwrap();

    let result = report
        .builtin::<EditClassifierResult>("edit_classifier")
        .expect("edit_classifier result missing");

    assert!(!result.operations.is_empty(), "no operations classified");

    // At least one operation should have Expansion as top category
    let has_expansion_top = result
        .operations
        .iter()
        .any(|op| op.categories.first().map(|(cat, _)| *cat) == Some(IntentCategory::Expansion));
    assert!(
        has_expansion_top,
        "insertion should have Expansion as top category, got: {:?}",
        result
            .operations
            .iter()
            .map(|o| &o.categories[0])
            .collect::<Vec<_>>()
    );
}

// ── Additional integration tests ──────────────────────────────

#[test]
fn semantic_similarity_scores_valid_range() {
    let coord = make_coordinator();
    let (diff, engine) = make_diff_and_engine(
        common::diff_pairs::REWRITE_BEFORE,
        common::diff_pairs::REWRITE_AFTER,
    );
    let ctx = AnalysisContext::new(&diff, &engine);
    let report = coord.run(&ctx).unwrap();

    let semantic = report
        .builtin::<SemanticResult>("semantic")
        .expect("semantic result missing");

    assert!(
        (0.0..=1.0).contains(&semantic.overall_similarity),
        "overall_similarity {} out of [0, 1]",
        semantic.overall_similarity
    );

    for score in &semantic.operation_scores {
        assert!(
            (0.0..=1.0).contains(&score.similarity),
            "similarity {} out of [0, 1] for op {}",
            score.similarity,
            score.operation_index
        );
    }
}

#[test]
fn readability_scores_populated() {
    let coord = make_coordinator();
    let (diff, engine) = make_diff_and_engine(common::SIMPLE_PROSE, common::TECHNICAL_PROSE);
    let ctx = AnalysisContext::new(&diff, &engine);
    let report = coord.run(&ctx).unwrap();

    let readability = report
        .builtin::<ReadabilityResult>("readability")
        .expect("readability result missing");

    // Simple prose should have higher FRE than technical prose
    assert!(
        readability.source_scores.flesch_reading_ease
            > readability.target_scores.flesch_reading_ease,
        "simple prose should be more readable than technical prose: src={} tgt={}",
        readability.source_scores.flesch_reading_ease,
        readability.target_scores.flesch_reading_ease
    );

    // Delta should be negative (FRE decreased)
    assert!(
        readability.delta.flesch_reading_ease < 0.0,
        "delta FRE should be negative: {}",
        readability.delta.flesch_reading_ease
    );
}

#[test]
fn report_metadata_completeness() {
    let coord = make_coordinator();
    let (diff, engine) = make_diff_and_engine("hello world", "hello there");
    let ctx = AnalysisContext::new(&diff, &engine);
    let report = coord.run(&ctx).unwrap();

    assert_eq!(report.execution_metadata.len(), 4);

    // All should have non-negative duration and valid order indices
    let indices: Vec<usize> = report
        .execution_metadata
        .iter()
        .map(|m| m.order_index)
        .collect();
    assert_eq!(indices.len(), 4);
    for &idx in &indices {
        assert!(idx < 4, "order_index {} out of range", idx);
    }
}

#[test]
fn run_filtered_subset() {
    let coord = make_coordinator();
    let (diff, engine) = make_diff_and_engine("hello world", "hello there");
    let ctx = AnalysisContext::new(&diff, &engine);

    let report = coord
        .run_filtered(&ctx, &["semantic", "edit_classifier"])
        .unwrap();

    assert_eq!(report.result_count(), 2);
    assert!(report.has_result("semantic"));
    assert!(report.has_result("edit_classifier"));
    assert!(!report.has_result("stylistic"));
    assert!(!report.has_result("readability"));
}

#[test]
fn edit_classifier_always_returns_seven_categories() {
    let coord = make_coordinator();
    let (diff, engine) = make_diff_and_engine(
        common::diff_pairs::REWRITE_BEFORE,
        common::diff_pairs::REWRITE_AFTER,
    );
    let ctx = AnalysisContext::new(&diff, &engine);
    let report = coord.run(&ctx).unwrap();

    let result = report
        .builtin::<EditClassifierResult>("edit_classifier")
        .expect("edit_classifier result missing");

    for op in &result.operations {
        assert_eq!(
            op.categories.len(),
            7,
            "expected 7 categories, got {} for op {}",
            op.categories.len(),
            op.operation_index
        );

        // Sorted descending
        for window in op.categories.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "categories not sorted: {:?} < {:?}",
                window[0],
                window[1]
            );
        }

        // Sum ~1.0
        let sum: f64 = op.categories.iter().map(|(_, c)| c).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "category sum should be ~1.0, got {} for op {}",
            sum,
            op.operation_index
        );
    }
}
