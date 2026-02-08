//! AnalysisCoordinator: synchronous analyzer execution with panic isolation.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, string::ToString, vec::Vec};

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use hashbrown::HashSet;

use super::report::{AnalyzerExecutionMeta, AnalyzerStatus};
use super::{
    AnalysisContext, AnalysisError, AnalysisReport, AnalyzerDependency, ExecutionPlanner,
    PluginRegistry,
};

/// Orchestrates analyzer execution in topological order with panic isolation.
///
/// Each analyzer is run inside `catch_unwind` so a panic in one analyzer
/// does not crash the entire pipeline. Dependents of failed or panicked
/// analyzers are automatically skipped.
pub struct AnalysisCoordinator {
    registry: PluginRegistry,
}

impl AnalysisCoordinator {
    /// Create a coordinator with the given registry.
    pub fn new(registry: PluginRegistry) -> Self {
        Self { registry }
    }

    /// Access the underlying registry.
    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    /// Run all registered analyzers in topological order.
    pub fn run(&self, context: &AnalysisContext) -> Result<AnalysisReport, AnalysisError> {
        let order = ExecutionPlanner::plan(&self.registry)?;
        self.execute_in_order(context, &order)
    }

    /// Run only the specified analyzers and their transitive Required dependencies.
    pub fn run_filtered(
        &self,
        context: &AnalysisContext,
        analyzer_ids: &[&str],
    ) -> Result<AnalysisReport, AnalysisError> {
        let order = ExecutionPlanner::plan_filtered(&self.registry, analyzer_ids)?;
        self.execute_in_order(context, &order)
    }

    fn execute_in_order(
        &self,
        context: &AnalysisContext,
        order: &[String],
    ) -> Result<AnalysisReport, AnalysisError> {
        let mut report = AnalysisReport::new();
        let mut failed_set: HashSet<String> = HashSet::new();

        for (order_index, analyzer_id) in order.iter().enumerate() {
            let analyzer = match self.registry.get(analyzer_id) {
                Some(a) => a,
                None => {
                    report.execution_metadata.push(AnalyzerExecutionMeta {
                        analyzer_id: analyzer_id.clone(),
                        status: AnalyzerStatus::Failed(format!(
                            "analyzer '{}' not found in registry",
                            analyzer_id
                        )),
                        duration: core::time::Duration::ZERO,
                        order_index,
                    });
                    failed_set.insert(analyzer_id.clone());
                    continue;
                }
            };

            // Check if any Required dependency has failed
            let mut skip_reason = None;
            for dep in analyzer.dependencies() {
                if let AnalyzerDependency::Required(ref dep_id) = dep {
                    if failed_set.contains(dep_id) {
                        skip_reason = Some(dep_id.clone());
                        break;
                    }
                }
            }

            if let Some(failed_dep) = skip_reason {
                report.execution_metadata.push(AnalyzerExecutionMeta {
                    analyzer_id: analyzer_id.clone(),
                    status: AnalyzerStatus::Skipped(format!("dependency '{}' failed", failed_dep)),
                    duration: core::time::Duration::ZERO,
                    order_index,
                });
                failed_set.insert(analyzer_id.clone());
                continue;
            }

            // Execute with panic isolation
            let start = Instant::now();
            let result = catch_unwind(AssertUnwindSafe(|| analyzer.analyze(context, &report)));
            let duration = start.elapsed();

            match result {
                Ok(Ok(value)) => {
                    report.insert(analyzer_id.clone(), value);
                    report.execution_metadata.push(AnalyzerExecutionMeta {
                        analyzer_id: analyzer_id.clone(),
                        status: AnalyzerStatus::Success,
                        duration,
                        order_index,
                    });
                }
                Ok(Err(err)) => {
                    report.execution_metadata.push(AnalyzerExecutionMeta {
                        analyzer_id: analyzer_id.clone(),
                        status: AnalyzerStatus::Failed(format!("{}", err)),
                        duration,
                        order_index,
                    });
                    failed_set.insert(analyzer_id.clone());
                }
                Err(panic_payload) => {
                    let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                        format!("panic: {}", s)
                    } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        format!("panic: {}", s)
                    } else {
                        "panic: <unknown>".to_string()
                    };
                    report.execution_metadata.push(AnalyzerExecutionMeta {
                        analyzer_id: analyzer_id.clone(),
                        status: AnalyzerStatus::Failed(msg),
                        duration,
                        order_index,
                    });
                    failed_set.insert(analyzer_id.clone());
                }
            }
        }

        Ok(report)
    }
}

impl core::fmt::Debug for AnalysisCoordinator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AnalysisCoordinator")
            .field("registry", &self.registry)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalyzerMeta, AnalyzerPlugin};
    use crate::diff::edit_operation::EditOperation;
    use crate::diff::result::{DiffMetadata, DiffResult, DiffStatistics};
    use crate::metrics::{MetricRegistry, MetricsEngine};
    use crate::process::ProcessedText;
    use crate::text_store::TextStoreBuilder;
    use std::sync::Arc;

    // ── Test helpers ──────────────────────────────────────────

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
            Ok(Box::new(42u32))
        }
    }

    struct PanicAnalyzer {
        id: String,
    }

    impl AnalyzerPlugin for PanicAnalyzer {
        fn id(&self) -> &str {
            &self.id
        }
        fn meta(&self) -> AnalyzerMeta {
            AnalyzerMeta {
                id: self.id.clone(),
                name: "panic".into(),
                version: "1.0.0".into(),
            }
        }
        fn analyze(
            &self,
            _ctx: &AnalysisContext,
            _report: &AnalysisReport,
        ) -> Result<Box<dyn core::any::Any + Send + Sync>, AnalysisError> {
            panic!("boom");
        }
    }

    struct FailAnalyzer {
        id: String,
    }

    impl AnalyzerPlugin for FailAnalyzer {
        fn id(&self) -> &str {
            &self.id
        }
        fn meta(&self) -> AnalyzerMeta {
            AnalyzerMeta {
                id: self.id.clone(),
                name: "fail".into(),
                version: "1.0.0".into(),
            }
        }
        fn analyze(
            &self,
            _ctx: &AnalysisContext,
            _report: &AnalysisReport,
        ) -> Result<Box<dyn core::any::Any + Send + Sync>, AnalysisError> {
            Err(AnalysisError::AnalyzerFailed {
                analyzer: self.id.clone(),
                reason: "intentional failure".into(),
            })
        }
    }

    fn make_processed_text(text: &str) -> ProcessedText {
        let store = TextStoreBuilder::new().build();
        ProcessedText {
            original: text.to_string(),
            layers: vec![],
            normalized: text.to_string(),
            tokens: vec![],
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        }
    }

    fn make_test_context() -> (DiffResult, MetricsEngine) {
        let source = make_processed_text("hello world");
        let target = make_processed_text("hello there");
        let ops = vec![
            EditOperation::equal(0, 1, 0, 1),
            EditOperation::replace(1, 2, 1, 2),
        ];
        let stats = DiffStatistics::from_operations(&ops, 2, 2);
        let diff = DiffResult {
            operations: ops,
            statistics: stats,
            metadata: DiffMetadata {
                algorithm_name: "test".into(),
                is_approximate: false,
                threshold_used: None,
            },
            source: Arc::new(source),
            target: Arc::new(target),
        };
        let engine = MetricsEngine::new(MetricRegistry::new());
        (diff, engine)
    }

    // ── Tests ─────────────────────────────────────────────────

    #[test]
    fn run_single_analyzer_success() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        let coord = AnalysisCoordinator::new(reg);
        let (diff, engine) = make_test_context();
        let ctx = AnalysisContext::new(&diff, &engine);
        let report = coord.run(&ctx).unwrap();
        assert_eq!(report.result_count(), 1);
        assert_eq!(report.succeeded().len(), 1);
    }

    #[test]
    fn panic_analyzer_caught() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(PanicAnalyzer {
            id: "panicker".into(),
        }))
        .unwrap();
        let coord = AnalysisCoordinator::new(reg);
        let (diff, engine) = make_test_context();
        let ctx = AnalysisContext::new(&diff, &engine);
        let report = coord.run(&ctx).unwrap();
        assert_eq!(report.result_count(), 0);
        assert_eq!(report.failed().len(), 1);
        let meta = &report.failed()[0];
        assert!(matches!(&meta.status, AnalyzerStatus::Failed(msg) if msg.contains("panic: boom")));
    }

    #[test]
    fn fail_analyzer_recorded() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(FailAnalyzer {
            id: "failer".into(),
        }))
        .unwrap();
        let coord = AnalysisCoordinator::new(reg);
        let (diff, engine) = make_test_context();
        let ctx = AnalysisContext::new(&diff, &engine);
        let report = coord.run(&ctx).unwrap();
        assert_eq!(report.result_count(), 0);
        assert_eq!(report.failed().len(), 1);
    }

    #[test]
    fn dependent_skipped_on_failure() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        reg.register(Box::new(FailAnalyzer { id: "b".into() }))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new(
            "c",
            vec![AnalyzerDependency::Required("b".into())],
        )))
        .unwrap();
        let coord = AnalysisCoordinator::new(reg);
        let (diff, engine) = make_test_context();
        let ctx = AnalysisContext::new(&diff, &engine);
        let report = coord.run(&ctx).unwrap();
        assert_eq!(report.succeeded().len(), 1); // a
        assert_eq!(report.failed().len(), 1); // b
        assert_eq!(report.skipped().len(), 1); // c
    }

    #[test]
    fn dependent_skipped_on_panic() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        reg.register(Box::new(PanicAnalyzer { id: "b".into() }))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new(
            "c",
            vec![AnalyzerDependency::Required("b".into())],
        )))
        .unwrap();
        let coord = AnalysisCoordinator::new(reg);
        let (diff, engine) = make_test_context();
        let ctx = AnalysisContext::new(&diff, &engine);
        let report = coord.run(&ctx).unwrap();
        assert_eq!(report.succeeded().len(), 1); // a
        assert_eq!(report.failed().len(), 1); // b panicked
        assert_eq!(report.skipped().len(), 1); // c skipped
    }

    #[test]
    fn execution_order_matches_plan() {
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
        let coord = AnalysisCoordinator::new(reg);
        let (diff, engine) = make_test_context();
        let ctx = AnalysisContext::new(&diff, &engine);
        let report = coord.run(&ctx).unwrap();
        let c_meta = report
            .execution_metadata
            .iter()
            .find(|m| m.analyzer_id == "c")
            .unwrap();
        let b_meta = report
            .execution_metadata
            .iter()
            .find(|m| m.analyzer_id == "b")
            .unwrap();
        let a_meta = report
            .execution_metadata
            .iter()
            .find(|m| m.analyzer_id == "a")
            .unwrap();
        assert_eq!(c_meta.order_index, 0);
        assert_eq!(b_meta.order_index, 1);
        assert_eq!(a_meta.order_index, 2);
    }

    #[test]
    fn run_filtered_subset() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new("b", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new("c", vec![])))
            .unwrap();
        let coord = AnalysisCoordinator::new(reg);
        let (diff, engine) = make_test_context();
        let ctx = AnalysisContext::new(&diff, &engine);
        let report = coord.run_filtered(&ctx, &["a", "c"]).unwrap();
        assert_eq!(report.result_count(), 2);
        assert!(report.has_result("a"));
        assert!(!report.has_result("b"));
        assert!(report.has_result("c"));
    }

    #[test]
    fn order_index_increments() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("x", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new("y", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new("z", vec![])))
            .unwrap();
        let coord = AnalysisCoordinator::new(reg);
        let (diff, engine) = make_test_context();
        let ctx = AnalysisContext::new(&diff, &engine);
        let report = coord.run(&ctx).unwrap();
        let indices: Vec<usize> = report
            .execution_metadata
            .iter()
            .map(|m| m.order_index)
            .collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }
}
