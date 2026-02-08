//! ExecutionPlanner: Kahn's algorithm topological sort for analyzer DAG.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec, vec::Vec};

use hashbrown::{HashMap, HashSet};

use super::registry::PluginRegistry;
use super::{AnalysisError, AnalyzerDependency};

/// Computes execution order for analyzers based on their dependency graph.
///
/// Uses Kahn's algorithm with alphabetical tie-breaking for deterministic output.
pub struct ExecutionPlanner;

impl ExecutionPlanner {
    /// Compute a full topological execution order for all registered analyzers.
    ///
    /// Uses Kahn's algorithm. Ties are broken alphabetically for determinism.
    pub fn plan(registry: &PluginRegistry) -> Result<Vec<String>, AnalysisError> {
        let ids = registry.analyzer_ids();
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // Build in-degree map and forward edges (dependency -> dependent)
        let mut in_degree: HashMap<&str, usize> = ids.iter().map(|&id| (id, 0usize)).collect();
        let mut forward_edges: HashMap<&str, Vec<&str>> =
            ids.iter().map(|&id| (id, Vec::new())).collect();

        for &id in &ids {
            let analyzer = registry.get(id).unwrap();
            for dep in analyzer.dependencies() {
                let dep_id = dep.id();
                // Only include edges for deps that exist in registry
                if registry.contains(dep_id) {
                    forward_edges.get_mut(dep_id).unwrap().push(id);
                    *in_degree.get_mut(id).unwrap() += 1;
                }
            }
        }

        // Initialize queue with zero-degree nodes, sorted alphabetically
        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();
        queue.sort();

        let mut result: Vec<String> = Vec::with_capacity(ids.len());

        while let Some(node) = queue.first().copied() {
            queue.remove(0);
            result.push(node.to_string());

            let mut successors = forward_edges.get(node).cloned().unwrap_or_default();
            successors.sort(); // deterministic ordering
            for &succ in &successors {
                let deg = in_degree.get_mut(succ).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    // Insert in sorted position
                    let pos = queue.iter().position(|&q| q > succ).unwrap_or(queue.len());
                    queue.insert(pos, succ);
                }
            }
        }

        // Cycle check
        if result.len() < ids.len() {
            let mut stuck: Vec<&str> = in_degree
                .iter()
                .filter(|(_, deg)| **deg > 0)
                .map(|(id, _)| *id)
                .collect();
            stuck.sort();
            return Err(AnalysisError::CycleDetected(format!(
                "cycle among: {}",
                stuck.join(", ")
            )));
        }

        Ok(result)
    }

    /// Compute a filtered execution order including only the requested analyzers
    /// and their transitive Required dependencies.
    pub fn plan_filtered(
        registry: &PluginRegistry,
        analyzer_ids: &[&str],
    ) -> Result<Vec<String>, AnalysisError> {
        // Collect transitive closure of Required dependencies
        let mut needed: HashSet<String> = HashSet::new();
        let mut stack: Vec<String> = analyzer_ids.iter().map(|s| s.to_string()).collect();

        while let Some(id) = stack.pop() {
            if needed.contains(&id) {
                continue;
            }
            if let Some(analyzer) = registry.get(&id) {
                for dep in analyzer.dependencies() {
                    if let AnalyzerDependency::Required(ref dep_id) = dep {
                        if registry.contains(dep_id) && !needed.contains(dep_id) {
                            stack.push(dep_id.clone());
                        }
                    }
                }
            }
            needed.insert(id);
        }

        // Get full topological order, filter to needed
        let full_order = Self::plan(registry)?;
        Ok(full_order
            .into_iter()
            .filter(|id| needed.contains(id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{
        AnalysisContext, AnalysisError, AnalysisReport, AnalyzerDependency, AnalyzerMeta,
        AnalyzerPlugin,
    };

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
            Ok(Box::new(()))
        }
    }

    #[test]
    fn plan_independent_analyzers() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("c", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new("b", vec![])))
            .unwrap();
        let order = ExecutionPlanner::plan(&reg).unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn plan_linear_chain() {
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

    #[test]
    fn plan_diamond() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("d", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new(
            "b",
            vec![AnalyzerDependency::Required("d".into())],
        )))
        .unwrap();
        reg.register(Box::new(MockAnalyzer::new(
            "c",
            vec![AnalyzerDependency::Required("d".into())],
        )))
        .unwrap();
        reg.register(Box::new(MockAnalyzer::new(
            "a",
            vec![
                AnalyzerDependency::Required("b".into()),
                AnalyzerDependency::Required("c".into()),
            ],
        )))
        .unwrap();
        let order = ExecutionPlanner::plan(&reg).unwrap();
        assert_eq!(order, vec!["d", "b", "c", "a"]);
    }

    #[test]
    fn plan_empty_registry() {
        let reg = PluginRegistry::new();
        let order = ExecutionPlanner::plan(&reg).unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn plan_filtered_subset() {
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
        reg.register(Box::new(MockAnalyzer::new("x", vec![])))
            .unwrap();
        let order = ExecutionPlanner::plan_filtered(&reg, &["a"]).unwrap();
        assert_eq!(order, vec!["c", "b", "a"]);
    }

    #[test]
    fn plan_filtered_no_deps() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("x", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new("y", vec![])))
            .unwrap();
        let order = ExecutionPlanner::plan_filtered(&reg, &["x"]).unwrap();
        assert_eq!(order, vec!["x"]);
    }
}
