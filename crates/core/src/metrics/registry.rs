//! Metric registry with DFS-based cycle detection on the static dependency graph.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, string::ToString, vec::Vec};

use hashbrown::HashMap;

use super::Metric;
use super::error::MetricError;

/// Registry of metric implementations, with dependency validation.
pub struct MetricRegistry {
    metrics: HashMap<String, Box<dyn Metric>>,
}

impl core::fmt::Debug for MetricRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetricRegistry")
            .field("count", &self.metrics.len())
            .field("ids", &self.metrics.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Node coloring for DFS cycle detection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    White,
    Gray,
    Black,
}

impl MetricRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Register a metric. Returns error if the ID is already taken.
    pub fn register(&mut self, metric: Box<dyn Metric>) -> Result<(), MetricError> {
        let id = metric.id().to_string();
        if self.metrics.contains_key(&id) {
            return Err(MetricError::AlreadyRegistered(id));
        }
        self.metrics.insert(id, metric);
        Ok(())
    }

    /// Register a metric, replacing any existing metric with the same ID.
    pub fn register_with_override(&mut self, metric: Box<dyn Metric>) {
        let id = metric.id().to_string();
        self.metrics.insert(id, metric);
    }

    /// Validate the static dependency graph: check all deps exist and detect cycles.
    pub fn validate(&self) -> Result<(), MetricError> {
        // Check all static dependencies exist
        for (id, metric) in &self.metrics {
            for dep in metric.static_dependencies() {
                if !self.metrics.contains_key(*dep) {
                    return Err(MetricError::DependencyNotFound(format!(
                        "{dep} (required by {id})"
                    )));
                }
            }
        }

        // DFS cycle detection with coloring
        let mut colors: HashMap<&str, Color> = HashMap::new();
        for id in self.metrics.keys() {
            colors.insert(id.as_str(), Color::White);
        }

        let mut path: Vec<&str> = Vec::new();
        for id in self.metrics.keys() {
            if colors[id.as_str()] == Color::White {
                self.dfs_visit(id.as_str(), &mut colors, &mut path)?;
            }
        }

        Ok(())
    }

    fn dfs_visit<'a>(
        &'a self,
        id: &'a str,
        colors: &mut HashMap<&'a str, Color>,
        path: &mut Vec<&'a str>,
    ) -> Result<(), MetricError> {
        colors.insert(id, Color::Gray);
        path.push(id);

        if let Some(metric) = self.metrics.get(id) {
            for dep in metric.static_dependencies() {
                match colors.get(*dep) {
                    Some(Color::Gray) => {
                        // Found a cycle - build cycle description
                        let cycle_start = path.iter().position(|&p| p == *dep).unwrap();
                        let mut cycle_path: Vec<&str> =
                            path[cycle_start..].iter().copied().collect();
                        cycle_path.push(dep);
                        return Err(MetricError::CycleDetected(cycle_path.join(" -> ")));
                    }
                    Some(Color::White) => {
                        self.dfs_visit(dep, colors, path)?;
                    }
                    _ => {} // Black: already fully explored
                }
            }
        }

        path.pop();
        colors.insert(id, Color::Black);
        Ok(())
    }

    /// Look up a metric by ID.
    pub fn get(&self, id: &str) -> Option<&dyn Metric> {
        self.metrics.get(id).map(|m| m.as_ref())
    }

    /// Iterate over all registered metric IDs.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.metrics.keys().map(|s| s.as_str())
    }

    /// Number of registered metrics.
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{DependencyKind, MetricInput, MetricValue};

    /// Test metric with configurable ID and dependencies.
    struct TestMetric {
        id: &'static str,
        deps: &'static [&'static str],
    }

    impl Metric for TestMetric {
        fn id(&self) -> &str {
            self.id
        }
        fn compute(
            &self,
            _input: &MetricInput<'_>,
            _deps: &HashMap<String, MetricValue>,
        ) -> Result<MetricValue, MetricError> {
            Ok(MetricValue::Integer(1))
        }
        fn static_dependencies(&self) -> &[&str] {
            self.deps
        }
        fn dependency_kind(&self) -> DependencyKind {
            DependencyKind::Static
        }
    }

    #[test]
    fn register_succeeds() {
        let mut reg = MetricRegistry::new();
        assert!(reg.is_empty());
        reg.register(Box::new(TestMetric { id: "a", deps: &[] }))
            .unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.get("a").is_some());
    }

    #[test]
    fn register_duplicate_errors() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(TestMetric { id: "a", deps: &[] }))
            .unwrap();
        let err = reg
            .register(Box::new(TestMetric { id: "a", deps: &[] }))
            .unwrap_err();
        assert!(matches!(err, MetricError::AlreadyRegistered(_)));
    }

    #[test]
    fn register_with_override_replaces() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(TestMetric { id: "a", deps: &[] }))
            .unwrap();
        reg.register_with_override(Box::new(TestMetric { id: "a", deps: &[] }));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn validate_detects_two_node_cycle() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(TestMetric {
            id: "a",
            deps: &["b"],
        }))
        .unwrap();
        reg.register(Box::new(TestMetric {
            id: "b",
            deps: &["a"],
        }))
        .unwrap();
        let err = reg.validate().unwrap_err();
        assert!(matches!(err, MetricError::CycleDetected(_)));
    }

    #[test]
    fn validate_detects_three_node_cycle() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(TestMetric {
            id: "a",
            deps: &["b"],
        }))
        .unwrap();
        reg.register(Box::new(TestMetric {
            id: "b",
            deps: &["c"],
        }))
        .unwrap();
        reg.register(Box::new(TestMetric {
            id: "c",
            deps: &["a"],
        }))
        .unwrap();
        let err = reg.validate().unwrap_err();
        assert!(matches!(err, MetricError::CycleDetected(_)));
    }

    #[test]
    fn validate_detects_self_referential() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(TestMetric {
            id: "a",
            deps: &["a"],
        }))
        .unwrap();
        let err = reg.validate().unwrap_err();
        assert!(matches!(err, MetricError::CycleDetected(_)));
    }

    #[test]
    fn validate_detects_missing_dependency() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(TestMetric {
            id: "a",
            deps: &["nonexistent"],
        }))
        .unwrap();
        let err = reg.validate().unwrap_err();
        assert!(matches!(err, MetricError::DependencyNotFound(_)));
    }

    #[test]
    fn validate_passes_for_valid_graph() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(TestMetric { id: "a", deps: &[] }))
            .unwrap();
        reg.register(Box::new(TestMetric {
            id: "b",
            deps: &["a"],
        }))
        .unwrap();
        reg.register(Box::new(TestMetric {
            id: "c",
            deps: &["a", "b"],
        }))
        .unwrap();
        reg.validate().unwrap();
    }

    #[test]
    fn validate_passes_diamond_no_cycle() {
        let mut reg = MetricRegistry::new();
        // Diamond: a -> b, a -> c, b -> d, c -> d
        reg.register(Box::new(TestMetric { id: "d", deps: &[] }))
            .unwrap();
        reg.register(Box::new(TestMetric {
            id: "b",
            deps: &["d"],
        }))
        .unwrap();
        reg.register(Box::new(TestMetric {
            id: "c",
            deps: &["d"],
        }))
        .unwrap();
        reg.register(Box::new(TestMetric {
            id: "a",
            deps: &["b", "c"],
        }))
        .unwrap();
        reg.validate().unwrap();
    }

    #[test]
    fn validate_empty_registry() {
        let reg = MetricRegistry::new();
        reg.validate().unwrap();
    }

    #[test]
    fn ids_iteration() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(TestMetric { id: "x", deps: &[] }))
            .unwrap();
        reg.register(Box::new(TestMetric { id: "y", deps: &[] }))
            .unwrap();
        let mut ids: Vec<&str> = reg.ids().collect();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn get_returns_none_for_missing() {
        let reg = MetricRegistry::new();
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn debug_format() {
        let reg = MetricRegistry::new();
        let dbg = format!("{:?}", reg);
        assert!(dbg.contains("MetricRegistry"));
    }
}
