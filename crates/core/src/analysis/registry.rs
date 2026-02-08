//! PluginRegistry: analyzer registration with incremental cycle detection.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, string::ToString, vec, vec::Vec};

use hashbrown::HashMap;

use super::{AnalysisError, AnalyzerDependency, AnalyzerPlugin};

/// Registry of analyzer plugins with validation on every `register()` call.
///
/// Validates: duplicate IDs, Required dependency existence, and DFS cycle detection.
/// Optional dependencies that don't exist are silently accepted.
pub struct PluginRegistry {
    analyzers: HashMap<String, Box<dyn AnalyzerPlugin>>,
}

impl core::fmt::Debug for PluginRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PluginRegistry")
            .field("count", &self.analyzers.len())
            .field("ids", &self.analyzers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Node coloring for DFS cycle detection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    White,
    Gray,
    Black,
}

impl PluginRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            analyzers: HashMap::new(),
        }
    }

    /// Register an analyzer with full validation.
    ///
    /// Checks: duplicate ID, Required dependency existence, cycle detection.
    /// Optional missing dependencies are silently accepted.
    pub fn register(&mut self, analyzer: Box<dyn AnalyzerPlugin>) -> Result<(), AnalysisError> {
        let id = analyzer.id().to_string();

        // 1. Duplicate check
        if self.analyzers.contains_key(&id) {
            return Err(AnalysisError::AlreadyRegistered(id));
        }

        // 2. Required dependency existence check
        for dep in analyzer.dependencies() {
            if let AnalyzerDependency::Required(ref dep_id) = dep {
                if !self.analyzers.contains_key(dep_id) {
                    return Err(AnalysisError::DependencyNotFound {
                        analyzer: id,
                        dependency: dep_id.clone(),
                    });
                }
            }
            // Optional missing: silently accepted
        }

        // 3. Temporarily insert, then cycle-check
        self.analyzers.insert(id.clone(), analyzer);

        if let Some(cycle_path) = self.has_cycle() {
            // Roll back insertion
            self.analyzers.remove(&id);
            return Err(AnalysisError::CycleDetected(cycle_path));
        }

        Ok(())
    }

    /// Remove an analyzer from the registry, returning it if present.
    pub fn unregister(&mut self, id: &str) -> Option<Box<dyn AnalyzerPlugin>> {
        self.analyzers.remove(id)
    }

    /// Look up an analyzer by ID.
    pub fn get(&self, id: &str) -> Option<&dyn AnalyzerPlugin> {
        self.analyzers.get(id).map(|b| b.as_ref())
    }

    /// Check if an analyzer is registered.
    pub fn contains(&self, id: &str) -> bool {
        self.analyzers.contains_key(id)
    }

    /// All registered analyzer IDs.
    pub fn analyzer_ids(&self) -> Vec<&str> {
        self.analyzers.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered analyzers.
    pub fn len(&self) -> usize {
        self.analyzers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.analyzers.is_empty()
    }

    /// DFS 3-color cycle detection on the full dependency graph.
    fn has_cycle(&self) -> Option<String> {
        let ids: Vec<String> = self.analyzers.keys().cloned().collect();
        let mut colors: HashMap<&str, Color> =
            ids.iter().map(|k| (k.as_str(), Color::White)).collect();

        for id in &ids {
            if colors.get(id.as_str()) == Some(&Color::White) {
                let mut path = Vec::new();
                if let Some(cycle) = self.dfs_visit(id.as_str(), &mut colors, &mut path) {
                    return Some(cycle);
                }
            }
        }
        None
    }

    fn dfs_visit<'a>(
        &'a self,
        node: &'a str,
        colors: &mut HashMap<&'a str, Color>,
        path: &mut Vec<&'a str>,
    ) -> Option<String> {
        colors.insert(node, Color::Gray);
        path.push(node);

        if let Some(analyzer) = self.analyzers.get(node) {
            // Collect dependency IDs to owned strings to avoid borrowing temporaries
            let dep_ids: Vec<String> = analyzer
                .dependencies()
                .iter()
                .map(|dep| dep.id().to_string())
                .collect();

            for dep_id_owned in &dep_ids {
                let dep_id = dep_id_owned.as_str();
                // Only follow edges to nodes that exist in registry
                if !self.analyzers.contains_key(dep_id) {
                    continue;
                }
                // Look up color via the registry key (which has 'a lifetime)
                let dep_key = self
                    .analyzers
                    .get_key_value(dep_id)
                    .map(|(k, _)| k.as_str());
                let dep_key = match dep_key {
                    Some(k) => k,
                    None => continue,
                };
                match colors.get(dep_key) {
                    Some(Color::Gray) => {
                        let cycle_start = path.iter().position(|&n| n == dep_key).unwrap();
                        let cycle: Vec<&str> = path[cycle_start..].to_vec();
                        return Some(format!("{} -> {}", cycle.join(" -> "), dep_key));
                    }
                    Some(Color::White) | None => {
                        if let Some(cycle) = self.dfs_visit(dep_key, colors, path) {
                            return Some(cycle);
                        }
                    }
                    Some(Color::Black) => {}
                }
            }
        }

        path.pop();
        colors.insert(node, Color::Black);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisContext, AnalysisReport, AnalyzerMeta};

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
    fn register_single() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn register_duplicate_id() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        let err = reg
            .register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap_err();
        assert!(matches!(err, AnalysisError::AlreadyRegistered(_)));
    }

    #[test]
    fn register_with_satisfied_dependency() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("b", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new(
            "a",
            vec![AnalyzerDependency::Required("b".into())],
        )))
        .unwrap();
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn register_missing_required_dependency() {
        let mut reg = PluginRegistry::new();
        let err = reg
            .register(Box::new(MockAnalyzer::new(
                "a",
                vec![AnalyzerDependency::Required("b".into())],
            )))
            .unwrap_err();
        assert!(matches!(err, AnalysisError::DependencyNotFound { .. }));
    }

    #[test]
    fn register_circular_dependency() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("b", vec![])))
            .unwrap();
        reg.register(Box::new(MockAnalyzer::new(
            "a",
            vec![AnalyzerDependency::Required("b".into())],
        )))
        .unwrap();
        // Now re-register b with dep on a to create cycle
        reg.unregister("b");
        let err = reg
            .register(Box::new(MockAnalyzer::new(
                "b",
                vec![AnalyzerDependency::Required("a".into())],
            )))
            .unwrap_err();
        assert!(matches!(err, AnalysisError::CycleDetected(_)));
    }

    #[test]
    fn register_three_node_cycle() {
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
        reg.unregister("c");
        let err = reg
            .register(Box::new(MockAnalyzer::new(
                "c",
                vec![AnalyzerDependency::Required("a".into())],
            )))
            .unwrap_err();
        assert!(matches!(err, AnalysisError::CycleDetected(_)));
    }

    #[test]
    fn register_optional_missing_ok() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new(
            "a",
            vec![AnalyzerDependency::Optional("nonexistent".into())],
        )))
        .unwrap();
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn unregister_returns_analyzer() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        let removed = reg.unregister("a");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn contains_and_get() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MockAnalyzer::new("a", vec![])))
            .unwrap();
        assert!(reg.contains("a"));
        assert!(!reg.contains("z"));
        assert_eq!(reg.get("a").unwrap().id(), "a");
        assert!(reg.get("z").is_none());
    }
}
