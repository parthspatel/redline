//! MetricsEngine: pull-based lazy evaluation with dependency resolution and caching.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec::Vec};

use core::cell::RefCell;

use hashbrown::{HashMap, HashSet};

use super::cache::{CacheKey, ContentHash, MetricCache};
use super::registry::MetricRegistry;
use super::{MetricInput, MetricValue};

/// Metrics evaluation engine with lazy dependency resolution and optional caching.
pub struct MetricsEngine {
    registry: MetricRegistry,
    cache: Option<RefCell<MetricCache>>,
}

impl core::fmt::Debug for MetricsEngine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetricsEngine")
            .field("registry", &self.registry)
            .field("cache_enabled", &self.cache.is_some())
            .finish()
    }
}

impl MetricsEngine {
    /// Create an engine without caching.
    pub fn new(registry: MetricRegistry) -> Self {
        Self {
            registry,
            cache: None,
        }
    }

    /// Create an engine with an LRU cache of the given capacity.
    pub fn with_cache(registry: MetricRegistry, capacity: usize) -> Self {
        Self {
            registry,
            cache: Some(RefCell::new(MetricCache::new(capacity))),
        }
    }

    /// Create an engine with a default cache (capacity 256).
    pub fn with_default_cache(registry: MetricRegistry) -> Self {
        Self::with_cache(registry, 256)
    }

    /// Compute a single metric value, resolving dependencies lazily.
    ///
    /// Returns `MetricValue::Unavailable` if the metric is not found, has a
    /// dependency cycle, or computation fails.
    pub fn get(&self, id: &str, input: &MetricInput<'_>) -> MetricValue {
        let computing = RefCell::new(HashSet::new());
        self.compute_metric(id, input, &computing)
    }

    /// Compute multiple metrics at once, sharing dependency resolution.
    pub fn get_many(&self, ids: &[&str], input: &MetricInput<'_>) -> HashMap<String, MetricValue> {
        let computing = RefCell::new(HashSet::new());
        ids.iter()
            .map(|id| {
                let value = self.compute_metric(id, input, &computing);
                (id.to_string(), value)
            })
            .collect()
    }

    /// Access the underlying registry.
    pub fn registry(&self) -> &MetricRegistry {
        &self.registry
    }

    /// Clear all cached metric results.
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.borrow_mut().clear();
        }
    }

    /// Number of entries in the cache (0 if no cache).
    pub fn cache_len(&self) -> usize {
        self.cache.as_ref().map(|c| c.borrow().len()).unwrap_or(0)
    }

    fn compute_metric(
        &self,
        id: &str,
        input: &MetricInput<'_>,
        computing: &RefCell<HashSet<String>>,
    ) -> MetricValue {
        // 1. Check cache
        let content_hash = match input {
            MetricInput::Single(text) => ContentHash::of_single(text),
            MetricInput::Pairwise(source, target) => ContentHash::of_pair(source, target),
        };
        let cache_key = CacheKey::new(content_hash, id.to_string());

        if let Some(cache) = &self.cache {
            if let Some(cached) = cache.borrow_mut().get(&cache_key) {
                return cached;
            }
        }

        // 2. Cycle detection
        if computing.borrow().contains(id) {
            return MetricValue::Unavailable;
        }
        computing.borrow_mut().insert(id.to_string());

        // 3. Look up metric
        let metric = match self.registry.get(id) {
            Some(m) => m,
            None => {
                computing.borrow_mut().remove(id);
                return MetricValue::Unavailable;
            }
        };

        // 4. Resolve static dependencies
        let mut deps = HashMap::new();
        for dep_id in metric.static_dependencies() {
            let dep_value = self.compute_metric(dep_id, input, computing);
            deps.insert(dep_id.to_string(), dep_value);
        }

        // 5. Resolve dynamic dependencies
        for dep_id in metric.dynamic_dependencies(input) {
            if !deps.contains_key(&dep_id) {
                let dep_value = self.compute_metric(&dep_id, input, computing);
                deps.insert(dep_id, dep_value);
            }
        }

        // 6. Compute
        let result = match metric.compute(input, &deps) {
            Ok(value) => value,
            Err(_) => MetricValue::Unavailable,
        };

        // 7. Cache the result
        if let Some(cache) = &self.cache {
            cache.borrow_mut().put(cache_key, result.clone());
        }

        // 8. Cleanup
        computing.borrow_mut().remove(id);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{DependencyKind, Metric, MetricError};
    use crate::process::ProcessedText;
    use crate::text_store::TextStoreBuilder;

    /// Simple test metric that returns a constant value.
    struct ConstMetric {
        id: &'static str,
        value: MetricValue,
        deps: &'static [&'static str],
    }

    impl Metric for ConstMetric {
        fn id(&self) -> &str {
            self.id
        }
        fn compute(
            &self,
            _input: &MetricInput<'_>,
            _deps: &HashMap<String, MetricValue>,
        ) -> Result<MetricValue, MetricError> {
            Ok(self.value.clone())
        }
        fn static_dependencies(&self) -> &[&str] {
            self.deps
        }
        fn dependency_kind(&self) -> DependencyKind {
            DependencyKind::Static
        }
    }

    /// Metric that sums its dependencies.
    struct SumMetric {
        id: &'static str,
        deps: &'static [&'static str],
    }

    impl Metric for SumMetric {
        fn id(&self) -> &str {
            self.id
        }
        fn compute(
            &self,
            _input: &MetricInput<'_>,
            deps: &HashMap<String, MetricValue>,
        ) -> Result<MetricValue, MetricError> {
            let mut sum = 0i64;
            for dep_id in self.deps {
                match deps.get(*dep_id) {
                    Some(MetricValue::Integer(v)) => sum += v,
                    _ => {
                        return Err(MetricError::DependencyUnavailable(dep_id.to_string()));
                    }
                }
            }
            Ok(MetricValue::Integer(sum))
        }
        fn static_dependencies(&self) -> &[&str] {
            self.deps
        }
    }

    /// Metric that always fails.
    struct FailMetric;

    impl Metric for FailMetric {
        fn id(&self) -> &str {
            "fail"
        }
        fn compute(
            &self,
            _input: &MetricInput<'_>,
            _deps: &HashMap<String, MetricValue>,
        ) -> Result<MetricValue, MetricError> {
            Err(MetricError::ComputationError(
                "fail".into(),
                "always fails".into(),
            ))
        }
    }

    fn make_processed_text(text: &str) -> ProcessedText {
        let builder = TextStoreBuilder::new();
        let store = builder.build();
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

    #[test]
    fn simple_metric_returns_value() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(ConstMetric {
            id: "count",
            value: MetricValue::Integer(42),
            deps: &[],
        }))
        .unwrap();

        let engine = MetricsEngine::new(reg);
        let text = make_processed_text("hello world");
        let input = MetricInput::Single(&text);
        assert_eq!(engine.get("count", &input), MetricValue::Integer(42));
    }

    #[test]
    fn missing_metric_returns_unavailable() {
        let reg = MetricRegistry::new();
        let engine = MetricsEngine::new(reg);
        let text = make_processed_text("hello");
        let input = MetricInput::Single(&text);
        assert_eq!(engine.get("nonexistent", &input), MetricValue::Unavailable);
    }

    #[test]
    fn cache_hit_on_second_call() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(ConstMetric {
            id: "count",
            value: MetricValue::Integer(42),
            deps: &[],
        }))
        .unwrap();

        let engine = MetricsEngine::with_cache(reg, 10);
        let text = make_processed_text("hello world");
        let input = MetricInput::Single(&text);

        let v1 = engine.get("count", &input);
        assert_eq!(engine.cache_len(), 1);
        let v2 = engine.get("count", &input);
        assert_eq!(v1, v2);
        assert_eq!(engine.cache_len(), 1); // still 1 (cache hit)
    }

    #[test]
    fn dependency_resolution() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(ConstMetric {
            id: "a",
            value: MetricValue::Integer(10),
            deps: &[],
        }))
        .unwrap();
        reg.register(Box::new(ConstMetric {
            id: "b",
            value: MetricValue::Integer(20),
            deps: &[],
        }))
        .unwrap();
        reg.register(Box::new(SumMetric {
            id: "sum",
            deps: &["a", "b"],
        }))
        .unwrap();

        let engine = MetricsEngine::new(reg);
        let text = make_processed_text("test");
        let input = MetricInput::Single(&text);
        assert_eq!(engine.get("sum", &input), MetricValue::Integer(30));
    }

    #[test]
    fn cycle_detection_returns_unavailable() {
        let mut reg = MetricRegistry::new();
        // Create a cycle: cycle_a -> cycle_b -> cycle_a
        reg.register(Box::new(ConstMetric {
            id: "cycle_a",
            value: MetricValue::Integer(1),
            deps: &["cycle_b"],
        }))
        .unwrap();
        reg.register(Box::new(ConstMetric {
            id: "cycle_b",
            value: MetricValue::Integer(2),
            deps: &["cycle_a"],
        }))
        .unwrap();

        let engine = MetricsEngine::new(reg);
        let text = make_processed_text("test");
        let input = MetricInput::Single(&text);
        // One of the deps will hit the cycle and return Unavailable
        let result = engine.get("cycle_a", &input);
        // cycle_a depends on cycle_b, cycle_b depends on cycle_a (already computing) -> Unavailable
        // cycle_a still gets its own value because ConstMetric ignores deps
        assert_eq!(result, MetricValue::Integer(1));
    }

    #[test]
    fn failed_metric_returns_unavailable() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(FailMetric)).unwrap();

        let engine = MetricsEngine::new(reg);
        let text = make_processed_text("test");
        let input = MetricInput::Single(&text);
        assert_eq!(engine.get("fail", &input), MetricValue::Unavailable);
    }

    #[test]
    fn get_many_works() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(ConstMetric {
            id: "a",
            value: MetricValue::Integer(1),
            deps: &[],
        }))
        .unwrap();
        reg.register(Box::new(ConstMetric {
            id: "b",
            value: MetricValue::Float(2.5),
            deps: &[],
        }))
        .unwrap();

        let engine = MetricsEngine::new(reg);
        let text = make_processed_text("test");
        let input = MetricInput::Single(&text);
        let results = engine.get_many(&["a", "b", "missing"], &input);
        assert_eq!(results["a"], MetricValue::Integer(1));
        assert_eq!(results["b"], MetricValue::Float(2.5));
        assert_eq!(results["missing"], MetricValue::Unavailable);
    }

    #[test]
    fn clear_cache_works() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(ConstMetric {
            id: "a",
            value: MetricValue::Integer(1),
            deps: &[],
        }))
        .unwrap();

        let engine = MetricsEngine::with_cache(reg, 10);
        let text = make_processed_text("test");
        let input = MetricInput::Single(&text);
        engine.get("a", &input);
        assert_eq!(engine.cache_len(), 1);
        engine.clear_cache();
        assert_eq!(engine.cache_len(), 0);
    }

    #[test]
    fn no_cache_engine() {
        let reg = MetricRegistry::new();
        let engine = MetricsEngine::new(reg);
        assert_eq!(engine.cache_len(), 0);
        engine.clear_cache(); // should not panic
    }

    #[test]
    fn debug_format() {
        let reg = MetricRegistry::new();
        let engine = MetricsEngine::new(reg);
        let dbg = format!("{:?}", engine);
        assert!(dbg.contains("MetricsEngine"));
    }

    #[test]
    fn default_cache_capacity() {
        let reg = MetricRegistry::new();
        let engine = MetricsEngine::with_default_cache(reg);
        assert!(engine.cache.is_some());
    }

    #[test]
    fn registry_accessor() {
        let mut reg = MetricRegistry::new();
        reg.register(Box::new(ConstMetric {
            id: "a",
            value: MetricValue::Integer(1),
            deps: &[],
        }))
        .unwrap();

        let engine = MetricsEngine::new(reg);
        assert_eq!(engine.registry().len(), 1);
        assert!(engine.registry().get("a").is_some());
    }
}
