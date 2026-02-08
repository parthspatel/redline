//! Metrics engine: trait definitions, types, registry, cache, and evaluation engine.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec, vec::Vec};

use hashbrown::HashMap;

use crate::process::ProcessedText;

pub mod builtins;
pub mod cache;
pub mod counts;
pub mod engine;
pub mod error;
pub mod readability;
pub mod registry;
pub mod similarity;

pub use builtins::register_builtins;
pub use cache::{CacheKey, ContentHash, MetricCache};
pub use engine::MetricsEngine;
pub use error::MetricError;
pub use registry::MetricRegistry;

/// Value returned by a metric computation.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricValue {
    /// An integer result (e.g., word count).
    Integer(i64),
    /// A floating-point result (e.g., readability score).
    Float(f64),
    /// The metric could not be computed (dependency failure, cycle, error).
    Unavailable,
}

impl MetricValue {
    /// Returns the integer value if this is `Integer`, or `None`.
    #[inline]
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            MetricValue::Integer(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the float value if this is `Float`, or `None`.
    #[inline]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            MetricValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns `true` if this is `Unavailable`.
    #[inline]
    pub fn is_unavailable(&self) -> bool {
        matches!(self, MetricValue::Unavailable)
    }
}

/// Input to a metric computation.
#[derive(Debug)]
pub enum MetricInput<'a> {
    /// A single text for single-text metrics.
    Single(&'a ProcessedText),
    /// A pair of texts for pairwise/comparison metrics.
    Pairwise(&'a ProcessedText, &'a ProcessedText),
}

/// Whether a metric's dependencies are known at compile time or discovered at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyKind {
    /// All dependencies are known statically and validated at registration.
    Static,
    /// Additional dependencies may be discovered at compute time.
    Dynamic,
}

/// Trait for a single metric computation.
///
/// All metrics implement this trait and are registered with a [`MetricRegistry`].
/// The engine resolves dependencies lazily and caches results.
///
/// # Object Safety
///
/// This trait is object-safe (`dyn Metric`).
pub trait Metric: Send + Sync {
    /// Unique identifier for this metric (e.g., `"word_count"`, `"flesch_kincaid"`).
    fn id(&self) -> &str;

    /// Compute the metric value given input text and resolved dependency values.
    fn compute(
        &self,
        input: &MetricInput<'_>,
        dependencies: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError>;

    /// Static dependencies known at compile time. Validated at registration.
    fn static_dependencies(&self) -> &[&str] {
        &[]
    }

    /// Dynamic dependencies discovered at compute time. Validated at runtime.
    fn dynamic_dependencies(&self, _input: &MetricInput<'_>) -> Vec<String> {
        vec![]
    }

    /// Whether this metric has static-only or dynamic dependencies.
    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }

    /// Relative cost hint for scheduling (0.0 = free, 1.0 = expensive).
    fn cost(&self) -> f32 {
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_value_integer() {
        let v = MetricValue::Integer(42);
        assert_eq!(v.as_integer(), Some(42));
        assert_eq!(v.as_float(), None);
        assert!(!v.is_unavailable());
    }

    #[test]
    fn metric_value_float() {
        let v = MetricValue::Float(3.14);
        assert_eq!(v.as_integer(), None);
        assert_eq!(v.as_float(), Some(3.14));
        assert!(!v.is_unavailable());
    }

    #[test]
    fn metric_value_unavailable() {
        let v = MetricValue::Unavailable;
        assert_eq!(v.as_integer(), None);
        assert_eq!(v.as_float(), None);
        assert!(v.is_unavailable());
    }

    #[test]
    fn metric_value_clone_and_eq() {
        let v1 = MetricValue::Integer(10);
        let v2 = v1.clone();
        assert_eq!(v1, v2);

        let v3 = MetricValue::Unavailable;
        let v4 = v3.clone();
        assert_eq!(v3, v4);
    }

    #[test]
    fn metric_trait_is_object_safe() {
        fn _assert_object_safe(_: &dyn Metric) {}
    }

    #[test]
    fn dependency_kind_copy() {
        let k = DependencyKind::Static;
        let k2 = k;
        assert_eq!(k, k2);
    }

    #[test]
    fn metric_value_negative_integer() {
        let v = MetricValue::Integer(-5);
        assert_eq!(v.as_integer(), Some(-5));
    }

    #[test]
    fn metric_value_zero_float() {
        let v = MetricValue::Float(0.0);
        assert_eq!(v.as_float(), Some(0.0));
        assert!(!v.is_unavailable());
    }
}
