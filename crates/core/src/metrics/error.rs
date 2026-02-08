//! Error types for the metrics engine.

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Errors from metric registration, validation, and computation.
#[derive(Debug, thiserror::Error)]
pub enum MetricError {
    /// Metric ID not found in registry.
    #[error("metric '{0}' not found in registry")]
    NotFound(String),

    /// Metric ID already registered (use `register_with_override` to replace).
    #[error("metric '{0}' already registered (use register_with_override to replace)")]
    AlreadyRegistered(String),

    /// Static dependency graph contains a cycle.
    #[error("dependency cycle detected: {0}")]
    CycleDetected(String),

    /// A declared dependency is not registered.
    #[error("dependency '{0}' not found in registry")]
    DependencyNotFound(String),

    /// A dependency computed to Unavailable.
    #[error("dependency '{0}' unavailable")]
    DependencyUnavailable(String),

    /// Input type mismatch (e.g., pairwise metric received single input).
    #[error("invalid input for metric '{0}': {1}")]
    InvalidInput(String, String),

    /// Generic computation error.
    #[error("computation error in metric '{0}': {1}")]
    ComputationError(String, String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        assert_eq!(
            MetricError::NotFound("foo".into()).to_string(),
            "metric 'foo' not found in registry"
        );
        assert_eq!(
            MetricError::AlreadyRegistered("bar".into()).to_string(),
            "metric 'bar' already registered (use register_with_override to replace)"
        );
        assert_eq!(
            MetricError::CycleDetected("a -> b -> a".into()).to_string(),
            "dependency cycle detected: a -> b -> a"
        );
        assert_eq!(
            MetricError::DependencyNotFound("dep".into()).to_string(),
            "dependency 'dep' not found in registry"
        );
        assert_eq!(
            MetricError::DependencyUnavailable("dep".into()).to_string(),
            "dependency 'dep' unavailable"
        );
        assert_eq!(
            MetricError::InvalidInput("m".into(), "expected pairwise".into()).to_string(),
            "invalid input for metric 'm': expected pairwise"
        );
        assert_eq!(
            MetricError::ComputationError("m".into(), "division by zero".into()).to_string(),
            "computation error in metric 'm': division by zero"
        );
    }

    #[test]
    fn errors_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetricError>();
    }
}
