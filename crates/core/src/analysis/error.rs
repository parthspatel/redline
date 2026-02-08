//! Error types for the analysis framework.

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Errors from analyzer registration, dependency resolution, and execution.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    /// Analyzer ID not found in registry.
    #[error("analyzer '{0}' not found in registry")]
    NotFound(String),

    /// Analyzer ID already registered.
    #[error("analyzer '{0}' already registered")]
    AlreadyRegistered(String),

    /// Dependency cycle detected in analyzer graph.
    #[error("dependency cycle detected: {0}")]
    CycleDetected(String),

    /// A required dependency is not registered.
    #[error("required dependency '{dependency}' not found for analyzer '{analyzer}'")]
    DependencyNotFound {
        analyzer: String,
        dependency: String,
    },

    /// An analyzer failed during execution.
    #[error("analyzer '{analyzer}' failed: {reason}")]
    AnalyzerFailed { analyzer: String, reason: String },

    /// An analyzer panicked during execution.
    #[error("analyzer '{analyzer}' panicked: {message}")]
    AnalyzerPanicked { analyzer: String, message: String },

    /// Analyzer was skipped because a required dependency failed.
    #[error("analyzer '{analyzer}' skipped: depends on failed '{dependency}'")]
    AnalyzerSkipped {
        analyzer: String,
        dependency: String,
    },

    /// No analyzers registered.
    #[error("no analyzers registered")]
    NoAnalyzers,
}
