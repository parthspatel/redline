//! Error types for the orchestration pipeline.

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Errors that can occur during orchestration pipeline execution.
#[derive(Debug, thiserror::Error)]
pub enum OrchestrateError {
    /// Configuration validation failed.
    #[error("configuration error: {0}")]
    Config(String),

    /// Pipeline processing failed.
    #[error("pipeline processing failed: {0}")]
    Process(#[from] crate::error::ProcessError),

    /// Diff computation failed.
    #[error("diff computation failed: {0}")]
    Diff(#[from] crate::diff::DiffError),

    /// Metrics computation failed.
    #[error("metrics computation failed: {0}")]
    Metrics(#[from] crate::metrics::MetricError),

    /// Analysis failed.
    #[error("analysis failed: {0}")]
    Analysis(#[from] crate::analysis::AnalysisError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_error_display() {
        let err = OrchestrateError::Config("bad value".into());
        assert_eq!(err.to_string(), "configuration error: bad value");
    }

    #[test]
    fn orchestrate_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OrchestrateError>();
    }

    #[test]
    fn process_error_converts() {
        let proc_err = crate::error::ProcessError::Configuration("test".into());
        let orch_err: OrchestrateError = proc_err.into();
        assert!(matches!(orch_err, OrchestrateError::Process(_)));
    }

    #[test]
    fn diff_error_converts() {
        let diff_err = crate::diff::DiffError::Cancelled;
        let orch_err: OrchestrateError = diff_err.into();
        assert!(matches!(orch_err, OrchestrateError::Diff(_)));
    }

    #[test]
    fn metrics_error_converts() {
        let metric_err = crate::metrics::MetricError::NotFound("test".into());
        let orch_err: OrchestrateError = metric_err.into();
        assert!(matches!(orch_err, OrchestrateError::Metrics(_)));
    }

    #[test]
    fn analysis_error_converts() {
        let analysis_err = crate::analysis::AnalysisError::NotFound("test".into());
        let orch_err: OrchestrateError = analysis_err.into();
        assert!(matches!(orch_err, OrchestrateError::Analysis(_)));
    }
}
