//! Error types for diff computation.

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Errors that can occur during diff computation.
#[derive(Debug, thiserror::Error)]
pub enum DiffError {
    /// Diff computation was cancelled (e.g., by a progress callback).
    #[error("diff cancelled by user")]
    Cancelled,

    /// Invalid input was provided to the diff algorithm.
    #[error("invalid diff input: {0}")]
    InvalidInput(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_display() {
        assert_eq!(DiffError::Cancelled.to_string(), "diff cancelled by user");
    }

    #[test]
    fn invalid_input_display() {
        let err = DiffError::InvalidInput("empty source".into());
        assert_eq!(err.to_string(), "invalid diff input: empty source");
    }

    #[test]
    fn diff_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DiffError>();
    }
}
