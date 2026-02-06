//! Error types for redline-core.

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Top-level error type for redline-core operations.
#[derive(Debug)]
pub enum RedlineError {}

impl core::fmt::Display for RedlineError {
    fn fmt(&self, _f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_error_converts_to_redline_error() {
        let err: RedlineError = StoreError::IdNotFound(42).into();
        assert!(matches!(
            err,
            RedlineError::Store(StoreError::IdNotFound(42))
        ));
    }

    #[test]
    fn tokenize_error_converts() {
        let err: RedlineError = TokenizeError::EmptyResult.into();
        assert!(matches!(
            err,
            RedlineError::Tokenize(TokenizeError::EmptyResult)
        ));
    }

    #[test]
    fn normalize_error_converts() {
        let err: RedlineError = NormalizeError::CompositionFailed.into();
        assert!(matches!(
            err,
            RedlineError::Normalize(NormalizeError::CompositionFailed)
        ));
    }

    #[test]
    fn config_error_converts() {
        let err: RedlineError = ConfigError::Invalid("test".into()).into();
        assert!(matches!(err, RedlineError::Config(ConfigError::Invalid(_))));
    }

    #[test]
    fn error_messages_are_descriptive() {
        let err = StoreError::IdNotFound(42);
        let msg = err.to_string();
        assert!(msg.contains("42"), "Should contain the ID: {msg}");

        let err = TokenizeError::InvalidOffset {
            offset: 10,
            length: 5,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("10") && msg.contains("5"),
            "Should contain both values: {msg}"
        );
    }

    #[test]
    fn errors_are_send_sync() {
        fn assert_send_sync<T: Send + Sync + core::fmt::Debug + core::fmt::Display>() {}
        assert_send_sync::<RedlineError>();
        assert_send_sync::<StoreError>();
        assert_send_sync::<TokenizeError>();
        assert_send_sync::<NormalizeError>();
        assert_send_sync::<ConfigError>();
    }

    #[test]
    fn store_error_variants() {
        let _ = StoreError::IdNotFound(0);
        let _ = StoreError::CapacityExceeded("full".into());
        let _ = StoreError::Frozen;
    }

    #[test]
    fn tokenize_error_variants() {
        let _ = TokenizeError::InvalidOffset {
            offset: 0,
            length: 0,
        };
        let _ = TokenizeError::EmptyResult;
        let _ = TokenizeError::SpanOutOfBounds {
            start: 0,
            end: 0,
            length: 0,
        };
    }

    #[test]
    fn normalize_error_variants() {
        let _ = NormalizeError::CompositionFailed;
        let _ = NormalizeError::InvalidPosition(0);
    }

    #[test]
    fn config_error_variants() {
        let _ = ConfigError::UnknownFeature("x".into());
        let _ = ConfigError::Invalid("y".into());
    }
}
