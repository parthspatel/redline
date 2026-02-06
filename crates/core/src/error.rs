//! Error types for redline-core.

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Top-level error type for redline-core operations.
#[derive(Debug, thiserror::Error)]
pub enum RedlineError {
    /// Text store error (interning, lookup, capacity).
    #[error(transparent)]
    Store(#[from] StoreError),

    /// Tokenization error.
    #[error(transparent)]
    Tokenize(#[from] TokenizeError),

    /// Normalization error.
    #[error(transparent)]
    Normalize(#[from] NormalizeError),

    /// Configuration error.
    #[error(transparent)]
    Config(#[from] ConfigError),
}

/// Errors from the TextStore string interning system.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// String ID not found in the store.
    #[error("string id {0} not found in text store")]
    IdNotFound(u32),

    /// Store capacity exceeded.
    #[error("text store capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// Store is frozen (promoted to immutable).
    #[error("text store is frozen; use TextStoreBuilder for mutations")]
    Frozen,
}

/// Errors during tokenization.
#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    /// Byte offset is invalid for the given text length.
    #[error("invalid offset {offset} for text of length {length}")]
    InvalidOffset { offset: u32, length: u32 },

    /// Tokenizer produced no tokens.
    #[error("tokenizer produced empty result")]
    EmptyResult,

    /// Span exceeds text bounds.
    #[error("span {start}..{end} out of bounds for text of length {length}")]
    SpanOutOfBounds { start: u32, end: u32, length: u32 },
}

/// Errors during text normalization.
#[derive(Debug, thiserror::Error)]
pub enum NormalizeError {
    /// CharMapping composition failed.
    #[error("character mapping composition failed")]
    CompositionFailed,

    /// Position is invalid in the normalized text.
    #[error("invalid position {0} in normalized text")]
    InvalidPosition(u32),
}

/// Configuration errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Unknown feature name.
    #[error("unknown feature: {0}")]
    UnknownFeature(String),

    /// Invalid configuration value.
    #[error("invalid configuration: {0}")]
    Invalid(String),
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
