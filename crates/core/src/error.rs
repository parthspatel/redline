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

    /// Pipeline processing error.
    #[error(transparent)]
    Process(#[from] ProcessError),

    /// Diff computation error.
    #[error(transparent)]
    Diff(#[from] crate::diff::error::DiffError),
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

    /// Input text is empty (where empty input is not allowed).
    #[error("tokenizer received empty input")]
    EmptyInput,

    /// Tokenization failed with a reason.
    #[error("tokenization failed: {0}")]
    Failed(String),
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

    /// Input text is empty.
    #[error("normalizer received empty input")]
    EmptyInput,

    /// Produced mapping is invalid.
    #[error("normalizer produced invalid character mapping")]
    InvalidMapping,
}

/// Errors from the text processing pipeline.
#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    /// A normalizer in the pipeline failed.
    #[error("normalization failed in '{normalizer}': {source}")]
    NormalizationFailed {
        normalizer: String,
        #[source]
        source: NormalizeError,
    },

    /// The tokenizer failed.
    #[error("tokenization failed in '{tokenizer}': {source}")]
    TokenizationFailed {
        tokenizer: String,
        #[source]
        source: TokenizeError,
    },

    /// Pipeline configuration error.
    #[error("pipeline configuration error: {0}")]
    Configuration(String),

    /// Named layer not found.
    #[error("layer '{0}' not found in pipeline")]
    LayerNotFound(String),
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
    fn process_error_converts() {
        let err: RedlineError = ProcessError::Configuration("test".into()).into();
        assert!(matches!(
            err,
            RedlineError::Process(ProcessError::Configuration(_))
        ));
    }

    #[test]
    fn process_error_normalization_failed() {
        let err = ProcessError::NormalizationFailed {
            normalizer: "lowercaser".into(),
            source: NormalizeError::EmptyInput,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("lowercaser"),
            "Should contain normalizer name: {msg}"
        );
    }

    #[test]
    fn process_error_tokenization_failed() {
        let err = ProcessError::TokenizationFailed {
            tokenizer: "word".into(),
            source: TokenizeError::EmptyInput,
        };
        let msg = err.to_string();
        assert!(msg.contains("word"), "Should contain tokenizer name: {msg}");
    }

    #[test]
    fn process_error_layer_not_found() {
        let err = ProcessError::LayerNotFound("nfkc".into());
        let msg = err.to_string();
        assert!(msg.contains("nfkc"), "Should contain layer name: {msg}");
    }

    #[test]
    fn errors_are_send_sync() {
        fn assert_send_sync<T: Send + Sync + core::fmt::Debug + core::fmt::Display>() {}
        assert_send_sync::<RedlineError>();
        assert_send_sync::<StoreError>();
        assert_send_sync::<TokenizeError>();
        assert_send_sync::<NormalizeError>();
        assert_send_sync::<ConfigError>();
        assert_send_sync::<ProcessError>();
        assert_send_sync::<crate::diff::error::DiffError>();
    }

    #[test]
    fn diff_error_converts() {
        let _: RedlineError = crate::diff::error::DiffError::Cancelled.into();
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
        let _ = TokenizeError::EmptyInput;
        let _ = TokenizeError::Failed("reason".into());
    }

    #[test]
    fn normalize_error_variants() {
        let _ = NormalizeError::CompositionFailed;
        let _ = NormalizeError::InvalidPosition(0);
        let _ = NormalizeError::EmptyInput;
        let _ = NormalizeError::InvalidMapping;
    }

    #[test]
    fn config_error_variants() {
        let _ = ConfigError::UnknownFeature("x".into());
        let _ = ConfigError::Invalid("y".into());
    }

    #[test]
    fn process_error_variants() {
        let _ = ProcessError::NormalizationFailed {
            normalizer: "test".into(),
            source: NormalizeError::EmptyInput,
        };
        let _ = ProcessError::TokenizationFailed {
            tokenizer: "test".into(),
            source: TokenizeError::EmptyInput,
        };
        let _ = ProcessError::Configuration("test".into());
        let _ = ProcessError::LayerNotFound("test".into());
    }

    // ── Edge case tests (02.1-01) ───────────────────────────────────

    #[test]
    fn edge_all_error_display_strings_non_empty() {
        let errors: Vec<Box<dyn core::fmt::Display>> = vec![
            Box::new(StoreError::IdNotFound(0)),
            Box::new(StoreError::CapacityExceeded("cap".into())),
            Box::new(StoreError::Frozen),
            Box::new(TokenizeError::InvalidOffset {
                offset: 0,
                length: 0,
            }),
            Box::new(TokenizeError::EmptyResult),
            Box::new(TokenizeError::SpanOutOfBounds {
                start: 0,
                end: 0,
                length: 0,
            }),
            Box::new(TokenizeError::EmptyInput),
            Box::new(TokenizeError::Failed("reason".into())),
            Box::new(NormalizeError::CompositionFailed),
            Box::new(NormalizeError::InvalidPosition(0)),
            Box::new(NormalizeError::EmptyInput),
            Box::new(NormalizeError::InvalidMapping),
            Box::new(ConfigError::UnknownFeature("x".into())),
            Box::new(ConfigError::Invalid("y".into())),
            Box::new(ProcessError::Configuration("c".into())),
            Box::new(ProcessError::LayerNotFound("l".into())),
        ];
        for err in &errors {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "Display string should not be empty");
        }
    }

    #[test]
    fn edge_store_error_display_formats() {
        assert_eq!(
            StoreError::IdNotFound(42).to_string(),
            "string id 42 not found in text store"
        );
        assert_eq!(
            StoreError::CapacityExceeded("full".into()).to_string(),
            "text store capacity exceeded: full"
        );
        assert_eq!(
            StoreError::Frozen.to_string(),
            "text store is frozen; use TextStoreBuilder for mutations"
        );
    }

    #[test]
    fn edge_tokenize_error_display_formats() {
        assert_eq!(
            TokenizeError::InvalidOffset {
                offset: 10,
                length: 5
            }
            .to_string(),
            "invalid offset 10 for text of length 5"
        );
        assert_eq!(
            TokenizeError::EmptyResult.to_string(),
            "tokenizer produced empty result"
        );
        assert_eq!(
            TokenizeError::SpanOutOfBounds {
                start: 5,
                end: 20,
                length: 10
            }
            .to_string(),
            "span 5..20 out of bounds for text of length 10"
        );
        assert_eq!(
            TokenizeError::EmptyInput.to_string(),
            "tokenizer received empty input"
        );
        assert_eq!(
            TokenizeError::Failed("oops".into()).to_string(),
            "tokenization failed: oops"
        );
    }

    #[test]
    fn edge_normalize_error_display_formats() {
        assert_eq!(
            NormalizeError::CompositionFailed.to_string(),
            "character mapping composition failed"
        );
        assert_eq!(
            NormalizeError::InvalidPosition(99).to_string(),
            "invalid position 99 in normalized text"
        );
        assert_eq!(
            NormalizeError::EmptyInput.to_string(),
            "normalizer received empty input"
        );
        assert_eq!(
            NormalizeError::InvalidMapping.to_string(),
            "normalizer produced invalid character mapping"
        );
    }

    #[test]
    fn edge_normalize_error_invalid_position_stores_value() {
        let err = NormalizeError::InvalidPosition(42);
        match err {
            NormalizeError::InvalidPosition(pos) => assert_eq!(pos, 42),
            _ => panic!("Expected InvalidPosition variant"),
        }
    }

    #[test]
    fn edge_normalize_error_invalid_position_various_values() {
        for val in [0u32, 1, 100, u32::MAX] {
            let err = NormalizeError::InvalidPosition(val);
            match err {
                NormalizeError::InvalidPosition(pos) => assert_eq!(pos, val),
                _ => panic!("Expected InvalidPosition variant"),
            }
        }
    }

    #[test]
    fn edge_config_error_display_formats() {
        assert_eq!(
            ConfigError::UnknownFeature("foo".into()).to_string(),
            "unknown feature: foo"
        );
        assert_eq!(
            ConfigError::Invalid("bar".into()).to_string(),
            "invalid configuration: bar"
        );
    }

    #[test]
    fn edge_process_error_display_formats() {
        assert_eq!(
            ProcessError::Configuration("bad".into()).to_string(),
            "pipeline configuration error: bad"
        );
        assert_eq!(
            ProcessError::LayerNotFound("missing".into()).to_string(),
            "layer 'missing' not found in pipeline"
        );
    }

    #[test]
    fn edge_redline_error_display_transparent() {
        let inner = StoreError::IdNotFound(7);
        let inner_msg = inner.to_string();
        let outer: RedlineError = inner.into();
        assert_eq!(outer.to_string(), inner_msg);
    }

    #[test]
    fn edge_all_redline_error_from_conversions() {
        let _: RedlineError = StoreError::Frozen.into();
        let _: RedlineError = TokenizeError::EmptyResult.into();
        let _: RedlineError = NormalizeError::EmptyInput.into();
        let _: RedlineError = ConfigError::Invalid("x".into()).into();
        let _: RedlineError = ProcessError::Configuration("x".into()).into();
    }

    #[test]
    fn edge_debug_output_non_empty() {
        let errors: Vec<Box<dyn core::fmt::Debug>> = vec![
            Box::new(StoreError::IdNotFound(0)),
            Box::new(TokenizeError::EmptyResult),
            Box::new(NormalizeError::CompositionFailed),
            Box::new(ConfigError::Invalid("x".into())),
            Box::new(ProcessError::Configuration("x".into())),
            Box::new(RedlineError::from(StoreError::Frozen)),
        ];
        for err in &errors {
            let msg = format!("{:?}", err);
            assert!(!msg.is_empty(), "Debug output should not be empty");
        }
    }
}
