//! Text processing pipeline: orchestrates normalization and tokenization.

pub mod layer;
pub mod processed_text;

pub use layer::{ExecutionMode, NormalizationLayer};
pub use processed_text::ProcessedText;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::ToString, vec::Vec};

use crate::error::ProcessError;
use crate::normalize::Normalizer;
use crate::tokenize::Tokenizer;

/// Orchestrator chaining normalizers then a tokenizer.
/// Stub -- full implementation in Plan 07.
pub struct TextProcessor {
    normalizers: Vec<Box<dyn Normalizer>>,
    tokenizer: Box<dyn Tokenizer>,
    mode: ExecutionMode,
}

impl TextProcessor {
    /// Create a new text processor with the given normalizers, tokenizer, and mode.
    pub fn new(
        normalizers: Vec<Box<dyn Normalizer>>,
        tokenizer: Box<dyn Tokenizer>,
        mode: ExecutionMode,
    ) -> Self {
        Self {
            normalizers,
            tokenizer,
            mode,
        }
    }

    /// Process input text through the pipeline. Currently a stub.
    pub fn process(&self, _input: &str) -> Result<ProcessedText, ProcessError> {
        Err(ProcessError::Configuration(
            "TextProcessor not yet implemented".to_string(),
        ))
    }
}
