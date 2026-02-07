//! Normalization layer and execution mode types.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::char_mapping::CharMapping;

/// Execution mode for the normalization pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    /// Only store layers required by configured analyzers (default).
    #[default]
    Minimal,
    /// Store all normalization layers.
    All,
}

/// A named normalization layer in the pipeline.
#[derive(Debug, Clone)]
pub struct NormalizationLayer {
    /// Name of the normalizer that produced this layer.
    pub name: String,
    /// Text after this normalization step.
    pub text: String,
    /// CharMapping from the PREVIOUS layer to this layer.
    pub mapping: CharMapping,
}
