//! ProcessedText: result of the full text processing pipeline.

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::sync::Arc;

use core::fmt;

use crate::char_mapping::CharMapping;
use crate::error::ProcessError;
use crate::text_store::TextStore;
use crate::token::Token;

use super::layer::NormalizationLayer;

/// Result of text processing through the full pipeline.
pub struct ProcessedText {
    /// Original input text.
    pub original: String,
    /// Normalization layers (ordered by pipeline execution).
    pub layers: Vec<NormalizationLayer>,
    /// Final normalized text.
    pub normalized: String,
    /// Tokens produced from the final normalized text.
    pub tokens: Vec<Token>,
    /// Composed CharMapping: original -> final normalized. None if no normalizers ran.
    pub composed_mapping: Option<CharMapping>,
    /// Text store containing all interned strings.
    pub text_store: Arc<TextStore>,
    /// Names of normalizers configured but NOT stored (Minimal mode).
    pub skipped_normalizers: Vec<String>,
}

impl fmt::Debug for ProcessedText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessedText")
            .field("original", &self.original)
            .field("layers", &self.layers)
            .field("normalized", &self.normalized)
            .field("tokens", &self.tokens)
            .field("composed_mapping", &self.composed_mapping)
            .field(
                "text_store",
                &format_args!("TextStore({} entries)", self.text_store.len()),
            )
            .field("skipped_normalizers", &self.skipped_normalizers)
            .finish()
    }
}

impl ProcessedText {
    /// Look up a layer by normalizer name.
    #[inline]
    pub fn layer(&self, name: &str) -> Option<&NormalizationLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Compute the composed CharMapping from original text to the output
    /// of the named normalization layer.
    pub fn mapping_to_layer(&self, name: &str) -> Result<CharMapping, ProcessError> {
        let target_idx = self
            .layers
            .iter()
            .position(|l| l.name == name)
            .ok_or_else(|| ProcessError::LayerNotFound(name.to_string()))?;

        let mut composed: Option<CharMapping> = None;
        for layer in &self.layers[..=target_idx] {
            composed = Some(match composed {
                None => layer.mapping.clone(),
                Some(prev) => prev.compose(&layer.mapping).map_err(|source| {
                    ProcessError::NormalizationFailed {
                        normalizer: layer.name.clone(),
                        source,
                    }
                })?,
            });
        }

        composed.ok_or_else(|| ProcessError::LayerNotFound(name.to_string()))
    }
}
