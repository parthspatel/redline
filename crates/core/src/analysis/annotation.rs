//! Annotation and region summary types for analysis results.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// An annotation attached to a specific edit operation.
///
/// Annotations carry a confidence score (0.0–1.0) indicating the
/// estimated probability that the annotation is correct.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Annotation {
    /// Index into `DiffResult::operations` that this annotation applies to.
    pub operation_index: usize,
    /// Human-readable label for the annotation.
    pub label: String,
    /// Confidence score in range [0.0, 1.0].
    pub confidence: f64,
    /// Optional detailed description of the finding.
    pub detail: Option<String>,
}

impl Annotation {
    /// Create a new annotation. Confidence is clamped to [0.0, 1.0].
    pub fn new(operation_index: usize, label: impl Into<String>, confidence: f64) -> Self {
        Self {
            operation_index,
            label: label.into(),
            confidence: confidence.clamp(0.0, 1.0),
            detail: None,
        }
    }

    /// Set the detail for this annotation.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// A summary of analysis findings for a contiguous region of operations.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RegionSummary {
    /// Indices into `DiffResult::operations` covered by this region.
    pub operation_indices: Vec<usize>,
    /// Human-readable summary of the region.
    pub label: String,
    /// Confidence score in range [0.0, 1.0].
    pub confidence: f64,
    /// Optional detailed description.
    pub detail: Option<String>,
}

impl RegionSummary {
    /// Create a new region summary. Confidence is clamped to [0.0, 1.0].
    pub fn new(operation_indices: Vec<usize>, label: impl Into<String>, confidence: f64) -> Self {
        Self {
            operation_indices,
            label: label.into(),
            confidence: confidence.clamp(0.0, 1.0),
            detail: None,
        }
    }

    /// Set the detail for this summary.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}
