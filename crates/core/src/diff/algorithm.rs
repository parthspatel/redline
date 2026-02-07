//! DiffAlgorithm trait and AlgorithmOutput for pluggable diff algorithms.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::ops::ControlFlow;

use super::edit_operation::EditOperation;
use super::error::DiffError;

/// Output from a diff algorithm computation.
#[derive(Debug, Clone)]
pub struct AlgorithmOutput {
    /// The sequence of edit operations.
    pub operations: Vec<EditOperation>,
    /// Whether the result is approximate (e.g., due to threshold cutoff).
    pub is_approximate: bool,
    /// If an approximation threshold was used, the value.
    pub threshold_used: Option<usize>,
}

/// Trait for pluggable diff algorithms.
///
/// Algorithms operate on pre-interned `&[u32]` token index slices.
/// The `DiffComputer` handles interning from `Token`/`TextStore` before calling `compute`.
pub trait DiffAlgorithm: Send + Sync {
    /// Returns the name of this algorithm (e.g., "myers", "histogram").
    fn name(&self) -> &str;

    /// Compute the diff between `source` and `target` token index sequences.
    ///
    /// The optional `progress` callback receives a value in `[0.0, 1.0]` and
    /// can return `ControlFlow::Break(())` to cancel the computation.
    fn compute(
        &self,
        source: &[u32],
        target: &[u32],
        progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
    ) -> Result<AlgorithmOutput, DiffError>;
}
