//! Diff computation module: algorithms, types, and coordination.

pub mod algorithm;
pub mod common;
pub mod computer;
pub mod edit_operation;
pub mod error;
pub mod myers;
pub mod result;

pub use algorithm::{AlgorithmOutput, DiffAlgorithm};
pub use common::{apply_operations, build_token_indices, strip_common_affixes};
pub use computer::DiffComputer;
pub use edit_operation::{EditKind, EditOperation};
pub use error::DiffError;
pub use myers::Myers;
pub use result::{DiffMetadata, DiffResult, DiffStatistics};
