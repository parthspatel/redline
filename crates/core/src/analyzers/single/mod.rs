//! Single diff analyzers organized by analysis type

pub mod semantic;
pub mod readability;
pub mod stylistic;
pub mod category;
pub mod features;

// Re-export all analyzers
pub use semantic::*;
pub use readability::*;
pub use stylistic::*;
pub use category::*;
pub use features::*;
