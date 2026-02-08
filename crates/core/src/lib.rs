//! redline-core: High-performance text diffing & analysis framework.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Compile-time enforcement: python and wasm features are mutually exclusive (FOUN-07)
#[cfg(all(feature = "python", feature = "wasm"))]
compile_error!("Features `python` and `wasm` are mutually exclusive.");

pub mod char_mapping;
pub mod diff;
pub mod error;
pub mod metrics;
pub mod normalize;
pub mod process;
pub mod span;
pub mod text_store;
pub mod token;
pub mod tokenize;

pub use char_mapping::CharMapping;
pub use diff::{
    DiffAlgorithm, DiffComputer, DiffError, DiffResult, EditKind, EditOperation, Histogram, Myers,
};
pub use error::RedlineError;
pub use metrics::{
    DependencyKind, Metric, MetricCache, MetricInput, MetricRegistry, MetricValue, MetricsEngine,
    register_builtins,
};
pub use normalize::{NormalizationResult, Normalizer};
pub use process::{ExecutionMode, NormalizationLayer, ProcessedText, TextProcessor};
pub use span::Span;
pub use text_store::{StringId, TextStore, TextStoreBuilder};
pub use token::{Token, TokenKind};
pub use tokenize::Tokenizer;
