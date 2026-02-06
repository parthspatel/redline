//! redline-core: High-performance text diffing & analysis framework.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Compile-time enforcement: python and wasm features are mutually exclusive (FOUN-07)
#[cfg(all(feature = "python", feature = "wasm"))]
compile_error!("Features `python` and `wasm` are mutually exclusive.");

pub mod char_mapping;
pub mod error;
pub mod span;
pub mod text_store;

pub use char_mapping::CharMapping;
pub use error::RedlineError;
pub use span::Span;
pub use text_store::{StringId, TextStore, TextStoreBuilder};
