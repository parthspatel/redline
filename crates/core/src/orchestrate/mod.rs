//! Orchestration pipeline: configuration, filtering, and result types.

pub mod config;
pub mod error;
pub mod filter;
pub mod result;

pub use config::{ConfigBuilder, Preset, RedlineConfig};
pub use error::OrchestrateError;
pub use filter::{Filter, FilterContext};
pub use result::RedlineResult;
