//! Orchestration pipeline: configuration, caching, filtering, and result types.

pub mod cache;
pub mod config;
pub mod error;
pub mod filter;
pub mod result;

pub use cache::CacheManager;
pub use config::{ConfigBuilder, Preset, RedlineConfig};
pub use error::OrchestrateError;
pub use filter::{Filter, FilterContext};
pub use result::RedlineResult;
