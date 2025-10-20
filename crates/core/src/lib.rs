//! # TextDiff Library
//!
//! A comprehensive text diffing library with extensible normalization pipelines
//! and tokenization strategies. Maintains full character-level mappings from
//! normalized/tokenized text back to original input.
//!
//! ## Core Concepts
//!
//! - **Normalizers**: Transform text while maintaining character mappings
//! - **Pipelines**: Chain normalizers to create multi-layer transformations
//! - **Tokenizers**: Split normalized text into tokens for diffing
//! - **DiffEngine**: Compute diffs using configurable algorithms
//!
//! ## Example
//!
//! ```rust
//! use redline_core::{DiffEngine, DiffConfig, DiffAlgorithm, TextPipeline};
//! use redline_core::normalizers::Lowercase;
//! use redline_core::tokenizers::WordTokenizer;
//!
//! let config = DiffConfig::default()
//!     .with_algorithm(DiffAlgorithm::Myers)
//!     .with_pipeline(
//!         TextPipeline::new()
//!             .add_normalizer(Box::new(Lowercase))
//!     )
//!     .with_tokenizer(Box::new(WordTokenizer::new()));
//!
//! let engine = DiffEngine::new(config);
//! let diff = engine.diff("Hello World", "Hello Rust");
//! ```

pub mod algorithm;
pub mod analyzer;
pub mod config;
pub mod diff;
pub mod engine;
pub mod execution;
pub mod mapping;
pub mod metrics;
pub mod util;
pub mod normalizers;
pub mod pipeline;
pub mod tokenizers;

// Token alignment utilities (for SpaCy analyzers)
#[cfg(feature = "spacy")]
pub mod token_alignment;

// Re-export main types
pub use config::{DiffAlgorithm, DiffConfig};
pub use diff::{ChangeCategory, DiffOperation, DiffResult, EditType};
pub use engine::DiffEngine;
pub use pipeline::{NormalizationLayer, TextPipeline};
// pub use analyzer::{AnalysisResult, AnalysisReport};

/// Main entry point for computing diffs between two strings
///
/// # Arguments
///
/// * `original` - The original text
/// * `modified` - The modified text
/// * `config` - Optional configuration (uses default if None)
///
/// # Returns
///
/// A complete `DiffResult` containing all analysis and mappings
///
/// # Example
///
/// ```rust
/// use redline_core::compute_diff;
///
/// let result = compute_diff("Hello World", "Hello Rust", None);
/// println!("Changes: {}", result.summary());
/// ```
pub fn compute_diff(original: &str, modified: &str, config: Option<DiffConfig>) -> DiffResult {
    let config = config.unwrap_or_default();
    let engine = DiffEngine::new(config);
    engine.diff(original, modified)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_diff() {
        let result = compute_diff("hello world", "hello rust", None);
        assert!(result.operations.len() > 0);
    }

    #[test]
    fn test_with_normalization() {
        let config = DiffConfig::default()
            .with_pipeline(TextPipeline::new().add_normalizer(Box::new(normalizers::Lowercase)));

        let result = compute_diff("Hello World", "HELLO WORLD", Some(config));
        assert!(result.semantic_similarity > 0.95);
    }
}
