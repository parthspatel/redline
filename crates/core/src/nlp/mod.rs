//! Natural Language Processing utilities
//!
//! This module provides NLP integrations for syntactic and semantic analysis.

#[cfg(feature = "spacy")]
pub mod spacy;

#[cfg(feature = "spacy")]
pub use spacy::SpacyClient;
