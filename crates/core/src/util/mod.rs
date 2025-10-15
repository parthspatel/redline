//! Natural Language Processing utilities
//!
//! This module provides NLP integrations for syntactic and semantic analysis.

#[cfg(feature = "spacy")]
pub mod spacy;
pub mod syntactic_token;
pub mod token_statistics;

#[cfg(feature = "spacy")]
pub use spacy::SpacyClient;

pub use syntactic_token::SyntacticToken;
