//! SpaCy integration for POS tagging and dependency parsing
//!
//! This module provides a clean interface to SpaCy without caching or heuristics.
//! Caching is handled by TextMetrics, heuristics by analyzers.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use super::syntactic_token::SyntacticToken;

/// Helper macro for PyO3 error mapping to reduce boilerplate
macro_rules! py_err {
    ($operation:expr, $e:expr) => {
        format!("Failed to {}: {}", $operation, $e)
    };
}

/// SpaCy client for NLP operations
///
/// This is a pure utility that:
/// - Loads and manages SpaCy models
/// - Converts text to syntactic tokens
/// - Handles PYTHON_HOME configuration
///
/// It does NOT:
/// - Cache results (that's in TextMetrics)
/// - Implement heuristics (that's in analyzers)
#[derive(Clone)]
pub struct SpacyClient {
    model_name: String,
    nlp: Arc<Mutex<Option<Py<PyAny>>>>,
}

impl SpacyClient {
    /// Create a new SpaCy client with the specified model
    ///
    /// Common models:
    /// - "en_core_web_sm" - Small English model (fast, ~15MB)
    /// - "en_core_web_md" - Medium English model (balanced, ~45MB)
    /// - "en_core_web_lg" - Large English model (accurate, ~800MB)
    ///
    /// Make sure to download the model first:
    /// ```bash
    /// python -m spacy download en_core_web_sm
    /// ```
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            nlp: Arc::new(Mutex::new(None)),
        }
    }

    /// Analyze text and return syntactic tokens
    ///
    /// This performs POS tagging, lemmatization, and dependency parsing.
    /// Results are NOT cached here - use TextMetrics for caching.
    pub fn analyze(&self, text: &str) -> Result<Vec<SyntacticToken>, String> {
        let nlp = self.get_or_load_nlp()?;

        Python::with_gil(|py| {
            let doc = nlp
                .bind(py)
                .call1((text,))
                .map_err(|e| py_err!("process text", e))?;

            let mut tokens = Vec::new();

            // Iterate through tokens in the doc
            let doc_iter = doc.iter().map_err(|e| py_err!("get iterator", e))?;

            for token_result in doc_iter {
                let token = token_result.map_err(|e| py_err!("iterate token", e))?;

                let text = token
                    .getattr("text")
                    .map_err(|e| py_err!("get text", e))?
                    .extract::<String>()
                    .map_err(|e| py_err!("extract text", e))?;

                let lemma = token
                    .getattr("lemma_")
                    .map_err(|e| py_err!("get lemma", e))?
                    .extract::<String>()
                    .map_err(|e| py_err!("extract lemma", e))?;

                let pos = token
                    .getattr("pos_")
                    .map_err(|e| py_err!("get POS", e))?
                    .extract::<String>()
                    .map_err(|e| py_err!("extract POS", e))?;

                let tag = token
                    .getattr("tag_")
                    .map_err(|e| py_err!("get tag", e))?
                    .extract::<String>()
                    .map_err(|e| py_err!("extract tag", e))?;

                let dep = token
                    .getattr("dep_")
                    .map_err(|e| py_err!("get dep", e))?
                    .extract::<String>()
                    .map_err(|e| py_err!("extract dep", e))?;

                let head = token
                    .getattr("head")
                    .map_err(|e| py_err!("get head", e))?
                    .getattr("i")
                    .map_err(|e| py_err!("get head index", e))?
                    .extract::<usize>()
                    .map_err(|e| py_err!("extract head index", e))?;

                let is_stop = token
                    .getattr("is_stop")
                    .map_err(|e| py_err!("get is_stop", e))?
                    .extract::<bool>()
                    .map_err(|e| py_err!("extract is_stop", e))?;

                let is_punct = token
                    .getattr("is_punct")
                    .map_err(|e| py_err!("get is_punct", e))?
                    .extract::<bool>()
                    .map_err(|e| py_err!("extract is_punct", e))?;

                tokens.push(SyntacticToken {
                    text,
                    lemma,
                    pos,
                    tag,
                    dep,
                    head,
                    is_stop,
                    is_punct,
                });
            }

            Ok(tokens)
        })
    }

    /// Initialize or get the SpaCy model
    fn get_or_load_nlp(&self) -> Result<Py<PyAny>, String> {
        let mut nlp_guard = self.nlp.lock().unwrap();

        if let Some(ref nlp) = *nlp_guard {
            return Python::with_gil(|py| Ok(nlp.clone_ref(py)));
        }

        // Load the model
        Python::with_gil(|py| {
            // Configure Python path if PYTHON_HOME is set
            if let Ok(python_home) = std::env::var("PYTHON_HOME") {
                Self::configure_python_path(py, &python_home)?;
            }

            let spacy = py.import_bound("spacy").map_err(|e| {
                format!(
                    "Failed to import spacy: {}. Make sure spacy is installed: pip install spacy",
                    e
                )
            })?;

            let nlp = spacy
                .call_method1("load", (&self.model_name,))
                .map_err(|e| {
                    format!(
                        "Failed to load SpaCy model '{}': {}. Download it with: python -m spacy download {}",
                        self.model_name, e, self.model_name
                    )
                })?;

            let nlp_py: Py<PyAny> = nlp.into();
            *nlp_guard = Some(nlp_py.clone_ref(py));
            Ok(nlp_py)
        })
    }

    /// Configure Python's sys.path to use a virtual environment
    ///
    /// This function:
    /// 1. Detects the Python version from sys.version_info
    /// 2. Constructs the site-packages path using PYTHON_HOME
    /// 3. Prepends it to sys.path so the venv packages are found first
    fn configure_python_path(py: Python, python_home: &str) -> Result<(), String> {
        // Get sys module
        let sys = py
            .import_bound("sys")
            .map_err(|e| py_err!("import sys", e))?;

        // Get Python version (major.minor)
        let version_info = sys
            .getattr("version_info")
            .map_err(|e| py_err!("get version_info", e))?;

        let major = version_info
            .getattr("major")
            .map_err(|e| py_err!("get major version", e))?
            .extract::<i32>()
            .map_err(|e| py_err!("extract major version", e))?;

        let minor = version_info
            .getattr("minor")
            .map_err(|e| py_err!("get minor version", e))?
            .extract::<i32>()
            .map_err(|e| py_err!("extract minor version", e))?;

        // Construct site-packages path
        let site_packages = format!(
            "{}/lib/python{}.{}/site-packages",
            python_home, major, minor
        );

        // Get sys.path
        let path = sys
            .getattr("path")
            .map_err(|e| py_err!("get sys.path", e))?;

        // Insert at the beginning so venv packages take precedence
        path.call_method1("insert", (0, &site_packages))
            .map_err(|e| py_err!("insert into sys.path", e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spacy_client_creation() {
        let client = SpacyClient::new("en_core_web_sm");
        assert_eq!(client.model_name, "en_core_web_sm");
    }
}
