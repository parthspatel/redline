//! DiffComputer: coordinates diff algorithms with token interning and result construction.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::ToString, sync::Arc};
#[cfg(feature = "std")]
use std::sync::Arc;

use core::ops::ControlFlow;

use crate::process::ProcessedText;

use super::algorithm::DiffAlgorithm;
use super::common::build_token_indices;
use super::error::DiffError;
use super::result::{DiffMetadata, DiffResult, DiffStatistics};

/// Coordinator that wires a diff algorithm to ProcessedText inputs.
pub struct DiffComputer {
    algorithm: Box<dyn DiffAlgorithm>,
}

impl DiffComputer {
    /// Create a DiffComputer with the default Myers algorithm.
    pub fn new() -> Self {
        Self {
            algorithm: Box::new(super::myers::Myers::new()),
        }
    }

    /// Create a DiffComputer with a specific algorithm.
    pub fn with_algorithm(algorithm: Box<dyn DiffAlgorithm>) -> Self {
        Self { algorithm }
    }

    /// Compute a diff between two processed texts.
    pub fn compute(
        &self,
        source: Arc<ProcessedText>,
        target: Arc<ProcessedText>,
        progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
    ) -> Result<DiffResult, DiffError> {
        let (src_ids, tgt_ids) = build_token_indices(
            &source.tokens,
            &source.text_store,
            &target.tokens,
            &target.text_store,
        );

        let output = self.algorithm.compute(&src_ids, &tgt_ids, progress)?;

        let statistics = DiffStatistics::from_operations(
            &output.operations,
            source.tokens.len(),
            target.tokens.len(),
        );

        let metadata = DiffMetadata {
            algorithm_name: self.algorithm.name().to_string(),
            is_approximate: output.is_approximate,
            threshold_used: output.threshold_used,
        };

        Ok(DiffResult {
            operations: output.operations,
            statistics,
            metadata,
            source,
            target,
        })
    }
}

impl Default for DiffComputer {
    fn default() -> Self {
        Self::new()
    }
}

// DiffComputer holds Box<dyn DiffAlgorithm> which requires Send+Sync via supertrait.
// Manual Debug since Box<dyn DiffAlgorithm> doesn't implement Debug.
impl core::fmt::Debug for DiffComputer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DiffComputer")
            .field("algorithm", &self.algorithm.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::algorithm::AlgorithmOutput;
    use crate::diff::edit_operation::EditOperation;
    use crate::process::ProcessedText;
    use crate::span::Span;
    use crate::text_store::TextStoreBuilder;
    use crate::token::{Token, TokenKind};

    struct IdentityAlgorithm;

    impl DiffAlgorithm for IdentityAlgorithm {
        fn name(&self) -> &str {
            "identity"
        }

        fn compute(
            &self,
            source: &[u32],
            target: &[u32],
            _progress: Option<&mut dyn FnMut(f64) -> ControlFlow<()>>,
        ) -> Result<AlgorithmOutput, DiffError> {
            let operations = if source == target {
                if source.is_empty() {
                    vec![]
                } else {
                    vec![EditOperation::equal(
                        0,
                        source.len() as u32,
                        0,
                        target.len() as u32,
                    )]
                }
            } else {
                vec![EditOperation::replace(
                    0,
                    source.len() as u32,
                    0,
                    target.len() as u32,
                )]
            };

            Ok(AlgorithmOutput {
                operations,
                is_approximate: false,
                threshold_used: None,
            })
        }
    }

    fn make_processed(text: &str, words: &[&str]) -> Arc<ProcessedText> {
        let mut builder = TextStoreBuilder::new();
        let mut tokens = Vec::new();
        let mut offset = 0u32;
        for &w in words {
            let id = builder.intern(w);
            let end = offset + w.len() as u32;
            tokens.push(Token::new(id, Span::new(offset, end), TokenKind::Regular));
            offset = end + 1;
        }
        let store = builder.build();
        Arc::new(ProcessedText {
            original: text.to_string(),
            layers: vec![],
            normalized: text.to_string(),
            tokens,
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        })
    }

    #[test]
    fn compute_identical_texts() {
        let computer = DiffComputer::with_algorithm(Box::new(IdentityAlgorithm));
        let source = make_processed("hello world", &["hello", "world"]);
        let target = make_processed("hello world", &["hello", "world"]);

        let result = computer.compute(source, target, None).unwrap();
        assert!((result.similarity() - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.statistics.equal_count, 1);
        assert_eq!(result.statistics.edit_distance, 0);
        assert_eq!(result.metadata.algorithm_name, "identity");
        assert!(!result.is_approximate());
    }

    #[test]
    fn compute_different_texts() {
        let computer = DiffComputer::with_algorithm(Box::new(IdentityAlgorithm));
        let source = make_processed("hello", &["hello"]);
        let target = make_processed("world", &["world"]);

        let result = computer.compute(source, target, None).unwrap();
        assert!(result.similarity().abs() < f64::EPSILON);
        assert_eq!(result.statistics.replace_count, 1);
    }

    #[test]
    fn compute_empty_texts() {
        let computer = DiffComputer::with_algorithm(Box::new(IdentityAlgorithm));
        let source = make_processed("", &[]);
        let target = make_processed("", &[]);

        let result = computer.compute(source, target, None).unwrap();
        assert!((result.similarity() - 1.0).abs() < f64::EPSILON);
        assert!(result.operations.is_empty());
    }

    #[test]
    fn new_uses_myers_default() {
        let computer = DiffComputer::new();
        assert_eq!(computer.algorithm.name(), "myers");
    }

    #[test]
    fn default_uses_myers() {
        let computer = DiffComputer::default();
        assert_eq!(computer.algorithm.name(), "myers");
    }

    #[test]
    fn new_computes_correct_diff() {
        let computer = DiffComputer::new();
        let source = make_processed("hello world", &["hello", "world"]);
        let target = make_processed("hello earth", &["hello", "earth"]);

        let result = computer.compute(source, target, None).unwrap();
        assert_eq!(result.metadata.algorithm_name, "myers");
        assert_eq!(result.statistics.equal_count, 1);
        assert_eq!(result.statistics.replace_count, 1);
    }
}
