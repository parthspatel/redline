//! Text processing pipeline: orchestrates normalization and tokenization.

pub mod layer;
pub mod processed_text;

pub use layer::{ExecutionMode, NormalizationLayer};
pub use processed_text::ProcessedText;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, string::ToString, vec, vec::Vec};

use crate::char_mapping::CharMapping;
use crate::error::ProcessError;
use crate::normalize::Normalizer;
use crate::text_store::TextStoreBuilder;
use crate::tokenize::Tokenizer;

/// Orchestrator chaining normalizers then a tokenizer.
pub struct TextProcessor {
    normalizers: Vec<Box<dyn Normalizer>>,
    tokenizer: Box<dyn Tokenizer>,
    mode: ExecutionMode,
}

impl TextProcessor {
    /// Create a new text processor. Validates normalizer name uniqueness.
    pub fn new(
        normalizers: Vec<Box<dyn Normalizer>>,
        tokenizer: Box<dyn Tokenizer>,
        mode: ExecutionMode,
    ) -> Result<Self, ProcessError> {
        // Validate normalizer name uniqueness
        let mut seen = Vec::new();
        for norm in &normalizers {
            let name = norm.name();
            if seen.contains(&name) {
                return Err(ProcessError::Configuration(format!(
                    "duplicate normalizer name: '{name}'"
                )));
            }
            seen.push(name);
        }
        Ok(Self {
            normalizers,
            tokenizer,
            mode,
        })
    }

    /// Process input text through the normalizer chain then tokenizer.
    pub fn process(&self, input: &str) -> Result<ProcessedText, ProcessError> {
        if input.is_empty() {
            let store = TextStoreBuilder::new();
            return Ok(ProcessedText {
                original: String::new(),
                layers: vec![],
                normalized: String::new(),
                tokens: vec![],
                composed_mapping: None,
                text_store: store.build(),
                skipped_normalizers: vec![],
            });
        }

        let mut current_text = input.to_string();
        let mut layers: Vec<NormalizationLayer> = Vec::new();
        let mut composed: Option<CharMapping> = None;
        let mut skipped: Vec<String> = Vec::new();

        for normalizer in &self.normalizers {
            let result = normalizer.normalize(&current_text).map_err(|source| {
                ProcessError::NormalizationFailed {
                    normalizer: normalizer.name().to_string(),
                    source,
                }
            })?;

            // Compose mapping
            composed = Some(match &composed {
                None => result.mapping.clone(),
                Some(prev) => prev.compose(&result.mapping).map_err(|source| {
                    ProcessError::NormalizationFailed {
                        normalizer: normalizer.name().to_string(),
                        source,
                    }
                })?,
            });

            match self.mode {
                ExecutionMode::All => {
                    layers.push(NormalizationLayer {
                        name: normalizer.name().to_string(),
                        text: result.text.clone(),
                        mapping: result.mapping,
                    });
                }
                ExecutionMode::Minimal => {
                    skipped.push(normalizer.name().to_string());
                }
            }

            current_text = result.text;
        }

        // Build mapping for tokenizer: composed if available, otherwise identity
        let final_mapping = match &composed {
            Some(m) => m.clone(),
            None => CharMapping::identity(current_text.len() as u32).map_err(|source| {
                ProcessError::NormalizationFailed {
                    normalizer: "identity".to_string(),
                    source,
                }
            })?,
        };

        let mut store = TextStoreBuilder::new();
        let tokens = self
            .tokenizer
            .tokenize(&current_text, &final_mapping, &mut store)
            .map_err(|source| ProcessError::TokenizationFailed {
                tokenizer: self.tokenizer.name().to_string(),
                source,
            })?;

        Ok(ProcessedText {
            original: input.to_string(),
            layers,
            normalized: current_text,
            tokens,
            composed_mapping: composed,
            text_store: store.build(),
            skipped_normalizers: skipped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{NormalizeError, TokenizeError};
    use crate::normalize::lowercase::Lowercase;
    use crate::normalize::whitespace::WhitespaceNormalizer;
    use crate::normalize::{NormalizationResult, Normalizer};
    use crate::token::Token;

    use crate::tokenize::word::WordTokenizer;

    fn processor_lc_ws() -> TextProcessor {
        TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
            ExecutionMode::All,
        )
        .unwrap()
    }

    // ── Construction tests ───────────────────────────────────────────

    #[test]
    fn new_with_unique_names_succeeds() {
        let result = TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
            ExecutionMode::All,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn new_with_duplicate_names_fails() {
        let result = TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(Lowercase)],
            Box::new(WordTokenizer),
            ExecutionMode::All,
        );
        assert!(matches!(result, Err(ProcessError::Configuration(_))));
    }

    #[test]
    fn new_with_no_normalizers_succeeds() {
        let result = TextProcessor::new(vec![], Box::new(WordTokenizer), ExecutionMode::All);
        assert!(result.is_ok());
    }

    // ── Pipeline execution tests ─────────────────────────────────────

    #[test]
    fn process_basic_pipeline() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();
        assert_eq!(result.normalized, "hello world");
        assert_eq!(result.tokens.len(), 2);
        assert_eq!(
            result.text_store.resolve(result.tokens[0].text_id).unwrap(),
            "hello"
        );
        assert_eq!(
            result.text_store.resolve(result.tokens[1].text_id).unwrap(),
            "world"
        );
    }

    #[test]
    fn process_no_normalizers() {
        let proc = TextProcessor::new(vec![], Box::new(WordTokenizer), ExecutionMode::All).unwrap();
        let result = proc.process("hello world").unwrap();
        assert_eq!(result.normalized, "hello world");
        assert_eq!(result.tokens.len(), 2);
        assert!(result.composed_mapping.is_none());
    }

    #[test]
    fn process_empty_input() {
        let proc = processor_lc_ws();
        let result = proc.process("").unwrap();
        assert!(result.tokens.is_empty());
        assert!(result.layers.is_empty());
        assert_eq!(result.normalized, "");
        assert_eq!(result.original, "");
    }

    // ── CharMapping composition tests ────────────────────────────────

    #[test]
    fn composed_mapping_through_two_normalizers() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();
        let mapping = result.composed_mapping.as_ref().unwrap();
        // Original 'W' at byte 7 -> lowercase 'w' at byte 7, then whitespace collapses
        // "hello  world" -> "hello world" where 'w' is at byte 6
        let norm_pos = mapping.to_normalized(7).unwrap();
        assert_eq!(norm_pos, 6);
    }

    #[test]
    fn composed_mapping_identity_with_no_normalizers() {
        let proc = TextProcessor::new(vec![], Box::new(WordTokenizer), ExecutionMode::All).unwrap();
        let result = proc.process("hello").unwrap();
        assert!(result.composed_mapping.is_none());
    }

    // ── Layer tests ──────────────────────────────────────────────────

    #[test]
    fn layers_stored_in_all_mode() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();
        assert_eq!(result.layers.len(), 2);
    }

    #[test]
    fn layer_by_name() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();
        let layer = result.layer("lowercase").unwrap();
        assert_eq!(layer.text, "hello  world");
    }

    #[test]
    fn layer_not_found_returns_none() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();
        assert!(result.layer("nonexistent").is_none());
    }

    #[test]
    fn mapping_to_layer_composes_correctly() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();

        // mapping_to_layer("lowercase") = mapping from original to post-lowercase
        let lc_mapping = result.mapping_to_layer("lowercase").unwrap();
        // 'H' at 0 -> 'h' at 0 (identity for lowercase of same-length chars)
        assert_eq!(lc_mapping.to_normalized(0).unwrap(), 0);

        // mapping_to_layer("whitespace") = mapping from original to post-whitespace
        let ws_mapping = result.mapping_to_layer("whitespace").unwrap();
        // Original 'W' at byte 7 -> normalized 'w' at byte 6
        assert_eq!(ws_mapping.to_normalized(7).unwrap(), 6);
    }

    #[test]
    fn mapping_to_layer_not_found() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();
        let err = result.mapping_to_layer("nonexistent").unwrap_err();
        assert!(matches!(err, ProcessError::LayerNotFound(_)));
    }

    // ── Minimal mode tests ───────────────────────────────────────────

    #[test]
    fn minimal_mode_stores_no_layers() {
        let proc = TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
            ExecutionMode::Minimal,
        )
        .unwrap();
        let result = proc.process("Hello  WORLD").unwrap();
        assert!(result.layers.is_empty());
    }

    #[test]
    fn minimal_mode_records_skipped() {
        let proc = TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
            ExecutionMode::Minimal,
        )
        .unwrap();
        let result = proc.process("Hello  WORLD").unwrap();
        assert_eq!(result.skipped_normalizers.len(), 2);
        assert_eq!(result.skipped_normalizers[0], "lowercase");
        assert_eq!(result.skipped_normalizers[1], "whitespace");
    }

    #[test]
    fn all_mode_records_no_skipped() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();
        assert!(result.skipped_normalizers.is_empty());
    }

    // ── Error propagation tests ──────────────────────────────────────

    struct FailingNormalizer;
    impl Normalizer for FailingNormalizer {
        fn normalize(&self, _input: &str) -> Result<NormalizationResult, NormalizeError> {
            Err(NormalizeError::CompositionFailed)
        }
        fn name(&self) -> &str {
            "failing"
        }
    }

    struct FailingTokenizer;
    impl Tokenizer for FailingTokenizer {
        fn tokenize(
            &self,
            _text: &str,
            _mapping: &CharMapping,
            _store: &mut TextStoreBuilder,
        ) -> Result<Vec<Token>, TokenizeError> {
            Err(TokenizeError::EmptyResult)
        }
        fn name(&self) -> &str {
            "failing"
        }
    }

    #[test]
    fn normalizer_error_propagates() {
        let proc = TextProcessor::new(
            vec![Box::new(FailingNormalizer)],
            Box::new(WordTokenizer),
            ExecutionMode::All,
        )
        .unwrap();
        let err = proc.process("hello").unwrap_err();
        match err {
            ProcessError::NormalizationFailed { normalizer, .. } => {
                assert_eq!(normalizer, "failing");
            }
            other => panic!("Expected NormalizationFailed, got: {other:?}"),
        }
    }

    #[test]
    fn tokenizer_error_propagates() {
        let proc =
            TextProcessor::new(vec![], Box::new(FailingTokenizer), ExecutionMode::All).unwrap();
        let err = proc.process("hello").unwrap_err();
        match err {
            ProcessError::TokenizationFailed { tokenizer, .. } => {
                assert_eq!(tokenizer, "failing");
            }
            other => panic!("Expected TokenizationFailed, got: {other:?}"),
        }
    }

    // ── ProcessedText field tests ────────────────────────────────────

    #[test]
    fn processed_text_has_text_store() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello World").unwrap();
        for tok in &result.tokens {
            assert!(result.text_store.resolve(tok.text_id).is_ok());
        }
    }

    #[test]
    fn processed_text_original_preserved() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello  WORLD").unwrap();
        assert_eq!(result.original, "Hello  WORLD");
    }

    // ── Minimal mode still produces correct tokens ───────────────────

    #[test]
    fn minimal_mode_still_normalizes_and_tokenizes() {
        let proc = TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
            ExecutionMode::Minimal,
        )
        .unwrap();
        let result = proc.process("Hello  WORLD").unwrap();
        assert_eq!(result.normalized, "hello world");
        assert_eq!(result.tokens.len(), 2);
    }

    // ── Edge case tests (02.1-03) ────────────────────────────────────

    #[test]
    fn edge_process_single_char() {
        let proc = processor_lc_ws();
        let result = proc.process("A").unwrap();
        assert_eq!(result.normalized, "a");
        assert_eq!(result.tokens.len(), 1);
        assert_eq!(
            result.text_store.resolve(result.tokens[0].text_id).unwrap(),
            "a"
        );
    }

    #[test]
    fn edge_process_called_twice_is_reusable() {
        let proc = processor_lc_ws();
        let r1 = proc.process("Hello World").unwrap();
        let r2 = proc.process("Foo Bar").unwrap();
        assert_eq!(r1.tokens.len(), 2);
        assert_eq!(r2.tokens.len(), 2);
        assert_eq!(r1.normalized, "hello world");
        assert_eq!(r2.normalized, "foo bar");
    }

    #[test]
    fn edge_composed_mapping_present_with_normalizers() {
        let proc = processor_lc_ws();
        let result = proc.process("Hello").unwrap();
        assert!(result.composed_mapping.is_some());
    }
}
