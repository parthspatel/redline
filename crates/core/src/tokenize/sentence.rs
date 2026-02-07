//! Sentence tokenizer using UAX#29 sentence boundaries.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use unicode_segmentation::UnicodeSegmentation;

use crate::char_mapping::CharMapping;
use crate::error::TokenizeError;
use crate::span::Span;
use crate::text_store::TextStoreBuilder;
use crate::token::{Token, TokenKind};

use super::Tokenizer;

/// Tokenizes text into sentences using UAX#29 sentence boundary rules.
///
/// Uses `unicode-segmentation` for boundary detection — no hand-rolled logic.
/// Filters segments to those containing at least one alphanumeric character,
/// and trims trailing whitespace from token text.
#[derive(Debug, Clone, Copy, Default)]
pub struct SentenceTokenizer;

impl Tokenizer for SentenceTokenizer {
    fn tokenize(
        &self,
        text: &str,
        _mapping: &CharMapping,
        store: &mut TextStoreBuilder,
    ) -> Result<Vec<Token>, TokenizeError> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let mut tokens = Vec::new();

        for (byte_offset, segment) in text.split_sentence_bound_indices() {
            // Only include segments with at least one alphanumeric character
            if !segment.chars().any(|c| c.is_alphanumeric()) {
                continue;
            }

            let trimmed = segment.trim_end();
            if trimmed.is_empty() {
                continue;
            }

            let text_id = store.intern(trimmed);
            let span = Span::new(byte_offset as u32, (byte_offset + trimmed.len()) as u32);
            tokens.push(Token::new(text_id, span, TokenKind::Regular));
        }

        Ok(tokens)
    }

    fn name(&self) -> &str {
        "sentence"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize_sentences(text: &str) -> (Vec<Token>, TextStoreBuilder) {
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let mut store = TextStoreBuilder::new();
        let tokens = SentenceTokenizer
            .tokenize(text, &mapping, &mut store)
            .unwrap();
        (tokens, store)
    }

    #[test]
    fn single_sentence() {
        let (tokens, store) = tokenize_sentences("Hello world.");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "Hello world.");
    }

    #[test]
    fn two_sentences() {
        let (tokens, store) = tokenize_sentences("Hello. World.");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "Hello.");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "World.");
    }

    #[test]
    fn question_and_statement() {
        let (tokens, store) = tokenize_sentences("How are you? I am fine.");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "How are you?");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "I am fine.");
    }

    #[test]
    fn exclamation() {
        let (tokens, store) = tokenize_sentences("Wow! That's great.");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "Wow!");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "That's great.");
    }

    #[test]
    fn decimal_not_split() {
        let (tokens, store) = tokenize_sentences("The value is 3.14 approximately.");
        assert_eq!(tokens.len(), 1);
        assert_eq!(
            store.resolve(tokens[0].text_id).unwrap(),
            "The value is 3.14 approximately."
        );
    }

    #[test]
    fn abbreviation_handling() {
        // UAX#29 behavior — abbreviation handling depends on Unicode rules, not custom logic.
        // unicode-segmentation splits on "Dr." as a sentence boundary.
        let (tokens, store) = tokenize_sentences("Dr. Smith went home.");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "Dr.");
        assert_eq!(
            store.resolve(tokens[1].text_id).unwrap(),
            "Smith went home."
        );
    }

    #[test]
    fn empty_returns_empty() {
        let mapping = CharMapping::identity(1).unwrap();
        let mut store = TextStoreBuilder::new();
        let tokens = SentenceTokenizer
            .tokenize("", &mapping, &mut store)
            .unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn whitespace_only_returns_empty() {
        let (tokens, _) = tokenize_sentences("   ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn cjk_sentence() {
        let (tokens, store) = tokenize_sentences("你好。世界。");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "你好。");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "世界。");
    }

    #[test]
    fn name_returns_sentence() {
        assert_eq!(SentenceTokenizer.name(), "sentence");
    }

    // ── Edge case tests (02.1-03) ────────────────────────────────────

    #[test]
    fn edge_no_sentence_boundary_no_period() {
        let (tokens, store) = tokenize_sentences("hello world");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello world");
    }

    #[test]
    fn edge_single_word_no_punctuation() {
        let (tokens, store) = tokenize_sentences("Hello");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "Hello");
    }

    #[test]
    fn tokens_interned() {
        let (tokens, _) = tokenize_sentences("Hello. Hello.");
        // Same sentence text shares StringId
        assert_eq!(tokens[0].text_id, tokens[1].text_id);
    }

    #[test]
    fn spans_cover_text() {
        let text = "Hello. World.";
        let (tokens, _) = tokenize_sentences(text);
        for tok in &tokens {
            assert!(tok.span.end() as usize <= text.len());
            assert!(tok.span.start() < tok.span.end());
        }
        // Non-overlapping
        for pair in tokens.windows(2) {
            assert!(pair[0].span.end() <= pair[1].span.start());
        }
    }

    #[test]
    fn all_tokens_are_regular() {
        let (tokens, _) = tokenize_sentences("Hello. World.");
        for tok in &tokens {
            assert_eq!(tok.kind, TokenKind::Regular);
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn spans_within_bounds(s in "\\PC{1,100}") {
            let mapping = CharMapping::identity(s.len() as u32).unwrap();
            let mut store = TextStoreBuilder::new();
            let tokens = SentenceTokenizer.tokenize(&s, &mapping, &mut store).unwrap();
            for tok in &tokens {
                prop_assert!(tok.span.end() as usize <= s.len());
            }
        }
    }
}
