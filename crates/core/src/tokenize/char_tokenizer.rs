//! Character (grapheme cluster) tokenizer.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use unicode_segmentation::UnicodeSegmentation;

use crate::char_mapping::CharMapping;
use crate::error::TokenizeError;
use crate::span::Span;
use crate::text_store::TextStoreBuilder;
use crate::token::{Token, TokenKind};

use super::Tokenizer;

/// Tokenizes text into individual Extended Grapheme Clusters.
///
/// Uses `unicode-segmentation` to correctly handle multi-codepoint graphemes
/// like emoji sequences and combining character sequences.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharTokenizer;

impl Tokenizer for CharTokenizer {
    fn tokenize(
        &self,
        text: &str,
        mapping: &CharMapping,
        store: &mut TextStoreBuilder,
    ) -> Result<Vec<Token>, TokenizeError> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let mut tokens = Vec::new();

        for (byte_offset, grapheme) in text.grapheme_indices(true) {
            let start = byte_offset as u32;
            let end = start + grapheme.len() as u32;

            // Map normalized positions back to original positions for the span
            let orig_start = mapping
                .to_original(start)
                .map_err(|_| TokenizeError::Failed("CharMapping lookup failed".into()))?;
            let orig_end = if end as usize == text.len() {
                // For the last grapheme, use the original text's byte length
                // (grapheme.len() may differ from original byte length due to case folding)
                mapping.original_len()
            } else {
                mapping
                    .to_original(end)
                    .map_err(|_| TokenizeError::Failed("CharMapping lookup failed".into()))?
            };

            let id = store.intern(grapheme);
            tokens.push(Token::new(
                id,
                Span::new(orig_start, orig_end),
                TokenKind::Regular,
            ));
        }

        Ok(tokens)
    }

    fn name(&self) -> &str {
        "char"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize(text: &str) -> (Vec<Token>, TextStoreBuilder) {
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let mut store = TextStoreBuilder::new();
        let tokens = CharTokenizer.tokenize(text, &mapping, &mut store).unwrap();
        (tokens, store)
    }

    #[test]
    fn ascii_chars() {
        let (tokens, store) = tokenize("abc");
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "a");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "b");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "c");
    }

    #[test]
    fn spans_are_correct() {
        let (tokens, _) = tokenize("abc");
        assert_eq!(tokens[0].span, Span::new(0, 1));
        assert_eq!(tokens[1].span, Span::new(1, 2));
        assert_eq!(tokens[2].span, Span::new(2, 3));
    }

    #[test]
    fn unicode_multibyte() {
        let (tokens, store) = tokenize("über");
        assert_eq!(tokens.len(), 4);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "ü"); // 2 bytes
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "b");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "e");
        assert_eq!(store.resolve(tokens[3].text_id).unwrap(), "r");
    }

    #[test]
    fn emoji_grapheme_clusters() {
        // Family emoji is a single grapheme cluster
        let text = "a👨‍👩‍👧b";
        let (tokens, store) = tokenize(text);
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "a");
        assert_eq!(
            store.resolve(tokens[1].text_id).unwrap(),
            "👨\u{200d}👩\u{200d}👧"
        );
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "b");
    }

    #[test]
    fn cjk_characters() {
        let (tokens, store) = tokenize("你好");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "你");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "好");
    }

    #[test]
    fn empty_input_returns_empty() {
        let mapping = CharMapping::identity(1).unwrap(); // dummy, won't be used
        let mut store = TextStoreBuilder::new();
        let tokens = CharTokenizer.tokenize("", &mapping, &mut store).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn single_char() {
        let (tokens, store) = tokenize("x");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "x");
        assert_eq!(tokens[0].span, Span::new(0, 1));
    }

    #[test]
    fn all_tokens_are_regular() {
        let (tokens, _) = tokenize("hello");
        for tok in &tokens {
            assert_eq!(tok.kind, TokenKind::Regular);
        }
    }

    #[test]
    fn deduplication_in_store() {
        let (tokens, store) = tokenize("aab");
        // 'a' interned once, reused
        assert_eq!(tokens[0].text_id, tokens[1].text_id);
        assert_ne!(tokens[0].text_id, tokens[2].text_id);
        assert_eq!(store.len(), 2); // "a" and "b"
    }

    #[test]
    fn name() {
        assert_eq!(CharTokenizer.name(), "char");
    }

    // ── Edge case tests (02.1-03) ────────────────────────────────────

    #[test]
    fn edge_single_multibyte_char() {
        let (tokens, store) = tokenize("\u{00E9}");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "\u{00E9}");
        assert_eq!(tokens[0].span, Span::new(0, 2));
    }

    #[test]
    fn edge_zwj_emoji_standalone() {
        let text = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        let (tokens, store) = tokenize(text);
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), text);
        assert_eq!(tokens[0].span, Span::new(0, text.len() as u32));
    }

    #[test]
    fn edge_very_long_string_1000_chars() {
        let long: String = core::iter::repeat('a').take(1000).collect();
        let (tokens, _) = tokenize(&long);
        assert_eq!(tokens.len(), 1000);
        assert_eq!(tokens[0].span, Span::new(0, 1));
        assert_eq!(tokens[999].span, Span::new(999, 1000));
    }

    #[test]
    fn edge_whitespace_chars_are_graphemes() {
        let (tokens, store) = tokenize("   ");
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), " ");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn token_count_equals_grapheme_count(s in "\\PC{1,100}") {
            let grapheme_count = s.graphemes(true).count();
            let mapping = CharMapping::identity(s.len() as u32).unwrap();
            let mut store = TextStoreBuilder::new();
            let tokens = CharTokenizer.tokenize(&s, &mapping, &mut store).unwrap();
            prop_assert_eq!(tokens.len(), grapheme_count);
        }

        #[test]
        fn tokens_reconstruct_text(s in "[a-zA-Z0-9 ]{1,50}") {
            let mapping = CharMapping::identity(s.len() as u32).unwrap();
            let mut store = TextStoreBuilder::new();
            let tokens = CharTokenizer.tokenize(&s, &mapping, &mut store).unwrap();
            let reconstructed: String = tokens
                .iter()
                .map(|t| store.resolve(t.text_id).unwrap())
                .collect();
            prop_assert_eq!(s.as_str(), reconstructed.as_str());
        }
    }
}
