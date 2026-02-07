//! Word tokenizer: splits on whitespace boundaries.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::char_mapping::CharMapping;
use crate::error::TokenizeError;
use crate::span::Span;
use crate::text_store::TextStoreBuilder;
use crate::token::{Token, TokenKind};

use super::Tokenizer;

/// Tokenizes text into words by splitting on whitespace.
///
/// Punctuation is NOT split from words — "hello," is a single token.
/// This is a locked design decision per the plan.
#[derive(Debug, Clone, Copy, Default)]
pub struct WordTokenizer;

impl Tokenizer for WordTokenizer {
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
        let mut word_start: Option<usize> = None;

        for (byte_offset, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if let Some(start) = word_start.take() {
                    let word = &text[start..byte_offset];
                    let token = make_token(
                        word,
                        start as u32,
                        byte_offset as u32,
                        mapping,
                        store,
                        false,
                    )?;
                    tokens.push(token);
                }
            } else if word_start.is_none() {
                word_start = Some(byte_offset);
            }
        }

        // Flush last word
        if let Some(start) = word_start {
            let word = &text[start..];
            let token = make_token(word, start as u32, text.len() as u32, mapping, store, true)?;
            tokens.push(token);
        }

        Ok(tokens)
    }

    fn name(&self) -> &str {
        "word"
    }
}

/// Create a Token from a word slice, mapping normalized positions to original.
fn make_token(
    word: &str,
    norm_start: u32,
    norm_end: u32,
    mapping: &CharMapping,
    store: &mut TextStoreBuilder,
    is_last: bool,
) -> Result<Token, TokenizeError> {
    let orig_start = mapping
        .to_original(norm_start)
        .map_err(|_| TokenizeError::Failed("CharMapping lookup failed".into()))?;

    // For end position: if this is the last token, use original text length
    // to handle byte-length changes from case folding (e.g., Ⱥ 2 bytes → ⱥ 3 bytes).
    // Otherwise, map the normalized end position to original.
    let orig_end = if is_last {
        mapping.original_len()
    } else {
        mapping
            .to_original(norm_end)
            .map_err(|_| TokenizeError::Failed("CharMapping lookup failed".into()))?
    };

    let id = store.intern(word);
    Ok(Token::new(
        id,
        Span::new(orig_start, orig_end),
        TokenKind::Regular,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize(text: &str) -> (Vec<Token>, TextStoreBuilder) {
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let mut store = TextStoreBuilder::new();
        let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
        (tokens, store)
    }

    #[test]
    fn simple_words() {
        let (tokens, store) = tokenize("hello world");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "world");
    }

    #[test]
    fn spans_are_correct() {
        let (tokens, _) = tokenize("hello world");
        assert_eq!(tokens[0].span, Span::new(0, 5));
        assert_eq!(tokens[1].span, Span::new(6, 11));
    }

    #[test]
    fn multiple_spaces() {
        let (tokens, store) = tokenize("a  b   c");
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "a");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "b");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "c");
    }

    #[test]
    fn tabs_and_newlines() {
        let (tokens, store) = tokenize("a\tb\nc");
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "a");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "b");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "c");
    }

    #[test]
    fn punctuation_stays_with_word() {
        let (tokens, store) = tokenize("hello, world!");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello,");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "world!");
    }

    #[test]
    fn cjk_words() {
        // CJK has no whitespace between characters, so entire string is one token
        let (tokens, store) = tokenize("你好世界");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "你好世界");
    }

    #[test]
    fn cjk_with_spaces() {
        let (tokens, store) = tokenize("你好 世界");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "你好");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "世界");
    }

    #[test]
    fn emoji_word() {
        let (tokens, store) = tokenize("hello 😀🎉 world");
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "😀🎉");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "world");
    }

    #[test]
    fn empty_input_returns_empty() {
        let mapping = CharMapping::identity(1).unwrap();
        let mut store = TextStoreBuilder::new();
        let tokens = WordTokenizer.tokenize("", &mapping, &mut store).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn single_word() {
        let (tokens, store) = tokenize("hello");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello");
        assert_eq!(tokens[0].span, Span::new(0, 5));
    }

    #[test]
    fn leading_trailing_whitespace() {
        let (tokens, store) = tokenize("  hello  ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello");
    }

    #[test]
    fn all_whitespace_returns_empty() {
        let mapping = CharMapping::identity(3).unwrap();
        let mut store = TextStoreBuilder::new();
        let tokens = WordTokenizer.tokenize("   ", &mapping, &mut store).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn all_tokens_are_regular() {
        let (tokens, _) = tokenize("hello world");
        for tok in &tokens {
            assert_eq!(tok.kind, TokenKind::Regular);
        }
    }

    #[test]
    fn deduplication_in_store() {
        let (tokens, store) = tokenize("the cat and the dog");
        // "the" appears twice -> same StringId
        assert_eq!(tokens[0].text_id, tokens[3].text_id);
        assert_eq!(store.len(), 4); // "the", "cat", "and", "dog"
    }

    #[test]
    fn name() {
        assert_eq!(WordTokenizer.name(), "word");
    }

    // ── Edge case tests (02.1-03) ────────────────────────────────────

    #[test]
    fn edge_punctuation_only() {
        let (tokens, store) = tokenize("!!!");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "!!!");
    }

    #[test]
    fn edge_trailing_space_ignored() {
        let (tokens, store) = tokenize("hello ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello");
    }

    #[test]
    fn edge_leading_space_ignored() {
        let (tokens, store) = tokenize(" hello");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello");
    }

    #[test]
    fn edge_tab_separator() {
        let (tokens, store) = tokenize("hello\tworld");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "world");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn tokens_reconstruct_text_with_spaces(s in "[a-z]{1,10}( [a-z]{1,10}){0,5}") {
            let mapping = CharMapping::identity(s.len() as u32).unwrap();
            let mut store = TextStoreBuilder::new();
            let tokens = WordTokenizer.tokenize(&s, &mapping, &mut store).unwrap();
            let reconstructed: String = tokens
                .iter()
                .map(|t| store.resolve(t.text_id).unwrap())
                .collect::<Vec<_>>()
                .join(" ");
            prop_assert_eq!(s.as_str(), reconstructed.as_str());
        }

        #[test]
        fn word_count_matches_split(s in "[a-z ]{1,50}") {
            let expected_count = s.split_whitespace().count();
            let mapping = CharMapping::identity(s.len() as u32).unwrap();
            let mut store = TextStoreBuilder::new();
            let tokens = WordTokenizer.tokenize(&s, &mapping, &mut store).unwrap();
            prop_assert_eq!(tokens.len(), expected_count);
        }
    }
}
