//! Word n-gram tokenizer: computes n-grams over a base tokenizer's output.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};

use core::fmt;

use crate::char_mapping::CharMapping;
use crate::error::TokenizeError;
use crate::span::Span;
use crate::text_store::TextStoreBuilder;
use crate::token::{Token, TokenKind};

use super::Tokenizer;

/// Computes word-level n-grams by wrapping a base tokenizer.
///
/// Runs the base tokenizer first, then slides a window of size `n` over
/// the resulting tokens. N-gram text is space-joined from base token texts.
pub struct WordNGramTokenizer {
    base: Box<dyn Tokenizer>,
    n: usize,
    name: String,
}

impl fmt::Debug for WordNGramTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WordNGramTokenizer")
            .field("n", &self.n)
            .field("name", &self.name)
            .finish()
    }
}

impl WordNGramTokenizer {
    /// Create a new word n-gram tokenizer wrapping a base tokenizer.
    pub fn new(base: Box<dyn Tokenizer>, n: usize) -> Self {
        let name = format!("word_{n}gram");
        Self { base, n, name }
    }

    /// Create with a custom name.
    pub fn with_name(base: Box<dyn Tokenizer>, n: usize, name: String) -> Self {
        Self { base, n, name }
    }
}

impl Tokenizer for WordNGramTokenizer {
    fn tokenize(
        &self,
        text: &str,
        mapping: &CharMapping,
        store: &mut TextStoreBuilder,
    ) -> Result<Vec<Token>, TokenizeError> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let base_tokens = self.base.tokenize(text, mapping, store)?;

        if base_tokens.len() < self.n {
            return Ok(vec![]);
        }

        let mut ngram_tokens = Vec::with_capacity(base_tokens.len() - self.n + 1);

        for window in base_tokens.windows(self.n) {
            let span = Span::new(
                window.first().unwrap().span.start(),
                window.last().unwrap().span.end(),
            );

            let ngram_text: String = window
                .iter()
                .map(|t| store.resolve(t.text_id).unwrap())
                .collect::<Vec<&str>>()
                .join(" ");

            let text_id = store.intern(&ngram_text);
            ngram_tokens.push(Token::new(text_id, span, TokenKind::Regular));
        }

        Ok(ngram_tokens)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bigram_tokenize(text: &str) -> (Vec<Token>, TextStoreBuilder) {
        let base = Box::new(crate::tokenize::word::WordTokenizer);
        let ngram = WordNGramTokenizer::new(base, 2);
        let mut store = TextStoreBuilder::new();
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = ngram.tokenize(text, &mapping, &mut store).unwrap();
        (tokens, store)
    }

    fn ngram_tokenize(text: &str, n: usize) -> (Vec<Token>, TextStoreBuilder) {
        let base = Box::new(crate::tokenize::word::WordTokenizer);
        let ngram = WordNGramTokenizer::new(base, n);
        let mut store = TextStoreBuilder::new();
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = ngram.tokenize(text, &mapping, &mut store).unwrap();
        (tokens, store)
    }

    #[test]
    fn bigram_basic() {
        let (tokens, store) = bigram_tokenize("hello world foo");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello world");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "world foo");
    }

    #[test]
    fn trigram_basic() {
        let (tokens, store) = ngram_tokenize("a b c d", 3);
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "a b c");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "b c d");
    }

    #[test]
    fn unigram() {
        let (tokens, store) = ngram_tokenize("hello world", 1);
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "world");
    }

    #[test]
    fn n_greater_than_tokens() {
        let (tokens, _) = ngram_tokenize("hello world", 5);
        assert!(tokens.is_empty());
    }

    #[test]
    fn n_equals_tokens() {
        let (tokens, store) = ngram_tokenize("a b c", 3);
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "a b c");
    }

    #[test]
    fn single_word_bigram() {
        let (tokens, _) = ngram_tokenize("hello", 2);
        assert!(tokens.is_empty());
    }

    #[test]
    fn span_covers_gap() {
        let (tokens, _) = bigram_tokenize("hello world");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].span, Span::new(0, 11));
    }

    #[test]
    fn ngram_text_interned() {
        let (tokens, _) = bigram_tokenize("a b a b");
        // "a b" appears at positions 0 and 2 in the bigram sequence
        assert_eq!(tokens[0].text_id, tokens[2].text_id);
    }

    #[test]
    fn empty_returns_empty() {
        let base = Box::new(crate::tokenize::word::WordTokenizer);
        let ngram = WordNGramTokenizer::new(base, 2);
        let mapping = CharMapping::identity(1).unwrap();
        let mut store = TextStoreBuilder::new();
        let tokens = ngram.tokenize("", &mapping, &mut store).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn name_default() {
        let base = Box::new(crate::tokenize::word::WordTokenizer);
        let ngram = WordNGramTokenizer::new(base, 2);
        assert_eq!(ngram.name(), "word_2gram");
    }

    #[test]
    fn name_custom() {
        let base = Box::new(crate::tokenize::word::WordTokenizer);
        let ngram = WordNGramTokenizer::with_name(base, 2, "my_ngram".into());
        assert_eq!(ngram.name(), "my_ngram");
    }

    #[test]
    fn all_tokens_regular_kind() {
        let (tokens, _) = bigram_tokenize("hello world foo");
        for tok in &tokens {
            assert_eq!(tok.kind, TokenKind::Regular);
        }
    }

    // ── Edge case tests (02.1-03) ────────────────────────────────────

    #[test]
    fn edge_n2_on_two_tokens_returns_one_bigram() {
        let (tokens, store) = ngram_tokenize("hello world", 2);
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "hello world");
    }

    #[test]
    fn edge_space_joined_text_verified() {
        let (tokens, store) = ngram_tokenize("the quick brown fox", 2);
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "the quick");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "quick brown");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "brown fox");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn ngram_count_formula(s in "[a-z]{1,5}( [a-z]{1,5}){0,8}", n in 1usize..5) {
            let word_count = s.split_whitespace().count();
            let base = Box::new(crate::tokenize::word::WordTokenizer);
            let ngram = WordNGramTokenizer::new(base, n);
            let mapping = CharMapping::identity(s.len() as u32).unwrap();
            let mut store = TextStoreBuilder::new();
            let tokens = ngram.tokenize(&s, &mapping, &mut store).unwrap();
            let expected = if word_count >= n { word_count - n + 1 } else { 0 };
            prop_assert_eq!(tokens.len(), expected);
        }

        #[test]
        fn spans_within_bounds(s in "[a-z]{1,5}( [a-z]{1,5}){1,5}") {
            let (tokens, _) = {
                let base = Box::new(crate::tokenize::word::WordTokenizer);
                let ngram = WordNGramTokenizer::new(base, 2);
                let mapping = CharMapping::identity(s.len() as u32).unwrap();
                let mut store = TextStoreBuilder::new();
                let tokens = ngram.tokenize(&s, &mapping, &mut store).unwrap();
                (tokens, store)
            };
            for tok in &tokens {
                prop_assert!(tok.span.end() as usize <= s.len());
            }
        }
    }
}
