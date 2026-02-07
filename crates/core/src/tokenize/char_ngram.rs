//! Character n-gram tokenizer: computes n-grams over grapheme clusters.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};

use core::fmt;

use crate::char_mapping::CharMapping;
use crate::error::TokenizeError;
use crate::span::Span;
use crate::text_store::TextStoreBuilder;
use crate::token::{Token, TokenKind};

use super::Tokenizer;

/// Computes character-level n-grams over grapheme clusters.
///
/// Wraps a base tokenizer (default: `CharTokenizer`) per the locked
/// composability decision. N-gram text is sliced directly from the source
/// text since grapheme clusters are contiguous.
pub struct CharNGramTokenizer {
    base: Box<dyn Tokenizer>,
    n: usize,
    name: String,
}

impl fmt::Debug for CharNGramTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharNGramTokenizer")
            .field("n", &self.n)
            .field("name", &self.name)
            .finish()
    }
}

impl CharNGramTokenizer {
    /// Create a new char n-gram tokenizer using `CharTokenizer` as base.
    pub fn new(n: usize) -> Self {
        let base = Box::new(super::char_tokenizer::CharTokenizer);
        let name = format!("char_{n}gram");
        Self { base, n, name }
    }

    /// Create with a custom base tokenizer.
    pub fn with_base(base: Box<dyn Tokenizer>, n: usize) -> Self {
        let name = format!("char_{n}gram");
        Self { base, n, name }
    }

    /// Create with a custom name.
    pub fn with_name(base: Box<dyn Tokenizer>, n: usize, name: String) -> Self {
        Self { base, n, name }
    }
}

impl Tokenizer for CharNGramTokenizer {
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

            // Slice directly from source text — grapheme clusters are contiguous
            let start = span.start() as usize;
            let end = span.end() as usize;
            let ngram_text = &text[start..end];
            let text_id = store.intern(ngram_text);
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

    fn char_bigram(text: &str) -> (Vec<Token>, TextStoreBuilder) {
        let ngram = CharNGramTokenizer::new(2);
        let mut store = TextStoreBuilder::new();
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = ngram.tokenize(text, &mapping, &mut store).unwrap();
        (tokens, store)
    }

    fn char_ngram(text: &str, n: usize) -> (Vec<Token>, TextStoreBuilder) {
        let ngram = CharNGramTokenizer::new(n);
        let mut store = TextStoreBuilder::new();
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = ngram.tokenize(text, &mapping, &mut store).unwrap();
        (tokens, store)
    }

    #[test]
    fn bigram_ascii() {
        let (tokens, store) = char_bigram("abc");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "ab");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "bc");
    }

    #[test]
    fn trigram_ascii() {
        let (tokens, store) = char_ngram("abcd", 3);
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "abc");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "bcd");
    }

    #[test]
    fn unigram_chars() {
        let (tokens, store) = char_ngram("abc", 1);
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "a");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "b");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "c");
    }

    #[test]
    fn n_greater_than_graphemes() {
        let (tokens, _) = char_ngram("abc", 5);
        assert!(tokens.is_empty());
    }

    #[test]
    fn n_equals_graphemes() {
        let (tokens, store) = char_ngram("abc", 3);
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "abc");
    }

    #[test]
    fn single_char_bigram() {
        let (tokens, _) = char_ngram("a", 2);
        assert!(tokens.is_empty());
    }

    #[test]
    fn cjk_bigram() {
        // Each CJK char is 3 bytes
        let (tokens, store) = char_bigram("你好世");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "你好");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "好世");
        // Byte spans: 你=0..3, 好=3..6, 世=6..9
        assert_eq!(tokens[0].span, Span::new(0, 6));
        assert_eq!(tokens[1].span, Span::new(3, 9));
    }

    #[test]
    fn emoji_bigram() {
        // Each emoji is 4 bytes
        let (tokens, store) = char_bigram("😀😁😂");
        assert_eq!(tokens.len(), 2);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "😀😁");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "😁😂");
        assert_eq!(tokens[0].span, Span::new(0, 8));
        assert_eq!(tokens[1].span, Span::new(4, 12));
    }

    #[test]
    fn span_correctness() {
        let (tokens, _) = char_bigram("abc");
        assert_eq!(tokens[0].span, Span::new(0, 2));
        assert_eq!(tokens[1].span, Span::new(1, 3));
    }

    #[test]
    fn empty_returns_empty() {
        let ngram = CharNGramTokenizer::new(2);
        let mapping = CharMapping::identity(1).unwrap();
        let mut store = TextStoreBuilder::new();
        let tokens = ngram.tokenize("", &mapping, &mut store).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn name_default() {
        let ngram = CharNGramTokenizer::new(2);
        assert_eq!(ngram.name(), "char_2gram");
    }

    #[test]
    fn name_custom() {
        let base = Box::new(super::super::char_tokenizer::CharTokenizer);
        let ngram = CharNGramTokenizer::with_name(base, 2, "my_ngram".into());
        assert_eq!(ngram.name(), "my_ngram");
    }

    #[test]
    fn mixed_width_chars() {
        // "aé" where é is 2 bytes -> 2 graphemes -> 1 bigram
        let (tokens, store) = char_bigram("a\u{00E9}");
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "a\u{00E9}");
        assert_eq!(tokens[0].span, Span::new(0, 3)); // 'a'=1 byte + 'é'=2 bytes
    }

    #[test]
    fn combining_char_bigram() {
        // "e\u{0301}a" = 2 graphemes: "é" (e + combining acute) and "a"
        let text = "e\u{0301}a";
        let (tokens, store) = char_bigram(text);
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), text);
    }

    #[test]
    fn all_tokens_regular() {
        let (tokens, _) = char_bigram("abcd");
        for tok in &tokens {
            assert_eq!(tok.kind, TokenKind::Regular);
        }
    }

    #[test]
    fn deduplication() {
        let (tokens, _) = char_bigram("aba");
        // "ab" and "ba" are different
        assert_ne!(tokens[0].text_id, tokens[1].text_id);

        // But "ab" in "abab" should dedup
        let (tokens2, _) = char_bigram("abab");
        assert_eq!(tokens2[0].text_id, tokens2[2].text_id); // both "ab"
    }

    // ── Edge case tests (02.1-03) ────────────────────────────────────

    #[test]
    fn edge_multi_byte_only_bigram() {
        let (tokens, store) = char_bigram("\u{00E9}\u{00E0}");
        assert_eq!(tokens.len(), 1);
        assert_eq!(
            store.resolve(tokens[0].text_id).unwrap(),
            "\u{00E9}\u{00E0}"
        );
        assert_eq!(tokens[0].span, Span::new(0, 4));
    }

    #[test]
    fn edge_zwj_emoji_in_ngrams() {
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        let text = format!("a{}", family);
        let (tokens, store) = char_bigram(&text);
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), &text);
        assert_eq!(tokens[0].span, Span::new(0, text.len() as u32));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use unicode_segmentation::UnicodeSegmentation;

    proptest! {
        #[test]
        fn ngram_count_formula(s in "\\PC{1,50}", n in 1usize..4) {
            let grapheme_count = s.graphemes(true).count();
            let ngram = CharNGramTokenizer::new(n);
            let mapping = CharMapping::identity(s.len() as u32).unwrap();
            let mut store = TextStoreBuilder::new();
            let tokens = ngram.tokenize(&s, &mapping, &mut store).unwrap();
            let expected = if grapheme_count >= n { grapheme_count - n + 1 } else { 0 };
            prop_assert_eq!(tokens.len(), expected);
        }

        #[test]
        fn spans_within_bounds(s in "\\PC{2,50}") {
            let (tokens, _) = {
                let ngram = CharNGramTokenizer::new(2);
                let mapping = CharMapping::identity(s.len() as u32).unwrap();
                let mut store = TextStoreBuilder::new();
                let tokens = ngram.tokenize(&s, &mapping, &mut store).unwrap();
                (tokens, store)
            };
            for tok in &tokens {
                prop_assert!(tok.span.end() as usize <= s.len());
                prop_assert!(tok.span.start() < tok.span.end());
            }
        }
    }
}
