//! Integration test matrix: Normalizer × Tokenizer combinations.
//!
//! Tests semantically meaningful normalizer-tokenizer combos through the
//! TextProcessor pipeline, verifying correct output and CharMapping integrity.

mod common;

use redline_core::normalize::{
    Lowercase, Normalizer, RemoveDiacritics, RemoveDigits, RemovePunctuation, UnicodeNormalizer,
    WhitespaceNormalizer,
};
use redline_core::process::{ExecutionMode, TextProcessor};
use redline_core::tokenize::{
    CharNGramTokenizer, CharTokenizer, SentenceTokenizer, Tokenizer, WordNGramTokenizer,
    WordTokenizer,
};

// ── Test input constants (from shared fixtures) ─────────────────────

use common::multilang::{
    ASCII, CJK, EMOJI, MIXED_SCRIPT as MIXED, RTL, UNICODE_ACCENTED as UNICODE, VIETNAMESE,
};

// ── Assertion helpers ────────────────────────────────────────────────

/// Validate pipeline for tokenizers whose spans reference ORIGINAL positions.
fn assert_pipeline_valid(
    input: &str,
    normalizers: Vec<Box<dyn Normalizer>>,
    tokenizer: Box<dyn Tokenizer>,
) {
    let norm_count = normalizers.len();
    let processor = TextProcessor::new(normalizers, tokenizer, ExecutionMode::All).unwrap();
    let result = processor.process(input).unwrap();

    assert!(!result.tokens.is_empty(), "tokens empty for: {:?}", input);
    assert!(!result.normalized.is_empty());

    for tok in &result.tokens {
        let start = tok.span.start() as usize;
        let end = tok.span.end() as usize;
        assert!(start <= end, "span start {start} > end {end}");
        assert!(
            end <= input.len(),
            "span end {end} > input len {}",
            input.len()
        );
    }

    if norm_count > 0 {
        assert!(result.composed_mapping.is_some());
    }
    assert_eq!(result.layers.len(), norm_count);

    for tok in &result.tokens {
        assert!(result.text_store.resolve(tok.text_id).is_ok());
    }
}

/// Validate pipeline for SentenceTokenizer (spans reference NORMALIZED positions).
fn assert_sentence_pipeline_valid(input: &str, normalizers: Vec<Box<dyn Normalizer>>) {
    let norm_count = normalizers.len();
    let processor =
        TextProcessor::new(normalizers, Box::new(SentenceTokenizer), ExecutionMode::All).unwrap();
    let result = processor.process(input).unwrap();

    assert!(!result.normalized.is_empty());

    for tok in &result.tokens {
        let start = tok.span.start() as usize;
        let end = tok.span.end() as usize;
        assert!(start <= end);
        assert!(end <= result.normalized.len());
    }

    if norm_count > 0 {
        assert!(result.composed_mapping.is_some());
    }
    assert_eq!(result.layers.len(), norm_count);

    for tok in &result.tokens {
        assert!(result.text_store.resolve(tok.text_id).is_ok());
    }
}

// ═══════════════════════════════════════════════════════════════════
// CharTokenizer combos
// ═══════════════════════════════════════════════════════════════════

mod char_tokenizer_combos {
    use super::*;

    #[test]
    fn lowercase_char() {
        assert_pipeline_valid(ASCII, vec![Box::new(Lowercase)], Box::new(CharTokenizer));
    }

    #[test]
    fn lowercase_whitespace_char() {
        assert_pipeline_valid(
            ASCII,
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(CharTokenizer),
        );
    }

    #[test]
    fn unicode_nfc_lowercase_char() {
        assert_pipeline_valid(
            UNICODE,
            vec![Box::new(UnicodeNormalizer::default()), Box::new(Lowercase)],
            Box::new(CharTokenizer),
        );
    }

    #[test]
    fn remove_diacritics_char() {
        assert_pipeline_valid(
            UNICODE,
            vec![Box::new(RemoveDiacritics)],
            Box::new(CharTokenizer),
        );
    }

    #[test]
    fn remove_punctuation_char() {
        assert_pipeline_valid(
            ASCII,
            vec![Box::new(RemovePunctuation)],
            Box::new(CharTokenizer),
        );
    }

    #[test]
    fn lowercase_whitespace_nfc_char() {
        assert_pipeline_valid(
            ASCII,
            vec![
                Box::new(Lowercase),
                Box::new(WhitespaceNormalizer),
                Box::new(UnicodeNormalizer::default()),
            ],
            Box::new(CharTokenizer),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// WordTokenizer combos
// ═══════════════════════════════════════════════════════════════════

mod word_tokenizer_combos {
    use super::*;

    #[test]
    fn lowercase_word() {
        assert_pipeline_valid(ASCII, vec![Box::new(Lowercase)], Box::new(WordTokenizer));
    }

    #[test]
    fn lowercase_whitespace_word() {
        assert_pipeline_valid(
            ASCII,
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
        );
    }

    #[test]
    fn remove_punctuation_lowercase_word() {
        assert_pipeline_valid(
            ASCII,
            vec![Box::new(RemovePunctuation), Box::new(Lowercase)],
            Box::new(WordTokenizer),
        );
    }

    #[test]
    fn remove_digits_lowercase_word() {
        assert_pipeline_valid(
            ASCII,
            vec![Box::new(RemoveDigits), Box::new(Lowercase)],
            Box::new(WordTokenizer),
        );
    }

    #[test]
    fn nfc_lowercase_whitespace_word() {
        assert_pipeline_valid(
            ASCII,
            vec![
                Box::new(UnicodeNormalizer::default()),
                Box::new(Lowercase),
                Box::new(WhitespaceNormalizer),
            ],
            Box::new(WordTokenizer),
        );
    }

    #[test]
    fn remove_diacritics_lowercase_word() {
        assert_pipeline_valid(
            UNICODE,
            vec![Box::new(RemoveDiacritics), Box::new(Lowercase)],
            Box::new(WordTokenizer),
        );
    }

    #[test]
    fn realistic_paragraph_lowercase_whitespace_word() {
        assert_pipeline_valid(
            common::TECHNICAL_PROSE,
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// SentenceTokenizer combos
// ═══════════════════════════════════════════════════════════════════

mod sentence_tokenizer_combos {
    use super::*;

    #[test]
    fn lowercase_sentence() {
        assert_sentence_pipeline_valid(ASCII, vec![Box::new(Lowercase)]);
    }

    #[test]
    fn whitespace_lowercase_sentence() {
        assert_sentence_pipeline_valid(
            ASCII,
            vec![Box::new(WhitespaceNormalizer), Box::new(Lowercase)],
        );
    }

    #[test]
    fn nfc_sentence() {
        assert_sentence_pipeline_valid(UNICODE, vec![Box::new(UnicodeNormalizer::default())]);
    }

    #[test]
    fn remove_punctuation_sentence() {
        assert_sentence_pipeline_valid(ASCII, vec![Box::new(RemovePunctuation)]);
    }

    #[test]
    fn realistic_paragraph_sentence() {
        assert_sentence_pipeline_valid(common::LITERARY_PROSE, vec![Box::new(Lowercase)]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// WordNGramTokenizer combos
// ═══════════════════════════════════════════════════════════════════

mod word_ngram_combos {
    use super::*;

    #[test]
    fn lowercase_whitespace_bigram() {
        assert_pipeline_valid(
            ASCII,
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordNGramTokenizer::new(Box::new(WordTokenizer), 2)),
        );
    }

    #[test]
    fn remove_punctuation_lowercase_bigram() {
        assert_pipeline_valid(
            ASCII,
            vec![Box::new(RemovePunctuation), Box::new(Lowercase)],
            Box::new(WordNGramTokenizer::new(Box::new(WordTokenizer), 2)),
        );
    }

    #[test]
    fn lowercase_trigram() {
        assert_pipeline_valid(
            ASCII,
            vec![Box::new(Lowercase)],
            Box::new(WordNGramTokenizer::new(Box::new(WordTokenizer), 3)),
        );
    }

    #[test]
    fn nfc_lowercase_bigram() {
        assert_pipeline_valid(
            UNICODE,
            vec![Box::new(UnicodeNormalizer::default()), Box::new(Lowercase)],
            Box::new(WordNGramTokenizer::new(Box::new(WordTokenizer), 2)),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// CharNGramTokenizer combos
// ═══════════════════════════════════════════════════════════════════

mod char_ngram_combos {
    use super::*;

    #[test]
    fn lowercase_char_bigram() {
        assert_pipeline_valid(
            "Hello World",
            vec![Box::new(Lowercase)],
            Box::new(CharNGramTokenizer::new(2)),
        );
    }

    #[test]
    fn lowercase_whitespace_char_trigram() {
        // Input with no multi-space so WhitespaceNormalizer is noop
        assert_pipeline_valid(
            "hello world test",
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(CharNGramTokenizer::new(3)),
        );
    }

    #[test]
    fn nfc_lowercase_char_bigram() {
        assert_pipeline_valid(
            "Hello World",
            vec![Box::new(UnicodeNormalizer::default()), Box::new(Lowercase)],
            Box::new(CharNGramTokenizer::new(2)),
        );
    }

    #[test]
    fn remove_diacritics_lowercase_char_bigram() {
        assert_pipeline_valid(
            "Hello World",
            vec![Box::new(RemoveDiacritics), Box::new(Lowercase)],
            Box::new(CharNGramTokenizer::new(2)),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Unicode-focused combos
// ═══════════════════════════════════════════════════════════════════

mod unicode_combos {
    use super::*;

    #[test]
    fn cjk_lowercase_whitespace_word() {
        assert_pipeline_valid(
            CJK,
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
        );
    }

    #[test]
    fn emoji_lowercase_char() {
        assert_pipeline_valid(EMOJI, vec![Box::new(Lowercase)], Box::new(CharTokenizer));
    }

    #[test]
    fn rtl_lowercase_word() {
        assert_pipeline_valid(RTL, vec![Box::new(Lowercase)], Box::new(WordTokenizer));
    }

    #[test]
    fn vietnamese_remove_diacritics_lowercase_word() {
        assert_pipeline_valid(
            VIETNAMESE,
            vec![Box::new(RemoveDiacritics), Box::new(Lowercase)],
            Box::new(WordTokenizer),
        );
    }

    #[test]
    fn mixed_script_nfc_lowercase_char() {
        assert_pipeline_valid(
            MIXED,
            vec![Box::new(UnicodeNormalizer::default()), Box::new(Lowercase)],
            Box::new(CharTokenizer),
        );
    }

    #[test]
    fn chinese_paragraph_word() {
        assert_pipeline_valid(
            common::multilang::CHINESE_PARAGRAPH,
            vec![Box::new(Lowercase)],
            Box::new(WordTokenizer),
        );
    }

    #[test]
    fn french_accented_remove_diacritics_word() {
        assert_pipeline_valid(
            common::multilang::ACCENTED_FRENCH,
            vec![Box::new(RemoveDiacritics), Box::new(Lowercase)],
            Box::new(WordTokenizer),
        );
    }
}
