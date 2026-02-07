//! Unicode edge case tests for Phase 2: Text Processing (TEXT-08)
//!
//! CJK, emoji (ZWJ, flags, skin tones, keycap), RTL (Arabic, Hebrew),
//! combining marks, and property-based testing of the full pipeline.

use redline_core::char_mapping::CharMapping;
use redline_core::normalize::diacritics::RemoveDiacritics;
use redline_core::normalize::lowercase::Lowercase;
use redline_core::normalize::unicode::UnicodeNormalizer;
use redline_core::normalize::whitespace::WhitespaceNormalizer;
use redline_core::process::{ExecutionMode, TextProcessor};
use redline_core::span::Span;
use redline_core::text_store::TextStoreBuilder;
use redline_core::tokenize::Tokenizer;
use redline_core::tokenize::char_tokenizer::CharTokenizer;
use redline_core::tokenize::word::WordTokenizer;

fn tokenize_chars(text: &str) -> (Vec<redline_core::Token>, TextStoreBuilder) {
    let mut store = TextStoreBuilder::new();
    let mapping = CharMapping::identity(text.len() as u32).unwrap();
    let tokens = CharTokenizer.tokenize(text, &mapping, &mut store).unwrap();
    (tokens, store)
}

// ── CJK Tests ────────────────────────────────────────────────────────

mod cjk {
    use super::*;

    #[test]
    fn char_tokenizer_chinese() {
        let (tokens, store) = tokenize_chars("你好世界");
        assert_eq!(tokens.len(), 4);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "你");
        assert_eq!(tokens[0].span, Span::new(0, 3));
        assert_eq!(tokens[1].span, Span::new(3, 6));
        assert_eq!(tokens[2].span, Span::new(6, 9));
        assert_eq!(tokens[3].span, Span::new(9, 12));
    }

    #[test]
    fn char_tokenizer_japanese() {
        let (tokens, store) = tokenize_chars("東京都");
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "東");
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "京");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "都");
    }

    #[test]
    fn word_tokenizer_cjk_no_space() {
        let mut store = TextStoreBuilder::new();
        let text = "你好世界";
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn word_tokenizer_cjk_with_space() {
        let mut store = TextStoreBuilder::new();
        let text = "你好 世界";
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn cjk_through_full_pipeline() {
        let processor = TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(CharTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        let result = processor.process("你好 世界").unwrap();
        // Lowercase is a noop for CJK, whitespace normalizes
        assert_eq!(result.normalized, "你好 世界");
        // CharTokenizer: 你, 好, ' ', 世, 界 = 5 graphemes
        assert_eq!(result.tokens.len(), 5);
    }
}

// ── Emoji Tests ──────────────────────────────────────────────────────

mod emoji {
    use super::*;

    #[test]
    fn zwj_family_emoji() {
        // 👨‍👩‍👧‍👦 is a single grapheme cluster via ZWJ
        let text = "👨\u{200d}👩\u{200d}👧\u{200d}👦";
        let (tokens, store) = tokenize_chars(text);
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), text);
    }

    #[test]
    fn flag_emoji() {
        // 🇺🇸 = U+1F1FA U+1F1F8 (regional indicators)
        let text = "🇺🇸";
        let (tokens, _) = tokenize_chars(text);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].span, Span::new(0, 8)); // 2 x 4 bytes
    }

    #[test]
    fn skin_tone_emoji() {
        // 👋🏽 = U+1F44B U+1F3FD
        let text = "👋🏽";
        let (tokens, _) = tokenize_chars(text);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].span, Span::new(0, 8)); // 2 x 4 bytes
    }

    #[test]
    fn multiple_emoji() {
        let (tokens, _) = tokenize_chars("😀😁😂🤣");
        assert_eq!(tokens.len(), 4);
        for (i, tok) in tokens.iter().enumerate() {
            assert_eq!(tok.span, Span::new(i as u32 * 4, (i as u32 + 1) * 4));
        }
    }

    #[test]
    fn mixed_text_and_emoji() {
        let mut store = TextStoreBuilder::new();
        let text = "hello 🌍 world";
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "🌍");
    }

    #[test]
    fn emoji_through_pipeline() {
        let processor = TextProcessor::new(
            vec![Box::new(Lowercase)],
            Box::new(CharTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        let result = processor.process("Hi 👨\u{200d}👩\u{200d}👧").unwrap();
        // Lowercase: "hi 👨‍👩‍👧"
        // CharTokenizer: 'h', 'i', ' ', family_emoji = 4 tokens
        assert_eq!(result.tokens.len(), 4);
    }
}

// ── RTL Tests ────────────────────────────────────────────────────────

mod rtl {
    use super::*;

    #[test]
    fn arabic_word_tokenizer() {
        let mut store = TextStoreBuilder::new();
        let text = "مرحبا بالعالم";
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
        assert_eq!(tokens.len(), 2);
        for token in &tokens {
            let start = token.span.start() as usize;
            let end = token.span.end() as usize;
            assert!(text.is_char_boundary(start));
            assert!(text.is_char_boundary(end));
        }
    }

    #[test]
    fn hebrew_char_tokenizer() {
        let (tokens, store) = tokenize_chars("שלום");
        assert_eq!(tokens.len(), 4);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "ש");
        // Hebrew chars are 2 bytes each
        assert_eq!(tokens[0].span, Span::new(0, 2));
        assert_eq!(tokens[1].span, Span::new(2, 4));
    }

    #[test]
    fn arabic_through_pipeline() {
        let processor = TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        let result = processor.process("مرحبا  بالعالم").unwrap();
        assert_eq!(result.tokens.len(), 2);
        // Whitespace should collapse the double space
        assert!(result.normalized.contains(" "));
        assert!(!result.normalized.contains("  "));
    }

    #[test]
    fn mixed_rtl_ltr() {
        let mut store = TextStoreBuilder::new();
        let text = "Hello مرحبا World";
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "Hello");
        assert_eq!(store.resolve(tokens[2].text_id).unwrap(), "World");
    }
}

// ── Combining Marks Tests ────────────────────────────────────────────

mod combining_marks {
    use super::*;

    #[test]
    fn combining_acute_accent() {
        // e + combining acute = 1 grapheme
        let text = "e\u{0301}";
        let (tokens, _) = tokenize_chars(text);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].span, Span::new(0, 3)); // 1 + 2 bytes
    }

    #[test]
    fn multiple_combining_marks() {
        // a + combining tilde + combining acute = 1 grapheme
        let text = "a\u{0303}\u{0301}";
        let (tokens, _) = tokenize_chars(text);
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn combining_marks_in_word() {
        let mut store = TextStoreBuilder::new();
        let text = "cafe\u{0301}";
        let mapping = CharMapping::identity(text.len() as u32).unwrap();
        let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(store.resolve(tokens[0].text_id).unwrap(), text);
    }

    #[test]
    fn diacritics_removal_through_pipeline() {
        let processor = TextProcessor::new(
            vec![
                Box::new(UnicodeNormalizer::default()),
                Box::new(RemoveDiacritics),
            ],
            Box::new(WordTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        let result = processor.process("r\u{00E9}sum\u{00E9}").unwrap();
        assert_eq!(result.normalized, "resume");
    }

    #[test]
    fn decomposed_then_diacritics_removal() {
        let processor = TextProcessor::new(
            vec![
                Box::new(UnicodeNormalizer::default()),
                Box::new(RemoveDiacritics),
            ],
            Box::new(CharTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        // e + combining acute -> NFC -> é -> remove diacritics -> e
        let result = processor.process("e\u{0301}").unwrap();
        assert_eq!(result.normalized, "e");
        assert_eq!(result.tokens.len(), 1);
    }
}

// ── CharTokenizer Pipeline Tests (TEXT-08 grapheme-aware) ────────────

mod char_tokenizer_pipeline {
    use super::*;

    #[test]
    fn cjk_through_normalizer_to_char_tokenizer() {
        let processor = TextProcessor::new(
            vec![Box::new(UnicodeNormalizer::default())],
            Box::new(CharTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        let result = processor.process("你好世界").unwrap();
        assert_eq!(result.tokens.len(), 4);
    }

    #[test]
    fn emoji_preserved_through_lowercase() {
        let processor = TextProcessor::new(
            vec![Box::new(Lowercase)],
            Box::new(CharTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        let result = processor.process("Hi 👨\u{200d}👩\u{200d}👧").unwrap();
        // 'h', 'i', ' ', family = 4 graphemes
        assert_eq!(result.tokens.len(), 4);
    }

    #[test]
    fn diacritics_then_char_tokenizer() {
        let processor = TextProcessor::new(
            vec![
                Box::new(UnicodeNormalizer::default()),
                Box::new(RemoveDiacritics),
            ],
            Box::new(CharTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        let result = processor.process("caf\u{00E9}").unwrap();
        assert_eq!(result.normalized, "cafe");
        assert_eq!(result.tokens.len(), 4);
    }

    #[test]
    fn rtl_through_pipeline_to_char_tokenizer() {
        let processor = TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(CharTokenizer),
            ExecutionMode::All,
        )
        .unwrap();

        let result = processor.process("שלום  עולם").unwrap();
        // "שלום עולם" after whitespace collapse
        // CharTokenizer: ש, ל, ו, ם, ' ', ע, ו, ל, ם = 9 graphemes
        assert_eq!(result.tokens.len(), 9);
    }
}

// ── Proptests ────────────────────────────────────────────────────────

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn pipeline_never_panics(input in "\\PC{1,100}") {
            let processor = TextProcessor::new(
                vec![
                    Box::new(Lowercase),
                    Box::new(WhitespaceNormalizer),
                    Box::new(UnicodeNormalizer::default()),
                ],
                Box::new(WordTokenizer),
                ExecutionMode::All,
            ).unwrap();

            let _ = processor.process(&input);
        }

        #[test]
        fn pipeline_token_spans_valid(input in "\\PC{1,50}") {
            let processor = TextProcessor::new(
                vec![Box::new(Lowercase)],
                Box::new(CharTokenizer),
                ExecutionMode::All,
            ).unwrap();

            if let Ok(result) = processor.process(&input) {
                for tok in &result.tokens {
                    let start = tok.span.start() as usize;
                    let end = tok.span.end() as usize;
                    prop_assert!(start <= end);
                    // Spans refer to original text positions (mapped through CharMapping)
                    prop_assert!(end <= input.len(), "end={} > input.len()={}, normalized.len()={}, input={:?}", end, input.len(), result.normalized.len(), input);
                }
            }
        }

        #[test]
        fn char_tokenizer_grapheme_count(input in "\\PC{1,50}") {
            use unicode_segmentation::UnicodeSegmentation;
            let processor = TextProcessor::new(
                vec![Box::new(Lowercase)],
                Box::new(CharTokenizer),
                ExecutionMode::All,
            ).unwrap();

            if let Ok(result) = processor.process(&input) {
                let expected = result.normalized.graphemes(true).count();
                prop_assert_eq!(result.tokens.len(), expected);
            }
        }
    }
}
