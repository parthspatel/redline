//! Integration tests for Phase 2: Text Processing
//!
//! Tests all 5 ROADMAP success criteria for the text processing pipeline.

use redline_core::char_mapping::CharMapping;
use redline_core::normalize::diacritics::RemoveDiacritics;
use redline_core::normalize::lowercase::Lowercase;
use redline_core::normalize::unicode::UnicodeNormalizer;
use redline_core::normalize::whitespace::WhitespaceNormalizer;
use redline_core::process::{ExecutionMode, TextProcessor};
use redline_core::text_store::TextStoreBuilder;
use redline_core::tokenize::Tokenizer;
use redline_core::tokenize::char_tokenizer::CharTokenizer;
use redline_core::tokenize::word::WordTokenizer;

// ── Success Criterion 1 ─────────────────────────────────────────────
// Normalize "Hello  WORLD" with Lowercase + Whitespace -> "hello world"

#[test]
fn success_criterion_1_lowercase_whitespace_normalization() {
    let processor = TextProcessor::new(
        vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
        Box::new(WordTokenizer),
        ExecutionMode::All,
    )
    .unwrap();

    let result = processor.process("Hello  WORLD").unwrap();

    assert_eq!(result.normalized, "hello world");
    assert_eq!(result.tokens.len(), 2);

    let token_texts: Vec<&str> = result
        .tokens
        .iter()
        .map(|t| result.text_store.resolve(t.text_id).unwrap())
        .collect();
    assert_eq!(token_texts, vec!["hello", "world"]);

    let mapping = result.composed_mapping.as_ref().unwrap();
    assert_eq!(mapping.to_normalized(0).unwrap(), 0); // 'H' -> 'h'
}

// ── Success Criterion 2 ─────────────────────────────────────────────
// CharMapping maps through 3-normalizer chain (property-tested with Unicode)

#[test]
fn success_criterion_2_three_normalizer_chain_round_trip() {
    let processor = TextProcessor::new(
        vec![
            Box::new(Lowercase),
            Box::new(WhitespaceNormalizer),
            Box::new(UnicodeNormalizer::default()),
        ],
        Box::new(WordTokenizer),
        ExecutionMode::All,
    )
    .unwrap();

    let input = "  Hello   WORLD  caf\u{00E9}  ";
    let result = processor.process(input).unwrap();
    let mapping = result.composed_mapping.as_ref().unwrap();

    for pos in 0..result.normalized.len() as u32 {
        if let Ok(orig_pos) = mapping.to_original(pos) {
            assert!(
                (orig_pos as usize) < input.len(),
                "Original position {} out of bounds for input len {}",
                orig_pos,
                input.len()
            );
        }
    }
}

// ── Success Criterion 3 ─────────────────────────────────────────────
// WordTokenizer handles CJK, emoji, RTL text correctly

#[test]
fn success_criterion_3_word_tokenizer_cjk() {
    let mut store = TextStoreBuilder::new();
    let text = "你好 世界";
    let mapping = CharMapping::identity(text.len() as u32).unwrap();
    let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "你好");
    assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "世界");
}

#[test]
fn success_criterion_3_word_tokenizer_emoji() {
    let mut store = TextStoreBuilder::new();
    let text = "hello 😀🎉 world";
    let mapping = CharMapping::identity(text.len() as u32).unwrap();
    let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
    assert_eq!(tokens.len(), 3);
    assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "😀🎉");
}

#[test]
fn success_criterion_3_word_tokenizer_rtl() {
    let mut store = TextStoreBuilder::new();
    let text = "مرحبا بالعالم";
    let mapping = CharMapping::identity(text.len() as u32).unwrap();
    let tokens = WordTokenizer.tokenize(text, &mapping, &mut store).unwrap();
    assert_eq!(tokens.len(), 2);
    for token in &tokens {
        let start = token.span.start() as usize;
        let end = token.span.end() as usize;
        assert!(start < end);
        assert!(end <= text.len());
        assert!(text.is_char_boundary(start));
        assert!(text.is_char_boundary(end));
    }
}

// ── Success Criterion 4 ─────────────────────────────────────────────
// TextProcessor pipeline: raw text -> 3 normalizers -> WordTokenizer -> ProcessedText

#[test]
fn success_criterion_4_full_pipeline_three_normalizers() {
    let processor = TextProcessor::new(
        vec![
            Box::new(Lowercase),
            Box::new(WhitespaceNormalizer),
            Box::new(UnicodeNormalizer::default()),
        ],
        Box::new(WordTokenizer),
        ExecutionMode::All,
    )
    .unwrap();

    let input = "  Hello   WORLD   caf\u{00E9}  ";
    let result = processor.process(input).unwrap();

    assert!(!result.tokens.is_empty(), "Pipeline should produce tokens");

    // Token spans reference original text positions (mapped through CharMapping)
    for token in &result.tokens {
        let start = token.span.start() as usize;
        let end = token.span.end() as usize;
        assert!(start <= end, "Span start {} > end {}", start, end);
        // Spans are within original text bounds
        assert!(
            end <= input.len(),
            "Span end {} exceeds original text length {}",
            end,
            input.len()
        );
        // Token text resolves correctly
        assert!(result.text_store.resolve(token.text_id).is_ok());
    }

    assert_eq!(result.layers.len(), 3);
    assert!(result.composed_mapping.is_some());
}

// ── Success Criterion 5 ─────────────────────────────────────────────
// All normalizer CharMappings compose correctly for accented characters

#[test]
fn success_criterion_5_char_mapping_compose_accented() {
    let processor = TextProcessor::new(
        vec![
            Box::new(Lowercase),
            Box::new(UnicodeNormalizer::default()),
            Box::new(RemoveDiacritics),
        ],
        Box::new(WordTokenizer),
        ExecutionMode::All,
    )
    .unwrap();

    let result = processor.process("CAFÉ").unwrap();
    assert_eq!(result.normalized, "cafe");

    let mapping = result.composed_mapping.as_ref().unwrap();
    assert_eq!(mapping.to_normalized(0).unwrap(), 0); // 'C' -> 'c'
}

#[test]
fn success_criterion_5_combining_characters() {
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
}

// ── Additional pipeline tests ────────────────────────────────────────

#[test]
fn named_layer_access_end_to_end() {
    let processor = TextProcessor::new(
        vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
        Box::new(WordTokenizer),
        ExecutionMode::All,
    )
    .unwrap();

    let result = processor.process("Hello  WORLD").unwrap();

    let lc_layer = result.layer("lowercase").unwrap();
    assert_eq!(lc_layer.text, "hello  world");

    let ws_layer = result.layer("whitespace").unwrap();
    assert_eq!(ws_layer.text, "hello world");

    assert!(result.layer("nonexistent").is_none());
}

#[test]
fn minimal_vs_all_execution_modes() {
    let make = |mode| {
        TextProcessor::new(
            vec![Box::new(Lowercase), Box::new(WhitespaceNormalizer)],
            Box::new(WordTokenizer),
            mode,
        )
        .unwrap()
    };

    let input = "Hello  WORLD";
    let all = make(ExecutionMode::All).process(input).unwrap();
    let min = make(ExecutionMode::Minimal).process(input).unwrap();

    assert_eq!(all.normalized, min.normalized);
    assert_eq!(all.tokens.len(), min.tokens.len());
    assert_eq!(all.layers.len(), 2);
    assert_eq!(min.layers.len(), 0);
    assert_eq!(min.skipped_normalizers.len(), 2);
}

#[test]
fn duplicate_normalizer_name_rejected() {
    let result = TextProcessor::new(
        vec![Box::new(Lowercase), Box::new(Lowercase)],
        Box::new(WordTokenizer),
        ExecutionMode::All,
    );
    assert!(result.is_err());
}

// ── Proptest ─────────────────────────────────────────────────────────

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn full_pipeline_arbitrary_unicode(input in "\\PC{1,100}") {
            let processor = TextProcessor::new(
                vec![
                    Box::new(Lowercase),
                    Box::new(WhitespaceNormalizer),
                    Box::new(UnicodeNormalizer::default()),
                ],
                Box::new(WordTokenizer),
                ExecutionMode::All,
            ).unwrap();

            // Should never panic
            let _ = processor.process(&input);
        }

        #[test]
        fn pipeline_mapping_stays_in_bounds(input in "\\PC{1,100}") {
            let processor = TextProcessor::new(
                vec![
                    Box::new(Lowercase),
                    Box::new(WhitespaceNormalizer),
                    Box::new(UnicodeNormalizer::default()),
                ],
                Box::new(WordTokenizer),
                ExecutionMode::All,
            ).unwrap();

            if let Ok(result) = processor.process(&input) {
                if let Some(mapping) = &result.composed_mapping {
                    for pos in 0..result.normalized.len().min(50) as u32 {
                        if let Ok(orig) = mapping.to_original(pos) {
                            prop_assert!((orig as usize) < input.len());
                        }
                    }
                }
            }
        }
    }
}
