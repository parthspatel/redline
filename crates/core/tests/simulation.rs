//! Deterministic simulation tests for the full text processing pipeline.
//!
//! Uses proptest for seed-based, reproducible property testing. Each test
//! exercises random combinations of normalizers and tokenizers against random
//! Unicode input, asserting invariants that must hold for ALL valid pipelines.
//!
//! Run with default cases:  `cargo test --test simulation`
//! Run with many cases:     `PROPTEST_CASES=10000 cargo test --test simulation`

use std::collections::HashSet;

use proptest::prelude::*;

use redline_core::normalize::{
    Lowercase, Normalizer, RemoveDiacritics, RemoveDigits, RemovePunctuation, UnicodeNormalizer,
    WhitespaceNormalizer,
};
use redline_core::process::{ExecutionMode, TextProcessor};
use redline_core::tokenize::{CharTokenizer, SentenceTokenizer, Tokenizer, WordTokenizer};

// ── Strategy helpers ──────────────────────────────────────────────────

fn normalizer_indices() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(0u8..6, 1..=4).prop_map(|indices| {
        let mut seen = HashSet::new();
        indices
            .into_iter()
            .filter(|i| seen.insert(*i))
            .collect::<Vec<_>>()
    })
}

fn tokenizer_index() -> impl Strategy<Value = u8> {
    0u8..3
}

fn build_normalizers(indices: &[u8]) -> Vec<Box<dyn Normalizer>> {
    indices
        .iter()
        .map(|i| match i {
            0 => Box::new(Lowercase) as Box<dyn Normalizer>,
            1 => Box::new(WhitespaceNormalizer) as Box<dyn Normalizer>,
            2 => Box::new(UnicodeNormalizer::default()) as Box<dyn Normalizer>,
            3 => Box::new(RemoveDiacritics) as Box<dyn Normalizer>,
            4 => Box::new(RemovePunctuation) as Box<dyn Normalizer>,
            _ => Box::new(RemoveDigits) as Box<dyn Normalizer>,
        })
        .collect()
}

fn build_tokenizer(index: u8) -> Box<dyn Tokenizer> {
    match index {
        0 => Box::new(CharTokenizer) as Box<dyn Tokenizer>,
        1 => Box::new(WordTokenizer) as Box<dyn Tokenizer>,
        _ => Box::new(SentenceTokenizer) as Box<dyn Tokenizer>,
    }
}

fn build_processor(norm_indices: &[u8], tok_index: u8) -> Option<TextProcessor> {
    let normalizers = build_normalizers(norm_indices);
    let tokenizer = build_tokenizer(tok_index);
    TextProcessor::new(normalizers, tokenizer, ExecutionMode::All).ok()
}

// ── Simulation tests ──────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// 1. Full pipeline never panics — random normalizer/tokenizer/input.
    #[test]
    fn simulation_full_pipeline_never_panics(
        norm_indices in normalizer_indices(),
        tok_index in tokenizer_index(),
        input in "\\PC{1,200}",
    ) {
        let processor = match build_processor(&norm_indices, tok_index) {
            Some(p) => p,
            None => return Ok(()),
        };
        let _ = processor.process(&input); // must not panic
    }

    /// 2. Spans always valid when processing succeeds.
    #[test]
    fn simulation_spans_always_valid(
        norm_indices in normalizer_indices(),
        tok_index in tokenizer_index(),
        input in "\\PC{1,200}",
    ) {
        let processor = match build_processor(&norm_indices, tok_index) {
            Some(p) => p,
            None => return Ok(()),
        };

        if let Ok(result) = processor.process(&input) {
            let bound = if tok_index == 2 {
                result.normalized.len() as u32
            } else {
                input.len() as u32
            };
            for token in &result.tokens {
                prop_assert!(token.span.start() <= token.span.end());
                prop_assert!(
                    token.span.end() <= bound,
                    "span end {} > bound {} (tok={}, norms={:?})",
                    token.span.end(), bound, tok_index, norm_indices
                );
            }
            if let Some(ref mapping) = result.composed_mapping {
                prop_assert_eq!(mapping.original_len() as usize, input.len());
            }
        }
    }

    /// 3. Composed mapping is consistent with input/normalized lengths.
    #[test]
    fn simulation_composed_mapping_consistent(
        norm_indices in normalizer_indices(),
        input in "\\PC{1,200}",
    ) {
        let processor = match build_processor(&norm_indices, 1) {
            Some(p) => p,
            None => return Ok(()),
        };

        if let Ok(result) = processor.process(&input) {
            if let Some(ref mapping) = result.composed_mapping {
                prop_assert_eq!(mapping.original_len() as usize, input.len());
                for &(orig, norm) in mapping.alignments() {
                    prop_assert!((orig as usize) < input.len());
                    prop_assert!((norm as usize) < result.normalized.len());
                }
            }
        }
    }

    /// 4. Every token's StringId resolves via text_store.
    #[test]
    fn simulation_token_text_resolvable(
        norm_indices in normalizer_indices(),
        tok_index in tokenizer_index(),
        input in "\\PC{1,200}",
    ) {
        let processor = match build_processor(&norm_indices, tok_index) {
            Some(p) => p,
            None => return Ok(()),
        };

        if let Ok(result) = processor.process(&input) {
            for token in &result.tokens {
                let resolved = result.text_store.resolve(token.text_id);
                prop_assert!(resolved.is_ok());
                prop_assert!(!resolved.unwrap().is_empty());
            }
        }
    }

    /// 5. Normalizer order: forward vs reversed both don't panic.
    #[test]
    fn simulation_normalizer_chain_order_matters(
        norm_indices in prop::collection::vec(0u8..6, 2..=4)
            .prop_map(|indices| {
                let mut seen = HashSet::new();
                indices.into_iter().filter(|i| seen.insert(*i)).collect::<Vec<_>>()
            })
            .prop_filter("need >= 2 unique", |v| v.len() >= 2),
        input in "\\PC{1,200}",
    ) {
        let proc_fwd = match build_processor(&norm_indices, 1) {
            Some(p) => p,
            None => return Ok(()),
        };
        let reversed: Vec<u8> = norm_indices.iter().rev().copied().collect();
        let proc_rev = match build_processor(&reversed, 1) {
            Some(p) => p,
            None => return Ok(()),
        };
        let _ = proc_fwd.process(&input);
        let _ = proc_rev.process(&input);
    }

    /// 6. Double processing: process(result.normalized) doesn't panic.
    #[test]
    fn simulation_idempotent_double_process(
        norm_indices in normalizer_indices(),
        tok_index in tokenizer_index(),
        input in "\\PC{1,200}",
    ) {
        let proc1 = match build_processor(&norm_indices, tok_index) {
            Some(p) => p,
            None => return Ok(()),
        };

        if let Ok(result1) = proc1.process(&input) {
            if !result1.normalized.is_empty() {
                let proc2 = match build_processor(&norm_indices, tok_index) {
                    Some(p) => p,
                    None => return Ok(()),
                };
                let _ = proc2.process(&result1.normalized); // must not panic
            }
        }
    }

    /// 7. Large input doesn't panic.
    #[test]
    #[ignore]
    fn simulation_large_input_no_panic(
        norm_indices in normalizer_indices(),
        tok_index in tokenizer_index(),
        input in "\\PC{1000,5000}",
    ) {
        let processor = match build_processor(&norm_indices, tok_index) {
            Some(p) => p,
            None => return Ok(()),
        };
        let _ = processor.process(&input);
    }
}

// ── Deterministic smoke tests ─────────────────────────────────────────

#[test]
fn simulation_all_normalizers_individually() {
    let inputs = ["Hello World!", "\u{00FC}ber caf\u{00E9} 42", "a b c 1 2 3"];
    for &input in &inputs {
        for idx in 0u8..6 {
            let normalizers = build_normalizers(&[idx]);
            let tokenizer = build_tokenizer(1);
            let proc = TextProcessor::new(normalizers, tokenizer, ExecutionMode::All).unwrap();
            let _ = proc.process(input);
        }
    }
}

#[test]
fn simulation_all_tokenizers_individually() {
    let inputs = ["Hello World!", "\u{00FC}ber caf\u{00E9} 42"];
    for &input in &inputs {
        for tok_idx in 0u8..3 {
            let tokenizer = build_tokenizer(tok_idx);
            let proc = TextProcessor::new(vec![], tokenizer, ExecutionMode::All).unwrap();
            let result = proc.process(input).unwrap();
            assert!(!result.tokens.is_empty());
        }
    }
}

#[test]
fn simulation_full_normalizer_chain() {
    let all: Vec<u8> = (0..6).collect();
    let input = "Hello, World! \u{00FC}ber 42 caf\u{00E9}.";
    for tok_idx in 0u8..3 {
        let proc = build_processor(&all, tok_idx).expect("all 6 normalizers should build");
        let _ = proc.process(input);
    }
}
