//! Integration tests for the metrics engine with count metrics.

use redline_core::TextProcessor;
use redline_core::metrics::counts::*;
use redline_core::metrics::{MetricInput, MetricRegistry, MetricValue, MetricsEngine};
use redline_core::process::ProcessedText;
use redline_core::tokenize::WordTokenizer;

fn register_all_counts(reg: &mut MetricRegistry) {
    reg.register(Box::new(WordCountMetric)).unwrap();
    reg.register(Box::new(CharCountMetric)).unwrap();
    reg.register(Box::new(ByteCountMetric)).unwrap();
    reg.register(Box::new(SentenceCountMetric)).unwrap();
    reg.register(Box::new(SyllableCountMetric)).unwrap();
    reg.register(Box::new(UniqueWordCountMetric)).unwrap();
    reg.register(Box::new(ParagraphCountMetric)).unwrap();
    reg.register(Box::new(LineCountMetric)).unwrap();
    reg.register(Box::new(LetterCountMetric)).unwrap();
    reg.register(Box::new(DigitCountMetric)).unwrap();
    reg.register(Box::new(WhitespaceCountMetric)).unwrap();
    reg.register(Box::new(PunctuationCountMetric)).unwrap();
}

fn process_text(text: &str) -> ProcessedText {
    let processor = TextProcessor::new(
        vec![],
        Box::new(WordTokenizer),
        redline_core::ExecutionMode::All,
    )
    .unwrap();
    processor.process(text).unwrap()
}

#[test]
fn engine_all_count_metrics() {
    let mut reg = MetricRegistry::new();
    register_all_counts(&mut reg);
    reg.validate().unwrap();

    let engine = MetricsEngine::with_cache(reg, 64);
    let text = process_text("The quick brown fox jumps over the lazy dog.");
    let input = MetricInput::Single(&text);

    // Word count: 9 tokens
    assert_eq!(engine.get("word_count", &input), MetricValue::Integer(9));

    // Char count: "The quick brown fox jumps over the lazy dog." = 44 chars
    assert_eq!(engine.get("char_count", &input), MetricValue::Integer(44));

    // Byte count: all ASCII = same as char count
    assert_eq!(engine.get("byte_count", &input), MetricValue::Integer(44));

    // Sentence count: 1 period
    assert_eq!(
        engine.get("sentence_count", &input),
        MetricValue::Integer(1)
    );

    // Letter count: 35 letters (no spaces or period)
    assert_eq!(engine.get("letter_count", &input), MetricValue::Integer(35));

    // Digit count: 0
    assert_eq!(engine.get("digit_count", &input), MetricValue::Integer(0));

    // Whitespace count: 8 spaces
    assert_eq!(
        engine.get("whitespace_count", &input),
        MetricValue::Integer(8)
    );

    // Punctuation count: 1 period
    assert_eq!(
        engine.get("punctuation_count", &input),
        MetricValue::Integer(1)
    );

    // Line count: 1 line (no newlines)
    assert_eq!(engine.get("line_count", &input), MetricValue::Integer(1));

    // Paragraph count: 1 paragraph (no double newlines)
    assert_eq!(
        engine.get("paragraph_count", &input),
        MetricValue::Integer(1)
    );
}

#[test]
fn engine_cache_hit() {
    let mut reg = MetricRegistry::new();
    register_all_counts(&mut reg);

    let engine = MetricsEngine::with_cache(reg, 64);
    let text = process_text("Hello world.");
    let input = MetricInput::Single(&text);

    // First call: cache miss
    let v1 = engine.get("word_count", &input);
    let cache_after_first = engine.cache_len();

    // Second call: cache hit
    let v2 = engine.get("word_count", &input);
    let cache_after_second = engine.cache_len();

    assert_eq!(v1, v2);
    assert_eq!(cache_after_first, cache_after_second);
}

#[test]
fn engine_get_many_all_counts() {
    let mut reg = MetricRegistry::new();
    register_all_counts(&mut reg);

    let engine = MetricsEngine::with_cache(reg, 64);
    let text = process_text("Hello, world!");
    let input = MetricInput::Single(&text);

    let ids = &[
        "word_count",
        "char_count",
        "byte_count",
        "sentence_count",
        "syllable_count",
        "unique_word_count",
        "paragraph_count",
        "line_count",
        "letter_count",
        "digit_count",
        "whitespace_count",
        "punctuation_count",
    ];

    let results = engine.get_many(ids, &input);
    assert_eq!(results.len(), 12);

    // All should be Integer (not Unavailable)
    for (id, value) in &results {
        assert!(
            !value.is_unavailable(),
            "metric '{id}' returned Unavailable"
        );
    }
}

#[test]
fn engine_empty_text() {
    let mut reg = MetricRegistry::new();
    register_all_counts(&mut reg);

    let engine = MetricsEngine::new(reg);
    let text = process_text("");
    let input = MetricInput::Single(&text);

    assert_eq!(engine.get("word_count", &input), MetricValue::Integer(0));
    assert_eq!(engine.get("char_count", &input), MetricValue::Integer(0));
    assert_eq!(engine.get("byte_count", &input), MetricValue::Integer(0));
    assert_eq!(
        engine.get("sentence_count", &input),
        MetricValue::Integer(0)
    );
    assert_eq!(
        engine.get("syllable_count", &input),
        MetricValue::Integer(0)
    );
    assert_eq!(
        engine.get("unique_word_count", &input),
        MetricValue::Integer(0)
    );
}

#[test]
fn engine_unicode_text() {
    let mut reg = MetricRegistry::new();
    register_all_counts(&mut reg);

    let engine = MetricsEngine::new(reg);
    let text = process_text("cafe\u{0301} naive\u{0308}");
    let input = MetricInput::Single(&text);

    // char_count counts Unicode codepoints
    let char_val = engine.get("char_count", &input);
    assert!(char_val.as_integer().unwrap() > 0);

    // byte_count >= char_count for multi-byte chars
    let byte_val = engine.get("byte_count", &input);
    assert!(byte_val.as_integer().unwrap() >= char_val.as_integer().unwrap());
}
