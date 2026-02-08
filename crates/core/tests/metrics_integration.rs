//! Comprehensive integration tests for the full metrics engine.

use hashbrown::HashMap;
use redline_core::metrics::DependencyKind;
use redline_core::metrics::error::MetricError;
use redline_core::process::ProcessedText;
use redline_core::tokenize::WordTokenizer;
use redline_core::{
    ExecutionMode, Metric, MetricInput, MetricRegistry, MetricValue, MetricsEngine, TextProcessor,
    register_builtins,
};

fn process(s: &str) -> ProcessedText {
    TextProcessor::new(vec![], Box::new(WordTokenizer), ExecutionMode::All)
        .unwrap()
        .process(s)
        .unwrap()
}

const SINGLE_METRIC_IDS: &[&str] = &[
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
    "avg_word_length",
    "avg_sentence_length",
    "vocabulary_richness",
    "lexical_density",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "coleman_liau",
    "ari",
];

const PAIRWISE_METRIC_IDS: &[&str] = &[
    "jaccard_similarity",
    "cosine_similarity",
    "dice_coefficient",
    "overlap_coefficient",
    "length_ratio",
    "word_count_diff",
    "char_count_diff",
    "levenshtein_distance",
    "damerau_levenshtein",
    "hamming_distance",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "readability_delta",
    "grade_level_delta",
    "similarity_ratio",
];

// ── Full Pipeline: Single-Text ──────────────────────────────────────

#[test]
fn full_pipeline_single_text() {
    let mut reg = MetricRegistry::new();
    register_builtins(&mut reg).unwrap();

    let engine = MetricsEngine::with_cache(reg, 256);
    let text = process(
        "The quick brown fox jumps over the lazy dog. \
         She sells seashells by the seashore. \
         How much wood would a woodchuck chuck?",
    );
    let input = MetricInput::Single(&text);

    let results = engine.get_many(SINGLE_METRIC_IDS, &input);
    assert_eq!(results.len(), SINGLE_METRIC_IDS.len());

    for id in SINGLE_METRIC_IDS {
        assert!(
            !results[*id].is_unavailable(),
            "metric '{id}' returned Unavailable"
        );
    }

    // Spot checks
    let word_count = results["word_count"].as_integer().unwrap();
    assert!(word_count > 0, "word_count should be > 0");

    let fre = results["flesch_reading_ease"].as_float().unwrap();
    assert!(
        fre != 0.0,
        "flesch_reading_ease should be nonzero for real text"
    );

    // Caching: second call should not increase cache_len
    let cache_before = engine.cache_len();
    let _ = engine.get_many(SINGLE_METRIC_IDS, &input);
    assert_eq!(engine.cache_len(), cache_before);
}

// ── Full Pipeline: Pairwise ─────────────────────────────────────────

#[test]
fn full_pipeline_pairwise() {
    let mut reg = MetricRegistry::new();
    register_builtins(&mut reg).unwrap();

    let engine = MetricsEngine::with_cache(reg, 256);
    let source = process("The quick brown fox jumps over the lazy dog.");
    let target = process("A fast red fox leaps across the sleepy hound.");
    let input = MetricInput::Pairwise(&source, &target);

    // hamming requires equal-length; these texts have same token count (9 each)
    let results = engine.get_many(PAIRWISE_METRIC_IDS, &input);

    for id in PAIRWISE_METRIC_IDS {
        assert!(
            !results[*id].is_unavailable(),
            "pairwise metric '{id}' returned Unavailable"
        );
    }

    // Similarity metrics in [0.0, 1.0]
    for id in &[
        "jaccard_similarity",
        "cosine_similarity",
        "dice_coefficient",
        "overlap_coefficient",
        "jaro_similarity",
        "jaro_winkler_similarity",
        "similarity_ratio",
    ] {
        let v = results[*id].as_float().unwrap();
        assert!((0.0..=1.0).contains(&v), "{id} = {v}, expected [0,1]");
    }

    // Distance metrics >= 0
    let lev = results["levenshtein_distance"].as_integer().unwrap();
    assert!(lev >= 0, "levenshtein should be >= 0");
}

// ── Identical Texts ─────────────────────────────────────────────────

#[test]
fn identical_texts_pairwise() {
    let mut reg = MetricRegistry::new();
    register_builtins(&mut reg).unwrap();

    let engine = MetricsEngine::new(reg);
    let text = process("The cat sat on the mat.");
    let input = MetricInput::Pairwise(&text, &text);

    assert_eq!(
        engine.get("jaccard_similarity", &input),
        MetricValue::Float(1.0)
    );
    assert_eq!(
        engine.get("levenshtein_distance", &input),
        MetricValue::Integer(0)
    );
    assert_eq!(
        engine.get("hamming_distance", &input),
        MetricValue::Integer(0)
    );
    assert_eq!(
        engine.get("word_count_diff", &input),
        MetricValue::Integer(0)
    );
    assert_eq!(
        engine.get("char_count_diff", &input),
        MetricValue::Integer(0)
    );

    let jaro = engine.get("jaro_similarity", &input).as_float().unwrap();
    assert!((jaro - 1.0).abs() < 0.001);

    let jw = engine
        .get("jaro_winkler_similarity", &input)
        .as_float()
        .unwrap();
    assert!((jw - 1.0).abs() < 0.001);

    let ratio = engine.get("similarity_ratio", &input).as_float().unwrap();
    assert!((ratio - 1.0).abs() < 0.001);

    let rd = engine.get("readability_delta", &input).as_float().unwrap();
    assert!((rd - 0.0).abs() < 0.001);
}

// ── Lazy Evaluation (no cache) ──────────────────────────────────────

#[test]
fn lazy_evaluation_no_cache() {
    let mut reg = MetricRegistry::new();
    register_builtins(&mut reg).unwrap();

    let engine = MetricsEngine::new(reg);
    let text = process("Hello world.");
    let input = MetricInput::Single(&text);

    assert_eq!(engine.get("word_count", &input), MetricValue::Integer(2));
    assert_eq!(engine.cache_len(), 0); // no cache
}

// ── Dependency Resolution Chain ─────────────────────────────────────

#[test]
fn dependency_chain_with_cache() {
    let mut reg = MetricRegistry::new();
    register_builtins(&mut reg).unwrap();

    let engine = MetricsEngine::with_cache(reg, 256);
    let text = process("The cat sat on the mat. The dog lay on the rug.");
    let input = MetricInput::Single(&text);

    // Request flesch_reading_ease — depends on word_count, sentence_count, syllable_count
    let fre = engine.get("flesch_reading_ease", &input);
    assert!(!fre.is_unavailable());

    // Cache should contain flesch + its 3 deps = at least 4 entries
    assert!(
        engine.cache_len() >= 4,
        "cache has {} entries",
        engine.cache_len()
    );

    // Dependencies should be cached too
    let wc = engine.get("word_count", &input);
    assert!(!wc.is_unavailable());
}

// ── Custom Metric Integration ───────────────────────────────────────

struct TextLengthCategoryMetric;

impl Metric for TextLengthCategoryMetric {
    fn id(&self) -> &str {
        "text_length_category"
    }

    fn compute(
        &self,
        input: &MetricInput<'_>,
        deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        match input {
            MetricInput::Single(_) => {
                let words = deps
                    .get("word_count")
                    .and_then(|v| v.as_integer())
                    .ok_or_else(|| MetricError::DependencyUnavailable("word_count".into()))?;
                // 0=empty, 1=short(<10), 2=medium(10-50), 3=long(>50)
                let category = if words == 0 {
                    0
                } else if words < 10 {
                    1
                } else if words <= 50 {
                    2
                } else {
                    3
                };
                Ok(MetricValue::Integer(category))
            }
            MetricInput::Pairwise(_, _) => Err(MetricError::InvalidInput(
                "text_length_category".into(),
                "expected Single input".into(),
            )),
        }
    }

    fn static_dependencies(&self) -> &[&str] {
        &["word_count"]
    }

    fn dependency_kind(&self) -> DependencyKind {
        DependencyKind::Static
    }
}

#[test]
fn custom_metric_with_builtins() {
    let mut reg = MetricRegistry::new();
    register_builtins(&mut reg).unwrap();
    reg.register(Box::new(TextLengthCategoryMetric)).unwrap();
    reg.validate().unwrap();

    let engine = MetricsEngine::new(reg);
    let text = process("Hello world.");
    let input = MetricInput::Single(&text);

    let category = engine.get("text_length_category", &input);
    assert_eq!(category, MetricValue::Integer(1)); // 2 words = short
}

// ── Override Test ───────────────────────────────────────────────────

struct ConstWordCount;

impl Metric for ConstWordCount {
    fn id(&self) -> &str {
        "word_count"
    }

    fn compute(
        &self,
        _input: &MetricInput<'_>,
        _deps: &HashMap<String, MetricValue>,
    ) -> Result<MetricValue, MetricError> {
        Ok(MetricValue::Integer(999))
    }
}

#[test]
fn override_builtin_metric() {
    let mut reg = MetricRegistry::new();
    register_builtins(&mut reg).unwrap();
    reg.register_with_override(Box::new(ConstWordCount));

    let engine = MetricsEngine::new(reg);
    let text = process("Hello world.");
    let input = MetricInput::Single(&text);

    // Overridden word_count returns 999
    assert_eq!(engine.get("word_count", &input), MetricValue::Integer(999));

    // Dependent metrics use the overridden value
    let awl = engine.get("avg_word_length", &input);
    // avg_word_length = char_count / word_count
    // word_count is now 999, char_count is real
    assert!(!awl.is_unavailable());
}

// ── Error Handling ──────────────────────────────────────────────────

#[test]
fn nonexistent_metric_returns_unavailable() {
    let mut reg = MetricRegistry::new();
    register_builtins(&mut reg).unwrap();

    let engine = MetricsEngine::new(reg);
    let text = process("test");
    let input = MetricInput::Single(&text);

    assert_eq!(
        engine.get("nonexistent_metric", &input),
        MetricValue::Unavailable
    );
}

#[test]
fn missing_dependency_validation_error() {
    struct BadMetric;
    impl Metric for BadMetric {
        fn id(&self) -> &str {
            "bad_metric"
        }
        fn compute(
            &self,
            _: &MetricInput<'_>,
            _: &HashMap<String, MetricValue>,
        ) -> Result<MetricValue, MetricError> {
            Ok(MetricValue::Integer(0))
        }
        fn static_dependencies(&self) -> &[&str] {
            &["does_not_exist"]
        }
    }

    let mut reg = MetricRegistry::new();
    reg.register(Box::new(BadMetric)).unwrap();
    assert!(reg.validate().is_err());
}

// ── Property Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn single_text_counts_non_negative(text in "[a-zA-Z .!?]{0,200}") {
            let mut reg = MetricRegistry::new();
            register_builtins(&mut reg).unwrap();
            let engine = MetricsEngine::new(reg);
            let processed = process(&text);
            let input = MetricInput::Single(&processed);

            let wc = engine.get("word_count", &input).as_integer().unwrap();
            prop_assert!(wc >= 0, "word_count={wc}");

            let uc = engine.get("unique_word_count", &input).as_integer().unwrap();
            prop_assert!(uc <= wc, "unique_word_count={uc} > word_count={wc}");
        }

        #[test]
        fn pairwise_similarity_in_range(
            a in "[a-z ]{1,50}",
            b in "[a-z ]{1,50}",
        ) {
            let mut reg = MetricRegistry::new();
            register_builtins(&mut reg).unwrap();
            let engine = MetricsEngine::new(reg);
            let pa = process(&a);
            let pb = process(&b);
            let input = MetricInput::Pairwise(&pa, &pb);

            let jaccard = engine.get("jaccard_similarity", &input).as_float().unwrap();
            prop_assert!((0.0..=1.0).contains(&jaccard), "jaccard={jaccard}");

            let jaro = engine.get("jaro_similarity", &input).as_float().unwrap();
            prop_assert!((0.0..=1.0).contains(&jaro), "jaro={jaro}");

            let lev = engine.get("levenshtein_distance", &input).as_integer().unwrap();
            prop_assert!(lev >= 0, "levenshtein={lev}");
        }

        #[test]
        fn self_similarity_identity(text in "[a-z ]{1,50}") {
            let mut reg = MetricRegistry::new();
            register_builtins(&mut reg).unwrap();
            let engine = MetricsEngine::new(reg);
            let processed = process(&text);
            let input = MetricInput::Pairwise(&processed, &processed);

            let lev = engine.get("levenshtein_distance", &input).as_integer().unwrap();
            prop_assert_eq!(lev, 0, "levenshtein(a,a) should be 0");

            let jaccard = engine.get("jaccard_similarity", &input).as_float().unwrap();
            prop_assert!((jaccard - 1.0).abs() < 0.001, "jaccard(a,a) should be 1.0, got {jaccard}");
        }

        #[test]
        fn no_panics_for_any_input(text in "\\PC{0,100}") {
            let mut reg = MetricRegistry::new();
            register_builtins(&mut reg).unwrap();
            let engine = MetricsEngine::new(reg);
            let processed = process(&text);
            let input = MetricInput::Single(&processed);

            // Just ensure no panics
            for id in super::SINGLE_METRIC_IDS {
                let _ = engine.get(id, &input);
            }
        }
    }
}
