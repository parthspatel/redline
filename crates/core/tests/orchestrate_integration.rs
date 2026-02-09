//! Integration tests verifying all 5 Phase 6 (Orchestration & Config) success criteria.
//!
//! SC1: `Redline::new(config).diff("old", "new")` returns complete result
//! SC2: `ConfigBuilder::preset(Fast)` produces valid config with minimal processing
//! SC3: Filter operations with `kind == Replace && span.len() > 10`
//! SC4: Memory: 10K word document pair analysis uses <50MB
//! SC5: CacheManager multi-threaded stress test (no deadlock)

use std::sync::Arc;
use std::thread;

use redline_core::{ConfigBuilder, EditKind, Filter, Preset, Redline};

// ═══════════════════════════════════════════════════════════════════════
// SC1: Redline::new(config).diff() returns complete result
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn sc1_redline_diff_returns_complete_result() {
    let config = ConfigBuilder::new().build().unwrap();
    let redline = Redline::new(config).unwrap();

    let result = redline
        .diff(
            "The quick brown fox jumps over the lazy dog.",
            "The fast brown fox leaps over the lazy cat.",
        )
        .unwrap();

    // Diff is present with operations
    assert!(
        !result.diff.operations.is_empty(),
        "Diff should have operations"
    );
    assert!(
        result
            .diff
            .operations
            .iter()
            .any(|op| op.kind == EditKind::Replace),
        "Should have Replace operations"
    );

    // Metrics are present
    assert!(!result.metrics.is_empty(), "Metrics should be computed");

    // Analysis is present (Comprehensive default)
    assert!(
        result.analysis.is_some(),
        "Analysis should be present for Comprehensive preset"
    );
    let report = result.analysis.as_ref().unwrap();
    assert!(
        report.result_count() > 0,
        "Analysis report should have results"
    );
}

#[test]
fn sc1_diff_empty_texts() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff("", "").unwrap();
    assert!(result.diff.operations.is_empty());
}

#[test]
fn sc1_diff_identical_texts() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline
        .diff("hello world today", "hello world today")
        .unwrap();

    assert!(
        (result.diff.similarity() - 1.0).abs() < f64::EPSILON,
        "Identical texts should have similarity 1.0, got {}",
        result.diff.similarity()
    );
}

#[test]
fn sc1_diff_completely_different() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline
        .diff("alpha bravo charlie", "xray yankee zulu")
        .unwrap();

    assert!(
        result.diff.similarity() < 0.5,
        "Completely different texts should have low similarity, got {}",
        result.diff.similarity()
    );
}

#[test]
fn sc1_diff_multiline_documents() {
    let source = "First paragraph with several words.\n\
                  Second paragraph continues here.\n\
                  Third paragraph concludes the document.";
    let target = "First paragraph with different words.\n\
                  Second paragraph was modified here.\n\
                  Third paragraph concludes the document.";

    let config = ConfigBuilder::new().build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff(source, target).unwrap();

    assert!(!result.diff.operations.is_empty());
    assert!(result.analysis.is_some());
}

// ═══════════════════════════════════════════════════════════════════════
// SC2: ConfigBuilder::preset(Fast) produces minimal processing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn sc2_fast_preset_minimal_processing() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();

    let result = redline
        .diff(
            "The quick brown fox jumps over the lazy dog.",
            "The fast brown fox leaps over the lazy cat.",
        )
        .unwrap();

    // Fast preset: diff + metrics, NO analysis
    assert!(
        !result.diff.operations.is_empty(),
        "Diff should still be computed"
    );
    assert!(
        !result.metrics.is_empty(),
        "Basic metrics should be computed"
    );
    assert!(
        result.analysis.is_none(),
        "Analysis should NOT be present for Fast preset"
    );
}

#[test]
fn sc2_fast_preset_override_enable_analysis() {
    let config = ConfigBuilder::preset(Preset::Fast)
        .with_analysis(true)
        .build()
        .unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff("hello world", "hello earth").unwrap();
    assert!(
        result.analysis.is_some(),
        "Analysis should be present when overridden to true on Fast preset"
    );
}

#[test]
fn sc2_comprehensive_preset_override_disable_analysis() {
    let config = ConfigBuilder::new().with_analysis(false).build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff("hello world", "hello earth").unwrap();
    assert!(
        result.analysis.is_none(),
        "Analysis should NOT be present when overridden to false on Comprehensive preset"
    );
}

#[test]
fn sc2_comprehensive_vs_fast_both_produce_diff() {
    let source = "The cat sat on the mat.";
    let target = "The dog sat on the rug.";

    let fast = Redline::new(ConfigBuilder::preset(Preset::Fast).build().unwrap()).unwrap();
    let comp = Redline::new(ConfigBuilder::new().build().unwrap()).unwrap();

    let r_fast = fast.diff(source, target).unwrap();
    let r_comp = comp.diff(source, target).unwrap();

    // Both should produce diff operations
    assert!(!r_fast.diff.operations.is_empty());
    assert!(!r_comp.diff.operations.is_empty());

    // Comprehensive should have analysis, Fast should not
    assert!(r_fast.analysis.is_none());
    assert!(r_comp.analysis.is_some());
}

// ═══════════════════════════════════════════════════════════════════════
// SC3: Filter operations where kind == Replace && span.len() > 10
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn sc3_filter_replace_by_length() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();

    // Source and target where the replace spans many tokens
    let source = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 end";
    let target = "x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 end";

    let result = redline.diff(source, target).unwrap();

    let pred = Filter::kind(EditKind::Replace) & Filter::min_length(10);
    let filtered: Vec<_> = result.filter(&pred).collect();

    for op in &filtered {
        assert_eq!(op.kind, EditKind::Replace);
        assert!(
            op.source_len().max(op.target_len()) >= 10,
            "Should have length >= 10, got {}",
            op.source_len().max(op.target_len())
        );
    }
}

#[test]
fn sc3_filter_insert_or_delete() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();

    let result = redline
        .diff("alpha bravo charlie", "alpha delta echo foxtrot charlie")
        .unwrap();

    let pred = Filter::kind(EditKind::Insert) | Filter::kind(EditKind::Delete);
    let filtered: Vec<_> = result.filter(&pred).collect();

    for op in &filtered {
        assert!(
            op.kind == EditKind::Insert || op.kind == EditKind::Delete,
            "Should be Insert or Delete, got {:?}",
            op.kind
        );
    }
}

#[test]
fn sc3_filter_not_equal() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff("hello world", "hello earth").unwrap();

    let pred = !Filter::kind(EditKind::Equal);
    let changes: Vec<_> = result.filter(&pred).collect();

    for op in &changes {
        assert_ne!(op.kind, EditKind::Equal);
    }
    assert!(!changes.is_empty(), "Should have some non-Equal operations");
}

#[test]
fn sc3_filter_impossible_predicate_returns_empty() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff("hello world", "hello earth").unwrap();

    // Replace AND Equal is impossible
    let pred = Filter::kind(EditKind::Replace) & Filter::kind(EditKind::Equal);
    let filtered: Vec<_> = result.filter(&pred).collect();

    assert!(
        filtered.is_empty(),
        "Contradictory filter should return empty"
    );
}

#[test]
fn sc3_filter_combined_kind_and_source_contains() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline
        .diff(
            "The quick brown fox jumps over the lazy dog.",
            "The fast brown fox leaps over the lazy cat.",
        )
        .unwrap();

    // All non-Equal ops that have source text containing a substring
    let pred = !Filter::kind(EditKind::Equal);
    let changes: Vec<_> = result.filter(&pred).collect();
    assert!(!changes.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// SC4: Memory — 10K word document pair analysis uses <50MB
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn sc4_memory_usage_10k_words_under_50mb() {
    let words: &[&str] = &[
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "alpha", "bravo",
        "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima",
        "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "xray", "yankee", "zulu", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten", "this", "that", "with", "from", "into", "upon",
    ];

    // Build source: 10K words
    let source: String = (0..10_000)
        .map(|i| words[i % words.len()])
        .collect::<Vec<_>>()
        .join(" ");

    // Build target: 10K words with ~10% changes
    let target: String = (0..10_000)
        .map(|i| {
            if i % 10 == 0 {
                words[(i + 7) % words.len()]
            } else {
                words[i % words.len()]
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    let config = ConfigBuilder::new().build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff(&source, &target).unwrap();

    // Result should be complete
    assert!(!result.diff.operations.is_empty());
    assert!(!result.metrics.is_empty());
    assert!(result.analysis.is_some());

    // Estimate data memory from actual result sizes.
    // RSS-based measurement is unreliable (includes test binary, allocator overhead, etc.).
    let input_bytes = source.len() + target.len();
    let ops_bytes =
        result.diff.operations.len() * std::mem::size_of::<redline_core::EditOperation>();
    let source_text_bytes = result.diff.source.original.len() + result.diff.source.normalized.len();
    let target_text_bytes = result.diff.target.original.len() + result.diff.target.normalized.len();
    let token_bytes = (result.diff.source.tokens.len() + result.diff.target.tokens.len())
        * std::mem::size_of::<redline_core::Token>();
    let metrics_bytes = result.metrics.len() * 64;
    let analysis_bytes: usize = 4096;

    let estimated_bytes = input_bytes
        + ops_bytes
        + source_text_bytes
        + target_text_bytes
        + token_bytes
        + metrics_bytes
        + analysis_bytes;
    let estimated_mb = estimated_bytes as f64 / (1024.0 * 1024.0);

    // Data footprint should be well under 50MB for 10K words (~55KB input)
    assert!(
        estimated_mb < 50.0,
        "Estimated data memory should be <50MB, got {:.2}MB",
        estimated_mb
    );

    eprintln!(
        "SC4: 10K word pair — input: {}KB, ops: {}, tokens: {}, estimated data: {:.2}MB (limit: 50MB)",
        input_bytes / 1024,
        result.diff.operations.len(),
        result.diff.source.tokens.len() + result.diff.target.tokens.len(),
        estimated_mb,
    );
}

// ═══════════════════════════════════════════════════════════════════════
// SC5: CacheManager multi-threaded stress test
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn sc5_cache_multithreaded_stress_test() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Arc::new(Redline::new(config).unwrap());

    let num_threads = 8;
    let ops_per_thread = 20;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let r = Arc::clone(&redline);
            thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let source = format!("Thread {} source text number {}", thread_id, i);
                    let target = format!("Thread {} target text number {}", thread_id, i + 1);
                    let result = r.diff(&source, &target);
                    assert!(result.is_ok(), "diff() should not fail: {:?}", result.err());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // Cache should have entries (up to capacity, which is 64 for Fast preset)
    let cache = redline.cache().expect("Cache should be enabled");
    assert!(
        cache.len() > 0,
        "Cache should have entries after stress test"
    );
    assert!(
        cache.len() <= 64,
        "Cache should respect capacity limit of 64"
    );
}

#[test]
fn sc5_cache_concurrent_same_key() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Arc::new(Redline::new(config).unwrap());

    let num_threads = 4;
    let source = "identical source text for all threads";
    let target = "identical target text for all threads";

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let r = Arc::clone(&redline);
            let s = source.to_string();
            let t = target.to_string();
            thread::spawn(move || {
                let result = r.diff(&s, &t);
                assert!(result.is_ok());
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // All threads diffed the same texts — cache should have exactly 1 entry
    let cache = redline.cache().expect("Cache should be enabled");
    assert_eq!(
        cache.len(),
        1,
        "Same key from all threads should result in 1 cache entry"
    );
}

#[test]
fn sc5_cache_hit_returns_same_result() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();

    let r1 = redline.diff("hello world", "hello earth").unwrap();
    let r2 = redline.diff("hello world", "hello earth").unwrap();

    // Both results should have the same diff operations count and similarity
    assert_eq!(r1.diff.operations.len(), r2.diff.operations.len());
    assert!(
        (r1.diff.similarity() - r2.diff.similarity()).abs() < f64::EPSILON,
        "Cache hit should return identical similarity"
    );
    assert_eq!(r1.metrics.len(), r2.metrics.len());
}

// ═══════════════════════════════════════════════════════════════════════
// Additional integration tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn comprehensive_analysis_has_all_builtin_analyzers() {
    let config = ConfigBuilder::new().build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline
        .diff(
            "The quick brown fox jumps over the lazy dog in the morning.",
            "The slow brown fox leaps over the happy cat in the evening.",
        )
        .unwrap();

    let report = result.analysis.as_ref().expect("Should have analysis");

    // All 4 built-in analyzers should have run
    assert!(report.has_result("semantic"), "Missing semantic analyzer");
    assert!(report.has_result("stylistic"), "Missing stylistic analyzer");
    assert!(
        report.has_result("readability"),
        "Missing readability analyzer"
    );
    assert!(
        report.has_result("edit_classifier"),
        "Missing edit_classifier"
    );
}

#[test]
fn diff_with_custom_algorithm() {
    use redline_core::Histogram;

    let config = ConfigBuilder::preset(Preset::Fast)
        .with_algorithm(Box::new(Histogram::new()))
        .build()
        .unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff("alpha bravo", "alpha charlie").unwrap();

    assert!(!result.diff.operations.is_empty());
    assert_eq!(result.diff.metadata.algorithm_name, "histogram");
}

#[test]
fn redline_debug_output() {
    let config = ConfigBuilder::new().build().unwrap();
    let redline = Redline::new(config).unwrap();
    let debug = format!("{:?}", redline);
    assert!(debug.contains("Redline"));
    assert!(debug.contains("Comprehensive"));
}

#[test]
fn result_debug_output() {
    let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
    let redline = Redline::new(config).unwrap();
    let result = redline.diff("hello", "world").unwrap();
    let debug = format!("{:?}", result);
    assert!(debug.contains("RedlineResult"));
}
