//! Integration tests for diff pipeline: known outputs, DiffComputer, statistics, performance safety.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Arc;

use redline_core::diff::{
    DiffAlgorithm, DiffComputer, DiffError, EditKind, Histogram, Myers, apply_operations,
};
use redline_core::process::ProcessedText;
use redline_core::tokenize::WordTokenizer;
use redline_core::{ExecutionMode, TextProcessor};

/// Helper to create ProcessedText from a string (no normalizers, WordTokenizer).
fn process_text(input: &str) -> Arc<ProcessedText> {
    let processor =
        TextProcessor::new(vec![], Box::new(WordTokenizer), ExecutionMode::All).unwrap();
    Arc::new(processor.process(input).unwrap())
}

// ---- Known-output correctness (success criteria 2, 3) ----

#[test]
fn the_cat_sat_vs_the_dog_sat_myers() {
    let source = process_text("the cat sat");
    let target = process_text("the dog sat on the mat");
    let result = DiffComputer::new().compute(source, target, None).unwrap();

    let kinds: Vec<EditKind> = result.operations.iter().map(|op| op.kind).collect();
    assert_eq!(
        kinds,
        vec![
            EditKind::Equal,
            EditKind::Replace,
            EditKind::Equal,
            EditKind::Insert
        ],
        "Expected [Equal, Replace, Equal, Insert], got {:?}",
        kinds
    );
}

#[test]
fn histogram_equivalent() {
    let source = process_text("the cat sat");
    let target = process_text("the dog sat on the mat");
    let result = DiffComputer::with_algorithm(Box::new(Histogram::new()))
        .compute(source.clone(), target.clone(), None)
        .unwrap();

    // Roundtrip verification via raw algorithm (DiffComputer interns internally)
    let (src_ids, tgt_ids) = redline_core::diff::build_token_indices(
        &source.tokens,
        &source.text_store,
        &target.tokens,
        &target.text_store,
    );
    let output = Histogram::new().compute(&src_ids, &tgt_ids, None).unwrap();
    let reconstructed = apply_operations(&src_ids, &tgt_ids, &output.operations);
    assert_eq!(reconstructed, tgt_ids, "Histogram roundtrip failed");
    assert!(!result.is_approximate());
}

#[test]
fn identical_texts() {
    let source = process_text("hello world");
    let target = process_text("hello world");
    let result = DiffComputer::new().compute(source, target, None).unwrap();
    assert!(
        (result.statistics.similarity_ratio - 1.0).abs() < f64::EPSILON,
        "Identical texts should have similarity 1.0"
    );
    assert_eq!(result.statistics.edit_distance, 0);
    for op in &result.operations {
        assert_eq!(op.kind, EditKind::Equal);
    }
}

#[test]
fn completely_different() {
    let source = process_text("aaa bbb ccc");
    let target = process_text("xxx yyy zzz");
    let result = DiffComputer::new().compute(source, target, None).unwrap();
    assert!(
        result.statistics.similarity_ratio.abs() < f64::EPSILON,
        "Completely different texts should have similarity 0.0"
    );
    // No Equal operations — all tokens differ
    assert_eq!(
        result.statistics.equal_count, 0,
        "Completely different texts should have no Equal ops"
    );
    assert_eq!(result.statistics.lcs_length, 0);
}

#[test]
fn empty_vs_nonempty() {
    let empty = process_text("");
    let nonempty = process_text("hello world");

    // Empty source -> all inserts
    let result = DiffComputer::new()
        .compute(empty.clone(), nonempty.clone(), None)
        .unwrap();
    assert!(result.operations.is_empty() || result.statistics.insert_count > 0);

    // Empty target -> all deletes
    let result = DiffComputer::new().compute(nonempty, empty, None).unwrap();
    assert!(result.operations.is_empty() || result.statistics.delete_count > 0);
}

// ---- Statistics validation ----

#[test]
fn statistics_validation() {
    let source = process_text("the cat sat");
    let target = process_text("the dog sat on the mat");
    let result = DiffComputer::new().compute(source, target, None).unwrap();
    let stats = &result.statistics;

    assert!(
        stats.equal_count > 0,
        "Should have equal ops ('the', 'sat')"
    );
    assert!(
        stats.replace_count > 0,
        "Should have replace ops ('cat'->'dog')"
    );
    assert!(
        stats.insert_count > 0,
        "Should have insert ops ('on the mat')"
    );
    assert!(
        stats.similarity_ratio > 0.0 && stats.similarity_ratio < 1.0,
        "Similarity should be between 0 and 1 exclusive, got {}",
        stats.similarity_ratio
    );
    assert!(stats.lcs_length >= 2, "LCS should be >= 2 ('the', 'sat')");
}

// ---- Hunks ----

#[test]
fn hunks_test() {
    let source = process_text("a b c d e f g h i j");
    let target = process_text("a b c x e f g h i j");
    let result = DiffComputer::new().compute(source, target, None).unwrap();

    // With context=1, should get one hunk around the 'd'->'x' change
    let hunks = result.hunks(1);
    assert_eq!(hunks.len(), 1, "Should have exactly 1 hunk");
    assert!(
        hunks[0].iter().any(|op| op.kind != EditKind::Equal),
        "Hunk should contain non-Equal ops"
    );

    // With context=0, should get just the change operations
    let hunks_zero = result.hunks(0);
    assert_eq!(hunks_zero.len(), 1);
    for op in hunks_zero[0] {
        assert_ne!(
            op.kind,
            EditKind::Equal,
            "Context=0 hunks should not include Equal ops"
        );
    }
}

// ---- D-threshold safety (success criteria 5) ----

#[test]
fn d_threshold_safety_10k_dissimilar() {
    let source: Vec<u32> = (0..10000).collect();
    let target: Vec<u32> = (10000..20000).collect();

    let start = std::time::Instant::now();
    let output = Myers::with_threshold(100)
        .compute(&source, &target, None)
        .unwrap();
    let elapsed = start.elapsed();

    assert!(
        output.is_approximate,
        "Should be approximate with threshold=100"
    );
    assert_eq!(output.threshold_used, Some(100));
    assert!(
        elapsed.as_secs() < 5,
        "10K dissimilar tokens with threshold should complete in <5s, took {:?}",
        elapsed
    );
}

// ---- Cancellation ----

#[test]
fn cancellation() {
    let source: Vec<u32> = (0..100).collect();
    let target: Vec<u32> = (100..200).collect();

    let mut called = false;
    let result = Myers::without_threshold().compute(
        &source,
        &target,
        Some(&mut |_progress: f64| {
            called = true;
            ControlFlow::Break(())
        }),
    );

    match result {
        Err(DiffError::Cancelled) => {
            assert!(called, "Progress callback should have been called");
        }
        other => panic!("Expected DiffError::Cancelled, got {:?}", other),
    }
}

// ---- HashMap storable (success criteria 6) ----

#[test]
fn hashmap_storable() {
    let source = process_text("hello world");
    let target = process_text("hello earth");
    let result = DiffComputer::new().compute(source, target, None).unwrap();

    let mut map: HashMap<String, _> = HashMap::new();
    let cloned = result.clone();
    map.insert("test".to_string(), result);
    assert_eq!(map["test"].operations, cloned.operations);
}

// ---- Metadata ----

#[test]
fn metadata_algorithm_name_myers() {
    let source = process_text("hello");
    let target = process_text("world");
    let result = DiffComputer::new().compute(source, target, None).unwrap();
    assert_eq!(result.metadata.algorithm_name, "myers");
}

#[test]
fn metadata_algorithm_name_histogram() {
    let source = process_text("hello");
    let target = process_text("world");
    let result = DiffComputer::with_algorithm(Box::new(Histogram::new()))
        .compute(source, target, None)
        .unwrap();
    assert_eq!(result.metadata.algorithm_name, "histogram");
}

#[test]
fn is_approximate_flag() {
    // Normal diff -> not approximate
    let source = process_text("hello world");
    let target = process_text("hello earth");
    let result = DiffComputer::new().compute(source, target, None).unwrap();
    assert!(!result.is_approximate());

    // Threshold-hit diff -> approximate
    let source_ids: Vec<u32> = (0..1000).collect();
    let target_ids: Vec<u32> = (1000..2000).collect();
    let output = Myers::with_threshold(5)
        .compute(&source_ids, &target_ids, None)
        .unwrap();
    assert!(output.is_approximate);
}
