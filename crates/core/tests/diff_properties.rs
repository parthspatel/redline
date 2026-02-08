//! Property-based tests for diff algorithms: roundtrip, completeness, consistency.

use proptest::prelude::*;
use redline_core::diff::{
    DiffAlgorithm, DiffStatistics, EditKind, EditOperation, Histogram, Myers, apply_operations,
};

/// Generate (source, target) token pairs with configurable vocabulary size.
fn arb_token_pair(max_len: usize) -> impl Strategy<Value = (Vec<u32>, Vec<u32>)> {
    (1..=20usize).prop_flat_map(move |vocab_size| {
        let source = prop::collection::vec(0..vocab_size as u32, 0..max_len);
        let target = prop::collection::vec(0..vocab_size as u32, 0..max_len);
        (source, target)
    })
}

/// Verify that operations completely cover source and target with no gaps.
fn assert_operations_complete(source_len: usize, target_len: usize, ops: &[EditOperation]) {
    let mut src_pos = 0u32;
    let mut tgt_pos = 0u32;
    for op in ops {
        assert_eq!(
            op.source_start, src_pos,
            "Source gap at {src_pos}: op={op:?}"
        );
        assert_eq!(
            op.target_start, tgt_pos,
            "Target gap at {tgt_pos}: op={op:?}"
        );
        src_pos = op.source_end;
        tgt_pos = op.target_end;
    }
    assert_eq!(src_pos, source_len as u32, "Source not fully covered");
    assert_eq!(tgt_pos, target_len as u32, "Target not fully covered");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn myers_roundtrip((source, target) in arb_token_pair(200)) {
        let output = Myers::new().compute(&source, &target, None).unwrap();
        let reconstructed = apply_operations(&source, &target, &output.operations);
        prop_assert_eq!(reconstructed, target);
    }

    #[test]
    fn histogram_roundtrip((source, target) in arb_token_pair(200)) {
        let output = Histogram::new().compute(&source, &target, None).unwrap();
        let reconstructed = apply_operations(&source, &target, &output.operations);
        prop_assert_eq!(reconstructed, target);
    }

    #[test]
    fn myers_no_affixes_roundtrip((source, target) in arb_token_pair(200)) {
        let output = Myers::new().without_common_affixes()
            .compute(&source, &target, None).unwrap();
        let reconstructed = apply_operations(&source, &target, &output.operations);
        prop_assert_eq!(reconstructed, target);
    }

    #[test]
    fn myers_operations_complete((source, target) in arb_token_pair(200)) {
        let output = Myers::new().compute(&source, &target, None).unwrap();
        assert_operations_complete(source.len(), target.len(), &output.operations);
    }

    #[test]
    fn histogram_operations_complete((source, target) in arb_token_pair(200)) {
        let output = Histogram::new().compute(&source, &target, None).unwrap();
        assert_operations_complete(source.len(), target.len(), &output.operations);
    }

    #[test]
    fn equal_ops_consistent((source, target) in arb_token_pair(200)) {
        for algo in &[
            Myers::new().compute(&source, &target, None).unwrap(),
            Histogram::new().compute(&source, &target, None).unwrap(),
        ] {
            for op in &algo.operations {
                if op.kind == EditKind::Equal {
                    let src_slice = &source[op.source_start as usize..op.source_end as usize];
                    let tgt_slice = &target[op.target_start as usize..op.target_end as usize];
                    prop_assert_eq!(src_slice, tgt_slice,
                        "Equal op mismatch: src={:?}, tgt={:?}", src_slice, tgt_slice);
                }
            }
        }
    }

    #[test]
    fn statistics_consistency((source, target) in arb_token_pair(200)) {
        let output = Myers::new().compute(&source, &target, None).unwrap();
        let stats = DiffStatistics::from_operations(
            &output.operations, source.len(), target.len(),
        );

        prop_assert!(stats.similarity_ratio >= 0.0 && stats.similarity_ratio <= 1.0,
            "similarity_ratio out of range: {}", stats.similarity_ratio);
        prop_assert_eq!(stats.edit_distance,
            stats.delete_count + stats.insert_count + stats.replace_count);
        prop_assert!(stats.lcs_length <= source.len().min(target.len()),
            "lcs_length {} > min(src={}, tgt={})", stats.lcs_length, source.len(), target.len());

        if source == target {
            prop_assert!((stats.similarity_ratio - 1.0).abs() < f64::EPSILON,
                "Identical inputs should have similarity 1.0, got {}", stats.similarity_ratio);
            prop_assert_eq!(stats.edit_distance, 0,
                "Identical inputs should have edit_distance 0");
        }
    }

    #[test]
    fn identical_inputs(source in prop::collection::vec(0..20u32, 0..200)) {
        let output = Myers::new().compute(&source, &source, None).unwrap();
        for op in &output.operations {
            prop_assert_eq!(op.kind, EditKind::Equal,
                "Identical inputs should produce only Equal ops, got {:?}", op);
        }
    }
}

#[test]
fn empty_inputs_both_algorithms() {
    let empty: Vec<u32> = vec![];
    let nonempty: Vec<u32> = vec![1, 2, 3];

    for algo_name in ["myers", "histogram"] {
        let cases: Vec<(Vec<u32>, Vec<u32>)> = vec![
            (empty.clone(), empty.clone()),
            (empty.clone(), nonempty.clone()),
            (nonempty.clone(), empty.clone()),
        ];

        for (source, target) in &cases {
            let output = match algo_name {
                "myers" => Myers::new().compute(source, target, None).unwrap(),
                "histogram" => Histogram::new().compute(source, target, None).unwrap(),
                _ => unreachable!(),
            };
            let reconstructed = apply_operations(source, target, &output.operations);
            assert_eq!(
                reconstructed, *target,
                "{algo_name}: roundtrip failed for {:?} vs {:?}",
                source, target
            );
        }
    }
}
