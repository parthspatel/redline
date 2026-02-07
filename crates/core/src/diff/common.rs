//! Common utilities for diff computation: prefix/suffix stripping, token interning, apply_operations.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::text_store::TextStore;
use crate::token::Token;

use super::edit_operation::{EditKind, EditOperation};

/// Intern tokens from two (possibly different) TextStores into a shared u32 ID space.
///
/// Returns `(source_ids, target_ids)` where tokens with the same text get the same
/// u32 ID regardless of which TextStore they came from.
pub fn build_token_indices(
    source_tokens: &[Token],
    source_store: &TextStore,
    target_tokens: &[Token],
    target_store: &TextStore,
) -> (Vec<u32>, Vec<u32>) {
    let mut intern: HashMap<&str, u32> = HashMap::new();
    let mut next_id: u32 = 0;

    let source_ids: Vec<u32> = source_tokens
        .iter()
        .map(|tok| {
            let text = source_store.resolve(tok.text_id).unwrap();
            *intern.entry(text).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();

    let target_ids: Vec<u32> = target_tokens
        .iter()
        .map(|tok| {
            let text = target_store.resolve(tok.text_id).unwrap();
            *intern.entry(text).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();

    (source_ids, target_ids)
}

/// Strip common prefix and suffix from two token ID slices.
///
/// Returns `(prefix_len, suffix_len, inner_source, inner_target)`.
pub fn strip_common_affixes<'a>(
    source: &'a [u32],
    target: &'a [u32],
) -> (usize, usize, &'a [u32], &'a [u32]) {
    let prefix_len = source
        .iter()
        .zip(target.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let source_rest = &source[prefix_len..];
    let target_rest = &target[prefix_len..];

    let suffix_len = source_rest
        .iter()
        .rev()
        .zip(target_rest.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();

    let inner_source = &source_rest[..source_rest.len() - suffix_len];
    let inner_target = &target_rest[..target_rest.len() - suffix_len];

    (prefix_len, suffix_len, inner_source, inner_target)
}

/// Reconstruct the target sequence from source, target, and a list of edit operations.
///
/// Useful for property-test roundtrip verification: `apply_operations(src, tgt, diff(src, tgt)) == tgt`.
pub fn apply_operations(source: &[u32], target: &[u32], ops: &[EditOperation]) -> Vec<u32> {
    let mut result = Vec::new();
    for op in ops {
        match op.kind {
            EditKind::Equal => {
                result.extend_from_slice(&source[op.source_start as usize..op.source_end as usize]);
            }
            EditKind::Delete => {}
            EditKind::Insert | EditKind::Replace => {
                result.extend_from_slice(&target[op.target_start as usize..op.target_end as usize]);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Span;
    use crate::text_store::TextStoreBuilder;
    use crate::token::{Token, TokenKind};

    #[test]
    fn build_token_indices_same_text_same_id() {
        let mut b1 = TextStoreBuilder::new();
        let hello1 = b1.intern("hello");
        let world1 = b1.intern("world");
        let store1 = b1.build();

        let mut b2 = TextStoreBuilder::new();
        let hello2 = b2.intern("hello");
        let foo2 = b2.intern("foo");
        let store2 = b2.build();

        let src = vec![
            Token::new(hello1, Span::new(0, 5), TokenKind::Regular),
            Token::new(world1, Span::new(6, 11), TokenKind::Regular),
        ];
        let tgt = vec![
            Token::new(hello2, Span::new(0, 5), TokenKind::Regular),
            Token::new(foo2, Span::new(6, 9), TokenKind::Regular),
        ];

        let (src_ids, tgt_ids) = build_token_indices(&src, &store1, &tgt, &store2);
        assert_eq!(src_ids[0], tgt_ids[0]); // "hello" == "hello"
        assert_ne!(src_ids[1], tgt_ids[1]); // "world" != "foo"
    }

    #[test]
    fn build_token_indices_empty() {
        let store = TextStoreBuilder::new().build();
        let (s, t) = build_token_indices(&[], &store, &[], &store);
        assert!(s.is_empty());
        assert!(t.is_empty());
    }

    #[test]
    fn strip_prefix_only() {
        let (p, s, is, it) = strip_common_affixes(&[0, 1, 2, 3], &[0, 1, 5, 6]);
        assert_eq!(p, 2);
        assert_eq!(s, 0);
        assert_eq!(is, &[2, 3]);
        assert_eq!(it, &[5, 6]);
    }

    #[test]
    fn strip_both_sides() {
        let (p, s, is, it) = strip_common_affixes(&[0, 1, 2, 3], &[0, 5, 6, 3]);
        assert_eq!(p, 1);
        assert_eq!(s, 1);
        assert_eq!(is, &[1, 2]);
        assert_eq!(it, &[5, 6]);
    }

    #[test]
    fn strip_identical() {
        let (p, s, is, it) = strip_common_affixes(&[0, 1, 2, 3], &[0, 1, 2, 3]);
        assert_eq!(p, 4);
        assert_eq!(s, 0);
        assert!(is.is_empty());
        assert!(it.is_empty());
    }

    #[test]
    fn strip_empty() {
        let e: [u32; 0] = [];
        let (p, s, is, it) = strip_common_affixes(&e, &e);
        assert_eq!(p, 0);
        assert_eq!(s, 0);
        assert!(is.is_empty());
        assert!(it.is_empty());
    }

    #[test]
    fn strip_no_common() {
        let (p, s, is, it) = strip_common_affixes(&[1, 2, 3], &[4, 5, 6]);
        assert_eq!(p, 0);
        assert_eq!(s, 0);
        assert_eq!(is, &[1, 2, 3]);
        assert_eq!(it, &[4, 5, 6]);
    }

    #[test]
    fn apply_operations_roundtrip() {
        let source = [0, 1, 2, 3, 4];
        let target = [0, 1, 5, 6, 4];
        let ops = vec![
            EditOperation::equal(0, 2, 0, 2),
            EditOperation::replace(2, 4, 2, 4),
            EditOperation::equal(4, 5, 4, 5),
        ];
        assert_eq!(apply_operations(&source, &target, &ops), target);
    }

    #[test]
    fn apply_operations_delete_and_insert() {
        let source = [0, 1, 2];
        let target = [0, 3, 4, 2];
        let ops = vec![
            EditOperation::equal(0, 1, 0, 1),
            EditOperation::delete(1, 2, 1),
            EditOperation::insert(2, 1, 3),
            EditOperation::equal(2, 3, 3, 4),
        ];
        assert_eq!(apply_operations(&source, &target, &ops), target);
    }

    #[test]
    fn apply_operations_all_equal() {
        let data = [1, 2, 3];
        let ops = vec![EditOperation::equal(0, 3, 0, 3)];
        assert_eq!(apply_operations(&data, &data, &ops), data);
    }

    #[test]
    fn apply_operations_empty() {
        let e: [u32; 0] = [];
        assert!(apply_operations(&e, &e, &[]).is_empty());
    }
}
