//! Integration tests verifying all Phase 1 success criteria.

use redline_core::text_store::{StringId, TextStore, TextStoreBuilder};
use redline_core::{CharMapping, RedlineError, Span, Token, TokenKind};
use static_assertions::assert_impl_all;
use std::sync::Arc;

// ── Success Criterion 1: Token traits ────────────────────────────────────────
assert_impl_all!(Token: Copy, Clone, core::fmt::Debug, PartialEq);
assert_impl_all!(StringId: Copy, Clone, core::fmt::Debug, PartialEq, Eq, core::hash::Hash);
assert_impl_all!(Span: Copy, Clone, core::fmt::Debug, PartialEq, Eq, core::hash::Hash);

#[test]
fn criterion_1_token_size_and_traits() {
    // Token must be <= 24 bytes and Copy
    assert!(core::mem::size_of::<Token>() <= 24);

    // Use the TextStoreBuilder to obtain a valid StringId
    let mut builder = TextStoreBuilder::new();
    let id = builder.intern("test");

    let t = Token::new(id, Span::new(0, 5), TokenKind::Regular);
    let t2 = t; // Copy
    let t3 = t; // Still valid
    assert_eq!(t2, t3);
}

// ── Success Criterion 2: TextStore 10K ───────────────────────────────────────
#[test]
fn criterion_2_textstore_10k_strings() {
    let mut builder = TextStoreBuilder::new();
    let mut ids = Vec::new();

    for i in 0..10_000u32 {
        let s = format!("token_{i}");
        ids.push((builder.intern(&s), s));
    }

    // Dedup: re-intern "token_0" returns same ID
    let id_dup = builder.intern("token_0");
    assert_eq!(id_dup, ids[0].0);

    // Promote to Arc<TextStore>
    let store: Arc<TextStore> = builder.build();

    // All 10K lookups return correct values
    for (id, expected) in &ids {
        assert_eq!(store.resolve(*id).unwrap(), expected.as_str());
    }

    assert_eq!(store.len(), 10_000);
}

// ── Success Criterion 3: CharMapping 3-compose round-trip ────────────────────
#[test]
fn criterion_3_char_mapping_compose_round_trip() {
    // Simulate 3 normalization steps:
    // Step 1: Remove leading whitespace (shift positions left by 2)
    //   "  Hello World" -> "Hello World"
    //   orig 2->norm 0, orig 3->norm 1, ... orig 12->norm 10
    let m1 = CharMapping::new((0..11).map(|i| (i + 2, i)).collect()).unwrap();

    // Step 2: Lowercase (identity mapping, positions don't change)
    //   "Hello World" -> "hello world"
    let m2 = CharMapping::identity(11).unwrap();

    // Step 3: Collapse double space (if any) — identity for this input
    let m3 = CharMapping::identity(11).unwrap();

    // Compose all 3
    let composed_12 = m1.compose(&m2).unwrap();
    let composed_123 = composed_12.compose(&m3).unwrap();

    // Verify: original position 2 -> normalized position 0
    assert_eq!(composed_123.to_normalized(2).unwrap(), 0);
    // Verify: original position 7 -> normalized position 5
    assert_eq!(composed_123.to_normalized(7).unwrap(), 5);

    // Verify round-trip: to_original(to_normalized(pos)) == pos
    for orig_pos in 2..13u32 {
        let norm = composed_123.to_normalized(orig_pos);
        if let Some(n) = norm.ok() {
            let back = composed_123.to_original(n);
            if let Some(o) = back.ok() {
                assert_eq!(o, orig_pos, "round-trip failed for position {orig_pos}");
            }
        }
    }
}

// ── Success Criterion 4: Types are Send + Sync ───────────────────────────────
#[test]
fn criterion_4_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Token>();
    assert_send_sync::<StringId>();
    assert_send_sync::<Span>();
    assert_send_sync::<TextStore>();
    assert_send_sync::<RedlineError>();
}

// ── Success Criterion 5: Miri-safe arena operations ──────────────────────────
#[test]
fn criterion_5_miri_safe_store_operations() {
    let mut builder = TextStoreBuilder::new();

    // Intern strings of varying sizes
    let mut ids = Vec::new();
    for i in 0..100u32 {
        let s: String = (0..i).map(|_| 'x').collect();
        ids.push((builder.intern(&s), s));
    }

    // Promote to Arc<TextStore>
    let store = builder.build();

    // Verify all lookups
    for (id, expected) in &ids {
        assert_eq!(store.resolve(*id).unwrap(), expected.as_str());
    }

    // Clone Arc, verify both references work
    let store2 = Arc::clone(&store);
    for (id, expected) in &ids {
        assert_eq!(store2.resolve(*id).unwrap(), expected.as_str());
    }
}

// ── End-to-end tokenization simulation ───────────────────────────────────────
#[test]
fn end_to_end_tokenization_simulation() {
    let mut builder = TextStoreBuilder::new();

    // Simulate tokenizing "Hello, world!"
    // Tokens: [CLS], Hello, ",", world, "!", [SEP]
    let id_cls = builder.intern("[CLS]");
    let id_hello = builder.intern("Hello");
    let id_comma = builder.intern(",");
    let id_world = builder.intern("world");
    let id_bang = builder.intern("!");
    let id_sep = builder.intern("[SEP]");

    let tokens = vec![
        Token::new(id_cls, Span::new(0, 0), TokenKind::Special),
        Token::new(id_hello, Span::new(0, 5), TokenKind::Regular),
        Token::new(id_comma, Span::new(5, 6), TokenKind::Regular),
        Token::new(id_world, Span::new(8, 13), TokenKind::Regular), // skip space at 7
        Token::new(id_bang, Span::new(13, 14), TokenKind::Regular),
        Token::new(id_sep, Span::new(14, 14), TokenKind::Special),
    ];

    // Build store
    let store = builder.build();

    // Verify tokens reference correct strings
    assert_eq!(store.resolve(tokens[0].text_id).unwrap(), "[CLS]");
    assert_eq!(store.resolve(tokens[1].text_id).unwrap(), "Hello");
    assert_eq!(store.resolve(tokens[3].text_id).unwrap(), "world");
    assert_eq!(store.resolve(tokens[5].text_id).unwrap(), "[SEP]");

    // Verify special tokens
    assert!(tokens[0].is_special());
    assert!(tokens[5].is_special());
    assert!(!tokens[1].is_special());

    // Verify span containment
    let full_span = Span::new(0, 14);
    for t in &tokens {
        assert!(full_span.contains(&t.span));
    }

    // Verify adjacent tokens don't overlap (except specials which may have empty spans)
    let real_tokens: Vec<_> = tokens.iter().filter(|t| !t.span.is_empty()).collect();
    for w in real_tokens.windows(2) {
        assert!(
            !w[0].span.overlaps(&w[1].span),
            "tokens {:?} and {:?} should not overlap",
            w[0],
            w[1]
        );
    }
}
