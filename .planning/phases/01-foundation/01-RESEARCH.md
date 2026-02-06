# Phase 1: Foundation - Research

**Researched:** 2026-02-06
**Domain:** Rust core types, arena allocation, string interning, error handling, feature flags, WASM/Miri CI
**Confidence:** HIGH

## Summary

Phase 1 establishes the core type system and build infrastructure. The standard approach uses bumpalo 3.19 for arena allocation with pointer-stable `alloc_str`, hashbrown 0.16 for the interning lookup table, thiserror 2.0 for error hierarchies with no_std support, and static_assertions for compile-time trait verification. The two-phase TextStore pattern (mutable builder -> immutable Arc-wrapped reader) solves bumpalo's `!Sync` limitation.

**Primary recommendation:** Build TextStore as two-phase: `TextStoreBuilder` (mutable, owns Bump + HashMap, does interning) that freezes into `Arc<TextStore>` (immutable, owns leaked bump data + Vec for index lookup).

## Standard Stack

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| bumpalo | 3.19 | Arena allocation for string interning | `alloc_str` for stable pointers, `!Sync` by design, no_std compatible. Features: `collections`, `allocator-api2` |
| hashbrown | 0.16 | Hash map for string deduplication | SwissTable, `allocator-api2` on stable, FoldHash default (changed from AHash). `raw_entry` API deprecated -- use standard `HashMap` entry API |
| thiserror | 2.0 | Error type derivation | `no_std` support via `default-features = false`. `core::error::Error` (stable Rust 1.81+) |
| static_assertions | 1.1 | Compile-time trait/size checks | `assert_impl_all!`, `assert_eq_size!`, zero runtime cost |
| proptest | 1.10 | Property-based testing | For CharMapping composition round-trip tests |

### Cargo.toml Dependencies

```toml
[dependencies]
bumpalo = { version = "3.19", features = ["collections", "allocator-api2"] }
hashbrown = { version = "0.16", default-features = false, features = ["allocator-api2"] }
thiserror = { version = "2.0", default-features = false }
static_assertions = "1.1"

[dev-dependencies]
proptest = "1.10"
```

## Architecture Patterns

### Project Structure

```
crates/core/src/
  lib.rs              # Crate root, feature flags, re-exports
  types/
    mod.rs            # Module declarations
    string_id.rs      # StringId newtype (u32, Copy)
    span.rs           # Span type (byte offsets)
    token.rs          # Token type (StringId, Span, TokenKind)
    token_kind.rs     # TokenKind enum
  store/
    mod.rs
    text_store.rs     # TextStoreBuilder + TextStore (two-phase)
    char_mapping.rs   # CharMapping with composition
  error/
    mod.rs            # RedlineError hierarchy
```

### Pattern 1: Two-Phase TextStore (Build/Read)

**Builder phase** (mutable, single-threaded):
- Owns `Bump` arena + `HashMap<&'static str, StringId>` + `Vec<&'static str>`
- `intern(&mut self, s: &str) -> StringId`: allocates in bump, deduplicates via HashMap
- Uses `unsafe { transmute }` to extend bump-allocated `&'bump str` to `&'static str`

**Read phase** (immutable, thread-safe via Arc):
- `build(self) -> Arc<TextStore>`: consumes builder, moves Bump into TextStore
- `resolve(&self, id: StringId) -> &str`: Vec index lookup, O(1)
- `Arc<TextStore>` is `Send + Sync`

**Safety:** The transmute is sound because:
1. Bumpalo allocations are pointer-stable (new chunks, old never moved/freed)
2. The Bump is moved into TextStore, living as long as the references
3. TextStore is only accessible through Arc, preventing use-after-free

### Pattern 2: Token with Copy Semantics

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub text: StringId,        // 4 bytes
    pub span: Span,            // 8 bytes (2x u32)
    pub kind: TokenKind,       // 8 bytes (enum discriminant + StringId)
}
// Total: ~20 bytes, likely 24 with padding

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Word,
    Punctuation,
    Whitespace,
    Number,
    Symbol,
    Unknown,
    Custom(StringId),  // NOT &'static str -- allows dynamic token kinds
}
```

### Pattern 3: Span with Byte Offsets

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: u32,  // byte offset (u32 = 4GB max)
    pub end: u32,    // exclusive
}
```

Methods: `new`, `len`, `is_empty`, `contains`, `overlaps`, `merge`, `split`, `slice_from`

### Pattern 4: CharMapping with Composition

Vec of `(u32, u32)` entries mapping original -> normalized byte offsets. Sorted by original position. Binary search for lookups with linear interpolation for gaps. `compose()` chains two mappings: A->B composed with B->C yields A->C.

**Open question:** Interpolation strategy for positions between mapped entries. Start simple, property-test, refine.

### Pattern 5: Feature Flags

```rust
// lib.rs
#[cfg(all(feature = "python", feature = "wasm"))]
compile_error!("Features `python` and `wasm` are mutually exclusive.");
```

Features: `default = ["std"]`, `std`, `async`, `serde`, `tracing`, `python`, `wasm`

### Pattern 6: Error Hierarchy

```rust
#[derive(Debug, Error)]
pub enum RedlineError {
    #[error("Text store error: {0}")]
    Store(#[from] StoreError),
    #[error("Tokenization error: {0}")]
    Tokenize(#[from] TokenizeError),
    #[error("Normalization error: {0}")]
    Normalize(#[from] NormalizeError),
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
}
```

Sub-errors: `StoreError` (StringNotFound, CapacityExceeded), `TokenizeError` (InvalidSpan, EmptyInput), `NormalizeError` (InvalidUnicode), `ConfigError` (UnknownTokenKind)

## Anti-Patterns to Avoid

- **SmallVec in Token** -- not Copy, use single Span
- **TokenKind::Custom(&'static str)** -- prevents dynamic kinds, use Custom(StringId)
- **Sharing Bump across threads** -- Bump is !Sync, use two-phase pattern
- **hashbrown raw_entry API** -- deprecated in 0.15+, use standard entry API
- **Relying on Drop in bump arena** -- bumpalo doesn't run Drop for arena values

## Pitfalls

1. **Bump !Sync** -- use two-phase pattern, never wrap in Arc<Mutex<Bump>>
2. **Token Copy + SmallVec** -- SmallVec not Copy, resolved: single Span
3. **transmute without keeping arena alive** -- move Bump into TextStore
4. **Feature unification** -- Cargo unifies features additively; compile_error! is safety net
5. **hashbrown 0.16 default hasher** -- changed to FoldHash (no action needed)
6. **Miri and bumpalo** -- test with both Stacked Borrows and Tree Borrows
7. **WASM + std deps** -- ensure no_std support, test with --no-default-features --features wasm

## Don't Hand-Roll

| Problem | Use Instead |
|---------|-------------|
| String dedup | hashbrown HashMap |
| Error boilerplate | thiserror 2.0 derive |
| Compile-time checks | static_assertions |
| Property testing | proptest |
| Bump allocation | bumpalo |

The transmute to 'static in TextStore is the ONLY hand-rolled unsafe needed.

## CI Requirements

- `cargo test --all-features` and `cargo test --no-default-features`
- `cargo check --target wasm32-unknown-unknown --no-default-features --features wasm`
- `cargo +nightly miri test` (Stacked Borrows)
- `MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test` (Tree Borrows)

## Open Questions

1. CharMapping interpolation for positions between mapped entries -- start simple, property-test
2. TextStore transmute under Tree Borrows -- test with Miri flag
3. Token exact padded size -- verify with `assert_eq_size!`
4. static_assertions maintenance -- stable macro crate, unlikely to need updates

---

## RESEARCH COMPLETE

**Confidence:** HIGH
**Ready for planning.**
