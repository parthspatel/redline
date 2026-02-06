//! Two-phase text interning: [`TextStoreBuilder`] (mutable) → [`TextStore`] (immutable, shared).
//!
//! Strings are interned as [`StringId`] values during the builder phase,
//! then the builder is promoted to an immutable [`TextStore`] wrapped in `Arc`.
//!
//! # Example
//!
//! ```
//! use redline_core::text_store::{TextStoreBuilder, TextStore, StringId};
//! use std::sync::Arc;
//!
//! let mut builder = TextStoreBuilder::new();
//! let id_hello = builder.intern("hello");
//! let id_world = builder.intern("world");
//!
//! // Deduplication: same string → same ID
//! assert_eq!(builder.intern("hello"), id_hello);
//!
//! // Promote to immutable, shared store
//! let store: Arc<TextStore> = builder.build();
//! assert_eq!(store.resolve(id_hello).unwrap(), "hello");
//! assert_eq!(store.resolve(id_world).unwrap(), "world");
//! ```

#[cfg(not(feature = "std"))]
use alloc::{borrow::ToOwned, string::String, sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::sync::Arc;

use hashbrown::HashMap;

use crate::error::StoreError;

// ── StringId ─────────────────────────────────────────────────────────────────

/// Opaque identifier for an interned string.
///
/// This is a lightweight `Copy` type wrapping a `u32` index.
/// It is only meaningful within the [`TextStoreBuilder`] or [`TextStore`] that created it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StringId(u32);

impl StringId {
    /// Returns the raw `u32` value of this identifier.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

// ── TextStoreBuilder ─────────────────────────────────────────────────────────

/// Mutable builder for interning strings with deduplication.
///
/// Use [`intern`](Self::intern) to add strings and receive [`StringId`] handles,
/// then call [`build`](Self::build) to promote into an immutable, shared [`TextStore`].
pub struct TextStoreBuilder {
    /// Owned strings indexed by `StringId`. The index into this vec IS the `StringId`.
    strings: Vec<String>,
    /// Deduplication map: string content → existing StringId.
    dedup: HashMap<String, StringId>,
}

impl TextStoreBuilder {
    /// Create a new empty builder.
    #[inline]
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            dedup: HashMap::new(),
        }
    }

    /// Intern a string, returning its [`StringId`].
    ///
    /// If the string has already been interned, returns the existing ID.
    /// Otherwise, stores the string and returns a fresh ID.
    pub fn intern(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.dedup.get(s) {
            return id;
        }
        let id = StringId(self.strings.len() as u32);
        self.strings.push(s.to_owned());
        self.dedup.insert(s.to_owned(), id);
        id
    }

    /// Resolve a [`StringId`] back to its string slice.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::IdNotFound`] if the ID is not valid for this builder.
    #[inline]
    pub fn resolve(&self, id: StringId) -> Result<&str, StoreError> {
        self.strings
            .get(id.0 as usize)
            .map(|s| s.as_str())
            .ok_or(StoreError::IdNotFound(id.0))
    }

    /// Returns the number of unique strings interned.
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns `true` if no strings have been interned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Promote this builder into an immutable, shared [`TextStore`].
    ///
    /// The builder is consumed. The returned `Arc<TextStore>` can be cheaply cloned
    /// and shared across threads.
    pub fn build(self) -> Arc<TextStore> {
        Arc::new(TextStore {
            strings: self.strings,
        })
    }
}

impl Default for TextStoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── TextStore ────────────────────────────────────────────────────────────────

/// Immutable, shared string store.
///
/// Created via [`TextStoreBuilder::build`]. All [`StringId`] values obtained
/// from the builder remain valid for lookups on this store.
pub struct TextStore {
    /// Immutable collection of interned strings.
    strings: Vec<String>,
}

impl TextStore {
    /// Resolve a [`StringId`] back to its string slice.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::IdNotFound`] if the ID is not valid for this store.
    #[inline]
    pub fn resolve(&self, id: StringId) -> Result<&str, StoreError> {
        self.strings
            .get(id.0 as usize)
            .map(|s| s.as_str())
            .ok_or(StoreError::IdNotFound(id.0))
    }

    /// Returns the number of unique strings in the store.
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns `true` if the store contains no strings.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    // ── Static trait assertions ──────────────────────────────────────
    assert_impl_all!(StringId: Copy, Clone, core::fmt::Debug, PartialEq, Eq, core::hash::Hash, Send, Sync);

    // ── StringId ─────────────────────────────────────────────────────

    #[test]
    fn string_id_as_u32() {
        let id = StringId(42);
        assert_eq!(id.as_u32(), 42);
    }

    // ── Builder: intern & resolve ────────────────────────────────────

    #[test]
    fn builder_intern_and_resolve() {
        let mut builder = TextStoreBuilder::new();
        let id = builder.intern("hello");
        assert_eq!(builder.resolve(id).unwrap(), "hello");
    }

    #[test]
    fn builder_deduplication() {
        let mut builder = TextStoreBuilder::new();
        let id1 = builder.intern("same");
        let id2 = builder.intern("same");
        assert_eq!(id1, id2, "same string must yield same StringId");
    }

    #[test]
    fn builder_different_strings_different_ids() {
        let mut builder = TextStoreBuilder::new();
        let id_a = builder.intern("alpha");
        let id_b = builder.intern("beta");
        assert_ne!(
            id_a, id_b,
            "different strings must yield different StringIds"
        );
    }

    #[test]
    fn builder_is_empty_and_len() {
        let mut builder = TextStoreBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);

        builder.intern("one");
        assert!(!builder.is_empty());
        assert_eq!(builder.len(), 1);

        builder.intern("two");
        assert_eq!(builder.len(), 2);

        // Duplicate should not increase len
        builder.intern("one");
        assert_eq!(builder.len(), 2);
    }

    // ── Promote to TextStore ─────────────────────────────────────────

    #[test]
    fn promote_and_resolve() {
        let mut builder = TextStoreBuilder::new();
        let id_hello = builder.intern("hello");
        let id_world = builder.intern("world");

        let store: Arc<TextStore> = builder.build();

        assert_eq!(store.resolve(id_hello).unwrap(), "hello");
        assert_eq!(store.resolve(id_world).unwrap(), "world");
    }

    #[test]
    fn store_len_matches_unique_strings() {
        let mut builder = TextStoreBuilder::new();
        builder.intern("a");
        builder.intern("b");
        builder.intern("a"); // duplicate

        let store = builder.build();
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn arc_is_shared() {
        let mut builder = TextStoreBuilder::new();
        let id = builder.intern("shared");
        let store = builder.build();
        let store2 = Arc::clone(&store);

        assert_eq!(store.resolve(id).unwrap(), "shared");
        assert_eq!(store2.resolve(id).unwrap(), "shared");
    }

    // ── Error cases ──────────────────────────────────────────────────

    #[test]
    fn builder_id_not_found() {
        let builder = TextStoreBuilder::new();
        let bad_id = StringId(999);
        let err = builder.resolve(bad_id).unwrap_err();
        assert!(matches!(err, StoreError::IdNotFound(999)));
    }

    #[test]
    fn store_id_not_found() {
        let mut builder = TextStoreBuilder::new();
        builder.intern("only");
        let store = builder.build();

        let bad_id = StringId(999);
        let err = store.resolve(bad_id).unwrap_err();
        assert!(matches!(err, StoreError::IdNotFound(999)));
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn intern_empty_string() {
        let mut builder = TextStoreBuilder::new();
        let id = builder.intern("");
        assert_eq!(builder.resolve(id).unwrap(), "");

        let store = builder.build();
        assert_eq!(store.resolve(id).unwrap(), "");
    }

    #[test]
    fn intern_unicode() {
        let mut builder = TextStoreBuilder::new();
        let id_emoji = builder.intern("hello \u{1F600}");
        let id_cjk = builder.intern("\u{4F60}\u{597D}"); // 你好
        let id_arabic = builder.intern("\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}"); // مرحبا

        let store = builder.build();
        assert_eq!(store.resolve(id_emoji).unwrap(), "hello \u{1F600}");
        assert_eq!(store.resolve(id_cjk).unwrap(), "\u{4F60}\u{597D}");
        assert_eq!(
            store.resolve(id_arabic).unwrap(),
            "\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}"
        );
    }

    // ── Scale: 10K strings ───────────────────────────────────────────

    #[test]
    fn ten_thousand_strings_interned_and_resolved() {
        let mut builder = TextStoreBuilder::new();
        let mut ids = Vec::new();

        for i in 0..10_000u32 {
            let s = format!("string_{i}");
            ids.push((builder.intern(&s), s));
        }

        // Verify all lookups succeed on builder
        for (id, expected) in &ids {
            assert_eq!(builder.resolve(*id).unwrap(), expected.as_str());
        }

        // Promote and verify all lookups succeed on store
        let store = builder.build();
        for (id, expected) in &ids {
            assert_eq!(store.resolve(*id).unwrap(), expected.as_str());
        }

        assert_eq!(store.len(), 10_000);
    }
}
