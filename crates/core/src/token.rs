//! Token type for representing text segments with their interned string ID,
//! byte span, and classification kind.

use crate::span::Span;
use crate::text_store::StringId;

// ── TokenKind ────────────────────────────────────────────────────────────────

/// Classification of a token within a tokenized sequence.
///
/// Uses `repr(u8)` to guarantee a compact, stable representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TokenKind {
    /// A normal content token.
    Regular = 0,
    /// An unknown or out-of-vocabulary token.
    Unknown = 1,
    /// A special control token (e.g., `[CLS]`, `[SEP]`).
    Special = 2,
    /// A continuation / subword token (e.g., `##ing`).
    Continuation = 3,
}

// ── Token ────────────────────────────────────────────────────────────────────

/// A single token in a tokenized text sequence.
///
/// Combines an interned string ID, a byte-offset span, and a classification kind.
/// The struct is `Copy` and kept at or below 24 bytes for cache-friendly storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Token {
    /// The interned string this token refers to.
    pub text_id: StringId,
    /// The byte-offset span of this token within the original text.
    pub span: Span,
    /// The classification of this token.
    pub kind: TokenKind,
}

impl Token {
    /// Create a new token with the given string ID, span, and kind.
    #[inline]
    pub fn new(text_id: StringId, span: Span, kind: TokenKind) -> Self {
        Self {
            text_id,
            span,
            kind,
        }
    }

    /// Returns `true` if this is a special control token (e.g., `[CLS]`, `[SEP]`).
    #[inline]
    pub fn is_special(&self) -> bool {
        matches!(self.kind, TokenKind::Special)
    }

    /// Returns `true` if this is a continuation / subword token.
    #[inline]
    pub fn is_continuation(&self) -> bool {
        matches!(self.kind, TokenKind::Continuation)
    }

    /// Returns `true` if this is an unknown / out-of-vocabulary token.
    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self.kind, TokenKind::Unknown)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    // ── Static trait assertions ──────────────────────────────────────
    assert_impl_all!(Token: Copy, Clone, core::fmt::Debug, PartialEq, Send, Sync);
    assert_impl_all!(TokenKind: Copy, Clone, core::fmt::Debug, PartialEq, Eq, core::hash::Hash);

    // ── Size constraint ──────────────────────────────────────────────
    const _: () = assert!(core::mem::size_of::<Token>() <= 24);

    // ── Construction & accessors ─────────────────────────────────────

    #[test]
    fn token_creation_and_accessors() {
        let id = StringId(42);
        let span = Span::new(10, 20);
        let kind = TokenKind::Regular;

        let tok = Token::new(id, span, kind);

        assert_eq!(tok.text_id, id);
        assert_eq!(tok.span, span);
        assert_eq!(tok.kind, kind);
    }

    // ── Convenience predicates ───────────────────────────────────────

    #[test]
    fn token_is_special() {
        let tok = Token::new(StringId(0), Span::new(0, 0), TokenKind::Special);
        assert!(tok.is_special());
        assert!(!tok.is_continuation());
        assert!(!tok.is_unknown());
    }

    #[test]
    fn token_is_continuation() {
        let tok = Token::new(StringId(0), Span::new(0, 0), TokenKind::Continuation);
        assert!(tok.is_continuation());
        assert!(!tok.is_special());
        assert!(!tok.is_unknown());
    }

    #[test]
    fn token_is_unknown() {
        let tok = Token::new(StringId(0), Span::new(0, 0), TokenKind::Unknown);
        assert!(tok.is_unknown());
        assert!(!tok.is_special());
        assert!(!tok.is_continuation());
    }

    // ── repr(u8) values ──────────────────────────────────────────────

    #[test]
    fn token_kind_repr() {
        assert_eq!(TokenKind::Regular as u8, 0);
        assert_eq!(TokenKind::Unknown as u8, 1);
        assert_eq!(TokenKind::Special as u8, 2);
        assert_eq!(TokenKind::Continuation as u8, 3);
    }

    // ── Equality ─────────────────────────────────────────────────────

    #[test]
    fn token_equality() {
        let a = Token::new(StringId(1), Span::new(0, 5), TokenKind::Regular);
        let b = Token::new(StringId(1), Span::new(0, 5), TokenKind::Regular);
        assert_eq!(a, b);

        // Different StringId
        let c = Token::new(StringId(2), Span::new(0, 5), TokenKind::Regular);
        assert_ne!(a, c);

        // Different span
        let d = Token::new(StringId(1), Span::new(0, 6), TokenKind::Regular);
        assert_ne!(a, d);

        // Different kind
        let e = Token::new(StringId(1), Span::new(0, 5), TokenKind::Special);
        assert_ne!(a, e);
    }

    // ── Copy semantics ───────────────────────────────────────────────

    #[test]
    fn token_is_copy() {
        let a = Token::new(StringId(7), Span::new(3, 8), TokenKind::Continuation);
        let b = a; // Copy
        let c = a; // Still valid — a was not moved
        assert_eq!(b, c);
        assert_eq!(a, b); // a is still usable
    }
}
