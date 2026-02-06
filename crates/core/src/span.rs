//! Byte-offset span type for text regions.
//!
//! `Span` represents a half-open byte range `[start, end)` using `u32` offsets,
//! keeping the struct at exactly 8 bytes and `Copy`-friendly.

/// A half-open byte range `[start, end)` within a text.
///
/// Uses `u32` offsets to keep the struct at 8 bytes total, supporting texts up
/// to ~4 GiB. `Span` is `Copy` and suitable for use in token representations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Span {
    start: u32,
    end: u32,
}

impl Span {
    /// Create a new `Span` from `start..end`.
    ///
    /// # Panics
    /// Panics if `start > end`.
    pub fn new(_start: u32, _end: u32) -> Self {
        todo!()
    }

    /// Create an empty span at the given position.
    pub fn empty(_pos: u32) -> Self {
        todo!()
    }

    /// Returns the start offset (inclusive).
    pub fn start(&self) -> u32 {
        self.start
    }

    /// Returns the end offset (exclusive).
    pub fn end(&self) -> u32 {
        self.end
    }

    /// Returns the length of this span in bytes.
    pub fn len(&self) -> u32 {
        todo!()
    }

    /// Returns `true` if this span is empty (zero length).
    pub fn is_empty(&self) -> bool {
        todo!()
    }

    /// Returns `true` if this span fully contains `other`.
    pub fn contains(&self, _other: &Span) -> bool {
        todo!()
    }

    /// Returns `true` if this span contains the given byte position.
    pub fn contains_position(&self, _pos: u32) -> bool {
        todo!()
    }

    /// Returns `true` if this span overlaps with `other`.
    pub fn overlaps(&self, _other: &Span) -> bool {
        todo!()
    }

    /// Merge this span with `other`, returning the smallest span covering both.
    pub fn merge(&self, _other: &Span) -> Span {
        todo!()
    }

    /// Split this span at a relative offset within the span.
    ///
    /// # Panics
    /// Panics if `offset > self.len()`.
    pub fn split_at(&self, _offset: u32) -> (Span, Span) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    // Compile-time trait verification
    assert_impl_all!(Span: Copy, Clone, core::fmt::Debug, PartialEq, Eq, core::hash::Hash, Send, Sync);

    #[test]
    fn new_creates_valid_span() {
        let s = Span::new(2, 10);
        assert_eq!(s.start(), 2);
        assert_eq!(s.end(), 10);
        assert_eq!(s.len(), 8);
        assert!(!s.is_empty());
    }

    #[test]
    fn empty_creates_zero_length_span() {
        let s = Span::empty(5);
        assert_eq!(s.start(), 5);
        assert_eq!(s.end(), 5);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn new_with_equal_start_end_is_empty() {
        let s = Span::new(3, 3);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    #[should_panic]
    fn new_panics_when_start_exceeds_end() {
        Span::new(10, 5);
    }

    #[test]
    fn contains_span_inside() {
        let outer = Span::new(0, 20);
        let inner = Span::new(5, 15);
        assert!(outer.contains(&inner));
    }

    #[test]
    fn contains_span_equal() {
        let s = Span::new(0, 10);
        assert!(s.contains(&s));
    }

    #[test]
    fn does_not_contain_larger_span() {
        let inner = Span::new(5, 15);
        let outer = Span::new(0, 20);
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn contains_empty_span_at_boundary() {
        let s = Span::new(0, 10);
        let empty = Span::empty(0);
        assert!(s.contains(&empty));
        let empty_end = Span::empty(10);
        assert!(s.contains(&empty_end));
    }

    #[test]
    fn overlaps_partial() {
        let a = Span::new(0, 10);
        let b = Span::new(5, 15);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn overlaps_full_containment() {
        let outer = Span::new(0, 20);
        let inner = Span::new(5, 15);
        assert!(outer.overlaps(&inner));
        assert!(inner.overlaps(&outer));
    }

    #[test]
    fn adjacent_spans_do_not_overlap() {
        let a = Span::new(0, 5);
        let b = Span::new(5, 10);
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn disjoint_spans_do_not_overlap() {
        let a = Span::new(0, 5);
        let b = Span::new(10, 15);
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn empty_spans_do_not_overlap() {
        let a = Span::empty(5);
        let b = Span::empty(5);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn merge_overlapping() {
        let a = Span::new(0, 10);
        let b = Span::new(5, 15);
        let m = a.merge(&b);
        assert_eq!(m.start(), 0);
        assert_eq!(m.end(), 15);
    }

    #[test]
    fn merge_disjoint() {
        let a = Span::new(0, 5);
        let b = Span::new(10, 15);
        let m = a.merge(&b);
        assert_eq!(m.start(), 0);
        assert_eq!(m.end(), 15);
    }

    #[test]
    fn merge_is_commutative() {
        let a = Span::new(3, 7);
        let b = Span::new(1, 5);
        assert_eq!(a.merge(&b), b.merge(&a));
    }

    #[test]
    fn split_at_middle() {
        let s = Span::new(10, 20);
        let (left, right) = s.split_at(5);
        assert_eq!(left, Span::new(10, 15));
        assert_eq!(right, Span::new(15, 20));
    }

    #[test]
    fn split_at_start() {
        let s = Span::new(10, 20);
        let (left, right) = s.split_at(0);
        assert_eq!(left, Span::new(10, 10));
        assert_eq!(right, Span::new(10, 20));
    }

    #[test]
    fn split_at_end() {
        let s = Span::new(10, 20);
        let (left, right) = s.split_at(10);
        assert_eq!(left, Span::new(10, 20));
        assert_eq!(right, Span::new(20, 20));
    }

    #[test]
    #[should_panic]
    fn split_at_beyond_length_panics() {
        let s = Span::new(10, 20);
        s.split_at(11);
    }

    #[test]
    fn contains_position_inside() {
        let s = Span::new(5, 15);
        assert!(s.contains_position(5));
        assert!(s.contains_position(10));
        assert!(s.contains_position(14));
    }

    #[test]
    fn contains_position_at_end_is_exclusive() {
        let s = Span::new(5, 15);
        assert!(!s.contains_position(15));
    }

    #[test]
    fn contains_position_outside() {
        let s = Span::new(5, 15);
        assert!(!s.contains_position(4));
        assert!(!s.contains_position(16));
    }

    #[test]
    fn empty_span_contains_no_positions() {
        let s = Span::empty(5);
        assert!(!s.contains_position(5));
    }

    #[test]
    fn span_is_eight_bytes() {
        assert_eq!(core::mem::size_of::<Span>(), 8);
    }
}
