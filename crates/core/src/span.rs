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
    #[inline]
    pub fn new(start: u32, end: u32) -> Self {
        assert!(
            start <= end,
            "Span start ({start}) must not exceed end ({end})"
        );
        Self { start, end }
    }

    /// Create an empty span at the given position.
    #[inline]
    pub fn empty(pos: u32) -> Self {
        Self {
            start: pos,
            end: pos,
        }
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
    #[inline]
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    /// Returns `true` if this span is empty (zero length).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Returns `true` if this span fully contains `other`.
    #[inline]
    pub fn contains(&self, other: &Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Returns `true` if this span contains the given byte position.
    ///
    /// The range is half-open: `start` is inclusive, `end` is exclusive.
    #[inline]
    pub fn contains_position(&self, pos: u32) -> bool {
        self.start <= pos && pos < self.end
    }

    /// Returns `true` if this span overlaps with `other`.
    ///
    /// Adjacent spans (where one's end equals the other's start) do **not** overlap.
    /// Empty spans never overlap with anything.
    #[inline]
    pub fn overlaps(&self, other: &Span) -> bool {
        // Two half-open intervals [a, b) and [c, d) overlap iff a < d && c < b.
        // This naturally handles empty spans: if a == b or c == d the strict
        // inequalities cannot both be satisfied.
        self.start < other.end && other.start < self.end
    }

    /// Merge this span with `other`, returning the smallest span covering both.
    #[inline]
    pub fn merge(&self, other: &Span) -> Span {
        let start = self.start.min(other.start);
        let end = self.end.max(other.end);
        Span { start, end }
    }

    /// Split this span at a relative offset within the span.
    ///
    /// Returns `(left, right)` where `left` covers `[start, start + offset)` and
    /// `right` covers `[start + offset, end)`.
    ///
    /// # Panics
    /// Panics if `offset > self.len()`.
    #[inline]
    pub fn split_at(&self, offset: u32) -> (Span, Span) {
        assert!(
            offset <= self.len(),
            "split offset ({offset}) exceeds span length ({})",
            self.len()
        );
        let mid = self.start + offset;
        (
            Span {
                start: self.start,
                end: mid,
            },
            Span {
                start: mid,
                end: self.end,
            },
        )
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
