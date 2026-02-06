//! Character position mapping between original and normalized text.
//!
//! [`CharMapping`] tracks how character positions shift during normalization
//! (e.g., lowercasing, Unicode normalization, whitespace collapsing). It stores
//! a sorted list of `(original, normalized)` position pairs and supports
//! bidirectional lookup and composition of multiple mappings.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::NormalizeError;

/// Maps character positions between original and normalized text.
///
/// Internally stores a sorted `Vec<(u32, u32)>` of `(original_pos, normalized_pos)`
/// alignment pairs. Supports bidirectional lookup via binary search and
/// composition of multiple mappings.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CharMapping {
    /// Sorted alignment pairs: (original_position, normalized_position).
    alignments: Vec<(u32, u32)>,
}

impl CharMapping {
    /// Create a new `CharMapping` from a list of `(original, normalized)` pairs.
    ///
    /// The pairs are sorted by original position for efficient lookup.
    ///
    /// # Errors
    /// Returns [`NormalizeError::CompositionFailed`] if `alignments` is empty.
    pub fn new(mut alignments: Vec<(u32, u32)>) -> Result<Self, NormalizeError> {
        if alignments.is_empty() {
            return Err(NormalizeError::CompositionFailed);
        }
        // Sort by original position (first element) for binary search in to_normalized.
        alignments.sort_unstable_by_key(|&(orig, _)| orig);
        Ok(Self { alignments })
    }

    /// Create an identity mapping for text of the given length.
    ///
    /// Every position maps to itself: `[(0,0), (1,1), ..., (len-1, len-1)]`.
    /// If `len` is 0, returns an error.
    ///
    /// # Errors
    /// Returns [`NormalizeError::CompositionFailed`] if `len` is 0.
    pub fn identity(len: u32) -> Result<Self, NormalizeError> {
        if len == 0 {
            return Err(NormalizeError::CompositionFailed);
        }
        let alignments: Vec<(u32, u32)> = (0..len).map(|i| (i, i)).collect();
        Ok(Self { alignments })
    }

    /// Map an original position to its normalized position.
    ///
    /// Uses binary search on the original (first) component of each pair.
    ///
    /// # Errors
    /// Returns [`NormalizeError::InvalidPosition`] if the position is not found.
    pub fn to_normalized(&self, original: u32) -> Result<u32, NormalizeError> {
        self.alignments
            .binary_search_by_key(&original, |&(orig, _)| orig)
            .map(|idx| self.alignments[idx].1)
            .map_err(|_| NormalizeError::InvalidPosition(original))
    }

    /// Map a normalized position back to its original position.
    ///
    /// Performs a linear scan since normalized positions may not be sorted
    /// (e.g., after reordering normalizations). For typical use cases where
    /// normalized positions are monotonically increasing, this is still efficient.
    ///
    /// # Errors
    /// Returns [`NormalizeError::InvalidPosition`] if the position is not found.
    pub fn to_original(&self, normalized: u32) -> Result<u32, NormalizeError> {
        // Normalized positions may not be contiguous or sorted in the same order
        // as the alignment vec (which is sorted by original). We do a linear scan.
        for &(orig, norm) in &self.alignments {
            if norm == normalized {
                return Ok(orig);
            }
        }
        Err(NormalizeError::InvalidPosition(normalized))
    }

    /// Compose this mapping with another, producing a new mapping that goes
    /// directly from `self`'s original positions to `other`'s normalized positions.
    ///
    /// For each `(orig, mid)` in `self`, looks up `mid` in `other` to find `final_pos`,
    /// yielding `(orig, final_pos)` in the result.
    ///
    /// # Errors
    /// Returns [`NormalizeError::CompositionFailed`] if any intermediate position
    /// cannot be found in `other`, or if the result would be empty.
    pub fn compose(&self, other: &CharMapping) -> Result<CharMapping, NormalizeError> {
        let mut result = Vec::with_capacity(self.alignments.len());
        for &(orig, mid) in &self.alignments {
            let final_pos = other
                .to_normalized(mid)
                .map_err(|_| NormalizeError::CompositionFailed)?;
            result.push((orig, final_pos));
        }
        CharMapping::new(result)
    }

    /// Returns the number of alignment pairs.
    #[inline]
    pub fn len(&self) -> usize {
        self.alignments.len()
    }

    /// Returns `true` if there are no alignment pairs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.alignments.is_empty()
    }

    /// Returns a reference to the underlying alignment pairs.
    #[inline]
    pub fn alignments(&self) -> &[(u32, u32)] {
        &self.alignments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────

    #[test]
    fn identity_round_trip() {
        let mapping = CharMapping::identity(5).unwrap();
        assert_eq!(mapping.len(), 5);
        assert!(!mapping.is_empty());
        for i in 0..5u32 {
            assert_eq!(mapping.to_normalized(i).unwrap(), i);
            assert_eq!(mapping.to_original(i).unwrap(), i);
        }
    }

    #[test]
    fn new_with_simple_offset() {
        // Normalization shifted every position by +2
        let pairs = vec![(0, 2), (1, 3), (2, 4), (3, 5)];
        let mapping = CharMapping::new(pairs).unwrap();
        assert_eq!(mapping.len(), 4);

        assert_eq!(mapping.to_normalized(0).unwrap(), 2);
        assert_eq!(mapping.to_normalized(3).unwrap(), 5);
        assert_eq!(mapping.to_original(2).unwrap(), 0);
        assert_eq!(mapping.to_original(5).unwrap(), 3);
    }

    #[test]
    fn empty_alignments_returns_error() {
        let result = CharMapping::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn identity_zero_length_returns_error() {
        let result = CharMapping::identity(0);
        assert!(result.is_err());
    }

    // ── Accessors ────────────────────────────────────────────────────

    #[test]
    fn alignments_returns_underlying_pairs() {
        let pairs = vec![(0, 0), (1, 2), (2, 4)];
        let mapping = CharMapping::new(pairs.clone()).unwrap();
        assert_eq!(mapping.alignments(), &pairs[..]);
    }

    #[test]
    fn is_empty_on_valid_mapping() {
        let mapping = CharMapping::identity(1).unwrap();
        assert!(!mapping.is_empty());
    }

    // ── Lookup errors ────────────────────────────────────────────────

    #[test]
    fn to_normalized_invalid_position() {
        let mapping = CharMapping::identity(3).unwrap();
        let err = mapping.to_normalized(99).unwrap_err();
        assert!(matches!(err, NormalizeError::InvalidPosition(99)));
    }

    #[test]
    fn to_original_invalid_position() {
        let mapping = CharMapping::identity(3).unwrap();
        let err = mapping.to_original(99).unwrap_err();
        assert!(matches!(err, NormalizeError::InvalidPosition(99)));
    }

    // ── Composition: two mappings ────────────────────────────────────

    #[test]
    fn compose_two_mappings() {
        // First normalization: shift +1
        let m1 = CharMapping::new(vec![(0, 1), (1, 2), (2, 3)]).unwrap();
        // Second normalization: shift +10
        let m2 = CharMapping::new(vec![(1, 11), (2, 12), (3, 13)]).unwrap();

        let composed = m1.compose(&m2).unwrap();
        assert_eq!(composed.len(), 3);

        // original -> final: 0->11, 1->12, 2->13
        assert_eq!(composed.to_normalized(0).unwrap(), 11);
        assert_eq!(composed.to_normalized(1).unwrap(), 12);
        assert_eq!(composed.to_normalized(2).unwrap(), 13);

        // And back
        assert_eq!(composed.to_original(11).unwrap(), 0);
        assert_eq!(composed.to_original(12).unwrap(), 1);
        assert_eq!(composed.to_original(13).unwrap(), 2);
    }

    // ── SUCCESS CRITERION: Compose THREE mappings with round-trip ─────

    #[test]
    fn compose_three_mappings_round_trip() {
        // Step 1: original positions 0..5
        // Normalization A: strip leading char -> shift by -1 (but we model as new positions)
        let a = CharMapping::new(vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]).unwrap();

        // Normalization B: double spacing -> positions *2
        let b = CharMapping::new(vec![(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]).unwrap();

        // Normalization C: add prefix of length 3 -> shift +3
        let c = CharMapping::new(vec![(0, 3), (2, 5), (4, 7), (6, 9), (8, 11)]).unwrap();

        // Compose: a -> b -> c
        let ab = a.compose(&b).unwrap();
        let abc = ab.compose(&c).unwrap();

        assert_eq!(abc.len(), 5);

        // Verify full chain: original -> final
        assert_eq!(abc.to_normalized(0).unwrap(), 3);
        assert_eq!(abc.to_normalized(1).unwrap(), 5);
        assert_eq!(abc.to_normalized(2).unwrap(), 7);
        assert_eq!(abc.to_normalized(3).unwrap(), 9);
        assert_eq!(abc.to_normalized(4).unwrap(), 11);

        // Verify round-trip: final -> original
        assert_eq!(abc.to_original(3).unwrap(), 0);
        assert_eq!(abc.to_original(5).unwrap(), 1);
        assert_eq!(abc.to_original(7).unwrap(), 2);
        assert_eq!(abc.to_original(9).unwrap(), 3);
        assert_eq!(abc.to_original(11).unwrap(), 4);
    }

    // ── Composition error cases ──────────────────────────────────────

    #[test]
    fn compose_fails_when_intermediate_not_found() {
        let m1 = CharMapping::new(vec![(0, 100)]).unwrap();
        let m2 = CharMapping::new(vec![(0, 0)]).unwrap(); // doesn't have 100
        let result = m1.compose(&m2);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Identity mapping always round-trips for any valid position.
        #[test]
        fn identity_always_round_trips(len in 1u32..=1000, pos_frac in 0.0f64..1.0) {
            let mapping = CharMapping::identity(len).unwrap();
            let pos = (pos_frac * (len - 1) as f64) as u32;
            let normalized = mapping.to_normalized(pos).unwrap();
            let original = mapping.to_original(normalized).unwrap();
            prop_assert_eq!(original, pos);
        }

        /// Composing any mapping with identity yields the same mapping.
        #[test]
        fn compose_with_identity_is_noop(
            pairs in prop::collection::vec((0u32..100, 0u32..100), 1..50)
        ) {
            // Deduplicate by first element to ensure valid mapping
            let mut seen_orig = std::collections::HashSet::new();
            let mut seen_norm = std::collections::HashSet::new();
            let unique_pairs: Vec<(u32, u32)> = pairs
                .into_iter()
                .filter(|(a, b)| seen_orig.insert(*a) && seen_norm.insert(*b))
                .collect();

            if unique_pairs.is_empty() {
                return Ok(());
            }

            let mapping = CharMapping::new(unique_pairs).unwrap();

            // Build identity that covers all normalized positions
            let max_norm = mapping.alignments().iter().map(|(_, n)| *n).max().unwrap();
            let identity = CharMapping::identity(max_norm + 1).unwrap();

            let composed = mapping.compose(&identity).unwrap();
            prop_assert_eq!(composed.alignments(), mapping.alignments());
        }
    }
}
