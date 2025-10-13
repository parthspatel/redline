//! Character-level mapping system
//!
//! Maintains bidirectional mappings between original and normalized text positions.
//! Supports complex transformations where characters may be:
//! - Removed (e.g., stripping whitespace)
//! - Added (e.g., expanding contractions)
//! - Modified (e.g., case changes)
//! - Merged (e.g., multiple spaces to one)

use std::collections::BTreeMap;

/// Represents a span of characters in text
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharSpan {
    /// Start position (inclusive)
    pub start: usize,
    /// End position (exclusive)
    pub end: usize,
}

impl CharSpan {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    pub fn contains(&self, pos: usize) -> bool {
        pos >= self.start && pos < self.end
    }

    /// Create a point span (zero-width)
    pub fn point(pos: usize) -> Self {
        Self::new(pos, pos)
    }
}

/// Maps a position in normalized text to position(s) in the original text
#[derive(Debug, Clone)]
pub enum CharMapping {
    /// One-to-one mapping (simple transformation like lowercase)
    Direct(usize),

    /// One-to-many mapping (one normalized char maps to multiple original chars)
    /// Example: combining diacritics, ligatures
    Expanded(Vec<usize>),

    /// Many-to-one mapping (multiple original chars map to one normalized char)
    /// Example: whitespace normalization (multiple spaces → one space)
    Collapsed(CharSpan),

    /// Inserted character (exists in normalized but not in original)
    /// Example: adding text, expanding abbreviations
    Inserted,

    /// Deleted character (exists in original but not in normalized)
    /// Stores original position
    Deleted(usize),
}

impl CharMapping {
    /// Get the primary original position (if any)
    pub fn primary_position(&self) -> Option<usize> {
        match self {
            CharMapping::Direct(pos) => Some(*pos),
            CharMapping::Expanded(positions) => positions.first().copied(),
            CharMapping::Collapsed(span) => Some(span.start),
            CharMapping::Inserted => None,
            CharMapping::Deleted(pos) => Some(*pos),
        }
    }

    /// Get all original positions covered by this mapping
    pub fn all_positions(&self) -> Vec<usize> {
        match self {
            CharMapping::Direct(pos) => vec![*pos],
            CharMapping::Expanded(positions) => positions.clone(),
            CharMapping::Collapsed(span) => (span.start..span.end).collect(),
            CharMapping::Inserted => vec![],
            CharMapping::Deleted(pos) => vec![*pos],
        }
    }
}

/// Bidirectional mapping between original and normalized text
#[derive(Debug, Clone)]
pub struct CharacterMap {
    /// Map from normalized position to original position(s)
    /// Key: position in normalized text
    /// Value: mapping to original text
    normalized_to_original: BTreeMap<usize, CharMapping>,

    /// Map from original position to normalized position(s)
    /// Key: position in original text
    /// Value: position(s) in normalized text (can be empty if deleted)
    original_to_normalized: BTreeMap<usize, Vec<usize>>,

    /// Length of the original text
    original_len: usize,

    /// Length of the normalized text
    normalized_len: usize,
}

impl CharacterMap {
    /// Create a new empty character map
    pub fn new(original_len: usize, normalized_len: usize) -> Self {
        Self {
            normalized_to_original: BTreeMap::new(),
            original_to_normalized: BTreeMap::new(),
            original_len,
            normalized_len,
        }
    }

    /// Create an identity mapping (no transformation)
    pub fn identity(len: usize) -> Self {
        let mut map = Self::new(len, len);
        for i in 0..len {
            map.add_direct_mapping(i, i);
        }
        map
    }

    /// Add a direct one-to-one mapping
    pub fn add_direct_mapping(&mut self, normalized_pos: usize, original_pos: usize) {
        self.normalized_to_original
            .insert(normalized_pos, CharMapping::Direct(original_pos));
        self.original_to_normalized
            .entry(original_pos)
            .or_default()
            .push(normalized_pos);
    }

    /// Add a collapsed mapping (many original chars → one normalized char)
    pub fn add_collapsed_mapping(&mut self, normalized_pos: usize, original_span: CharSpan) {
        self.normalized_to_original
            .insert(normalized_pos, CharMapping::Collapsed(original_span));

        for orig_pos in original_span.start..original_span.end {
            self.original_to_normalized
                .entry(orig_pos)
                .or_default()
                .push(normalized_pos);
        }
    }

    /// Add an expanded mapping (one original char → many normalized chars)
    pub fn add_expanded_mapping(&mut self, normalized_positions: Vec<usize>, original_pos: usize) {
        for &norm_pos in &normalized_positions {
            self.normalized_to_original
                .insert(norm_pos, CharMapping::Expanded(vec![original_pos]));
        }

        self.original_to_normalized
            .insert(original_pos, normalized_positions);
    }

    /// Add an inserted character (only in normalized, not in original)
    pub fn add_insertion(&mut self, normalized_pos: usize) {
        self.normalized_to_original
            .insert(normalized_pos, CharMapping::Inserted);
    }

    /// Add a deleted character (only in original, not in normalized)
    pub fn add_deletion(&mut self, original_pos: usize) {
        self.original_to_normalized.insert(original_pos, vec![]);
    }

    /// Map a normalized position to original position(s)
    pub fn normalized_to_original(&self, normalized_pos: usize) -> Option<&CharMapping> {
        self.normalized_to_original.get(&normalized_pos)
    }

    /// Map an original position to normalized position(s)
    pub fn original_to_normalized(&self, original_pos: usize) -> Option<&[usize]> {
        self.original_to_normalized
            .get(&original_pos)
            .map(|v| v.as_slice())
    }

    /// Map a span in normalized text to span(s) in original text
    pub fn map_span_to_original(&self, normalized_span: CharSpan) -> Vec<CharSpan> {
        let mut original_spans = Vec::new();
        let mut current_span: Option<CharSpan> = None;

        for norm_pos in normalized_span.start..normalized_span.end {
            if let Some(mapping) = self.normalized_to_original(norm_pos) {
                for orig_pos in mapping.all_positions() {
                    match current_span {
                        None => {
                            current_span = Some(CharSpan::new(orig_pos, orig_pos + 1));
                        }
                        Some(ref mut span) => {
                            if orig_pos == span.end {
                                // Extend current span
                                span.end = orig_pos + 1;
                            } else {
                                // Start a new span
                                original_spans.push(*span);
                                current_span = Some(CharSpan::new(orig_pos, orig_pos + 1));
                            }
                        }
                    }
                }
            }
        }

        if let Some(span) = current_span {
            original_spans.push(span);
        }

        original_spans
    }

    /// Compose two mappings: original -> intermediate -> final
    pub fn compose(&self, other: &CharacterMap) -> CharacterMap {
        let mut composed = CharacterMap::new(self.original_len, other.normalized_len);

        // For each position in the final (other's normalized) text
        for final_pos in 0..other.normalized_len {
            if let Some(intermediate_mapping) = other.normalized_to_original(final_pos) {
                // Get position(s) in intermediate (self's normalized) text
                let intermediate_positions = intermediate_mapping.all_positions();

                // Map back to original through self
                let mut original_positions = Vec::new();
                for intermediate_pos in intermediate_positions {
                    if let Some(original_mapping) = self.normalized_to_original(intermediate_pos) {
                        original_positions.extend(original_mapping.all_positions());
                    }
                }

                // Create appropriate mapping based on cardinality
                match original_positions.len() {
                    0 => composed.add_insertion(final_pos),
                    1 => composed.add_direct_mapping(final_pos, original_positions[0]),
                    _ => {
                        let span = CharSpan::new(
                            *original_positions.first().unwrap(),
                            *original_positions.last().unwrap() + 1,
                        );
                        composed.add_collapsed_mapping(final_pos, span);
                    }
                }
            }
        }

        composed
    }

    pub fn original_len(&self) -> usize {
        self.original_len
    }

    pub fn normalized_len(&self) -> usize {
        self.normalized_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_mapping() {
        let map = CharacterMap::identity(5);
        assert_eq!(
            map.normalized_to_original(0).unwrap().primary_position(),
            Some(0)
        );
        assert_eq!(map.original_to_normalized(0), Some(&[0][..]));
    }

    #[test]
    fn test_collapsed_mapping() {
        let mut map = CharacterMap::new(5, 3);
        map.add_collapsed_mapping(1, CharSpan::new(1, 4)); // Multiple chars -> one

        let mapping = map.normalized_to_original(1).unwrap();
        assert_eq!(mapping.all_positions(), vec![1, 2, 3]);
    }

    #[test]
    fn test_span_mapping() {
        let mut map = CharacterMap::new(10, 10);
        for i in 0..10 {
            map.add_direct_mapping(i, i);
        }

        let span = CharSpan::new(2, 5);
        let original_spans = map.map_span_to_original(span);
        assert_eq!(original_spans.len(), 1);
        assert_eq!(original_spans[0], CharSpan::new(2, 5));
    }

    #[test]
    fn test_composition() {
        // First mapping: identity for length 5
        let map1 = CharacterMap::identity(5);

        // Second mapping: collapse positions 2-4 to position 2
        let mut map2 = CharacterMap::new(5, 3);
        map2.add_direct_mapping(0, 0);
        map2.add_direct_mapping(1, 1);
        map2.add_collapsed_mapping(2, CharSpan::new(2, 5));

        // Compose: should map directly from original to final
        let composed = map1.compose(&map2);
        assert_eq!(composed.original_len(), 5);
        assert_eq!(composed.normalized_len(), 3);
    }
}
