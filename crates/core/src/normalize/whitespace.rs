//! Whitespace normalizer: collapses runs and trims.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::char_mapping::CharMapping;
use crate::error::NormalizeError;

use super::{NormalizationResult, Normalizer};

/// Collapses whitespace runs to single spaces and trims leading/trailing whitespace.
#[derive(Debug, Clone, Copy, Default)]
pub struct WhitespaceNormalizer;

impl Normalizer for WhitespaceNormalizer {
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
        if input.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mut normalized = String::with_capacity(input.len());
        let mut alignments = Vec::new();
        let mut norm_byte_offset: u32 = 0;
        let mut in_whitespace = false;
        let mut leading = true;

        for (orig_byte_offset, ch) in input.char_indices() {
            if ch.is_whitespace() {
                if leading {
                    continue;
                }
                if !in_whitespace {
                    alignments.push((orig_byte_offset as u32, norm_byte_offset));
                    normalized.push(' ');
                    norm_byte_offset += 1;
                    in_whitespace = true;
                }
            } else {
                leading = false;
                in_whitespace = false;
                alignments.push((orig_byte_offset as u32, norm_byte_offset));
                normalized.push(ch);
                norm_byte_offset += ch.len_utf8() as u32;
            }
        }

        // Trim trailing space
        if normalized.ends_with(' ') {
            normalized.pop();
            alignments.pop();
        }

        if normalized.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mapping = CharMapping::new(alignments, input.len() as u32)
            .map_err(|_| NormalizeError::InvalidMapping)?;
        Ok(NormalizationResult {
            text: normalized,
            mapping,
        })
    }

    fn name(&self) -> &str {
        "whitespace"
    }

    fn cost(&self) -> f32 {
        0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(input: &str) -> NormalizationResult {
        WhitespaceNormalizer.normalize(input).unwrap()
    }

    #[test]
    fn collapse_double_space() {
        let r = normalize("a  b");
        assert_eq!(r.text, "a b");
    }

    #[test]
    fn collapse_triple_space() {
        let r = normalize("a   b");
        assert_eq!(r.text, "a b");
    }

    #[test]
    fn collapse_tabs() {
        let r = normalize("a\t\tb");
        assert_eq!(r.text, "a b");
    }

    #[test]
    fn collapse_newlines() {
        let r = normalize("a\n\nb");
        assert_eq!(r.text, "a b");
    }

    #[test]
    fn collapse_mixed_whitespace() {
        let r = normalize("a \t\n b");
        assert_eq!(r.text, "a b");
    }

    #[test]
    fn trim_leading() {
        let r = normalize("  hello");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn trim_trailing() {
        let r = normalize("hello  ");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn trim_both() {
        let r = normalize("  hello  ");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn empty_input_error() {
        let err = WhitespaceNormalizer.normalize("").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn all_whitespace_error() {
        let err = WhitespaceNormalizer.normalize("   ").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn single_word_no_change() {
        let r = normalize("hello");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn unicode_whitespace_nbsp() {
        // U+00A0 Non-breaking space
        let r = normalize("a\u{00A0}b");
        assert_eq!(r.text, "a b");
    }

    #[test]
    fn name_and_cost() {
        let n = WhitespaceNormalizer;
        assert_eq!(n.name(), "whitespace");
        assert!((n.cost() - 0.1).abs() < f32::EPSILON);
    }

    // ── Edge case tests (02.1-02) ────────────────────────────────────

    #[test]
    fn edge_single_space_only() {
        let err = WhitespaceNormalizer.normalize(" ").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn edge_tabs_and_newlines_only() {
        let err = WhitespaceNormalizer.normalize("\t\n\t\n").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn edge_em_space_unicode() {
        let r = normalize("hello\u{2003}world");
        assert_eq!(r.text, "hello world");
    }

    #[test]
    fn edge_mixed_tab_newline_space() {
        let r = normalize("hello\t\nworld");
        assert_eq!(r.text, "hello world");
    }

    #[test]
    fn edge_quintuple_space_collapse_with_mapping() {
        let r = normalize("a     b");
        assert_eq!(r.text, "a b");
        assert_eq!(r.mapping.to_normalized(0).unwrap(), 0);
        assert_eq!(r.mapping.to_normalized(1).unwrap(), 1);
        assert_eq!(r.mapping.to_normalized(6).unwrap(), 2);
        assert_eq!(r.mapping.original_len(), 7);
    }

    #[test]
    fn edge_leading_trailing_preserved_inner() {
        let r = normalize("  hello world  ");
        assert_eq!(r.text, "hello world");
    }

    #[test]
    fn edge_single_char_no_whitespace() {
        let r = normalize("x");
        assert_eq!(r.text, "x");
        assert_eq!(r.mapping.original_len(), 1);
        assert_eq!(r.mapping.to_normalized(0).unwrap(), 0);
    }

    #[test]
    fn edge_unicode_whitespace_multiple_types() {
        let r = normalize("a\u{00A0}\u{3000}\u{2009}b");
        assert_eq!(r.text, "a b");
    }

    #[test]
    fn edge_mapping_round_trip_collapsed() {
        let r = normalize("hello   world");
        assert_eq!(r.text, "hello world");
        for (norm_byte, _) in r.text.char_indices() {
            let orig = r.mapping.to_original(norm_byte as u32).unwrap();
            let back = r.mapping.to_normalized(orig).unwrap();
            assert_eq!(back, norm_byte as u32);
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn whitespace_idempotent(s in "\\PC{1,100}") {
            let r1 = WhitespaceNormalizer.normalize(&s);
            if let Ok(r1) = r1 {
                let r2 = WhitespaceNormalizer.normalize(&r1.text).unwrap();
                prop_assert_eq!(r1.text, r2.text);
            }
        }

        #[test]
        fn whitespace_no_leading_trailing(s in "[a-z ]{1,50}") {
            if let Ok(r) = WhitespaceNormalizer.normalize(&s) {
                prop_assert!(!r.text.starts_with(' '), "Leading space in: {:?}", r.text);
                prop_assert!(!r.text.ends_with(' '), "Trailing space in: {:?}", r.text);
            }
        }
    }
}
