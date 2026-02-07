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
