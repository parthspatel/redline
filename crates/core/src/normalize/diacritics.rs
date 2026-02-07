//! Remove diacritics (accent marks) normalizer.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use unicode_normalization::UnicodeNormalization;

use crate::char_mapping::CharMapping;
use crate::error::NormalizeError;

use super::{NormalizationResult, Normalizer};

/// Removes diacritical marks (accents) by NFD decomposition then filtering combining marks.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoveDiacritics;

/// Check if a character is a Unicode combining mark (category M).
fn is_combining_mark(ch: char) -> bool {
    // Combining Diacritical Marks: U+0300..U+036F
    // Combining Diacritical Marks Extended: U+1AB0..U+1AFF
    // Combining Diacritical Marks Supplement: U+1DC0..U+1DFF
    // Combining Diacritical Marks for Symbols: U+20D0..U+20FF
    // Combining Half Marks: U+FE20..U+FE2F
    let cp = ch as u32;
    (0x0300..=0x036F).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
}

impl Normalizer for RemoveDiacritics {
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
        if input.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mut result = String::with_capacity(input.len());
        let mut alignments = Vec::new();
        let mut result_offset: u32 = 0;

        // Track original byte offset for each NFD char
        for (orig_byte, ch) in input.char_indices() {
            for nfd_ch in ch.nfd() {
                if !is_combining_mark(nfd_ch) {
                    alignments.push((orig_byte as u32, result_offset));
                    result.push(nfd_ch);
                    result_offset += nfd_ch.len_utf8() as u32;
                }
            }
        }

        if result.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        alignments.sort_unstable();
        alignments.dedup();

        let mapping = CharMapping::new(alignments, input.len() as u32)
            .map_err(|_| NormalizeError::InvalidMapping)?;
        Ok(NormalizationResult {
            text: result,
            mapping,
        })
    }

    fn name(&self) -> &str {
        "diacritics"
    }

    fn cost(&self) -> f32 {
        0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(input: &str) -> NormalizationResult {
        RemoveDiacritics.normalize(input).unwrap()
    }

    #[test]
    fn cafe_to_cafe() {
        let r = normalize("caf\u{00E9}");
        assert_eq!(r.text, "cafe");
    }

    #[test]
    fn naive_to_naive() {
        let r = normalize("na\u{00EF}ve");
        assert_eq!(r.text, "naive");
    }

    #[test]
    fn uber_to_uber() {
        let r = normalize("\u{00FC}ber");
        assert_eq!(r.text, "uber");
    }

    #[test]
    fn resume_to_resume() {
        let r = normalize("r\u{00E9}sum\u{00E9}");
        assert_eq!(r.text, "resume");
    }

    #[test]
    fn pinata_to_pinata() {
        let r = normalize("pi\u{00F1}ata");
        assert_eq!(r.text, "pinata");
    }

    #[test]
    fn cjk_unchanged() {
        let r = normalize("你好");
        assert_eq!(r.text, "你好");
    }

    #[test]
    fn ascii_unchanged() {
        let r = normalize("hello");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn emoji_unchanged() {
        let r = normalize("😀🎉");
        assert_eq!(r.text, "😀🎉");
    }

    #[test]
    fn empty_input_error() {
        let err = RemoveDiacritics.normalize("").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn name_and_cost() {
        let n = RemoveDiacritics;
        assert_eq!(n.name(), "diacritics");
        assert!((n.cost() - 0.3).abs() < f32::EPSILON);
    }

    // ── Edge case tests (02.1-02) ────────────────────────────────────

    #[test]
    fn edge_no_diacritics_noop() {
        let r = normalize("hello");
        assert_eq!(r.text, "hello");
        assert_eq!(r.mapping.original_len(), 5);
        for i in 0..5u32 {
            assert_eq!(r.mapping.to_normalized(i).unwrap(), i);
        }
    }

    #[test]
    fn edge_single_accented_char() {
        let r = normalize("\u{00E9}");
        assert_eq!(r.text, "e");
        assert_eq!(r.mapping.original_len(), 2);
        assert_eq!(r.mapping.to_normalized(0).unwrap(), 0);
    }

    #[test]
    fn edge_vietnamese_stacked_diacritics() {
        let r = normalize("Vi\u{1EC7}t");
        assert_eq!(r.text, "Viet");
    }

    #[test]
    fn edge_mixed_accented_and_plain() {
        let r = normalize("caf\u{00E9} r\u{00E9}sum\u{00E9}");
        assert_eq!(r.text, "cafe resume");
    }

    #[test]
    fn edge_all_combining_marks_empty_input() {
        let err = RemoveDiacritics
            .normalize("\u{0301}\u{0302}\u{0303}")
            .unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn edge_single_char_no_diacritics() {
        let r = normalize("a");
        assert_eq!(r.text, "a");
        assert_eq!(r.mapping.original_len(), 1);
        assert_eq!(r.mapping.len(), 1);
    }

    #[test]
    fn edge_mapping_correctness_mixed() {
        let r = normalize("\u{00E0}b");
        assert_eq!(r.text, "ab");
        assert_eq!(r.mapping.to_original(0).unwrap(), 0);
        assert_eq!(r.mapping.to_original(1).unwrap(), 2);
    }

    #[test]
    fn edge_multiple_diacritics_on_one_char() {
        let r = normalize("\u{1ED3}");
        assert_eq!(r.text, "o");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn diacritics_idempotent(s in "\\PC{1,100}") {
            if let Ok(r1) = RemoveDiacritics.normalize(&s) {
                let r2 = RemoveDiacritics.normalize(&r1.text).unwrap();
                prop_assert_eq!(r1.text, r2.text);
            }
        }
    }
}
