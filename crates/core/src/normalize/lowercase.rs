//! Lowercase normalizer with byte-offset CharMapping.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::char_mapping::CharMapping;
use crate::error::NormalizeError;

use super::{NormalizationResult, Normalizer};

/// Lowercases all Unicode characters, tracking byte offset changes in CharMapping.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lowercase;

impl Normalizer for Lowercase {
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
        if input.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mut normalized = String::with_capacity(input.len());
        let mut alignments = Vec::new();
        let mut norm_byte_offset: u32 = 0;

        for (orig_byte_offset, ch) in input.char_indices() {
            for lower_ch in ch.to_lowercase() {
                alignments.push((orig_byte_offset as u32, norm_byte_offset));
                normalized.push(lower_ch);
                norm_byte_offset += lower_ch.len_utf8() as u32;
            }
        }

        let mapping = CharMapping::new(alignments, input.len() as u32)
            .map_err(|_| NormalizeError::InvalidMapping)?;
        Ok(NormalizationResult {
            text: normalized,
            mapping,
        })
    }

    fn name(&self) -> &str {
        "lowercase"
    }

    fn cost(&self) -> f32 {
        0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(input: &str) -> NormalizationResult {
        Lowercase.normalize(input).unwrap()
    }

    #[test]
    fn lowercase_ascii() {
        let r = normalize("Hello World");
        assert_eq!(r.text, "hello world");
        // Round-trip all positions
        for i in 0..r.text.len() as u32 {
            let orig = r.mapping.to_original(i).unwrap();
            let back = r.mapping.to_normalized(orig).unwrap();
            assert_eq!(back, i);
        }
    }

    #[test]
    fn lowercase_german() {
        let r = normalize("STRASSE");
        assert_eq!(r.text, "strasse");
    }

    #[test]
    fn lowercase_sharp_s_unchanged() {
        let r = normalize("Straße");
        assert_eq!(r.text, "straße");
    }

    #[test]
    fn lowercase_accented() {
        let r = normalize("Über");
        assert_eq!(r.text, "über");
    }

    #[test]
    fn lowercase_cjk_noop() {
        let r = normalize("你好世界");
        assert_eq!(r.text, "你好世界");
    }

    #[test]
    fn lowercase_emoji_noop() {
        let r = normalize("😀🎉");
        assert_eq!(r.text, "😀🎉");
    }

    #[test]
    fn lowercase_empty_input() {
        let err = Lowercase.normalize("").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn lowercase_already_lowercase() {
        let r = normalize("hello");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn lowercase_name_and_cost() {
        let n = Lowercase;
        assert_eq!(n.name(), "lowercase");
        assert!((n.cost() - 0.1).abs() < f32::EPSILON);
    }

    // ── Edge case tests (02.1-02) ────────────────────────────────────

    #[test]
    fn edge_single_char() {
        let r = normalize("A");
        assert_eq!(r.text, "a");
        assert_eq!(r.mapping.original_len(), 1);
        assert_eq!(r.mapping.to_normalized(0).unwrap(), 0);
        assert_eq!(r.mapping.to_original(0).unwrap(), 0);
    }

    #[test]
    fn edge_byte_length_change_latin_a_stroke() {
        // Ⱥ (U+023A) is 2 bytes UTF-8, ⱥ (U+2C65) is 3 bytes UTF-8
        let input = "\u{023A}";
        assert_eq!(input.len(), 2);
        let r = normalize(input);
        assert_eq!(r.text, "\u{2C65}");
        assert_eq!(r.text.len(), 3);
        assert_eq!(r.mapping.original_len(), 2);
        assert_eq!(r.mapping.to_normalized(0).unwrap(), 0);
    }

    #[test]
    fn edge_byte_length_change_turkish_i() {
        // İ (U+0130) is 2 bytes UTF-8, lowercases to i + combining dot above (3 bytes)
        let input = "\u{0130}";
        assert_eq!(input.len(), 2);
        let r = normalize(input);
        assert_eq!(r.text.len(), 3);
        assert_eq!(r.mapping.original_len(), 2);
        assert_eq!(r.mapping.to_original(0).unwrap(), 0);
        assert_eq!(r.mapping.to_original(1).unwrap(), 0);
    }

    #[test]
    fn edge_noop_mapping_identity() {
        let r = normalize("hello");
        assert_eq!(r.text, "hello");
        assert_eq!(r.mapping.original_len(), 5);
        assert_eq!(r.mapping.len(), 5);
        for i in 0..5u32 {
            assert_eq!(r.mapping.to_normalized(i).unwrap(), i);
            assert_eq!(r.mapping.to_original(i).unwrap(), i);
        }
    }

    #[test]
    fn edge_single_unicode_omega() {
        let r = normalize("\u{03A9}");
        assert_eq!(r.text, "\u{03C9}");
        assert_eq!(r.text.len(), 2);
        assert_eq!(r.mapping.original_len(), 2);
        assert_eq!(r.mapping.to_normalized(0).unwrap(), 0);
        assert_eq!(r.mapping.to_original(0).unwrap(), 0);
    }

    #[test]
    fn edge_cyrillic_mixed_script() {
        let r = normalize("\u{041F}\u{0420}\u{0418}\u{0412}\u{0415}\u{0422}");
        assert_eq!(r.text, "\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}");
        for (norm_byte, _) in r.text.char_indices() {
            let orig = r.mapping.to_original(norm_byte as u32).unwrap();
            let back = r.mapping.to_normalized(orig).unwrap();
            assert_eq!(back, norm_byte as u32);
        }
    }

    #[test]
    fn edge_mapping_round_trip_with_byte_expansion() {
        let r = normalize("A\u{03A9}\u{2C60}");
        assert_eq!(r.text, "a\u{03C9}\u{2C61}");
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
        fn lowercase_idempotent(s in "\\PC{1,100}") {
            let r1 = Lowercase.normalize(&s);
            if let Ok(r1) = r1 {
                let r2 = Lowercase.normalize(&r1.text).unwrap();
                prop_assert_eq!(r1.text, r2.text);
            }
        }

        #[test]
        fn lowercase_mapping_round_trips(s in "[A-Za-z\\p{L}]{1,50}") {
            let r = Lowercase.normalize(&s).unwrap();
            // Every normalized char boundary should map back to a valid original position
            for (norm_byte, _) in r.text.char_indices() {
                let orig = r.mapping.to_original(norm_byte as u32);
                prop_assert!(orig.is_ok(), "Failed to map normalized pos {} back", norm_byte);
            }
        }
    }
}
