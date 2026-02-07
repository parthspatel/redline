//! Remove digits normalizer.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::char_mapping::CharMapping;
use crate::error::NormalizeError;

use super::{NormalizationResult, Normalizer};

/// Removes ASCII digit characters (0-9), preserving byte-offset CharMapping.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoveDigits;

impl Normalizer for RemoveDigits {
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
        if input.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mut result = String::with_capacity(input.len());
        let mut alignments = Vec::new();
        let mut result_offset: u32 = 0;

        for (orig_byte, ch) in input.char_indices() {
            if !ch.is_ascii_digit() {
                alignments.push((orig_byte as u32, result_offset));
                result.push(ch);
                result_offset += ch.len_utf8() as u32;
            }
        }

        if result.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mapping = CharMapping::new(alignments, input.len() as u32)
            .map_err(|_| NormalizeError::InvalidMapping)?;
        Ok(NormalizationResult {
            text: result,
            mapping,
        })
    }

    fn name(&self) -> &str {
        "digits"
    }

    fn cost(&self) -> f32 {
        0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(input: &str) -> NormalizationResult {
        RemoveDigits.normalize(input).unwrap()
    }

    #[test]
    fn remove_digits_from_text() {
        let r = normalize("hello123world");
        assert_eq!(r.text, "helloworld");
    }

    #[test]
    fn remove_leading_digits() {
        let r = normalize("42answer");
        assert_eq!(r.text, "answer");
    }

    #[test]
    fn remove_trailing_digits() {
        let r = normalize("answer42");
        assert_eq!(r.text, "answer");
    }

    #[test]
    fn remove_scattered_digits() {
        let r = normalize("a1b2c3");
        assert_eq!(r.text, "abc");
    }

    #[test]
    fn no_digits_noop() {
        let r = normalize("hello world");
        assert_eq!(r.text, "hello world");
    }

    #[test]
    fn ascii_only_no_unicode_digits() {
        // Unicode digits like ① ② should NOT be removed
        let r = normalize("item\u{2460}");
        assert_eq!(r.text, "item\u{2460}");
    }

    #[test]
    fn cjk_unchanged() {
        let r = normalize("你好世界");
        assert_eq!(r.text, "你好世界");
    }

    #[test]
    fn emoji_unchanged() {
        let r = normalize("😀🎉");
        assert_eq!(r.text, "😀🎉");
    }

    #[test]
    fn all_digits_error() {
        let err = RemoveDigits.normalize("12345").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn empty_input_error() {
        let err = RemoveDigits.normalize("").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn mapping_positions_valid() {
        let r = normalize("a1b");
        // "a1b" -> "ab", a at orig 0 -> norm 0, b at orig 2 -> norm 1
        assert_eq!(r.mapping.to_normalized(0).unwrap(), 0); // 'a'
        assert_eq!(r.mapping.to_normalized(2).unwrap(), 1); // 'b'
        assert_eq!(r.mapping.to_original(0).unwrap(), 0);
        assert_eq!(r.mapping.to_original(1).unwrap(), 2);
    }

    #[test]
    fn name_and_cost() {
        let n = RemoveDigits;
        assert_eq!(n.name(), "digits");
        assert!((n.cost() - 0.1).abs() < f32::EPSILON);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn digits_idempotent(s in "\\PC{1,100}") {
            if let Ok(r1) = RemoveDigits.normalize(&s) {
                let r2 = RemoveDigits.normalize(&r1.text).unwrap();
                prop_assert_eq!(r1.text, r2.text);
            }
        }

        #[test]
        fn result_has_no_ascii_digits(s in "[a-z0-9 ]{1,50}") {
            if let Ok(r) = RemoveDigits.normalize(&s) {
                for ch in r.text.chars() {
                    prop_assert!(!ch.is_ascii_digit(), "Found digit '{}' in result: {:?}", ch, r.text);
                }
            }
        }
    }
}
