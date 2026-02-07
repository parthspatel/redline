//! Remove punctuation normalizer.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::char_mapping::CharMapping;
use crate::error::NormalizeError;

use super::{NormalizationResult, Normalizer};

/// Removes ASCII punctuation characters, preserving byte-offset CharMapping.
#[derive(Debug, Clone, Copy, Default)]
pub struct RemovePunctuation;

/// Returns `true` if `ch` is an ASCII punctuation character.
#[inline]
fn is_ascii_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
}

impl Normalizer for RemovePunctuation {
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
        if input.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mut result = String::with_capacity(input.len());
        let mut alignments = Vec::new();
        let mut result_offset: u32 = 0;

        for (orig_byte, ch) in input.char_indices() {
            if !is_ascii_punctuation(ch) {
                alignments.push((orig_byte as u32, result_offset));
                result.push(ch);
                result_offset += ch.len_utf8() as u32;
            }
        }

        if result.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mapping = CharMapping::new(alignments).map_err(|_| NormalizeError::InvalidMapping)?;
        Ok(NormalizationResult {
            text: result,
            mapping,
        })
    }

    fn name(&self) -> &str {
        "punctuation"
    }

    fn cost(&self) -> f32 {
        0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(input: &str) -> NormalizationResult {
        RemovePunctuation.normalize(input).unwrap()
    }

    #[test]
    fn remove_period() {
        let r = normalize("hello.");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn remove_comma() {
        let r = normalize("a, b");
        assert_eq!(r.text, "a b");
    }

    #[test]
    fn remove_multiple() {
        let r = normalize("hello, world! how's it?");
        assert_eq!(r.text, "hello world hows it");
    }

    #[test]
    fn remove_exclamation_question() {
        let r = normalize("wow!?");
        assert_eq!(r.text, "wow");
    }

    #[test]
    fn remove_brackets_parens() {
        let r = normalize("[hello] (world)");
        assert_eq!(r.text, "hello world");
    }

    #[test]
    fn ascii_only_no_unicode_punct() {
        // Unicode punctuation like « » should NOT be removed
        let r = normalize("«hello»");
        assert_eq!(r.text, "«hello»");
    }

    #[test]
    fn no_punctuation_noop() {
        let r = normalize("hello world");
        assert_eq!(r.text, "hello world");
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
    fn all_punctuation_error() {
        let err = RemovePunctuation.normalize("...!!!").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn empty_input_error() {
        let err = RemovePunctuation.normalize("").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn mapping_positions_valid() {
        let r = normalize("a.b");
        // "a.b" -> "ab", a at orig 0 -> norm 0, b at orig 2 -> norm 1
        assert_eq!(r.mapping.to_normalized(0).unwrap(), 0); // 'a'
        assert_eq!(r.mapping.to_normalized(2).unwrap(), 1); // 'b'
        assert_eq!(r.mapping.to_original(0).unwrap(), 0);
        assert_eq!(r.mapping.to_original(1).unwrap(), 2);
    }

    #[test]
    fn name_and_cost() {
        let n = RemovePunctuation;
        assert_eq!(n.name(), "punctuation");
        assert!((n.cost() - 0.1).abs() < f32::EPSILON);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn punctuation_idempotent(s in "\\PC{1,100}") {
            if let Ok(r1) = RemovePunctuation.normalize(&s) {
                let r2 = RemovePunctuation.normalize(&r1.text).unwrap();
                prop_assert_eq!(r1.text, r2.text);
            }
        }

        #[test]
        fn result_has_no_ascii_punctuation(s in "[a-z.,!? ]{1,50}") {
            if let Ok(r) = RemovePunctuation.normalize(&s) {
                for ch in r.text.chars() {
                    prop_assert!(!ch.is_ascii_punctuation(), "Found punct '{}' in result: {:?}", ch, r.text);
                }
            }
        }
    }
}
