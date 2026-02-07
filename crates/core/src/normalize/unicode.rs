//! Unicode normalization (NFC/NFD/NFKC/NFKD) with CharMapping.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use unicode_normalization::UnicodeNormalization;

use crate::char_mapping::CharMapping;
use crate::error::NormalizeError;

use super::{NormalizationResult, Normalizer};

/// Unicode normalization form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalizationForm {
    /// Canonical Decomposition, followed by Canonical Composition.
    #[default]
    NFC,
    /// Canonical Decomposition.
    NFD,
    /// Compatibility Decomposition, followed by Canonical Composition.
    NFKC,
    /// Compatibility Decomposition.
    NFKD,
}

/// Normalizes text to a Unicode normalization form with CharMapping.
#[derive(Debug, Clone, Copy)]
pub struct UnicodeNormalizer {
    pub form: NormalizationForm,
}

impl Default for UnicodeNormalizer {
    fn default() -> Self {
        Self {
            form: NormalizationForm::NFC,
        }
    }
}

impl UnicodeNormalizer {
    pub fn new(form: NormalizationForm) -> Self {
        Self { form }
    }

    fn normalize_str(&self, input: &str) -> String {
        match self.form {
            NormalizationForm::NFC => input.nfc().collect(),
            NormalizationForm::NFD => input.nfd().collect(),
            NormalizationForm::NFKC => input.nfkc().collect(),
            NormalizationForm::NFKD => input.nfkd().collect(),
        }
    }
}

/// Build a CharMapping between original and normalized text by aligning via NFD decomposition.
fn build_normalization_mapping(
    original: &str,
    normalized: &str,
) -> Result<CharMapping, NormalizeError> {
    // Decompose both to NFD and align by NFD codepoints
    let orig_nfd: Vec<(usize, char)> = original
        .char_indices()
        .flat_map(|(byte_off, ch)| ch.nfd().map(move |nfd_ch| (byte_off, nfd_ch)))
        .collect();

    let norm_nfd: Vec<(usize, char)> = normalized
        .char_indices()
        .flat_map(|(byte_off, ch)| ch.nfd().map(move |nfd_ch| (byte_off, nfd_ch)))
        .collect();

    let mut alignments = Vec::new();
    let len = orig_nfd.len().min(norm_nfd.len());
    for i in 0..len {
        alignments.push((orig_nfd[i].0 as u32, norm_nfd[i].0 as u32));
    }

    alignments.sort_unstable();
    alignments.dedup();

    if alignments.is_empty() {
        return Err(NormalizeError::InvalidMapping);
    }

    CharMapping::new(alignments, original.len() as u32).map_err(|_| NormalizeError::InvalidMapping)
}

impl Normalizer for UnicodeNormalizer {
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
        if input.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let normalized = self.normalize_str(input);
        let mapping = build_normalization_mapping(input, &normalized)?;

        Ok(NormalizationResult {
            text: normalized,
            mapping,
        })
    }

    fn name(&self) -> &str {
        match self.form {
            NormalizationForm::NFC => "unicode-nfc",
            NormalizationForm::NFD => "unicode-nfd",
            NormalizationForm::NFKC => "unicode-nfkc",
            NormalizationForm::NFKD => "unicode-nfkd",
        }
    }

    fn cost(&self) -> f32 {
        0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nfc(input: &str) -> NormalizationResult {
        UnicodeNormalizer::new(NormalizationForm::NFC)
            .normalize(input)
            .unwrap()
    }

    fn nfd(input: &str) -> NormalizationResult {
        UnicodeNormalizer::new(NormalizationForm::NFD)
            .normalize(input)
            .unwrap()
    }

    fn nfkc(input: &str) -> NormalizationResult {
        UnicodeNormalizer::new(NormalizationForm::NFKC)
            .normalize(input)
            .unwrap()
    }

    #[test]
    fn nfc_composes_combining() {
        // e + combining acute -> é
        let r = nfc("e\u{0301}");
        assert_eq!(r.text, "\u{00E9}");
    }

    #[test]
    fn nfc_composes_n_tilde() {
        let r = nfc("n\u{0303}");
        assert_eq!(r.text, "\u{00F1}");
    }

    #[test]
    fn nfc_ascii_unchanged() {
        let r = nfc("hello");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn nfc_already_nfc() {
        let r = nfc("café");
        assert_eq!(r.text, "café");
    }

    #[test]
    fn nfd_decomposes() {
        let r = nfd("\u{00E9}"); // é
        assert_eq!(r.text, "e\u{0301}");
    }

    #[test]
    fn nfd_decomposes_n_tilde() {
        let r = nfd("\u{00F1}"); // ñ
        assert_eq!(r.text, "n\u{0303}");
    }

    #[test]
    fn nfd_ascii_unchanged() {
        let r = nfd("hello");
        assert_eq!(r.text, "hello");
    }

    #[test]
    fn nfkc_fi_ligature() {
        let r = nfkc("\u{FB01}"); // ﬁ
        assert_eq!(r.text, "fi");
    }

    #[test]
    fn nfkc_fullwidth() {
        let r = nfkc("\u{FF21}"); // fullwidth A
        assert_eq!(r.text, "A");
    }

    #[test]
    fn nfkd_fi_ligature() {
        let r = UnicodeNormalizer::new(NormalizationForm::NFKD)
            .normalize("\u{FB01}")
            .unwrap();
        assert_eq!(r.text, "fi");
    }

    #[test]
    fn empty_input_error() {
        let err = UnicodeNormalizer::default().normalize("").unwrap_err();
        assert!(matches!(err, NormalizeError::EmptyInput));
    }

    #[test]
    fn cjk_unchanged() {
        let r = nfc("你好世界");
        assert_eq!(r.text, "你好世界");
    }

    #[test]
    fn emoji_unchanged() {
        let r = nfc("😀🎉");
        assert_eq!(r.text, "😀🎉");
    }

    #[test]
    fn default_is_nfc() {
        let n = UnicodeNormalizer::default();
        assert_eq!(n.form, NormalizationForm::NFC);
    }

    #[test]
    fn name_matches_form() {
        assert_eq!(
            UnicodeNormalizer::new(NormalizationForm::NFC).name(),
            "unicode-nfc"
        );
        assert_eq!(
            UnicodeNormalizer::new(NormalizationForm::NFD).name(),
            "unicode-nfd"
        );
        assert_eq!(
            UnicodeNormalizer::new(NormalizationForm::NFKC).name(),
            "unicode-nfkc"
        );
        assert_eq!(
            UnicodeNormalizer::new(NormalizationForm::NFKD).name(),
            "unicode-nfkd"
        );
    }

    // ── Edge case tests (02.1-02) ────────────────────────────────────

    #[test]
    fn edge_single_precomposed_nfc_noop() {
        let r = nfc("\u{00E9}");
        assert_eq!(r.text, "\u{00E9}");
        assert_eq!(r.mapping.original_len(), 2);
    }

    #[test]
    fn edge_decomposed_to_nfc() {
        let input = "e\u{0301}";
        assert_eq!(input.len(), 3);
        let r = nfc(input);
        assert_eq!(r.text, "\u{00E9}");
        assert_eq!(r.text.len(), 2);
    }

    #[test]
    fn edge_nfd_precomposed_to_decomposed() {
        let r = nfd("\u{00E9}");
        assert_eq!(r.text, "e\u{0301}");
        assert_eq!(r.text.len(), 3);
    }

    #[test]
    fn edge_nfkc_fi_ligature_expands() {
        let input = "\u{FB01}";
        assert_eq!(input.len(), 3);
        let r = nfkc(input);
        assert_eq!(r.text, "fi");
        assert_eq!(r.text.len(), 2);
    }

    #[test]
    fn edge_nfkd_fi_ligature_expands() {
        let r = UnicodeNormalizer::new(NormalizationForm::NFKD)
            .normalize("\u{FB01}")
            .unwrap();
        assert_eq!(r.text, "fi");
    }

    #[test]
    fn edge_ascii_through_nfc_noop() {
        let r = nfc("hello world 123");
        assert_eq!(r.text, "hello world 123");
        for (byte, _) in r.text.char_indices() {
            let orig = r.mapping.to_original(byte as u32).unwrap();
            assert_eq!(orig, byte as u32);
        }
    }

    #[test]
    fn edge_single_char_nfc() {
        let r = nfc("a");
        assert_eq!(r.text, "a");
        assert_eq!(r.mapping.original_len(), 1);
    }

    #[test]
    fn edge_nfc_nfd_round_trip() {
        let input = "caf\u{00E9} na\u{00EF}ve";
        let nfd_result = nfd(input);
        let nfc_of_nfd = nfc(&nfd_result.text);
        let nfc_direct = nfc(input);
        assert_eq!(nfc_of_nfd.text, nfc_direct.text);
    }

    #[test]
    fn edge_all_four_forms_on_same_input() {
        let input = "caf\u{00E9}";
        let r_nfc = nfc(input);
        let r_nfd = nfd(input);
        let r_nfkc = nfkc(input);
        let r_nfkd = UnicodeNormalizer::new(NormalizationForm::NFKD)
            .normalize(input)
            .unwrap();
        assert_eq!(r_nfc.text, r_nfkc.text);
        assert_eq!(r_nfd.text, r_nfkd.text);
        assert_ne!(r_nfc.text, r_nfd.text);
    }

    #[test]
    fn edge_hangul_decomposition() {
        let input = "\u{D55C}";
        let r = nfd(input);
        assert!(r.text.len() >= input.len());
        assert!(r.text.chars().count() > 1);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn nfc_idempotent(s in "\\PC{1,100}") {
            let n = UnicodeNormalizer::new(NormalizationForm::NFC);
            if let Ok(r1) = n.normalize(&s) {
                let r2 = n.normalize(&r1.text).unwrap();
                prop_assert_eq!(r1.text, r2.text);
            }
        }

        #[test]
        fn nfd_idempotent(s in "\\PC{1,100}") {
            let n = UnicodeNormalizer::new(NormalizationForm::NFD);
            if let Ok(r1) = n.normalize(&s) {
                let r2 = n.normalize(&r1.text).unwrap();
                prop_assert_eq!(r1.text, r2.text);
            }
        }
    }
}
