//! Pairwise similarity and distance metrics.
//!
//! Set-based (plan 04-06): jaccard, cosine, dice, overlap, length_ratio,
//! word_count_diff, char_count_diff.
//!
//! Edit-distance (plan 04-07): levenshtein, damerau_levenshtein, hamming,
//! jaro, jaro_winkler.

use crate::process::ProcessedText;

pub mod char_count_diff;
pub mod cosine;
pub mod damerau_levenshtein;
pub mod dice;
pub mod hamming;
pub mod jaccard;
pub mod jaro;
pub mod jaro_winkler;
pub mod length_ratio;
pub mod levenshtein;
pub mod overlap;
pub mod word_count_diff;

pub use char_count_diff::CharCountDiffMetric;
pub use cosine::CosineSimilarityMetric;
pub use damerau_levenshtein::DamerauLevenshteinDistanceMetric;
pub use dice::DiceCoefficientMetric;
pub use hamming::HammingDistanceMetric;
pub use jaccard::JaccardSimilarityMetric;
pub use jaro::JaroSimilarityMetric;
pub use jaro_winkler::JaroWinklerSimilarityMetric;
pub use length_ratio::LengthRatioMetric;
pub use levenshtein::LevenshteinDistanceMetric;
pub use overlap::OverlapCoefficientMetric;
pub use word_count_diff::WordCountDiffMetric;

/// Extract token text strings from a ProcessedText.
/// Returns the normalized text slice for each token.
/// This is needed because StringId is only meaningful within one TextStore,
/// so pairwise metrics must compare by actual text content.
pub(crate) fn token_texts<'a>(text: &'a ProcessedText) -> Vec<&'a str> {
    text.tokens
        .iter()
        .map(|t| {
            let start = t.span.start() as usize;
            let end = (t.span.end() as usize).min(text.normalized.len());
            &text.normalized[start..end]
        })
        .collect()
}
