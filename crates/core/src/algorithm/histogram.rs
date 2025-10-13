use crate::algorithm::DiffAlgorithm;
use crate::tokenizers::Token;
use crate::DiffOperation;

/// Histogram diff algorithm (enhanced patience)
pub struct HistogramAlgorithm;

impl HistogramAlgorithm {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HistogramAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffAlgorithm for HistogramAlgorithm {
    fn compute(&self, original: &[Token], modified: &[Token]) -> Vec<DiffOperation> {
        // Build histogram of token occurrences
        use crate::algorithm::myers::MyersAlgorithm;
        use crate::algorithm::patience;
        use std::collections::HashMap;

        let mut orig_histogram: HashMap<&str, usize> = HashMap::new();
        let mut mod_histogram: HashMap<&str, usize> = HashMap::new();

        for token in original {
            *orig_histogram.entry(&token.text).or_insert(0) += 1;
        }

        for token in modified {
            *mod_histogram.entry(&token.text).or_insert(0) += 1;
        }

        // Find low-occurrence common elements
        let mut low_occurrence_matches = Vec::new();

        for (i, orig_token) in original.iter().enumerate() {
            let orig_count = orig_histogram.get(orig_token.text.as_str()).unwrap_or(&0);

            for (j, mod_token) in modified.iter().enumerate() {
                if orig_token.text == mod_token.text {
                    let mod_count = mod_histogram.get(mod_token.text.as_str()).unwrap_or(&0);
                    let occurrence = *orig_count.min(mod_count);

                    if occurrence > 0 {
                        low_occurrence_matches.push((i, j, occurrence));
                    }
                }
            }
        }

        // Sort by occurrence (prefer low occurrence)
        low_occurrence_matches.sort_by_key(|(_, _, occ)| *occ);

        if low_occurrence_matches.is_empty() {
            return MyersAlgorithm::new().compute(original, modified);
        }

        // Use the lowest occurrence matches similar to patience
        let matches: Vec<(usize, usize)> = low_occurrence_matches
            .into_iter()
            .take(10) // Limit to avoid too many matches
            .map(|(i, j, _)| (i, j))
            .collect();

        patience::patience_diff_recursive(
            original,
            modified,
            &matches,
            0,
            original.len(),
            0,
            modified.len(),
        )
    }
}
