use crate::algorithm::myers::MyersAlgorithm;
use crate::algorithm::DiffAlgorithm;
use crate::tokenizers::Token;
use crate::{algorithm, DiffOperation, EditType};

/// Patience diff algorithm
pub struct PatienceAlgorithm;

impl PatienceAlgorithm {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PatienceAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffAlgorithm for PatienceAlgorithm {
    fn compute(&self, original: &[Token], modified: &[Token]) -> Vec<DiffOperation> {
        // Find unique matching lines
        let unique_matches = algorithm::find_unique_matches(original, modified);

        if unique_matches.is_empty() {
            // Fall back to Myers
            return MyersAlgorithm::new().compute(original, modified);
        }

        // Recursively diff the regions between unique matches
        patience_diff_recursive(
            original,
            modified,
            &unique_matches,
            0,
            original.len(),
            0,
            modified.len(),
        )
    }
}

pub fn patience_diff_recursive(
    original: &[Token],
    modified: &[Token],
    matches: &[(usize, usize)],
    orig_start: usize,
    orig_end: usize,
    mod_start: usize,
    mod_end: usize,
) -> Vec<DiffOperation> {
    if orig_start >= orig_end && mod_start >= mod_end {
        return vec![];
    }

    // Find matches in this region
    let region_matches: Vec<_> = matches
        .iter()
        .filter(|(o, m)| *o >= orig_start && *o < orig_end && *m >= mod_start && *m < mod_end)
        .copied()
        .collect();

    if region_matches.is_empty() {
        // No matches, use simple diff
        // Ensure valid ranges before slicing
        if orig_start > orig_end || mod_start > mod_end {
            return vec![];
        }
        return MyersAlgorithm::new().compute(
            &original[orig_start..orig_end],
            &modified[mod_start..mod_end],
        );
    }

    let mut operations = Vec::new();
    let mut last_orig = orig_start;
    let mut last_mod = mod_start;

    for (match_orig, match_mod) in region_matches {
        // Recursively diff the region before this match
        operations.extend(patience_diff_recursive(
            original, modified, matches, last_orig, match_orig, last_mod, match_mod,
        ));

        // Add the match as an equal operation
        let token = &original[match_orig];
        operations.push(
            DiffOperation::new(EditType::Equal)
                .with_original(token.text.clone(), token.normalized_span)
                .with_modified(token.text.clone(), token.normalized_span),
        );

        last_orig = match_orig + 1;
        last_mod = match_mod + 1;
    }

    // Diff the remaining region
    operations.extend(patience_diff_recursive(
        original, modified, matches, last_orig, orig_end, last_mod, mod_end,
    ));

    operations
}
