//! Diff algorithms implementation
//!
//! Provides various diff algorithms for computing the difference between token sequences.
pub mod histogram;
pub mod myers;
pub mod patience;

use crate::diff::{DiffOperation, EditType};
use crate::tokenizers::Token;

/// Trait for diff algorithms
pub trait DiffAlgorithm {
    /// Compute diff between two token sequences
    fn compute(&self, original: &[Token], modified: &[Token]) -> Vec<DiffOperation>;
}

// ============================================================================
// Helper Functions
// ============================================================================

fn longest_common_subsequence(original: &[Token], modified: &[Token]) -> Vec<(usize, usize)> {
    let n = original.len();
    let m = modified.len();

    // DP table
    let mut dp = vec![vec![0; m + 1]; n + 1];

    for i in 1..=n {
        for j in 1..=m {
            if original[i - 1].text == modified[j - 1].text {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to find LCS
    let mut lcs = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 && j > 0 {
        if original[i - 1].text == modified[j - 1].text {
            lcs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    lcs.reverse();
    lcs
}

fn build_operations_from_lcs(
    original: &[Token],
    modified: &[Token],
    lcs: &[(usize, usize)],
) -> Vec<DiffOperation> {
    let mut operations = Vec::new();
    let mut orig_idx = 0;
    let mut mod_idx = 0;

    for &(lcs_orig, lcs_mod) in lcs {
        // Add deletions before this LCS match
        while orig_idx < lcs_orig {
            let token = &original[orig_idx];
            operations.push(
                DiffOperation::new(EditType::Delete)
                    .with_original(token.text.clone(), token.normalized_span),
            );
            orig_idx += 1;
        }

        // Add insertions before this LCS match
        while mod_idx < lcs_mod {
            let token = &modified[mod_idx];
            operations.push(
                DiffOperation::new(EditType::Insert)
                    .with_modified(token.text.clone(), token.normalized_span),
            );
            mod_idx += 1;
        }

        // Add equal operation for LCS match
        let orig_token = &original[lcs_orig];
        let mod_token = &modified[lcs_mod];
        operations.push(
            DiffOperation::new(EditType::Equal)
                .with_original(orig_token.text.clone(), orig_token.normalized_span)
                .with_modified(mod_token.text.clone(), mod_token.normalized_span),
        );

        orig_idx += 1;
        mod_idx += 1;
    }

    // Add remaining deletions
    while orig_idx < original.len() {
        let token = &original[orig_idx];
        operations.push(
            DiffOperation::new(EditType::Delete)
                .with_original(token.text.clone(), token.normalized_span),
        );
        orig_idx += 1;
    }

    // Add remaining insertions
    while mod_idx < modified.len() {
        let token = &modified[mod_idx];
        operations.push(
            DiffOperation::new(EditType::Insert)
                .with_modified(token.text.clone(), token.normalized_span),
        );
        mod_idx += 1;
    }

    operations
}

fn find_unique_matches(original: &[Token], modified: &[Token]) -> Vec<(usize, usize)> {
    use std::collections::HashMap;

    let mut orig_counts: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut mod_counts: HashMap<&str, Vec<usize>> = HashMap::new();

    for (i, token) in original.iter().enumerate() {
        orig_counts.entry(&token.text).or_default().push(i);
    }

    for (j, token) in modified.iter().enumerate() {
        mod_counts.entry(&token.text).or_default().push(j);
    }

    let mut unique_matches = Vec::new();

    for (text, orig_positions) in &orig_counts {
        if orig_positions.len() == 1 {
            if let Some(mod_positions) = mod_counts.get(text) {
                if mod_positions.len() == 1 {
                    unique_matches.push((orig_positions[0], mod_positions[0]));
                }
            }
        }
    }

    // Sort by position
    unique_matches.sort();

    // Filter to get longest increasing subsequence
    find_lis(unique_matches)
}

fn find_lis(matches: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if matches.is_empty() {
        return vec![];
    }

    // Simple LIS using patience sorting concept
    let mut lis = vec![matches[0]];

    for &(orig_pos, mod_pos) in &matches[1..] {
        if mod_pos > lis.last().unwrap().1 {
            lis.push((orig_pos, mod_pos));
        }
    }

    lis
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::TextPipeline;
    use crate::tokenizers::{Tokenizer, WordTokenizer};

    pub fn create_tokens(text: &str) -> Vec<Token> {
        let pipeline = TextPipeline::new();
        let layers = pipeline.process(text);
        let tokenizer = WordTokenizer::new();
        tokenizer.tokenize(&layers)
    }
}
