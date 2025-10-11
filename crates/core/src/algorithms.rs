//! Diff algorithms implementation
//!
//! Provides various diff algorithms for computing the difference between token sequences.

use crate::diff::{DiffOperation, EditType};
use crate::tokenizers::Token;

/// Trait for diff algorithms
pub trait DiffAlgorithm {
    /// Compute diff between two token sequences
    fn compute(&self, original: &[Token], modified: &[Token]) -> Vec<DiffOperation>;
}

/// Myers O(ND) diff algorithm
pub struct MyersAlgorithm;

impl MyersAlgorithm {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MyersAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffAlgorithm for MyersAlgorithm {
    fn compute(&self, original: &[Token], modified: &[Token]) -> Vec<DiffOperation> {
        // Simplified Myers algorithm implementation
        let n = original.len();
        let m = modified.len();
        
        if n == 0 && m == 0 {
            return vec![];
        }
        
        if n == 0 {
            // All insertions
            return modified
                .iter()
                .map(|token| {
                    DiffOperation::new(EditType::Insert)
                        .with_modified(
                            token.text.clone(),
                            token.normalized_span,
                        )
                })
                .collect();
        }
        
        if m == 0 {
            // All deletions
            return original
                .iter()
                .map(|token| {
                    DiffOperation::new(EditType::Delete)
                        .with_original(
                            token.text.clone(),
                            token.normalized_span,
                        )
                })
                .collect();
        }
        
        // Use simple LCS-based diff for now
        let lcs = longest_common_subsequence(original, modified);
        build_operations_from_lcs(original, modified, &lcs)
    }
}

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
        let unique_matches = find_unique_matches(original, modified);
        
        if unique_matches.is_empty() {
            // Fall back to Myers
            return MyersAlgorithm::new().compute(original, modified);
        }
        
        // Recursively diff the regions between unique matches
        patience_diff_recursive(original, modified, &unique_matches, 0, original.len(), 0, modified.len())
    }
}

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
        
        patience_diff_recursive(original, modified, &matches, 0, original.len(), 0, modified.len())
    }
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
                    .with_original(token.text.clone(), token.normalized_span)
            );
            orig_idx += 1;
        }
        
        // Add insertions before this LCS match
        while mod_idx < lcs_mod {
            let token = &modified[mod_idx];
            operations.push(
                DiffOperation::new(EditType::Insert)
                    .with_modified(token.text.clone(), token.normalized_span)
            );
            mod_idx += 1;
        }
        
        // Add equal operation for LCS match
        let orig_token = &original[lcs_orig];
        let mod_token = &modified[lcs_mod];
        operations.push(
            DiffOperation::new(EditType::Equal)
                .with_original(orig_token.text.clone(), orig_token.normalized_span)
                .with_modified(mod_token.text.clone(), mod_token.normalized_span)
        );
        
        orig_idx += 1;
        mod_idx += 1;
    }
    
    // Add remaining deletions
    while orig_idx < original.len() {
        let token = &original[orig_idx];
        operations.push(
            DiffOperation::new(EditType::Delete)
                .with_original(token.text.clone(), token.normalized_span)
        );
        orig_idx += 1;
    }
    
    // Add remaining insertions
    while mod_idx < modified.len() {
        let token = &modified[mod_idx];
        operations.push(
            DiffOperation::new(EditType::Insert)
                .with_modified(token.text.clone(), token.normalized_span)
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

fn patience_diff_recursive(
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
            original,
            modified,
            matches,
            last_orig,
            match_orig,
            last_mod,
            match_mod,
        ));
        
        // Add the match as an equal operation
        let token = &original[match_orig];
        operations.push(
            DiffOperation::new(EditType::Equal)
                .with_original(token.text.clone(), token.normalized_span)
                .with_modified(token.text.clone(), token.normalized_span)
        );
        
        last_orig = match_orig + 1;
        last_mod = match_mod + 1;
    }
    
    // Diff the remaining region
    operations.extend(patience_diff_recursive(
        original,
        modified,
        matches,
        last_orig,
        orig_end,
        last_mod,
        mod_end,
    ));
    
    operations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::TextPipeline;
    use crate::tokenizers::{Tokenizer, WordTokenizer};

    fn create_tokens(text: &str) -> Vec<Token> {
        let pipeline = TextPipeline::new();
        let layers = pipeline.process(text);
        let tokenizer = WordTokenizer::new();
        tokenizer.tokenize(&layers)
    }

    #[test]
    fn test_myers_basic() {
        let orig = create_tokens("hello world");
        let modified = create_tokens("hello rust");
        
        let algo = MyersAlgorithm::new();
        let ops = algo.compute(&orig, &modified);
        
        assert!(ops.len() > 0);
    }

    #[test]
    fn test_all_deletions() {
        let orig = create_tokens("hello world");
        let modified = create_tokens("");
        
        let algo = MyersAlgorithm::new();
        let ops = algo.compute(&orig, &modified);
        
        assert_eq!(ops.len(), 2); // Two deletions
        assert!(ops.iter().all(|op| op.edit_type == EditType::Delete));
    }

    #[test]
    fn test_all_insertions() {
        let orig = create_tokens("");
        let modified = create_tokens("hello world");
        
        let algo = MyersAlgorithm::new();
        let ops = algo.compute(&orig, &modified);
        
        assert_eq!(ops.len(), 2); // Two insertions
        assert!(ops.iter().all(|op| op.edit_type == EditType::Insert));
    }
}
