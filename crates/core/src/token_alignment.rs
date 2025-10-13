//! Token alignment utilities for comparing syntactic tokens
//!
//! This module provides algorithms to align tokens between original and modified text
//! based on their content (text/lemma) rather than just position.

use crate::analyzer::classifiers::SyntacticToken;

/// Represents an alignment between two token sequences
#[derive(Debug, Clone)]
pub enum TokenAlignment {
    /// Tokens match at same relative position (sequential order preserved)
    Match { orig_idx: usize, mod_idx: usize },
    /// Token only exists in original (deletion)
    Deletion { orig_idx: usize },
    /// Token only exists in modified (insertion)
    Insertion { mod_idx: usize },
    /// Token was replaced (similar position, different content)
    Replacement { orig_idx: usize, mod_idx: usize },
    /// Tokens match but order changed (reordering detected)
    Reorder { orig_idx: usize, mod_idx: usize },
}

/// Align two token sequences using LCS with reordering detection
///
/// This aligns tokens based on their text content, handling:
/// - Exact matches (same text)
/// - Lemma matches (same lemma, different form)
/// - Insertions and deletions
/// - Reorderings (same tokens, different order)
///
/// Strategy:
/// 1. Build a match matrix showing which tokens can match
/// 2. Find the longest common subsequence (LCS) of matching tokens
/// 3. Tokens in LCS that are out of order are marked as Reorder
/// 4. Other tokens are marked as Insert/Delete/Replacement
pub fn align_tokens(orig: &[SyntacticToken], modified: &[SyntacticToken]) -> Vec<TokenAlignment> {
    // Step 1: Build match and similarity matrices
    let mut match_matrix = vec![vec![false; modified.len()]; orig.len()];
    let mut similar_matrix = vec![vec![false; modified.len()]; orig.len()];
    for (i, orig_tok) in orig.iter().enumerate() {
        for (j, mod_tok) in modified.iter().enumerate() {
            match_matrix[i][j] = tokens_match(orig_tok, mod_tok);
            similar_matrix[i][j] = tokens_similar(orig_tok, mod_tok);
        }
    }

    // Step 2: Find all matching pairs (exact matches first, then similar)
    let mut matches: Vec<(usize, usize)> = Vec::new();
    let mut similar_pairs: Vec<(usize, usize)> = Vec::new();

    for i in 0..orig.len() {
        for j in 0..modified.len() {
            if match_matrix[i][j] {
                matches.push((i, j));
            } else if similar_matrix[i][j] {
                similar_pairs.push((i, j));
            }
        }
    }

    // Step 3: Greedy matching - prefer exact matches, then similar pairs, avoid conflicts
    let mut used_orig = vec![false; orig.len()];
    let mut used_mod = vec![false; modified.len()];
    let mut matched_pairs: Vec<(usize, usize)> = Vec::new();
    let mut replacement_pairs: Vec<(usize, usize)> = Vec::new();

    // Sort by diagonal distance (prefer matches on same relative positions)
    matches.sort_by_key(|(i, j)| {
        let i_ratio = *i as f64 / orig.len().max(1) as f64;
        let j_ratio = *j as f64 / modified.len().max(1) as f64;
        ((i_ratio - j_ratio).abs() * 1000.0) as usize
    });

    similar_pairs.sort_by_key(|(i, j)| {
        let i_ratio = *i as f64 / orig.len().max(1) as f64;
        let j_ratio = *j as f64 / modified.len().max(1) as f64;
        ((i_ratio - j_ratio).abs() * 1000.0) as usize
    });

    // First, process exact matches
    for (i, j) in matches {
        if !used_orig[i] && !used_mod[j] {
            matched_pairs.push((i, j));
            used_orig[i] = true;
            used_mod[j] = true;
        }
    }

    // Then, process similar pairs as replacements
    for (i, j) in similar_pairs {
        if !used_orig[i] && !used_mod[j] {
            replacement_pairs.push((i, j));
            used_orig[i] = true;
            used_mod[j] = true;
        }
    }

    // Sort both matched and replacement pairs by original index
    matched_pairs.sort_by_key(|(i, _)| *i);
    replacement_pairs.sort_by_key(|(i, _)| *i);

    // Step 4: Build alignment in output order, detecting reorderings
    // We need to process in a way that accounts for both orig and mod indices
    let mut alignments = Vec::new();

    // Mark which modified indices are matched or replaced (to avoid marking them as insertions)
    let mut matched_mod_indices: std::collections::HashSet<usize> =
        matched_pairs.iter().map(|(_, mod_idx)| *mod_idx).collect();
    for (_, mod_idx) in &replacement_pairs {
        matched_mod_indices.insert(*mod_idx);
    }

    // Build alignment by walking through original sequence
    // First, determine which matches preserve order and which are reordered
    let mut is_reordered_match = vec![false; matched_pairs.len()];

    // A match is in-order if its mod_idx is strictly increasing compared to all previous matches
    // AND strictly less than all following matches
    for i in 0..matched_pairs.len() {
        let (_, current_mod_idx) = matched_pairs[i];

        // Check if any previous match has a greater mod_idx (out of order with respect to past)
        for j in 0..i {
            let (_, prev_mod_idx) = matched_pairs[j];
            if current_mod_idx < prev_mod_idx {
                is_reordered_match[i] = true;
                break;
            }
        }

        // Also check if any following match has a smaller mod_idx (out of order with respect to future)
        if !is_reordered_match[i] {
            for j in (i + 1)..matched_pairs.len() {
                let (_, next_mod_idx) = matched_pairs[j];
                if current_mod_idx > next_mod_idx {
                    is_reordered_match[i] = true;
                    break;
                }
            }
        }
    }

    // Now build the alignments
    let mut pair_iter = matched_pairs.iter().enumerate().peekable();
    let mut replacement_iter = replacement_pairs.iter().peekable();

    for orig_idx in 0..orig.len() {
        // Check if this orig_idx has a match
        if let Some(&(pair_idx, &(match_orig_idx, match_mod_idx))) = pair_iter.peek() {
            if match_orig_idx == orig_idx {
                // We have a match for this position

                if is_reordered_match[pair_idx] {
                    alignments.push(TokenAlignment::Reorder {
                        orig_idx,
                        mod_idx: match_mod_idx,
                    });
                } else {
                    alignments.push(TokenAlignment::Match {
                        orig_idx,
                        mod_idx: match_mod_idx,
                    });
                }

                pair_iter.next(); // Consume this pair
                continue;
            }
        }

        // Check if this orig_idx has a replacement
        if let Some(&&(repl_orig_idx, repl_mod_idx)) = replacement_iter.peek() {
            if repl_orig_idx == orig_idx {
                // We have a replacement for this position
                alignments.push(TokenAlignment::Replacement {
                    orig_idx,
                    mod_idx: repl_mod_idx,
                });
                replacement_iter.next(); // Consume this pair
                continue;
            }
        }

        // No match or replacement - this is a deletion
        alignments.push(TokenAlignment::Deletion { orig_idx });
    }

    // Add any unmatched modified tokens as insertions
    for mod_idx in 0..modified.len() {
        if !matched_mod_indices.contains(&mod_idx) {
            alignments.push(TokenAlignment::Insertion { mod_idx });
        }
    }

    alignments
}

/// Check if two tokens match (same text or same lemma with same tag)
fn tokens_match(orig: &SyntacticToken, modified: &SyntacticToken) -> bool {
    // Exact text match
    if orig.text.eq_ignore_ascii_case(&modified.text) {
        return true;
    }

    // Lemma match only if tag also matches
    // This ensures "do" (VBP) doesn't match "does" (VBZ) even though they have same lemma
    // But allows matching different inflections that are truly equivalent
    if orig.lemma == modified.lemma && !orig.lemma.is_empty() && orig.tag == modified.tag {
        return true;
    }

    false
}

/// Check if two tokens are similar enough to be considered a replacement
/// (same lemma but different tag = grammar/tense change)
fn tokens_similar(orig: &SyntacticToken, modified: &SyntacticToken) -> bool {
    // Already match? Then they're similar
    if tokens_match(orig, modified) {
        return true;
    }

    // Same lemma but different tag/form = replacement candidate
    if orig.lemma == modified.lemma && !orig.lemma.is_empty() && orig.tag != modified.tag {
        return true;
    }

    false
}

/// Find the next matching token in a sequence
fn find_next_match(tokens: &[SyntacticToken], target: &SyntacticToken) -> Option<usize> {
    tokens.iter().position(|tok| tokens_match(tok, target))
}

/// Count POS changes using aligned tokens
pub fn count_pos_changes_aligned(
    orig: &[SyntacticToken],
    modified: &[SyntacticToken],
    alignments: &[TokenAlignment],
) -> usize {
    let mut changes = 0;

    for alignment in alignments {
        match alignment {
            TokenAlignment::Match { orig_idx, mod_idx }
            | TokenAlignment::Reorder { orig_idx, mod_idx } => {
                // Check if POS changed for matched/reordered tokens
                if orig[*orig_idx].pos != modified[*mod_idx].pos {
                    changes += 1;
                }
            }
            TokenAlignment::Replacement { orig_idx, mod_idx } => {
                // Replacement counts as POS change
                if orig[*orig_idx].pos != modified[*mod_idx].pos {
                    changes += 1;
                }
            }
            TokenAlignment::Deletion { .. } | TokenAlignment::Insertion { .. } => {
                // Insertions and deletions count as changes
                changes += 1;
            }
        }
    }

    changes
}

/// Count dependency changes using aligned tokens
pub fn count_dep_changes_aligned(
    orig: &[SyntacticToken],
    modified: &[SyntacticToken],
    alignments: &[TokenAlignment],
) -> usize {
    let mut changes = 0;

    for alignment in alignments {
        match alignment {
            TokenAlignment::Match { orig_idx, mod_idx } => {
                // Check if dependency changed for matched tokens
                if orig[*orig_idx].dep != modified[*mod_idx].dep {
                    changes += 1;
                }
            }
            TokenAlignment::Reorder { orig_idx, mod_idx } => {
                // Reordering always changes dependencies due to position change
                // Even if the dep label is same, the head indices will be different
                changes += 1;

                // Also check if the dep label itself changed
                if orig[*orig_idx].dep != modified[*mod_idx].dep {
                    // Already counted above, but this is even worse
                }
            }
            TokenAlignment::Replacement { orig_idx, mod_idx } => {
                // Replacement counts as dependency change
                if orig[*orig_idx].dep != modified[*mod_idx].dep {
                    changes += 1;
                }
            }
            TokenAlignment::Deletion { .. } | TokenAlignment::Insertion { .. } => {
                // Insertions and deletions count as changes
                changes += 1;
            }
        }
    }

    changes
}

/// Find grammar fixes using aligned tokens
pub fn find_grammar_fixes_aligned(
    orig: &[SyntacticToken],
    modified: &[SyntacticToken],
    alignments: &[TokenAlignment],
) -> Vec<String> {
    let mut fixes = Vec::new();

    for alignment in alignments {
        if let TokenAlignment::Match { orig_idx, mod_idx }
        | TokenAlignment::Reorder { orig_idx, mod_idx }
        | TokenAlignment::Replacement { orig_idx, mod_idx } = alignment
        {
            let orig_tok = &orig[*orig_idx];
            let mod_tok = &modified[*mod_idx];

            // Only check if lemmas match (same word, different form)
            if orig_tok.lemma != mod_tok.lemma {
                continue;
            }

            // Subject-verb agreement fixes
            if (orig_tok.dep == "nsubj" || orig_tok.dep == "ROOT")
                && orig_tok.pos == "VERB"
                && mod_tok.pos == "VERB"
            {
                if orig_tok.tag != mod_tok.tag {
                    fixes.push(format!(
                        "Verb form correction: '{}' → '{}'",
                        orig_tok.text, mod_tok.text
                    ));
                }
            }

            // Auxiliary verb corrections (was/were, is/are)
            if orig_tok.pos == "AUX" && mod_tok.pos == "AUX" {
                if orig_tok.text.to_lowercase() != mod_tok.text.to_lowercase() {
                    fixes.push(format!(
                        "Auxiliary verb correction: '{}' → '{}'",
                        orig_tok.text, mod_tok.text
                    ));
                }
            }

            // Tense consistency
            if orig_tok.pos == "VERB" && mod_tok.pos == "VERB" {
                if orig_tok.tag != mod_tok.tag {
                    fixes.push(format!(
                        "Tense correction: '{}' → '{}'",
                        orig_tok.text, mod_tok.text
                    ));
                }
            }
        }

        // Article corrections (a/an) - check even for non-matched tokens
        if let TokenAlignment::Replacement { orig_idx, mod_idx } = alignment {
            let orig_tok = &orig[*orig_idx];
            let mod_tok = &modified[*mod_idx];

            if orig_tok.pos == "DET" && mod_tok.pos == "DET" {
                let orig_text = orig_tok.text.to_lowercase();
                let mod_text = mod_tok.text.to_lowercase();
                if (orig_text == "a" && mod_text == "an") || (orig_text == "an" && mod_text == "a")
                {
                    fixes.push(format!(
                        "Article correction: '{}' → '{}'",
                        orig_tok.text, mod_tok.text
                    ));
                }
            }

            // Pronoun corrections
            if orig_tok.pos == "PRON" && mod_tok.pos == "PRON" {
                fixes.push(format!(
                    "Pronoun correction: '{}' → '{}'",
                    orig_tok.text, mod_tok.text
                ));
            }
        }
    }

    fixes
}

/// Calculate structural similarity using aligned tokens
pub fn calculate_structural_similarity_aligned(
    orig: &[SyntacticToken],
    modified: &[SyntacticToken],
    alignments: &[TokenAlignment],
) -> f64 {
    if orig.is_empty() && modified.is_empty() {
        return 1.0;
    }

    if orig.is_empty() || modified.is_empty() {
        return 0.0;
    }

    let total_tokens = orig.len().max(modified.len());
    let mut matching_structures = 0;

    for alignment in alignments {
        match alignment {
            TokenAlignment::Match { orig_idx, mod_idx } => {
                // Check if both POS and dependency match AND order preserved
                if orig[*orig_idx].pos == modified[*mod_idx].pos
                    && orig[*orig_idx].dep == modified[*mod_idx].dep
                {
                    matching_structures += 1;
                }
            }
            TokenAlignment::Reorder { .. } => {
                // Reordering breaks structural similarity even if POS/dep match
                // Don't count as matching structure
            }
            _ => {
                // Insertions, deletions, replacements don't match
            }
        }
    }

    matching_structures as f64 / total_tokens as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_token(text: &str, lemma: &str, pos: &str, dep: &str) -> SyntacticToken {
        SyntacticToken {
            text: text.to_string(),
            lemma: lemma.to_string(),
            pos: pos.to_string(),
            tag: pos.to_string(),
            dep: dep.to_string(),
            head: 0,
            is_stop: false,
            is_punct: false,
        }
    }

    #[test]
    fn test_exact_match() {
        let orig = vec![
            create_token("The", "the", "DET", "det"),
            create_token("cat", "cat", "NOUN", "nsubj"),
        ];
        let modified = orig.clone();

        let alignments = align_tokens(&orig, &modified);

        assert_eq!(alignments.len(), 2);
        assert!(matches!(alignments[0], TokenAlignment::Match { .. }));
        assert!(matches!(alignments[1], TokenAlignment::Match { .. }));
    }

    #[test]
    fn test_insertion() {
        let orig = vec![create_token("cat", "cat", "NOUN", "nsubj")];
        let modified = vec![
            create_token("The", "the", "DET", "det"),
            create_token("cat", "cat", "NOUN", "nsubj"),
        ];

        let alignments = align_tokens(&orig, &modified);

        assert_eq!(alignments.len(), 2);

        // The algorithm processes original sequence first, then adds insertions
        // So we get: Match(cat), Insertion(The)
        assert!(
            matches!(alignments[0], TokenAlignment::Match { .. }),
            "Expected Match at position 0, got {:?}",
            alignments[0]
        );
        assert!(
            matches!(alignments[1], TokenAlignment::Insertion { .. }),
            "Expected Insertion at position 1, got {:?}",
            alignments[1]
        );
    }

    #[test]
    fn test_deletion() {
        let orig = vec![
            create_token("The", "the", "DET", "det"),
            create_token("cat", "cat", "NOUN", "nsubj"),
        ];
        let modified = vec![create_token("cat", "cat", "NOUN", "nsubj")];

        let alignments = align_tokens(&orig, &modified);

        assert_eq!(alignments.len(), 2);
        assert!(matches!(alignments[0], TokenAlignment::Deletion { .. }));
        assert!(matches!(alignments[1], TokenAlignment::Match { .. }));
    }

    #[test]
    fn test_lemma_match_same_tag() {
        let orig = vec![create_token("walk", "walk", "VERB", "ROOT")];
        let modified = vec![create_token("walks", "walk", "VERB", "ROOT")];

        let alignments = align_tokens(&orig, &modified);

        assert_eq!(alignments.len(), 1);
        assert!(matches!(alignments[0], TokenAlignment::Match { .. }));
    }

    #[test]
    fn test_replacement_same_lemma_different_tag() {
        // "do" (VBP) -> "does" (VBZ) should be a replacement
        let orig = vec![create_token("do", "do", "AUX", "aux")];
        let modified = vec![create_token("does", "do", "AUX", "aux")];

        // Change tags to simulate different verb forms
        let mut orig_with_tag = orig.clone();
        let mut mod_with_tag = modified.clone();
        orig_with_tag[0].tag = "VBP".to_string();
        mod_with_tag[0].tag = "VBZ".to_string();

        let alignments = align_tokens(&orig_with_tag, &mod_with_tag);

        assert_eq!(alignments.len(), 1);
        assert!(
            matches!(alignments[0], TokenAlignment::Replacement { .. }),
            "Expected Replacement for 'do' -> 'does' with different tags, got {:?}",
            alignments[0]
        );
    }

    #[test]
    fn test_word_order_change() {
        // "big red" -> "red big" should detect reordering
        let orig = vec![
            create_token("The", "the", "DET", "det"),
            create_token("big", "big", "ADJ", "amod"),
            create_token("red", "red", "ADJ", "amod"),
            create_token("car", "car", "NOUN", "ROOT"),
        ];
        let modified = vec![
            create_token("The", "the", "DET", "det"),
            create_token("red", "red", "ADJ", "amod"),
            create_token("big", "big", "ADJ", "amod"),
            create_token("car", "car", "NOUN", "ROOT"),
        ];

        let alignments = align_tokens(&orig, &modified);

        assert_eq!(alignments.len(), 4);

        // Check that both "big" and "red" are marked as reordered
        let reorder_count = alignments
            .iter()
            .filter(|a| matches!(a, TokenAlignment::Reorder { .. }))
            .count();

        assert_eq!(
            reorder_count, 2,
            "Expected 2 reorderings for swapped adjectives, got {}",
            reorder_count
        );
    }

    #[test]
    fn test_complex_alignment() {
        // Mix of match, insertion, deletion, and replacement
        let orig = vec![
            create_token("She", "she", "PRON", "nsubj"),
            create_token("do", "do", "AUX", "aux"),
            create_token("not", "not", "PART", "neg"),
            create_token("like", "like", "VERB", "ROOT"),
        ];
        let mut modified = vec![
            create_token("She", "she", "PRON", "nsubj"),
            create_token("does", "do", "AUX", "aux"),
            create_token("not", "not", "PART", "neg"),
            create_token("like", "like", "VERB", "ROOT"),
            create_token("apples", "apple", "NOUN", "dobj"),
        ];

        // Set different tags for do/does
        let mut orig_with_tags = orig.clone();
        orig_with_tags[1].tag = "VBP".to_string();
        modified[1].tag = "VBZ".to_string();

        let alignments = align_tokens(&orig_with_tags, &modified);

        // Should have: Match(She), Replacement(do->does), Match(not), Match(like), Insertion(apples)
        assert_eq!(alignments.len(), 5);

        // Count each type
        let matches = alignments
            .iter()
            .filter(|a| matches!(a, TokenAlignment::Match { .. }))
            .count();
        let replacements = alignments
            .iter()
            .filter(|a| matches!(a, TokenAlignment::Replacement { .. }))
            .count();
        let insertions = alignments
            .iter()
            .filter(|a| matches!(a, TokenAlignment::Insertion { .. }))
            .count();

        assert_eq!(matches, 3, "Expected 3 matches");
        assert_eq!(replacements, 1, "Expected 1 replacement (do->does)");
        assert_eq!(insertions, 1, "Expected 1 insertion (apples)");
    }

    #[test]
    fn test_count_pos_changes() {
        let orig = vec![
            create_token("quick", "quick", "ADJ", "amod"),
            create_token("fox", "fox", "NOUN", "nsubj"),
        ];
        let modified = vec![
            create_token("quickly", "quick", "ADV", "advmod"), // ADJ -> ADV
            create_token("fox", "fox", "NOUN", "nsubj"),
        ];

        let alignments = align_tokens(&orig, &modified);
        let changes = count_pos_changes_aligned(&orig, &modified, &alignments);

        assert_eq!(changes, 1, "Expected 1 POS change (ADJ -> ADV)");
    }

    #[test]
    fn test_structural_similarity_with_reorder() {
        // Reordered tokens should reduce structural similarity
        let orig = vec![
            create_token("big", "big", "ADJ", "amod"),
            create_token("red", "red", "ADJ", "amod"),
        ];
        let modified = vec![
            create_token("red", "red", "ADJ", "amod"),
            create_token("big", "big", "ADJ", "amod"),
        ];

        let alignments = align_tokens(&orig, &modified);
        let similarity = calculate_structural_similarity_aligned(&orig, &modified, &alignments);

        // With reordering, structural similarity should be 0 (no matches with preserved order)
        assert!(
            similarity < 0.5,
            "Expected low structural similarity with reordering, got {}",
            similarity
        );
    }

    #[test]
    fn test_grammar_fix_detection() {
        // Test auxiliary verb correction
        let mut orig = vec![create_token("do", "do", "AUX", "aux")];
        let mut modified = vec![create_token("does", "do", "AUX", "aux")];

        orig[0].tag = "VBP".to_string();
        modified[0].tag = "VBZ".to_string();

        let alignments = align_tokens(&orig, &modified);
        let fixes = find_grammar_fixes_aligned(&orig, &modified, &alignments);

        assert!(
            !fixes.is_empty(),
            "Expected grammar fix detection for do->does"
        );
        assert!(
            fixes[0].contains("Auxiliary verb correction"),
            "Expected auxiliary verb correction message, got: {}",
            fixes[0]
        );
    }
}
