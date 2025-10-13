//! Token alignment utilities for comparing syntactic tokens
//!
//! This module provides algorithms to align tokens between original and modified text
//! based on their content (text/lemma) rather than just position.

use crate::analyzer::classifiers::SyntacticToken;

/// Represents an alignment between two token sequences
#[derive(Debug, Clone)]
pub enum TokenAlignment {
    /// Tokens match (same position in both sequences)
    Match { orig_idx: usize, mod_idx: usize },
    /// Token only exists in original (deletion)
    Deletion { orig_idx: usize },
    /// Token only exists in modified (insertion)
    Insertion { mod_idx: usize },
    /// Token was replaced (similar position, different content)
    Replacement { orig_idx: usize, mod_idx: usize },
}

/// Align two token sequences using a simple longest common subsequence approach
///
/// This aligns tokens based on their text content, handling:
/// - Exact matches (same text)
/// - Lemma matches (same lemma, different form)
/// - Insertions and deletions
pub fn align_tokens(orig: &[SyntacticToken], modified: &[SyntacticToken]) -> Vec<TokenAlignment> {
    let mut alignments = Vec::new();
    let mut i = 0; // Index in original
    let mut j = 0; // Index in modified

    while i < orig.len() || j < modified.len() {
        if i >= orig.len() {
            // Rest are insertions
            alignments.push(TokenAlignment::Insertion { mod_idx: j });
            j += 1;
        } else if j >= modified.len() {
            // Rest are deletions
            alignments.push(TokenAlignment::Deletion { orig_idx: i });
            i += 1;
        } else {
            let orig_tok = &orig[i];
            let mod_tok = &modified[j];

            // Check for exact match
            if tokens_match(orig_tok, mod_tok) {
                alignments.push(TokenAlignment::Match {
                    orig_idx: i,
                    mod_idx: j,
                });
                i += 1;
                j += 1;
            } else {
                // Look ahead to see if we can find a match
                let next_orig_match = find_next_match(&orig[i..], mod_tok);
                let next_mod_match = find_next_match(&modified[j..], orig_tok);

                match (next_orig_match, next_mod_match) {
                    (Some(0), _) | (_, Some(0)) => {
                        // Current tokens should be aligned (replacement)
                        alignments.push(TokenAlignment::Replacement {
                            orig_idx: i,
                            mod_idx: j,
                        });
                        i += 1;
                        j += 1;
                    }
                    (None, None) => {
                        // No matches found ahead, treat as replacement
                        alignments.push(TokenAlignment::Replacement {
                            orig_idx: i,
                            mod_idx: j,
                        });
                        i += 1;
                        j += 1;
                    }
                    (Some(orig_offset), None) => {
                        // Found match in original ahead, delete current original tokens
                        for _ in 0..orig_offset {
                            alignments.push(TokenAlignment::Deletion { orig_idx: i });
                            i += 1;
                        }
                    }
                    (None, Some(mod_offset)) => {
                        // Found match in modified ahead, insert modified tokens
                        for _ in 0..mod_offset {
                            alignments.push(TokenAlignment::Insertion { mod_idx: j });
                            j += 1;
                        }
                    }
                    (Some(orig_offset), Some(mod_offset)) => {
                        // Both found, choose closer one
                        if orig_offset <= mod_offset {
                            for _ in 0..orig_offset {
                                alignments.push(TokenAlignment::Deletion { orig_idx: i });
                                i += 1;
                            }
                        } else {
                            for _ in 0..mod_offset {
                                alignments.push(TokenAlignment::Insertion { mod_idx: j });
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    alignments
}

/// Check if two tokens match (same text or same lemma)
fn tokens_match(orig: &SyntacticToken, modified: &SyntacticToken) -> bool {
    // Exact text match
    if orig.text.eq_ignore_ascii_case(&modified.text) {
        return true;
    }

    // Lemma match (same word, different form)
    if orig.lemma == modified.lemma && !orig.lemma.is_empty() {
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
            TokenAlignment::Match { orig_idx, mod_idx } => {
                // Check if POS changed for matched tokens
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
        if let TokenAlignment::Match { orig_idx, mod_idx } = alignment {
            // Check if both POS and dependency match
            if orig[*orig_idx].pos == modified[*mod_idx].pos
                && orig[*orig_idx].dep == modified[*mod_idx].dep
            {
                matching_structures += 1;
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
        assert!(matches!(alignments[0], TokenAlignment::Insertion { .. }));
        assert!(matches!(alignments[1], TokenAlignment::Match { .. }));
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
    fn test_lemma_match() {
        let orig = vec![create_token("walk", "walk", "VERB", "ROOT")];
        let modified = vec![create_token("walked", "walk", "VERB", "ROOT")];

        let alignments = align_tokens(&orig, &modified);

        assert_eq!(alignments.len(), 1);
        assert!(matches!(alignments[0], TokenAlignment::Match { .. }));
    }
}
