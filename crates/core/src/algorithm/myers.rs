use crate::algorithm::DiffAlgorithm;
use crate::tokenizers::Token;
use crate::{algorithm, DiffOperation, EditType};

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
                        .with_modified(token.text.clone(), token.normalized_span)
                })
                .collect();
        }

        if m == 0 {
            // All deletions
            return original
                .iter()
                .map(|token| {
                    DiffOperation::new(EditType::Delete)
                        .with_original(token.text.clone(), token.normalized_span)
                })
                .collect();
        }

        // Use simple LCS-based diff for now
        let lcs = algorithm::longest_common_subsequence(original, modified);
        algorithm::build_operations_from_lcs(original, modified, &lcs)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::create_tokens;
    use super::*;

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
