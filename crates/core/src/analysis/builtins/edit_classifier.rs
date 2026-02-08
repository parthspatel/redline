//! EditClassifier: 7-category intent taxonomy with ranked confidence scores.
//!
//! Full implementation in plan 05-04.

/// The 7 intent categories for classifying edit operations.
///
/// This enum is a locked design decision per 05-CONTEXT.md.
/// Later plans will expand `EditClassifierResult` but must NOT change these variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntentCategory {
    /// Fixing a typo, grammar, or factual error.
    Correction,
    /// Adding new content (sentences, paragraphs).
    Expansion,
    /// Removing content without replacement.
    Deletion,
    /// Substantially rewriting existing content.
    Rewrite,
    /// Minor improvements to wording or clarity.
    Refinement,
    /// Whitespace, punctuation, or structural changes only.
    Formatting,
    /// Does not fit other categories.
    Other,
}

/// Result of edit classification analysis.
#[derive(Debug, Clone)]
pub struct EditClassifierResult;
