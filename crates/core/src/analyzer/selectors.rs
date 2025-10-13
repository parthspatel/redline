//! Selectors for grouping diff operations
//!
//! Provides ways to select and group operations from diffs for aggregate analysis

use crate::diff::{DiffOperation, DiffResult, EditType};
use crate::mapping::CharSpan;

/// Trait for selecting operations from a diff
pub trait DiffSelector: Send + Sync {
    /// Select operations from a diff result
    fn select<'a>(&self, diff: &'a DiffResult) -> Vec<&'a DiffOperation>;

    /// Get the name of this selector
    fn name(&self) -> &str;

    /// Clone into a box
    fn clone_box(&self) -> Box<dyn DiffSelector>;
}

impl Clone for Box<dyn DiffSelector> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ============================================================================
// Whole Document Selector
// ============================================================================

/// Selects all operations from a diff
#[derive(Clone)]
pub struct WholeDocumentSelector;

impl DiffSelector for WholeDocumentSelector {
    fn select<'a>(&self, diff: &'a DiffResult) -> Vec<&'a DiffOperation> {
        diff.operations.iter().collect()
    }

    fn name(&self) -> &str {
        "whole_document"
    }

    fn clone_box(&self) -> Box<dyn DiffSelector> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Paragraph Selector
// ============================================================================

/// Selects operations within specific paragraphs
#[derive(Clone)]
pub struct ParagraphSelector {
    /// Which paragraph(s) to select (0-indexed)
    pub paragraphs: Vec<usize>,
}

impl ParagraphSelector {
    pub fn new(paragraphs: Vec<usize>) -> Self {
        Self { paragraphs }
    }

    /// Select a single paragraph
    pub fn single(paragraph: usize) -> Self {
        Self {
            paragraphs: vec![paragraph],
        }
    }

    /// Select a range of paragraphs
    pub fn range(start: usize, end: usize) -> Self {
        Self {
            paragraphs: (start..end).collect(),
        }
    }

    /// Identify paragraph boundaries in text
    fn get_paragraph_spans(text: &str) -> Vec<CharSpan> {
        let mut spans = Vec::new();
        let mut current_start = 0;
        let mut in_paragraph = false;

        for (i, line) in text.lines().enumerate() {
            let line_start = text[..text.lines().take(i).map(|l| l.len() + 1).sum()].len();
            // let line_end = line_start + line.len();

            if line.trim().is_empty() {
                if in_paragraph {
                    spans.push(CharSpan::new(current_start, line_start));
                    in_paragraph = false;
                }
            } else if !in_paragraph {
                current_start = line_start;
                in_paragraph = true;
            }
        }

        // Add final paragraph if text doesn't end with blank line
        if in_paragraph {
            spans.push(CharSpan::new(current_start, text.len()));
        }

        spans
    }

    /// Check if an operation overlaps with selected paragraphs
    fn operation_in_paragraphs(&self, op: &DiffOperation, para_spans: &[CharSpan]) -> bool {
        let op_span = op.original_span.or(op.modified_span);

        if let Some(span) = op_span {
            for &para_idx in &self.paragraphs {
                if let Some(para_span) = para_spans.get(para_idx) {
                    // Check for overlap
                    if span.start < para_span.end && span.end > para_span.start {
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl DiffSelector for ParagraphSelector {
    fn select<'a>(&self, diff: &'a DiffResult) -> Vec<&'a DiffOperation> {
        // Get paragraph boundaries from original text
        let para_spans = Self::get_paragraph_spans(&diff.original_text);

        diff.operations
            .iter()
            .filter(|op| self.operation_in_paragraphs(op, &para_spans))
            .collect()
    }

    fn name(&self) -> &str {
        "paragraph"
    }

    fn clone_box(&self) -> Box<dyn DiffSelector> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Section Selector
// ============================================================================

/// Selects operations within sections (defined by headers)
#[derive(Clone)]
pub struct SectionSelector {
    /// Section identifiers (header text or indices)
    pub sections: Vec<String>,
}

impl SectionSelector {
    pub fn new(sections: Vec<String>) -> Self {
        Self { sections }
    }

    /// Select by section header text
    pub fn by_headers(headers: Vec<impl Into<String>>) -> Self {
        Self {
            sections: headers.into_iter().map(|h| h.into()).collect(),
        }
    }

    /// Identify sections in text based on headers
    /// Simple heuristic: lines starting with # or all caps
    fn get_section_spans(text: &str) -> Vec<(String, CharSpan)> {
        let mut sections = Vec::new();
        let mut current_header = String::new();
        let mut current_start = 0;

        let lines: Vec<&str> = text.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let line_start = text[..text.lines().take(i).map(|l| l.len() + 1).sum()].len();

            if Self::is_header_line(line) {
                // Save previous section if exists
                if !current_header.is_empty() {
                    sections.push((
                        current_header.clone(),
                        CharSpan::new(current_start, line_start),
                    ));
                }

                // Start new section
                current_header = line.trim().to_string();
                current_start = line_start;
            }
        }

        // Add final section
        if !current_header.is_empty() {
            sections.push((current_header, CharSpan::new(current_start, text.len())));
        }

        sections
    }

    fn is_header_line(line: &str) -> bool {
        let trimmed = line.trim();

        // Markdown headers
        if trimmed.starts_with('#') {
            return true;
        }

        // All caps (with at least 3 words)
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.len() >= 2
            && words
                .iter()
                .all(|w| w.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()))
        {
            return true;
        }

        false
    }

    fn operation_in_sections(&self, op: &DiffOperation, sections: &[(String, CharSpan)]) -> bool {
        let op_span = op.original_span.or(op.modified_span);

        if let Some(span) = op_span {
            for (header, section_span) in sections {
                // Check if header matches any of our target sections
                let header_matches = self
                    .sections
                    .iter()
                    .any(|target| header.to_lowercase().contains(&target.to_lowercase()));

                if header_matches {
                    // Check for overlap
                    if span.start < section_span.end && span.end > section_span.start {
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl DiffSelector for SectionSelector {
    fn select<'a>(&self, diff: &'a DiffResult) -> Vec<&'a DiffOperation> {
        let sections = Self::get_section_spans(&diff.original_text);

        diff.operations
            .iter()
            .filter(|op| self.operation_in_sections(op, &sections))
            .collect()
    }

    fn name(&self) -> &str {
        "section"
    }

    fn clone_box(&self) -> Box<dyn DiffSelector> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Edit Type Selector
// ============================================================================

/// Selects operations by edit type
#[derive(Clone)]
pub struct EditTypeSelector {
    pub edit_types: Vec<EditType>,
}

impl EditTypeSelector {
    pub fn new(edit_types: Vec<EditType>) -> Self {
        Self { edit_types }
    }

    /// Select only insertions
    pub fn insertions() -> Self {
        Self {
            edit_types: vec![EditType::Insert],
        }
    }

    /// Select only deletions
    pub fn deletions() -> Self {
        Self {
            edit_types: vec![EditType::Delete],
        }
    }

    /// Select only modifications
    pub fn modifications() -> Self {
        Self {
            edit_types: vec![EditType::Modify],
        }
    }

    /// Select all changes (no Equal)
    pub fn changes() -> Self {
        Self {
            edit_types: vec![EditType::Insert, EditType::Delete, EditType::Modify],
        }
    }
}

impl DiffSelector for EditTypeSelector {
    fn select<'a>(&self, diff: &'a DiffResult) -> Vec<&'a DiffOperation> {
        diff.operations
            .iter()
            .filter(|op| self.edit_types.contains(&op.edit_type))
            .collect()
    }

    fn name(&self) -> &str {
        "edit_type"
    }

    fn clone_box(&self) -> Box<dyn DiffSelector> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Position Range Selector
// ============================================================================

/// Selects operations within a specific character range
#[derive(Clone)]
pub struct PositionRangeSelector {
    pub start: usize,
    pub end: usize,
}

impl PositionRangeSelector {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    fn operation_in_range(&self, op: &DiffOperation) -> bool {
        if let Some(span) = op.original_span.or(op.modified_span) {
            // Check for overlap with our range
            span.start < self.end && span.end > self.start
        } else {
            false
        }
    }
}

impl DiffSelector for PositionRangeSelector {
    fn select<'a>(&self, diff: &'a DiffResult) -> Vec<&'a DiffOperation> {
        diff.operations
            .iter()
            .filter(|op| self.operation_in_range(op))
            .collect()
    }

    fn name(&self) -> &str {
        "position_range"
    }

    fn clone_box(&self) -> Box<dyn DiffSelector> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Composite Selector
// ============================================================================

/// Combines multiple selectors with AND/OR logic
#[derive(Clone)]
pub enum CompositeSelector {
    And(Vec<Box<dyn DiffSelector>>),
    Or(Vec<Box<dyn DiffSelector>>),
}

impl CompositeSelector {
    pub fn and(selectors: Vec<Box<dyn DiffSelector>>) -> Self {
        Self::And(selectors)
    }

    pub fn or(selectors: Vec<Box<dyn DiffSelector>>) -> Self {
        Self::Or(selectors)
    }
}

impl DiffSelector for CompositeSelector {
    fn select<'a>(&self, diff: &'a DiffResult) -> Vec<&'a DiffOperation> {
        match self {
            CompositeSelector::And(selectors) => {
                if selectors.is_empty() {
                    return vec![];
                }

                // Start with first selector's results
                let mut result: std::collections::HashSet<*const DiffOperation> = selectors[0]
                    .select(diff)
                    .into_iter()
                    .map(|op| op as *const _)
                    .collect();

                // Intersect with remaining selectors
                for selector in &selectors[1..] {
                    let selected: std::collections::HashSet<*const DiffOperation> = selector
                        .select(diff)
                        .into_iter()
                        .map(|op| op as *const _)
                        .collect();

                    result.retain(|op| selected.contains(op));
                }

                // Convert back to references
                diff.operations
                    .iter()
                    .filter(|op| result.contains(&(*op as *const _)))
                    .collect()
            }
            CompositeSelector::Or(selectors) => {
                let mut result: std::collections::HashSet<*const DiffOperation> =
                    std::collections::HashSet::new();

                for selector in selectors {
                    for op in selector.select(diff) {
                        result.insert(op as *const _);
                    }
                }

                // Convert back to references
                diff.operations
                    .iter()
                    .filter(|op| result.contains(&(*op as *const _)))
                    .collect()
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            CompositeSelector::And(_) => "composite_and",
            CompositeSelector::Or(_) => "composite_or",
        }
    }

    fn clone_box(&self) -> Box<dyn DiffSelector> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiffEngine;

    #[test]
    fn test_whole_document_selector() {
        let engine = DiffEngine::default();
        let diff = engine.diff("hello world", "hello rust");

        let selector = WholeDocumentSelector;
        let selected = selector.select(&diff);

        assert_eq!(selected.len(), diff.operations.len());
    }

    #[test]
    fn test_edit_type_selector() {
        let engine = DiffEngine::default();
        let diff = engine.diff("hello world", "hello rust world");

        let selector = EditTypeSelector::insertions();
        let selected = selector.select(&diff);

        assert!(selected.iter().all(|op| op.edit_type == EditType::Insert));
    }

    #[test]
    fn test_paragraph_selector() {
        let text1 = "Paragraph 1\n\nParagraph 2\n\nParagraph 3";
        let text2 = "Paragraph 1\n\nModified Para 2\n\nParagraph 3";

        let engine = DiffEngine::default();
        let diff = engine.diff(text1, text2);

        let selector = ParagraphSelector::single(1); // Select second paragraph
        let selected = selector.select(&diff);

        // Should only get operations from paragraph 2
        assert!(selected.len() > 0);
    }
}
