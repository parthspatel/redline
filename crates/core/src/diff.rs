//! Diff result types and structures

use crate::mapping::CharSpan;
use crate::tokenizers::Token;
use std::fmt;

/// Type of edit operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditType {
    /// Content was inserted
    Insert,
    /// Content was deleted
    Delete,
    /// Content was modified (delete + insert)
    Modify,
    /// Content remained unchanged
    Equal,
}

/// Category of change based on analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeCategory {
    /// Semantic/meaning change
    Semantic,
    /// Stylistic change (tone, word choice, etc.)
    Stylistic,
    /// Formatting/presentation change
    Formatting,
    /// Syntactic/grammar change
    Syntactic,
    /// Organizational change (reordering)
    Organizational,
    /// Mixed category
    Mixed(Vec<ChangeCategory>),
    /// Unknown/unclassified
    Unknown,
}

/// A single diff operation
#[derive(Debug, Clone)]
pub struct DiffOperation {
    /// Type of operation
    pub edit_type: EditType,
    
    /// Original text (for delete/modify)
    pub original_text: Option<String>,
    
    /// Modified text (for insert/modify)
    pub modified_text: Option<String>,
    
    /// Span in original text
    pub original_span: Option<CharSpan>,
    
    /// Span in modified text
    pub modified_span: Option<CharSpan>,
    
    /// Tokens involved (if tokenization was used)
    pub original_tokens: Vec<Token>,
    pub modified_tokens: Vec<Token>,
    
    /// Classification of this change
    pub category: ChangeCategory,
    
    /// Confidence score for classification (0.0 to 1.0)
    pub confidence: f64,
}

impl DiffOperation {
    pub fn new(edit_type: EditType) -> Self {
        Self {
            edit_type,
            original_text: None,
            modified_text: None,
            original_span: None,
            modified_span: None,
            original_tokens: Vec::new(),
            modified_tokens: Vec::new(),
            category: ChangeCategory::Unknown,
            confidence: 0.0,
        }
    }

    pub fn with_original(mut self, text: String, span: CharSpan) -> Self {
        self.original_text = Some(text);
        self.original_span = Some(span);
        self
    }

    pub fn with_modified(mut self, text: String, span: CharSpan) -> Self {
        self.modified_text = Some(text);
        self.modified_span = Some(span);
        self
    }

    pub fn with_category(mut self, category: ChangeCategory, confidence: f64) -> Self {
        self.category = category;
        self.confidence = confidence;
        self
    }

    /// Get a human-readable description of this operation
    pub fn description(&self) -> String {
        match self.edit_type {
            EditType::Insert => {
                format!("Insert: \"{}\"", self.modified_text.as_ref().unwrap_or(&String::new()))
            }
            EditType::Delete => {
                format!("Delete: \"{}\"", self.original_text.as_ref().unwrap_or(&String::new()))
            }
            EditType::Modify => {
                format!(
                    "Modify: \"{}\" → \"{}\"",
                    self.original_text.as_ref().unwrap_or(&String::new()),
                    self.modified_text.as_ref().unwrap_or(&String::new())
                )
            }
            EditType::Equal => "Equal".to_string(),
        }
    }
}

/// Statistics about the diff
#[derive(Debug, Clone, Default)]
pub struct DiffStatistics {
    /// Total characters in original
    pub original_length: usize,
    
    /// Total characters in modified
    pub modified_length: usize,
    
    /// Number of insertions
    pub insertions: usize,
    
    /// Number of deletions
    pub deletions: usize,
    
    /// Number of modifications
    pub modifications: usize,
    
    /// Total edit distance
    pub edit_distance: usize,
    
    /// Percentage of text changed (0.0 to 1.0)
    pub change_percentage: f64,
    
    /// Number of tokens changed (if tokenized)
    pub tokens_changed: usize,
    
    /// Total number of tokens
    pub total_tokens: usize,
}

impl DiffStatistics {
    pub fn new(original_length: usize, modified_length: usize) -> Self {
        Self {
            original_length,
            modified_length,
            ..Default::default()
        }
    }

    /// Calculate the change percentage
    pub fn calculate_change_percentage(&mut self) {
        let total_changes = self.insertions + self.deletions + self.modifications;
        let max_length = self.original_length.max(self.modified_length);
        
        self.change_percentage = if max_length > 0 {
            total_changes as f64 / max_length as f64
        } else {
            0.0
        };
    }
}

/// Analysis results for semantic, stylistic, etc.
#[derive(Debug, Clone, Default)]
pub struct DiffAnalysis {
    /// Semantic similarity score (0.0 to 1.0)
    pub semantic_similarity: f64,
    
    /// Stylistic change score (0.0 = no style change, 1.0 = complete style change)
    pub stylistic_change: f64,
    
    /// Readability change (positive = more readable, negative = less readable)
    pub readability_change: f64,
    
    /// Tone change description
    pub tone_change: Option<String>,
    
    /// Detected edit intents
    pub edit_intents: Vec<String>,
    
    /// Custom analysis metrics
    pub custom_metrics: Vec<(String, f64)>,
}

/// Complete diff result
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// List of all diff operations
    pub operations: Vec<DiffOperation>,
    
    /// Statistics about the diff
    pub statistics: DiffStatistics,
    
    /// Analysis results
    pub analysis: DiffAnalysis,
    
    /// Original text
    pub original_text: String,
    
    /// Modified text
    pub modified_text: String,
    
    /// Semantic similarity (shortcut to analysis.semantic_similarity)
    pub semantic_similarity: f64,
}

impl DiffResult {
    pub fn new(original_text: String, modified_text: String) -> Self {
        let stats = DiffStatistics::new(original_text.len(), modified_text.len());
        
        Self {
            operations: Vec::new(),
            statistics: stats,
            analysis: DiffAnalysis::default(),
            original_text,
            modified_text,
            semantic_similarity: 1.0,
        }
    }

    /// Add an operation to the diff
    pub fn add_operation(&mut self, op: DiffOperation) {
        match op.edit_type {
            EditType::Insert => self.statistics.insertions += 1,
            EditType::Delete => self.statistics.deletions += 1,
            EditType::Modify => self.statistics.modifications += 1,
            EditType::Equal => {}
        }
        
        self.operations.push(op);
    }

    /// Finalize the diff result (calculate derived values)
    pub fn finalize(&mut self) {
        self.statistics.calculate_change_percentage();
        self.statistics.edit_distance = 
            self.statistics.insertions + 
            self.statistics.deletions + 
            self.statistics.modifications;
        
        self.semantic_similarity = self.analysis.semantic_similarity;
    }

    /// Get a summary of the diff
    pub fn summary(&self) -> String {
        format!(
            "Diff Summary: {} insertions, {} deletions, {} modifications. \
             Change: {:.1}%, Semantic similarity: {:.2}",
            self.statistics.insertions,
            self.statistics.deletions,
            self.statistics.modifications,
            self.statistics.change_percentage * 100.0,
            self.semantic_similarity
        )
    }

    /// Check if the diff is empty (no changes)
    pub fn is_empty(&self) -> bool {
        self.operations.iter().all(|op| op.edit_type == EditType::Equal)
    }

    /// Get only the changed operations (exclude Equal)
    pub fn changed_operations(&self) -> Vec<&DiffOperation> {
        self.operations
            .iter()
            .filter(|op| op.edit_type != EditType::Equal)
            .collect()
    }

    /// Group operations by category
    pub fn operations_by_category(&self) -> Vec<(ChangeCategory, Vec<&DiffOperation>)> {
        use std::collections::HashMap;
        
        let mut grouped: HashMap<String, Vec<&DiffOperation>> = HashMap::new();
        
        for op in &self.operations {
            let key = format!("{:?}", op.category);
            grouped.entry(key).or_default().push(op);
        }
        
        grouped.into_iter()
            .map(|(_, ops)| {
                let category = ops.first().unwrap().category.clone();
                (category, ops)
            })
            .collect()
    }
}

impl fmt::Display for DiffResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Diff Result ===")?;
        writeln!(f, "{}", self.summary())?;
        writeln!(f, "\nOperations:")?;
        
        for (i, op) in self.operations.iter().enumerate() {
            writeln!(f, "  {}. {}", i + 1, op.description())?;
        }
        
        Ok(())
    }
}

/// Unified diff format output
pub struct UnifiedDiff {
    /// Original filename/identifier
    pub original_name: String,
    /// Modified filename/identifier
    pub modified_name: String,
    /// Hunks of changes
    pub hunks: Vec<DiffHunk>,
}

/// A hunk represents a contiguous block of changes
#[derive(Debug, Clone)]
pub struct DiffHunk {
    /// Starting line in original
    pub original_start: usize,
    /// Number of lines in original
    pub original_count: usize,
    /// Starting line in modified
    pub modified_start: usize,
    /// Number of lines in modified
    pub modified_count: usize,
    /// Lines in this hunk (with +/- prefixes)
    pub lines: Vec<String>,
}

impl UnifiedDiff {
    pub fn new(original_name: impl Into<String>, modified_name: impl Into<String>) -> Self {
        Self {
            original_name: original_name.into(),
            modified_name: modified_name.into(),
            hunks: Vec::new(),
        }
    }

    /// Add a hunk to the diff
    pub fn add_hunk(&mut self, hunk: DiffHunk) {
        self.hunks.push(hunk);
    }

    /// Format as unified diff string
    pub fn format(&self) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("--- {}\n", self.original_name));
        output.push_str(&format!("+++ {}\n", self.modified_name));
        
        for hunk in &self.hunks {
            output.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                hunk.original_start,
                hunk.original_count,
                hunk.modified_start,
                hunk.modified_count
            ));
            
            for line in &hunk.lines {
                output.push_str(line);
                output.push('\n');
            }
        }
        
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_operation() {
        let op = DiffOperation::new(EditType::Modify)
            .with_original("hello".to_string(), CharSpan::new(0, 5))
            .with_modified("world".to_string(), CharSpan::new(0, 5));
        
        assert_eq!(op.edit_type, EditType::Modify);
        assert_eq!(op.original_text, Some("hello".to_string()));
        assert_eq!(op.modified_text, Some("world".to_string()));
    }

    #[test]
    fn test_diff_result() {
        let mut result = DiffResult::new("hello".to_string(), "world".to_string());
        
        result.add_operation(
            DiffOperation::new(EditType::Modify)
                .with_original("hello".to_string(), CharSpan::new(0, 5))
                .with_modified("world".to_string(), CharSpan::new(0, 5))
        );
        
        result.finalize();
        
        assert_eq!(result.statistics.modifications, 1);
        assert!(!result.is_empty());
    }
}
