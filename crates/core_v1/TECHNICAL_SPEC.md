# Redline Core v1.0 - Technical Specification

**Version:** 1.0  
**Author:** Parth Patel  
**Date:** 2025-10-15  
**Status:** Proposed

---

## Table of Contents

1. [Overview](#overview)
2. [Foundation Layer](#foundation-layer)
3. [Text Processing Layer](#text-processing-layer)
4. [Diff Computation Layer](#diff-computation-layer)
5. [Metrics Engine](#metrics-engine)
6. [Analysis Framework](#analysis-framework)
7. [Caching System](#caching-system)
8. [Orchestration Layer](#orchestration-layer)
9. [Configuration System](#configuration-system)
10. [Error Handling](#error-handling)
11. [API Reference](#api-reference)
12. [Performance Considerations](#performance-considerations)
13. [Implementation Phases](#implementation-phases)

---

## Overview

This document provides detailed technical specifications for Redline Core v1.0, including complete API definitions, data structures, algorithms, and implementation guidelines.

### Design Goals

- **Zero-cost abstractions**: Minimal runtime overhead
- **Memory efficiency**: <50MB for typical workloads
- **Composability**: All components are independently usable
- **Type safety**: Leverage Rust's type system for correctness
- **Testability**: Every component is unit-testable

---

## Foundation Layer

### 1. TextStore - String Management

The `TextStore` provides efficient string storage through arena allocation and string interning.

#### API

```rust
/// Manages text storage with interning and arena allocation
pub struct TextStore {
    arena: bumpalo::Bump,
    interner: hashbrown::HashMap<&'static str, StringId>,
    id_to_str: Vec<&'static str>,
    next_id: u32,
}

/// Unique identifier for an interned string
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct StringId(u32);

impl TextStore {
    /// Create a new text store with default capacity
    pub fn new() -> Self;
    
    /// Create with specified initial capacity
    pub fn with_capacity(capacity: usize) -> Self;
    
    /// Intern a string, returning its unique ID
    /// If the string already exists, returns existing ID
    pub fn intern(&mut self, s: &str) -> StringId;
    
    /// Get the string for a given ID
    /// 
    /// # Panics
    /// Panics if the ID is invalid
    pub fn get(&self, id: StringId) -> &str;
    
    /// Try to get the string for a given ID
    pub fn try_get(&self, id: StringId) -> Option<&str>;
    
    /// Allocate a string in the arena (for unique strings)
    /// Returns a reference with arena lifetime
    pub fn alloc(&mut self, s: &str) -> &str;
    
    /// Get or intern a string
    pub fn get_or_intern(&mut self, s: &str) -> StringId;
    
    /// Number of unique strings stored
    pub fn len(&self) -> usize;
    
    /// Check if empty
    pub fn is_empty(&self) -> bool;
    
    /// Clear all stored strings and reset
    pub fn clear(&mut self);
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize;
}

impl Default for TextStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference to text in a store
#[derive(Copy, Clone, Debug)]
pub struct TextRef<'store> {
    id: StringId,
    store: &'store TextStore,
}

impl<'store> TextRef<'store> {
    /// Create a new text reference
    pub fn new(id: StringId, store: &'store TextStore) -> Self;
    
    /// Get the string value
    pub fn as_str(&self) -> &'store str;
    
    /// Get the string ID
    pub fn id(&self) -> StringId;
}

impl<'store> std::ops::Deref for TextRef<'store> {
    type Target = str;
    
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl<'store> AsRef<str> for TextRef<'store> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<'store> PartialEq for TextRef<'store> {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: compare IDs
        self.id == other.id
    }
}

impl<'store> Eq for TextRef<'store> {}
```

#### Implementation Notes

1. **Arena Allocation**: Use `bumpalo` crate for fast bump allocation
2. **String Interning**: HashMap with FxHash for fast lookups
3. **ID Assignment**: Monotonic counter, IDs never reused within same store
4. **Memory Layout**: Strings are tightly packed in arena, minimal overhead

#### Example Usage

```rust
let mut store = TextStore::new();

// Intern common strings
let the_id = store.intern("the");
let a_id = store.intern("a");

// Reusing interned string returns same ID
let the_id_2 = store.intern("the");
assert_eq!(the_id, the_id_2);

// Get string back
assert_eq!(store.get(the_id), "the");

// Create text references
let text_ref = TextRef::new(the_id, &store);
assert_eq!(text_ref.as_str(), "the");
```

---

### 2. Span - Position Tracking

#### API

```rust
/// Represents a span in text (byte positions, exclusive end)
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Span {
    /// Start position (inclusive)
    pub start: usize,
    /// End position (exclusive)
    pub end: usize,
}

impl Span {
    /// Create a new span
    pub const fn new(start: usize, end: usize) -> Self;
    
    /// Create a point span (zero-width)
    pub const fn point(pos: usize) -> Self;
    
    /// Create a span from start with length
    pub const fn from_len(start: usize, len: usize) -> Self;
    
    /// Get the length of the span
    pub const fn len(&self) -> usize;
    
    /// Check if the span is empty
    pub const fn is_empty(&self) -> bool;
    
    /// Check if the span contains a position
    pub const fn contains(&self, pos: usize) -> bool;
    
    /// Check if this span overlaps with another
    pub const fn overlaps(&self, other: &Span) -> bool;
    
    /// Get the intersection of two spans
    pub fn intersection(&self, other: &Span) -> Option<Span>;
    
    /// Get the union of two spans (smallest span containing both)
    pub fn union(&self, other: &Span) -> Span;
    
    /// Extend the span to include a position
    pub fn extend(&mut self, pos: usize);
    
    /// Extend the span to include another span
    pub fn extend_span(&mut self, other: &Span);
    
    /// Shift the span by an offset (can be negative)
    pub fn shift(&self, offset: isize) -> Span;
    
    /// Split the span at a position
    pub fn split_at(&self, pos: usize) -> Option<(Span, Span)>;
}

impl From<std::ops::Range<usize>> for Span {
    fn from(range: std::ops::Range<usize>) -> Self {
        Self::new(range.start, range.end)
    }
}

impl From<Span> for std::ops::Range<usize> {
    fn from(span: Span) -> Self {
        span.start..span.end
    }
}
```

#### Example Usage

```rust
let span1 = Span::new(0, 5);
let span2 = Span::new(3, 8);

assert!(span1.overlaps(&span2));
assert_eq!(span1.intersection(&span2), Some(Span::new(3, 5)));
assert_eq!(span1.union(&span2), Span::new(0, 8));
```

---

### 3. CharMapping - Position Mapping

#### API

```rust
/// Bidirectional mapping between character positions
#[derive(Debug, Clone)]
pub struct CharMapping {
    /// Forward mapping: normalized position -> original position(s)
    forward: Vec<MappingEntry>,
    
    /// Reverse mapping: original position -> normalized position(s)
    reverse: hashbrown::HashMap<usize, SmallVec<[usize; 2]>>,
    
    /// Original text length
    original_len: usize,
    
    /// Normalized text length
    normalized_len: usize,
}

/// Type of character mapping
#[derive(Debug, Clone)]
pub enum MappingEntry {
    /// Direct 1:1 mapping
    Direct(usize),
    
    /// Many-to-one: multiple original chars -> one normalized char
    Collapsed(Span),
    
    /// One-to-many: one original char -> multiple normalized chars
    Expanded(SmallVec<[usize; 4]>),
    
    /// Inserted character (no original)
    Inserted,
    
    /// Deleted character (stored in reverse map only)
    Deleted(usize),
}

impl CharMapping {
    /// Create a new empty mapping
    pub fn new(original_len: usize, normalized_len: usize) -> Self;
    
    /// Create an identity mapping
    pub fn identity(len: usize) -> Self;
    
    /// Add a direct 1:1 mapping
    pub fn add_direct(&mut self, normalized_pos: usize, original_pos: usize);
    
    /// Add a collapsed mapping (many original -> one normalized)
    pub fn add_collapsed(&mut self, normalized_pos: usize, original_span: Span);
    
    /// Add an expanded mapping (one original -> many normalized)
    pub fn add_expanded(&mut self, normalized_positions: &[usize], original_pos: usize);
    
    /// Add an insertion (character only in normalized)
    pub fn add_insertion(&mut self, normalized_pos: usize);
    
    /// Add a deletion (character only in original)
    pub fn add_deletion(&mut self, original_pos: usize);
    
    /// Map normalized position to original position(s)
    pub fn normalized_to_original(&self, pos: usize) -> Option<&MappingEntry>;
    
    /// Map original position to normalized position(s)
    pub fn original_to_normalized(&self, pos: usize) -> Option<&[usize]>;
    
    /// Map a normalized span to original span(s)
    pub fn map_span_to_original(&self, span: Span) -> Vec<Span>;
    
    /// Map an original span to normalized span(s)
    pub fn map_span_to_normalized(&self, span: Span) -> Vec<Span>;
    
    /// Compose two mappings: A -> B -> C produces A -> C
    pub fn compose(&self, other: &CharMapping) -> CharMapping;
    
    /// Get original text length
    pub fn original_len(&self) -> usize;
    
    /// Get normalized text length
    pub fn normalized_len(&self) -> usize;
}
```

#### Example Usage

```rust
// Map "Hello  World" -> "hello world"
let mut mapping = CharMapping::new(12, 11);

// "Hello" -> "hello" (direct mappings)
for i in 0..5 {
    mapping.add_direct(i, i);
}

// "  " -> " " (collapsed)
mapping.add_collapsed(5, Span::new(5, 7));

// "World" -> "world"
for i in 0..5 {
    mapping.add_direct(6 + i, 7 + i);
}

// Query mappings
assert_eq!(
    mapping.normalized_to_original(5),
    Some(&MappingEntry::Collapsed(Span::new(5, 7)))
);
```

---

## Text Processing Layer

### 1. Normalizer Trait

```rust
/// Trait for text normalizers
pub trait Normalizer: Send + Sync {
    /// Normalize text, returning normalized string and character mapping
    fn normalize(&self, input: &str) -> NormalizationResult;
    
    /// Get the name of this normalizer
    fn name(&self) -> &str;
    
    /// Get metadata about this normalizer
    fn metadata(&self) -> NormalizerMetadata {
        NormalizerMetadata::default()
    }
    
    /// Estimate computational cost (0.0 to 1.0, default 0.1)
    fn cost(&self) -> f32 {
        0.1
    }
}

/// Result of normalization
#[derive(Debug)]
pub struct NormalizationResult {
    /// Normalized text
    pub text: String,
    
    /// Character mapping
    pub mapping: CharMapping,
}

/// Metadata about a normalizer
#[derive(Debug, Clone, Default)]
pub struct NormalizerMetadata {
    /// Human-readable description
    pub description: String,
    
    /// Additional configuration parameters
    pub parameters: Vec<(String, String)>,
    
    /// Whether this normalizer is deterministic
    pub deterministic: bool,
}

/// Helper trait for cloning normalizers
pub trait NormalizerClone {
    fn clone_box(&self) -> Box<dyn Normalizer>;
}

impl<T> NormalizerClone for T
where
    T: Normalizer + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn Normalizer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Normalizer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
```

### 2. Built-in Normalizers

```rust
/// Lowercase normalizer
#[derive(Debug, Clone, Copy, Default)]
pub struct Lowercase;

impl Normalizer for Lowercase {
    fn normalize(&self, input: &str) -> NormalizationResult {
        let text = input.to_lowercase();
        let mapping = build_case_mapping(input, &text);
        NormalizationResult { text, mapping }
    }
    
    fn name(&self) -> &str {
        "lowercase"
    }
}

/// Uppercase normalizer
#[derive(Debug, Clone, Copy, Default)]
pub struct Uppercase;

/// Unicode normalization (NFC, NFD, NFKC, NFKD)
#[derive(Debug, Clone, Copy)]
pub struct UnicodeNormalizer {
    pub form: UnicodeNormalizationForm,
}

#[derive(Debug, Clone, Copy)]
pub enum UnicodeNormalizationForm {
    NFC,  // Canonical composition
    NFD,  // Canonical decomposition
    NFKC, // Compatibility composition
    NFKD, // Compatibility decomposition
}

/// Whitespace normalizer
#[derive(Debug, Clone)]
pub struct WhitespaceNormalizer {
    /// Collapse multiple whitespace to single space
    pub collapse: bool,
    
    /// Trim leading/trailing whitespace
    pub trim: bool,
    
    /// Normalize newlines to spaces
    pub normalize_newlines: bool,
}

impl WhitespaceNormalizer {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn collapse() -> Self {
        Self {
            collapse: true,
            ..Default::default()
        }
    }
    
    pub fn trim() -> Self {
        Self {
            trim: true,
            ..Default::default()
        }
    }
    
    pub fn all() -> Self {
        Self {
            collapse: true,
            trim: true,
            normalize_newlines: true,
        }
    }
}

/// Remove diacritics (accents)
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoveDiacritics;

/// Remove punctuation
#[derive(Debug, Clone)]
pub struct RemovePunctuation {
    /// Set of punctuation to remove (default: all ASCII punctuation)
    pub punctuation: std::collections::HashSet<char>,
}

/// Remove digits
#[derive(Debug, Clone, Copy, Default)]
pub struct RemoveDigits;

/// Custom normalizer with user-provided function
pub struct CustomNormalizer {
    name: String,
    func: Arc<dyn Fn(&str) -> String + Send + Sync>,
}
```

### 3. Tokenizer Trait

```rust
/// Trait for tokenizers
pub trait Tokenizer: Send + Sync {
    /// Tokenize processed text
    fn tokenize<'store>(
        &self,
        text: &str,
        mapping: &CharMapping,
        store: &'store mut TextStore,
    ) -> Vec<Token<'store>>;
    
    /// Get the name of this tokenizer
    fn name(&self) -> &str;
    
    /// Get metadata
    fn metadata(&self) -> TokenizerMetadata {
        TokenizerMetadata::default()
    }
    
    /// Estimate computational cost (0.0 to 1.0)
    fn cost(&self) -> f32 {
        0.1
    }
}

/// Token with lifetime tied to text store
#[derive(Debug, Clone)]
pub struct Token<'store> {
    /// Reference to token text
    pub text: TextRef<'store>,
    
    /// Span in normalized text
    pub span: Span,
    
    /// Span(s) in original text
    pub original_spans: SmallVec<[Span; 1]>,
    
    /// Token index in sequence
    pub index: usize,
    
    /// Token kind
    pub kind: TokenKind,
}

/// Type of token
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Word,
    Punctuation,
    Whitespace,
    Number,
    Symbol,
    Sentence,
    Paragraph,
    Custom(&'static str),
}

impl<'store> Token<'store> {
    /// Create a new token
    pub fn new(
        text: TextRef<'store>,
        span: Span,
        original_spans: SmallVec<[Span; 1]>,
        index: usize,
        kind: TokenKind,
    ) -> Self;
    
    /// Get the token text as a string slice
    pub fn as_str(&self) -> &'store str;
    
    /// Get the token's length in bytes
    pub fn len(&self) -> usize;
    
    /// Check if token is empty
    pub fn is_empty(&self) -> bool;
}
```

### 4. Built-in Tokenizers

```rust
/// Character-level tokenizer
#[derive(Debug, Clone, Copy, Default)]
pub struct CharTokenizer;

/// Word tokenizer
#[derive(Debug, Clone)]
pub struct WordTokenizer {
    /// Include punctuation as separate tokens
    pub include_punctuation: bool,
    
    /// Include whitespace as separate tokens
    pub include_whitespace: bool,
    
    /// Minimum word length
    pub min_length: usize,
}

impl WordTokenizer {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_punctuation() -> Self {
        Self {
            include_punctuation: true,
            ..Default::default()
        }
    }
    
    pub fn words_only() -> Self {
        Self {
            include_punctuation: false,
            include_whitespace: false,
            ..Default::default()
        }
    }
}

/// Sentence tokenizer
#[derive(Debug, Clone, Default)]
pub struct SentenceTokenizer {
    /// Sentence boundary detection strategy
    pub strategy: SentenceBoundaryStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum SentenceBoundaryStrategy {
    /// Simple: split on . ! ?
    Simple,
    
    /// Advanced: handle abbreviations, quotes, etc.
    Advanced,
}

/// N-gram tokenizer
#[derive(Debug, Clone)]
pub struct NGramTokenizer {
    /// N-gram size
    pub n: usize,
    
    /// Character-level or word-level
    pub level: NGramLevel,
    
    /// Include incomplete n-grams at boundaries
    pub include_incomplete: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum NGramLevel {
    Character,
    Word,
}
```

### 5. TextProcessor - Orchestrator

```rust
/// Orchestrates text normalization and tokenization
pub struct TextProcessor {
    /// Normalization pipeline
    normalizers: Vec<Box<dyn Normalizer>>,
    
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer>,
    
    /// Text store for string interning
    text_store: TextStore,
}

impl TextProcessor {
    /// Create a new processor with given normalizers and tokenizer
    pub fn new(
        normalizers: Vec<Box<dyn Normalizer>>,
        tokenizer: Box<dyn Tokenizer>,
    ) -> Self;
    
    /// Process text through the pipeline
    pub fn process(&mut self, text: &str) -> ProcessedText;
    
    /// Get the text store
    pub fn text_store(&self) -> &TextStore;
    
    /// Get mutable text store
    pub fn text_store_mut(&mut self) -> &mut TextStore;
}

/// Result of text processing
pub struct ProcessedText {
    /// Original text
    pub original: String,
    
    /// Normalized text (final result)
    pub normalized: String,
    
    /// Tokens
    pub tokens: Vec<Token<'static>>, // Owned tokens with static lifetime
    
    /// Normalization layers
    pub layers: Vec<NormalizationLayer>,
    
    /// Final mapping from normalized to original
    pub mapping: CharMapping,
    
    /// Reference-counted text store
    pub text_store: Arc<TextStore>,
}

/// A layer in the normalization pipeline
#[derive(Debug, Clone)]
pub struct NormalizationLayer {
    /// Text at this layer
    pub text: String,
    
    /// Mapping to previous layer
    pub mapping: CharMapping,
    
    /// Name of the normalizer that created this layer
    pub normalizer_name: String,
}

impl ProcessedText {
    /// Get a specific layer by index (0 = original)
    pub fn layer(&self, index: usize) -> Option<&str>;
    
    /// Get the final layer
    pub fn final_layer(&self) -> &str;
    
    /// Map a position in final layer to original position(s)
    pub fn map_to_original(&self, pos: usize) -> Vec<usize>;
    
    /// Map a span in final layer to original span(s)
    pub fn map_span_to_original(&self, span: Span) -> Vec<Span>;
}
```

---

## Diff Computation Layer

### 1. DiffComputer

```rust
/// Computes diff operations between two processed texts
pub struct DiffComputer {
    /// Algorithm to use
    algorithm: Box<dyn DiffAlgorithm>,
}

impl DiffComputer {
    /// Create a new diff computer with specified algorithm
    pub fn new(algorithm: Box<dyn DiffAlgorithm>) -> Self;
    
    /// Create with Myers algorithm (default)
    pub fn myers() -> Self;
    
    /// Create with Histogram algorithm
    pub fn histogram() -> Self;
    
    /// Create with Patience algorithm
    pub fn patience() -> Self;
    
    /// Compute diff operations
    pub fn compute(
        &self,
        original: &ProcessedText,
        modified: &ProcessedText,
    ) -> Result<DiffResult, DiffError>;
}
```

### 2. DiffAlgorithm Trait

```rust
/// Trait for diff algorithms
pub trait DiffAlgorithm: Send + Sync {
    /// Compute edit operations
    fn compute<'a>(
        &self,
        original_tokens: &'a [Token<'a>],
        modified_tokens: &'a [Token<'a>],
    ) -> Vec<EditOperation<'a>>;
    
    /// Get algorithm name
    fn name(&self) -> &str;
    
    /// Estimate computational complexity (for choosing algorithm)
    fn complexity(&self, original_len: usize, modified_len: usize) -> f64;
}

/// Myers O(ND) algorithm
#[derive(Debug, Clone, Copy, Default)]
pub struct MyersAlgorithm;

/// Histogram diff algorithm
#[derive(Debug, Clone, Default)]
pub struct HistogramAlgorithm {
    /// Maximum recursion depth
    pub max_depth: usize,
}

/// Patience diff algorithm
#[derive(Debug, Clone, Default)]
pub struct PatienceAlgorithm;
```

### 3. EditOperation

```rust
/// A single edit operation in a diff
#[derive(Debug, Clone)]
pub struct EditOperation<'a> {
    /// Type of operation
    pub kind: EditKind,
    
    /// Tokens in the original text
    pub original_tokens: &'a [Token<'a>],
    
    /// Tokens in the modified text
    pub modified_tokens: &'a [Token<'a>],
    
    /// Span in original text
    pub original_span: Option<Span>,
    
    /// Span in modified text
    pub modified_span: Option<Span>,
}

/// Type of edit operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditKind {
    /// Content unchanged
    Equal,
    
    /// Content deleted
    Delete,
    
    /// Content inserted
    Insert,
    
    /// Content replaced (delete + insert)
    Replace,
}

impl<'a> EditOperation<'a> {
    /// Create a new edit operation
    pub fn new(kind: EditKind) -> Self;
    
    /// Create an equal operation
    pub fn equal(tokens: &'a [Token<'a>], span: Span) -> Self;
    
    /// Create a delete operation
    pub fn delete(tokens: &'a [Token<'a>], span: Span) -> Self;
    
    /// Create an insert operation
    pub fn insert(tokens: &'a [Token<'a>], span: Span) -> Self;
    
    /// Create a replace operation
    pub fn replace(
        original_tokens: &'a [Token<'a>],
        original_span: Span,
        modified_tokens: &'a [Token<'a>],
        modified_span: Span,
    ) -> Self;
    
    /// Check if this operation represents a change
    pub fn is_change(&self) -> bool;
    
    /// Get the text content of this operation
    pub fn text(&self) -> (&str, &str);
}
```

### 4. DiffResult

```rust
/// Result of diff computation
#[derive(Debug)]
pub struct DiffResult {
    /// Edit operations
    pub operations: Vec<EditOperation<'static>>,
    
    /// Statistics
    pub statistics: DiffStatistics,
    
    /// Original processed text
    pub original: Arc<ProcessedText>,
    
    /// Modified processed text
    pub modified: Arc<ProcessedText>,
}

/// Statistics about a diff
#[derive(Debug, Clone, Default)]
pub struct DiffStatistics {
    /// Number of equal operations
    pub equal_count: usize,
    
    /// Number of delete operations
    pub delete_count: usize,
    
    /// Number of insert operations
    pub insert_count: usize,
    
    /// Number of replace operations
    pub replace_count: usize,
    
    /// Total edit distance
    pub edit_distance: usize,
    
    /// Percentage of text changed (0.0 to 1.0)
    pub change_ratio: f64,
    
    /// Number of tokens in original
    pub original_tokens: usize,
    
    /// Number of tokens in modified
    pub modified_tokens: usize,
}

impl DiffResult {
    /// Create a new diff result
    pub fn new(
        operations: Vec<EditOperation<'static>>,
        original: Arc<ProcessedText>,
        modified: Arc<ProcessedText>,
    ) -> Self;
    
    /// Check if diff is empty (no changes)
    pub fn is_empty(&self) -> bool;
    
    /// Get only the changed operations
    pub fn changes(&self) -> impl Iterator<Item = &EditOperation<'static>>;
    
    /// Get similarity ratio (inverse of change ratio)
    pub fn similarity(&self) -> f64;
}
```

---

## Metrics Engine

### 1. MetricsEngine

```rust
/// Centralized metrics computation and caching
pub struct MetricsEngine {
    /// Cache manager
    cache: Arc<CacheManager>,
    
    /// Registered metrics
    registry: MetricRegistry,
}

impl MetricsEngine {
    /// Create a new metrics engine
    pub fn new() -> Self;
    
    /// Create with custom cache
    pub fn with_cache(cache: Arc<CacheManager>) -> Self;
    
    /// Compute text metrics for a single text
    pub fn compute_text_metrics(&self, text: &str) -> Arc<TextMetrics>;
    
    /// Compute pairwise metrics for two texts
    pub fn compute_pairwise_metrics(
        &self,
        original: &str,
        modified: &str,
    ) -> Arc<PairwiseMetrics>;
    
    /// Get or compute a specific metric
    pub fn get_metric<M: Metric>(&self, key: &MetricKey) -> Result<M::Value, MetricError>;
    
    /// Register a custom metric
    pub fn register_metric<M: Metric>(&mut self, metric: M);
    
    /// Clear cache
    pub fn clear_cache(&mut self);
}
```

### 2. Metrics Types

```rust
/// Metrics for a single text
#[derive(Debug, Clone)]
pub struct TextMetrics {
    // Basic counts
    pub char_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub line_count: usize,
    
    // Character-level
    pub letter_count: usize,
    pub digit_count: usize,
    pub whitespace_count: usize,
    pub punctuation_count: usize,
    
    // Linguistic
    pub syllable_count: usize,
    pub avg_word_length: f64,
    pub avg_sentence_length: f64,
    pub unique_words: usize,
    pub vocabulary_richness: f64, // unique / total
    
    // Readability
    pub flesch_reading_ease: f64,
    pub flesch_kincaid_grade: f64,
    pub gunning_fog_index: f64,
    pub smog_index: f64,
    
    // Complexity
    pub lexical_density: f64,
    pub function_word_ratio: f64,
    
    // Features
    pub has_negation: bool,
    pub has_modal_verbs: bool,
    pub has_passive_voice: bool,
}

/// Pairwise comparison metrics
#[derive(Debug, Clone)]
pub struct PairwiseMetrics {
    /// Metrics for original text
    pub original: Arc<TextMetrics>,
    
    /// Metrics for modified text
    pub modified: Arc<TextMetrics>,
    
    // Similarity metrics
    pub char_similarity: f64,
    pub word_similarity: f64,
    pub semantic_similarity: f64,
    
    // Distance metrics
    pub levenshtein_distance: usize,
    pub damerau_levenshtein_distance: usize,
    pub jaro_winkler_similarity: f64,
    
    // Set-based metrics
    pub jaccard_similarity: f64,
    pub cosine_similarity: f64,
    pub dice_coefficient: f64,
    
    // Statistical
    pub length_ratio: f64,
    pub word_count_diff: isize,
    pub char_count_diff: isize,
    
    // Readability change
    pub readability_delta: f64,
    pub grade_level_delta: f64,
    
    // Linguistic features
    pub negation_changed: bool,
    pub tense_changed: bool,
    pub voice_changed: bool,
}

/// Key for identifying a metric computation
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum MetricKey {
    Text(ContentHash),
    Pairwise(ContentHash, ContentHash),
    Custom(String, Vec<u8>),
}

/// Content hash for cache keys
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Compute hash of content
    pub fn of(content: &str) -> Self;
    
    /// Compute hash of bytes
    pub fn of_bytes(bytes: &[u8]) -> Self;
}
```

### 3. Metric Trait

```rust
/// Trait for custom metrics
pub trait Metric: Send + Sync {
    /// Type of value this metric produces
    type Value: Send + Sync + Clone;
    
    /// Unique identifier for this metric
    fn id(&self) -> &str;
    
    /// Compute the metric
    fn compute(&self, ctx: &MetricContext) -> Result<Self::Value, MetricError>;
    
    /// Dependencies (other metrics this metric needs)
    fn dependencies(&self) -> Vec<&str> {
        Vec::new()
    }
    
    /// Estimated cost (0.0 to 1.0)
    fn cost(&self) -> f32 {
        0.1
    }
}

/// Context for metric computation
pub struct MetricContext<'a> {
    /// Original text
    pub original: &'a str,
    
    /// Modified text
    pub modified: Option<&'a str>,
    
    /// Processed text (if available)
    pub processed_original: Option<&'a ProcessedText>,
    pub processed_modified: Option<&'a ProcessedText>,
    
    /// Diff result (if available)
    pub diff: Option<&'a DiffResult>,
    
    /// Previously computed metrics
    pub computed_metrics: &'a HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}
```

---

## Analysis Framework

### 1. AnalyzerPlugin Trait

```rust
/// Plugin for analyzing diffs
pub trait AnalyzerPlugin: Send + Sync {
    /// Metadata about this analyzer
    fn metadata(&self) -> AnalyzerMetadata;
    
    /// Dependencies (metrics, other analyzers)
    fn dependencies(&self) -> Vec<Dependency>;
    
    /// Synchronous analysis
    fn analyze(&self, ctx: &AnalysisContext) -> Result<AnalysisResult, AnalysisError>;
    
    /// Asynchronous analysis (optional, defaults to sync version)
    fn analyze_async<'a>(
        &'a self,
        ctx: &'a AnalysisContext,
    ) -> Pin<Box<dyn Future<Output = Result<AnalysisResult, AnalysisError>> + Send + 'a>> {
        Box::pin(async move { self.analyze(ctx) })
    }
    
    /// Whether this analyzer supports async execution
    fn supports_async(&self) -> bool {
        false
    }
}

/// Metadata about an analyzer
#[derive(Debug, Clone)]
pub struct AnalyzerMetadata {
    /// Unique identifier
    pub id: String,
    
    /// Human-readable name
    pub name: String,
    
    /// Description
    pub description: String,
    
    /// Version
    pub version: String,
    
    /// Author
    pub author: Option<String>,
    
    /// Estimated computational cost (0.0 to 1.0)
    pub cost: f32,
    
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Dependency specification
#[derive(Debug, Clone)]
pub enum Dependency {
    /// Depends on a metric
    Metric(String),
    
    /// Depends on another analyzer
    Analyzer(String),
    
    /// Depends on specific data
    Data(DataRequirement),
}

#[derive(Debug, Clone)]
pub enum DataRequirement {
    ProcessedText,
    DiffResult,
    Tokens,
    Metrics,
}
```

### 2. AnalysisContext

```rust
/// Context provided to analyzers
pub struct AnalysisContext {
    /// The diff result
    pub diff: Arc<DiffResult>,
    
    /// Computed metrics
    pub metrics: Arc<PairwiseMetrics>,
    
    /// Additional data
    pub data: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl AnalysisContext {
    /// Create a new context
    pub fn new(diff: Arc<DiffResult>, metrics: Arc<PairwiseMetrics>) -> Self;
    
    /// Add data to context
    pub fn with_data<T: Send + Sync + 'static>(
        mut self,
        key: impl Into<String>,
        value: T,
    ) -> Self;
    
    /// Get data from context
    pub fn get<T: 'static>(&self, key: &str) -> Option<&T>;
}
```

### 3. AnalysisResult

```rust
/// Result of an analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// ID of the analyzer that produced this result
    pub analyzer_id: String,
    
    /// Computed metrics/scores
    pub metrics: HashMap<String, f64>,
    
    /// Categorical outputs
    pub categories: HashMap<String, String>,
    
    /// Insights (human-readable findings)
    pub insights: Vec<Insight>,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    
    /// Arbitrary data
    pub data: HashMap<String, serde_json::Value>,
}

/// An insight from analysis
#[derive(Debug, Clone)]
pub struct Insight {
    /// Severity level
    pub level: InsightLevel,
    
    /// Category
    pub category: String,
    
    /// Message
    pub message: String,
    
    /// Associated span (if applicable)
    pub span: Option<Span>,
    
    /// Confidence (0.0 to 1.0)
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightLevel {
    Info,
    Note,
    Warning,
    Critical,
}
```

### 4. AnalysisCoordinator

```rust
/// Coordinates execution of multiple analyzers
pub struct AnalysisCoordinator {
    /// Registry of available plugins
    registry: PluginRegistry,
    
    /// Cache manager
    cache: Arc<CacheManager>,
    
    /// Execution planner
    planner: ExecutionPlanner,
}

impl AnalysisCoordinator {
    /// Create a new coordinator
    pub fn new() -> Self;
    
    /// Register an analyzer plugin
    pub fn register(&mut self, plugin: Box<dyn AnalyzerPlugin>) -> Result<(), AnalysisError>;
    
    /// Unregister a plugin
    pub fn unregister(&mut self, id: &str) -> Result<(), AnalysisError>;
    
    /// List registered plugins
    pub fn list_plugins(&self) -> Vec<AnalyzerMetadata>;
    
    /// Analyze with specified plugins (synchronous)
    pub fn analyze(
        &self,
        ctx: &AnalysisContext,
        plugin_ids: &[String],
    ) -> Result<AnalysisReport, AnalysisError>;
    
    /// Analyze with parallel execution (asynchronous)
    pub async fn analyze_async(
        &self,
        ctx: &AnalysisContext,
        plugin_ids: &[String],
    ) -> Result<AnalysisReport, AnalysisError>;
    
    /// Analyze with all registered plugins
    pub fn analyze_all(&self, ctx: &AnalysisContext) -> Result<AnalysisReport, AnalysisError>;
}

/// Report containing all analysis results
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// Individual analyzer results
    pub results: Vec<AnalysisResult>,
    
    /// Aggregated metrics
    pub summary: HashMap<String, f64>,
    
    /// All insights combined
    pub insights: Vec<Insight>,
    
    /// Execution metadata
    pub metadata: ExecutionMetadata,
}

#[derive(Debug, Clone)]
pub struct ExecutionMetadata {
    /// Total execution time
    pub duration: std::time::Duration,
    
    /// Number of analyzers run
    pub analyzer_count: usize,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
}
```

### 5. Built-in Analyzers

```rust
/// Semantic similarity analyzer
pub struct SemanticAnalyzer {
    /// Similarity threshold
    pub threshold: f64,
}

/// Stylistic change analyzer
pub struct StylisticAnalyzer {
    /// Features to analyze
    pub features: Vec<StylisticFeature>,
}

#[derive(Debug, Clone, Copy)]
pub enum StylisticFeature {
    Formality,
    Conciseness,
    Tone,
    Complexity,
}

/// Readability analyzer
pub struct ReadabilityAnalyzer {
    /// Target grade level
    pub target_grade: Option<f64>,
}

/// Edit classification analyzer
pub struct EditClassifier {
    /// Classification strategy
    pub strategy: ClassificationStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum ClassificationStrategy {
    RuleBased,
    MachineLearning,
    Hybrid,
}

/// Intent detection analyzer
pub struct IntentAnalyzer {
    /// Intent categories to detect
    pub categories: Vec<IntentCategory>,
}

#[derive(Debug, Clone, Copy)]
pub enum IntentCategory {
    Clarification,
    Expansion,
    Reduction,
    Correction,
    Reformulation,
    Stylistic,
}
```

---

## Caching System

### 1. CacheManager

```rust
/// Manages caching of computed values
pub struct CacheManager {
    /// Metrics cache
    metrics_cache: Arc<RwLock<LruCache<MetricKey, Arc<dyn Any + Send + Sync>>>>,
    
    /// Analysis results cache
    analysis_cache: Arc<RwLock<LruCache<AnalysisCacheKey, Arc<AnalysisResult>>>>,
    
    /// Cache policy
    policy: CachePolicy,
    
    /// Statistics
    stats: Arc<RwLock<CacheStatistics>>,
}

impl CacheManager {
    /// Create with default policy
    pub fn new() -> Self;
    
    /// Create with custom policy
    pub fn with_policy(policy: CachePolicy) -> Self;
    
    /// Get or compute a metric
    pub fn get_or_compute_metric<F, T>(
        &self,
        key: &MetricKey,
        compute: F,
    ) -> Arc<T>
    where
        F: FnOnce() -> T,
        T: Send + Sync + 'static;
    
    /// Get or compute analysis result
    pub fn get_or_compute_analysis<F>(
        &self,
        key: &AnalysisCacheKey,
        compute: F,
    ) -> Arc<AnalysisResult>
    where
        F: FnOnce() -> AnalysisResult;
    
    /// Invalidate cache entries
    pub fn invalidate(&self, predicate: impl Fn(&MetricKey) -> bool);
    
    /// Clear all caches
    pub fn clear(&self);
    
    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics;
}

/// Cache policy configuration
#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// Maximum cache size in bytes
    pub max_size: usize,
    
    /// TTL for entries
    pub ttl: Option<std::time::Duration>,
    
    /// Eviction strategy
    pub eviction: EvictionStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum EvictionStrategy {
    LRU,
    LFU,
    FIFO,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_size: usize,
}

impl CacheStatistics {
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
}
```

---

## Orchestration Layer

### 1. DiffOrchestrator

```rust
/// High-level orchestrator for complete diff workflow
pub struct DiffOrchestrator {
    /// Text processor
    text_processor: TextProcessor,
    
    /// Diff computer
    diff_computer: DiffComputer,
    
    /// Metrics engine
    metrics_engine: MetricsEngine,
    
    /// Analysis coordinator
    analysis_coordinator: AnalysisCoordinator,
}

impl DiffOrchestrator {
    /// Create a new orchestrator with default configuration
    pub fn new() -> Self;
    
    /// Create with custom configuration
    pub fn with_config(config: &OrchestratorConfig) -> Result<Self, ConfigError>;
    
    /// Compute diff only (no analysis)
    pub fn diff(
        &mut self,
        original: &str,
        modified: &str,
    ) -> Result<DiffResult, RedlineError>;
    
    /// Compute diff with full analysis
    pub fn diff_with_analysis(
        &mut self,
        original: &str,
        modified: &str,
        analyzers: &[String],
    ) -> Result<DiffReport, RedlineError>;
    
    /// Async version with parallel analysis
    pub async fn diff_with_analysis_async(
        &mut self,
        original: &str,
        modified: &str,
        analyzers: &[String],
    ) -> Result<DiffReport, RedlineError>;
    
    /// Batch diff multiple pairs
    pub fn batch_diff(
        &mut self,
        pairs: &[(&str, &str)],
    ) -> Vec<Result<DiffResult, RedlineError>>;
    
    /// Access components
    pub fn text_processor(&self) -> &TextProcessor;
    pub fn diff_computer(&self) -> &DiffComputer;
    pub fn metrics_engine(&self) -> &MetricsEngine;
    pub fn analysis_coordinator(&self) -> &AnalysisCoordinator;
}

/// Configuration for orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Text processing configuration
    pub text_processing: TextProcessingConfig,
    
    /// Diff algorithm
    pub diff_algorithm: DiffAlgorithmConfig,
    
    /// Metrics configuration
    pub metrics: MetricsConfig,
    
    /// Analysis configuration
    pub analysis: AnalysisConfig,
    
    /// Caching configuration
    pub cache: CachePolicy,
}
```

---

## Configuration System

### 1. Configuration Builder

```rust
/// Fluent builder for configuration
pub struct ConfigBuilder {
    text_processing: TextProcessingConfig,
    diff_algorithm: DiffAlgorithmConfig,
    metrics: MetricsConfig,
    analysis: AnalysisConfig,
    cache: CachePolicy,
}

impl ConfigBuilder {
    /// Start with a preset
    pub fn preset(preset: Preset) -> Self;
    
    /// Start from scratch
    pub fn new() -> Self;
    
    /// Configure text processing
    pub fn text_processing(mut self, f: impl FnOnce(TextProcessingBuilder) -> TextProcessingConfig) -> Self;
    
    /// Configure diff algorithm
    pub fn diff_algorithm(mut self, algo: DiffAlgorithmConfig) -> Self;
    
    /// Configure metrics
    pub fn metrics(mut self, f: impl FnOnce(MetricsBuilder) -> MetricsConfig) -> Self;
    
    /// Configure analysis
    pub fn analysis(mut self, f: impl FnOnce(AnalysisBuilder) -> AnalysisConfig) -> Self;
    
    /// Configure cache
    pub fn cache(mut self, policy: CachePolicy) -> Self;
    
    /// Build the configuration
    pub fn build(self) -> Result<OrchestratorConfig, ConfigError>;
}

/// Presets
#[derive(Debug, Clone, Copy)]
pub enum Preset {
    /// Minimal (fastest, no analysis)
    Fast,
    
    /// Syntactic comparison
    Syntactic,
    
    /// Semantic comparison
    Semantic,
    
    /// Full analysis
    Comprehensive,
}

/// Example usage:
///
/// let config = ConfigBuilder::preset(Preset::Semantic)
///     .text_processing(|t| t
///         .normalize(|n| n.lowercase().collapse_whitespace())
///         .tokenize(Tokenization::Words)
///     )
///     .analysis(|a| a
///         .add_analyzer("semantic")
///         .add_analyzer("readability")
///     )
///     .build()?;
```

---

## Error Handling

```rust
/// Main error type
#[derive(Debug, thiserror::Error)]
pub enum RedlineError {
    #[error("Text processing error: {0}")]
    TextProcessing(#[from] TextProcessingError),
    
    #[error("Diff computation error: {0}")]
    DiffComputation(#[from] DiffError),
    
    #[error("Metrics error: {0}")]
    Metrics(#[from] MetricError),
    
    #[error("Analysis error: {0}")]
    Analysis(#[from] AnalysisError),
    
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum TextProcessingError {
    #[error("Normalization failed: {0}")]
    Normalization(String),
    
    #[error("Tokenization failed: {0}")]
    Tokenization(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

#[derive(Debug, thiserror::Error)]
pub enum DiffError {
    #[error("Algorithm error: {0}")]
    Algorithm(String),
    
    #[error("Invalid tokens: {0}")]
    InvalidTokens(String),
}

#[derive(Debug, thiserror::Error)]
pub enum MetricError {
    #[error("Metric '{0}' not found")]
    NotFound(String),
    
    #[error("Computation failed: {0}")]
    Computation(String),
    
    #[error("Dependency error: {0}")]
    Dependency(String),
}

#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("Analyzer '{0}' not found")]
    NotFound(String),
    
    #[error("Analysis failed: {0}")]
    Failed(String),
    
    #[error("Dependency not satisfied: {0}")]
    DependencyNotSatisfied(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid configuration: {0}")]
    Invalid(String),
    
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Validation failed: {0}")]
    Validation(String),
}
```

---

## API Reference

### Public API (Main Entry Points)

```rust
// Simple API
pub fn diff(original: &str, modified: &str) -> Result<DiffResult>;

pub fn diff_with_config(
    original: &str,
    modified: &str,
    config: &OrchestratorConfig,
) -> Result<DiffResult>;

pub fn diff_with_analysis(
    original: &str,
    modified: &str,
    analyzers: &[&str],
) -> Result<DiffReport>;

// Async API
pub async fn diff_async(original: &str, modified: &str) -> Result<DiffResult>;

pub async fn diff_with_analysis_async(
    original: &str,
    modified: &str,
    analyzers: &[&str],
) -> Result<DiffReport>;

// Builder API
let orchestrator = DiffOrchestrator::builder()
    .preset(Preset::Semantic)
    .build()?;

let result = orchestrator.diff(original, modified)?;
```

---

## Performance Considerations

### Memory Usage

- **Target**: <50MB for typical workload (10K words)
- **String interning**: Reduces memory by ~40%
- **Arena allocation**: Reduces allocator overhead
- **Reference counting**: Share immutable data

### CPU Usage

- **Lazy evaluation**: Compute metrics only when needed
- **Caching**: Avoid redundant computations
- **Parallel analysis**: Utilize multiple cores
- **SIMD**: Use for string comparisons where applicable

### Benchmarks (Target)

| Operation | Throughput | Latency |
|-----------|-----------|---------|
| Simple diff (100 words) | 1666 ops/s | 0.6ms |
| With normalization | 1000 ops/s | 1.0ms |
| Full analysis | 50 ops/s | 20ms |
| Large doc (10K words) | 6.6 ops/s | 150ms |

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- ✅ TextStore with string interning
- ✅ Span and position types
- ✅ CharMapping
- ✅ Basic error types

### Phase 2: Text Processing (Week 3-4)
- Normalizer trait and built-ins
- Tokenizer trait and built-ins
- TextProcessor orchestrator
- Pipeline with layers

### Phase 3: Diff Computation (Week 5-6)
- DiffAlgorithm trait
- Myers algorithm
- Histogram algorithm
- DiffComputer
- EditOperation types

### Phase 4: Metrics (Week 7-8)
- Metric trait
- TextMetrics computation
- PairwiseMetrics computation
- MetricsEngine with caching

### Phase 5: Analysis Framework (Week 9-10)
- AnalyzerPlugin trait
- AnalysisCoordinator
- Built-in analyzers
- Plugin registry

### Phase 6: Caching & Optimization (Week 11-12)
- CacheManager
- Content-addressed caching
- LRU eviction
- Performance tuning

### Phase 7: Orchestration & API (Week 13-14)
- DiffOrchestrator
- Configuration system
- Builder APIs
- Async support

### Phase 8: Testing & Documentation (Week 15-16)
- Unit tests
- Integration tests
- Property-based tests
- Documentation
- Examples

---

**Document Status**: Draft for Implementation  
**Next Review**: After Phase 1 completion
