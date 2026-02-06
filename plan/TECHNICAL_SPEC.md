# Redline Core v1.0 - Technical Specification

**Version:** 1.0
**Author:** Parth Patel
**Date:** 2025-10-23
**Status:** In Review

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
10. [Async Design and Error Handling](#async-design-and-error-handling)
    - [Async Runtime Considerations](#async-runtime-considerations)
    - [Error Handling Strategy](#error-handling-strategy)
    - [Async Error Propagation](#async-error-propagation)
    - [Error Recovery](#error-recovery)
11. [API Reference](#api-reference)
12. [Python Bindings (PyO3)](#python-bindings-pyo3)
    - [Module Structure](#module-structure)
    - [Core Type Bindings](#core-type-bindings)
    - [Python Plugin Traits](#python-plugin-traits)
    - [Async Support (pyo3-asyncio)](#async-support-pyo3-asyncio)
    - [Error Handling](#error-handling)
    - [Python Module Definition](#python-module-definition)
    - [Python Type Stubs](#python-type-stubs-__init__pyi)
    - [Automated Stub Generation](#automated-stub-generation)
13. [Thread Safety and Memory Safety](#thread-safety-and-memory-safety)
14. [Performance Considerations](#performance-considerations)
15. [Required Dependencies](#required-dependencies)
16. [Implementation Phases](#implementation-phases)
17. [Document Change Log](#document-change-log)

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
    ///
    /// Tokens are created by interning text into the store and storing the StringId.
    /// This allows tokens to be Copy and have no lifetime dependencies.
    fn tokenize(
        &self,
        text: &str,
        mapping: &CharMapping,
        store: &mut TextStore,
    ) -> Vec<Token>;

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

/// Token with ID reference to text in store
///
/// Design Note: Tokens store a StringId rather than borrowing text directly.
/// This allows tokens to be Copy, freely moved/cloned, and stored in caches,
/// while actual text access happens through the TextStore via the StringId.
///
/// Performance: Text access via ID is O(1) array lookup (~2ns).
/// Memory: ~56-64 bytes per token (no lifetime overhead).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token {
    /// ID of token text in the text store
    pub text_id: StringId,

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

impl Token {
    /// Create a new token
    pub fn new(
        text_id: StringId,
        span: Span,
        original_spans: SmallVec<[Span; 1]>,
        index: usize,
        kind: TokenKind,
    ) -> Self;

    /// Get the token text as a string slice (requires TextStore reference)
    pub fn text<'a>(&self, store: &'a TextStore) -> &'a str {
        store.get(self.text_id)
    }

    /// Get the token's length in bytes (requires TextStore reference)
    pub fn len(&self, store: &TextStore) -> usize {
        store.get(self.text_id).len()
    }

    /// Check if token is empty (requires TextStore reference)
    pub fn is_empty(&self, store: &TextStore) -> bool {
        store.get(self.text_id).is_empty()
    }

    /// Get the text ID
    pub fn text_id(&self) -> StringId {
        self.text_id
    }
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

    /// Tokens (contain StringIds referencing the text store)
    pub tokens: Vec<Token>,

    /// Normalization layers
    pub layers: Vec<NormalizationLayer>,

    /// Final mapping from normalized to original
    pub mapping: CharMapping,

    /// Reference-counted text store for looking up token text
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
    ///
    /// Since tokens are Copy and contain only IDs, they can be freely
    /// copied into the EditOperations without lifetime constraints.
    fn compute(
        &self,
        original_tokens: &[Token],
        modified_tokens: &[Token],
    ) -> Vec<EditOperation>;

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
///
/// Design Note: Uses slices to reference tokens stored in ProcessedText,
/// avoiding duplication. Since Token is Copy (contains only StringIds),
/// individual tokens can be cheaply copied when needed, while slices
/// provide zero-copy access to token ranges.
#[derive(Debug, Clone, Copy)]
pub struct EditOperation<'a> {
    /// Type of operation
    pub kind: EditKind,

    /// Tokens in the original text (slice into ProcessedText)
    pub original_tokens: &'a [Token],

    /// Tokens in the modified text (slice into ProcessedText)
    pub modified_tokens: &'a [Token],

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
    pub fn equal(tokens: &'a [Token], span: Span) -> Self;

    /// Create a delete operation
    pub fn delete(tokens: &'a [Token], span: Span) -> Self;

    /// Create an insert operation
    pub fn insert(tokens: &'a [Token], span: Span) -> Self;

    /// Create a replace operation
    pub fn replace(
        original_tokens: &'a [Token],
        original_span: Span,
        modified_tokens: &'a [Token],
        modified_span: Span,
    ) -> Self;

    /// Check if this operation represents a change
    pub fn is_change(&self) -> bool {
        !matches!(self.kind, EditKind::Equal)
    }

    /// Get reconstructed text for this operation (requires TextStore)
    pub fn text<'b>(&self, store: &'b TextStore) -> (String, String) {
        let original = self.original_tokens
            .iter()
            .map(|t| t.text(store))
            .collect::<Vec<_>>()
            .join("");
        let modified = self.modified_tokens
            .iter()
            .map(|t| t.text(store))
            .collect::<Vec<_>>()
            .join("");
        (original, modified)
    }
}
```

### 4. DiffResult

```rust
/// Result of diff computation
///
/// Design Note: The lifetime 'a ties operations to the ProcessedText
/// stored in the Arc. Since Token is Copy, operations can be cheaply
/// copied, but they reference token slices in the ProcessedText.
#[derive(Debug, Clone)]
pub struct DiffResult<'a> {
    /// Edit operations (reference tokens in ProcessedText)
    pub operations: Vec<EditOperation<'a>>,

    /// Statistics
    pub statistics: DiffStatistics,

    /// Original processed text (keeps tokens alive)
    pub original: Arc<ProcessedText>,

    /// Modified processed text (keeps tokens alive)
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

impl<'a> DiffResult<'a> {
    /// Create a new diff result
    pub fn new(
        operations: Vec<EditOperation<'a>>,
        original: Arc<ProcessedText>,
        modified: Arc<ProcessedText>,
    ) -> Self;

    /// Check if diff is empty (no changes)
    pub fn is_empty(&self) -> bool {
        self.statistics.edit_distance == 0
    }

    /// Get only the changed operations
    pub fn changes(&self) -> impl Iterator<Item = &EditOperation<'a>> {
        self.operations.iter().filter(|op| op.is_change())
    }

    /// Get similarity ratio (inverse of change ratio)
    pub fn similarity(&self) -> f64 {
        1.0 - self.statistics.change_ratio
    }

    /// Get the text store for accessing token text
    pub fn text_store(&self) -> &Arc<TextStore> {
        &self.original.text_store
    }
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
    fn analyze<'a>(&self, ctx: &AnalysisContext<'a>) -> Result<AnalysisResult, AnalysisError>;

    /// Asynchronous analysis (optional, defaults to sync version)
    fn analyze_async<'a, 'async_trait>(
        &'a self,
        ctx: &'a AnalysisContext<'async_trait>,
    ) -> Pin<Box<dyn Future<Output = Result<AnalysisResult, AnalysisError>> + Send + 'a>>
    where
        'async_trait: 'a,
    {
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
///
/// Design Note: The lifetime 'a is inherited from DiffResult, which references
/// tokens in the ProcessedText. The Arc keeps the ProcessedText alive.
pub struct AnalysisContext<'a> {
    /// The diff result
    pub diff: Arc<DiffResult<'a>>,

    /// Computed metrics
    pub metrics: Arc<PairwiseMetrics>,

    /// Additional data
    pub data: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl<'a> AnalysisContext<'a> {
    /// Create a new context
    pub fn new(diff: Arc<DiffResult<'a>>, metrics: Arc<PairwiseMetrics>) -> Self;

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
    pub fn analyze<'a>(
        &self,
        ctx: &AnalysisContext<'a>,
        plugin_ids: &[String],
    ) -> Result<AnalysisReport, AnalysisError>;

    /// Analyze with parallel execution (asynchronous)
    pub async fn analyze_async<'a>(
        &self,
        ctx: &AnalysisContext<'a>,
        plugin_ids: &[String],
    ) -> Result<AnalysisReport, AnalysisError>;

    /// Analyze with all registered plugins
    pub fn analyze_all<'a>(
        &self,
        ctx: &AnalysisContext<'a>,
    ) -> Result<AnalysisReport, AnalysisError>;
}

/// Report containing all analysis results
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// Individual analyzer results
    pub results: Vec<AnalysisResult>,

    /// Errors that occurred during analysis
    pub errors: Vec<AnalysisError>,

    /// Aggregated metrics
    pub summary: HashMap<String, f64>,

    /// All insights combined
    pub insights: Vec<Insight>,

    /// Execution metadata
    pub metadata: ExecutionMetadata,

    /// Whether the analysis completed fully
    pub completed: bool,

    /// Partial completion percentage (0.0-1.0)
    pub completion_ratio: f64,
}

impl AnalysisReport {
    /// Check if any results were produced
    pub fn has_results(&self) -> bool {
        !self.results.is_empty()
    }

    /// Check if analysis was fully successful
    pub fn is_complete(&self) -> bool {
        self.completed && self.errors.is_empty()
    }
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

### Configuration Validation

The `build()` method validates configuration before creating the orchestrator:

```rust
impl ConfigBuilder {
    pub fn build(self) -> Result<OrchestratorConfig, ConfigError> {
        // Validate text processing configuration
        self.validate_text_processing()?;

        // Validate algorithm selection
        self.validate_diff_algorithm()?;

        // Validate analyzer dependencies
        self.validate_analyzer_dependencies()?;

        // Validate cache policy
        self.validate_cache_policy()?;

        Ok(OrchestratorConfig { /* ... */ })
    }

    fn validate_text_processing(&self) -> Result<(), ConfigError> {
        // Check normalizer chain isn't empty
        if self.text_processing.normalizers.is_empty() {
            return Err(ConfigError::Validation(
                "At least one normalizer required".into()
            ));
        }

        // Check tokenizer is set
        if self.text_processing.tokenizer.is_none() {
            return Err(ConfigError::MissingField("tokenizer".into()));
        }

        // Warn about conflicting normalizers
        let has_lowercase = self.text_processing.normalizers
            .iter()
            .any(|n| n.name() == "lowercase");
        let has_uppercase = self.text_processing.normalizers
            .iter()
            .any(|n| n.name() == "uppercase");

        if has_lowercase && has_uppercase {
            return Err(ConfigError::Invalid(
                "Cannot use both lowercase and uppercase normalizers".into()
            ));
        }

        Ok(())
    }

    fn validate_diff_algorithm(&self) -> Result<(), ConfigError> {
        // Check for very large inputs with slow algorithms
        if let Some(max_len) = self.diff_algorithm.max_input_length {
            match self.diff_algorithm.algorithm {
                DiffAlgorithmType::Myers if max_len > 100_000 => {
                    // Myers is O(ND), can be slow for large D
                    return Err(ConfigError::Validation(
                        "Myers algorithm not recommended for inputs >100K tokens. \
                         Consider Histogram or Patience.".into()
                    ));
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn validate_analyzer_dependencies(&self) -> Result<(), ConfigError> {
        // Build dependency graph
        let mut graph = DependencyGraph::new();

        for analyzer_id in &self.analysis.enabled_analyzers {
            if let Some(analyzer) = self.analysis.available_analyzers.get(analyzer_id) {
                for dep in analyzer.dependencies() {
                    graph.add_edge(analyzer_id, &dep);
                }
            } else {
                return Err(ConfigError::NotFound(
                    format!("Analyzer '{}' not found", analyzer_id)
                ));
            }
        }

        // Check for cycles
        if let Some(cycle) = graph.find_cycle() {
            return Err(ConfigError::Validation(
                format!("Circular dependency detected: {}", cycle.join(" -> "))
            ));
        }

        // Check all dependencies are available
        for analyzer_id in &self.analysis.enabled_analyzers {
            for dep in self.analysis.available_analyzers[analyzer_id].dependencies() {
                match dep {
                    Dependency::Analyzer(dep_id) => {
                        if !self.analysis.enabled_analyzers.contains(&dep_id) {
                            return Err(ConfigError::DependencyNotSatisfied(
                                format!("Analyzer '{}' requires '{}' which is not enabled",
                                    analyzer_id, dep_id)
                            ));
                        }
                    }
                    Dependency::Metric(metric_name) => {
                        // Check metric is computable
                        if !self.metrics.available_metrics.contains(&metric_name) {
                            return Err(ConfigError::DependencyNotSatisfied(
                                format!("Analyzer '{}' requires metric '{}' which is not available",
                                    analyzer_id, metric_name)
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn validate_cache_policy(&self) -> Result<(), ConfigError> {
        // Check cache size is reasonable
        if self.cache.max_size > 10_000_000_000 {  // 10GB
            return Err(ConfigError::Validation(
                "Cache size > 10GB is not recommended".into()
            ));
        }

        // Check TTL is reasonable
        if let Some(ttl) = self.cache.ttl {
            if ttl.as_secs() < 1 {
                return Err(ConfigError::Validation(
                    "Cache TTL must be at least 1 second".into()
                ));
            }
        }

        Ok(())
    }
}
```

### Validation Rules Summary

| Rule | Check | Error Type |
|------|-------|------------|
| **Normalizers** | At least one normalizer | `MissingField` |
| **Tokenizer** | Tokenizer must be set | `MissingField` |
| **Conflicting normalizers** | Uppercase + Lowercase | `Invalid` |
| **Algorithm scaling** | Myers with >100K tokens | `Validation` (warning) |
| **Analyzer existence** | All enabled analyzers exist | `NotFound` |
| **Dependency cycles** | No circular dependencies | `Validation` |
| **Dependency availability** | Required analyzers enabled | `DependencyNotSatisfied` |
| **Metric availability** | Required metrics computable | `DependencyNotSatisfied` |
| **Cache size** | < 10GB | `Validation` (warning) |
| **Cache TTL** | >= 1 second | `Validation` |

---

## Async Design and Error Handling

### Async Runtime Considerations

The library is **runtime-agnostic** but optimized for `tokio`:

```rust
// Works with any async runtime
pub async fn diff_async(original: &str, modified: &str) -> Result<DiffResult>;

// Internal: Uses tokio for CPU-bound work
async fn run_expensive_analysis<F, T>(compute: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(compute)
        .await
        .expect("Task panicked")
}
```

### Error Handling Strategy

#### Structured Errors with Context

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

    #[error("Async runtime error: {0}")]
    AsyncRuntime(String),

    #[error("Operation cancelled")]
    Cancelled,

    #[error("Timeout after {0:?}")]
    Timeout(std::time::Duration),
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

    #[error("Component not found: {0}")]
    NotFound(String),

    #[error("Dependency not satisfied: {0}")]
    DependencyNotSatisfied(String),
}
```

### Async Error Propagation

#### Graceful Degradation

When some analyzers fail, continue with others:

```rust
pub async fn analyze_async_with_fallback<'a>(
    &self,
    ctx: &AnalysisContext<'a>,
    plugin_ids: &[String],
) -> AnalysisReport {
    let results: Vec<_> = futures::future::join_all(
        plugin_ids.iter().map(|id| async {
            self.run_analyzer_safe(id, ctx).await
        })
    ).await;

    // Separate successes from failures
    let (successes, failures): (Vec<_>, Vec<_>) = results
        .into_iter()
        .partition(Result::is_ok);

    AnalysisReport {
        results: successes.into_iter().map(Result::unwrap).collect(),
        errors: failures.into_iter().map(Result::unwrap_err).collect(),
        // ...
    }
}
```

#### Timeout Support

```rust
use tokio::time::{timeout, Duration};

pub async fn analyze_with_timeout<'a>(
    &self,
    ctx: &AnalysisContext<'a>,
    plugin_ids: &[String],
    duration: Duration,
) -> Result<AnalysisReport, RedlineError> {
    timeout(duration, self.analyze_async(ctx, plugin_ids))
        .await
        .map_err(|_| RedlineError::Timeout(duration))?
}
```

#### Cancellation Safety

All async operations should be cancellation-safe:

```rust
// ✅ Cancellation-safe: No partial state changes
pub async fn diff_async(original: &str, modified: &str) -> Result<DiffResult> {
    // All mutations happen in local variables
    let processed_original = process_text(original).await?;
    let processed_modified = process_text(modified).await?;

    // Only return when fully complete
    Ok(DiffResult::new(processed_original, processed_modified))
}

// ❌ NOT cancellation-safe: Mutates shared state
pub async fn bad_diff_async(&mut self, original: &str) -> Result<()> {
    self.cache.insert(...);  // If cancelled here, cache is inconsistent!
    let result = compute().await?;
    self.results.push(result);  // Might never execute
}
```

### Error Recovery

#### Retry Logic for Transient Failures

```rust
async fn analyze_with_retry<'a>(
    &self,
    ctx: &AnalysisContext<'a>,
    max_retries: usize,
) -> Result<AnalysisResult, AnalysisError> {
    let mut attempts = 0;

    loop {
        match self.analyze(ctx) {
            Ok(result) => return Ok(result),
            Err(e) if attempts < max_retries && e.is_transient() => {
                attempts += 1;
                tokio::time::sleep(Duration::from_millis(100 * attempts)).await;
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}
```

#### Partial Results

```rust
pub struct AnalysisReport {
    /// Successful results
    pub results: Vec<AnalysisResult>,

    /// Errors that occurred
    pub errors: Vec<AnalysisError>,

    /// Whether the analysis completed fully
    pub completed: bool,

    /// Partial completion percentage (0.0-1.0)
    pub completion_ratio: f64,
}

impl AnalysisReport {
    /// Check if any results were produced
    pub fn has_results(&self) -> bool {
        !self.results.is_empty()
    }

    /// Check if analysis was fully successful
    pub fn is_complete(&self) -> bool {
        self.completed && self.errors.is_empty()
    }
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

## Python Bindings (PyO3)

### Overview

Complete Python bindings expose the entire Redline Core API and allow Python users to implement custom plugins for all extensible components.

### Module Structure

```
python/
├── src/
│   ├── lib.rs                  # PyO3 module definition
│   │
│   ├── foundation/             # Foundation bindings
│   │   ├── mod.rs
│   │   ├── text_store.rs       # PyTextStore
│   │   ├── span.rs             # PySpan
│   │   └── types.rs            # PyStringId, etc.
│   │
│   ├── text/                   # Text processing bindings
│   │   ├── mod.rs
│   │   ├── normalizer.rs       # PyNormalizer trait + adapters
│   │   ├── tokenizer.rs        # PyTokenizer trait + adapters
│   │   ├── token.rs            # PyToken
│   │   └── processed_text.rs  # PyProcessedText
│   │
│   ├── diff/                   # Diff bindings
│   │   ├── mod.rs
│   │   ├── algorithm.rs        # PyDiffAlgorithm trait + adapters
│   │   ├── result.rs           # PyDiffResult
│   │   └── operation.rs        # PyEditOperation
│   │
│   ├── metrics/                # Metrics bindings
│   │   ├── mod.rs
│   │   ├── metric.rs           # PyMetric trait
│   │   ├── text_metrics.rs    # PyTextMetrics
│   │   └── pairwise.rs         # PyPairwiseMetrics
│   │
│   ├── analysis/               # Analysis bindings
│   │   ├── mod.rs
│   │   ├── plugin.rs           # PyAnalyzerPlugin trait + adapters
│   │   ├── context.rs          # PyAnalysisContext
│   │   └── result.rs           # PyAnalysisResult
│   │
│   ├── orchestration/          # Orchestration bindings
│   │   └── orchestrator.rs    # PyDiffOrchestrator
│   │
│   ├── config/                 # Configuration bindings
│   │   └── builder.rs          # PyConfigBuilder
│   │
│   ├── error.rs                # Python exception types
│   ├── conversions.rs          # Rust ↔ Python conversions
│   └── async_bridge.rs         # Tokio ↔ asyncio bridge
│
└── python/
    └── redline_core/
        ├── __init__.py         # Python package
        ├── __init__.pyi        # Type stubs
        └── py.typed            # PEP 561 marker
```

### Core Type Bindings

#### PyToken

```rust
use pyo3::prelude::*;

#[pyclass(name = "Token")]
#[derive(Clone)]
pub struct PyToken {
    pub(crate) inner: Token,
    pub(crate) store: Arc<TextStore>,  // Keep store alive
}

#[pymethods]
impl PyToken {
    /// Get the token text
    #[getter]
    fn text(&self) -> PyResult<String> {
        Ok(self.inner.text(&self.store).to_string())
    }

    /// Get the token span
    #[getter]
    fn span(&self) -> PyResult<PySpan> {
        Ok(PySpan { inner: self.inner.span })
    }

    /// Get the token index
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    /// Get the token kind
    #[getter]
    fn kind(&self) -> String {
        format!("{:?}", self.inner.kind)
    }

    fn __repr__(&self) -> String {
        format!("Token(text='{}', span={}..{}, index={})",
            self.inner.text(&self.store),
            self.inner.span.start,
            self.inner.span.end,
            self.inner.index)
    }

    fn __str__(&self) -> String {
        self.inner.text(&self.store).to_string()
    }
}
```

#### PyDiffResult

```rust
#[pyclass(name = "DiffResult")]
pub struct PyDiffResult {
    pub(crate) inner: Arc<DiffResult<'static>>,
}

#[pymethods]
impl PyDiffResult {
    /// Get all edit operations
    #[getter]
    fn operations(&self) -> Vec<PyEditOperation> {
        self.inner.operations.iter()
            .map(|op| PyEditOperation::from_rust(op, self.inner.clone()))
            .collect()
    }

    /// Get statistics
    #[getter]
    fn statistics(&self) -> PyDiffStatistics {
        PyDiffStatistics {
            inner: self.inner.statistics.clone()
        }
    }

    /// Get similarity ratio
    fn similarity(&self) -> f64 {
        self.inner.similarity()
    }

    /// Check if diff is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get only changed operations
    fn changes(&self) -> Vec<PyEditOperation> {
        self.inner.changes()
            .map(|op| PyEditOperation::from_rust(op, self.inner.clone()))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.operations.len()
    }

    fn __repr__(&self) -> String {
        format!("DiffResult(operations={}, similarity={:.2}%)",
            self.inner.operations.len(),
            self.similarity() * 100.0)
    }
}
```

### Python Plugin Traits

#### PyNormalizer

```rust
/// Python base class for normalizers
#[pyclass(subclass)]
pub struct PyNormalizer;

#[pymethods]
impl PyNormalizer {
    #[new]
    fn new() -> Self {
        PyNormalizer
    }

    /// Get normalizer name (must be overridden)
    fn name(&self) -> PyResult<String> {
        Err(PyNotImplementedError::new_err("Subclass must implement name()"))
    }

    /// Normalize text (must be overridden)
    fn normalize(&self, _text: String) -> PyResult<PyNormalizationResult> {
        Err(PyNotImplementedError::new_err("Subclass must implement normalize()"))
    }

    /// Get metadata (optional)
    fn metadata(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("description", "Custom normalizer")?;
            dict.set_item("deterministic", true)?;
            Ok(dict.into())
        })
    }

    /// Get computational cost estimate (optional)
    fn cost(&self) -> f32 {
        0.1
    }
}

/// Adapter that wraps Python normalizer to implement Rust trait
pub struct PyNormalizerAdapter {
    py_object: Py<PyAny>,
}

impl Normalizer for PyNormalizerAdapter {
    fn normalize(&self, input: &str) -> NormalizationResult {
        Python::with_gil(|py| {
            let result = self.py_object
                .call_method1(py, "normalize", (input,))
                .map_err(|e| TextProcessingError::Normalization(e.to_string()))?;

            let py_result: PyNormalizationResult = result.extract(py)
                .map_err(|e| TextProcessingError::Normalization(
                    format!("Invalid return type: {}", e)
                ))?;

            Ok(py_result.into_rust())
        })
    }

    fn name(&self) -> &str {
        Python::with_gil(|py| {
            self.py_object
                .call_method0(py, "name")
                .and_then(|name| name.extract::<String>(py))
                .unwrap_or_else(|_| "unknown".to_string())
        })
        .leak() // TODO: Store in adapter
    }
}
```

#### PyAnalyzerPlugin

```rust
/// Python base class for analyzer plugins
#[pyclass(subclass)]
pub struct PyAnalyzerPlugin;

#[pymethods]
impl PyAnalyzerPlugin {
    #[new]
    fn new() -> Self {
        PyAnalyzerPlugin
    }

    /// Get analyzer metadata (must be overridden)
    fn metadata(&self) -> PyResult<Py<PyDict>> {
        Err(PyNotImplementedError::new_err("Subclass must implement metadata()"))
    }

    /// Get dependencies (optional)
    fn dependencies(&self) -> PyResult<Vec<String>> {
        Ok(Vec::new())
    }

    /// Synchronous analysis (must be overridden)
    fn analyze(&self, _ctx: &PyAnalysisContext) -> PyResult<PyAnalysisResult> {
        Err(PyNotImplementedError::new_err("Subclass must implement analyze()"))
    }

    /// Asynchronous analysis (optional)
    fn analyze_async<'py>(
        &self,
        py: Python<'py>,
        ctx: &PyAnalysisContext,
    ) -> PyResult<&'py PyAny> {
        // Default: run sync version on thread pool
        let ctx = ctx.clone();
        let analyzer = self.clone();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            analyzer.analyze(&ctx)
        })
    }

    /// Whether this analyzer supports async
    fn supports_async(&self) -> bool {
        false
    }
}

/// Adapter that wraps Python analyzer to implement Rust trait
pub struct PyAnalyzerAdapter {
    py_object: Py<PyAny>,
    metadata_cache: OnceLock<AnalyzerMetadata>,
}

impl AnalyzerPlugin for PyAnalyzerAdapter {
    fn metadata(&self) -> AnalyzerMetadata {
        self.metadata_cache.get_or_init(|| {
            Python::with_gil(|py| {
                let dict = self.py_object
                    .call_method0(py, "metadata")
                    .and_then(|d| d.extract::<HashMap<String, Py<PyAny>>>(py))
                    .unwrap_or_default();

                AnalyzerMetadata {
                    id: dict.get("id")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or_else(|| "unknown".to_string()),
                    name: dict.get("name")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or_else(|| "Unknown".to_string()),
                    description: dict.get("description")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or_default(),
                    version: dict.get("version")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or_else(|| "0.1.0".to_string()),
                    author: dict.get("author")
                        .and_then(|v| v.extract(py).ok()),
                    cost: dict.get("cost")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or(0.5),
                    tags: dict.get("tags")
                        .and_then(|v| v.extract(py).ok())
                        .unwrap_or_default(),
                }
            })
        }).clone()
    }

    fn dependencies(&self) -> Vec<Dependency> {
        Python::with_gil(|py| {
            self.py_object
                .call_method0(py, "dependencies")
                .and_then(|deps| deps.extract::<Vec<String>>(py))
                .unwrap_or_default()
                .into_iter()
                .map(|s| Dependency::Analyzer(s))
                .collect()
        })
    }

    fn analyze<'a>(&self, ctx: &AnalysisContext<'a>) -> Result<AnalysisResult, AnalysisError> {
        Python::with_gil(|py| {
            let py_ctx = PyAnalysisContext::from_rust(ctx);

            let result = self.py_object
                .call_method1(py, "analyze", (py_ctx,))
                .map_err(|e| AnalysisError::Failed(
                    format!("Python analyzer failed: {}", e)
                ))?;

            let py_result: PyAnalysisResult = result.extract(py)
                .map_err(|e| AnalysisError::Failed(
                    format!("Invalid return type: {}", e)
                ))?;

            Ok(py_result.into_rust())
        })
    }
}
```

### Async Support (pyo3-asyncio)

```rust
use pyo3_asyncio::tokio::{future_into_py, get_runtime};

#[pymethods]
impl PyDiffOrchestrator {
    /// Async diff operation
    fn diff_async<'py>(
        &self,
        py: Python<'py>,
        original: String,
        modified: String,
    ) -> PyResult<&'py PyAny> {
        let orch = self.inner.clone();

        future_into_py(py, async move {
            let result = orch.diff_async(&original, &modified)
                .await
                .map_err(to_py_err)?;

            Ok(Python::with_gil(|py| {
                PyDiffResult { inner: Arc::new(result) }
                    .into_py(py)
            }))
        })
    }

    /// Async analysis
    fn analyze_async<'py>(
        &self,
        py: Python<'py>,
        diff: &PyDiffResult,
        analyzers: Vec<String>,
    ) -> PyResult<&'py PyAny> {
        let orch = self.inner.clone();
        let diff_ref = diff.inner.clone();

        future_into_py(py, async move {
            // Create context
            let ctx = AnalysisContext::new(
                diff_ref.clone(),
                Arc::new(PairwiseMetrics::default()),
            );

            let report = orch.analysis_coordinator()
                .analyze_async(&ctx, &analyzers)
                .await
                .map_err(to_py_err)?;

            Ok(Python::with_gil(|py| {
                PyAnalysisReport::from(report).into_py(py)
            }))
        })
    }
}
```

### Error Handling

```rust
use pyo3::create_exception;
use pyo3::exceptions::PyException;

// Create exception hierarchy
create_exception!(redline_core, RedlineError, PyException);
create_exception!(redline_core, TextProcessingError, RedlineError);
create_exception!(redline_core, NormalizationError, TextProcessingError);
create_exception!(redline_core, TokenizationError, TextProcessingError);
create_exception!(redline_core, DiffError, RedlineError);
create_exception!(redline_core, MetricError, RedlineError);
create_exception!(redline_core, AnalysisError, RedlineError);
create_exception!(redline_core, ConfigError, RedlineError);

/// Convert Rust errors to Python exceptions
pub fn to_py_err(err: crate::error::RedlineError) -> PyErr {
    match err {
        crate::error::RedlineError::TextProcessing(e) => {
            match e {
                crate::error::TextProcessingError::Normalization(msg) => {
                    NormalizationError::new_err(msg)
                }
                crate::error::TextProcessingError::Tokenization(msg) => {
                    TokenizationError::new_err(msg)
                }
                _ => TextProcessingError::new_err(e.to_string()),
            }
        }
        crate::error::RedlineError::DiffComputation(e) => {
            DiffError::new_err(e.to_string())
        }
        crate::error::RedlineError::Metrics(e) => {
            MetricError::new_err(e.to_string())
        }
        crate::error::RedlineError::Analysis(e) => {
            AnalysisError::new_err(e.to_string())
        }
        crate::error::RedlineError::Config(e) => {
            ConfigError::new_err(e.to_string())
        }
        _ => RedlineError::new_err(err.to_string()),
    }
}

/// Register exception types with Python module
pub fn register_exceptions(py: Python, module: &PyModule) -> PyResult<()> {
    module.add("RedlineError", py.get_type::<RedlineError>())?;
    module.add("TextProcessingError", py.get_type::<TextProcessingError>())?;
    module.add("NormalizationError", py.get_type::<NormalizationError>())?;
    module.add("TokenizationError", py.get_type::<TokenizationError>())?;
    module.add("DiffError", py.get_type::<DiffError>())?;
    module.add("MetricError", py.get_type::<MetricError>())?;
    module.add("AnalysisError", py.get_type::<AnalysisError>())?;
    module.add("ConfigError", py.get_type::<ConfigError>())?;
    Ok(())
}
```

### Python Module Definition

```rust
use pyo3::prelude::*;

/// Main Python module
#[pymodule]
fn redline_core(py: Python, m: &PyModule) -> PyResult<()> {
    // Register exception types
    register_exceptions(py, m)?;

    // Foundation types
    m.add_class::<PyTextStore>()?;
    m.add_class::<PyStringId>()?;
    m.add_class::<PySpan>()?;

    // Text processing
    m.add_class::<PyNormalizer>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyToken>()?;
    m.add_class::<PyProcessedText>()?;

    // Diff computation
    m.add_class::<PyDiffAlgorithm>()?;
    m.add_class::<PyDiffResult>()?;
    m.add_class::<PyEditOperation>()?;
    m.add_class::<PyDiffStatistics>()?;

    // Metrics
    m.add_class::<PyMetric>()?;
    m.add_class::<PyTextMetrics>()?;
    m.add_class::<PyPairwiseMetrics>()?;

    // Analysis
    m.add_class::<PyAnalyzerPlugin>()?;
    m.add_class::<PyAnalysisContext>()?;
    m.add_class::<PyAnalysisResult>()?;
    m.add_class::<PyAnalysisReport>()?;
    m.add_class::<PyInsight>()?;

    // Orchestration
    m.add_class::<PyDiffOrchestrator>()?;

    // Configuration
    m.add_class::<PyConfigBuilder>()?;
    m.add_class::<PyPreset>()?;

    // Convenience functions
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(diff_async, m)?)?;

    Ok(())
}

/// Convenience function: Simple diff
#[pyfunction]
fn diff(py: Python, original: String, modified: String) -> PyResult<PyDiffResult> {
    py.allow_threads(|| {
        let orch = DiffOrchestrator::new();
        orch.diff(&original, &modified)
            .map(|r| PyDiffResult { inner: Arc::new(r) })
            .map_err(to_py_err)
    })
}

/// Convenience function: Async diff
#[pyfunction]
fn diff_async<'py>(
    py: Python<'py>,
    original: String,
    modified: String,
) -> PyResult<&'py PyAny> {
    future_into_py(py, async move {
        let orch = DiffOrchestrator::new();
        let result = orch.diff_async(&original, &modified)
            .await
            .map_err(to_py_err)?;

        Ok(Python::with_gil(|py| {
            PyDiffResult { inner: Arc::new(result) }.into_py(py)
        }))
    })
}
```

### Python Type Stubs (__init__.pyi)

```python
"""
Type stubs for redline_core Python bindings.

This file provides type hints for IDE support and static analysis.
"""

from typing import List, Dict, Optional, Union, Protocol
from enum import Enum

# Exceptions
class RedlineError(Exception): ...
class TextProcessingError(RedlineError): ...
class NormalizationError(TextProcessingError): ...
class TokenizationError(TextProcessingError): ...
class DiffError(RedlineError): ...
class MetricError(RedlineError): ...
class AnalysisError(RedlineError): ...
class ConfigError(RedlineError): ...

# Foundation
class StringId:
    def __int__(self) -> int: ...

class Span:
    start: int
    end: int
    def __init__(self, start: int, end: int): ...
    def len(self) -> int: ...
    def is_empty(self) -> bool: ...
    def contains(self, pos: int) -> bool: ...

class Token:
    text: str
    span: Span
    index: int
    kind: str
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# Text Processing
class Normalizer(Protocol):
    def name(self) -> str: ...
    def normalize(self, text: str) -> NormalizationResult: ...
    def metadata(self) -> Dict[str, any]: ...
    def cost(self) -> float: ...

class Tokenizer(Protocol):
    def name(self) -> str: ...
    def tokenize(self, text: str) -> List[Token]: ...

class ProcessedText:
    original: str
    normalized: str
    tokens: List[Token]

# Diff
class EditKind(Enum):
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"
    REPLACE = "replace"

class EditOperation:
    kind: EditKind
    original_tokens: List[Token]
    modified_tokens: List[Token]

class DiffStatistics:
    equal_count: int
    delete_count: int
    insert_count: int
    replace_count: int
    edit_distance: int
    change_ratio: float

class DiffResult:
    operations: List[EditOperation]
    statistics: DiffStatistics
    def similarity(self) -> float: ...
    def is_empty(self) -> bool: ...
    def changes(self) -> List[EditOperation]: ...

# Analysis
class InsightLevel(Enum):
    INFO = "info"
    NOTE = "note"
    WARNING = "warning"
    CRITICAL = "critical"

class Insight:
    level: InsightLevel
    category: str
    message: str
    confidence: float

class AnalysisResult:
    analyzer_id: str
    metrics: Dict[str, float]
    categories: Dict[str, str]
    insights: List[Insight]
    confidence: float

class AnalysisContext:
    diff: DiffResult
    metrics: Dict[str, float]

class AnalyzerPlugin(Protocol):
    def metadata(self) -> Dict[str, any]: ...
    def dependencies(self) -> List[str]: ...
    def analyze(self, ctx: AnalysisContext) -> AnalysisResult: ...
    async def analyze_async(self, ctx: AnalysisContext) -> AnalysisResult: ...
    def supports_async(self) -> bool: ...

class AnalysisReport:
    results: List[AnalysisResult]
    errors: List[AnalysisError]
    insights: List[Insight]
    completed: bool

# Orchestration
class DiffOrchestrator:
    @staticmethod
    def new() -> "DiffOrchestrator": ...
    @staticmethod
    def builder() -> "ConfigBuilder": ...

    def diff(self, original: str, modified: str) -> DiffResult: ...
    async def diff_async(self, original: str, modified: str) -> DiffResult: ...

    def analyze(
        self,
        diff: DiffResult,
        analyzers: List[str]
    ) -> AnalysisReport: ...

    async def analyze_async(
        self,
        diff: DiffResult,
        analyzers: List[str]
    ) -> AnalysisReport: ...

    def register_normalizer(self, normalizer: Normalizer) -> None: ...
    def register_tokenizer(self, tokenizer: Tokenizer) -> None: ...
    def register_analyzer(self, analyzer: AnalyzerPlugin) -> None: ...

# Configuration
class Preset(Enum):
    FAST = "fast"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    COMPREHENSIVE = "comprehensive"

class ConfigBuilder:
    def preset(self, preset: Preset) -> "ConfigBuilder": ...
    def add_normalizer(self, normalizer: Normalizer) -> "ConfigBuilder": ...
    def add_tokenizer(self, tokenizer: Tokenizer) -> "ConfigBuilder": ...
    def add_analyzer(self, analyzer: AnalyzerPlugin) -> "ConfigBuilder": ...
    def build(self) -> DiffOrchestrator: ...

# Convenience functions
def diff(original: str, modified: str) -> DiffResult: ...
async def diff_async(original: str, modified: str) -> DiffResult: ...
```

### Automated Stub Generation

To ensure Python tooling (mypy, pyright, pylance) works correctly, we automatically generate `.pyi` files from the compiled extension.

#### Build Dependencies

```toml
[dev-dependencies]
# For generating .pyi stubs from compiled extension
pyo3-stub-gen = "0.5"
```

#### Stub Generation Process

```bash
# Build the extension first
maturin develop --release

# Generate stubs from compiled module
pyo3-stubgen redline_core -o ./python/

# Verify stubs
mypy --strict python/redline_core/__init__.pyi
```

#### Integration with maturin

```python
# build.py - Custom build script for maturin
import subprocess
import sys
from pathlib import Path

def generate_stubs():
    """Generate .pyi stub files after building extension"""

    # Build extension
    subprocess.run(
        ["maturin", "develop", "--release"],
        check=True
    )

    # Generate stubs
    subprocess.run(
        [
            sys.executable, "-m", "pyo3_stub_gen",
            "redline_core",
            "-o", "./python/",
            "--black",  # Format with black
        ],
        check=True
    )

    # Add py.typed marker
    py_typed = Path("python/redline_core/py.typed")
    py_typed.touch()

    print("✓ Generated type stubs at python/redline_core/")

if __name__ == "__main__":
    generate_stubs()
```

#### GitHub Actions CI for Stub Validation

```yaml
# .github/workflows/python-stubs.yml
name: Validate Python Stubs

on: [push, pull_request]

jobs:
  validate-stubs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Rust
        uses: actions-rust-lang/setup-rust-toolchain@v1

      - name: Install maturin
        run: pip install maturin pyo3-stub-gen

      - name: Build extension and generate stubs
        run: |
          maturin develop --release
          python -m pyo3_stub_gen redline_core -o ./python/

      - name: Install type checkers
        run: pip install mypy pyright

      - name: Validate stubs with mypy
        run: mypy --strict python/redline_core/

      - name: Validate stubs with pyright
        run: pyright python/redline_core/

      - name: Check stub completeness
        run: python scripts/check_stub_coverage.py
```

#### Stub Completeness Checker

```python
# scripts/check_stub_coverage.py
"""Ensure all exported types have stubs"""

import sys
from pathlib import Path
import importlib
import inspect

def check_stub_coverage():
    """Verify all public APIs have type stubs"""

    import redline_core

    # Get all public exports
    public_attrs = [
        name for name in dir(redline_core)
        if not name.startswith('_')
    ]

    # Read stub file
    stub_path = Path("python/redline_core/__init__.pyi")
    stub_content = stub_path.read_text()

    missing = []
    for attr in public_attrs:
        if attr not in stub_content:
            obj = getattr(redline_core, attr)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                missing.append(attr)

    if missing:
        print("❌ Missing stubs for:")
        for name in missing:
            print(f"  - {name}")
        sys.exit(1)

    print(f"✓ All {len(public_attrs)} public APIs have stubs")

if __name__ == "__main__":
    check_stub_coverage()
```

#### Package Distribution

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "redline-core"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Rust",
    "Typing :: Typed",  # Important: indicates type stubs included
]

[tool.maturin]
python-source = "python"
module-name = "redline_core"

# Include .pyi files in distribution
include = [
    "python/redline_core/*.pyi",
    "python/redline_core/py.typed",
]
```

#### Verification of Stub Installation

After installation, users can verify type stubs work:

```python
# test_stubs.py
"""Verify type stubs are properly installed"""

import redline_core
from redline_core import DiffOrchestrator, Token, DiffResult

# Type checker should validate these
orchestrator: DiffOrchestrator = DiffOrchestrator.new()
result: DiffResult = orchestrator.diff("hello", "world")

# This should raise type error with mypy/pyright
# result: int = orchestrator.diff("hello", "world")  # Type error!

print("✓ Type stubs working correctly")
```

```bash
# Run static type checking
mypy test_stubs.py
pyright test_stubs.py

# Both should pass with no errors
```

#### IDE Integration

With proper stubs, IDEs provide:

1. **Autocomplete**:
   ```python
   orch = DiffOrchestrator.new()
   orch.  # IDE suggests: diff, diff_async, analyze, register_normalizer, etc.
   ```

2. **Type Hints**:
   ```python
   def process_diff(result: DiffResult) -> float:
       return result.similarity()  # IDE knows return type is float
   ```

3. **Error Detection**:
   ```python
   orch.diff(123, 456)  # IDE error: Expected str, got int
   ```

4. **Documentation**:
   ```python
   DiffOrchestrator.  # IDE shows docstrings from stubs
   ```

### Stub Generation Best Practices

1. **Generate on every build**: Ensure stubs stay in sync with Rust code
2. **Validate in CI**: Catch stub issues before release
3. **Version in git**: Commit generated stubs so users without Rust can develop
4. **Document types**: Add comprehensive docstrings to PyO3 bindings
5. **Test with multiple checkers**: Support both mypy and pyright

### Example Generated Stub

pyo3-stub-gen will produce stubs like:

```python
# redline_core.pyi (auto-generated)
from typing import List, Dict, Optional, Protocol
import asyncio

class Token:
    """A token in the processed text."""

    @property
    def text(self) -> str: ...

    @property
    def span(self) -> Span: ...

    @property
    def index(self) -> int: ...

    @property
    def kind(self) -> str: ...

class DiffOrchestrator:
    """Main orchestrator for diff operations."""

    @staticmethod
    def new() -> DiffOrchestrator: ...

    @staticmethod
    def builder() -> ConfigBuilder: ...

    def diff(self, original: str, modified: str) -> DiffResult: ...

    def diff_async(
        self,
        original: str,
        modified: str
    ) -> asyncio.Future[DiffResult]: ...

    def register_normalizer(self, normalizer: Normalizer) -> None: ...

# ... etc
```

---

## Thread Safety and Memory Safety

### Thread Safety

All core types are designed with thread safety in mind:

#### Send + Sync Types

```rust
// Core types that are Send + Sync
impl Send for TextStore {}
impl Sync for TextStore {}  // With interior mutability via RwLock

impl Send for Token {}
impl Sync for Token {}  // Copy type, trivially Sync

impl<'a> Send for DiffResult<'a> {}
impl<'a> Sync for DiffResult<'a> {}

impl Send for AnalysisResult {}
impl Sync for AnalysisResult {}
```

#### Shared Ownership with Arc

```rust
// Thread-safe reference counting for shared data
pub struct ProcessedText {
    pub text_store: Arc<TextStore>,  // Shared across threads
    // ...
}

pub struct DiffResult<'a> {
    pub original: Arc<ProcessedText>,  // Clone Arc, not data
    pub modified: Arc<ProcessedText>,
}
```

#### Interior Mutability

For cache management and concurrent access:

```rust
pub struct CacheManager {
    metrics_cache: Arc<RwLock<LruCache<...>>>,  // Read-write lock
    stats: Arc<RwLock<CacheStatistics>>,
}

impl CacheManager {
    pub fn get_or_compute_metric<F, T>(&self, key: &MetricKey, compute: F) -> Arc<T>
    where
        F: FnOnce() -> T,
        T: Send + Sync + 'static,
    {
        // Try read lock first (fast path)
        if let Some(cached) = self.metrics_cache.read().unwrap().get(key) {
            return Arc::clone(cached);
        }

        // Write lock for insertion (slow path)
        let mut cache = self.metrics_cache.write().unwrap();
        // Double-check after acquiring write lock
        cache.entry(key.clone()).or_insert_with(|| Arc::new(compute()))
    }
}
```

### Memory Safety

#### Arena Allocation Safety

The `TextStore` uses arena allocation with careful lifetime management:

```rust
pub struct TextStore {
    arena: bumpalo::Bump,  // Bump allocator
    // SAFETY: Strings allocated in arena live as long as TextStore
}

impl TextStore {
    pub fn intern(&mut self, s: &str) -> StringId {
        // SAFETY: String is allocated in arena and won't move
        // StringId is only valid for this TextStore's lifetime
        let allocated: &str = self.arena.alloc_str(s);
        let id = self.next_id;
        self.next_id += 1;
        self.id_to_str.push(allocated);
        id
    }
}
```

**Safety Invariants:**
1. StringIds are only valid for their originating TextStore
2. TextStore owns all allocated strings
3. Tokens store StringId (Copy), not references
4. Text access requires passing &TextStore, enforcing lifetime

#### No Unsafe Code in Public API

The library aims for **zero unsafe code** in the public API surface:

- ✅ All public methods use safe Rust
- ✅ Arena allocation hidden behind safe abstraction
- ✅ String interning uses safe data structures
- ⚠️ Internal optimizations may use unsafe (with justification)

#### Memory Leaks Prevention

```rust
impl Drop for TextStore {
    fn drop(&mut self) {
        // Arena allocator automatically frees all memory
        // No manual cleanup needed for interned strings
    }
}
```

### Concurrency Patterns

#### Parallel Analysis

```rust
pub async fn analyze_async<'a>(
    &self,
    ctx: &AnalysisContext<'a>,
    plugin_ids: &[String],
) -> Result<AnalysisReport, AnalysisError> {
    use futures::future::join_all;

    let futures: Vec<_> = plugin_ids
        .iter()
        .filter_map(|id| self.registry.get(id))
        .map(|plugin| async move {
            if plugin.supports_async() {
                plugin.analyze_async(ctx).await
            } else {
                // Run sync analyzer on thread pool
                tokio::task::spawn_blocking(move || plugin.analyze(ctx)).await?
            }
        })
        .collect();

    let results = join_all(futures).await;
    // Aggregate results...
}
```

#### Lock-Free Reads

Where possible, use lock-free data structures:

```rust
// Immutable after construction - no locks needed
pub struct DiffResult<'a> {
    pub operations: Vec<EditOperation<'a>>,  // Immutable
    pub original: Arc<ProcessedText>,        // Immutable inner data
}

// Can be shared across threads without locking
let result = Arc::new(diff_result);
let result_clone = Arc::clone(&result);
tokio::spawn(async move {
    analyze(result_clone).await
});
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

## Required Dependencies

The following external crates are required for implementation:

### Core Dependencies

```toml
[dependencies]
# Error handling
thiserror = "1.0"

# Arena allocation for TextStore
bumpalo = "3.14"

# Fast hash maps
hashbrown = "0.14"

# Small vector optimization
smallvec = { version = "1.11", features = ["union", "const_generics"] }

# String interning
string-interner = "0.15"

# Async runtime (optional, for async features)
tokio = { version = "1.35", optional = true, features = ["rt", "sync", "time", "macros"] }
futures = { version = "0.3", optional = true }

# Serialization (optional)
serde = { version = "1.0", optional = true, features = ["derive"] }
serde_json = { version = "1.0", optional = true }

# Caching
lru = "0.12"

# Metrics and observability (optional)
tracing = { version = "0.1", optional = true }

# Python bindings (optional)
pyo3 = { version = "0.20", optional = true, features = ["extension-module", "abi3", "multiple-pymethods"] }
pyo3-asyncio = { version = "0.20", optional = true, features = ["tokio-runtime"] }
pythonize = { version = "0.20", optional = true }  # Serde ↔ Python conversion
```

### Development Dependencies

```toml
[dev-dependencies]
# Testing
proptest = "1.4"  # Property-based testing
criterion = "0.5"  # Benchmarking
test-case = "3.3"  # Parameterized tests

# Async testing
tokio-test = "0.4"

# Test data generation
fake = "2.9"
rand = "0.8"

# Python stub generation
pyo3-stub-gen = "0.5"  # Generate .pyi files from PyO3 bindings
```

### Feature Flags

```toml
[features]
default = ["std"]
std = []

# Async support
async = ["tokio", "futures"]

# Serialization
serde = ["dep:serde", "serde_json"]

# Observability
tracing = ["dep:tracing"]

# Python bindings
python = ["pyo3", "pyo3-asyncio", "pythonize", "async"]

# ML-based analyzers (separate crate)
ml = ["redline-ml"]

# Full feature set
full = ["async", "serde", "tracing", "python"]
```

### Python Package Build Dependencies

```toml
[build-dependencies]
# For building Python wheels (when python feature enabled)
maturin = { version = "1.4", optional = true }

[package.metadata.maturin]
# Python package metadata
name = "redline-core"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
```

### Import Conventions

Recommended imports at module level:

```rust
// Foundation
use crate::foundation::{TextStore, StringId, Span, CharMapping};

// Text processing
use crate::text::{Normalizer, Tokenizer, Token, ProcessedText};

// Diff computation
use crate::diff::{DiffComputer, DiffResult, EditOperation, EditKind};

// Metrics
use crate::metrics::{MetricsEngine, TextMetrics, PairwiseMetrics};

// Analysis
use crate::analysis::{
    AnalyzerPlugin, AnalysisContext, AnalysisResult,
    AnalysisCoordinator, AnalysisReport,
};

// Caching
use crate::cache::{CacheManager, CachePolicy};

// Orchestration
use crate::orchestration::DiffOrchestrator;

// Configuration
use crate::config::{ConfigBuilder, Preset, OrchestratorConfig};

// Errors
use crate::error::{
    RedlineError, TextProcessingError, DiffError,
    MetricError, AnalysisError, ConfigError,
};

// External
use std::sync::Arc;
use smallvec::SmallVec;
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4) - 1 month
**Goal**: Core data structures and memory management

- ✅ Project setup, module structure, CI/CD
- ✅ TextStore with string interning
- ✅ Span and position types with full API
- ✅ CharMapping with composition
- ✅ Basic error types with thiserror
- ✅ Initial benchmarks and profiling setup
- ✅ Foundation unit tests

**Deliverable**: Foundation layer with <50ns string lookup, comprehensive tests

### Phase 2: Text Processing (Weeks 5-8) - 1 month
**Goal**: Complete normalization and tokenization pipeline

- Normalizer trait and 5-6 built-in normalizers
- Tokenizer trait and 3-4 built-in tokenizers
- TextProcessor orchestrator
- Pipeline with traceability layers
- TextProcessing integration tests
- Documentation and examples

**Deliverable**: Working text processing pipeline with character mapping

### Phase 3: Diff Computation (Weeks 9-13) - 5 weeks
**Goal**: Multiple diff algorithms with benchmarks

- DiffAlgorithm trait
- Myers algorithm (O(ND))
- Histogram algorithm
- Patience algorithm (optional)
- DiffComputer with algorithm selection
- EditOperation types and utilities
- Diff algorithm benchmarks
- Property-based diff tests

**Deliverable**: Fast, correct diff computation with <1ms for 100 tokens

### Phase 4: Metrics Engine (Weeks 14-18) - 5 weeks
**Goal**: Comprehensive metrics with caching

- Metric trait and registry
- TextMetrics (20+ metrics)
- PairwiseMetrics (15+ metrics)
- MetricsEngine with lazy evaluation
- Content-addressed caching
- Metrics computation tests
- Performance benchmarks

**Deliverable**: Rich metrics suite with sub-millisecond cached lookups

### Phase 5: Analysis Framework (Weeks 19-24) - 6 weeks
**Goal**: Extensible analyzer plugin system

- AnalyzerPlugin trait with metadata
- Dependency resolution system
- AnalysisCoordinator with execution planning
- 3-4 built-in analyzers (semantic, stylistic, readability)
- Plugin registry
- Async analyzer support
- Analyzer integration tests

**Deliverable**: Working plugin system with parallel execution

### Phase 6: Orchestration & Configuration (Weeks 25-28) - 4 weeks
**Goal**: High-level API and configuration

- DiffOrchestrator
- Configuration builder with validation
- Preset configurations
- CacheManager with RwLock
- High-level convenience APIs
- Configuration examples

**Deliverable**: Complete, easy-to-use public API

### Phase 7: Async & Optimization (Weeks 29-32) - 4 weeks
**Goal**: Async support and performance tuning

- Async diff and analysis APIs
- Tokio integration for CPU-bound work
- Parallel analyzer execution
- Cache optimization
- Memory profiling and optimization
- SIMD exploration (if beneficial)
- Performance regression tests

**Deliverable**: Async support with 2x speedup target for parallel analysis

### Phase 7.5: Python Bindings (Weeks 33-37) - 5 weeks
**Goal**: Complete Python interface via PyO3

- PyO3 module setup and build system (maturin + pyproject.toml)
- Core type bindings (30+ PyClasses)
- Python plugin adapters for all extensible components:
  - PyNormalizer, PyTokenizer
  - PyDiffAlgorithm
  - PyAnalyzerPlugin
  - PyMetric (custom metrics)
- Async bridge (tokio ↔ asyncio with pyo3-asyncio)
- Error handling (Rust errors → Python exceptions)
- **Automated .pyi stub generation**:
  - Configure pyo3-stub-gen
  - Build script for automatic stub generation
  - CI validation with mypy and pyright
  - Stub completeness checking
  - Package py.typed marker
- Python test suite (pytest + pytest-asyncio)
- Python examples and documentation
- Python package publishing setup (PyPI with stubs)

**Deliverable**: Full-featured Python package with complete type hints and plugin extensibility

### Phase 8: Testing & Polish (Weeks 38-43) - 6 weeks
**Goal**: Production-ready quality

- Comprehensive unit test coverage (>80%)
- Integration test suite
- Property-based tests with proptest
- Fuzz testing
- Error handling edge cases
- API documentation
- User guide and examples
- Migration guide from v0
- Benchmark suite

**Deliverable**: Well-tested, documented v1.0.0 ready for release

### Phase 9: Beta & Feedback (Weeks 39-42) - 4 weeks
**Goal**: Real-world validation

- Internal dogfooding
- Beta release
- Gather feedback
- Bug fixes and polish
- Performance tuning based on real usage
- Documentation improvements

**Deliverable**: v1.0.0-rc1

### Phase 10: Release (Week 43) - 1 week
**Goal**: Official v1.0.0 release

- Final review
- Release notes
- Version tagging
- Crate publication
- Announcement

**Total Timeline**: ~11 months (48 weeks) with Python bindings

### Timeline Flexibility

- **Fast track** (aggressive): 8-9 months with dedicated full-time work
- **Realistic** (recommended): 11 months with part-time work or interruptions
- **Conservative**: 13 months accounting for unforeseen challenges
- **Rust-only** (no Python): 10 months (skip Phase 7.5)

### Key Milestones

| Month | Milestone | Deliverable |
|-------|-----------|-------------|
| 1 | Foundation complete | Core data structures |
| 2 | Text processing works | Normalization + tokenization |
| 3-4 | Diff computation ready | Multiple algorithms benchmarked |
| 5 | Metrics engine functional | Rich metrics with caching |
| 6-7 | Analysis plugins working | Extensible analyzer system |
| 8 | Public API finalized | Configuration + orchestration |
| 9 | Async + optimizations | Performance targets met |
| 9-10 | **Python bindings** | **Full Python API & plugins** |
| 11 | Beta release | Production-ready quality (Rust + Python) |

### Python Integration Benefits

- **5 additional weeks** for Python bindings (Phase 7.5)
- Enables ML models via Python (HuggingFace, spaCy, etc.)
- Opens library to Python ecosystem (largest ML/NLP community)
- Users can prototype analyzers in Python, optimize in Rust later
- Wider adoption potential

---

**Document Status**: In Review - Ready for Implementation
**Last Updated**: 2025-10-23
**Next Review**: After Phase 1 completion (Month 1)

## Document Change Log

### 2025-10-23 - Major Revision
- ✅ Fixed lifetime inconsistencies (ID-based tokens with sliced operations)
- ✅ Added comprehensive thread safety and memory safety sections
- ✅ Enhanced async design with cancellation, timeout, and error recovery
- ✅ Added detailed configuration validation with dependency checking
- ✅ Added required dependencies and import conventions
- ✅ Revised implementation timeline to realistic 10-month schedule
- ✅ Added structured error handling with context propagation
- ✅ Clarified memory management and arena allocation safety
- ✅ Enhanced AnalysisReport with error tracking and partial results
- ✅ Added concrete validation rules and examples
- ✅ **Added comprehensive Python bindings via PyO3**
  - Complete Python API exposure (30+ classes)
  - Python plugin system for all extensible components
  - Async Python support (asyncio ↔ tokio bridge)
  - Python exception hierarchy
  - **Automated .pyi stub generation system**:
    - pyo3-stub-gen integration
    - Build automation scripts
    - CI validation (mypy + pyright)
    - Stub completeness checking
    - py.typed marker for PEP 561 compliance
  - maturin build configuration
  - Extended timeline by 5 weeks for Python integration
