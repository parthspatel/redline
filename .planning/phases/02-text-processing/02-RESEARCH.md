# Phase 2: Text Processing - Research

**Researched:** 2026-02-06
**Domain:** Text normalization, tokenization, Unicode processing, pipeline orchestration, TDD in Rust
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Aggressiveness is Claude's discretion — no constraint on how aggressive each normalizer is
- The hard contract: every normalizer MUST produce a valid CharMapping that round-trips positions correctly
- Traceability is the invariant, not preservation of original form
- WordTokenizer splits on whitespace only — no punctuation splitting, no contraction handling
- Two NGram tokenizer variants: WordNGramTokenizer (n-grams over words), CharNGramTokenizer (n-grams over characters)
- NGram tokenizers are composable: configured with a base tokenizer that runs first, then n-grams are computed over its output
- This means NGram is not a standalone tokenizer — it wraps another
- All undefined/unexpected behaviors are FATAL
- Every failure is a typed error — no silent fallbacks, no skip-and-continue
- If a normalizer produces empty output, invalid mapping, or any unexpected state: return a typed error
- Caller is responsible for handling errors explicitly (this is Rust)
- ProcessedText MUST store intermediate normalization layers, not just final output
- Each layer stores: the text after that normalization step + CharMapping to previous layer
- Different analyzers and diff algorithms operate on specific layers
- Memory optimization: only store layers that are actually needed by configured analyzers — unused normalizer outputs are not retained
- This is the default "minimal" mode; user can explicitly request "all" layers
- Analyzers declare at configuration time (before execution) the minimally required processing steps
- Dependencies are declared by named layers (e.g., "lowercase", "whitespace") — not numeric indices
- The system resolves names to layer indices at configuration time
- Referencing a non-existent layer name is a fatal typed error at config time
- System runs the minimal union of all required normalization steps by default
- Steps that will NOT run are logged (not silently skipped)
- User can explicitly control "minimal" (default) or "all" execution mode
- Diff algorithms operate based on the dependency graph
- The diff layer is determined by what the requesting analyzer/consumer declared as dependencies

### Claude's Discretion
- How aggressive each built-in normalizer is (Lowercase, Whitespace, Unicode, Diacritics, Punctuation, Digits)
- Unicode normalization form choice (NFC vs NFD vs NFKC)
- How NGram tokenizer composes its token stream internally (new StringIds for n-grams vs index references)
- Exact layer storage format and composed CharMapping caching strategy
- CJK/emoji/RTL handling details within tokenizers

### Deferred Ideas (OUT OF SCOPE)
- Execution graph with full DAG resolution is partially a Phase 5 (Analysis Framework) concern — the dependency declaration mechanism is defined here, but the full ExecutionPlanner with topological sort lives in Phase 5
- Parallel normalizer execution for independent normalization chains — Phase 7 (Async)
</user_constraints>

## Summary

Phase 2 builds the text processing pipeline on top of Phase 1's foundation types (TextStore, Token, Span, CharMapping). The core task is implementing two trait hierarchies (Normalizer and Tokenizer) with 6+4 built-in implementations, an orchestrator (TextProcessor), and an output type (ProcessedText) that preserves intermediate normalization layers with full CharMapping traceability.

The key technical challenges are: (1) building correct CharMappings for normalizations that change character count (Unicode NFC/NFD, diacritics removal, whitespace collapsing), (2) composing CharMappings through multi-step pipelines, (3) grapheme-aware tokenization for CJK/emoji/RTL text, and (4) the named-layer dependency system with minimal execution mode. The standard approach uses `unicode-normalization` 0.1.25 for NFC/NFD/NFKC/NFKD operations and `unicode-segmentation` 1.12 for grapheme-cluster-aware tokenization.

The TDD approach requires writing tests FIRST for each trait, normalizer, tokenizer, and pipeline component. Property-based tests via `proptest` validate CharMapping round-trip correctness. Snapshot tests via `insta` capture complex ProcessedText output. Parameterized tests via `test-case` cover the 6 normalizer x multiple-input matrix efficiently.

**Primary recommendation:** Implement in strict TDD order: trait definitions (test trait contracts) -> normalizers one-by-one (test CharMapping round-trip for each) -> tokenizers (test boundary detection) -> TextProcessor orchestrator (test pipeline composition) -> ProcessedText with named layers (test layer addressing and minimal execution). Every normalizer MUST have a proptest that verifies CharMapping round-trip for arbitrary Unicode input.

## Standard Stack

### Core Dependencies (to add to Cargo.toml)

| Library | Version | Purpose | Why Standard | Confidence |
|---------|---------|---------|--------------|------------|
| unicode-normalization | 0.1.25 | NFC/NFD/NFKC/NFKD normalization | The Rust ecosystem standard for UAX#15. no_std compatible. Used by rustc itself. | HIGH |
| unicode-segmentation | 1.12.0 | Grapheme cluster, word, and sentence boundary detection | UAX#29 reference implementation for Rust. no_std compatible. Provides `grapheme_indices()`, `unicode_word_indices()`, `split_sentence_bound_indices()` with byte offsets. | HIGH |

### Dev Dependencies (to add to Cargo.toml)

| Library | Version | Purpose | Why Standard | Confidence |
|---------|---------|---------|--------------|------------|
| proptest | 1.6+ (already in Cargo.toml) | Property-based testing for CharMapping round-trips | Already used in Phase 1. Essential for correctness. | HIGH |
| test-case | 3.3 | Parameterized tests for normalizer/tokenizer variants | Clean `#[test_case]` attribute macro. 29M+ downloads. | HIGH |
| insta | 1.46 | Snapshot testing for ProcessedText output | De facto Rust snapshot testing. Inline snapshot support. Diff via `similar` crate. | HIGH |
| static_assertions | 1.1 (already in Cargo.toml) | Compile-time trait verification | Already used in Phase 1. | HIGH |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| unicode-normalization | icu_normalizer (ICU4X) | ICU4X is more comprehensive but much heavier dependency. unicode-normalization is lighter, sufficient for our needs. |
| unicode-normalization | unicode-normalization-alignments | Fork of unicode-normalization by HuggingFace team that returns alignment diff values alongside normalized chars. Version 0.1.12, last updated 2019. Too stale — build our own alignment tracking on top of standard crate instead. |
| unicode-segmentation | Manual regex-based splitting | UAX#29 has hundreds of edge cases. Hand-rolling is guaranteed incorrect for CJK/emoji/RTL. |
| insta | Manual assert_eq! on Debug output | Snapshot testing is much more maintainable for complex nested types like ProcessedText. |

### Installation (additions to crates/core/Cargo.toml)

```toml
[dependencies]
# ... existing deps ...
unicode-normalization = "0.1"
unicode-segmentation = "1.12"

[dev-dependencies]
# ... existing dev-deps ...
test-case = "3.3"
insta = { version = "1.46", features = ["yaml"] }
```

## Architecture Patterns

### Recommended Project Structure

```
crates/core/src/
├── lib.rs                    # Add pub mod normalize; pub mod tokenize; pub mod process;
├── normalize/
│   ├── mod.rs                # Normalizer trait + NormalizationResult + re-exports
│   ├── lowercase.rs          # Lowercase normalizer
│   ├── whitespace.rs         # WhitespaceNormalizer (collapse, trim, newline handling)
│   ├── unicode.rs            # UnicodeNormalizer (NFC/NFD/NFKC/NFKD)
│   ├── diacritics.rs         # RemoveDiacritics normalizer
│   ├── punctuation.rs        # RemovePunctuation normalizer
│   └── digits.rs             # RemoveDigits normalizer
├── tokenize/
│   ├── mod.rs                # Tokenizer trait + re-exports
│   ├── char.rs               # CharTokenizer
│   ├── word.rs               # WordTokenizer (whitespace-only splits)
│   ├── sentence.rs           # SentenceTokenizer (via unicode-segmentation)
│   ├── word_ngram.rs         # WordNGramTokenizer (wraps base tokenizer)
│   └── char_ngram.rs         # CharNGramTokenizer (wraps base tokenizer)
├── process/
│   ├── mod.rs                # TextProcessor orchestrator + re-exports
│   ├── processed_text.rs     # ProcessedText type with named layers
│   └── layer.rs              # NormalizationLayer, LayerName, ExecutionMode
├── char_mapping.rs           # (existing, may need extensions)
├── error.rs                  # (existing, needs new error variants)
├── span.rs                   # (existing)
├── text_store.rs             # (existing)
└── token.rs                  # (existing)
```

### Pattern 1: Normalizer Trait with NormalizationResult

**What:** Each normalizer takes `&str` input and returns `Result<NormalizationResult, NormalizeError>` containing the normalized text and a CharMapping from input positions to output positions. The CharMapping uses byte offsets (matching Span's byte-offset convention from Phase 1).

**When to use:** Every text normalization operation.

**Design (Claude's Discretion recommendation):**

```rust
/// Result of a normalization operation.
pub struct NormalizationResult {
    /// The normalized text.
    pub text: String,
    /// Mapping from original byte positions to normalized byte positions.
    pub mapping: CharMapping,
}

/// Trait for text normalizers. All normalizers are Send + Sync for thread safety.
pub trait Normalizer: Send + Sync {
    /// Apply normalization to input text.
    /// 
    /// # Errors
    /// Returns NormalizeError if normalization fails or produces invalid state.
    /// Empty output from non-empty input is a valid normalization (e.g., RemoveDigits on "123").
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError>;

    /// Human-readable name used as the layer identifier.
    /// This name is used for named-layer dependency resolution.
    fn name(&self) -> &str;

    /// Estimated computational cost (0.0 = trivial, 1.0 = expensive).
    /// Used for execution planning in later phases.
    fn cost(&self) -> f32 {
        0.5
    }
}
```

**Key constraint from CONTEXT.md:** Every normalizer MUST produce a valid CharMapping that round-trips positions correctly. This is the invariant.

### Pattern 2: CharMapping for Variable-Length Transformations

**What:** The existing CharMapping stores `Vec<(u32, u32)>` alignment pairs as (original, normalized) positions. For normalizations that change text length (Unicode NFC composes 2 chars into 1, diacritics removal deletes chars, whitespace collapsing merges chars), the CharMapping must handle N:1, 1:N, and 1:0 mappings.

**Critical insight:** The current CharMapping from Phase 1 stores position-to-position pairs. For normalizations that DELETE characters (RemovePunctuation, RemoveDiacritics, RemoveDigits), positions in the original that map to nothing need special handling. For normalizations that MERGE characters (whitespace collapsing), multiple original positions map to one normalized position.

**Current CharMapping limitation:** The Phase 1 implementation uses binary search on original positions and linear scan on normalized positions. It assumes bijective (1:1) mapping. For Phase 2, we need:
- **Many-to-one** (whitespace collapse: 3 spaces -> 1 space): multiple original positions map to same normalized position. The current `to_original(normalized)` linear scan returns the FIRST match, which may need to return a range.
- **One-to-zero** (character removal): original positions that have no normalized counterpart. These should NOT be in the alignment vec — they simply won't have a mapping entry.
- **One-to-many** (Unicode NFD decomposition: e-acute -> e + combining-accent): one original position maps to multiple normalized positions. Need multiple entries with same original.

**Recommendation:** Extend CharMapping or build alignment on top of it:
- For the "round-trip" contract, the important direction is normalized->original (given a position in processed text, find where it was in original). This direction is critical for the UI to highlight original text.
- The existing `to_original()` and `to_normalized()` are sufficient for 1:1 mappings. For many-to-one, we can store the first original position (the start of the collapsed range). For one-to-many, we store multiple entries with the same original position.
- The composition via `compose()` already handles these cases correctly since it operates on (orig, mid) pairs.

### Pattern 3: Tokenizer Trait with TextStoreBuilder

**What:** Tokenizers take normalized text and a `&mut TextStoreBuilder` to intern token strings, producing `Vec<Token>`. The tokenizer does NOT own the TextStoreBuilder — it receives a mutable reference.

**Design:**

```rust
/// Trait for tokenizers. All tokenizers are Send + Sync for thread safety.
pub trait Tokenizer: Send + Sync {
    /// Tokenize text, interning token strings into the store.
    ///
    /// The `mapping` parameter is the composed CharMapping from all prior
    /// normalizations, allowing tokens to be mapped back to original positions.
    ///
    /// # Errors
    /// Returns TokenizeError on failure.
    fn tokenize(
        &self,
        text: &str,
        mapping: &CharMapping,
        store: &mut TextStoreBuilder,
    ) -> Result<Vec<Token>, TokenizeError>;

    /// Human-readable name of this tokenizer.
    fn name(&self) -> &str;
}
```

**Key decision from CONTEXT.md:** WordTokenizer splits on whitespace only. NGram tokenizers wrap a base tokenizer.

### Pattern 4: NGram Tokenizer Composition

**What:** NGram tokenizers are NOT standalone — they wrap another tokenizer and compute n-grams over its output.

**Design:**

```rust
/// Word-level n-gram tokenizer. Wraps a base tokenizer, runs it first,
/// then computes n-grams over the resulting word tokens.
pub struct WordNGramTokenizer {
    base: Box<dyn Tokenizer>,
    n: usize,
}

/// Character-level n-gram tokenizer. Computes n-grams over characters.
pub struct CharNGramTokenizer {
    n: usize,
}
```

**For WordNGramTokenizer:** Run base tokenizer first, get `Vec<Token>`. Then slide a window of size `n` over those tokens. For each window, create a new token by concatenating the window's text (separated by space), interning into TextStoreBuilder, and using a Span that covers the combined range.

**For CharNGramTokenizer:** Iterate over characters (grapheme clusters for Unicode correctness). For each window of `n` grapheme clusters, intern the substring and create a token.

**Claude's Discretion recommendation:** Create NEW StringIds for n-gram tokens (the concatenated text). This is simpler and keeps n-gram tokens self-describing. The alternative (storing index references to base tokens) adds complexity without clear benefit since StringId deduplication handles repeated n-grams.

### Pattern 5: ProcessedText with Named Layers

**What:** ProcessedText stores the pipeline result with intermediate layers addressable by name.

**Design:**

```rust
/// Execution mode for the normalization pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Only run normalizers required by configured analyzers (default).
    Minimal,
    /// Run all configured normalizers, storing all layers.
    All,
}

/// A named normalization layer in the pipeline.
#[derive(Debug, Clone)]
pub struct NormalizationLayer {
    /// Name of the normalizer that produced this layer.
    pub name: String,
    /// Text after this normalization step.
    pub text: String,
    /// CharMapping from the PREVIOUS layer to this layer.
    pub mapping: CharMapping,
}

/// Result of text processing through the full pipeline.
pub struct ProcessedText {
    /// Original input text.
    pub original: String,
    /// Normalization layers (ordered by pipeline execution).
    /// In Minimal mode, only layers required by analyzers are stored.
    pub layers: Vec<NormalizationLayer>,
    /// Final normalized text (the last layer's output, or original if no normalizers).
    pub normalized: String,
    /// Tokens produced from the final normalized text.
    pub tokens: Vec<Token>,
    /// Composed CharMapping: original -> final normalized.
    pub composed_mapping: CharMapping,
    /// Text store containing all interned strings.
    pub text_store: std::sync::Arc<TextStore>,
    /// Names of normalizers that were configured but NOT run (Minimal mode).
    pub skipped_normalizers: Vec<String>,
}

impl ProcessedText {
    /// Look up a layer by normalizer name.
    pub fn layer(&self, name: &str) -> Option<&NormalizationLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Get the composed CharMapping from original to a specific named layer.
    pub fn mapping_to_layer(&self, name: &str) -> Result<CharMapping, NormalizeError> {
        // Compose mappings from layer 0 to the named layer
    }
}
```

### Pattern 6: TDD Red-Green-Refactor for Trait Systems

**What:** Test-driven development pattern for implementing trait-based normalizer/tokenizer systems in Rust.

**TDD cycle for each normalizer:**
1. **RED:** Write tests that define the normalizer's contract: input/output pairs, CharMapping correctness, edge cases (empty string, Unicode, emoji). Tests fail because normalizer doesn't exist.
2. **GREEN:** Implement the minimal normalizer to pass all tests.
3. **REFACTOR:** Clean up implementation, extract common patterns.
4. **PROPERTY:** Add proptest for CharMapping round-trip with arbitrary Unicode input.

**TDD cycle for each tokenizer:**
1. **RED:** Write tests defining tokenization boundaries: what characters split tokens, how spans are computed, edge cases.
2. **GREEN:** Implement tokenizer.
3. **REFACTOR:** Extract common token creation helpers.
4. **PROPERTY:** Add proptest verifying token spans cover full input (no gaps, no overlaps for non-overlapping tokenizers).

### Anti-Patterns to Avoid

- **Anti-pattern: Building CharMapping after normalization by diffing strings.** Instead, build CharMapping DURING normalization by tracking each character transformation as it happens. The normalizer knows exactly what it changed — use that knowledge.
- **Anti-pattern: Using char indices instead of byte offsets in CharMapping.** Span uses byte offsets (u32). CharMapping MUST also use byte offsets. `str::char_indices()` provides byte offsets.
- **Anti-pattern: Ignoring grapheme cluster boundaries in tokenizers.** A grapheme cluster like flag emoji (two regional indicators) or family emoji (person + ZWJ + person) is ONE visual unit. CharTokenizer should emit one token per grapheme cluster, not per char/codepoint.
- **Anti-pattern: Storing full text copies in every NormalizationLayer unconditionally.** The CONTEXT.md mandates minimal mode by default — only store layers needed by analyzers. Implement an `ExecutionMode` enum.
- **Anti-pattern: Using `to_lowercase()` directly without tracking byte offsets.** Lowercase can change byte length (e.g., German sharp-s `ß` -> `SS` in uppercase has different byte lengths). Build CharMapping char-by-char.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Unicode NFC/NFD/NFKC/NFKD | Custom normalization tables | `unicode-normalization` 0.1.25 | UAX#15 has hundreds of rules and edge cases. The crate handles all Unicode versions correctly. |
| Grapheme cluster detection | Regex-based character splitting | `unicode-segmentation` 1.12 (`.graphemes(true)`, `.grapheme_indices(true)`) | UAX#29 Extended Grapheme Clusters handle emoji ZWJ sequences, combining marks, regional indicators. Impossible to get right manually. |
| Sentence boundary detection | Split on `.!?` | `unicode-segmentation` 1.12 (`.unicode_sentences()`, `.split_sentence_bound_indices()`) | Abbreviations (Dr., U.S.), quotation marks, decimal numbers (3.14) make naive splitting incorrect. |
| Word boundary detection (for SentenceTokenizer internals) | Split on whitespace+punctuation | `unicode-segmentation` 1.12 (`.unicode_word_indices()`) | CJK text has NO whitespace between words. UAX#29 handles this. |
| Diacritics identification | Manual Unicode table lookup | NFD decomposition + `char::is_combining_mark()` or Unicode category check | The standard pattern: NFD decomposes accented chars, then filter `\p{Mn}` (Mark, Nonspacing) category. |
| Property-based testing | Manual random input generation | `proptest` 1.6+ | Shrinking, regression files, reproducibility. Already used in Phase 1. |
| Snapshot testing | Manual Debug string comparison | `insta` 1.46 with inline snapshots | Review workflow, human-readable diffs, cargo-insta tooling. |
| Parameterized tests | Copy-paste test functions | `test-case` 3.3 `#[test_case]` attribute | Clean matrix of inputs. Each test case gets its own name in test output. |

## Common Pitfalls

### Pitfall 1: CharMapping Byte Offset vs Character Index Confusion

**What goes wrong:** CharMapping stores (u32, u32) pairs. If you use character indices (number of chars before this one) instead of byte offsets, the mapping won't compose correctly with Span (which uses byte offsets) or with other normalizers.

**Why it happens:** `str::chars().enumerate()` gives character indices. `str::char_indices()` gives byte offsets. Easy to grab the wrong one.

**How to avoid:** ALWAYS use `str::char_indices()` or explicit byte offset tracking. Never use `chars().enumerate()` for position tracking. Add a test that verifies CharMapping positions align with `&text[start..end]` slice operations.

**Warning signs:** Tests pass for ASCII but fail for multi-byte UTF-8 (CJK, emoji).

### Pitfall 2: Lowercase/Uppercase Can Change Byte Length

**What goes wrong:** `'ß'.to_uppercase()` produces "SS" (2 chars, 2 bytes) from 1 char (2 bytes). Turkish `'İ'` (U+0130) lowercases to `'i'` + combining dot (2 chars). German `'ẞ'` (capital sharp S) lowercases to `'ß'`. These change byte lengths.

**Why it happens:** Case mapping is not 1:1 in Unicode. The simple `str::to_lowercase()` handles this correctly for the text, but the CharMapping must account for the length change.

**How to avoid:** Build CharMapping character by character:
```rust
let mut alignments = Vec::new();
let mut norm_offset = 0u32;
for (orig_offset, ch) in input.char_indices() {
    let lower = ch.to_lowercase();
    for lower_ch in lower {
        alignments.push((orig_offset as u32, norm_offset));
        norm_offset += lower_ch.len_utf8() as u32;
    }
}
```

**Warning signs:** German text with ß, Turkish text with İ/ı, tests that only use ASCII.

### Pitfall 3: Unicode Normalization Changes Character Count

**What goes wrong:** NFC composition: `'e'` + `'\u{0301}'` (combining acute) -> `'é'` (one codepoint). This is a 2:1 mapping. NFD decomposition: `'é'` -> `'e'` + `'\u{0301}'` (1:2 mapping). NFKC/NFKD also handle compatibility equivalences (e.g., `'ﬁ'` ligature -> `'f'` + `'i'`).

**Why it happens:** Unicode normalization is fundamentally about character count changes.

**How to avoid:** Build CharMapping by iterating original and normalized in parallel, tracking byte offsets of each character. The `unicode-normalization` crate's `.nfc()` etc. return iterators of `char` — you need to track which original chars produced which normalized chars.

**Recommended approach:** Decompose to NFD first (to get a canonical decomposed form), then for NFC, compose and track which decomposed chars merged. Alternatively, compare original and NFD char-by-char with `char_indices()`.

**Warning signs:** Tests pass for Latin text but fail for accented characters, CJK with compatibility forms, or mathematical symbols.

### Pitfall 4: Whitespace Collapse Many-to-One Mapping

**What goes wrong:** Collapsing "a   b" (3 spaces) to "a b" (1 space) means original positions 1, 2, 3 all map to normalized position 1. The CharMapping must handle this, and `to_original(1)` should return the first original position in the collapsed range.

**Why it happens:** Whitespace collapse is a many-to-one operation.

**How to avoid:** When collapsing whitespace, only add a mapping for the first whitespace character in a run. Skip the rest. This means `to_original(norm_pos_of_space)` returns the first original space.

```rust
let mut in_whitespace = false;
for (orig_offset, ch) in input.char_indices() {
    if ch.is_whitespace() {
        if !in_whitespace {
            // First whitespace in run: map to single space
            alignments.push((orig_offset as u32, norm_offset));
            norm_offset += 1; // single space byte
            in_whitespace = true;
        }
        // Subsequent whitespace: no mapping entry (consumed)
    } else {
        in_whitespace = false;
        alignments.push((orig_offset as u32, norm_offset));
        norm_offset += ch.len_utf8() as u32;
    }
}
```

### Pitfall 5: Empty String Input Edge Case

**What goes wrong:** A normalizer receives empty string `""`. CharMapping::new(vec![]) returns Err (empty alignments). But empty-to-empty is a valid normalization.

**Why it happens:** Phase 1's CharMapping requires non-empty alignments vector.

**How to avoid:** Two options:
1. **Special-case empty input** in each normalizer: return a sentinel CharMapping (e.g., identity of length 0 — but Phase 1 doesn't allow this).
2. **Handle at TextProcessor level**: if input is empty, skip normalization entirely, produce empty ProcessedText with no tokens.

**Recommendation:** Handle at TextProcessor level. If input is empty, return a ProcessedText with empty tokens, empty layers, and no CharMapping. Individual normalizers should return an error for empty input (per CONTEXT.md: "empty output is a typed error").

**IMPORTANT:** CONTEXT.md says "empty output, invalid mapping, or any unexpected state: return a typed error." But empty INPUT producing empty OUTPUT is different from non-empty input producing empty output. The TextProcessor should handle the empty-input case before calling normalizers.

### Pitfall 6: NGram Tokenizer Span Correctness

**What goes wrong:** A WordNGramTokenizer with n=2 over tokens ["hello", "world", "foo"] produces bigrams ["hello world", "world foo"]. The Span for "hello world" must cover the byte range from hello's start to world's end — including the whitespace between them. But the base tokens may not cover the whitespace.

**Why it happens:** Base tokenizer tokens may have gaps in their spans (e.g., WordTokenizer skips whitespace).

**How to avoid:** For word n-gram Spans, use `Span::new(first_token.span.start(), last_token.span.end())` — the merged span covers everything including gaps. The interned n-gram text is the concatenation with spaces, which may not exactly match the original text in the span range, but the span correctly identifies the source region.

### Pitfall 7: SentenceTokenizer Boundary Edge Cases

**What goes wrong:** "Dr. Smith went to Washington." should be ONE sentence, not two. "I said 'Hello.' She replied." has complex quotation boundary rules. "3.14 is pi." should not split at the decimal point.

**Why it happens:** Sentence boundary detection is surprisingly complex.

**How to avoid:** Use `unicode-segmentation`'s `.unicode_sentences()` which implements UAX#29 rules. Do NOT hand-roll sentence splitting. Test with abbreviations, decimal numbers, quotations, and ellipsis.

## Code Examples

### Example 1: Lowercase Normalizer with CharMapping (TDD Style)

**Tests first (RED):**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercase_ascii() {
        let norm = Lowercase;
        let result = norm.normalize("Hello World").unwrap();
        assert_eq!(result.text, "hello world");
        // Every position should round-trip
        for i in 0..result.text.len() as u32 {
            let orig = result.mapping.to_original(i).unwrap();
            let back = result.mapping.to_normalized(orig).unwrap();
            assert_eq!(back, i);
        }
    }

    #[test]
    fn lowercase_german_sharp_s() {
        // ß is already lowercase, should be unchanged
        let norm = Lowercase;
        let result = norm.normalize("Straße").unwrap();
        assert_eq!(result.text, "straße");
    }

    #[test]
    fn lowercase_preserves_mapping_for_multibyte() {
        let norm = Lowercase;
        let result = norm.normalize("Ä").unwrap(); // 2-byte char
        assert_eq!(result.text, "ä");
        assert_eq!(result.mapping.to_normalized(0).unwrap(), 0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn lowercase_mapping_round_trips(input in "\\PC{1,100}") {
            let norm = Lowercase;
            if let Ok(result) = norm.normalize(&input) {
                // Every normalized position should map back to a valid original position
                for (norm_offset, _) in result.text.char_indices() {
                    let orig = result.mapping.to_original(norm_offset as u32);
                    prop_assert!(orig.is_ok(), "Failed to_original for norm pos {}", norm_offset);
                }
            }
        }
    }
}
```

**Implementation (GREEN):**
```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct Lowercase;

impl Normalizer for Lowercase {
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
        if input.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let mut normalized = String::with_capacity(input.len());
        let mut alignments = Vec::new();
        let mut norm_byte_offset: u32 = 0;

        for (orig_byte_offset, ch) in input.char_indices() {
            for lower_ch in ch.to_lowercase() {
                alignments.push((orig_byte_offset as u32, norm_byte_offset));
                normalized.push(lower_ch);
                norm_byte_offset += lower_ch.len_utf8() as u32;
            }
        }

        let mapping = CharMapping::new(alignments)?;
        Ok(NormalizationResult { text: normalized, mapping })
    }

    fn name(&self) -> &str { "lowercase" }
    fn cost(&self) -> f32 { 0.1 }
}
```

### Example 2: Unicode Normalizer with Alignment Tracking

```rust
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy)]
pub enum NormalizationForm {
    NFC,
    NFD,
    NFKC,
    NFKD,
}

#[derive(Debug, Clone, Copy)]
pub struct UnicodeNormalizer {
    pub form: NormalizationForm,
}

impl Normalizer for UnicodeNormalizer {
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
        if input.is_empty() {
            return Err(NormalizeError::EmptyInput);
        }

        let normalized: String = match self.form {
            NormalizationForm::NFC => input.nfc().collect(),
            NormalizationForm::NFD => input.nfd().collect(),
            NormalizationForm::NFKC => input.nfkc().collect(),
            NormalizationForm::NFKD => input.nfkd().collect(),
        };

        // Build alignment by comparing original and normalized char-by-char.
        // This is the hard part — see Architecture Patterns section.
        let mapping = build_unicode_normalization_mapping(input, &normalized)?;

        Ok(NormalizationResult { text: normalized, mapping })
    }

    fn name(&self) -> &str {
        match self.form {
            NormalizationForm::NFC => "unicode_nfc",
            NormalizationForm::NFD => "unicode_nfd",
            NormalizationForm::NFKC => "unicode_nfkc",
            NormalizationForm::NFKD => "unicode_nfkd",
        }
    }
}
```

**Claude's Discretion recommendation for default form:** Use **NFC** as the default. NFC is the most common normalization form, produces the most compact representation, and is what most systems expect. Web content, filenames, and most text processing tools use NFC.

### Example 3: WordTokenizer (Whitespace-Only Splits)

```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct WordTokenizer;

impl Tokenizer for WordTokenizer {
    fn tokenize(
        &self,
        text: &str,
        mapping: &CharMapping,
        store: &mut TextStoreBuilder,
    ) -> Result<Vec<Token>, TokenizeError> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let mut tokens = Vec::new();
        let mut word_start: Option<usize> = None;

        for (byte_idx, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if let Some(start) = word_start {
                    // End of word
                    let word = &text[start..byte_idx];
                    let id = store.intern(word);
                    tokens.push(Token::new(
                        id,
                        Span::new(start as u32, byte_idx as u32),
                        TokenKind::Regular,
                    ));
                    word_start = None;
                }
            } else if word_start.is_none() {
                word_start = Some(byte_idx);
            }
        }

        // Handle last word (no trailing whitespace)
        if let Some(start) = word_start {
            let word = &text[start..];
            let id = store.intern(word);
            tokens.push(Token::new(
                id,
                Span::new(start as u32, text.len() as u32),
                TokenKind::Regular,
            ));
        }

        Ok(tokens)
    }

    fn name(&self) -> &str { "word" }
}
```

### Example 4: CharTokenizer with Grapheme Clusters

```rust
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Copy, Default)]
pub struct CharTokenizer;

impl Tokenizer for CharTokenizer {
    fn tokenize(
        &self,
        text: &str,
        _mapping: &CharMapping,
        store: &mut TextStoreBuilder,
    ) -> Result<Vec<Token>, TokenizeError> {
        let mut tokens = Vec::new();

        for (byte_idx, grapheme) in text.grapheme_indices(true) {
            let id = store.intern(grapheme);
            let end = byte_idx + grapheme.len();
            tokens.push(Token::new(
                id,
                Span::new(byte_idx as u32, end as u32),
                TokenKind::Regular,
            ));
        }

        Ok(tokens)
    }

    fn name(&self) -> &str { "char" }
}
```

### Example 5: Proptest for CharMapping Round-Trip Property

```rust
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy generating Unicode strings with interesting properties
    fn unicode_text() -> impl Strategy<Value = String> {
        prop_oneof![
            // ASCII
            "[a-zA-Z0-9 ]{1,100}",
            // Accented Latin
            "[a-zA-ZàáâãäåèéêëìíîïòóôõöùúûüýÿñçÀÁÂÃÄÅÈÉÊËÌÍÎÏ]{1,50}",
            // CJK
            "[\u{4e00}-\u{9fff}]{1,30}",
            // Emoji
            "[😀-😿🎉🔥💯]{1,20}",
            // Mixed
            "\\PC{1,100}",
        ]
    }

    proptest! {
        /// For any normalizer, the CharMapping must allow round-tripping
        /// every normalized position back to an original position.
        #[test]
        fn normalizer_mapping_round_trip(
            input in unicode_text(),
            normalizer_idx in 0usize..6,
        ) {
            let normalizer: Box<dyn Normalizer> = match normalizer_idx {
                0 => Box::new(Lowercase),
                1 => Box::new(WhitespaceNormalizer::default()),
                2 => Box::new(UnicodeNormalizer { form: NormalizationForm::NFC }),
                3 => Box::new(RemoveDiacritics),
                4 => Box::new(RemovePunctuation::default()),
                _ => Box::new(RemoveDigits),
            };

            if let Ok(result) = normalizer.normalize(&input) {
                // Property: every char boundary in normalized text has a valid mapping
                for (norm_offset, _) in result.text.char_indices() {
                    let orig = result.mapping.to_original(norm_offset as u32);
                    prop_assert!(
                        orig.is_ok(),
                        "Normalizer {:?} failed to_original at norm offset {} for input {:?}",
                        normalizer.name(), norm_offset, input
                    );
                }
            }
        }

        /// Composing two normalizer mappings preserves round-trip property.
        #[test]
        fn composed_mapping_round_trips(input in unicode_text()) {
            let n1 = Lowercase;
            let n2 = WhitespaceNormalizer::default();

            if let (Ok(r1), Ok(_)) = (n1.normalize(&input), Ok(())) {
                if let Ok(r2) = n2.normalize(&r1.text) {
                    if let Ok(composed) = r1.mapping.compose(&r2.mapping) {
                        for (norm_offset, _) in r2.text.char_indices() {
                            let orig = composed.to_original(norm_offset as u32);
                            prop_assert!(orig.is_ok());
                        }
                    }
                }
            }
        }
    }
}
```

### Example 6: Test Fixtures for Unicode Edge Cases

```rust
#[cfg(test)]
mod unicode_fixtures {
    use super::*;
    use test_case::test_case;

    // ── CJK ──────────────────────────────────────────────────────────
    #[test_case("你好世界" ; "chinese hello world")]
    #[test_case("東京都" ; "tokyo")]
    #[test_case("한국어" ; "korean")]
    fn cjk_text_normalizes_correctly(input: &str) {
        let norm = Lowercase;
        let result = norm.normalize(input).unwrap();
        // CJK has no case, so output should equal input
        assert_eq!(result.text, input);
    }

    // ── Emoji ────────────────────────────────────────────────────────
    #[test_case("😀" ; "simple emoji")]
    #[test_case("👨‍👩‍👧‍👦" ; "family emoji with ZWJ")]
    #[test_case("🇺🇸" ; "flag emoji regional indicators")]
    #[test_case("👋🏽" ; "skin tone modifier")]
    fn emoji_preserved_through_normalization(input: &str) {
        let norm = Lowercase;
        let result = norm.normalize(input).unwrap();
        assert_eq!(result.text, input);
    }

    // ── RTL ──────────────────────────────────────────────────────────
    #[test_case("مرحبا" ; "arabic hello")]
    #[test_case("שלום" ; "hebrew shalom")]
    fn rtl_text_normalizes_correctly(input: &str) {
        let norm = Lowercase;
        let result = norm.normalize(input).unwrap();
        assert_eq!(result.text.len(), input.len());
    }

    // ── Combining marks ──────────────────────────────────────────────
    #[test_case("e\u{0301}", "é" ; "e + combining acute to NFC")]
    #[test_case("n\u{0303}", "ñ" ; "n + combining tilde to NFC")]
    fn nfc_composes_combining_marks(input: &str, expected: &str) {
        let norm = UnicodeNormalizer { form: NormalizationForm::NFC };
        let result = norm.normalize(input).unwrap();
        assert_eq!(result.text, expected);
    }

    // ── Diacritics removal ───────────────────────────────────────────
    #[test_case("café", "cafe" ; "french cafe")]
    #[test_case("naïve", "naive" ; "diaeresis")]
    #[test_case("über", "uber" ; "german umlaut")]
    #[test_case("日本語", "日本語" ; "CJK unchanged")]
    fn diacritics_removed_correctly(input: &str, expected: &str) {
        let norm = RemoveDiacritics;
        let result = norm.normalize(input).unwrap();
        assert_eq!(result.text, expected);
    }

    // ── Whitespace collapse ──────────────────────────────────────────
    #[test_case("a  b", "a b" ; "double space")]
    #[test_case("a\t\tb", "a b" ; "tabs")]
    #[test_case("a\n\nb", "a b" ; "newlines")]
    #[test_case("  leading", "leading" ; "leading whitespace")]
    #[test_case("trailing  ", "trailing" ; "trailing whitespace")]
    fn whitespace_collapsed_correctly(input: &str, expected: &str) {
        let norm = WhitespaceNormalizer::default(); // collapse + trim
        let result = norm.normalize(input).unwrap();
        assert_eq!(result.text, expected);
    }
}
```

### Example 7: Integration Test for Full Pipeline

```rust
#[test]
fn full_pipeline_lowercase_then_word_tokenize() {
    let normalizers: Vec<Box<dyn Normalizer>> = vec![
        Box::new(Lowercase),
    ];
    let tokenizer: Box<dyn Tokenizer> = Box::new(WordTokenizer);

    let mut processor = TextProcessor::new(normalizers, tokenizer, ExecutionMode::All);
    let result = processor.process("Hello World").unwrap();

    assert_eq!(result.normalized, "hello world");
    assert_eq!(result.tokens.len(), 2);

    // Token text via store
    let store = &result.text_store;
    assert_eq!(store.resolve(result.tokens[0].text_id).unwrap(), "hello");
    assert_eq!(store.resolve(result.tokens[1].text_id).unwrap(), "world");

    // Span maps back to original via composed mapping
    let hello_span = result.tokens[0].span;
    let orig_start = result.composed_mapping.to_original(hello_span.start()).unwrap();
    assert_eq!(orig_start, 0); // "Hello" starts at 0 in original

    // Layer stored
    assert_eq!(result.layers.len(), 1);
    assert_eq!(result.layers[0].name, "lowercase");
    assert_eq!(result.layers[0].text, "hello world");
}
```

## TDD Strategy and Testing Patterns

### Test Organization

```
crates/core/src/
├── normalize/
│   ├── mod.rs                  # Unit tests for trait contract
│   ├── lowercase.rs            # Unit + proptest + test_case for Lowercase
│   ├── whitespace.rs           # Unit + proptest + test_case for Whitespace
│   ├── unicode.rs              # Unit + proptest + test_case for Unicode
│   ├── diacritics.rs           # Unit + proptest + test_case for Diacritics
│   ├── punctuation.rs          # Unit + proptest + test_case for Punctuation
│   └── digits.rs               # Unit + proptest + test_case for Digits
├── tokenize/
│   └── ...                     # Same pattern
└── process/
    └── ...                     # Integration tests in mod + snapshot tests

crates/core/tests/
├── integration.rs              # (existing Phase 1 tests)
├── text_processing.rs          # Phase 2 integration tests
└── unicode_edge_cases.rs       # Comprehensive Unicode test suite
```

### Property-Based Testing Strategy

| Property | What It Validates | Generator |
|----------|-------------------|-----------|
| CharMapping round-trip | `to_original(to_normalized(pos)) == pos` for all positions | Any unicode string |
| CharMapping composition associativity | `(a.compose(b)).compose(c) == a.compose(b.compose(c))` | Three normalizer chains |
| Token span coverage | Union of all token spans covers input text | Any unicode string via any tokenizer |
| Token span non-overlap | No two tokens have overlapping spans (for non-NGram tokenizers) | Any unicode string |
| Normalization idempotence | `normalize(normalize(x)) == normalize(x)` for deterministic normalizers | Any unicode string |
| Pipeline CharMapping integrity | Composed mapping covers all positions in final text | Multi-normalizer pipeline |

### Snapshot Testing Strategy

Use `insta` for ProcessedText output that is too complex for manual assertions:

```rust
#[test]
fn snapshot_full_pipeline_output() {
    let result = process_text_with_default_pipeline("Hello, World! 123");
    insta::assert_yaml_snapshot!(result, {
        ".text_store" => "[TextStore]", // redact non-deterministic parts
    });
}
```

### Integration Test Strategy

| Test Category | What It Tests | Example |
|---------------|---------------|---------|
| Pipeline composition | Multiple normalizers chain correctly | Lowercase -> Whitespace -> Unicode NFC |
| CharMapping traceability | End-to-end position mapping | Token in normalized text maps to correct original position |
| Named layer access | Layer lookup by name works | `processed.layer("lowercase")` returns correct layer |
| Minimal execution mode | Only required layers are computed | Configure 3 normalizers, require 1 -> only 1 layer stored |
| Error propagation | Pipeline errors surface correctly | Invalid normalizer in chain -> typed error |
| Unicode correctness | CJK, emoji, RTL work end-to-end | Full pipeline with CJK input produces correct tokens |

## Error Types to Add

The existing `error.rs` needs new variants for Phase 2:

```rust
/// Errors during text normalization.
#[derive(Debug, thiserror::Error)]
pub enum NormalizeError {
    /// Normalizer received empty input.
    #[error("normalizer received empty input")]
    EmptyInput,

    /// Normalizer produced output with invalid CharMapping.
    #[error("normalizer '{name}' produced invalid mapping: {reason}")]
    InvalidMapping { name: String, reason: String },

    /// Named layer not found during dependency resolution.
    #[error("normalizer layer '{0}' not found")]
    LayerNotFound(String),
}

/// Errors during tokenization.
#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    /// Tokenizer received empty input.
    #[error("tokenizer received empty input")]
    EmptyInput,

    /// Tokenizer encountered invalid state.
    #[error("tokenizer '{name}' failed: {reason}")]
    Failed { name: String, reason: String },
}

/// Errors during text processing pipeline.
#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    /// Normalization step failed.
    #[error("normalization step '{normalizer}' failed: {source}")]
    NormalizationFailed {
        normalizer: String,
        source: NormalizeError,
    },

    /// Tokenization failed.
    #[error("tokenization failed: {0}")]
    TokenizationFailed(#[from] TokenizeError),

    /// Pipeline configuration error.
    #[error("pipeline configuration error: {0}")]
    Configuration(String),
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `unicode-normalization-alignments` for position tracking | Build custom alignment on `unicode-normalization` 0.1.25 | 2020 (alignments crate abandoned) | Must build CharMapping manually during normalization |
| `unicode-segmentation` 1.11 | `unicode-segmentation` 1.12 | 2024 | Updated UAX#29 rules, better emoji handling |
| Manual sentence splitting on `.!?` | `unicode-segmentation` `.unicode_sentences()` | Established | Handles abbreviations, quotations, decimals |
| Manual Debug assertions | `insta` 1.46 snapshot testing | Established | Review workflow, cargo-insta CLI |
| `test-case` 3.0 | `test-case` 3.3 | 2024 | Latest stable, supports latest Rust |

## CharMapping Extension Decision

**Key question:** Does the Phase 1 CharMapping need to be extended for Phase 2?

**Answer:** The existing Phase 1 CharMapping is SUFFICIENT for Phase 2 with careful usage:

1. **Many-to-one (whitespace collapse):** Store only the first original position. Other original positions simply have no entry. `to_original(normalized_space_pos)` returns the first original space. WORKS.

2. **One-to-many (NFD decomposition):** Store multiple entries with the same original position but different normalized positions. `to_normalized(orig_pos)` returns the first normalized position via binary search. `to_original(norm_pos)` finds the right one via linear scan. WORKS but may return wrong original pos if multiple originals map to same normalized pos — need careful handling.

3. **One-to-zero (character removal):** Original positions with no normalized counterpart simply have no entry in alignments. `to_normalized(removed_pos)` returns InvalidPosition error. Callers must handle this. WORKS.

4. **Composition:** `compose()` works correctly for all these cases since it operates on (orig, mid) pairs.

**Potential extension:** Add a `to_original_range(normalized: u32) -> Result<(u32, u32), NormalizeError>` method that returns the RANGE of original positions that mapped to a given normalized position (useful for collapsed mappings). This is optional and can be added if needed.

## Claude's Discretion Recommendations

### Normalizer Aggressiveness

| Normalizer | Recommended Behavior | Rationale |
|------------|---------------------|-----------|
| **Lowercase** | `str::to_lowercase()` (full Unicode) | Simple, well-tested, handles Turkish İ etc. |
| **Whitespace** | Collapse runs + trim leading/trailing + normalize \t\n\r to space | Most aggressive useful default. Configurable via struct fields. |
| **Unicode** | Default to **NFC** | NFC is the W3C recommendation, most compact, web-standard. Configurable via `NormalizationForm` enum. |
| **RemoveDiacritics** | NFD decompose then filter `\p{Mn}` (Mark, Nonspacing) category | Standard approach. Handles all accented Latin, Cyrillic, Greek, etc. |
| **RemovePunctuation** | Remove chars where `char::is_ascii_punctuation()` is true | Conservative default (ASCII only). Configurable to include Unicode punctuation via `\p{P}` category. |
| **RemoveDigits** | Remove chars where `char::is_ascii_digit()` is true | Conservative default (ASCII 0-9 only). Configurable to include Unicode digits. |

### NGram Internal Representation

**Recommendation:** Create new StringIds for n-gram token text. For WordNGramTokenizer, the n-gram text is the space-joined base tokens. For CharNGramTokenizer, it's the substring of consecutive grapheme clusters.

**Rationale:** Self-describing tokens are simpler to work with downstream. The TextStoreBuilder deduplicates automatically, so repeated n-grams share the same StringId. Storing index references to base tokens would require a separate lookup mechanism that adds complexity without performance benefit (n-gram text is typically short).

### Layer Storage Format

**Recommendation:** `Vec<NormalizationLayer>` ordered by pipeline execution order. Each layer stores `name: String`, `text: String`, `mapping: CharMapping`. In Minimal mode, only layers in the required set are pushed to this Vec. The composed mapping is eagerly computed and stored separately.

**Caching strategy:** Compute the composed CharMapping eagerly during pipeline execution (compose after each normalizer step). Store the final composed mapping in `ProcessedText.composed_mapping`. Layer-specific composed mappings (original -> layer N) are computed lazily on demand via `mapping_to_layer(name)`.

### CJK/Emoji/RTL Handling in Tokenizers

- **CharTokenizer:** Use `grapheme_indices(true)` (extended grapheme clusters). This correctly handles ZWJ emoji sequences, flag emoji (regional indicators), skin tone modifiers as single tokens.
- **WordTokenizer:** Whitespace-only split. CJK text with no spaces becomes a single token. This is correct per CONTEXT.md ("splits on whitespace only").
- **SentenceTokenizer:** Use `unicode-segmentation`'s `.unicode_sentences()`. Handles CJK period (。), Arabic sentence boundaries, etc.
- **RTL text:** No special handling needed at the tokenizer level — byte offsets are directionally agnostic. The Span represents a byte range, which works correctly regardless of text direction.

## Open Questions

1. **CharMapping for completely removed text:** When RemovePunctuation removes ALL characters from input like "!!!", the result is empty. CONTEXT.md says "empty output is a typed error." Confirm this is correct even when the normalizer is working as designed. **Recommendation:** Return `NormalizeError::EmptyOutput` with context about why. The TextProcessor can decide whether to proceed with empty text or propagate the error.

2. **TextStoreBuilder mutability in Tokenizer trait:** The tokenizer needs `&mut TextStoreBuilder` to intern tokens. But TextProcessor may want to run tokenization after normalization is complete. The TextStoreBuilder must be alive and mutable at tokenization time. **Recommendation:** TextProcessor owns the TextStoreBuilder, passes `&mut` to tokenizer, then calls `.build()` to freeze into `Arc<TextStore>` before constructing ProcessedText.

3. **Layer name uniqueness:** What if two normalizers have the same `name()`? The named-layer system assumes unique names. **Recommendation:** TextProcessor validates name uniqueness at construction time. Duplicate names produce a `ProcessError::Configuration` error.

## Sources

### Primary (HIGH confidence)
- Phase 1 source code: `crates/core/src/{char_mapping,token,text_store,span,error}.rs` — verified API
- unicode-normalization 0.1.25 docs — NFC/NFD/NFKC/NFKD API
- unicode-segmentation 1.12 docs — UnicodeSegmentation trait with grapheme_indices, unicode_word_indices, unicode_sentences
- proptest 1.6+ docs — proptest!, prop_compose!, string strategies
- insta 1.46 — snapshot testing, inline snapshots, cargo-insta
- test-case 3.3 docs — parameterized test macro
- CONTEXT.md decisions — locked choices for normalizer/tokenizer behavior
- TECHNICAL_SPEC.md — original trait designs and ProcessedText structure

### Secondary (MEDIUM confidence)
- HuggingFace tokenizers NormalizedString — reference implementation for alignment tracking
- unicode-normalization-alignments — HuggingFace fork with alignment diffs, confirms approach of tracking alignment during normalization
- UAX#15 Unicode Normalization Forms — NFC/NFD/NFKC/NFKD specification
- UAX#29 Unicode Text Segmentation — grapheme, word, sentence boundary rules

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — unicode-normalization and unicode-segmentation are the established Rust crates, versions verified
- Architecture: HIGH — trait-based design validated against Phase 1 patterns and TECHNICAL_SPEC
- Pitfalls: HIGH — Unicode edge cases well-documented, CharMapping limitations understood from Phase 1 code review
- TDD patterns: HIGH — proptest already used in Phase 1, test-case and insta are standard Rust testing tools
- Code examples: MEDIUM — examples are representative but implementation details may vary

**Research date:** 2026-02-06
**Valid until:** 2026-03-06 (stable domain, Unicode standards don't change frequently)
