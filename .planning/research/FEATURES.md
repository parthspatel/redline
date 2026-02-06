# Feature Research

**Domain:** Text diffing and analysis framework (Rust library with Python bindings)
**Researched:** 2026-02-05
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist in any text diffing/analysis library. Missing these means the product feels incomplete and users will choose an existing alternative.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Myers diff algorithm | Industry standard default (Git default); every diff lib has it | MEDIUM | O(ND) complexity. `similar` and `imara-diff` both implement this. Linear-space variant preferred for memory. |
| Line-level diffing | Most basic diff operation; `diff`, `git diff`, `similar`, `difflib` all support it | LOW | Foundation operation. Must support before word/char level. |
| Word-level diffing | Expected for prose/document comparison. `similar`, `diff-match-patch`, `chopdiff` all provide it | MEDIUM | Requires tokenizer. `similar` supports line/word/char/grapheme granularity. |
| Character-level diffing | Catches subtle edits. `diff-match-patch`, `similar`, Kaleidoscope all provide it | LOW | Simplest tokenizer. Useful for fine-grained analysis. |
| Edit operations (Insert/Delete/Replace/Equal) | Every diff library outputs these. Standard output format | LOW | `EditKind` enum. `Replace` = `Delete` + `Insert` at same position. |
| Diff statistics (edit distance, change ratio, counts) | Expected metadata on any diff result | LOW | Count of each operation type, total edit distance, similarity ratio 0.0-1.0. `similar` provides ratio(). |
| Text normalization (lowercase, whitespace collapse, trim) | Basic preprocessing before comparison. `diff-match-patch` has cleanup passes, `mddiff` normalizes before diff | LOW | Lowercase, whitespace normalization, and trim are the minimum. Users expect case-insensitive diff. |
| Unicode support | Non-negotiable in 2026. `textdistance` explicitly supports Unicode grapheme clusters | LOW | Rust handles UTF-8 natively. Need grapheme-aware tokenization for correctness. |
| Levenshtein distance | Most widely known string distance metric. `strsim`, `textdistance` both implement it | LOW | Classic edit distance. Foundation for many other metrics. |
| Jaro-Winkler similarity | Standard fuzzy matching metric. `strsim` provides it, `textdistance` provides it | LOW | Common in record linkage and fuzzy search. |
| Jaccard similarity | Standard set-based similarity. `textdistance` provides token-based version | LOW | Word-set overlap metric. Expected for text comparison. |
| Cosine similarity | Standard vector-based similarity. `textdistance` provides token-based version | MEDIUM | TF-based or count-based. Expected for text analytics. |
| Basic readability scores (Flesch, Flesch-Kincaid) | `textstat`, `py-readability-metrics`, `rust_readability` all compute these | MEDIUM | Requires syllable counting. Flesch Reading Ease and Flesch-Kincaid Grade Level are minimum. |
| Builder pattern / configuration API | Expected in any Rust library. `similar` uses builder-style TextDiff creation | LOW | Ergonomic configuration. Presets for common use cases. |
| Structured error handling | Expected in modern Rust. `thiserror` is standard | LOW | `RedlineError` hierarchy with `thiserror`. No panics in library code. |
| Serde serialization support | Expected for any data-producing Rust library. Results must be serializable | LOW | Feature-gated behind `serde` feature flag. Enables JSON/TOML output. |
| Comprehensive documentation with examples | crates.io standard. `similar` and `textdistance` both have good docs | MEDIUM | Rustdoc with examples. README with quick start. |

### Differentiators (Competitive Advantage)

Features that set Redline apart from existing libraries. No single existing library combines these. This is where Redline's value proposition lives.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Normalization pipeline with traceability | Map any position in normalized text back to original positions through all normalization layers. No existing Rust library does this. `mddiff` does this for Markdown only. `chopdiff` does word-level mapping. Redline does full character-level bidirectional mapping through arbitrary normalizer chains. | HIGH | CharMapping composition across layers is the hard part. Each normalizer produces a mapping; mappings compose via `compose()`. This is the core architectural differentiator. |
| Token-level diff operations (not just line/word/char) | Diff operates on semantically meaningful tokens with original position traceability. Tokens carry StringId (zero-copy), span, original_spans, kind. No existing library ties diff operations back through normalization layers to original text positions. | HIGH | Requires TextStore + TextProcessor + DiffComputer integration. Token design (Copy with StringId) enables this without lifetime complexity. |
| Pluggable analyzer framework with dependency resolution | Register custom analyzers that declare dependencies on metrics and other analyzers. Coordinator resolves execution order, runs independent analyzers in parallel. Similar to spaCy's pipeline but for diff analysis. No Rust diff library has this. | HIGH | AnalyzerPlugin trait + PluginRegistry + ExecutionPlanner + AnalysisCoordinator. Topological sort for dependency resolution. |
| Comprehensive metrics engine (20+ text metrics, 15+ pairwise) | Single library provides readability scores, distance metrics, set-based similarity, linguistic features, and pairwise deltas. Currently requires combining `textdistance` + `strsim` + `rust_readability` + custom code. | HIGH | Flesch, Flesch-Kincaid, Gunning Fog, SMOG, Coleman-Liau, ARI, lexical density, vocabulary richness, plus all distance/similarity metrics. Lazy evaluation with caching. |
| Built-in semantic/stylistic/readability analyzers | Pre-built analyzers for common analysis tasks (semantic similarity, style change detection, readability delta, edit classification). No Rust library provides these. Python ecosystem has scattered tools but no unified framework. | HIGH | 4 built-in analyzers. SemanticAnalyzer, StylisticAnalyzer, ReadabilityAnalyzer, EditClassifier. Each implements AnalyzerPlugin trait. |
| Edit classification (Clarification/Expansion/Reduction/Correction/Reformulation/Stylistic) | Classify the intent behind each edit operation. Academic research exists but no production library provides this as a feature. Goes beyond mechanical "insert/delete" to semantic "why was this changed?" | HIGH | Rule-based + optional ML classification. IntentCategory enum. Requires analysis of surrounding context, not just the diff operation itself. |
| Zero-copy architecture (arena allocation + string interning) | TextStore with bumpalo arena + string interning reduces allocations by ~40%. Tokens are Copy (56-64 bytes) with StringId reference. `similar` is dependency-free but allocates strings. `imara-diff` is performant but doesn't intern strings. | MEDIUM | Bumpalo for arena, hashbrown for fast interning. O(1) string lookup (~2ns). Enables cache-friendly traversal. |
| Python bindings with full plugin extensibility | PyO3 bindings that expose not just the API but allow Python implementations of normalizers, tokenizers, analyzers, and custom metrics. Python plugins callable from Rust trait objects via adapter pattern. Similar to how Polars exposes Rust performance with Python ergonomics. | HIGH | PyO3 + maturin. PyNormalizerAdapter, PyAnalyzerAdapter wrap Python objects as Rust trait objects. GIL management for Rust-to-Python calls. pyo3-asyncio for async bridge. |
| Async analysis with parallel execution | Run independent analyzers concurrently. Expensive operations (ML models, external APIs) don't block. Graceful degradation when some analyzers fail. No existing diff library offers async analysis. | MEDIUM | tokio::task::spawn_blocking for CPU-bound work. futures::join_all for parallel analyzers. AnalysisReport includes partial results on failure. |
| Configuration presets (Fast/Syntactic/Semantic/Comprehensive) | One-line setup for common use cases. Users don't need to understand all options to get started. `diff-match-patch` has similar "cleanup" presets but not for full analysis pipelines. | LOW | DiffPreset enum with pre-configured normalizer chains, tokenizer selection, and analyzer sets. Builder override for customization. |
| Content-addressed caching with LRU eviction | Cache metrics and analysis results keyed by content hash. Avoid redundant computation when same text appears in multiple comparisons. No diff library provides built-in caching. | MEDIUM | CacheManager with RwLock for concurrent access. ContentHash (SHA-256 or xxhash) for keys. Configurable max size and TTL. |
| Histogram diff algorithm | Outperforms Myers by 10-100% in benchmarks. `imara-diff` implements it, `similar` does not. Git uses it as an alternative to Myers. Better at indicating intended changes. | MEDIUM | Port or adapt from imara-diff or implement from Git's C implementation. Falls back gracefully when no low-occurrence elements exist. |
| Batch diff API | Process multiple text pairs efficiently, reusing TextStore and caches across comparisons. No existing library explicitly optimizes for batch workloads. | LOW | `batch_diff(&mut self, pairs: &[(&str, &str)])`. Reuse TextStore interning across documents. |
| Custom metric registration | Users register domain-specific metrics that integrate with the dependency system and caching. Metric trait with `id()`, `compute()`, `dependencies()`, `cost()`. | MEDIUM | MetricRegistry. Custom metrics participate in lazy evaluation and caching. Enables domain-specific analysis without forking the library. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems. Deliberately NOT building these.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Built-in ML model integration (BERT, transformer embeddings) | "Semantic similarity using BERT embeddings" is the gold standard for NLP. The v0 had `bert` and `spacy` feature flags. | Massive dependency chain (libtorch/onnxruntime, 100MB+ models). Feature flags leak through API. Model loading is slow. Version conflicts with user's ML stack. Makes the core library enormous and hard to compile. | Expose via plugin system. Python users bring their own models via PyAnalyzerPlugin. Rust users implement AnalyzerPlugin with their preferred ML runtime. Provide example plugins, not built-in integration. |
| Streaming diff for arbitrarily large files | "Diff a 10GB log file" is an edge case request. | Fundamentally changes the architecture -- tokens must fit in memory for the diff algorithm. Character mapping becomes streaming. Caching becomes impossible. Most users work with document-sized text (KB to low MB). Premature optimization for an edge case. | Set a documented maximum input size. For very large files, recommend chunking at paragraph/section boundaries. Consider adding streaming in v2 after the core architecture proves stable. |
| Real-time collaborative diff (operational transform / CRDT) | "Track changes as users type" is a different problem domain. | Completely different algorithm family (OT/CRDT, not LCS/Myers). Would double the API surface. Conflict resolution is its own research area. Better served by dedicated libraries (automerge, yrs/yjs). | Out of scope. Redline computes diffs between snapshots. For real-time, use Automerge/Yjs and feed snapshots to Redline for analysis. |
| Syntax-aware / language-specific diff | "Understand that this is Python code, diff at the AST level." SemanticDiff and difftastic do this. | Requires parsers for every target language. Tree-sitter integration is complex. Redline is a text analysis framework, not a code analysis tool. Adding language awareness would dilute the text-first focus. | Out of scope for core. Could be a plugin: user provides a language-specific tokenizer that produces syntax-aware tokens. The diff algorithm doesn't need to know about syntax -- the tokenizer handles it. |
| GUI / visual diff output | "Show me a side-by-side colored diff." | Redline is a library, not an application. Rendering is a presentation concern. Adding HTML/terminal output couples the library to display technology. | Provide serializable DiffResult. Let consumers render however they want. Offer a `redline-pretty` example crate for terminal output. `similar` took this approach correctly. |
| Automatic language detection | "Detect whether input is English, French, etc." | Adds dependency on language detection models. Most users know their input language. Accuracy for short texts is poor. Better handled upstream. | Accept language as a configuration parameter. Users set `config.language("en")`. Normalizers and syllable counting use it. |

## Feature Dependencies

```
[TextStore (arena + string interning)]
    -->requires--> [Span, StringId, CharMapping] (foundation types)

[Normalizers (Lowercase, Whitespace, Unicode, etc.)]
    -->requires--> [TextStore, CharMapping]

[Tokenizers (Word, Char, Sentence, NGram)]
    -->requires--> [TextStore, Token, CharMapping]

[TextProcessor]
    -->requires--> [Normalizers, Tokenizers, TextStore]
    -->produces--> [ProcessedText (tokens + layers + mapping)]

[DiffComputer (Myers, Histogram)]
    -->requires--> [ProcessedText, EditOperation types]

[DiffResult]
    -->requires--> [DiffComputer, ProcessedText]

[MetricsEngine]
    -->requires--> [TextStore, ProcessedText]
    -->enhances--> [CacheManager]

[CacheManager]
    -->requires--> [ContentHash]
    -->enhances--> [MetricsEngine, AnalysisCoordinator]

[AnalyzerPlugin trait + PluginRegistry]
    -->requires--> [AnalysisContext, AnalysisResult, Dependency types]

[AnalysisCoordinator]
    -->requires--> [PluginRegistry, ExecutionPlanner, CacheManager]
    -->requires--> [DiffResult, MetricsEngine]

[Built-in Analyzers (Semantic, Stylistic, Readability, EditClassifier)]
    -->requires--> [AnalyzerPlugin trait, MetricsEngine, DiffResult]

[DiffOrchestrator]
    -->requires--> [TextProcessor, DiffComputer, MetricsEngine, AnalysisCoordinator]

[ConfigBuilder + Presets]
    -->requires--> [All component types for validation]

[Async Support]
    -->enhances--> [AnalysisCoordinator, DiffOrchestrator]
    -->requires--> [tokio feature flag]

[Python Bindings]
    -->requires--> [All Rust types stable]
    -->requires--> [PyO3, maturin, pyo3-async-runtimes]

[Serde Support]
    -->enhances--> [All result types]
    -->requires--> [serde feature flag]
```

## Competitor Feature Analysis

| Feature | similar (Rust) | imara-diff (Rust) | textdistance (Rust) | diff-match-patch (multi) | difflib (Python) | Redline (Our Approach) |
|---------|---------------|-------------------|--------------------|--------------------------|-----------------|-----------------------|
| Myers algorithm | Yes | Yes (optimized) | No (distance only) | Yes | Yes (Ratcliff) | Yes |
| Histogram algorithm | No | Yes | No | No | No | Yes |
| Patience algorithm | Yes | No | No | No | No | Deferred to v1.x |
| Line/Word/Char granularity | Yes (all 4) | Line-focused | N/A | Char-focused | Line-focused | All via pluggable tokenizers |
| Edit distance metrics | No (ratio only) | No | Yes (25+ algos) | No | Basic ratio | Yes (20+ metrics) |
| Readability scores | No | No | No | No | No | Yes (Flesch, FK, Fog, SMOG+) |
| Normalization pipeline | No | No | No | Basic cleanup | No | Yes with full traceability |
| Position mapping | No | No | No | No | No | Yes (CharMapping, bidirectional) |
| Plugin system | No | No | No | No | No | Yes (AnalyzerPlugin trait) |
| Dependency resolution | No | No | No | No | No | Yes (topological sort) |
| Caching | No | No | No | No | No | Yes (content-addressed LRU) |
| Async support | No | No | No | No | No | Yes (tokio, parallel analysis) |
| Python bindings | No | No | No | Yes (native) | Yes (stdlib) | Yes (PyO3 with plugin extensibility) |
| Configuration presets | No | No | No | No | No | Yes (Fast/Syntactic/Semantic/Comprehensive) |
| Zero-copy architecture | No | Yes (efficient) | Yes (no alloc) | No | No | Yes (arena + interning) |

**Key takeaway:** No existing library combines diffing + normalization traceability + comprehensive metrics + pluggable analysis. Each covers at most 2-3 of these areas. Redline's value is the integrated framework.

## Sources

### Rust Diff Libraries
- [similar (mitsuhiko)](https://github.com/mitsuhiko/similar) - Myers + Patience + LCS, multi-granularity, no dependencies
- [imara-diff](https://github.com/pascalkuthe/imara-diff) - Myers + Histogram, performance-focused, 10-100% faster than similar
- [diff crate](https://docs.rs/diff) - LCS-based slice/string diffing

### Rust String Distance/Similarity
- [textdistance.rs](https://github.com/life4/textdistance.rs) - 25+ algorithms, pure Rust, no_std, common interface
- [strsim-rs](https://github.com/rapidfuzz/strsim-rs) - Hamming, Levenshtein, Jaro-Winkler, Dice

### Text Readability
- [rust_readability](https://github.com/ian-nai/rust_readability) - Rust readability formulas (Coleman-Liau, ARI)
- [textstat (Python)](https://pypi.org/project/textstat/) - Comprehensive readability metrics
- [py-readability-metrics](https://github.com/cdimascio/py-readability-metrics) - 9 readability formulas

### Multi-Language Diff Libraries
- [diff-match-patch (Google)](https://github.com/google/diff-match-patch) - Myers + Bitap matching + patching, 8 languages
- [Python difflib](https://docs.python.org/3/library/difflib.html) - Standard library, Ratcliff/Obershelp

### Normalization-Aware Diff
- [mddiff](https://github.com/cscheffler/mddiff) - Markdown-specific normalization with transformation tracking
- [chopdiff](https://github.com/jlevy/chopdiff) - Word-level mapping for LLM text transforms

### Pipeline Architecture
- [spaCy](https://spacy.io/usage/processing-pipelines) - Plugin pipeline architecture reference

---
*Feature research for: Text diffing and analysis framework*
*Researched: 2026-02-05*
