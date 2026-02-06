# Architecture

**Analysis Date:** 2026-02-05

## Current State

The codebase is in **very early development** (v0.1.0). The existing code is a proof-of-concept for BERT-based sentence embeddings, not the full diffing library described in the design document. The actual diff/analysis architecture has been designed but not yet implemented.

## Existing Code Architecture

### What's Built

A single utility module with BERT model loading and embedding computation:

```
User Code
    │
    ▼
bert_v1.rs functions
    │
    ├── model_all_minilm_l6_v2() ──→ HuggingFace Hub (download)
    ├── model_all_mpnet_base_v2() ──→ HuggingFace Hub (download)
    ├── get_embeddings()          ──→ Candle (inference)
    └── cosine_similarity()       ──→ Candle (tensor math)
```

### Data Flow (Current)

1. Model loading: Download from HF Hub → Parse config.json → Load safetensors weights → Initialize BertModel
2. Embedding: Input text → Tokenize (HF tokenizers) → Pad to max length → BERT forward pass → Mean pooling → L2 normalize
3. Similarity: Two embeddings → Dot product → Scalar cosine similarity

## Planned Architecture (from DESIGN.md)

The design document describes a layered architecture for v1:

```
┌──────────────────────────────────────────┐
│           User API Layer                  │
│  (Builders, presets, convenience)         │
├──────────────────────────────────────────┤
│         Orchestration Layer               │
│  DiffOrchestrator, AnalysisCoordinator,  │
│  CacheManager                            │
├──────────────────────────────────────────┤
│          Core Engine Layer                │
│  TextProcessor, DiffComputer,            │
│  MetricsEngine                           │
├──────────────────────────────────────────┤
│         Foundation Layer                  │
│  TextStore (arena/interning),            │
│  CharMapping, Types (Token, Span)        │
└──────────────────────────────────────────┘
```

### Key Design Decisions (from DESIGN.md)

- **Zero-copy strings**: Arena-allocated `TextStore` with `StringId` references (4 bytes, Copy)
- **Plugin architecture**: Analyzers as `AnalyzerPlugin` trait objects with dependency/cost metadata
- **Async-first**: Tokio runtime for parallel analyzer execution
- **Smart caching**: Content-addressed cache with dependency-aware invalidation
- **Python bindings**: PyO3 adapter layer with GIL management and asyncio bridge
- **Structured errors**: `thiserror`-based error hierarchy (not yet implemented)

## Entry Points

- **Library**: `crates/core/src/lib.rs` (re-exports `mod utils`)
- **Example**: `crates/core/examples/hello_world.rs` (trivial, just prints "Hello, world!")
- **Test**: `crates/core/src/utils/bert_v1.rs` (embedded `#[cfg(test)]` module)

## Patterns

- **Error handling**: `anyhow::Result<T>` with `?` propagation
- **Device selection**: Runtime feature detection (`cfg!(feature = "metal")` + capability check)
- **Model loading**: Synchronous HF Hub API → file download → deserialization

---

*Architecture analysis: 2026-02-05*
