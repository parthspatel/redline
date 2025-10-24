# Redline Core v1 - Design Documentation

This directory contains the complete design and technical specification for Redline Core v1.0, a comprehensive refactoring and redesign of the v0 framework.

## Documents

### [DESIGN.md](./DESIGN.md)
The **Design Document** provides:
- High-level architecture and design principles
- Analysis of v0 strengths and weaknesses
- V1 design goals and objectives
- Module architecture overview
- Key component designs
- Performance optimization strategies
- Migration path from v0
- Testing strategy

**Start here** if you want to understand the *why* and *what* of v1.

### [TECHNICAL_SPEC.md](./TECHNICAL_SPEC.md)
The **Technical Specification** provides:
- Detailed API definitions for all components
- Complete data structure specifications
- Implementation guidelines
- Error handling patterns
- Performance targets and benchmarks
- Phase-by-phase implementation plan

**Read this** when you're ready to implement v1 or need detailed API references.

## Quick Summary

### What's Changing in v1?

#### Architecture
- **v0**: Monolithic `DiffEngine` that does everything
- **v1**: Layered architecture with clear separation of concerns
  - Foundation Layer: TextStore, Span, CharMapping
  - Text Processing: Normalizers, Tokenizers
  - Diff Computation: Pure diff algorithms
  - Metrics Engine: Centralized metric computation
  - Analysis Framework: Plugin-based analyzers
  - Orchestration: High-level workflow coordination

#### Performance
- **String Interning**: Reduce memory usage by 40-50%
- **Arena Allocation**: Faster allocations, better cache locality
- **Smart Caching**: Content-addressed caching for metrics
- **Lazy Evaluation**: Compute only what's needed
- **Async Support**: Parallel execution of analyzers

#### Developer Experience
- **Type Safety**: Stronger types, better error messages
- **Structured Errors**: Replace panics with proper error types
- **Better Config**: Cleaner builder API with presets
- **Plugin System**: Easy to add custom analyzers
- **Async-First**: Support for async operations

### Key Improvements

| Aspect | v0 | v1 |
|--------|----|----|
| **Memory** | Heavy string duplication | String interning + arena allocation |
| **Caching** | Ad-hoc, metrics struct | Content-addressed, dependency-aware |
| **Analyzers** | Two traits, tight coupling | Unified plugin system |
| **Async** | None | Full async support |
| **Errors** | Panics, unwraps | Structured error types |
| **Config** | 10+ boolean flags | Preset-based builder |
| **Extensibility** | Modify core code | Plugin registration |

### Performance Targets

| Operation | v0 | v1 Target | Improvement |
|-----------|-----|-----------|-------------|
| Simple diff (100 words) | 1.2ms | 0.6ms | 2x faster |
| With normalization | 2.5ms | 1.0ms | 2.5x faster |
| Full analysis | 50ms | 20ms | 2.5x faster |
| 10K word document | 500ms | 150ms | 3.3x faster |
| Memory (10K words) | ~80MB | <50MB | 37% reduction |

## Design Principles

### 1. Separation of Concerns
Each layer has a single, well-defined responsibility. Components are independently usable and testable.

### 2. Zero-Copy Where Possible
Use lifetimes, references, and string interning to minimize allocations.

### 3. Plugin Architecture
Analyzers are plugins that declare dependencies and can be registered dynamically.

### 4. Smart Caching
Content-addressed caching with dependency tracking to avoid redundant computations.

### 5. Async-First
Support both sync and async APIs, with parallel execution for independent operations.

### 6. Type Safety
Leverage Rust's type system to prevent errors at compile time.

## Example: v0 vs v1

### v0 API
```rust
use redline_core::{DiffEngine, DiffConfig, DiffAlgorithm, TextPipeline};
use redline_core::normalizers::Lowercase;
use redline_core::tokenizers::WordTokenizer;

let config = DiffConfig::default()
    .with_algorithm(DiffAlgorithm::Histogram)
    .with_pipeline(
        TextPipeline::new()
            .add_normalizer(Box::new(Lowercase))
    )
    .with_tokenizer(Box::new(WordTokenizer::new()))
    .with_semantic_similarity(true)
    .with_edit_classification(true)
    .with_style_analysis(true);

let engine = DiffEngine::new(config);
let result = engine.diff("Hello World", "Hello Rust");
```

### v1 API
```rust
use redline_core_v1::{DiffOrchestrator, ConfigBuilder, Preset};

// Using preset (simplest)
let mut orchestrator = DiffOrchestrator::with_preset(Preset::Semantic)?;
let report = orchestrator.diff_with_analysis(
    "Hello World",
    "Hello Rust",
    &["semantic", "stylistic"]
)?;

// Or with builder (custom)
let config = ConfigBuilder::preset(Preset::Semantic)
    .text_processing(|t| t
        .normalize(|n| n.lowercase().collapse_whitespace())
        .tokenize(Tokenization::Words)
    )
    .analysis(|a| a
        .add_analyzer("semantic")
        .add_analyzer("readability")
    )
    .build()?;

let mut orchestrator = DiffOrchestrator::with_config(&config)?;
let result = orchestrator.diff(original, modified)?;
```

## Module Structure

```
redline-core-v1/
├── foundation/        # Core types (TextStore, Span, CharMapping)
├── text/             # Text processing (normalizers, tokenizers)
├── diff/             # Diff algorithms and computation
├── metrics/          # Metric computation and caching
├── analysis/         # Analyzer plugins and framework
├── cache/            # Caching layer
├── orchestration/    # High-level workflow coordination
└── config/           # Configuration and builders
```

## Implementation Status

- [ ] Phase 1: Foundation Layer
- [ ] Phase 2: Text Processing Layer
- [ ] Phase 3: Diff Computation Layer
- [ ] Phase 4: Metrics Engine
- [ ] Phase 5: Analysis Framework
- [ ] Phase 6: Caching & Optimization
- [ ] Phase 7: Orchestration & API
- [ ] Phase 8: Testing & Documentation

## Migration from v0

### Compatibility Layer

v1 will provide a compatibility layer in `redline_core_v1::compat` that wraps the v1 API to provide a v0-compatible interface. This allows gradual migration.

```rust
// v0 code works with compatibility layer
use redline_core_v1::compat::{DiffEngine, DiffConfig};

let engine = DiffEngine::default();
let result = engine.diff("old", "new");
```

### Migration Steps

1. **Phase 1**: Use v1 with compat layer (no code changes)
2. **Phase 2**: Migrate to v1 builders (better API, same functionality)
3. **Phase 3**: Adopt v1 plugins (extensibility)
4. **Phase 4**: Use async APIs (performance)

## Contributing

When implementing v1:

1. **Follow the spec**: Stick to the APIs defined in TECHNICAL_SPEC.md
2. **Write tests first**: TDD approach for each component
3. **Benchmark**: Ensure performance targets are met
4. **Document**: Add rustdoc comments to all public APIs
5. **Examples**: Provide usage examples for each major feature

## Questions or Feedback?

If you have questions about the design or suggestions for improvements, please:

1. Review the design principles and rationale in DESIGN.md
2. Check the detailed API specs in TECHNICAL_SPEC.md
3. Open an issue with your question or proposal
4. Tag it with `v1-design` label

## References

### Related Documents
- [v0 Source Code](../core/src/)
- [v0 Examples](../core/examples/)

### External Resources
- [Myers Diff Algorithm](http://www.xmailserver.org/diff2.pdf)
- [Patience Diff](https://bramcohen.livejournal.com/73318.html)
- [String Interning in Rust](https://matklad.github.io/2020/03/22/fast-simple-rust-interner.html)
- [Arena Allocation](https://docs.rs/bumpalo/)

---

**Last Updated**: 2025-10-15  
**Status**: Design Phase  
**Next Milestone**: Begin Phase 1 Implementation
