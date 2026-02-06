# Testing Patterns

**Analysis Date:** 2026-02-05

## Framework

- Built-in Rust test framework (`cargo test`)
- No external test frameworks (criterion commented out in dev-dependencies)
- Standard `assert!`, `assert_eq!` macros

## Run Commands

```bash
cargo test                              # All workspace tests
cargo test --package redline-core       # Core crate only
cargo test -- --nocapture               # Show println output
cargo test --release --features metal   # With GPU acceleration
```

## Test Organization

- **Inline pattern**: `#[cfg(test)] mod tests` inside source files
- **Location**: Single test in `crates/core/src/utils/bert_v1.rs:172-226`
- No `tests/` directory (no integration tests)
- No benchmark suite (criterion is commented out)

## Current Tests

One test exists (`bert_v1::tests::main`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main() -> Result<()> {
        let (model, tokenizer) = model_all_mpnet_base_v2()?;
        let device = /* feature-gated device selection */;
        let sentences = [/* 4 test sentences */];
        let embeddings = get_embeddings(&model, &tokenizer, &sentences, &device)?;
        // Prints cosine similarities (no assertions)
        Ok(())
    }
}
```

**Characteristics**:
- Returns `Result<()>` for `?` operator usage
- Loads real model from HuggingFace Hub (requires network)
- Feature-gated device selection (metal/cuda/cpu)
- **No assertions** - only prints similarities for manual inspection
- Heavy test (~seconds, model download on first run)

## Mocking

- No mocking framework in use
- Tests use real model loading and inference
- No test doubles for HF Hub API or Candle

## Coverage

- No coverage tool configured (no tarpaulin, llvm-cov)
- No coverage thresholds

## Gaps

- Only 1 test for the entire codebase
- No unit tests for individual functions (`cosine_similarity`, `get_embeddings`)
- No error case testing
- No assertion-based validation
- Network dependency makes tests slow and non-deterministic
- No integration or E2E test infrastructure

---

*Testing analysis: 2026-02-05*
