# Coding Conventions

**Analysis Date:** 2026-02-05

## Naming

- **Files**: snake_case (`bert_v1.rs`, `mod.rs`)
- **Functions**: snake_case (`model_all_minilm_l6_v2`, `get_embeddings`, `cosine_similarity`)
- **Types**: PascalCase (`BertModel`, `Tokenizer`, `Device`, `Tensor`, `Config`)
- **Constants**: SCREAMING_CASE (`DTYPE`)
- **Variables**: snake_case (`model_id`, `token_ids`, `padded_ids`, `batch_size`)

## Import Organization

Grouped by domain, external crates first:

```rust
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
```

- Error type aliased: `use anyhow::{Error as E, Result}`
- Wildcard imports only in test modules: `use super::*`

## Error Handling

- **Return type**: `anyhow::Result<T>` for all fallible functions
- **Propagation**: `?` operator throughout
- **External errors**: `.map_err(E::msg)` for crates that don't implement `std::Error` (e.g., tokenizers)
- **No panics**: No `unwrap()` or `expect()` in library code (only in test println)

## Function Design

- Functions are 10-40 lines typically
- References for borrowed params: `model: &BertModel`, `sentences: &[&str]`
- Tuple returns for multi-value: `Result<(BertModel, Tokenizer)>`
- All functions in `bert_v1.rs` are private (no `pub`)

## Logging

- `println!` for informational output (no structured logging framework)
- Used during model loading to report device selection
- No log levels, no tracing spans

## Module Pattern

- `mod.rs` for directory modules
- `lib.rs` at crate root with module declarations
- `#[cfg(test)] mod tests` for inline unit tests

## Git Conventions

- **Commits**: Conventional commits enforced via commitizen hook
- **Formatting**: rustfmt pre-commit hook
- **Linting**: No clippy.toml; standard Rust warnings

## Code Style

- No rustfmt.toml - uses Rust defaults
- Comments use `//` for inline explanations of non-obvious logic
- No rustdoc `///` comments yet

---

*Convention analysis: 2026-02-05*
