# External Integrations

**Analysis Date:** 2026-02-05

## APIs & External Services

### HuggingFace Model Hub

- **Purpose**: Download and cache pre-trained sentence transformer models
- **Client**: `hf-hub` crate 0.4.3, synchronous API (`hf_hub::api::sync::Api`)
- **Auth**: None (public models only, no API key required)
- **Models used**:
  - `sentence-transformers/all-MiniLM-L6-v2` - Lightweight embeddings
  - `sentence-transformers/all-mpnet-base-v2` - Higher-quality embeddings
- **Files downloaded per model**: `config.json`, `tokenizer.json`, `model.safetensors`
- **Location in code**: `crates/core/src/utils/bert_v1.rs`

## Data Storage

- **Databases**: None - library is stateless, operates on in-memory text
- **File storage**: Local filesystem only
- **Model cache**: HuggingFace Hub cache at `$HOME/.cache/huggingface/hub/` (default, overridable via `HF_HOME`)

## Authentication

- None - library requires no credentials
- HuggingFace Hub uses unauthenticated public API endpoints

## Monitoring & Logging

- No structured logging framework (no tracing, env_logger, slog)
- `println!` for informational output during model loading
- `console` crate available as optional feature for terminal formatting

## CI/CD

- **Repository**: GitHub (`https://github.com/parthspatel/redline`)
- **CI pipeline**: None configured (no GitHub Actions workflows)
- **Publishing**: Not yet published to crates.io
- **Local tooling**: clippy, rustfmt, commitizen (via devenv git hooks)

## Environment Variables

- No required env vars
- Optional: `HF_HOME` to override HuggingFace cache location

## Network Requirements

- Internet required for first model download from HuggingFace Hub (HTTPS)
- Fully offline-capable after models are cached locally
- No other external API calls, webhooks, or service dependencies

---

*Integration audit: 2026-02-05*
