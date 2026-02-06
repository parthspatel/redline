# Directory Structure

**Analysis Date:** 2026-02-05

## Layout

```
redline/
├── Cargo.toml                          # Workspace root (resolver = "3")
├── Cargo.lock                          # Locked dependencies
├── rust-toolchain.toml                 # Nightly toolchain config
├── devenv.nix                          # Dev environment (Nix)
├── devenv.yaml                         # Devenv inputs & secrets
├── devenv.lock                         # Devenv lock file
├── .envrc                              # direnv integration
├── .gitignore
│
├── crates/
│   └── core/                           # redline-core crate (v0.1.0)
│       ├── Cargo.toml                  # Crate manifest
│       ├── src/
│       │   ├── lib.rs                  # Crate root (exports `mod utils`)
│       │   └── utils/
│       │       ├── mod.rs              # Module declarations (bert_v1, bert_v2)
│       │       ├── bert_v1.rs          # BERT embedding functions (226 lines)
│       │       └── bert_v2.rs          # Empty placeholder
│       └── examples/
│           └── hello_world.rs          # Trivial example
│
├── docs/
│   └── design/
│       ├── DESIGN.md                   # v1 design document (comprehensive)
│       └── TECHNICAL_SPEC.md           # Technical specification
│
├── flakes/
│   └── libtorch/                       # Nix flake for libtorch (GPU support)
│       ├── flake.nix
│       ├── flake.lock
│       ├── README.md
│       └── update_hashes.sh
│
├── .claude/
│   └── settings.local.json             # Claude Code local settings
│
├── .planning/                          # GSD planning docs (this mapping)
│   └── codebase/
│
└── .venv/                              # Python virtual environment (devenv-managed)
```

## Key Locations

| What | Where |
|------|-------|
| Workspace manifest | `Cargo.toml` |
| Core library | `crates/core/src/` |
| BERT utilities | `crates/core/src/utils/bert_v1.rs` |
| Design docs | `docs/design/` |
| Dev environment | `devenv.nix` |
| Toolchain config | `rust-toolchain.toml` |
| Examples | `crates/core/examples/` |

## Naming Conventions

- **Crate naming**: `redline-core` (Cargo), `redline_core` (Rust module)
- **Files**: snake_case (`bert_v1.rs`, `hello_world.rs`)
- **Modules**: `mod.rs` pattern for directories
- **Versioned files**: `bert_v1.rs`, `bert_v2.rs` (implementation iterations)

## Notable

- `bert_v2.rs` exists but is empty (placeholder for next iteration)
- `lib.rs` only has one line: `mod utils;`
- The design document (`docs/design/DESIGN.md`) is far more detailed than the actual code
- Workspace is set up for multiple crates but currently only has `core`

---

*Structure analysis: 2026-02-05*
