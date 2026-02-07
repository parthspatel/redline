# Phase 2: Text Processing - Context

**Gathered:** 2026-02-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Raw text is normalized and tokenized into token streams with full position traceability back to original text. Normalizer trait + 6 built-in normalizers, Tokenizer trait + 4 built-in tokenizers, TextProcessor orchestrator, ProcessedText output type with intermediate layers, CharMapping preservation through pipeline, Unicode correctness.

</domain>

<decisions>
## Implementation Decisions

### Normalizer behavior
- Aggressiveness is Claude's discretion — no constraint on how aggressive each normalizer is
- The hard contract: every normalizer MUST produce a valid CharMapping that round-trips positions correctly
- Traceability is the invariant, not preservation of original form

### Tokenizer boundary rules
- WordTokenizer splits on whitespace only — no punctuation splitting, no contraction handling
- Two NGram tokenizer variants:
  - WordNGramTokenizer — n-grams over words
  - CharNGramTokenizer — n-grams over characters
- NGram tokenizers are **composable**: configured with a base tokenizer that runs first, then n-grams are computed over its output
- This means NGram is not a standalone tokenizer — it wraps another

### Pipeline error handling
- All undefined/unexpected behaviors are FATAL
- Every failure is a typed error — no silent fallbacks, no skip-and-continue
- If a normalizer produces empty output, invalid mapping, or any unexpected state: return a typed error
- Caller is responsible for handling errors explicitly (this is Rust)

### ProcessedText intermediate layers
- ProcessedText MUST store intermediate normalization layers, not just final output
- Each layer stores: the text after that normalization step + CharMapping to previous layer
- Different analyzers and diff algorithms operate on specific layers
- Memory optimization: only store layers that are actually needed by configured analyzers — unused normalizer outputs are not retained
- This is the default "minimal" mode; user can explicitly request "all" layers

### Execution graph / dependency system
- Analyzers declare at configuration time (before execution) the minimally required processing steps
- Dependencies are declared by **named layers** (e.g., `"lowercase"`, `"whitespace"`) — not numeric indices
- The system resolves names to layer indices at configuration time
- Referencing a non-existent layer name is a fatal typed error at config time
- System runs the minimal union of all required normalization steps by default
- Steps that will NOT run are logged (not silently skipped)
- User can explicitly control "minimal" (default) or "all" execution mode

### Diff layer selection
- Diff algorithms operate based on the dependency graph
- If multiple normalizations are needed, diffing happens at those levels
- The diff layer is determined by what the requesting analyzer/consumer declared as dependencies

### Claude's Discretion
- How aggressive each built-in normalizer is (Lowercase, Whitespace, Unicode, Diacritics, Punctuation, Digits)
- Unicode normalization form choice (NFC vs NFD vs NFKC)
- How NGram tokenizer composes its token stream internally (new StringIds for n-grams vs index references)
- Exact layer storage format and composed CharMapping caching strategy
- CJK/emoji/RTL handling details within tokenizers

</decisions>

<specifics>
## Specific Ideas

- Named layer addressing for analyzer dependencies — type-safe handles resolved at config time
- "Minimal by default" execution: only compute what's needed, log what's skipped
- NGram tokenizer takes a base tokenizer as configuration — composable pipeline design
- Layer storage naturally bounded by minimal execution (only store what's needed)

</specifics>

<deferred>
## Deferred Ideas

- Execution graph with full DAG resolution is partially a Phase 5 (Analysis Framework) concern — the dependency declaration mechanism is defined here, but the full ExecutionPlanner with topological sort lives in Phase 5
- Parallel normalizer execution for independent normalization chains — Phase 7 (Async)

</deferred>

---

*Phase: 02-text-processing*
*Context gathered: 2026-02-06*
