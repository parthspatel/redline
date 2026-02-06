# Pitfalls Research

**Domain:** Text diffing and analysis framework (Rust)
**Project:** Redline Core v1.0
**Researched:** 2026-02-05
**Confidence:** HIGH

## Critical Pitfalls

These will cause compilation failures, unsoundness, or fundamental architectural problems if not addressed.

---

### Pitfall 1: Token Cannot Derive Copy With SmallVec Field

**What goes wrong:**
The spec declares `#[derive(Copy, Clone)]` on Token but includes `original_spans: SmallVec<[Span; 1]>`. SmallVec does NOT implement Copy because it can heap-allocate. This will not compile.

**How to avoid:**
Replace `SmallVec<[Span; 1]>` with `original_span: Span` (singular). Most tokens map 1:1. For rare multi-span cases, store a flag and look up the full mapping in CharMapping.

**Phase to address:** Foundation (Phase 1). **Blocking decision.**

**Confidence:** HIGH (verified: SmallVec docs)

---

### Pitfall 2: TextStore 'static Lifetime Is Unsound

**What goes wrong:**
The spec stores arena-allocated strings as `&'static str` via unsafe transmutation. If the TextStore is dropped while StringIds are still in use, those StringIds point to freed memory (use-after-free). The bumpalo `Bump` is `!Sync`, and `unsafe impl Sync for TextStore` would be unsound.

**How to avoid:**
1. Use `Arc<TextStore>` ownership chain: ProcessedText owns Arc<TextStore>, keeping it alive.
2. Separate TextStore into build phase (mutable, single-threaded) and read phase (immutable, shareable via Arc).
3. Never use `unsafe impl Sync` -- use proper phase separation instead.

**Phase to address:** Foundation (Phase 1). Run Miri on all TextStore tests.

**Confidence:** HIGH (verified: bumpalo docs, Rust ownership rules)

---

### Pitfall 3: Bump Allocator Is !Sync

**What goes wrong:**
`bumpalo::Bump` is `!Sync` (cannot be shared between threads). The spec's `TextStore` wraps a `Bump`. If TextStore needs to be shared for concurrent reading (e.g., parallel analyzers reading token text), wrapping in `Arc<TextStore>` works for reads but `Bump::alloc_str()` requires `&self` -- and `Bump` isn't Sync.

**How to avoid:**
Two-phase design: interning phase (single-threaded, &mut self) followed by read-only phase (Arc<TextStore>, immutable, safe to share). After processing completes, no more interning happens.

**Phase to address:** Foundation (Phase 1).

**Confidence:** HIGH (verified: bumpalo docs)

---

### Pitfall 4: EditOperation Lifetime Prevents Caching/Storage

**What goes wrong:**
`EditOperation<'a>` borrows `&'a [Token]` from ProcessedText. This lifetime parameter means DiffResult<'a> cannot be stored in a HashMap, cached, or sent across threads without the source ProcessedText being pinned.

**How to avoid:**
Use index ranges instead of borrowed slices: `original_range: Range<usize>` and `modified_range: Range<usize>`. The consumer looks up tokens via the Arc<ProcessedText>. DiffResult becomes `'static` and cacheable.

**Phase to address:** Foundation (Phase 1) for design, Phase 3 for implementation.

**Confidence:** HIGH

---

### Pitfall 5: PyO3 API Is Stale (0.20 vs 0.28)

**What goes wrong:**
The spec targets PyO3 0.20 which uses the GIL Refs API (`&PyAny`, `&PyDict`). This API was deprecated in 0.21 and removed in 0.23. All code examples in the spec's Python bindings section are wrong.

**How to avoid:**
Target PyO3 0.28. Rewrite all Python binding code to use `Bound<'py, T>` API. Replace `pyo3-asyncio` with `pyo3-async-runtimes`.

**Phase to address:** Pre-implementation spec update. Python Bindings phase.

**Confidence:** HIGH (verified: PyO3 migration guide)

---

### Pitfall 6: GIL Deadlock in Async Python Bridge

**What goes wrong:**
The spec's async Python code spawns tokio tasks that call `Python::with_gil()`. In multi-threaded tokio, the GIL can be held by thread A while thread B (running a tokio task) tries to acquire it, causing deadlock.

**How to avoid:**
1. Use `py.allow_threads()` before entering tokio runtime.
2. Only acquire GIL on the thread that owns the Python context.
3. Use `pyo3-async-runtimes::tokio::future_into_py()` which handles GIL correctly.
4. Test with multi-threaded tokio runtime from day one.

**Phase to address:** Python Bindings phase.

**Confidence:** HIGH (verified: PyO3 discussions)

---

### Pitfall 7: WASM Feature Flags Must Be Phase 1

**What goes wrong:**
Tokio, PyO3, and std::thread all panic or fail on `wasm32-unknown-unknown`. If feature flag architecture is deferred, retrofitting `#[cfg]` gates to all tokio/PyO3 code is expensive and error-prone.

**How to avoid:**
1. Add `cargo check --target wasm32-unknown-unknown --no-default-features --features wasm` to CI from Phase 1.
2. Feature-gate all tokio usage behind `#[cfg(feature = "async")]`.
3. Feature-gate all PyO3 usage behind `#[cfg(feature = "python")]`.
4. Use `compile_error!` macro to prevent impossible feature combinations.

**Phase to address:** Foundation (Phase 1) CI setup.

**Confidence:** HIGH (verified: rustc WASM target docs, tokio issues)

---

### Pitfall 8: Myers O(N^2) Performance Cliff

**What goes wrong:**
Myers algorithm is O(ND) where D is the edit distance. For dissimilar texts (D approaches N), this degrades to O(N^2). A 10K token comparison of completely different texts can take >10 seconds.

**How to avoid:**
1. Strip common prefix and suffix before running Myers (reduces N significantly).
2. Set a maximum D threshold (e.g., D_max = 2*N/3) and bail out to "everything changed" if exceeded.
3. Offer Histogram diff as the default for larger inputs (more consistent performance).
4. Benchmark with adversarial inputs (completely dissimilar 10K token texts).

**Phase to address:** Diff Computation (Phase 3).

**Confidence:** HIGH (verified: Myers 1986 paper)

---

## Major Pitfalls

These cause incorrect behavior, data corruption, or significant performance problems.

---

### Pitfall 9: Unicode CharMapping Breaks for Multi-Codepoint Graphemes

**What goes wrong:**
Unicode normalization can change the number of codepoints (e.g., NFC: e + combining accent -> single codepoint). CharMapping must handle 1:N and N:1 position mappings correctly, including for emoji (multi-codepoint sequences), combining characters, and ligatures.

**How to avoid:**
1. Use `unicode-normalization` crate for normalization, not hand-rolled code.
2. Use `unicode-segmentation` crate for grapheme cluster awareness in tokenization.
3. Span positions must be byte offsets, not character indices or grapheme indices.
4. Build a comprehensive CharMapping test suite with: accented characters, CJK, RTL text, emoji, zero-width joiners, combining characters.
5. Property-test CharMapping composition: for any sequence of normalizers, `composed_mapping.map_to_original(pos)` must round-trip correctly.

**Phase to address:** Foundation (Phase 1) for Span byte-offset convention. Text Processing (Phase 2) for normalizer CharMapping correctness.

**Confidence:** HIGH (verified: unicode-normalization docs, Unicode Standard Annex #29)

---

### Pitfall 10: RwLock Double-Checked Locking Causes Deadlock in CacheManager

**What goes wrong:**
The spec's CacheManager uses a read-then-write double-checked locking pattern with `std::sync::RwLock`. If the read lock guard's lifetime extends past the write lock attempt on the same thread, this deadlocks.

**How to avoid:**
1. Use `parking_lot::RwLock` instead of `std::sync::RwLock` (fair scheduling, prevents write starvation).
2. Isolate lock acquisition into separate functions that return owned data.
3. Never hold two locks simultaneously. If you must, always acquire in the same order.

**Phase to address:** Orchestration & Configuration (Phase 6).

**Confidence:** HIGH (verified: std::sync::RwLock docs, rust-lang issue #37612)

---

### Pitfall 11: Python Plugin Callbacks Cross the GIL Boundary Per Call

**What goes wrong:**
Python plugin adapters (PyNormalizer, PyTokenizer, PyAnalyzerPlugin) call Python functions from Rust. Each call requires acquiring the GIL, converting types, calling Python, converting back, and releasing GIL. For a tokenizer processing 10K tokens, that is 10K GIL acquisitions. Overhead can be 100-1000x slower than native Rust.

**How to avoid:**
1. Batch the Python interface -- pass entire texts or batches, not one token at a time.
2. Use `py.allow_threads()` for Rust computation between Python calls.
3. Cache Python plugin results aggressively.
4. Document expected performance: Python plugins will be 10-100x slower than Rust equivalents.

**Phase to address:** Python Bindings phase.

**Confidence:** MEDIUM (verified: PyO3 performance guide; specific overhead numbers are estimates)

---

### Pitfall 12: Analyzer Dependency Graph Allows Runtime Circular Dependencies

**What goes wrong:**
Static validation happens at `ConfigBuilder::build()` time, but `AnalysisCoordinator::register()` adds plugins after construction. A Python user could register analyzer A depending on B, then register B depending on A.

**How to avoid:**
1. Validate dependencies on every `register()` call, not just at build time.
2. Use topological sort for execution ordering. If toposort fails (cycle detected), return an error, not a panic.
3. Implement a maximum depth limit for dependency resolution.

**Phase to address:** Analysis Framework (Phase 5).

**Confidence:** MEDIUM

---

## Minor Pitfalls

### Pitfall 13: TokenKind::Custom(&'static str) Prevents Dynamic Token Kinds

The spec defines `TokenKind::Custom(&'static str)`, which means custom token kinds must be compile-time string literals. Python plugins cannot create custom token kinds at runtime.

**Fix:** Use `TokenKind::Custom(StringId)` -- store the kind name in the TextStore.

**Phase:** Foundation (Phase 1).

---

### Pitfall 14: ProcessedText Stores Original and Normalized as Owned Strings

The spec's `ProcessedText` has `pub original: String` and `pub normalized: String` -- heap-allocated strings duplicating what's already in the arena.

**Fix:** Store `original_id: StringId` and `normalized_id: StringId` in ProcessedText, with the actual strings in TextStore.

**Phase:** Text Processing (Phase 2).

---

### Pitfall 15: NormalizationLayer Stores Full Text Copies Per Layer

Each `NormalizationLayer` stores `pub text: String` -- the full text at that normalization stage. For a 5-normalizer pipeline on a 100KB document, this is 500KB of intermediate strings.

**Fix:** Make layer text storage optional (behind a `trace` feature flag or configuration option). In production mode, only store the final normalized text and the composed CharMapping.

**Phase:** Text Processing (Phase 2).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `unsafe impl Sync for TextStore` | Avoids refactoring to separate build/read phases | Potential data races, UB, memory corruption | Never |
| `unwrap()` on RwLock results | Cleaner code | Panic on poisoned lock | Only in tests |
| Owned Strings in ProcessedText | Simpler initial implementation | 2x memory usage | Acceptable in Phase 2 prototype, must refactor by Phase 6 |
| Skip WASM CI checks | Faster CI | WASM breakage discovered late | Never |
| Single-threaded tokio in tests | Tests pass without deadlock investigation | Hides GIL deadlock bugs | Never for async Python tests |
| Skip Unicode test fixtures | Faster initial test writing | False confidence | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| bumpalo + hashbrown | Using `&'static str` keys without understanding soundness | Use safe public API with proper phase separation |
| PyO3 + tokio | Spawning tokio tasks that call Python::with_gil | Release GIL with `py.allow_threads()` before entering tokio |
| PyO3 + maturin | Forgetting `extension-module` feature on PyO3 | Always set `features = ["extension-module"]` |
| WASM + lru cache | `std::time::Instant` used for TTL | Use `web-time` crate that works on both native and WASM |
| tokio + WASM | Using `tokio::spawn` on wasm32 | Feature-gate all tokio usage; sync-only API for WASM |
| proptest + WASM | proptest does not compile for wasm32-unknown-unknown | Gate proptest behind `#[cfg(not(target_arch = "wasm32"))]` |
| serde + PyO3 | Trying to serialize PyO3 Bound objects directly | Convert to Rust types first, then serialize |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Myers on dissimilar 10K+ token texts | Single diff takes >10 seconds | Common prefix/suffix strip + D-threshold cutoff | >5K tokens with >50% edit distance |
| GIL acquisition per Python plugin call | Batch processing 100x slower than expected | Batch plugin interface | >100 texts in batch |
| RwLock write starvation on cache | Cache misses under concurrent reads | Use parking_lot::RwLock | >8 concurrent readers |
| LRU cache unbounded size | Memory grows without limit | Set max_size based on item count | >10K unique text pairs cached |
| CharMapping composition chain | Composing 5 layers is O(n*layers) | Eagerly compose after each normalizer | >5 normalizers on >100KB text |
| Full NormalizationLayer storage | 500KB+ per ProcessedText | Optional tracing; only store final text in production | Documents >50KB with >3 normalizers |

---

## "Looks Done But Isn't" Checklist

- [ ] **TextStore:** Works for ASCII but crashes on multi-byte UTF-8 -- verify with CJK/emoji text
- [ ] **CharMapping:** Works for 1:1 but composition fails for Collapsed/Expanded -- verify with Unicode normalization round-trip
- [ ] **Myers diff:** Passes basic tests but hangs on adversarial input -- verify with timeout and >5K dissimilar tokens
- [ ] **Python bindings:** Work synchronously but deadlock in async -- verify with multi-threaded tokio runtime
- [ ] **WASM build:** Compiles with default features but fails with `--features wasm` -- verify in CI from Phase 1
- [ ] **CacheManager:** Works single-threaded but deadlocks under concurrent load -- verify with stress tests
- [ ] **Plugin system:** Handles happy path but panics on circular dependencies -- verify with adversarial registration
- [ ] **Token Copy:** Compiles in isolation but fails when SmallVec added -- verify with `static_assertions::assert_impl_all!(Token: Copy)`
- [ ] **Feature flags:** `--all-features` works but individual combinations fail -- verify each in CI matrix
- [ ] **Error types:** All defined but Python exception mapping incomplete -- verify every Rust error maps to Python exception

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| #1 Token Copy + SmallVec | Phase 1: Foundation | `static_assertions::assert_impl_all!(Token: Copy)` compiles |
| #2 TextStore 'static unsound | Phase 1: Foundation | Miri passes on all TextStore tests |
| #3 Bump !Sync | Phase 1: Foundation | No `unsafe impl Sync` in codebase |
| #4 EditOperation lifetime | Phase 1 (design), Phase 3 (impl) | DiffResult can be stored in HashMap |
| #5 PyO3 version mismatch | Pre-implementation spec update | `cargo build --features python` succeeds with PyO3 0.28 |
| #6 GIL deadlock | Python Bindings phase | Async Python test with multi-threaded tokio passes |
| #7 WASM feature flags | Phase 1: Foundation CI | `cargo check --target wasm32-unknown-unknown --features wasm` in CI |
| #8 Myers O(N^2) | Phase 3: Diff Computation | Benchmark with dissimilar 10K token texts < 500ms |
| #9 Unicode CharMapping | Phase 1 (Span), Phase 2 (normalizers) | Property tests with Unicode text pass |
| #10 RwLock deadlock | Phase 6: Orchestration | Multi-threaded stress test passes |
| #11 Python plugin GIL overhead | Python Bindings phase | Batch plugin benchmark within 5x of Rust equivalent |
| #12 Circular analyzer deps | Phase 5: Analysis Framework | register() returns error on cycle |
| #13 TokenKind::Custom | Phase 1: Foundation | Python plugin can create custom TokenKind |
| #14 ProcessedText owned strings | Phase 2: Text Processing | ProcessedText uses StringId, not String |
| #15 NormalizationLayer memory | Phase 2: Text Processing | Layer text storage gated behind feature/config flag |

---

## Sources

- [bumpalo documentation -- Bump is !Sync](https://docs.rs/bumpalo/latest/bumpalo/struct.Bump.html)
- [SmallVec documentation -- does not implement Copy](https://docs.rs/smallvec/latest/smallvec/struct.SmallVec.html)
- [PyO3 Migration Guide -- GIL Refs removal, Bound API](https://pyo3.rs/main/migration)
- [PyO3 Discussion #3045 -- GIL deadlock with tokio::spawn](https://github.com/PyO3/pyo3/discussions/3045)
- [pyo3-async-runtimes -- replacement for pyo3-asyncio](https://github.com/PyO3/pyo3-async-runtimes)
- [wasm32-unknown-unknown target documentation](https://doc.rust-lang.org/rustc/platform-support/wasm32-unknown-unknown.html)
- [tokio issue #5418 -- WASM panic](https://github.com/tokio-rs/tokio/issues/5418)
- [Myers 1986 -- An O(ND) Difference Algorithm](http://www.xmailserver.org/diff2.pdf)
- [std::sync::RwLock -- deadlock with pattern matching](https://github.com/rust-lang/rust/issues/37612)
- [parking_lot::RwLock -- fair scheduling](https://docs.rs/parking_lot/latest/parking_lot/type.RwLock.html)

---
*Pitfalls research for: Redline Core v1.0*
*Researched: 2026-02-05*
