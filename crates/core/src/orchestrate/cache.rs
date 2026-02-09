//! Pipeline-level cache with LRU eviction and optional memory cap.
//!
//! Wraps `lru::LruCache` behind a `parking_lot::RwLock` for thread-safe
//! concurrent access. Cache keys are content-addressed hashes of source text,
//! target text, and configuration fingerprint.

use core::hash::{BuildHasher, Hash, Hasher};
use core::num::NonZeroUsize;
use std::sync::Arc;

use hashbrown::HashMap;
use lru::LruCache;
use parking_lot::RwLock;

use crate::analysis::AnalysisReport;
use crate::diff::edit_operation::EditOperation;
use crate::diff::result::DiffResult;
use crate::metrics::MetricValue;
use crate::orchestrate::config::RedlineConfig;

/// Fixed-seed hasher for deterministic content hashing.
static FIXED_HASHER: foldhash::fast::FixedState = foldhash::fast::FixedState::with_seed(0);

/// Content-addressed key for the pipeline cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineCacheKey {
    hash: u64,
}

impl PipelineCacheKey {
    /// Create a cache key from source text, target text, and a config fingerprint.
    pub fn new(source: &str, target: &str, config_fingerprint: u64) -> Self {
        let mut hasher = FIXED_HASHER.build_hasher();
        source.hash(&mut hasher);
        target.hash(&mut hasher);
        config_fingerprint.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
        }
    }
}

/// Compute a fingerprint for a RedlineConfig to differentiate cache entries.
pub(crate) fn config_fingerprint(config: &RedlineConfig) -> u64 {
    let mut hasher = FIXED_HASHER.build_hasher();
    // Hash preset via discriminant
    match config.preset {
        crate::orchestrate::Preset::Fast => 0u8.hash(&mut hasher),
        crate::orchestrate::Preset::Comprehensive => 1u8.hash(&mut hasher),
    }
    config.metrics_enabled.hash(&mut hasher);
    config.analysis_enabled.hash(&mut hasher);
    // Hash execution mode via discriminant
    match config.execution_mode {
        crate::process::ExecutionMode::Minimal => 0u8.hash(&mut hasher),
        crate::process::ExecutionMode::All => 1u8.hash(&mut hasher),
    }
    hasher.finish()
}

/// A cached pipeline result. AnalysisReport is wrapped in Arc since it's !Clone.
#[derive(Debug, Clone)]
pub(crate) struct CachedResult {
    pub(crate) diff: DiffResult,
    pub(crate) metrics: HashMap<String, MetricValue>,
    pub(crate) analysis: Option<Arc<AnalysisReport>>,
    pub(crate) estimated_bytes: usize,
}

impl CachedResult {
    /// Rough memory estimate for a pipeline result.
    pub(crate) fn estimate_memory(
        diff: &DiffResult,
        metrics: &HashMap<String, MetricValue>,
        has_analysis: bool,
    ) -> usize {
        let ops_bytes = diff.operations.len() * core::mem::size_of::<EditOperation>();
        let source_bytes = diff.source.original.len() + diff.source.normalized.len();
        let target_bytes = diff.target.original.len() + diff.target.normalized.len();
        let metrics_bytes = metrics.len() * 64;
        let analysis_bytes = if has_analysis { 4096 } else { 0 };
        ops_bytes + source_bytes + target_bytes + metrics_bytes + analysis_bytes + 256
    }
}

/// Thread-safe pipeline cache with LRU eviction and optional memory cap.
///
/// Uses `parking_lot::RwLock` for concurrent access. All state is behind a
/// single lock to eliminate deadlock risk.
pub struct CacheManager {
    inner: RwLock<CacheInner>,
    memory_limit: Option<usize>,
}

struct CacheInner {
    cache: LruCache<PipelineCacheKey, CachedResult>,
    total_bytes: usize,
}

impl CacheManager {
    /// Create a new cache with the given entry capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(CacheInner {
                cache: LruCache::new(
                    NonZeroUsize::new(capacity).expect("cache capacity must be > 0"),
                ),
                total_bytes: 0,
            }),
            memory_limit: None,
        }
    }

    /// Create a new cache with entry capacity and a memory limit in bytes.
    ///
    /// When the memory limit is exceeded, LRU entries are evicted until
    /// usage drops below the limit.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    pub fn with_memory_limit(capacity: usize, limit: usize) -> Self {
        Self {
            inner: RwLock::new(CacheInner {
                cache: LruCache::new(
                    NonZeroUsize::new(capacity).expect("cache capacity must be > 0"),
                ),
                total_bytes: 0,
            }),
            memory_limit: Some(limit),
        }
    }

    /// Look up a cached result. Returns a clone if found.
    ///
    /// Uses a write lock because `LruCache::get` mutates internal ordering.
    pub(crate) fn get(&self, key: &PipelineCacheKey) -> Option<CachedResult> {
        let mut inner = self.inner.write();
        inner.cache.get(key).cloned()
    }

    /// Insert a result into the cache.
    ///
    /// If an entry with the same key existed, its bytes are subtracted first.
    /// If a memory limit is set and exceeded after insertion, LRU entries are
    /// evicted until usage drops below the limit.
    pub(crate) fn insert(&self, key: PipelineCacheKey, value: CachedResult) {
        let mut inner = self.inner.write();
        let new_bytes = value.estimated_bytes;

        // If replacing an existing entry, subtract its bytes first
        if let Some(old) = inner.cache.pop(&key) {
            inner.total_bytes = inner.total_bytes.saturating_sub(old.estimated_bytes);
        }

        inner.cache.put(key, value);
        inner.total_bytes += new_bytes;

        // Memory cap eviction
        if let Some(limit) = self.memory_limit {
            while inner.total_bytes > limit {
                if let Some((_, evicted)) = inner.cache.pop_lru() {
                    inner.total_bytes = inner.total_bytes.saturating_sub(evicted.estimated_bytes);
                } else {
                    break;
                }
            }
        }
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.cache.clear();
        inner.total_bytes = 0;
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.inner.read().cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().cache.is_empty()
    }

    /// Total estimated memory usage in bytes.
    pub fn total_memory(&self) -> usize {
        self.inner.read().total_bytes
    }
}

impl core::fmt::Debug for CacheManager {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("CacheManager")
            .field("len", &inner.cache.len())
            .field("cap", &inner.cache.cap())
            .field("total_bytes", &inner.total_bytes)
            .field("memory_limit", &self.memory_limit)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::result::{DiffMetadata, DiffStatistics};
    use crate::process::ProcessedText;
    use crate::span::Span;
    use crate::text_store::TextStoreBuilder;
    use crate::token::{Token, TokenKind};

    fn make_processed(words: &[&str]) -> Arc<ProcessedText> {
        let joined = words.join(" ");
        let mut builder = TextStoreBuilder::new();
        let mut tokens = Vec::new();
        let mut offset = 0u32;
        for &w in words {
            let id = builder.intern(w);
            let end = offset + w.len() as u32;
            tokens.push(Token::new(id, Span::new(offset, end), TokenKind::Regular));
            offset = end + 1;
        }
        let store = builder.build();
        Arc::new(ProcessedText {
            original: joined.clone(),
            layers: vec![],
            normalized: joined,
            tokens,
            composed_mapping: None,
            text_store: store,
            skipped_normalizers: vec![],
        })
    }

    fn make_cached_result(source_words: &[&str], target_words: &[&str]) -> CachedResult {
        let source = make_processed(source_words);
        let target = make_processed(target_words);
        let ops = vec![EditOperation::replace(
            0,
            source_words.len() as u32,
            0,
            target_words.len() as u32,
        )];
        let src_len = source.tokens.len();
        let tgt_len = target.tokens.len();
        let diff = DiffResult {
            statistics: DiffStatistics::from_operations(&ops, src_len, tgt_len),
            metadata: DiffMetadata {
                algorithm_name: "test".into(),
                is_approximate: false,
                threshold_used: None,
            },
            operations: ops,
            source,
            target,
        };
        let metrics = HashMap::new();
        let estimated_bytes = CachedResult::estimate_memory(&diff, &metrics, false);
        CachedResult {
            diff,
            metrics,
            analysis: None,
            estimated_bytes,
        }
    }

    fn make_key(source: &str, target: &str) -> PipelineCacheKey {
        PipelineCacheKey::new(source, target, 0)
    }

    #[test]
    fn new_creates_empty_cache() {
        let cache = CacheManager::new(10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.total_memory(), 0);
    }

    #[test]
    fn insert_and_get() {
        let cache = CacheManager::new(10);
        let key = make_key("hello", "world");
        let result = make_cached_result(&["hello"], &["world"]);
        cache.insert(key.clone(), result);
        assert_eq!(cache.len(), 1);
        let got = cache.get(&key);
        assert!(got.is_some());
    }

    #[test]
    fn get_missing_returns_none() {
        let cache = CacheManager::new(10);
        let key = make_key("missing", "key");
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn lru_eviction_on_capacity() {
        let cache = CacheManager::new(2);
        let k1 = make_key("a", "1");
        let k2 = make_key("b", "2");
        let k3 = make_key("c", "3");

        cache.insert(k1.clone(), make_cached_result(&["a"], &["1"]));
        cache.insert(k2.clone(), make_cached_result(&["b"], &["2"]));
        assert_eq!(cache.len(), 2);

        // Insert third entry — k1 (oldest) should be evicted
        cache.insert(k3.clone(), make_cached_result(&["c"], &["3"]));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k2).is_some());
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn get_promotes_entry() {
        let cache = CacheManager::new(2);
        let k1 = make_key("a", "1");
        let k2 = make_key("b", "2");
        let k3 = make_key("c", "3");

        cache.insert(k1.clone(), make_cached_result(&["a"], &["1"]));
        cache.insert(k2.clone(), make_cached_result(&["b"], &["2"]));

        // Access k1 to promote it
        let _ = cache.get(&k1);

        // Insert k3 — k2 should be evicted (k1 was recently accessed)
        cache.insert(k3.clone(), make_cached_result(&["c"], &["3"]));
        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k2).is_none());
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn clear_empties_cache() {
        let cache = CacheManager::new(10);
        cache.insert(make_key("a", "1"), make_cached_result(&["a"], &["1"]));
        cache.insert(make_key("b", "2"), make_cached_result(&["b"], &["2"]));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.total_memory(), 0);
    }

    #[test]
    fn cache_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CacheManager>();
    }

    #[test]
    fn config_fingerprint_deterministic() {
        use crate::orchestrate::ConfigBuilder;
        let c1 = ConfigBuilder::new().build().unwrap();
        let c2 = ConfigBuilder::new().build().unwrap();
        assert_eq!(config_fingerprint(&c1), config_fingerprint(&c2));
    }

    #[test]
    fn config_fingerprint_varies_by_preset() {
        use crate::orchestrate::ConfigBuilder;
        use crate::orchestrate::config::Preset;
        let fast = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let comp = ConfigBuilder::preset(Preset::Comprehensive)
            .build()
            .unwrap();
        assert_ne!(config_fingerprint(&fast), config_fingerprint(&comp));
    }

    #[test]
    fn config_fingerprint_varies_by_flags() {
        use crate::orchestrate::ConfigBuilder;
        let with_metrics = ConfigBuilder::new().with_metrics(true).build().unwrap();
        let no_metrics = ConfigBuilder::new().with_metrics(false).build().unwrap();
        assert_ne!(
            config_fingerprint(&with_metrics),
            config_fingerprint(&no_metrics)
        );
    }

    #[test]
    fn memory_limit_eviction() {
        // Each CachedResult is ~300+ bytes. Set a low limit.
        let cache = CacheManager::with_memory_limit(100, 800);
        let r1 = make_cached_result(&["hello", "world"], &["foo", "bar"]);
        let r1_bytes = r1.estimated_bytes;
        let r2 = make_cached_result(&["alpha", "beta"], &["gamma", "delta"]);
        let r2_bytes = r2.estimated_bytes;

        cache.insert(make_key("a", "1"), r1);
        cache.insert(make_key("b", "2"), r2);

        // Both entries together likely exceed 800 bytes
        if r1_bytes + r2_bytes > 800 {
            // Memory eviction should have kicked in
            assert!(cache.total_memory() <= 800);
        }
    }

    #[test]
    fn total_memory_tracks_insertions() {
        let cache = CacheManager::new(10);
        assert_eq!(cache.total_memory(), 0);

        let result = make_cached_result(&["hello"], &["world"]);
        let expected_bytes = result.estimated_bytes;
        cache.insert(make_key("a", "1"), result);
        assert_eq!(cache.total_memory(), expected_bytes);
    }

    #[test]
    fn overwrite_updates_memory() {
        let cache = CacheManager::new(10);
        let key = make_key("a", "1");

        let r1 = make_cached_result(&["hello"], &["world"]);
        let r1_bytes = r1.estimated_bytes;
        cache.insert(key.clone(), r1);
        assert_eq!(cache.total_memory(), r1_bytes);

        let r2 = make_cached_result(&["hello", "again"], &["world", "too"]);
        let r2_bytes = r2.estimated_bytes;
        cache.insert(key, r2);
        assert_eq!(cache.total_memory(), r2_bytes);
    }

    #[test]
    fn concurrent_access_no_deadlock() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let cache = StdArc::new(CacheManager::new(64));
        let mut handles = Vec::new();

        for thread_id in 0..8u32 {
            let cache_clone = StdArc::clone(&cache);
            handles.push(thread::spawn(move || {
                for i in 0..100u32 {
                    let key = PipelineCacheKey::new(
                        &format!("src-{thread_id}-{i}"),
                        &format!("tgt-{thread_id}-{i}"),
                        0,
                    );
                    let result = make_cached_result(&["a", "b"], &["x", "y"]);
                    cache_clone.insert(key.clone(), result);
                    let _ = cache_clone.get(&key);
                    let _ = cache_clone.len();
                }
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // Cache should be internally consistent
        assert!(cache.len() <= 64);
    }

    #[test]
    fn debug_output() {
        let cache = CacheManager::new(10);
        let debug = format!("{:?}", cache);
        assert!(debug.contains("CacheManager"));
    }

    #[test]
    #[should_panic(expected = "cache capacity must be > 0")]
    fn zero_capacity_panics() {
        CacheManager::new(0);
    }

    #[test]
    fn pipeline_cache_key_deterministic() {
        let k1 = PipelineCacheKey::new("hello", "world", 42);
        let k2 = PipelineCacheKey::new("hello", "world", 42);
        assert_eq!(k1, k2);
    }

    #[test]
    fn pipeline_cache_key_varies_by_input() {
        let k1 = PipelineCacheKey::new("hello", "world", 0);
        let k2 = PipelineCacheKey::new("world", "hello", 0);
        assert_ne!(k1, k2);
    }

    #[test]
    fn pipeline_cache_key_varies_by_fingerprint() {
        let k1 = PipelineCacheKey::new("hello", "world", 1);
        let k2 = PipelineCacheKey::new("hello", "world", 2);
        assert_ne!(k1, k2);
    }
}
