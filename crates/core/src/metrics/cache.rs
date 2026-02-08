//! Content-addressed caching for metric results with LRU eviction.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use core::hash::{BuildHasher, Hash, Hasher};
use core::num::NonZeroUsize;

use lru::LruCache;

use crate::process::ProcessedText;

use super::MetricValue;

/// Content hash of a ProcessedText, computed via FoldHash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(u64);

/// Fixed-seed hasher for deterministic content hashing across calls.
/// Uses FoldHash with seed 0 so the same content always produces the same hash.
static FIXED_HASHER: foldhash::fast::FixedState = foldhash::fast::FixedState::with_seed(0);

impl ContentHash {
    /// Hash a single ProcessedText.
    pub fn of_single(text: &ProcessedText) -> Self {
        let mut hasher = FIXED_HASHER.build_hasher();
        text.original.hash(&mut hasher);
        text.normalized.hash(&mut hasher);
        text.tokens.len().hash(&mut hasher);
        ContentHash(hasher.finish())
    }

    /// Hash a pair of ProcessedTexts.
    pub fn of_pair(source: &ProcessedText, target: &ProcessedText) -> Self {
        let mut hasher = FIXED_HASHER.build_hasher();
        source.original.hash(&mut hasher);
        source.normalized.hash(&mut hasher);
        source.tokens.len().hash(&mut hasher);
        target.original.hash(&mut hasher);
        target.normalized.hash(&mut hasher);
        target.tokens.len().hash(&mut hasher);
        ContentHash(hasher.finish())
    }

    /// Get the raw u64 hash value.
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Cache key combining content hash and metric ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub content_hash: ContentHash,
    pub metric_id: String,
}

impl CacheKey {
    /// Create a new cache key.
    pub fn new(content_hash: ContentHash, metric_id: String) -> Self {
        Self {
            content_hash,
            metric_id,
        }
    }
}

/// LRU cache for metric results, keyed by content hash + metric ID.
pub struct MetricCache {
    inner: LruCache<CacheKey, MetricValue>,
}

impl core::fmt::Debug for MetricCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetricCache")
            .field("len", &self.inner.len())
            .field("cap", &self.inner.cap())
            .finish()
    }
}

impl MetricCache {
    /// Create a new cache with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: LruCache::new(NonZeroUsize::new(capacity).expect("cache capacity must be > 0")),
        }
    }

    /// Look up a cached value (updates LRU ordering).
    pub fn get(&mut self, key: &CacheKey) -> Option<MetricValue> {
        self.inner.get(key).cloned()
    }

    /// Insert a value into the cache.
    pub fn put(&mut self, key: CacheKey, value: MetricValue) {
        self.inner.put(key, value);
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash(val: u64) -> ContentHash {
        ContentHash(val)
    }

    #[test]
    fn cache_put_and_get() {
        let mut cache = MetricCache::new(10);
        let key = CacheKey::new(make_hash(1), "word_count".into());
        cache.put(key.clone(), MetricValue::Integer(42));
        assert_eq!(cache.get(&key), Some(MetricValue::Integer(42)));
    }

    #[test]
    fn cache_miss_returns_none() {
        let mut cache = MetricCache::new(10);
        let key = CacheKey::new(make_hash(1), "missing".into());
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn cache_lru_eviction() {
        let mut cache = MetricCache::new(2);
        let k1 = CacheKey::new(make_hash(1), "a".into());
        let k2 = CacheKey::new(make_hash(2), "b".into());
        let k3 = CacheKey::new(make_hash(3), "c".into());

        cache.put(k1.clone(), MetricValue::Integer(1));
        cache.put(k2.clone(), MetricValue::Integer(2));
        assert_eq!(cache.len(), 2);

        // This should evict k1 (least recently used)
        cache.put(k3.clone(), MetricValue::Integer(3));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&k1), None);
        assert_eq!(cache.get(&k2), Some(MetricValue::Integer(2)));
        assert_eq!(cache.get(&k3), Some(MetricValue::Integer(3)));
    }

    #[test]
    fn cache_clear() {
        let mut cache = MetricCache::new(10);
        cache.put(
            CacheKey::new(make_hash(1), "a".into()),
            MetricValue::Integer(1),
        );
        cache.put(
            CacheKey::new(make_hash(2), "b".into()),
            MetricValue::Integer(2),
        );
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn content_hash_same_value() {
        let h1 = make_hash(42);
        let h2 = make_hash(42);
        assert_eq!(h1, h2);
        assert_eq!(h1.value(), 42);
    }

    #[test]
    fn content_hash_different_values() {
        let h1 = make_hash(1);
        let h2 = make_hash(2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn cache_key_equality() {
        let k1 = CacheKey::new(make_hash(1), "a".into());
        let k2 = CacheKey::new(make_hash(1), "a".into());
        let k3 = CacheKey::new(make_hash(1), "b".into());
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn cache_overwrites_existing_key() {
        let mut cache = MetricCache::new(10);
        let key = CacheKey::new(make_hash(1), "a".into());
        cache.put(key.clone(), MetricValue::Integer(1));
        cache.put(key.clone(), MetricValue::Integer(2));
        assert_eq!(cache.get(&key), Some(MetricValue::Integer(2)));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_debug_format() {
        let cache = MetricCache::new(10);
        let dbg = format!("{:?}", cache);
        assert!(dbg.contains("MetricCache"));
    }

    #[test]
    #[should_panic(expected = "cache capacity must be > 0")]
    fn cache_zero_capacity_panics() {
        MetricCache::new(0);
    }
}
