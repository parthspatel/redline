//! Orchestration pipeline: configuration, caching, filtering, and result types.
//!
//! The [`Redline`] struct is the single entry point for the full pipeline:
//! text processing → diff computation → metrics → analysis.

use std::sync::Arc;

use hashbrown::HashMap;

pub mod cache;
pub mod config;
pub mod error;
pub mod filter;
pub mod result;

pub use cache::CacheManager;
pub use config::{ConfigBuilder, Preset, RedlineConfig};
pub use error::OrchestrateError;
pub use filter::{Filter, FilterContext};
pub use result::RedlineResult;

use crate::analysis::{
    AnalysisContext, AnalysisCoordinator, PluginRegistry, register_builtin_analyzers,
};
use crate::diff::DiffComputer;
use crate::metrics::{MetricInput, MetricRegistry, MetricsEngine, register_builtins};
use crate::process::TextProcessor;
use cache::{CachedResult, PipelineCacheKey, config_fingerprint};

/// The single entry point for the Redline text diffing & analysis pipeline.
///
/// Constructed from a [`RedlineConfig`] (via [`ConfigBuilder`]), `Redline` wires
/// together text processing, diff computation, metrics evaluation, and analysis
/// into a single `diff()` call.
///
/// # Example
///
/// ```ignore
/// use redline_core::{Redline, ConfigBuilder, Preset, Filter, EditKind};
///
/// let redline = Redline::new(ConfigBuilder::preset(Preset::Fast).build()?)?;
/// let result = redline.diff("old text", "new text")?;
/// let replaces: Vec<_> = result.filter(&Filter::kind(EditKind::Replace)).collect();
/// ```
pub struct Redline {
    processor: TextProcessor,
    diff_computer: DiffComputer,
    metrics_enabled: bool,
    analysis_enabled: bool,
    cache: Option<CacheManager>,
    config_fingerprint: u64,
    preset: Preset,
}

impl Redline {
    /// Create a new pipeline orchestrator from a validated configuration.
    ///
    /// Consumes the config, extracting normalizers, tokenizer, and algorithm
    /// to build the `TextProcessor` and `DiffComputer`.
    pub fn new(config: RedlineConfig) -> Result<Self, OrchestrateError> {
        // Compute fingerprint BEFORE destructuring (needs to read all fields)
        let fingerprint = config_fingerprint(&config);

        // Extract scalar flags before moving trait objects out
        let preset = config.preset;
        let metrics_enabled = config.metrics_enabled;
        let analysis_enabled = config.analysis_enabled;
        let cache_enabled = config.cache_enabled;
        let cache_capacity = config.cache_capacity;
        let cache_memory_limit = config.cache_memory_limit;
        let execution_mode = config.execution_mode;

        // Build TextProcessor from config's normalizers + tokenizer
        let processor = TextProcessor::new(config.normalizers, config.tokenizer, execution_mode)?;

        // Build DiffComputer from config's algorithm
        let diff_computer = DiffComputer::with_algorithm(config.algorithm);

        // Build CacheManager if enabled
        let cache = if cache_enabled {
            let cm = if let Some(mem_limit) = cache_memory_limit {
                CacheManager::with_memory_limit(cache_capacity, mem_limit)
            } else {
                CacheManager::new(cache_capacity)
            };
            Some(cm)
        } else {
            None
        };

        Ok(Self {
            processor,
            diff_computer,
            metrics_enabled,
            analysis_enabled,
            cache,
            config_fingerprint: fingerprint,
            preset,
        })
    }

    /// Run the full pipeline on source and target text.
    ///
    /// Steps: cache lookup → text processing → diff → metrics → analysis → cache store.
    /// Returns a [`RedlineResult`] with diff operations, metrics, and optional analysis.
    pub fn diff(&self, source: &str, target: &str) -> Result<RedlineResult, OrchestrateError> {
        // 1. Check cache
        let cache_key = PipelineCacheKey::new(source, target, self.config_fingerprint);
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(RedlineResult {
                    diff: cached.diff,
                    metrics: cached.metrics,
                    analysis: cached.analysis,
                });
            }
        }

        // 2. Process texts
        let source_processed = Arc::new(self.processor.process(source)?);
        let target_processed = Arc::new(self.processor.process(target)?);

        // 3. Compute diff
        let diff_result =
            self.diff_computer
                .compute(source_processed.clone(), target_processed.clone(), None)?;

        // 4. Compute metrics (if enabled)
        let metrics = if self.metrics_enabled {
            let mut registry = MetricRegistry::new();
            register_builtins(&mut registry)?;
            let engine = MetricsEngine::with_default_cache(registry);

            let input = MetricInput::Pairwise(&source_processed, &target_processed);
            let ids: Vec<&str> = engine.registry().ids().collect();
            engine.get_many(&ids, &input)
        } else {
            HashMap::new()
        };

        // 5. Run analysis (if enabled)
        let analysis = if self.analysis_enabled {
            let mut analyzer_registry = PluginRegistry::new();
            register_builtin_analyzers(&mut analyzer_registry)?;

            // Analysis needs its own MetricsEngine for on-demand metric computation
            let mut metrics_registry = MetricRegistry::new();
            register_builtins(&mut metrics_registry)?;
            let metrics_engine = MetricsEngine::with_default_cache(metrics_registry);

            let analysis_context = AnalysisContext::new(&diff_result, &metrics_engine);
            let coordinator = AnalysisCoordinator::new(analyzer_registry);
            let report = coordinator.run(&analysis_context)?;
            Some(Arc::new(report))
        } else {
            None
        };

        // 6. Store in cache
        if let Some(ref cache) = self.cache {
            let estimated =
                CachedResult::estimate_memory(&diff_result, &metrics, analysis.is_some());
            let cached = CachedResult {
                diff: diff_result.clone(),
                metrics: metrics.clone(),
                analysis: analysis.clone(),
                estimated_bytes: estimated,
            };
            cache.insert(cache_key, cached);
        }

        // 7. Return result
        Ok(RedlineResult {
            diff: diff_result,
            metrics,
            analysis,
        })
    }

    /// Returns the preset this orchestrator was built from.
    pub fn preset(&self) -> Preset {
        self.preset
    }

    /// Returns whether metrics computation is enabled.
    pub fn metrics_enabled(&self) -> bool {
        self.metrics_enabled
    }

    /// Returns whether analysis is enabled.
    pub fn analysis_enabled(&self) -> bool {
        self.analysis_enabled
    }

    /// Returns a reference to the cache, if enabled.
    pub fn cache(&self) -> Option<&CacheManager> {
        self.cache.as_ref()
    }
}

impl core::fmt::Debug for Redline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Redline")
            .field("preset", &self.preset)
            .field("metrics_enabled", &self.metrics_enabled)
            .field("analysis_enabled", &self.analysis_enabled)
            .field("cache_enabled", &self.cache.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::EditKind;

    // ── Construction tests ───────────────────────────────────────────

    #[test]
    fn new_with_comprehensive_preset() {
        let config = ConfigBuilder::new().build().unwrap();
        let redline = Redline::new(config).unwrap();
        assert_eq!(redline.preset(), Preset::Comprehensive);
        assert!(redline.metrics_enabled());
        assert!(redline.analysis_enabled());
        assert!(redline.cache().is_some());
    }

    #[test]
    fn new_with_fast_preset() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        assert_eq!(redline.preset(), Preset::Fast);
        assert!(redline.metrics_enabled());
        assert!(!redline.analysis_enabled());
        assert!(redline.cache().is_some());
    }

    #[test]
    fn new_with_cache_disabled() {
        let config = ConfigBuilder::new().with_cache(false).build().unwrap();
        let redline = Redline::new(config).unwrap();
        assert!(redline.cache().is_none());
    }

    // ── diff() pipeline tests ────────────────────────────────────────

    #[test]
    fn diff_returns_result_with_operations() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("hello world", "hello earth").unwrap();
        assert!(!result.diff.operations.is_empty());
    }

    #[test]
    fn diff_fast_preset_no_analysis() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("old text", "new text").unwrap();
        assert!(result.analysis.is_none());
    }

    #[test]
    fn diff_comprehensive_has_analysis() {
        let config = ConfigBuilder::new().build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline
            .diff(
                "The quick brown fox jumps over the lazy dog.",
                "The slow brown fox leaps over the happy dog.",
            )
            .unwrap();
        assert!(result.analysis.is_some());
    }

    #[test]
    fn diff_has_metrics_when_enabled() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("hello world", "hello earth").unwrap();
        assert!(!result.metrics.is_empty());
    }

    #[test]
    fn diff_no_metrics_when_disabled() {
        let config = ConfigBuilder::preset(Preset::Fast)
            .with_metrics(false)
            .build()
            .unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("hello world", "hello earth").unwrap();
        assert!(result.metrics.is_empty());
    }

    #[test]
    fn diff_identical_texts() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("same text", "same text").unwrap();
        assert!(
            (result.diff.similarity() - 1.0).abs() < f64::EPSILON,
            "identical texts should have similarity 1.0, got {}",
            result.diff.similarity()
        );
    }

    #[test]
    fn diff_empty_texts() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("", "").unwrap();
        assert!(result.diff.operations.is_empty());
    }

    // ── Cache tests ──────────────────────────────────────────────────

    #[test]
    fn cache_hit_on_second_call() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();

        let _r1 = redline.diff("hello world", "hello earth").unwrap();
        assert_eq!(redline.cache().unwrap().len(), 1);

        let _r2 = redline.diff("hello world", "hello earth").unwrap();
        assert_eq!(redline.cache().unwrap().len(), 1); // still 1 — cache hit
    }

    #[test]
    fn cache_miss_different_inputs() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();

        let _r1 = redline.diff("aaa", "bbb").unwrap();
        let _r2 = redline.diff("ccc", "ddd").unwrap();
        assert_eq!(redline.cache().unwrap().len(), 2);
    }

    #[test]
    fn cache_disabled_no_storage() {
        let config = ConfigBuilder::preset(Preset::Fast)
            .with_cache(false)
            .build()
            .unwrap();
        let redline = Redline::new(config).unwrap();
        let _r1 = redline.diff("hello", "world").unwrap();
        assert!(redline.cache().is_none());
    }

    // ── Filter tests ─────────────────────────────────────────────────

    #[test]
    fn filter_on_result() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("hello world", "hello earth").unwrap();

        let pred = Filter::kind(EditKind::Replace);
        let replaces: Vec<_> = result.filter(&pred).collect();
        assert!(!replaces.is_empty());
        for op in &replaces {
            assert_eq!(op.kind, EditKind::Replace);
        }
    }

    #[test]
    fn filter_with_combinators() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("hello world", "hello earth").unwrap();

        let pred = Filter::kind(EditKind::Replace) & Filter::min_length(1);
        let matches: Vec<_> = result.filter(&pred).collect();
        assert!(!matches.is_empty());
    }

    #[test]
    fn filter_not_equal() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        let redline = Redline::new(config).unwrap();
        let result = redline.diff("hello world", "hello earth").unwrap();

        let pred = !Filter::kind(EditKind::Equal);
        let changes: Vec<_> = result.filter(&pred).collect();
        for op in &changes {
            assert_ne!(op.kind, EditKind::Equal);
        }
    }

    // ── Send + Sync ──────────────────────────────────────────────────

    #[test]
    fn redline_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Redline>();
        assert_sync::<Redline>();
    }

    // ── Debug ────────────────────────────────────────────────────────

    #[test]
    fn debug_output() {
        let config = ConfigBuilder::new().build().unwrap();
        let redline = Redline::new(config).unwrap();
        let debug = format!("{:?}", redline);
        assert!(debug.contains("Redline"));
        assert!(debug.contains("Comprehensive"));
    }
}
