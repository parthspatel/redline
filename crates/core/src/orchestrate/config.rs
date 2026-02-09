//! Configuration system for the Redline orchestration pipeline.
//!
//! Provides `ConfigBuilder` with a fluent API, `Preset` enum, and `RedlineConfig`.

use crate::diff::DiffAlgorithm;
use crate::normalize::Normalizer;
use crate::process::ExecutionMode;
use crate::tokenize::Tokenizer;

use super::error::OrchestrateError;

/// Preset configuration profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// Optimized for speed: minimal normalizers, no analysis.
    Fast,
    /// Full-featured: all normalizers, metrics, and analysis enabled.
    Comprehensive,
}

impl Default for Preset {
    fn default() -> Self {
        Preset::Comprehensive
    }
}

/// The finalized, validated configuration for a Redline pipeline run.
pub struct RedlineConfig {
    pub(crate) preset: Preset,
    pub(crate) normalizers: Vec<Box<dyn Normalizer>>,
    pub(crate) tokenizer: Box<dyn Tokenizer>,
    pub(crate) algorithm: Box<dyn DiffAlgorithm>,
    pub(crate) metrics_enabled: bool,
    pub(crate) analysis_enabled: bool,
    pub(crate) cache_enabled: bool,
    pub(crate) cache_capacity: usize,
    pub(crate) cache_memory_limit: Option<usize>,
    pub(crate) execution_mode: ExecutionMode,
}

impl RedlineConfig {
    /// Returns the preset this config was built from.
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

    /// Returns whether caching is enabled.
    pub fn cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Returns the cache capacity (max entries).
    pub fn cache_capacity(&self) -> usize {
        self.cache_capacity
    }

    /// Returns the cache memory limit in bytes, if set.
    pub fn cache_memory_limit(&self) -> Option<usize> {
        self.cache_memory_limit
    }

    /// Returns the execution mode.
    pub fn execution_mode(&self) -> ExecutionMode {
        self.execution_mode
    }
}

impl core::fmt::Debug for RedlineConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RedlineConfig")
            .field("preset", &self.preset)
            .field("normalizer_count", &self.normalizers.len())
            .field("metrics_enabled", &self.metrics_enabled)
            .field("analysis_enabled", &self.analysis_enabled)
            .field("cache_enabled", &self.cache_enabled)
            .field("cache_capacity", &self.cache_capacity)
            .field("cache_memory_limit", &self.cache_memory_limit)
            .field("execution_mode", &self.execution_mode)
            .finish()
    }
}

/// Builder for constructing a `RedlineConfig` with a fluent API.
///
/// Starts from a preset and allows overriding individual settings.
pub struct ConfigBuilder {
    preset: Preset,
    normalizers: Option<Vec<Box<dyn Normalizer>>>,
    tokenizer: Option<Box<dyn Tokenizer>>,
    algorithm: Option<Box<dyn DiffAlgorithm>>,
    metrics_enabled: Option<bool>,
    analysis_enabled: Option<bool>,
    cache_enabled: Option<bool>,
    cache_capacity: Option<usize>,
    cache_memory_limit: Option<usize>,
    execution_mode: Option<ExecutionMode>,
}

impl ConfigBuilder {
    /// Create a new builder with `Preset::Comprehensive` defaults.
    pub fn new() -> Self {
        Self {
            preset: Preset::Comprehensive,
            normalizers: None,
            tokenizer: None,
            algorithm: None,
            metrics_enabled: None,
            analysis_enabled: None,
            cache_enabled: None,
            cache_capacity: None,
            cache_memory_limit: None,
            execution_mode: None,
        }
    }

    /// Create a new builder starting from the given preset.
    pub fn preset(p: Preset) -> Self {
        Self {
            preset: p,
            normalizers: None,
            tokenizer: None,
            algorithm: None,
            metrics_enabled: None,
            analysis_enabled: None,
            cache_enabled: None,
            cache_capacity: None,
            cache_memory_limit: None,
            execution_mode: None,
        }
    }

    /// Override the normalizer stack.
    pub fn with_normalizers(mut self, n: Vec<Box<dyn Normalizer>>) -> Self {
        self.normalizers = Some(n);
        self
    }

    /// Override the tokenizer.
    pub fn with_tokenizer(mut self, t: Box<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(t);
        self
    }

    /// Override the diff algorithm.
    pub fn with_algorithm(mut self, a: Box<dyn DiffAlgorithm>) -> Self {
        self.algorithm = Some(a);
        self
    }

    /// Enable or disable metrics computation.
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.metrics_enabled = Some(enabled);
        self
    }

    /// Enable or disable analysis.
    pub fn with_analysis(mut self, enabled: bool) -> Self {
        self.analysis_enabled = Some(enabled);
        self
    }

    /// Enable or disable caching.
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = Some(enabled);
        self
    }

    /// Set the cache capacity (max number of entries).
    pub fn with_cache_capacity(mut self, cap: usize) -> Self {
        self.cache_capacity = Some(cap);
        self
    }

    /// Set the cache memory limit in bytes.
    pub fn with_cache_memory_limit(mut self, bytes: usize) -> Self {
        self.cache_memory_limit = Some(bytes);
        self
    }

    /// Set the execution mode for the text processing pipeline.
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = Some(mode);
        self
    }

    /// Build and validate the configuration.
    ///
    /// Expands preset defaults, overlays any explicit overrides, and validates
    /// the resulting configuration.
    pub fn build(self) -> Result<RedlineConfig, OrchestrateError> {
        // Phase 1: Expand preset defaults
        let (
            default_normalizers,
            default_tokenizer,
            default_algorithm,
            default_metrics_enabled,
            default_analysis_enabled,
            default_cache_enabled,
            default_cache_capacity,
            default_execution_mode,
        ) = match self.preset {
            Preset::Fast => (
                Vec::new() as Vec<Box<dyn Normalizer>>,
                Box::new(crate::tokenize::WordTokenizer) as Box<dyn Tokenizer>,
                Box::new(crate::diff::Myers::new()) as Box<dyn DiffAlgorithm>,
                true,
                false,
                true,
                64usize,
                ExecutionMode::Minimal,
            ),
            Preset::Comprehensive => (
                vec![
                    Box::new(crate::normalize::UnicodeNormalizer::default()) as Box<dyn Normalizer>,
                    Box::new(crate::normalize::Lowercase) as Box<dyn Normalizer>,
                    Box::new(crate::normalize::WhitespaceNormalizer) as Box<dyn Normalizer>,
                ],
                Box::new(crate::tokenize::WordTokenizer) as Box<dyn Tokenizer>,
                Box::new(crate::diff::Myers::new()) as Box<dyn DiffAlgorithm>,
                true,
                true,
                true,
                256usize,
                ExecutionMode::Minimal,
            ),
        };

        // Phase 2: Overlay overrides (overrides ALWAYS win)
        let normalizers = self.normalizers.unwrap_or(default_normalizers);
        let tokenizer = self.tokenizer.unwrap_or(default_tokenizer);
        let algorithm = self.algorithm.unwrap_or(default_algorithm);
        let metrics_enabled = self.metrics_enabled.unwrap_or(default_metrics_enabled);
        let analysis_enabled = self.analysis_enabled.unwrap_or(default_analysis_enabled);
        let cache_enabled = self.cache_enabled.unwrap_or(default_cache_enabled);
        let cache_capacity = self.cache_capacity.unwrap_or(default_cache_capacity);
        let cache_memory_limit = self.cache_memory_limit;
        let execution_mode = self.execution_mode.unwrap_or(default_execution_mode);

        // Phase 3: Validate
        if cache_enabled && cache_capacity == 0 {
            return Err(OrchestrateError::Config(
                "cache capacity must be > 0 when cache is enabled".into(),
            ));
        }
        if let Some(limit) = cache_memory_limit {
            if limit == 0 {
                return Err(OrchestrateError::Config(
                    "cache memory limit must be > 0".into(),
                ));
            }
        }

        Ok(RedlineConfig {
            preset: self.preset,
            normalizers,
            tokenizer,
            algorithm,
            metrics_enabled,
            analysis_enabled,
            cache_enabled,
            cache_capacity,
            cache_memory_limit,
            execution_mode,
        })
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::Histogram;
    use crate::normalize::RemoveDiacritics;
    use crate::tokenize::CharTokenizer;

    #[test]
    fn new_defaults_to_comprehensive() {
        let config = ConfigBuilder::new().build().unwrap();
        assert_eq!(config.preset(), Preset::Comprehensive);
        assert!(config.metrics_enabled());
        assert!(config.analysis_enabled());
    }

    #[test]
    fn preset_fast_disables_analysis() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        assert_eq!(config.preset(), Preset::Fast);
        assert!(!config.analysis_enabled());
    }

    #[test]
    fn preset_fast_enables_metrics() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        assert!(config.metrics_enabled());
    }

    #[test]
    fn override_fast_with_analysis() {
        let config = ConfigBuilder::preset(Preset::Fast)
            .with_analysis(true)
            .build()
            .unwrap();
        assert!(config.analysis_enabled());
    }

    #[test]
    fn override_comprehensive_disable_cache() {
        let config = ConfigBuilder::new().with_cache(false).build().unwrap();
        assert!(!config.cache_enabled());
    }

    #[test]
    fn override_cache_capacity() {
        let config = ConfigBuilder::new()
            .with_cache_capacity(1024)
            .build()
            .unwrap();
        assert_eq!(config.cache_capacity(), 1024);
    }

    #[test]
    fn build_validates_zero_cache_capacity() {
        let result = ConfigBuilder::new().with_cache_capacity(0).build();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("cache capacity"));
    }

    #[test]
    fn build_validates_zero_memory_limit() {
        let result = ConfigBuilder::new().with_cache_memory_limit(0).build();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("memory limit"));
    }

    #[test]
    fn default_preset_is_comprehensive() {
        assert_eq!(Preset::default(), Preset::Comprehensive);
    }

    #[test]
    fn preset_clone_copy() {
        let p = Preset::Fast;
        let p2 = p; // Copy
        let p3 = p.clone(); // Clone
        assert_eq!(p, p2);
        assert_eq!(p, p3);

        fn assert_copy<T: Copy>() {}
        fn assert_clone<T: Clone>() {}
        fn assert_debug<T: core::fmt::Debug>() {}
        fn assert_eq_trait<T: PartialEq>() {}
        assert_copy::<Preset>();
        assert_clone::<Preset>();
        assert_debug::<Preset>();
        assert_eq_trait::<Preset>();
    }

    #[test]
    fn custom_normalizers_override_defaults() {
        let config = ConfigBuilder::new()
            .with_normalizers(vec![Box::new(RemoveDiacritics)])
            .build()
            .unwrap();
        // Comprehensive default has 3 normalizers; override should have 1
        assert_eq!(config.normalizers.len(), 1);
    }

    #[test]
    fn custom_tokenizer_overrides_default() {
        let config = ConfigBuilder::new()
            .with_tokenizer(Box::new(CharTokenizer))
            .build()
            .unwrap();
        assert!(config.metrics_enabled());
    }

    #[test]
    fn custom_algorithm_overrides_default() {
        let config = ConfigBuilder::new()
            .with_algorithm(Box::new(Histogram::new()))
            .build()
            .unwrap();
        assert!(config.metrics_enabled());
    }

    #[test]
    fn zero_cache_capacity_ok_when_cache_disabled() {
        let config = ConfigBuilder::new()
            .with_cache(false)
            .with_cache_capacity(0)
            .build()
            .unwrap();
        assert!(!config.cache_enabled());
        assert_eq!(config.cache_capacity(), 0);
    }

    #[test]
    fn config_debug_output() {
        let config = ConfigBuilder::new().build().unwrap();
        let debug = format!("{:?}", config);
        assert!(debug.contains("RedlineConfig"));
        assert!(debug.contains("Comprehensive"));
    }

    #[test]
    fn builder_default_same_as_new() {
        let config1 = ConfigBuilder::new().build().unwrap();
        let config2 = ConfigBuilder::default().build().unwrap();
        assert_eq!(config1.preset(), config2.preset());
        assert_eq!(config1.metrics_enabled(), config2.metrics_enabled());
        assert_eq!(config1.analysis_enabled(), config2.analysis_enabled());
        assert_eq!(config1.cache_enabled(), config2.cache_enabled());
        assert_eq!(config1.cache_capacity(), config2.cache_capacity());
    }

    #[test]
    fn execution_mode_override() {
        let config = ConfigBuilder::new()
            .with_execution_mode(ExecutionMode::All)
            .build()
            .unwrap();
        assert_eq!(config.execution_mode(), ExecutionMode::All);
    }

    #[test]
    fn cache_memory_limit_accessor() {
        let config = ConfigBuilder::new()
            .with_cache_memory_limit(1024 * 1024)
            .build()
            .unwrap();
        assert_eq!(config.cache_memory_limit(), Some(1024 * 1024));
    }

    #[test]
    fn cache_memory_limit_default_none() {
        let config = ConfigBuilder::new().build().unwrap();
        assert_eq!(config.cache_memory_limit(), None);
    }

    #[test]
    fn fast_preset_has_empty_normalizers() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        assert_eq!(config.normalizers.len(), 0);
    }

    #[test]
    fn comprehensive_preset_has_three_normalizers() {
        let config = ConfigBuilder::preset(Preset::Comprehensive)
            .build()
            .unwrap();
        assert_eq!(config.normalizers.len(), 3);
    }

    #[test]
    fn fast_preset_cache_capacity_is_64() {
        let config = ConfigBuilder::preset(Preset::Fast).build().unwrap();
        assert_eq!(config.cache_capacity(), 64);
    }

    #[test]
    fn comprehensive_preset_cache_capacity_is_256() {
        let config = ConfigBuilder::preset(Preset::Comprehensive)
            .build()
            .unwrap();
        assert_eq!(config.cache_capacity(), 256);
    }
}
