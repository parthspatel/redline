//! PluginRegistry: analyzer registration with incremental cycle detection.
//!
//! Full implementation in plan 05-02.

use super::AnalyzerPlugin;

/// Registry of analyzer plugins.
pub struct PluginRegistry {
    analyzers: hashbrown::HashMap<String, Box<dyn AnalyzerPlugin>>,
}

impl core::fmt::Debug for PluginRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PluginRegistry")
            .field("count", &self.analyzers.len())
            .finish()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            analyzers: hashbrown::HashMap::new(),
        }
    }

    /// Number of registered analyzers.
    pub fn len(&self) -> usize {
        self.analyzers.len()
    }

    /// Returns true if no analyzers are registered.
    pub fn is_empty(&self) -> bool {
        self.analyzers.is_empty()
    }
}
