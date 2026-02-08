//! Analysis framework: plugin-based analyzer system with dependency resolution.
//!
//! Analyzers declare dependencies, register with a [`PluginRegistry`], and execute
//! in topological order. Results are collected into an [`AnalysisReport`] with typed
//! accessors for built-in analyzers and dynamic access for plugins.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

use hashbrown::HashMap;

pub mod annotation;
pub mod builtins;
pub mod context;
pub mod coordinator;
pub mod error;
pub mod planner;
pub mod registry;
pub mod report;

pub use annotation::{Annotation, RegionSummary};
pub use context::AnalysisContext;
pub use coordinator::AnalysisCoordinator;
pub use error::AnalysisError;
pub use planner::ExecutionPlanner;
pub use registry::PluginRegistry;
pub use report::{AnalysisReport, AnalyzerExecutionMeta, AnalyzerStatus};

/// Metadata describing an analyzer plugin.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalyzerMeta {
    /// Unique identifier (e.g., "semantic", "edit_classifier").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Version string.
    pub version: String,
}

/// Dependency declaration for an analyzer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnalyzerDependency {
    /// Must be present and succeed before this analyzer runs.
    Required(String),
    /// Used if present, but not required.
    Optional(String),
}

impl AnalyzerDependency {
    /// Returns the analyzer ID this dependency refers to.
    pub fn id(&self) -> &str {
        match self {
            AnalyzerDependency::Required(id) | AnalyzerDependency::Optional(id) => id,
        }
    }

    /// Returns true if this dependency is required.
    pub fn is_required(&self) -> bool {
        matches!(self, AnalyzerDependency::Required(_))
    }
}

/// Common result struct for dynamic/plugin analyzers.
///
/// Built-in analyzers return their own typed result structs; custom plugins
/// return `AnalysisResult`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalysisResult {
    /// Per-operation annotations.
    pub annotations: Vec<Annotation>,
    /// Region-level summaries.
    pub summaries: Vec<RegionSummary>,
    /// Named scores produced by this analyzer.
    pub scores: HashMap<String, f64>,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl AnalysisResult {
    /// Create an empty analysis result.
    pub fn new() -> Self {
        Self {
            annotations: Vec::new(),
            summaries: Vec::new(),
            scores: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

impl Default for AnalysisResult {
    fn default() -> Self {
        Self::new()
    }
}

/// The core trait for analyzer plugins.
///
/// All analyzers implement this trait. It is object-safe, `Send + Sync`,
/// enabling storage in trait-object collections and future parallel execution.
pub trait AnalyzerPlugin: Send + Sync {
    /// Unique identifier for this analyzer.
    fn id(&self) -> &str;

    /// Full metadata for this analyzer.
    fn meta(&self) -> AnalyzerMeta;

    /// Declare dependencies on other analyzers.
    fn dependencies(&self) -> Vec<AnalyzerDependency> {
        Vec::new()
    }

    /// Relative cost hint for scheduling (0.0 = free, 1.0 = expensive).
    fn cost(&self) -> f32 {
        0.5
    }

    /// Run this analyzer against the given context.
    ///
    /// Receives read-only access to the diff context and the report built so far.
    /// Returns a boxed result stored in the report. Built-in analyzers return their
    /// typed result struct; plugins return `AnalysisResult`.
    fn analyze(
        &self,
        context: &AnalysisContext,
        prior_results: &AnalysisReport,
    ) -> Result<Box<dyn core::any::Any + Send + Sync>, AnalysisError>;
}
