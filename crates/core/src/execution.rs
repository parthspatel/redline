//! Dependency-aware execution planning for metrics and analysis
//!
//! This module implements a sophisticated execution system that:
//! 1. Tracks dependencies between metrics, analyzers, and classifiers
//! 2. Builds a dependency graph
//! 3. Topologically sorts the graph to determine optimal execution order
//! 4. Executes in order to eliminate cache misses

use std::collections::{HashMap, HashSet, VecDeque};

/// Metrics that can be computed and cached
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    // Text-level metrics
    CharCount,
    WordCount,
    SentenceCount,
    SyllableCount,
    WhitespaceCount,
    PunctuationCount,

    // Derived metrics
    FleschReadingEase,
    FleschKincaidGrade,
    AvgWordLength,
    AvgSentenceLength,

    // Linguistic features
    StopwordRatio,
    WhitespaceRatio,
    HasNegation,

    // Pairwise comparison metrics
    CharSimilarity,
    WordOverlap,
    LevenshteinDistance,
    LengthRatio,

    // Diff metrics
    ReadabilityDiff,
    WordCountDiff,
    WhitespaceRatioDiff,
    NegationChanged,

    // Higher-level analysis
    SemanticSimilarity,
    StylisticChange,
    ReadabilityChange,
}

/// A node in the execution graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExecutionNode {
    /// Compute a specific metric
    Metric(MetricType),
    /// Run an analyzer
    Analyzer(String),
    /// Run a classifier
    Classifier(String),
}

/// Dependency information for an execution node
#[derive(Debug, Clone)]
pub struct NodeDependencies {
    pub node: ExecutionNode,
    pub depends_on: Vec<ExecutionNode>,
}

impl NodeDependencies {
    pub fn new(node: ExecutionNode) -> Self {
        Self {
            node,
            depends_on: Vec::new(),
        }
    }

    pub fn with_dependency(mut self, dep: ExecutionNode) -> Self {
        self.depends_on.push(dep);
        self
    }

    pub fn with_dependencies(mut self, deps: Vec<ExecutionNode>) -> Self {
        self.depends_on.extend(deps);
        self
    }
}

/// Execution plan that orders operations to minimize cache misses
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Ordered list of nodes to execute
    execution_order: Vec<ExecutionNode>,
    /// Dependency graph
    dependencies: HashMap<ExecutionNode, Vec<ExecutionNode>>,
}

impl ExecutionPlan {
    /// Create a new execution plan from a set of dependencies
    pub fn new(node_deps: Vec<NodeDependencies>) -> Result<Self, ExecutionError> {
        let mut dependencies: HashMap<ExecutionNode, Vec<ExecutionNode>> = HashMap::new();

        // Build dependency map
        for dep in node_deps {
            dependencies.insert(dep.node.clone(), dep.depends_on);
        }

        // Topologically sort to get execution order
        let execution_order = Self::topological_sort(&dependencies)?;

        Ok(Self {
            execution_order,
            dependencies,
        })
    }

    /// Get the execution order
    pub fn execution_order(&self) -> &[ExecutionNode] {
        &self.execution_order
    }

    /// Perform topological sort using Kahn's algorithm
    fn topological_sort(
        dependencies: &HashMap<ExecutionNode, Vec<ExecutionNode>>,
    ) -> Result<Vec<ExecutionNode>, ExecutionError> {
        // Build in-degree map and reverse dependency graph
        // dependencies maps: node -> [things node depends on]
        // We need: node -> [things that depend on node] (reverse)

        let mut in_degree: HashMap<ExecutionNode, usize> = HashMap::new();
        let mut reverse_deps: HashMap<ExecutionNode, Vec<ExecutionNode>> = HashMap::new();
        let mut all_nodes: HashSet<ExecutionNode> = HashSet::new();

        // Initialize all nodes with in-degree 0
        for (node, deps) in dependencies {
            all_nodes.insert(node.clone());
            in_degree.entry(node.clone()).or_insert(0);

            for dep in deps {
                all_nodes.insert(dep.clone());
                in_degree.entry(dep.clone()).or_insert(0);
            }
        }

        // Build reverse dependencies and count in-degrees
        // If B depends on A, then: A -> B in reverse graph, and B's in-degree increases
        for (node, deps) in dependencies {
            *in_degree.get_mut(node).unwrap() += deps.len();

            for dep in deps {
                reverse_deps
                    .entry(dep.clone())
                    .or_insert_with(Vec::new)
                    .push(node.clone());
            }
        }

        // Find nodes with no dependencies (in-degree = 0)
        let mut queue: VecDeque<ExecutionNode> = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(node, _)| node.clone())
            .collect();

        let mut sorted = Vec::new();

        // Process queue
        while let Some(node) = queue.pop_front() {
            sorted.push(node.clone());

            // Find nodes that depend on this node (using reverse graph)
            if let Some(dependents) = reverse_deps.get(&node) {
                for dependent in dependents {
                    // Decrement in-degree
                    if let Some(degree) = in_degree.get_mut(dependent) {
                        *degree -= 1;

                        // If no more dependencies, add to queue
                        if *degree == 0 {
                            queue.push_back(dependent.clone());
                        }
                    }
                }
            }
        }

        // Check for cycles
        if sorted.len() != all_nodes.len() {
            return Err(ExecutionError::CyclicDependency);
        }

        Ok(sorted)
    }

    /// Check if a metric needs to be computed based on the execution order
    pub fn should_compute(&self, node: &ExecutionNode) -> bool {
        self.execution_order.contains(node)
    }

    /// Get metrics that should be precomputed
    pub fn get_required_metrics(&self) -> Vec<MetricType> {
        self.execution_order
            .iter()
            .filter_map(|node| {
                if let ExecutionNode::Metric(metric) = node {
                    Some(*metric)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Builder for creating execution plans
pub struct ExecutionPlanBuilder {
    dependencies: Vec<NodeDependencies>,
}

impl ExecutionPlanBuilder {
    pub fn new() -> Self {
        Self {
            dependencies: Vec::new(),
        }
    }

    /// Add a node with its dependencies
    pub fn add_node(&mut self, node_deps: NodeDependencies) -> &mut Self {
        self.dependencies.push(node_deps);
        self
    }

    /// Build the execution plan
    pub fn build(self) -> Result<ExecutionPlan, ExecutionError> {
        ExecutionPlan::new(self.dependencies)
    }
}

impl Default for ExecutionPlanBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during execution planning
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionError {
    /// Circular dependency detected
    CyclicDependency,
    /// Invalid dependency reference
    InvalidDependency,
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::CyclicDependency => {
                write!(f, "Cyclic dependency detected in execution plan")
            }
            ExecutionError::InvalidDependency => write!(f, "Invalid dependency reference"),
        }
    }
}

impl std::error::Error for ExecutionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dependency_order() {
        let mut builder = ExecutionPlanBuilder::new();

        // B depends on A
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(
            MetricType::WordCount,
        )));
        builder.add_node(
            NodeDependencies::new(ExecutionNode::Metric(MetricType::WordOverlap))
                .with_dependency(ExecutionNode::Metric(MetricType::WordCount)),
        );

        let plan = builder.build().unwrap();
        let order = plan.execution_order();

        // WordCount should come before WordOverlap
        let word_count_pos = order
            .iter()
            .position(|n| n == &ExecutionNode::Metric(MetricType::WordCount))
            .unwrap();
        let word_overlap_pos = order
            .iter()
            .position(|n| n == &ExecutionNode::Metric(MetricType::WordOverlap))
            .unwrap();

        assert!(word_count_pos < word_overlap_pos);
    }

    #[test]
    fn test_complex_dependency_chain() {
        let mut builder = ExecutionPlanBuilder::new();

        // Chain: A -> B -> C
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(
            MetricType::CharCount,
        )));
        builder.add_node(
            NodeDependencies::new(ExecutionNode::Metric(MetricType::CharSimilarity))
                .with_dependency(ExecutionNode::Metric(MetricType::CharCount)),
        );
        builder.add_node(
            NodeDependencies::new(ExecutionNode::Analyzer("test".to_string()))
                .with_dependency(ExecutionNode::Metric(MetricType::CharSimilarity)),
        );

        let plan = builder.build().unwrap();
        let order = plan.execution_order();

        // Should be in order: CharCount -> CharSimilarity -> Analyzer
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], ExecutionNode::Metric(MetricType::CharCount));
        assert_eq!(order[1], ExecutionNode::Metric(MetricType::CharSimilarity));
        assert_eq!(order[2], ExecutionNode::Analyzer("test".to_string()));
    }

    #[test]
    fn test_parallel_dependencies() {
        let mut builder = ExecutionPlanBuilder::new();

        // A, B, C all independent
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(
            MetricType::CharCount,
        )));
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(
            MetricType::WordCount,
        )));
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(
            MetricType::SentenceCount,
        )));

        let plan = builder.build().unwrap();

        // Should have all three in some order (order doesn't matter since independent)
        assert_eq!(plan.execution_order().len(), 3);
    }

    #[test]
    fn test_get_required_metrics() {
        let mut builder = ExecutionPlanBuilder::new();

        // Manually add some metrics to test get_required_metrics
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(
            MetricType::WordOverlap,
        )));
        builder.add_node(NodeDependencies::new(ExecutionNode::Metric(
            MetricType::CharSimilarity,
        )));

        let plan = builder.build().unwrap();
        let metrics = plan.get_required_metrics();

        assert!(metrics.contains(&MetricType::WordOverlap));
        assert!(metrics.contains(&MetricType::CharSimilarity));
    }
}
