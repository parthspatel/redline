# Analyzers and Classifiers Guide

## Overview

The redline_core library provides comprehensive analysis and classification capabilities for understanding text changes. This document explains the two-tier analyzer system and the classification framework.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DiffResult                              │
│                                                              │
│  Operations • Statistics • Analysis                          │
└──────────────┬───────────────────────────────┬──────────────┘
               │                                │
        ┌──────▼──────┐                 ┌──────▼──────────┐
        │   Single    │                 │     Multi       │
        │  Analyzers  │                 │   Analyzers     │
        │             │                 │                 │
        │ • Semantic  │                 │ • Aggregate     │
        │ • Readabili │                 │ • Patterns      │
        │ • Stylistic │                 │ • Clustering    │
        │ • Category  │                 │ • Trends        │
        │ • Intent    │                 │                 │
        └─────────────┘                 └────────┬────────┘
                │                                 │
                └────────┬───────────────────────┘
                         │
                    ┌────▼─────┐
                    │Selectors │
                    │          │
                    │• Paragra │
                    │• Section │
                    │• Edit Ty │
                    └──────────┘
```

## Single Diff Analyzers

Operate on individual `DiffResult` objects to extract insights about a single comparison.

### 1. SemanticSimilarityAnalyzer

Analyzes how similar the texts are in meaning.

**Metrics:**
- `semantic_similarity`: 0.0-1.0 score
- `semantic_distance`: 1.0 - similarity

**Insights:**
- Similarity level classification
- Semantic change detection

**Example:**
```rust
use redline_core::analyzer::single::SemanticSimilarityAnalyzer;
use redline_core::analyzer::SingleDiffAnalyzer;

let analyzer = SemanticSimilarityAnalyzer::new()
    .with_threshold(0.7);

let result = analyzer.analyze(&diff);
println!("Similarity: {}", result.metrics["semantic_similarity"]);
```

### 2. ReadabilityAnalyzer

Measures readability using multiple standard metrics.

**Metrics:**
- `flesch_reading_ease`: 0-100 (higher = easier)
- `flesch_kincaid_grade`: US grade level
- `smog_index`: Years of education needed
- `automated_readability_index`: Grade level

**Insights:**
- Readability improvements/degradations
- Grade level changes

**Example:**
```rust
use redline_core::analyzer::single::ReadabilityAnalyzer;

let analyzer = ReadabilityAnalyzer::new();
let result = analyzer.analyze(&diff);

println!("Flesch score change: {}", 
    result.metrics["flesch_reading_ease_change"]);
```

### 3. StylisticAnalyzer

Analyzes stylistic properties of the text.

**Metrics:**
- `avg_word_length_change`
- `avg_sentence_length_change`
- `lexical_diversity_change`
- `punctuation_density_change`

**Insights:**
- Vocabulary sophistication changes
- Sentence complexity changes
- Word choice diversity

**Example:**
```rust
use redline_core::analyzer::single::StylisticAnalyzer;

let analyzer = StylisticAnalyzer::new();
let result = analyzer.analyze(&diff);
```

### 4. CategoryDistributionAnalyzer

Analyzes distribution of edit categories.

**Metrics:**
- `{category}_percentage`: % of changes in each category
- `{category}_count`: Number of operations per category
- `total_changes`: Total change operations

**Insights:**
- Dominant change category
- Distribution patterns

### 5. EditIntentClassifier

Classifies the intent behind edits.

**Detected Intents:**
- `clarification`: Making meaning clearer
- `expansion`: Adding content
- `condensation`: Removing content
- `correction`: Fixing errors
- `reformatting`: Changing presentation
- `stylistic_refinement`: Improving style
- `rewrite`: Complete rewording

**Example:**
```rust
use redline_core::analyzer::single::EditIntentClassifier;

let analyzer = EditIntentClassifier::new();
let result = analyzer.analyze(&diff);

// Check for specific intents
if result.metrics.contains_key("intent_expansion") {
    println!("User expanded the content");
}
```

## Multi-Diff Analyzers

Operate on collections of `DiffResult` objects to find patterns across multiple comparisons.

### 1. AggregateStatisticsAnalyzer

Computes aggregate statistics across diffs.

**Metrics:**
- `avg_insertions/deletions/modifications`
- `avg_semantic_similarity`
- `avg_change_percentage`
- `total_diffs`

**Example:**
```rust
use redline_core::analyzer::multi::AggregateStatisticsAnalyzer;
use redline_core::analyzer::MultiDiffAnalyzer;

let analyzer = AggregateStatisticsAnalyzer::new();
let diffs: Vec<&DiffResult> = diff_collection.iter().collect();
let result = analyzer.analyze(&diffs);
```

### 2. PatternDetectionAnalyzer

Detects common patterns across multiple edits.

**Detected Patterns:**
- Dominant edit types
- Consistent expansion/reduction
- Similarity clustering
- Category dominance

**Configuration:**
```rust
use redline_core::analyzer::multi::PatternDetectionAnalyzer;

let analyzer = PatternDetectionAnalyzer::new()
    .with_min_frequency(0.3); // Report patterns occurring in >30% of diffs
```

### 3. BehaviorClusteringAnalyzer

Groups diffs by editing behavior using k-means clustering.

**Features Used:**
- Semantic similarity
- Change percentage
- Insert/delete/modify ratios
- Stylistic change
- Readability change

**Cluster Types:**
- Minor refinements
- Moderate edits
- Substantial rewrites

**Example:**
```rust
use redline_core::analyzer::multi::BehaviorClusteringAnalyzer;

let analyzer = BehaviorClusteringAnalyzer::new(3); // 3 clusters
let result = analyzer.analyze(&diffs);

// Each cluster has size, percentage, and description
```

### 4. TemporalTrendAnalyzer

Analyzes trends over time (requires chronological order).

**Detected Trends:**
- `increasing`: Metric getting larger
- `decreasing`: Metric getting smaller
- `stable`: No significant trend

**Metrics:**
- Moving averages
- Trend directions
- Final values

**Example:**
```rust
use redline_core::analyzer::multi::TemporalTrendAnalyzer;

// Diffs should be in chronological order
let analyzer = TemporalTrendAnalyzer::new();
let result = analyzer.analyze(&chronological_diffs);
```

## Selectors

Filter operations from diffs for targeted analysis.

### WholeDocumentSelector

Selects all operations.

```rust
use redline_core::analyzer::selectors::WholeDocumentSelector;

let selector = WholeDocumentSelector;
let ops = selector.select(&diff);
```

### ParagraphSelector

Selects operations within specific paragraphs.

```rust
use redline_core::analyzer::selectors::ParagraphSelector;

// Select second paragraph
let selector = ParagraphSelector::single(1);

// Select paragraphs 2-5
let selector = ParagraphSelector::range(1, 5);

let ops = selector.select(&diff);
```

### SectionSelector

Selects operations within sections (by headers).

```rust
use redline_core::analyzer::selectors::SectionSelector;

let selector = SectionSelector::by_headers(vec!["Introduction", "Methods"]);
let ops = selector.select(&diff);
```

### EditTypeSelector

Selects by edit type.

```rust
use redline_core::analyzer::selectors::EditTypeSelector;

let selector = EditTypeSelector::insertions();
// or: deletions(), modifications(), changes()

let ops = selector.select(&diff);
```

### CompositeSelector

Combines multiple selectors with AND/OR logic.

```rust
use redline_core::analyzer::selectors::{CompositeSelector, EditTypeSelector, ParagraphSelector};

// Select insertions in paragraph 2
let selector = CompositeSelector::and(vec![
    Box::new(EditTypeSelector::insertions()),
    Box::new(ParagraphSelector::single(1)),
]);
```

## Classifiers

Classify changes into categories for detailed analysis.

### RuleBasedClassifier

Comprehensive rule-based classification.

**Categories:**
- `Semantic`: Meaning changes
- `Stylistic`: Style/tone changes
- `Formatting`: Presentation changes
- `Syntactic`: Grammar/spelling changes
- `Organizational`: Structure changes

**Example:**
```rust
use redline_core::classifiers::{RuleBasedClassifier, ChangeClassifier};

let classifier = RuleBasedClassifier::new();

for op in diff.operations {
    let classification = classifier.classify_operation(&op);
    println!("{:?} (confidence: {:.2})", 
             classification.category, 
             classification.confidence);
}
```

### NaiveBayesClassifier

ML-based classifier that can be trained on labeled data.

**Training:**
```rust
use redline_core::classifiers::ml::{
    NaiveBayesClassifier, 
    StandardFeatureExtractor,
    TrainingSample,
    create_training_samples,
};
use redline_core::ChangeCategory;

let extractor = StandardFeatureExtractor::new();
let mut classifier = NaiveBayesClassifier::new();

// Create training samples from labeled operations
let labeled_ops = vec![
    (op1, ChangeCategory::Semantic),
    (op2, ChangeCategory::Formatting),
    // ... more samples
];

let samples = create_training_samples(&labeled_ops, &extractor);
classifier.train(&samples);

// Now use the trained classifier
let classification = classifier.classify_operation(&new_op);
```

### EnsembleClassifier

Combines multiple classifiers.

```rust
use redline_core::classifiers::{EnsembleClassifier, RuleBasedClassifier};

let ensemble = EnsembleClassifier::new(vec![
    Box::new(RuleBasedClassifier::new()),
    Box::new(trained_nb_classifier),
])
.with_strategy("vote"); // or "average"
```

## Complete Workflow Example

```rust
use redline_core::{DiffEngine, AnalysisReport};
use redline_core::analyzer::single::*;
use redline_core::analyzer::multi::*;
use redline_core::analyzer::SingleDiffAnalyzer;

// 1. Create diffs
let engine = DiffEngine::default();
let diff1 = engine.diff("original 1", "edited 1");
let diff2 = engine.diff("original 2", "edited 2");

// 2. Single diff analysis
let mut report = AnalysisReport::new();

let analyzers: Vec<Box<dyn SingleDiffAnalyzer>> = vec![
    Box::new(SemanticSimilarityAnalyzer::new()),
    Box::new(ReadabilityAnalyzer::new()),
    Box::new(StylisticAnalyzer::new()),
    Box::new(EditIntentClassifier::new()),
];

for analyzer in &analyzers {
    report.add_result(analyzer.analyze(&diff1));
}

report.compute_summary();

// 3. Multi-diff analysis
let diffs = vec![&diff1, &diff2];

let multi_analyzers: Vec<Box<dyn MultiDiffAnalyzer>> = vec![
    Box::new(AggregateStatisticsAnalyzer::new()),
    Box::new(PatternDetectionAnalyzer::new()),
    Box::new(BehaviorClusteringAnalyzer::new(3)),
];

for analyzer in &multi_analyzers {
    let result = analyzer.analyze(&diffs);
    println!("{}: {:?}", analyzer.name(), result.insights);
}

// 4. Classification
let classifier = RuleBasedClassifier::new();
for op in diff1.changed_operations() {
    let class = classifier.classify_operation(op);
    println!("{:?}", class.category);
}
```

## Integration with User Edit Analysis

For analyzing user edits to AI-generated text:

```rust
// 1. Collect edit pairs
let generated_text = get_ai_generated_text();
let user_edited_text = get_user_edited_version();

// 2. Create diff
let diff = engine.diff(&generated_text, &user_edited_text);

// 3. Analyze
let semantic = SemanticSimilarityAnalyzer::new().analyze(&diff);
let intent = EditIntentClassifier::new().analyze(&diff);
let readability = ReadabilityAnalyzer::new().analyze(&diff);

// 4. Aggregate insights
if semantic.metrics["semantic_similarity"] < 0.5 {
    log("User made substantial content changes");
}

if intent.metrics.contains_key("intent_expansion") {
    log("User expanded the content");
}

if readability.metrics["flesch_reading_ease_change"] > 5.0 {
    log("User improved readability");
}

// 5. Over multiple edits, detect patterns
let all_diffs: Vec<&DiffResult> = /* collect all user edits */;
let pattern_analyzer = PatternDetectionAnalyzer::new();
let patterns = pattern_analyzer.analyze(&all_diffs);

// Use insights to improve generation
```

## Performance Notes

- **Single analyzers**: O(N) where N is diff size
- **Multi analyzers**: O(M×N) where M is number of diffs
- **Clustering**: O(M×K×I) where K is clusters, I is iterations
- **Classification**: O(1) per operation for rule-based, O(F) for ML where F is features

## Extension Points

### Custom Single Analyzer

```rust
use redline_core::analyzer::{SingleDiffAnalyzer, AnalysisResult};
use redline_core::diff::DiffResult;

#[derive(Clone)]
struct MyAnalyzer;

impl SingleDiffAnalyzer for MyAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult {
        let mut result = AnalysisResult::new("my_analyzer");
        // Your analysis logic
        result
    }

    fn name(&self) -> &str { "my_analyzer" }
    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer> {
        Box::new(self.clone())
    }
}
```

### Custom Classifier

```rust
use redline_core::classifiers::{ChangeClassifier, ClassificationResult};
use redline_core::diff::DiffOperation;

#[derive(Clone)]
struct MyClassifier;

impl ChangeClassifier for MyClassifier {
    fn classify_operation(&self, op: &DiffOperation) -> ClassificationResult {
        // Your classification logic
        ClassificationResult::new(category, confidence)
    }

    fn name(&self) -> &str { "my_classifier" }
    fn clone_box(&self) -> Box<dyn ChangeClassifier> {
        Box::new(self.clone())
    }
}
```

## Testing

Run the comprehensive example:
```bash
cargo run --example analyzers_demo
```

Run specific tests:
```bash
cargo test analyzers
cargo test classifiers
```
