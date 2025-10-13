# Modular SpaCy Analyzers

This document describes the modular SpaCy-based analyzers available in Redline. These analyzers provide focused, single-responsibility analysis of syntactic changes.

## Overview

The original `SpacySyntacticAnalyzer` combined multiple types of analysis into one analyzer. The modular approach breaks this into four specialized analyzers:

1. **SpacyPOSAnalyzer** - Part-of-speech tag analysis
2. **SpacyDependencyAnalyzer** - Dependency relation analysis  
3. **SpacyGrammarAnalyzer** - Grammar fix detection
4. **SpacyStructuralAnalyzer** - Structural similarity analysis

## Benefits of Modular Analyzers

### 1. Fine-Grained Control
Run only the analyses you need:
```rust
// Only check for grammar fixes
let grammar_analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
let result = grammar_analyzer.analyze(&diff);
```

### 2. Better Performance
Skip expensive analyses when not needed:
```rust
// Skip structural analysis if you only care about POS tags
let pos_analyzer = SpacyPOSAnalyzer::new("en_core_web_sm");
```

### 3. Clearer Insights
Each analyzer focuses on one aspect with detailed metrics:
```rust
// Get specific POS distribution metrics
let pos_result = pos_analyzer.analyze(&diff);
// Metrics: pos_changes, pos_distribution_divergence, pos_unchanged_ratio
```

### 4. Easier Testing
Test individual components in isolation:
```rust
#[test]
fn test_grammar_fixes() {
    let analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
    // Test only grammar detection logic
}
```

### 5. Flexible Composition
Combine analyzers as needed for your use case:
```rust
let analyzers: Vec<Box<dyn SingleDiffAnalyzer>> = vec![
    Box::new(SpacyGrammarAnalyzer::new("en_core_web_sm")),
    Box::new(SpacyStructuralAnalyzer::new("en_core_web_sm")),
];
```

## Analyzer Details

### SpacyPOSAnalyzer

Analyzes changes in part-of-speech tags between original and modified text.

**Metrics:**
- `pos_changes` - Number of POS tag changes
- `pos_distribution_divergence` - Divergence between POS distributions (0.0 to 1.0)
- `pos_unchanged_ratio` - Ratio of tokens with unchanged POS tags

**Use Cases:**
- Detecting word class changes (noun → verb)
- Analyzing POS distribution shifts
- Identifying text complexity changes

**Example:**
```rust
use redline_core::analyzer::classifiers::SpacyPOSAnalyzer;
use redline_core::analyzer::SingleDiffAnalyzer;

let analyzer = SpacyPOSAnalyzer::new("en_core_web_sm");
let result = analyzer.analyze(&diff);

println!("POS changes: {}", result.metrics["pos_changes"]);
```

**Output Example:**
```
Metrics:
  • pos_changes: 2.00
  • pos_distribution_divergence: 0.15
  • pos_unchanged_ratio: 0.85

Insights:
  • Minimal POS changes: 2 tag(s) changed
```

---

### SpacyDependencyAnalyzer

Analyzes changes in dependency relations between tokens.

**Metrics:**
- `dep_changes` - Number of dependency relation changes
- `dep_unchanged_ratio` - Ratio of tokens with unchanged dependencies
- `root_changes` - Number of changes to ROOT (main verb) dependencies

**Use Cases:**
- Detecting syntactic restructuring
- Analyzing clause changes
- Identifying sentence complexity shifts

**Example:**
```rust
use redline_core::analyzer::classifiers::SpacyDependencyAnalyzer;
use redline_core::analyzer::SingleDiffAnalyzer;

let analyzer = SpacyDependencyAnalyzer::new("en_core_web_sm");
let result = analyzer.analyze(&diff);

println!("Dependency changes: {}", result.metrics["dep_changes"]);
```

**Output Example:**
```
Metrics:
  • dep_changes: 5.00
  • root_changes: 1.00
  • dep_unchanged_ratio: 0.50

Insights:
  • Significant dependency changes: 5 relation(s) changed
  • Main clause structure changed (1 ROOT change(s))
  • Dependency changes:
    - 'dog': nsubj → nsubjpass
    - 'chased': ROOT → auxpass
```

---

### SpacyGrammarAnalyzer

Detects potential grammar corrections based on syntactic patterns.

**Metrics:**
- `grammar_fixes` - Total number of grammar fixes detected
- `verb_fixes` - Number of verb-related fixes
- `article_fixes` - Number of article corrections (a/an)
- `tense_fixes` - Number of tense corrections

**Grammar Patterns Detected:**
- Subject-verb agreement fixes
- Verb tense corrections
- Article corrections (a/an)
- Pronoun corrections (I/me, who/whom)
- Auxiliary verb corrections (was/were, is/are)

**Example:**
```rust
use redline_core::analyzer::classifiers::SpacyGrammarAnalyzer;
use redline_core::analyzer::SingleDiffAnalyzer;

let analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
let result = analyzer.analyze(&diff);

for insight in &result.insights {
    println!("{}", insight);
}
```

**Output Example:**
```
Metrics:
  • grammar_fixes: 2.00
  • verb_fixes: 1.00
  • article_fixes: 1.00
  • tense_fixes: 0.00

Insights:
  • Detected 2 potential grammar fix(es):
    - Verb form correction: 'have' → 'has'
    - Article correction: 'a' → 'an'
  • Fix categories: 1 verb, 1 article, 0 tense
```

---

### SpacyStructuralAnalyzer

Measures syntactic structural similarity based on dependency trees.

**Metrics:**
- `structural_similarity` - Overall syntactic similarity (0.0 to 1.0)
- `pos_dep_match_ratio` - Ratio of tokens with matching POS and dependency
- `sentence_structure_change` - Whether structure was significantly altered (0.0 or 1.0)

**Use Cases:**
- Detecting sentence restructuring
- Measuring paraphrasing similarity
- Identifying active/passive voice changes

**Example:**
```rust
use redline_core::analyzer::classifiers::SpacyStructuralAnalyzer;
use redline_core::analyzer::SingleDiffAnalyzer;

let analyzer = SpacyStructuralAnalyzer::new("en_core_web_sm");
let result = analyzer.analyze(&diff);

let similarity = result.metrics["structural_similarity"];
println!("Structural similarity: {:.1}%", similarity * 100.0);
```

**Output Example:**
```
Metrics:
  • structural_similarity: 0.12
  • sentence_structure_change: 1.00
  • pos_dep_match_ratio: 0.17

Insights:
  • Major syntactic restructuring detected
  • Sentence structure was significantly altered
  • Original clauses: ["dog [chased]"]
  • Modified clauses: ["[chased]"]
  • Structural match: 16.7% of tokens have identical POS+dependency
```

## Usage Examples

### Basic Usage

```rust
use redline_core::{DiffConfig, DiffEngine};
use redline_core::analyzer::classifiers::{
    SpacyPOSAnalyzer,
    SpacyDependencyAnalyzer,
    SpacyGrammarAnalyzer,
    SpacyStructuralAnalyzer,
};
use redline_core::analyzer::SingleDiffAnalyzer;

// Create diff engine
let config = DiffConfig::default();
let engine = DiffEngine::new(config);

// Create diff
let diff = engine.diff(
    "The student have completed their homework.",
    "The student has completed their homework."
);

// Use individual analyzers
let grammar_analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
let grammar_result = grammar_analyzer.analyze(&diff);

let pos_analyzer = SpacyPOSAnalyzer::new("en_core_web_sm");
let pos_result = pos_analyzer.analyze(&diff);
```

### Combined Analysis

```rust
// Create all analyzers
let analyzers: Vec<Box<dyn SingleDiffAnalyzer>> = vec![
    Box::new(SpacyPOSAnalyzer::new("en_core_web_sm")),
    Box::new(SpacyDependencyAnalyzer::new("en_core_web_sm")),
    Box::new(SpacyGrammarAnalyzer::new("en_core_web_sm")),
    Box::new(SpacyStructuralAnalyzer::new("en_core_web_sm")),
];

// Run all analyses
for analyzer in &analyzers {
    let result = analyzer.analyze(&diff);
    println!("\n{}: {}", analyzer.name(), analyzer.description());
    
    for (metric, value) in &result.metrics {
        println!("  {}: {:.2}", metric, value);
    }
}
```

### Grammar-Only Check

```rust
// Quick grammar check
let grammar_analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
let result = grammar_analyzer.analyze(&diff);

if result.metrics["grammar_fixes"] > 0.0 {
    println!("Grammar corrections detected:");
    for insight in &result.insights {
        if insight.contains("correction") {
            println!("  {}", insight);
        }
    }
}
```

### Structural Similarity Check

```rust
// Check if text was significantly restructured
let structural_analyzer = SpacyStructuralAnalyzer::new("en_core_web_sm");
let result = structural_analyzer.analyze(&diff);

let similarity = result.metrics["structural_similarity"];
if similarity < 0.7 {
    println!("Text was significantly restructured (similarity: {:.1}%)", 
             similarity * 100.0);
} else {
    println!("Text structure mostly preserved (similarity: {:.1}%)", 
             similarity * 100.0);
}
```

## Running the Demo

A comprehensive example demonstrating all modular analyzers is available:

```bash
PYO3_PYTHON=$(pwd)/.venv/bin/python PYTHON_HOME=$(pwd)/.venv \
    cargo run --example spacy_modular_demo --features spacy
```

This demo shows:
- Grammar fix detection (subject-verb agreement, articles)
- POS tag analysis (verb tense changes)
- Structural changes (active to passive voice)
- Dependency relation changes

## Setup Requirements

See [SPACY_SETUP.md](SPACY_SETUP.md) for complete setup instructions.

**Quick setup:**
```bash
# Create venv and install SpaCy
python3 -m venv .venv
source .venv/bin/activate
pip install spacy
python -m spacy download en_core_web_sm

# Build with venv Python
PYO3_PYTHON=$(pwd)/.venv/bin/python cargo build --features spacy

# Run with venv packages
PYTHON_HOME=$(pwd)/.venv cargo run --features spacy
```

## Performance Considerations

### Model Loading
- First analysis loads SpaCy model (1-2 seconds)
- Subsequent analyses reuse cached model (milliseconds)
- All analyzers share the same SpaCy client

### Optimization Tips

**1. Reuse analyzer instances:**
```rust
// Good: Create once, use many times
let analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
for diff in diffs {
    let result = analyzer.analyze(&diff);
}
```

**2. Use appropriate model size:**
```rust
// Fast but less accurate
let analyzer = SpacyPOSAnalyzer::new("en_core_web_sm");

// Slower but more accurate
let analyzer = SpacyPOSAnalyzer::new("en_core_web_lg");
```

**3. Run only needed analyzers:**
```rust
// Don't run all analyzers if you only need grammar checks
let grammar_analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
// Skip POS, dependency, and structural analyzers
```

## Comparison: Modular vs Combined

### Original Combined Analyzer
```rust
// SpacySyntacticAnalyzer - All-in-one
let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");
let result = analyzer.analyze(&diff);
// Metrics: pos_changes, dep_changes, structural_similarity, 
//          grammar_fixes, pos_distribution_divergence
```

### New Modular Approach
```rust
// Choose what you need
let pos_analyzer = SpacyPOSAnalyzer::new("en_core_web_sm");
let grammar_analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");

let pos_result = pos_analyzer.analyze(&diff);
let grammar_result = grammar_analyzer.analyze(&diff);
```

**Advantages:**
- ✅ More focused metrics
- ✅ Clearer insights
- ✅ Better performance (run only what you need)
- ✅ Easier to test
- ✅ More maintainable

**Migration:**
The original `SpacySyntacticAnalyzer` is still available for backward compatibility, but new code should use the modular analyzers.

## Thread Safety

All analyzers are thread-safe and can be cloned:

```rust
use std::sync::Arc;
use std::thread;

let analyzer = Arc::new(SpacyGrammarAnalyzer::new("en_core_web_sm"));

let handles: Vec<_> = (0..4)
    .map(|_| {
        let analyzer = Arc::clone(&analyzer);
        thread::spawn(move || {
            // Use analyzer in thread
        })
    })
    .collect();

for handle in handles {
    handle.join().unwrap();
}
```

## API Reference

### Common Methods

All analyzers implement the `SingleDiffAnalyzer` trait:

```rust
pub trait SingleDiffAnalyzer {
    fn analyze(&self, diff: &DiffResult) -> AnalysisResult;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn clone_box(&self) -> Box<dyn SingleDiffAnalyzer>;
}
```

### SpacyPOSAnalyzer Specific
```rust
impl SpacyPOSAnalyzer {
    pub fn new(model_name: impl Into<String>) -> Self;
    pub fn analyze_pos(&self, text: &str) -> Result<Vec<SyntacticToken>, String>;
    pub fn count_pos_changes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> usize;
    pub fn calculate_pos_divergence(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> f64;
    pub fn get_pos_distribution(&self, tokens: &[SyntacticToken]) -> HashMap<String, usize>;
}
```

### SpacyDependencyAnalyzer Specific
```rust
impl SpacyDependencyAnalyzer {
    pub fn new(model_name: impl Into<String>) -> Self;
    pub fn analyze_dependencies(&self, text: &str) -> Result<Vec<SyntacticToken>, String>;
    pub fn count_dep_changes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> usize;
    pub fn count_root_changes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> usize;
    pub fn get_dep_distribution(&self, tokens: &[SyntacticToken]) -> HashMap<String, usize>;
    pub fn find_dep_changes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> Vec<String>;
}
```

### SpacyGrammarAnalyzer Specific
```rust
impl SpacyGrammarAnalyzer {
    pub fn new(model_name: impl Into<String>) -> Self;
    pub fn analyze_grammar(&self, text: &str) -> Result<Vec<SyntacticToken>, String>;
    pub fn detect_grammar_fixes(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> Vec<String>;
    pub fn categorize_fixes(&self, fixes: &[String]) -> (usize, usize, usize); // (verb, article, tense)
}
```

### SpacyStructuralAnalyzer Specific
```rust
impl SpacyStructuralAnalyzer {
    pub fn new(model_name: impl Into<String>) -> Self;
    pub fn analyze_structure(&self, text: &str) -> Result<Vec<SyntacticToken>, String>;
    pub fn calculate_structural_similarity(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> f64;
    pub fn pos_dep_match_ratio(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> f64;
    pub fn has_significant_structural_change(&self, orig: &[SyntacticToken], modified: &[SyntacticToken]) -> bool;
    pub fn get_clause_structure(&self, tokens: &[SyntacticToken]) -> Vec<String>;
}
```

## References

- [SpaCy Documentation](https://spacy.io/)
- [SpaCy Dependency Labels](https://universaldependencies.org/u/dep/)
- [SpaCy POS Tags](https://universaldependencies.org/u/pos/)
- [Redline Setup Guide](SPACY_SETUP.md)
