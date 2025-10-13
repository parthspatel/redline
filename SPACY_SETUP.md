# SpaCy Integration Setup

This document describes how to set up and use the SpaCy-based syntactic analyzer in Redline.

## Prerequisites

1. **Python 3.7+** must be installed and available in your PATH
2. **SpaCy** Python library and language models

## Using a Virtual Environment (Recommended)

When using a Python virtual environment, you need to configure both build-time and runtime settings:

### Build Time: PYO3_PYTHON

PyO3 needs to know which Python interpreter to link against at **build time**:

```bash
export PYO3_PYTHON=$(pwd)/.venv/bin/python
```

This ensures PyO3 links to the same Python version as your venv.

### Runtime: PYTHON_HOME

At **runtime**, set `PYTHON_HOME` to configure `sys.path`:

```bash
export PYTHON_HOME=$(pwd)/.venv
```

The SpaCy analyzer will automatically detect the Python version and add the venv's site-packages to `sys.path`. This works with any Python version (3.7-3.12+).

### Complete Setup Example

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install SpaCy
pip install spacy
python -m spacy download en_core_web_sm

# Build with venv Python
export PYO3_PYTHON=$(pwd)/.venv/bin/python
cargo build --features spacy

# Run with venv packages
export PYTHON_HOME=$(pwd)/.venv
cargo run --example spacy_syntactic_demo --features spacy
```

Or combine both in one command:

```bash
PYO3_PYTHON=$(pwd)/.venv/bin/python PYTHON_HOME=$(pwd)/.venv cargo run --example spacy_syntactic_demo --features spacy
```

## Installation Steps

### 1. Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install SpaCy

```bash
pip install spacy
```

### 3. Download a SpaCy Language Model

For English text analysis, install the small English model:

```bash
python -m spacy download en_core_web_sm
```

For better accuracy, you can use larger models:

```bash
# Medium model (more accurate)
python -m spacy download en_core_web_md

# Large model (most accurate, includes word vectors)
python -m spacy download en_core_web_lg
```

For other languages, see the [SpaCy models documentation](https://spacy.io/models).

### 4. Build Redline with SpaCy Support

Compile the `redline-core` crate with the `spacy` feature enabled:

```bash
cargo build --features spacy
```

Or to enable all features including SpaCy:

```bash
cargo build --features full
```

## Usage

### Running with Virtual Environment

If you're using a virtual environment, you must build with `PYO3_PYTHON` and run with `PYTHON_HOME`:

```bash
# Build once with PYO3_PYTHON (links to venv Python)
PYO3_PYTHON=$(pwd)/.venv/bin/python cargo build --features spacy

# Run with PYTHON_HOME (configures sys.path)
PYTHON_HOME=$(pwd)/.venv cargo run --example spacy_syntactic_demo --features spacy
```

Or combine both in one command (will rebuild if needed):

```bash
PYO3_PYTHON=$(pwd)/.venv/bin/python PYTHON_HOME=$(pwd)/.venv cargo run --example spacy_syntactic_demo --features spacy
```

**Important**: Both environment variables are required when using a venv:
- `PYO3_PYTHON` tells PyO3 which Python to link against (build time)
- `PYTHON_HOME` configures where to find packages (runtime)

### Basic Example

```rust
use redline_core::analyzer::classifiers::SpacySyntacticAnalyzer;
use redline_core::analyzer::SingleDiffAnalyzer;
use redline_core::diff::DiffResult;
use redline_core::DiffEngine;

// Create the analyzer (specify the model name)
let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

// Create a diff engine and compute the diff
let engine = DiffEngine::new();
let diff = engine.diff(
    "The cat sit on the mat.",
    "The cat sat on the mat.",
);

// Analyze the diff
let result = analyzer.analyze(&diff);

// View the analysis
println!("Analyzer: {}", result.analyzer_name);
println!("Confidence: {:.2}", result.confidence);
println!("\nMetrics:");
for (key, value) in &result.metrics {
    println!("  {}: {:.2}", key, value);
}
println!("\nInsights:");
for insight in &result.insights {
    println!("  - {}", insight);
}
```

### Analyzing Text Directly

You can also use the analyzer to process text directly and get detailed syntactic information:

```rust
let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

// Get syntactic tokens (POS tags, dependencies, etc.)
match analyzer.analyze_text("The quick brown fox jumps over the lazy dog.") {
    Ok(tokens) => {
        println!("Found {} tokens:\n", tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            println!("Token #{}: {}", i + 1, token.text);
            println!("  Lemma: {}", token.lemma);
            println!("  POS: {} ({})", token.pos, token.tag);
            println!("  Dependency: {} -> head #{}", token.dep, token.head);
            println!("  Is stop word: {}", token.is_stop);
            println!("  Is punctuation: {}", token.is_punct);
            println!();
        }
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

### Comparing Syntactic Structures

You can directly compare the syntactic structures of two texts:

```rust
use redline_core::analyzer::classifiers::SpacySyntacticAnalyzer;

let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

let original = "She don't like apples.";
let modified = "She doesn't like apples.";

match analyzer.compare_syntactic_structures(original, modified) {
    Ok(comparison) => {
        println!("Syntactic Comparison Results:");
        println!("============================");
        println!("POS changes: {}", comparison.pos_changes);
        println!("Dependency changes: {}", comparison.dep_changes);
        println!("Structural similarity: {:.2}%", comparison.structural_similarity * 100.0);
        println!("POS distribution divergence: {:.2}", comparison.pos_distribution_divergence);
        
        if !comparison.grammar_fixes.is_empty() {
            println!("\nGrammar fixes detected ({}):", comparison.grammar_fixes.len());
            for fix in &comparison.grammar_fixes {
                println!("  - {}", fix);
            }
        }
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Metrics Provided

The `SpacySyntacticAnalyzer` provides the following metrics:

- **pos_changes**: Number of part-of-speech tag changes between original and modified text
- **dep_changes**: Number of dependency relation changes
- **structural_similarity**: Overall syntactic similarity score (0.0 to 1.0)
- **grammar_fixes**: Number of detected grammar corrections
- **pos_distribution_divergence**: Divergence between POS tag distributions (0.0 to 1.0)

## Insights Generated

The analyzer generates human-readable insights such as:

- Structural change assessment (minimal/moderate/significant)
- Grammar fix detection and descriptions
- POS distribution change notifications

## Troubleshooting

### "SpaCy feature not enabled" message

If you see this message, make sure you compiled with the `spacy` feature:

```bash
cargo build --features spacy
```

### "Failed to load SpaCy model" error

This usually means:

1. SpaCy is not installed: `pip install spacy`
2. The language model is not downloaded: `python -m spacy download en_core_web_sm`
3. Python is not in your PATH or PyO3 can't find it

### PyO3 initialization errors

Make sure your Python environment is correctly set up. If you're using a virtual environment, set `PYTHON_HOME`:

```bash
export PYTHON_HOME=$(pwd)/.venv
cargo run --features spacy
```

For build-time issues, you may need to set `PYO3_PYTHON`:

```bash
export PYO3_PYTHON=$(pwd)/.venv/bin/python
cargo build --features spacy
```

### Performance considerations

- The first call to the analyzer will be slow as it loads the SpaCy model
- Subsequent calls reuse the loaded model (cached in memory)
- For production use, consider using a larger model for better accuracy
- The analyzer is thread-safe and can be shared across threads

## Language Models

Different SpaCy models offer different trade-offs:

| Model | Size | Speed | Accuracy | Word Vectors |
|-------|------|-------|----------|--------------|
| `en_core_web_sm` | ~15MB | Fast | Good | No |
| `en_core_web_md` | ~45MB | Medium | Better | Yes (20k) |
| `en_core_web_lg` | ~800MB | Slower | Best | Yes (685k) |

For non-English text, replace `en` with the appropriate language code (e.g., `de_core_news_sm` for German, `fr_core_news_sm` for French).

## Advanced Configuration

### Using Different Models

You can specify different models for different use cases:

```rust
// Small model for quick analysis
let fast_analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

// Large model for detailed analysis
let detailed_analyzer = SpacySyntacticAnalyzer::new("en_core_web_lg");

// Multi-language analysis
let german_analyzer = SpacySyntacticAnalyzer::new("de_core_news_sm");
```

### Thread Safety

The analyzer uses `Arc<Mutex<>>` internally, so it's safe to clone and use across threads:

```rust
use std::sync::Arc;
use std::thread;

let analyzer = Arc::new(SpacySyntacticAnalyzer::new("en_core_web_sm"));

let handles: Vec<_> = (0..4)
    .map(|i| {
        let analyzer = Arc::clone(&analyzer);
        thread::spawn(move || {
            let text = format!("This is thread {}", i);
            analyzer.analyze_text(&text)
        })
    })
    .collect();

for handle in handles {
    let _ = handle.join();
}
```

## References

- [SpaCy Documentation](https://spacy.io/)
- [SpaCy Models](https://spacy.io/models)
- [PyO3 User Guide](https://pyo3.rs/)
