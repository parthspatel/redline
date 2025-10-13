# Redline Core Examples

This directory contains examples demonstrating various features of the redline-core library.

## Basic Examples

### basic_usage.rs

Demonstrates fundamental diff operations and configurations.

```bash
cargo run --example basic_usage
```

**Features:**
- Simple text diffing
- Normalization pipelines
- Custom configurations
- Analyzing different types of changes

### advanced_pipeline.rs

Shows advanced normalization and tokenization techniques.

```bash
cargo run --example advanced_pipeline
```

**Features:**
- Multi-stage normalization pipelines
- Different tokenization strategies (character, word, n-gram, sentence)
- Character n-gram analysis
- Sentence-level diffing

### analyzers_demo.rs

Comprehensive demonstration of analyzers and classifiers.

```bash
cargo run --example analyzers_demo
```

**Features:**
- Single diff analysis (semantic, readability, stylistic)
- Multi-diff analysis (patterns, clustering, trends)
- Using selectors to filter operations
- Change classification
- Complete user edit analysis workflow

## Advanced Examples

### spacy_syntactic_demo.rs

**⚠️ Requires additional setup** - See [SPACY_SETUP.md](../../SPACY_SETUP.md)

Demonstrates the SpaCy-based syntactic analyzer for advanced NLP analysis.

```bash
# Prerequisites
pip install spacy
python -m spacy download en_core_web_sm

# Run the example
cargo run --example spacy_syntactic_demo --features spacy
```

**Features:**
- Part-of-speech (POS) tagging
- Dependency parsing
- Grammar error detection
- Syntactic structure comparison
- POS distribution analysis
- Integration with diff analysis

**Example Output:**

The example demonstrates:

1. **Basic Analysis**: Token-level POS tags and dependencies
2. **Grammar Detection**: Identifies subject-verb agreement errors, tense corrections, etc.
3. **Structure Comparison**: Analyzes active vs. passive voice, clause reordering
4. **POS Distribution**: Statistical analysis of part-of-speech usage
5. **Real-World Edits**: Common grammatical corrections
6. **Diff Integration**: Full integration with the diff engine

## Running Examples

### Run a specific example:

```bash
cargo run --example <example_name>
```

### Run with specific features:

```bash
# SpaCy support
cargo run --example spacy_syntactic_demo --features spacy

# All features
cargo run --example analyzers_demo --features full
```

### List all examples:

```bash
cargo run --example
```

## Example Categories

| Example | Category | Features Required | Description |
|---------|----------|-------------------|-------------|
| `basic_usage` | Core | None | Basic diff operations |
| `advanced_pipeline` | Core | None | Advanced normalization and tokenization |
| `analyzers_demo` | Analysis | None | Comprehensive analyzer showcase |
| `spacy_syntactic_demo` | NLP | `spacy` | SpaCy-based syntactic analysis |

## Dependencies

### All Examples
- Rust 1.70+
- redline-core library

### SpaCy Examples
- Python 3.7+
- SpaCy library: `pip install spacy`
- SpaCy language model: `python -m spacy download en_core_web_sm`

## Troubleshooting

### "SpaCy feature not enabled"

Make sure to build with the `--features spacy` flag:

```bash
cargo run --example spacy_syntactic_demo --features spacy
```

### "Failed to import spacy"

Install SpaCy in your Python environment:

```bash
pip install spacy
```

### "Failed to load SpaCy model"

Download the required language model:

```bash
python -m spacy download en_core_web_sm
```

For other languages, see [SpaCy models](https://spacy.io/models).

### PyO3 initialization errors

Ensure Python is in your PATH and PyO3 can find it:

```bash
export PYTHON_SYS_EXECUTABLE=$(which python3)
cargo run --example spacy_syntactic_demo --features spacy
```

## Contributing

When adding new examples:

1. Create a new `.rs` file in this directory
2. Add it to `Cargo.toml` under `[[example]]`
3. If it requires features, add `required-features = ["feature-name"]`
4. Update this README with documentation
5. Add appropriate comments and documentation in the example code

## See Also

- [Main README](../../../README.md)
- [SpaCy Setup Guide](../../SPACY_SETUP.md)
- [API Documentation](https://docs.rs/redline-core)
