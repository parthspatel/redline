//! Basic usage example of the text_diff library

use redline_core::normalizers::{Lowercase, WhitespaceNormalizer};
use redline_core::tokenizers::WordTokenizer;
use redline_core::{compute_diff, DiffConfig, DiffEngine, TextPipeline};

const LINE: &str = "----------------------------------------";

fn main() {
    println!("=== Text Diff Library Examples ===\n");

    // Example 1: Simple diff with default configuration
    example_simple_diff();

    // Example 2: Diff with normalization
    example_with_normalization();

    // Example 3: Custom configuration
    example_custom_config();

    // Example 4: Analyzing changes
    example_analyze_changes();
}

fn example_simple_diff() {
    println!("Example 1: Simple Diff");

    println!("{}", LINE);

    let original = "Hello World! This is a test.";
    let modified = "Hello Rust! This is a test.";

    let result = compute_diff(original, modified, None);

    println!("Original: {}", original);
    println!("Modified: {}", modified);
    println!("\n{}", result.summary());
    println!("\nOperations:");
    for (i, op) in result.operations.iter().enumerate() {
        println!("  {}. {}", i + 1, op.description());
    }
    println!("\n");
}

fn example_with_normalization() {
    println!("Example 2: Diff with Normalization");
    println!("{}", LINE);

    let original = "HELLO   WORLD";
    let modified = "hello world";

    // Without normalization
    let result_raw = compute_diff(original, modified, None);
    println!("Without normalization:");
    println!(
        "  Semantic similarity: {:.2}",
        result_raw.semantic_similarity
    );

    // With normalization
    let config = DiffConfig::default().with_pipeline(
        TextPipeline::new()
            .add_normalizer(Box::new(Lowercase))
            .add_normalizer(Box::new(WhitespaceNormalizer::new().with_collapse(true))),
    );

    let result_normalized = compute_diff(original, modified, Some(config));
    println!("\nWith normalization (lowercase + whitespace):");
    println!(
        "  Semantic similarity: {:.2}",
        result_normalized.semantic_similarity
    );
    println!("\n");
}

fn example_custom_config() {
    println!("Example 3: Custom Configuration");
    println!("{}", LINE);

    let original = "The quick brown fox jumps over the lazy dog.";
    let modified = "The fast brown fox leaps over the sleepy dog.";

    let config = DiffConfig::default()
        .with_algorithm(redline_core::DiffAlgorithm::Histogram)
        .with_tokenizer(Box::new(WordTokenizer::new()))
        .with_semantic_similarity(true)
        .with_edit_classification(true)
        .with_style_analysis(true);

    let engine = DiffEngine::new(config);
    let result = engine.diff(original, modified);

    println!("Original: {}", original);
    println!("Modified: {}", modified);
    println!("\nAnalysis:");
    println!(
        "  Semantic similarity: {:.2}",
        result.analysis.semantic_similarity
    );
    println!(
        "  Stylistic change: {:.2}",
        result.analysis.stylistic_change
    );
    println!(
        "  Readability change: {:.2}",
        result.analysis.readability_change
    );

    println!("\nChanged operations:");
    for op in result.changed_operations() {
        println!(
            "  - {} (category: {:?}, confidence: {:.2})",
            op.description(),
            op.category,
            op.confidence
        );
    }
    println!("\n");
}

fn example_analyze_changes() {
    println!("Example 4: Analyzing Different Types of Changes");
    println!("{}", LINE);

    let test_cases = vec![
        (
            "The cat sat on the mat.",
            "The dog sat on the mat.",
            "Semantic change (cat -> dog)",
        ),
        ("Hello World", "hello world", "Formatting change (case)"),
        (
            "I dont know",
            "I don't know",
            "Syntactic change (apostrophe)",
        ),
        (
            "The quick brown fox",
            "The fast brown fox",
            "Stylistic change (quick -> fast)",
        ),
    ];

    for (original, modified, description) in test_cases {
        let result = compute_diff(original, modified, None);

        println!("\n{}", description);
        println!("  Original:  {}", original);
        println!("  Modified:  {}", modified);
        println!("  Similarity: {:.2}", result.semantic_similarity);
        println!(
            "  Changes: {} insertions, {} deletions, {} modifications",
            result.statistics.insertions,
            result.statistics.deletions,
            result.statistics.modifications
        );
    }
    println!();
}
