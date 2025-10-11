//! Advanced pipeline example showing custom normalizers and tokenizers

use redline_core::normalizers::{Lowercase, RemovePunctuation, WhitespaceNormalizer};
use redline_core::tokenizers::{
    CharacterTokenizer, NGramTokenizer, SentenceTokenizer, WordTokenizer,
};
use redline_core::{DiffConfig, DiffEngine, TextPipeline};

const LINE: &str = "----------------------------------------";

fn main() {
    println!("=== Advanced Pipeline Examples ===\n");

    // Example 1: Multi-stage normalization pipeline
    example_multi_stage_pipeline();

    // Example 2: Different tokenization strategies
    example_tokenization_strategies();

    // Example 3: Character n-gram diff
    example_character_ngrams();

    // Example 4: Sentence-level diff
    example_sentence_diff();
}

fn example_multi_stage_pipeline() {
    println!("Example 1: Multi-Stage Normalization Pipeline");
    println!("{}", LINE);

    let original = "Hello, WORLD!!! This    is   a TEST!!!";
    let modified = "hello world this is a test";

    // Build a comprehensive normalization pipeline
    let pipeline = TextPipeline::new()
        .add_normalizer(Box::new(Lowercase))
        .add_normalizer(Box::new(RemovePunctuation))
        .add_normalizer(Box::new(
            WhitespaceNormalizer::new()
                .with_collapse(true)
                .with_trim(true),
        ));

    let config = DiffConfig::default()
        .with_pipeline(pipeline)
        .with_tokenizer(Box::new(WordTokenizer::new()));

    let engine = DiffEngine::new(config);
    let result = engine.diff(original, modified);

    println!("Original: \"{}\"", original);
    println!("Modified: \"{}\"", modified);
    println!("\nPipeline stages:");
    println!("  1. Lowercase");
    println!("  2. Remove punctuation");
    println!("  3. Normalize whitespace (collapse + trim)");
    println!("\nResult:");
    println!("  Semantic similarity: {:.2}", result.semantic_similarity);
    println!(
        "  Changes: {:.1}%",
        result.statistics.change_percentage * 100.0
    );
    println!("\n");
}

fn example_tokenization_strategies() {
    println!("Example 2: Different Tokenization Strategies");
    println!("{}", LINE);

    let text1 = "hello world";
    let text2 = "hello rust";

    // Character-level tokenization
    {
        println!("\nCharacter-level tokenization:");
        let config = DiffConfig::default().with_tokenizer(Box::new(CharacterTokenizer));

        let engine = DiffEngine::new(config);
        let result = engine.diff(text1, text2);

        println!("  Total operations: {}", result.operations.len());
        println!("  Changed characters: {}", result.statistics.edit_distance);
    }

    // Word-level tokenization
    {
        println!("\nWord-level tokenization:");
        let config = DiffConfig::default().with_tokenizer(Box::new(WordTokenizer::new()));

        let engine = DiffEngine::new(config);
        let result = engine.diff(text1, text2);

        println!("  Total operations: {}", result.operations.len());
        println!("  Changed words: {}", result.changed_operations().len());
    }

    // Bigram tokenization
    {
        println!("\nCharacter bigram tokenization:");
        let config = DiffConfig::default()
            .with_tokenizer(Box::new(NGramTokenizer::new(2).character_level()));

        let engine = DiffEngine::new(config);
        let result = engine.diff(text1, text2);

        println!("  Total bigrams: {}", result.statistics.total_tokens);
    }

    println!("\n");
}

fn example_character_ngrams() {
    println!("Example 3: Character N-Gram Diff");
    println!("{}", LINE);

    let original = "algorithm";
    let modified = "logarithm";

    // Use character trigrams
    let config =
        DiffConfig::default().with_tokenizer(Box::new(NGramTokenizer::new(3).character_level()));

    let engine = DiffEngine::new(config);
    let result = engine.diff(original, modified);

    println!("Original: {}", original);
    println!("Modified: {}", modified);
    println!("\nCharacter trigram analysis:");
    println!("  Similarity: {:.2}", result.semantic_similarity);

    println!("\nChanged trigrams:");
    for op in result.changed_operations().iter().take(5) {
        println!("  {}", op.description());
    }
    println!("\n");
}

fn example_sentence_diff() {
    println!("Example 4: Sentence-Level Diff");
    println!("{}", LINE);

    let original =
        "This is the first sentence. This is the second sentence. This is the third sentence.";
    let modified =
        "This is the first sentence. This is a modified sentence. This is the third sentence.";

    let config = DiffConfig::default().with_tokenizer(Box::new(SentenceTokenizer));

    let engine = DiffEngine::new(config);
    let result = engine.diff(original, modified);

    println!("Original text:");
    println!("  {}", original);
    println!("\nModified text:");
    println!("  {}", modified);

    println!("\nSentence-level changes:");
    for op in result.changed_operations() {
        match op.edit_type {
            redline_core::EditType::Delete => {
                println!("  [-] {}", op.original_text.as_ref().unwrap());
            }
            redline_core::EditType::Insert => {
                println!("  [+] {}", op.modified_text.as_ref().unwrap());
            }
            redline_core::EditType::Modify => {
                println!(
                    "  [~] {} -> {}",
                    op.original_text.as_ref().unwrap(),
                    op.modified_text.as_ref().unwrap()
                );
            }
            _ => {}
        }
    }
    println!("\n");
}
