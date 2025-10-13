//! Comprehensive example showing analyzers and classifiers

use redline_core::normalizers::WhitespaceNormalizer;
use redline_core::tokenizers::WordTokenizer;
use redline_core::{DiffConfig, DiffEngine, TextPipeline};

use redline_core::analyzer::multi::{
    AggregateStatisticsAnalyzer, BehaviorClusteringAnalyzer, PatternDetectionAnalyzer,
    TemporalTrendAnalyzer,
};
use redline_core::analyzer::selectors::{DiffSelector, EditTypeSelector, ParagraphSelector};
// Import analyzers
use redline_core::analyzer::classifiers::{
    CategoryDistributionAnalyzer, EditIntentClassifier, ReadabilityAnalyzer,
    SemanticSimilarityAnalyzer, StylisticAnalyzer,
};
use redline_core::analyzer::{AnalysisReport, MultiDiffAnalyzer, SingleDiffAnalyzer};

// Import classifiers
use redline_core::analyzer::classifiers::{ChangeClassifier, RuleBasedClassifier};

const LINE: &str = "----------------------------------------";

fn main() {
    println!("=== Text Diff Analyzers & Classifiers Demo ===\n");

    // Example 1: Single diff analysis
    example_single_diff_analysis();

    // Example 2: Multi-diff analysis
    example_multi_diff_analysis();

    // Example 3: Using selectors
    example_with_selectors();

    // Example 4: Custom classification
    example_classification();

    // Example 5: Complete workflow for user edit analysis
    example_user_edit_analysis();
}

fn example_single_diff_analysis() {
    println!("Example 1: Single Diff Analysis");
    println!("{}", LINE);

    let generated = "The quick brown fox jumps over the lazy dog. This is a simple sentence.";
    let edited = "The fast brown fox leaps over the sleepy dog. This sentence is simple.";

    let engine = DiffEngine::default();
    let diff = engine.diff(generated, edited);

    println!("Original: {}", generated);
    println!("Edited:   {}", edited);
    println!();

    // Run multiple analyzers
    let analyzers: Vec<Box<dyn SingleDiffAnalyzer>> = vec![
        Box::new(SemanticSimilarityAnalyzer::new()),
        Box::new(ReadabilityAnalyzer::new()),
        Box::new(StylisticAnalyzer::new()),
        Box::new(CategoryDistributionAnalyzer::new()),
        Box::new(EditIntentClassifier::new()),
    ];

    let mut report = AnalysisReport::new();

    for analyzer in &analyzers {
        println!("\n📊 {} Analysis:", analyzer.name());
        let result = analyzer.analyze(&diff);

        // Print key metrics
        for (metric, value) in &result.metrics {
            println!("  • {}: {:.2}", metric, value);
        }

        // Print insights
        if !result.insights.is_empty() {
            println!("  Insights:");
            for insight in &result.insights {
                println!("    - {}", insight);
            }
        }

        report.add_result(result);
    }

    report.compute_summary();
    println!(
        "\n✅ Analysis complete! {} analyzers run.\n",
        report.results.len()
    );
}

fn example_multi_diff_analysis() {
    println!("\nExample 2: Multi-Diff Analysis");
    println!("{}", LINE);

    let engine = DiffEngine::default();

    // Simulate multiple user edits
    let edits = vec![
        ("AI generated content.", "Human edited content."),
        (
            "The system works correctly.",
            "The system functions properly.",
        ),
        ("Hello world!", "Hello world."),
        ("This is a TEST.", "This is a test."),
        (
            "Quick summary here.",
            "Here's a detailed explanation of the topic with multiple sentences.",
        ),
    ];

    let diffs: Vec<_> = edits
        .iter()
        .map(|(orig, modified)| engine.diff(orig, modified))
        .collect();

    let diff_refs: Vec<_> = diffs.iter().collect();

    println!("Analyzing {} edits...\n", diffs.len());

    // Run multi-diff analyzers
    let analyzers: Vec<Box<dyn MultiDiffAnalyzer>> = vec![
        Box::new(AggregateStatisticsAnalyzer::new()),
        Box::new(PatternDetectionAnalyzer::new()),
        Box::new(BehaviorClusteringAnalyzer::new(3)),
        Box::new(TemporalTrendAnalyzer::new()),
    ];

    for analyzer in &analyzers {
        println!("\n📈 {} Analysis:", analyzer.name());
        let result = analyzer.analyze(&diff_refs);

        // Print insights
        for insight in &result.insights {
            println!("  • {}", insight);
        }

        // Print key metrics
        if !result.metrics.is_empty() {
            println!("  Key Metrics:");
            for (metric, value) in result.metrics.iter().take(5) {
                println!("    - {}: {:.2}", metric, value);
            }
        }
    }

    println!();
}

fn example_with_selectors() {
    println!("\nExample 3: Using Selectors");
    println!("{}", LINE);

    let text1 = "# Introduction\nThis is the intro.\n\n# Main Content\nThis is the main section.\n\n# Conclusion\nFinal thoughts.";
    let text2 = "# Introduction\nThis is the intro.\n\n# Main Content\nThis section has been heavily modified with new content.\n\n# Conclusion\nFinal thoughts.";

    let engine = DiffEngine::default();
    let diff = engine.diff(text1, text2);

    println!("Analyzing specific sections...\n");

    // Select only insertions
    let insert_selector = EditTypeSelector::insertions();
    let insertions = insert_selector.select(&diff);
    println!("📝 Insertions only: {} operations", insertions.len());

    // Select by paragraph
    let para_selector = ParagraphSelector::single(1);
    let para_ops = para_selector.select(&diff);
    println!("📄 Paragraph 1: {} operations", para_ops.len());

    // Analyze just the selected operations
    if !insertions.is_empty() {
        println!("\nAnalyzing insertions:");
        for op in insertions.iter().take(3) {
            println!("  • {}", op.description());
        }
    }

    println!();
}

fn example_classification() {
    println!("\nExample 4: Change Classification");
    println!("{}", LINE);

    let engine = DiffEngine::default();

    let test_cases = vec![
        ("Hello World", "hello world", "Case change"),
        ("The cat sat.", "The dog sat.", "Semantic change"),
        ("I dont know", "I don't know", "Syntactic change"),
        ("quick", "fast", "Stylistic change"),
    ];

    let classifier = RuleBasedClassifier::new();

    println!("Classifying different types of changes:\n");

    for (original, modified, description) in test_cases {
        let diff = engine.diff(original, modified);

        println!("Test: {}", description);
        println!("  Original:  \"{}\"", original);
        println!("  Modified:  \"{}\"", modified);

        for op in diff.changed_operations() {
            let classification = classifier.classify_operation(op);
            println!(
                "  Classification: {:?} (confidence: {:.2})",
                classification.category, classification.confidence
            );
            if !classification.explanation.is_empty() {
                println!("  Explanation: {}", classification.explanation);
            }
        }
        println!();
    }
}

fn example_user_edit_analysis() {
    println!("\nExample 5: Complete User Edit Analysis Workflow");
    println!("{}", LINE);

    // Simulate AI-generated text that a user edited
    let generated = r#"
The quick brown fox jumps over the lazy dog. This sentence contains 
every letter of the alphabet. It is commonly used for testing purposes. 
The fox is very agile and quick.
"#;

    let edited = r#"
The swift brown fox leaps over the sleepy dog. This sentence demonstrates 
various letters of the alphabet and serves as an excellent test case. 
The fox displays remarkable agility and speed.
"#;

    println!("Analyzing user edits to AI-generated content...\n");
    println!(
        "Generated ({}  chars): {}",
        generated.len(),
        generated.trim()
    );
    println!("\nEdited ({} chars): {}", edited.len(), edited.trim());

    // Configure diff engine
    let config = DiffConfig::default()
        .with_pipeline(
            TextPipeline::new()
                .add_normalizer(Box::new(WhitespaceNormalizer::new().with_collapse(true))),
        )
        .with_tokenizer(Box::new(WordTokenizer::new()))
        .with_semantic_similarity(true)
        .with_edit_classification(true)
        .with_style_analysis(true);

    let engine = DiffEngine::new(config);
    let diff = engine.diff(generated, edited);

    println!("\n{}", LINE);
    println!("ANALYSIS RESULTS");
    println!("{}", LINE);

    // 1. Basic Statistics
    println!("\n1️⃣ Basic Statistics:");
    println!(
        "   • Semantic similarity: {:.1}%",
        diff.semantic_similarity * 100.0
    );
    println!(
        "   • Total changes: {:.1}%",
        diff.statistics.change_percentage * 100.0
    );
    println!("   • Insertions: {}", diff.statistics.insertions);
    println!("   • Deletions: {}", diff.statistics.deletions);
    println!("   • Modifications: {}", diff.statistics.modifications);

    // 2. Run single-diff analyzers
    println!("\n2️⃣ Detailed Analysis:");

    let similarity_analyzer = SemanticSimilarityAnalyzer::new();
    let similarity_result = similarity_analyzer.analyze(&diff);
    println!("\n   Semantic Similarity:");
    for insight in &similarity_result.insights {
        println!("     • {}", insight);
    }

    let readability_analyzer = ReadabilityAnalyzer::new();
    let readability_result = readability_analyzer.analyze(&diff);
    println!("\n   Readability:");
    for insight in readability_result.insights.iter().take(2) {
        println!("     • {}", insight);
    }

    let stylistic_analyzer = StylisticAnalyzer::new();
    let stylistic_result = stylistic_analyzer.analyze(&diff);
    println!("\n   Stylistic Changes:");
    for insight in stylistic_result.insights.iter().take(2) {
        println!("     • {}", insight);
    }

    let intent_analyzer = EditIntentClassifier::new();
    let intent_result = intent_analyzer.analyze(&diff);
    println!("\n   Edit Intents:");
    for insight in &intent_result.insights {
        println!("     • {}", insight);
    }

    // 3. Classify changes
    println!("\n3️⃣ Change Categories:");
    let classifier = RuleBasedClassifier::new();

    let mut category_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for op in diff.changed_operations() {
        let classification = classifier.classify_operation(op);
        let cat = format!("{:?}", classification.category);
        *category_counts.entry(cat).or_insert(0) += 1;
    }

    for (category, count) in category_counts.iter() {
        let pct = (*count as f64 / diff.changed_operations().len() as f64) * 100.0;
        println!("   • {}: {} operations ({:.1}%)", category, count, pct);
    }

    // 4. Key takeaways
    println!("\n4️⃣ Key Takeaways:");
    if diff.semantic_similarity > 0.7 {
        println!("   ✅ User preserved core meaning (high similarity)");
    } else {
        println!("   ⚠️  User made substantial semantic changes");
    }

    if diff.statistics.change_percentage < 0.3 {
        println!("   ✅ User made minor refinements (<30% changed)");
    } else {
        println!("   📝 User made significant revisions (>30% changed)");
    }

    println!("\n{}", LINE);
    println!("Analysis completed! Use these insights to improve generation.\n");
}
