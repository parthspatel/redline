//! Comprehensive example showing analyzers and classifiers

use redline_core::normalizers::WhitespaceNormalizer;
use redline_core::tokenizers::WordTokenizer;
use redline_core::{DiffConfig, DiffEngine, TextPipeline};

use redline_core::analyzer::group::{
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

// SpaCy analyzers (optional, enabled with 'spacy' feature)
#[cfg(feature = "spacy")]
use redline_core::analyzer::spacy::alignment::SpacyAlignmentAnalyzer;
#[cfg(feature = "spacy")]
use redline_core::analyzer::spacy::dependency_parse::SpacyDependencyAnalyzer;
#[cfg(feature = "spacy")]
use redline_core::analyzer::spacy::grammar::SpacyGrammarAnalyzer;
#[cfg(feature = "spacy")]
use redline_core::analyzer::spacy::part_of_speech::SpacyPOSAnalyzer;
#[cfg(feature = "spacy")]
use redline_core::analyzer::spacy::structural::SpacyStructuralAnalyzer;

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

    // Example 6: SpaCy syntactic analysis (if enabled)
    #[cfg(feature = "spacy")]
    example_spacy_analysis();
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

#[cfg(feature = "spacy")]
fn example_spacy_analysis() {
    println!("\nExample 6: SpaCy Syntactic Analysis");
    println!("{}", LINE);

    let engine = DiffEngine::default();

    // Test 1: Grammar fix detection
    println!("\n📝 Test 1: Grammar Fix Detection");
    let original1 = "She don't like apples";
    let corrected1 = "She doesn't like apples";
    let diff1 = engine.diff(original1, corrected1);

    println!("Original:  '{}'", original1);
    println!("Corrected: '{}'", corrected1);

    let grammar_analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
    let grammar_result = grammar_analyzer.analyze(&diff1);

    println!("\nGrammar Analysis:");
    if let Some(fixes) = grammar_result.metrics.get("grammar_fixes") {
        println!("  ✓ Grammar fixes detected: {}", fixes);
    }
    for insight in &grammar_result.insights {
        println!("  • {}", insight);
    }

    // Test 2: Word order detection
    println!("\n📝 Test 2: Word Order Changes");
    let original2 = "The big red car";
    let modified2 = "The red big car";
    let diff2 = engine.diff(original2, modified2);

    println!("Original: '{}'", original2);
    println!("Modified: '{}'", modified2);

    let alignment_analyzer = SpacyAlignmentAnalyzer::new("en_core_web_sm");
    let alignment_result = alignment_analyzer.analyze(&diff2);

    println!("\nAlignment Analysis:");
    if let Some(reorderings) = alignment_result.metrics.get("reorderings") {
        println!("  ⚠️  Reorderings detected: {}", reorderings);
    }
    if let Some(ratio) = alignment_result.metrics.get("alignment_ratio") {
        println!("  • Alignment ratio: {:.1}%", ratio * 100.0);
    }

    let structural_analyzer = SpacyStructuralAnalyzer::new("en_core_web_sm");
    let structural_result = structural_analyzer.analyze(&diff2);

    if let Some(similarity) = structural_result.metrics.get("structural_similarity") {
        println!("  • Structural similarity: {:.1}%", similarity * 100.0);
    }

    // Test 3: Comprehensive analysis
    println!("\n📝 Test 3: Comprehensive Syntactic Analysis");
    let original3 = "I seen the movie yesterday and it was good";
    let corrected3 = "I saw the movie yesterday and it was great";
    let diff3 = engine.diff(original3, corrected3);

    println!("Original:  '{}'", original3);
    println!("Corrected: '{}'", corrected3);

    let analyzers: Vec<Box<dyn SingleDiffAnalyzer>> = vec![
        Box::new(SpacyAlignmentAnalyzer::new("en_core_web_sm")),
        Box::new(SpacyPOSAnalyzer::new("en_core_web_sm")),
        Box::new(SpacyDependencyAnalyzer::new("en_core_web_sm")),
        Box::new(SpacyGrammarAnalyzer::new("en_core_web_sm")),
        Box::new(SpacyStructuralAnalyzer::new("en_core_web_sm")),
    ];

    println!("\nRunning all SpaCy analyzers:");
    for analyzer in &analyzers {
        let result = analyzer.analyze(&diff3);

        // Print key metrics
        println!("\n  {} - Key Metrics:", analyzer.name());
        for (metric, value) in result.metrics.iter().take(3) {
            println!("    • {}: {:.2}", metric, value);
        }

        // Print first insight if any
        if let Some(insight) = result.insights.first() {
            println!("    💡 {}", insight);
        }
    }

    println!("\n{}", LINE);
    println!("SpaCy analysis demonstrates:");
    println!("  ✓ Grammar error detection (verb forms, agreement)");
    println!("  ✓ Word order change detection");
    println!("  ✓ Structural similarity analysis");
    println!("  ✓ Fine-grained syntactic metrics\n");
}
