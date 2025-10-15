//! Demonstration of modular SpaCy analyzers
//!
//! This example shows how to use the individual SpaCy-based analyzers:
//! - SpacyPOSAnalyzer: Part-of-speech tag analysis
//! - SpacyDependencyAnalyzer: Dependency relation analysis
//! - SpacyGrammarAnalyzer: Grammar fix detection
//! - SpacyStructuralAnalyzer: Structural similarity analysis
//!
//! Run with:
//! ```bash
//! PYO3_PYTHON=$(pwd)/.venv/bin/python PYTHON_HOME=$(pwd)/.venv \
//!     cargo run --example spacy_modular_demo --features spacy
//! ```

use redline_core::analyzer::SingleDiffAnalyzer;
use redline_core::analyzer::spacy::{
    SpacyDependencyAnalyzer, SpacyGrammarAnalyzer, SpacyPOSAnalyzer, SpacyStructuralAnalyzer,
};
use redline_core::{DiffConfig, DiffEngine};

fn print_separator() {
    println!("\n{}", "=".repeat(60));
}

fn print_analysis(title: &str, result: &redline_core::analyzer::AnalysisResult) {
    println!("\n{}", title);
    println!("{}", "-".repeat(60));

    println!("\nMetrics:");
    for (key, value) in &result.metrics {
        println!("  • {}: {:.2}", key, value);
    }

    if !result.insights.is_empty() {
        println!("\nInsights:");
        for insight in &result.insights {
            println!("  • {}", insight);
        }
    }

    println!("\nConfidence: {:.1}%", result.confidence * 100.0);
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("🔍 Modular SpaCy Analyzers Demo");
    println!("{}", "=".repeat(60));

    // Create the diff engine
    let config = DiffConfig::default();
    let engine = DiffEngine::new(config);

    // Test cases
    let test_cases = vec![
        (
            "Grammar fix: subject-verb agreement",
            "The student have completed their homework.",
            "The student has completed their homework.",
        ),
        (
            "POS change: verb tense",
            "I walk to the store yesterday.",
            "I walked to the store yesterday.",
        ),
        (
            "Structural change: active to passive",
            "The dog chased the cat.",
            "The cat was chased by the dog.",
        ),
        (
            "Article correction",
            "I saw a elephant at the zoo.",
            "I saw an elephant at the zoo.",
        ),
        (
            "Pronoun correction",
            "Between you and I, this is interesting.",
            "Between you and me, this is interesting.",
        ),
    ];

    // Create analyzers
    println!("\n📊 Initializing analyzers...");
    let pos_analyzer = SpacyPOSAnalyzer::new("en_core_web_sm");
    let dep_analyzer = SpacyDependencyAnalyzer::new("en_core_web_sm");
    let grammar_analyzer = SpacyGrammarAnalyzer::new("en_core_web_sm");
    let structural_analyzer = SpacyStructuralAnalyzer::new("en_core_web_sm");

    // Run tests
    for (i, (description, original, modified)) in test_cases.iter().enumerate() {
        print_separator();
        println!("\nTest Case #{}: {}", i + 1, description);
        println!("  Original:  \"{}\"", original);
        println!("  Modified:  \"{}\"", modified);

        // Create diff
        let diff = engine.diff(original, modified);

        // Run all analyzers
        println!("\n🔎 Running all analyzers...");

        // 1. POS Analysis
        let pos_result = pos_analyzer.analyze(&diff);
        print_analysis("📝 Part-of-Speech Analysis", &pos_result);

        // 2. Dependency Analysis
        let dep_result = dep_analyzer.analyze(&diff);
        print_analysis("🔗 Dependency Analysis", &dep_result);

        // 3. Grammar Analysis
        let grammar_result = grammar_analyzer.analyze(&diff);
        print_analysis("✏️ Grammar Analysis", &grammar_result);

        // 4. Structural Analysis
        let structural_result = structural_analyzer.analyze(&diff);
        print_analysis("🏗️ Structural Analysis", &structural_result);
    }

    print_separator();
    println!("\n✅ Demo Complete!");
    print_separator();

    // Show comparison between modular and combined approach
    println!("\n💡 Benefits of Modular Analyzers:");
    println!("  • Fine-grained control: Use only what you need");
    println!("  • Better performance: Run specific analyses");
    println!("  • Clearer insights: Each analyzer focuses on one aspect");
    println!("  • Easier testing: Test individual components");
    println!("  • Flexible composition: Combine as needed");

    println!("\n📚 Available Analyzers:");
    println!("  • SpacyPOSAnalyzer - Part-of-speech tag changes");
    println!("  • SpacyDependencyAnalyzer - Dependency relation changes");
    println!("  • SpacyGrammarAnalyzer - Grammar fix detection");
    println!("  • SpacyStructuralAnalyzer - Structural similarity");
    println!("  • SpacySyntacticAnalyzer - Combined (legacy)");

    print_separator();
}
