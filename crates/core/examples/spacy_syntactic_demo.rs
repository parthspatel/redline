//! SpaCy Syntactic Analyzer Demo
//!
//! This example demonstrates the SpaCy-based syntactic analyzer which provides:
//! - Part-of-speech (POS) tagging
//! - Dependency parsing
//! - Grammar error detection
//! - Syntactic structure comparison
//!
//! ## Prerequisites
//! 1. Python 3.7+ with SpaCy installed: `pip install spacy`
//! 2. Download a SpaCy model: `python -m spacy download en_core_web_sm`
//! 3. Build with SpaCy feature: `cargo run --example spacy_syntactic_demo --features spacy`

#![allow(unused_imports)]

use redline_core::{DiffConfig, DiffEngine};

#[cfg(feature = "spacy")]
use redline_core::analyzer::SingleDiffAnalyzer;
#[cfg(feature = "spacy")]
use redline_core::analyzer::spacy::SpacySyntacticAnalyzer;

const LINE: &str = "========================================";
const SUBLINE: &str = "----------------------------------------";

fn main() {
    #[cfg(not(feature = "spacy"))]
    {
        println!("\n{}", LINE);
        println!("❌ SpaCy Feature Not Enabled");
        println!("{}", LINE);
        println!("\nThis example requires the 'spacy' feature to be enabled.");
        println!("\nTo run this example:");
        println!("  1. Install SpaCy: pip install spacy");
        println!("  2. Download model: python -m spacy download en_core_web_sm");
        println!("  3. Run with: cargo run --example spacy_syntactic_demo --features spacy");
        println!("\n{}", LINE);
        println!();
        return;
    }

    #[cfg(feature = "spacy")]
    {
        println!("\n{}", LINE);
        println!("🔍 SpaCy Syntactic Analyzer Demo");
        println!("{}", LINE);
        println!();

        // Example 1: Basic syntactic analysis
        example_basic_analysis();

        // Example 2: Grammar error detection
        example_grammar_detection();

        // Example 3: Syntactic structure comparison
        example_structure_comparison();

        // Example 4: POS tagging and dependencies
        example_pos_and_dependencies();

        // Example 5: Real-world editing scenarios
        example_real_world_edits();

        // Example 6: Diff analysis integration
        example_diff_integration();

        println!("\n{}", LINE);
        println!("✅ Demo Complete!");
        println!("{}", LINE);
        println!();
    }
}

#[cfg(feature = "spacy")]
fn example_basic_analysis() {

    println!("Example 1: Basic Syntactic Analysis");
    println!("{}", SUBLINE);

    let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

    let text = "The quick brown fox jumps over the lazy dog.";

    match analyzer.analyze_text(text) {
        Ok(tokens) => {
            println!("Text: \"{}\"\n", text);
            println!("Found {} tokens:\n", tokens.len());

            for (i, token) in tokens.iter().enumerate() {
                println!("Token #{}: \"{}\"", i + 1, token.text);
                println!("  Lemma:       {}", token.lemma);
                println!("  POS:         {} ({})", token.pos, token.tag);
                println!(
                    "  Dependency:  {} (head: token #{})",
                    token.dep,
                    token.head + 1
                );

                if token.is_stop {
                    println!("  [Stop word]");
                }
                if token.is_punct {
                    println!("  [Punctuation]");
                }
                println!();
            }
        }
        Err(e) => {
            println!("❌ Error: {}", e);
            println!("\nMake sure you have:");
            println!("  1. Installed SpaCy: pip install spacy");
            println!("  2. Downloaded the model: python -m spacy download en_core_web_sm");
        }
    }

    println!();
}

#[cfg(feature = "spacy")]
fn example_grammar_detection() {

    println!("\nExample 2: Grammar Error Detection");
    println!("{}", SUBLINE);

    let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

    let test_cases = vec![
        (
            "She don't like apples.",
            "She doesn't like apples.",
            "Subject-verb agreement",
        ),
        (
            "The cat sit on the mat.",
            "The cat sat on the mat.",
            "Verb tense correction",
        ),
        (
            "I seen it yesterday.",
            "I saw it yesterday.",
            "Irregular verb correction",
        ),
        (
            "He go to school every day.",
            "He goes to school every day.",
            "Third person singular",
        ),
    ];

    for (original, corrected, description) in test_cases {
        println!("\n📝 Test: {}", description);
        println!("  Original:  \"{}\"", original);
        println!("  Corrected: \"{}\"", corrected);

        match analyzer.compare_syntactic_structures(original, corrected) {
            Ok(comparison) => {
                println!("\n  Results:");
                println!("    POS changes:           {}", comparison.pos_changes);
                println!("    Dependency changes:    {}", comparison.dep_changes);
                println!(
                    "    Structural similarity: {:.1}%",
                    comparison.structural_similarity * 100.0
                );

                if !comparison.grammar_fixes.is_empty() {
                    println!("\n  ✅ Detected Grammar Fixes:");
                    for fix in &comparison.grammar_fixes {
                        println!("     • {}", fix);
                    }
                } else {
                    println!("\n  ℹ️  No specific grammar patterns detected");
                }
            }
            Err(e) => {
                println!("  ❌ Error: {}", e);
            }
        }
    }

    println!();
}

#[cfg(feature = "spacy")]
fn example_structure_comparison() {

    println!("\nExample 3: Syntactic Structure Comparison");
    println!("{}", SUBLINE);

    let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

    let comparisons = vec![
        (
            "The dog chased the cat.",
            "The cat was chased by the dog.",
            "Active vs. Passive Voice",
        ),
        (
            "I think that he is right.",
            "He is right, I think.",
            "Clause Reordering",
        ),
        ("The big red car.", "The red big car.", "Adjective Order"),
    ];

    for (text1, text2, description) in comparisons {
        println!("\n📊 Comparison: {}", description);
        println!("  Text 1: \"{}\"", text1);
        println!("  Text 2: \"{}\"", text2);

        match analyzer.compare_syntactic_structures(text1, text2) {
            Ok(comparison) => {
                println!("\n  Metrics:");
                println!(
                    "    Structural similarity:     {:.1}%",
                    comparison.structural_similarity * 100.0
                );
                println!(
                    "    POS distribution divergence: {:.3}",
                    comparison.pos_distribution_divergence
                );
                println!("    POS tag changes:           {}", comparison.pos_changes);
                println!("    Dependency changes:        {}", comparison.dep_changes);

                if comparison.structural_similarity > 0.8 {
                    println!("\n  ✅ High structural similarity - similar syntactic patterns");
                } else if comparison.structural_similarity > 0.5 {
                    println!("\n  ⚠️  Moderate structural changes detected");
                } else {
                    println!("\n  ⚠️  Significant structural differences");
                }
            }
            Err(e) => {
                println!("  ❌ Error: {}", e);
            }
        }
    }

    println!();
}

#[cfg(feature = "spacy")]
fn example_pos_and_dependencies() {
    use std::collections::HashMap;

    println!("\nExample 4: POS Distribution Analysis");
    println!("{}", SUBLINE);

    let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming technology.",
        "Run! Jump! Slide! Go!",
    ];

    for text in texts {
        println!("\nText: \"{}\"", text);

        match analyzer.analyze_text(text) {
            Ok(tokens) => {
                // Count POS tags
                let mut pos_counts: HashMap<String, usize> = HashMap::new();
                let mut dep_counts: HashMap<String, usize> = HashMap::new();

                for token in &tokens {
                    if !token.is_punct {
                        *pos_counts.entry(token.pos.clone()).or_insert(0) += 1;
                        *dep_counts.entry(token.dep.clone()).or_insert(0) += 1;
                    }
                }

                println!("\n  POS Tag Distribution:");
                let mut pos_vec: Vec<_> = pos_counts.iter().collect();
                pos_vec.sort_by(|a, b| b.1.cmp(a.1));
                for (pos, count) in pos_vec {
                    println!("    {}: {}", pos, count);
                }

                println!("\n  Dependency Relations:");
                let mut dep_vec: Vec<_> = dep_counts.iter().collect();
                dep_vec.sort_by(|a, b| b.1.cmp(a.1));
                for (dep, count) in dep_vec.iter().take(5) {
                    println!("    {}: {}", dep, count);
                }
            }
            Err(e) => {
                println!("  ❌ Error: {}", e);
            }
        }
    }

    println!();
}

#[cfg(feature = "spacy")]
fn example_real_world_edits() {

    println!("\nExample 5: Real-World Editing Scenarios");
    println!("{}", SUBLINE);

    let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

    let scenarios = vec![
        (
            "The company announced they was hiring.",
            "The company announced it was hiring.",
            "Pronoun-antecedent agreement",
        ),
        (
            "Between you and I, this is wrong.",
            "Between you and me, this is correct.",
            "Pronoun case correction",
        ),
        (
            "Neither the students nor the teacher were happy.",
            "Neither the students nor the teacher was happy.",
            "Complex subject-verb agreement",
        ),
        (
            "Who should I ask?",
            "Whom should I ask?",
            "Who/whom distinction",
        ),
    ];

    for (before, after, scenario) in scenarios {
        println!("\n🎯 Scenario: {}", scenario);
        println!("  Before: \"{}\"", before);
        println!("  After:  \"{}\"", after);

        match analyzer.compare_syntactic_structures(before, after) {
            Ok(comparison) => {
                println!("\n  Analysis:");
                println!(
                    "    Changes detected: {} POS, {} dependencies",
                    comparison.pos_changes, comparison.dep_changes
                );

                if !comparison.grammar_fixes.is_empty() {
                    println!("    Grammar corrections identified:");
                    for fix in &comparison.grammar_fixes {
                        println!("      • {}", fix);
                    }
                }

                if comparison.structural_similarity > 0.9 {
                    println!("\n  ✨ Minor structural change - likely a grammatical correction");
                }
            }
            Err(e) => {
                println!("  ❌ Error: {}", e);
            }
        }
    }

    println!();
}

#[cfg(feature = "spacy")]
fn example_diff_integration() {
    use redline_core::analyzer::SingleDiffAnalyzer;

    println!("\nExample 6: Integration with Diff Analysis");
    println!("{}", SUBLINE);

    let analyzer = SpacySyntacticAnalyzer::new("en_core_web_sm");

    let original = "The student have completed their homework and submitted it on time.";
    let revised = "The student has completed their homework and submitted it on time.";

    println!("Original: \"{}\"", original);
    println!("Revised:  \"{}\"\n", revised);

    // Create diff
    let engine = DiffEngine::default();
    let diff = engine.diff(original, revised);

    // Analyze with SpaCy
    let result = analyzer.analyze(&diff);

    println!("Syntactic Analysis Results:");
    println!("{}", SUBLINE);

    // Display metrics
    println!("\nMetrics:");
    for (metric, value) in &result.metrics {
        println!("  • {}: {:.2}", metric, value);
    }

    // Display insights
    if !result.insights.is_empty() {
        println!("\nInsights:");
        for insight in &result.insights {
            println!("  • {}", insight);
        }
    }

    // Display metadata
    if !result.metadata.is_empty() {
        println!("\nMetadata:");
        for (key, value) in &result.metadata {
            println!("  • {}: {}", key, value);
        }
    }

    println!("\nConfidence: {:.1}%", result.confidence * 100.0);

    println!();
}
