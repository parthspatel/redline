//! Example demonstrating offline training and online inference separation
//!
//! This example shows the recommended workflow:
//! 1. Offline: Train on historical similarity data to learn parameters
//! 2. Offline: Serialize learned parameters to JSON
//! 3. Online: Load parameters and use for fast inference

use redline_core::analyzer::ml::bert_semantic::{
    BertSemanticAnalyzer, BertSemanticTrainer, InferenceConfig, LearnedParameters,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BERT Semantic Analysis: Offline Training + Online Inference ===\n");

    // ========================================
    // PHASE 1: OFFLINE TRAINING
    // ========================================
    println!("📚 Phase 1: Offline Training");
    println!("----------------------------");

    // Create a trainer
    let mut trainer = BertSemanticTrainer::new()
        .with_threshold_percentile(0.5) // Use median as base
        .with_max_history(10000);

    // Simulate historical similarity data from your application
    // In production, this would come from logged similarity scores
    println!("Adding training samples...");

    // Bimodal distribution: low similarities (dissimilar) and high (similar)
    let low_similarities: Vec<f64> = (0..100).map(|i| 0.2 + (i as f64 / 100.0) * 0.2).collect();
    let high_similarities: Vec<f64> = (0..100).map(|i| 0.7 + (i as f64 / 100.0) * 0.25).collect();

    trainer.add_samples(&low_similarities);
    trainer.add_samples(&high_similarities);

    println!("✓ Added {} training samples", trainer.sample_count());

    // Train and learn parameters
    println!("\nTraining threshold and statistical parameters...");
    let learned_params = trainer.train()?;

    println!("✓ Training complete!");
    println!("  Learned threshold: {:.3}", learned_params.threshold);
    println!("  Mean: {:.3}", learned_params.mean);
    println!("  Std Dev: {:.3}", learned_params.std_dev);
    println!("  Sample count: {}", learned_params.sample_count);

    // Serialize to JSON for storage
    #[cfg(feature = "bert")]
    {
        let json = learned_params.to_json()?;
        println!("\n📝 Serialized parameters (first 100 chars):");
        println!("  {}", &json[..json.len().min(100)]);

        // In production, save to file:
        // std::fs::write("bert_params.json", json)?;
    }

    // ========================================
    // PHASE 2: ONLINE INFERENCE
    // ========================================
    println!("\n\n🚀 Phase 2: Online Inference");
    println!("----------------------------");

    // In production, load from file:
    // let json = std::fs::read_to_string("bert_params.json")?;
    // let params = LearnedParameters::from_json(&json)?;

    // Create analyzer with learned parameters
    let analyzer = BertSemanticAnalyzer::new(learned_params.clone()).with_config(InferenceConfig {
        outlier_z_threshold: 2.5,
        drift_std_multiplier: 2.0,
        entropy_bins: 10,
        include_all_metrics: true,
    });

    println!("✓ Analyzer initialized with pre-learned parameters");
    println!("  No runtime learning required!");
    println!("  Fast inference with statistical context");

    // ========================================
    // DEMONSTRATION
    // ========================================
    println!("\n\n📊 Inference Examples");
    println!("--------------------");

    // Example 1: Typical similarity (should be confident, no outlier)
    let typical_sim = 0.75;
    println!("\nExample 1: Typical similarity ({:.2})", typical_sim);
    demonstrate_inference(&analyzer, &learned_params, typical_sim);

    // Example 2: Low similarity (might be drift/outlier)
    let low_sim = 0.15;
    println!("\nExample 2: Low similarity ({:.2})", low_sim);
    demonstrate_inference(&analyzer, &learned_params, low_sim);

    // Example 3: Outlier high similarity
    let high_sim = 0.98;
    println!("\nExample 3: Very high similarity ({:.2})", high_sim);
    demonstrate_inference(&analyzer, &learned_params, high_sim);

    println!(
        "\n\n✅ Complete! Offline training enables fast, statistically-informed online inference."
    );

    Ok(())
}

fn demonstrate_inference(
    _analyzer: &BertSemanticAnalyzer,
    learned_params: &LearnedParameters,
    similarity: f64,
) {
    // Compute metrics using learned parameters
    let config = InferenceConfig::default();

    // Confidence based on learned distribution
    let distance_from_mean = (similarity - learned_params.mean).abs();
    let normalized_distance = if learned_params.std_dev > 0.0 {
        distance_from_mean / learned_params.std_dev
    } else {
        0.0
    };
    let confidence = 1.0 / (1.0 + normalized_distance);

    // Z-score for outlier detection
    let z_score = if learned_params.std_dev > 0.0 {
        (similarity - learned_params.mean) / learned_params.std_dev
    } else {
        0.0
    };
    let is_outlier = z_score.abs() > config.outlier_z_threshold;

    // Drift detection
    let drift_threshold =
        learned_params.mean - (config.drift_std_multiplier * learned_params.std_dev);
    let is_drift = similarity < drift_threshold;

    // Classification
    let classification = if similarity >= learned_params.threshold {
        "SIMILAR"
    } else {
        "DISSIMILAR"
    };

    println!("  Classification: {}", classification);
    println!("  Confidence: {:.2}%", confidence * 100.0);
    println!("  Z-score: {:.2}", z_score);
    println!("  Outlier: {}", if is_outlier { "YES ⚠️" } else { "No" });
    println!(
        "  Semantic Drift: {}",
        if is_drift { "YES ⚠️" } else { "No" }
    );
}
