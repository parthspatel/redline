//! Criterion benchmarks for diff algorithms: scaling, similarity sweep, algorithm comparison.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use redline_core::diff::{DiffAlgorithm, Histogram, Myers};

/// Deterministic token pair generator using simple xorshift PRNG.
fn generate_token_pair(len: usize, similarity: f64, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let vocab_size = 100u32;
    let mut state = seed;
    let mut next = || -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let source: Vec<u32> = (0..len)
        .map(|_| (next() % vocab_size as u64) as u32)
        .collect();
    let target: Vec<u32> = source
        .iter()
        .map(|&s| {
            if (next() % 1000) as f64 / 1000.0 < similarity {
                s
            } else {
                (next() % vocab_size as u64) as u32
            }
        })
        .collect();
    (source, target)
}

fn bench_myers_100(c: &mut Criterion) {
    let (source, target) = generate_token_pair(100, 0.8, 42);
    c.bench_function("myers_100_tokens_80pct", |b| {
        b.iter(|| Myers::new().compute(black_box(&source), black_box(&target), None))
    });
}

fn bench_myers_10k_dissimilar(c: &mut Criterion) {
    let (source, target) = generate_token_pair(10000, 0.0, 42);
    let mut group = c.benchmark_group("myers_10k_dissimilar");
    group.sample_size(10);
    group.bench_function("with_threshold_4096", |b| {
        b.iter(|| Myers::with_threshold(4096).compute(black_box(&source), black_box(&target), None))
    });
    group.finish();
}

fn bench_myers_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("myers_scaling_90pct");
    for &size in &[10, 50, 100, 500, 1000, 5000] {
        let (source, target) = generate_token_pair(size, 0.9, 42);
        group.bench_with_input(
            BenchmarkId::new("tokens", size),
            &(source, target),
            |b, (s, t)| b.iter(|| Myers::new().compute(black_box(s), black_box(t), None)),
        );
    }
    group.finish();
}

fn bench_histogram_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_scaling_90pct");
    for &size in &[10, 50, 100, 500, 1000, 5000] {
        let (source, target) = generate_token_pair(size, 0.9, 42);
        group.bench_with_input(
            BenchmarkId::new("tokens", size),
            &(source, target),
            |b, (s, t)| b.iter(|| Histogram::new().compute(black_box(s), black_box(t), None)),
        );
    }
    group.finish();
}

fn bench_affix_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("common_affix_impact_1000");
    let (source, target) = generate_token_pair(1000, 0.9, 42);
    group.bench_function("with_affixes", |b| {
        b.iter(|| Myers::new().compute(black_box(&source), black_box(&target), None))
    });
    group.bench_function("without_affixes", |b| {
        b.iter(|| {
            Myers::new().without_common_affixes().compute(
                black_box(&source),
                black_box(&target),
                None,
            )
        })
    });
    group.finish();
}

fn bench_similarity_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("myers_similarity_sweep_1000");
    for &sim in &[0.0f64, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0] {
        let (source, target) = generate_token_pair(1000, sim, 42);
        let label = format!("sim_{:.0}pct", sim * 100.0);
        group.bench_with_input(
            BenchmarkId::new("similarity", &label),
            &(source, target),
            |b, (s, t)| b.iter(|| Myers::new().compute(black_box(s), black_box(t), None)),
        );
    }
    group.finish();
}

fn bench_algo_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison_1000");
    let (source, target) = generate_token_pair(1000, 0.8, 42);
    group.bench_function("myers", |b| {
        b.iter(|| Myers::new().compute(black_box(&source), black_box(&target), None))
    });
    group.bench_function("histogram", |b| {
        b.iter(|| Histogram::new().compute(black_box(&source), black_box(&target), None))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_myers_100,
    bench_myers_10k_dissimilar,
    bench_myers_scaling,
    bench_histogram_scaling,
    bench_affix_impact,
    bench_similarity_sweep,
    bench_algo_comparison,
);
criterion_main!(benches);
