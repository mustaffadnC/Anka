//! Distance kernel benchmarks.
//!
//! Each case scans a **block of 4096 vectors** rather than timing one call. Two reasons: a
//! single 128-dimension distance takes tens of nanoseconds, which is close enough to the
//! measurement overhead to be misleading; and a scan is how the kernel is actually used, so the
//! ratio it produces is the ratio that matters to a query.
//!
//! 4096 x 128 floats is 2 MiB, which sits in L3 on the development machine. That is
//! deliberate: this benchmark is meant to measure the kernel, not DDR5. Memory-bound behaviour
//! at full dataset size shows up in the phase 2 numbers instead.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native"`; see docs/DESIGN.md, section 10.

use std::hint::black_box;

use anka_core::{Cosine, DotProduct, L2Squared, Metric};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Vectors per scanned block.
const BLOCK: usize = 4096;

/// Deterministic pseudo-random data, so successive runs compare against each other rather than
/// against a different dataset.
fn values(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
            (unit - 0.5) * 200.0
        })
        .collect()
}

/// Sums the distances from `query` to every vector in `block`.
///
/// Summing rather than discarding keeps the compiler from eliminating the loop, and costs one
/// addition per vector next to a whole distance computation.
fn scan(query: &[f32], block: &[f32], dim: usize, distance: impl Fn(&[f32], &[f32]) -> f32) -> f32 {
    block.chunks_exact(dim).map(|v| distance(query, v)).sum()
}

fn bench_metric<M: Metric>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("distance/{}", M::NAME));

    // 100 is GloVe (and exercises the scalar tail), 128 is SIFT, 768 is a modern embedding.
    for dim in [100usize, 128, 768] {
        let query = values(1, dim);
        let block = values(2, dim * BLOCK);

        // One "element" is one dimension of one distance computation, so the reported
        // throughput is directly comparable across widths.
        group.throughput(Throughput::Elements((dim * BLOCK) as u64));

        group.bench_with_input(BenchmarkId::new("reference", dim), &dim, |b, &dim| {
            b.iter(|| {
                scan(
                    black_box(&query),
                    black_box(&block),
                    dim,
                    M::distance_scalar,
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |b, &dim| {
            b.iter(|| scan(black_box(&query), black_box(&block), dim, M::distance));
        });
    }

    group.finish();
}

fn benchmarks(c: &mut Criterion) {
    bench_metric::<L2Squared>(c);
    bench_metric::<DotProduct>(c);
    bench_metric::<Cosine>(c);
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
