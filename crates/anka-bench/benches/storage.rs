//! What the storage accessor costs.
//!
//! Phase 3 needs a store whose vectors live in two places: a snapshot mapping, read-only, with
//! write-ahead-log replay appended after it. That made `as_slice()` impossible to keep — there is no
//! single slice — so hot loops now take a resolved `Vectors` view and index it.
//!
//! The concern is that this moves a branch *into* the loop. Before, the storage variant was matched
//! once per search and the loop ran over a plain `&[f32]`; now every access goes through `get`. This
//! benchmark asks whether that is measurable, comparing three ways of scanning the same data:
//!
//! - `raw` — plain slice chunking, which is what the code did before
//! - `contiguous` — the view over an owned or fully-mapped store
//! - `split` — the view over a hybrid store, where the branch actually decides something
//!
//! 4096 vectors at dim 128 is 2 MiB and stays in L3, deliberately: this is a question about a
//! branch, not about DDR5.

use std::hint::black_box;

use anka_core::{L2Squared, Metric, Vectors};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const DIM: usize = 128;
const COUNT: usize = 4096;

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

/// The pre-refactor shape: chunk a slice and hand each chunk to the kernel.
fn scan_raw(query: &[f32], data: &[f32], dim: usize) -> f32 {
    data.chunks_exact(dim)
        .map(|v| L2Squared::distance(query, v))
        .sum()
}

/// The post-refactor shape: index a resolved view.
fn scan_view(query: &[f32], view: Vectors<'_>) -> f32 {
    let count = view.len();
    (0..count)
        .map(|index| L2Squared::distance(query, view.get(index)))
        .sum()
}

fn accessor(c: &mut Criterion) {
    let query = values(1, DIM);
    let data = values(2, DIM * COUNT);

    // A hybrid store with the split in the middle, so half the accesses take each arm and the
    // branch is as unpredictable as it can realistically get.
    let (mapped, owned) = data.split_at(DIM * COUNT / 2);

    let mut group = c.benchmark_group("storage/scan");
    group.throughput(Throughput::Elements((DIM * COUNT) as u64));

    group.bench_function(BenchmarkId::new("raw", DIM), |b| {
        b.iter(|| scan_raw(black_box(&query), black_box(&data), DIM));
    });

    group.bench_function(BenchmarkId::new("contiguous", DIM), |b| {
        let view = Vectors::Contiguous {
            data: &data,
            dim: DIM,
        };
        b.iter(|| scan_view(black_box(&query), black_box(view)));
    });

    group.bench_function(BenchmarkId::new("split", DIM), |b| {
        let view = Vectors::Split {
            mapped,
            owned,
            dim: DIM,
            split: COUNT / 2,
        };
        b.iter(|| scan_view(black_box(&query), black_box(view)));
    });

    group.finish();
}

criterion_group!(benches, accessor);
criterion_main!(benches);
