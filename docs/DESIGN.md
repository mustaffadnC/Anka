# Anka — Design

This document describes what Anka is built out of and, where a choice was not obvious, why it
was made that way. Measurements live in [RESULTS.md](RESULTS.md); decisions with real
alternatives get their own record under [decisions/](decisions/).

---

## 1. Crate layout

```
crates/
├── anka-core     VectorStore, Storage, Metric, distance kernels, shared types
├── anka-index    HNSW, brute force, quantization
├── anka-store    WAL, snapshot, mmap
├── anka-cli      load, build, bench, query, checkpoint
├── anka-bench    criterion benchmarks
└── anka-server   HTTP API                                        (phase 6)
```

`anka-bench` is a workspace member rather than a top-level `benches/` directory, because Cargo
does not pick up a `benches/` folder at the workspace root — only inside a package.

## 2. Data model

### Identifiers

```rust
pub type ExternalId = u64;   // caller-supplied id, stable forever
pub type NodeId     = u32;   // graph slot, reassigned by compaction
```

These are deliberately distinct types. Deletion is implemented with tombstones, and compaction
eventually rebuilds the graph and compacts the `NodeId` space — while the caller's ids must not
move. `IdMap` (forward `HashMap` + reverse `Vec`) is therefore a first-class component with its
own snapshot section, not an afterthought.

### Vector storage

```rust
pub enum Storage {
    Owned(Vec<f32>),   // build / insert path
    Mapped(Mmap),      // loaded from a snapshot, zero-copy
}
```

Vectors are stored flat and row-major (`[v0_0..v0_d, v1_0..v1_d, ...]`), never as
`Vec<Vec<f32>>` — the latter turns every distance computation into pointer chasing and destroys
cache locality.

The `Mapped` variant is what makes snapshot loading cheap, and it is also what makes int8
quantization worthwhile: quantized codes stay resident in RAM while the full-precision vectors
needed for rescoring are paged in from the mapping on demand. `&[u8] → &[f32]` conversion goes
through `bytemuck`, which keeps the alignment and size checks in one audited place instead of
scattering `unsafe` transmutes across the codebase.

### Graph representation

Neighbour lists are flat arrays with a fixed-capacity slot per node — one allocation per layer,
cache-friendly, and directly serialisable (raw pointers are meaningless after a restart,
offsets are not).

The subtlety is that **upper layers are sparse**: with `M = 16`, roughly 1 node in 16 exists at
layer 1, 1 in 256 at layer 2, and so on. Allocating `count` slots per layer would waste about
68 MB per layer on a 1M-vector index — several hundred megabytes in total, dwarfing the graph
itself. Layers above 0 therefore carry a remap:

```rust
pub struct Layer {
    slot_of: Option<Vec<u32>>,  // layer 0: None (identity). Above: NodeId → slot, NO_SLOT if absent
    nodes: Vec<NodeId>,         // slot → NodeId (reverse map: iteration + serialisation)
    neighbors: Vec<u32>,        // per slot: [degree, n0, n1, ..., n_{max-1}]
    max_degree: usize,          // Mmax0 at layer 0, M above
}
```

Measured cost on 1M vectors with `M = 16, Mmax0 = 32`: 132 MB for layer 0, roughly 25 MB for
everything above it.

## 3. On-disk formats

### Snapshot (`collection.anka`)

```
[ Header 128B ]
  magic "ANKA" | format_version u32 | dim u32 | count u32 | live_count u32
  metric u8 | M u16 | Mmax0 u16 | ef_construction u16
  entry_point u32 | max_layer u8
  rng_seed u64            # reproducibility: layer assignment is seeded
  wal_seq u64             # this snapshot contains WAL records up to and including this seq
  header_crc32 u32        # covers the header only
  body_crc32 u32          # covers the body
  section_offsets 6 × u64 # vectors, layer_count, layers, tombstones, id_map, metadata
[ Vectors    ]  count × dim × 4B, fp32, starting at offset 128
[ LayerCount ]  count × 1B
[ Layers     ]  layer 0: dense slot array; layers ≥ 1: node_count + NodeId list + slot array
[ Tombstones ]  serialised roaring bitmap
[ IdMap      ]  entry_count + (ExternalId, NodeId) pairs
[ Metadata   ]  bincode
```

Two checksums with two different policies: `header_crc32` is verified on every open (cheap, and
it catches a truncated or foreign file immediately), while `body_crc32` is verified only on
`--verify`, in crash tests, and in CI. Scanning 700 MB on every open would defeat the entire
purpose of a lazy, zero-copy mapping. `section_offsets` exists so sections can be reached
without parsing sequentially, which a memory-mapped reader needs.

Snapshots are written atomically, and the order is not negotiable:

```
write collection.anka.tmp → fsync(file) → rename → fsync(directory)
```

Skipping the directory `fsync` means a crash can lose the rename even though the file contents
were durable.

### Write-ahead log (`wal.log`)

```
[ WAL header 16B ]  magic "AWAL" | format_version u32 | reserved

then a stream of records:
[ len u32 ][ crc32 u32 ][ seq u64 ][ op u8 ][ payload ]
  len   = byte count of (seq + op + payload), excluding len and crc32
  crc32 = crc32(seq ++ op ++ payload), excluding len

0x01 INSERT      ext_id u64 + level u8 + dim × f32 + meta_len u32 + meta
0x02 DELETE      ext_id u64
0x03 CHECKPOINT  snapshot_wal_seq u64
```

Two fields here are easy to leave out and painful to add later:

- **`seq`** — recovery means "replay records after the snapshot". Without a per-record sequence
  number there is no definition of *after*, and recovery cannot be written at all.
- **`level`** — HNSW layer assignment is random (`floor(-ln(U) · mL)`). If replay re-rolls the
  dice, the recovered graph is not the graph that was lost, and "identical results after
  restart" becomes untestable. The level is recorded, so replay is deterministic.

Recovery loads the snapshot, takes `S = header.wal_seq`, skips records with `seq ≤ S`, replays
the rest, and stops cleanly on the first of: fewer than 8 bytes remaining, payload shorter than
`len`, CRC mismatch, or a gap in `seq`. The tail from that point is truncated.

### Durability

| Mode | Behaviour | Guarantee |
|---|---|---|
| `always` | `fsync` after each record, *then* update memory | Process and OS crash (default) |
| `every_n` | `fsync` every N records | Process crash only |
| `never` | left to the OS | Process crash only |

The ordering matters as much as the `fsync` itself: if the in-memory index were updated first,
a search could return a record that never reached disk.

**What the crash test proves, and what it does not.** `kill -9` kills a *process*. Data that
already reached the kernel via `write()` survives in the page cache and is readable after
restart — so a `kill -9` test passes even with `fsync=never`. It demonstrates that recovery
handles a torn log and an interrupted snapshot; it does not demonstrate power-loss durability.
That would require fault injection, and this project does not claim it.

## 4. Distance metrics

```rust
/// CONTRACT: the returned value is a distance — SMALLER IS ALWAYS CLOSER.
/// Dot product returns -dot.
pub trait Metric {
    fn distance_scalar(a: &[f32], b: &[f32]) -> f32;  // reference, f64 accumulator
    fn distance(a: &[f32], b: &[f32]) -> f32;         // fast path (SIMD)
    fn preprocess(v: &mut [f32]) -> Result<(), VectorError>;  // cosine → normalise
}

pub enum MetricKind { L2Squared, Cosine, Dot }
```

The contract carries its weight: if the trait exposed the direction of comparison instead
(`is_similarity() -> bool`), then every heap, every stopping condition, every pruning step and
the rescoring pass would have to branch on it. `MetricKind` exists because the metric is a
runtime property — it comes out of a snapshot header or an HTTP request — while the trait is
static so the inner loops can monomorphise. Dispatch happens once, at that boundary.

Ordering is explicit, because `f32` has no `Ord` and `partial_cmp().unwrap()` is a latent panic:

```rust
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist).then(self.id.cmp(&o.id))
    }
}
```

Ties break towards the smaller id, everywhere. This is a prerequisite for matching a published
ground truth exactly rather than at 99.9%.

**Numerical tolerance.** SIMD and scalar kernels are compared with a *relative* tolerance
(`1e-6`), never an absolute one. On SIFT, components are 0–255 over 128 dimensions, so squared
L2 reaches ~8.3e6; with fp32 epsilon at ~1.19e-7 the expected absolute error after 128
accumulations is on the order of 10. An absolute threshold like `1e-5` cannot hold there, while
the same relative threshold works for both SIFT (~1e6) and normalised GloVe (~1).

The tolerance is relative to **what the sum accumulates**, `Σ|aᵢbᵢ|`, not to the result. For
squared L2 the two are the same thing, since every term is non-negative. For a dot product they
are not: with mixed signs the terms cancel, so the result can land arbitrarily close to zero
while the absolute error stays exactly where it was. A test written against the result passes on
positive data and fails the moment signs are involved.

## 5. Ground truth

Exact neighbour lists come from the reference kernel, and the published SIFT1M list is the only
external check this project has on its own arithmetic.

The bar is that our **distance profile** matches the published one: for every query and every
rank, the distance to our neighbour equals the distance to theirs. Both lists are sorted
ascending by distance, so that makes the two sorted distance sequences identical — and a list of
`k` items whose distance sequence equals that of a known-exact top-`k` is itself an exact
top-`k`. Anything less than exact would have to contain an item beyond the true `k`-th distance,
and that would show up in the profile.

Id equality is *not* the bar, because it is unachievable and because it does not imply
correctness. Measured on SIFT1M with `k = 100`:

| | siftsmall | SIFT1M |
|---|---|---|
| Positions with the same id | 9 982 / 10 000 | 980 481 / 1 000 000 |
| Rows identical in order | 91 / 100 | 4 446 / 10 000 |
| Rows with the same neighbour **set** | 100 / 100 | 9 889 / 10 000 |
| Differing positions at equal distance | 18 / 18 | 19 519 / 19 519 |
| Largest relative distance gap | `0.000e0` | `0.000e0` |

Every single disagreement is between neighbours at **bit-identical** distances. Two separate
effects produce them, and neither is a defect:

- **Tie order is arbitrary.** When two neighbours are exactly equidistant, which one comes first
  in the published list follows an undocumented rule. No tie-break reproduces it and none is
  more correct.
- **When a tie straddles rank `k`, the top-`k` set is not unique.** Several vectors compete for
  the last slot at the same distance; any choice is an exact answer. On SIFT1M this happens for
  111 of 10 000 queries.

This matters for recall as well as for correctness: `recall@k = |returned ∩ true_top_k| / k`
depends on the set, not the order, and the residual ambiguity is confined to the last slot of
111 queries.

## 6. HNSW notes

Implementation follows Malkov & Yashunin (arXiv:1603.09320), algorithms 1–5. Parameters:
`M = 16`, `Mmax0 = 2M = 32`, `mL = 1/ln(M)`, `ef_construction = 200`, query-time `ef` swept over
`[10 … 800]`.

Points where a straightforward reading of the paper produces a working-but-wrong index:

- **`SELECT_NEIGHBORS_HEURISTIC` (algorithm 4) is not optional.** Taking the nearest `M`
  candidates builds a graph in which every edge is short-range, no bridge survives between
  distant clusters, and greedy search gets stuck in local minima. The heuristic keeps a
  candidate only if it is closer to the query than to every already-selected neighbour —
  note that the comparison is `dist(e, q)` against `dist(e, r)`, not against `dist(r, q)`.
  `keepPrunedConnections` is enabled so degree does not fall below `M`; `extendCandidates` is
  not, because it slows construction considerably for little gain.
- **The pruning step inside insert is not optional either.** After adding bidirectional edges,
  every affected neighbour whose degree now exceeds `Mmax(lc)` must be re-selected through the
  same heuristic. Skip it and degree grows without bound as construction proceeds.
- **Visited set:** an epoch-stamped array, not a `HashSet` — clearing is O(1). The epoch starts
  at **1**, because the marks array is zero-initialised and an epoch of 0 would make the first
  query believe every node had already been visited.
- **Empty index and first insert** are special: there is no entry point to search from.
- **`ef` is clamped to `max(ef, k)`**, otherwise the beam is too narrow to return `k` results.
- **In `SEARCH_LAYER`, check `|W| < ef` before dereferencing `W.furthest()`** — the other order
  panics on an empty result set.

A `distance_computations` counter distinguishes algorithmic wins from micro-optimisation, but it
sits in the hottest loop, so it lives behind `#[cfg(feature = "stats")]`. Distance counts and
QPS come from different runs, and RESULTS.md says so wherever both appear.

## 7. Deletion and filtering (phase 4)

Deletion uses tombstones: the `NodeId` goes into a roaring bitmap and is filtered out of
results, but stays in the graph as a navigational bridge. As the tombstone ratio grows, finding
`k` live results takes a wider beam, so the effective beam is `ceil(ef / (1 - ratio))`. Past a
20% threshold the index is rebuilt; the rebuild serves from the old index and swaps atomically
via `arc-swap`, with writes that arrive mid-rebuild queued and replayed just before the swap.

Filtering evaluates a `Filter` tree into a roaring allow-list and then picks a strategy.
*Selectivity* here means **the fraction of vectors that pass** the filter, so 0.1% is the most
restrictive case (some literature uses the inverse convention).

| Strategy | When | How |
|---|---|---|
| Brute force | selectivity < 1% | scan the allow-list, skip the graph entirely |
| In-filtering | 1% ≤ selectivity < 60% | traverse the graph, exclude non-matches from *results* only |
| Post-filtering | selectivity ≥ 60% | search normally, over-fetch, filter, widen `ef` and retry if short |

The thresholds are configurable and will be set from measurement; the values above are starting
guesses.

The classic in-filtering mistake is excluding non-matching nodes from *traversal* as well as
from results. Those nodes are the graph's connective tissue — remove them and it fragments,
and recall collapses. Tombstones and in-filtering are in fact the same mechanism ("traverse but
do not collect") and share one abstraction.

Measuring any of this requires infrastructure that the datasets do not provide: SIFT and GloVe
carry no attributes, so metadata is generated synthetically with a fixed seed and controlled
selectivity, and **filtered ground truth** is computed by brute force over each allow-list.
Filtered recall cannot be measured against unfiltered ground truth — the reference set would
simply be the wrong set.

## 8. Quantization (phase 5)

Per-dimension quantiles (1%–99%) are used as **clipping bounds**, but the scale is **global**:
`scale = (hi - lo) / 255` with a single `offset`. The reason is arithmetic. With a per-dimension
scale, squared L2 becomes

```
Σ (a_d - b_d)² = Σ s_d² (qa_d - qb_d)²
```

— a weighted sum, which cannot collapse into one integer accumulation. With a global scale it is
`s² · Σ (qa_d - qb_d)²`: accumulate in integers, apply one multiply at the end. A per-*vector*
scale plus stored correction terms is the interesting middle ground and will be measured against
the global scheme.

The fast path is integer on both sides — the query is quantized with the same parameters, since
an integer kernel cannot consume an fp32 query. What is asymmetric is the **rescoring** step:
`k · rescore_factor` candidates are re-scored against full-precision vectors, so quantization
error only affects candidate *selection*, never the final ordering.

The kernel widens `u8 → i16` and uses `_mm256_madd_epi16` (i16×i16 → i32, no saturation) rather
than `_mm256_maddubs_epi16`. `maddubs` treats its first operand as unsigned and its second as
*signed*, and saturates the intermediate to i16: with `a ≤ 255` and `b ≤ 127`, a pair sum
reaches `2 · 255 · 127 = 64770`, well past i16's 32767. The result is silently wrong distances.
The widening path costs more instructions and stays correct:

```
Δ = a - b ∈ [-255, 255]                    fits i16
Δ² ≤ 65 025                                 fits i32
Σ over 128 dims ≤ 8 323 200 < 2³¹           i32 accumulator suffices
```

A dedicated test asserts the integer kernel agrees exactly with dequantize-then-fp32, which is
what catches a saturation bug.

**What "4x" refers to.** Rescoring needs the fp32 vectors, so they do not disappear — they stay
in the memory mapping. The 4x reduction is in *resident* vector data (512 MB → 128 MB on
SIFT1M). Total footprint across RAM and disk does not drop 4x, and RESULTS.md reports both
numbers separately.

## 9. Concurrency (phase 6)

Concurrent insert is out of scope, so there is a single writer: `RwLock<Collection>`, shared for
search, exclusive for writes. Search is CPU-bound, so it runs under `spawn_blocking` rather than
directly inside an async handler — otherwise a burst of queries starves the Tokio worker threads
and the whole server, health endpoint included, stops responding.

## 10. Development environment

Host: Windows 11 with an AMD Ryzen 5 7600X (Zen 4, 6C/12T), 32 GB DDR5.
Development and measurement happen in **WSL2 + Ubuntu**, which is also what CI runs, so POSIX
behaviour (`fsync`, directory `fsync`, `kill -9`, `/proc/self/status`) is the same in both.
WSL2 gets half the host's RAM by default — 15 GiB and 12 vCPUs here, comfortably more than the
~1.4 GB the largest datasets and their graphs need.

SIMD targets **AVX2**. Zen 4 does support AVX-512 (it was AMD's first consumer architecture to
do so, and `/proc/cpuinfo` reports `avx512f` on this machine), but no AVX-512 path is written:
Zen 4 double-pumps it over 256-bit datapaths, so on a bandwidth-bound distance kernel any gain
over AVX2 has to be measured rather than assumed, and a second SIMD path doubles the
correctness surface — each one needs its own scalar-equivalence property test. It stays a
stretch goal, and unlike most stretch goals it is one this hardware can actually validate.

Layout matters under WSL2, where `/mnt/c` is markedly slower than ext4:

| What | Where | Why |
|---|---|---|
| Source | `/mnt/c/...` | reachable from Windows editors |
| Build output | `CARGO_TARGET_DIR` on ext4 | compile times |
| Datasets | ext4, located via `$ANKA_DATASETS` | I/O measurements should mean something |
| Snapshot/WAL tests | ext4 (`$TMPDIR`) | realistic `fsync` semantics |

Benchmarks are built with `RUSTFLAGS="-C target-cpu=native"`; CI is not, since the runner is a
different machine and only correctness is checked there.

Two profiling caveats specific to this setup: WSL2 does not virtualise the PMU, so `perf`
hardware counters are unavailable — cache behaviour is measured with `cachegrind` (simulated,
useful as a ratio) or with AMD uProf against a Windows build, noted as such wherever it appears.
And `cpupower frequency-set` cannot work from inside a VM, so clock variance is handled with the
Windows power plan plus three repetitions and a median.

## 11. Reproducibility contract

1. Layer assignment uses `StdRng::seed_from_u64(seed)`; `thread_rng()` appears nowhere. The seed
   is recorded in the snapshot header and in RESULTS.md.
2. Assigned levels are written to the WAL, so replay is deterministic.
3. Ties break on `(dist, id)` everywhere.
4. The reference kernel accumulates in f64; SIMD is compared against it with a relative
   tolerance.
5. Toolchain pinned via `rust-toolchain.toml`; `Cargo.lock` is committed.
6. Dataset SHA256 sums are verified and reported.
7. `run_benchmarks.sh` writes an environment report per run: CPU, RAM, kernel and WSL version,
   `rustc -V`, `RUSTFLAGS`, seed, dataset hashes, and how clock variance was handled.
8. Construction may use rayon (time reported), but **query measurement is single-threaded** and
   the thread count is stated.
