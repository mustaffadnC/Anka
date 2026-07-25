# Anka

A vector search engine written from scratch in Rust: HNSW index, int8 scalar quantization,
metadata filtering, durability (WAL + snapshot), and an HTTP API.

**The point of this project is the measurement, not the implementation.** Plenty of
repositories say "I implemented HNSW". This one aims to say: *I implemented HNSW, plotted
recall/QPS Pareto curves on SIFT1M and GloVe-100, compared them against hnswlib under
identical parameters on identical hardware, and profiled where the difference comes from.*

> **Status: under construction.** Phase 0 (skeleton + dataset loading) is in progress.
> No performance numbers are published yet — and none will be published until they come from
> a script anyone can re-run. See [docs/RESULTS.md](docs/RESULTS.md) for what has actually
> been measured so far, and the roadmap below for what is coming.

---

## Why build this

Off-the-shelf libraries (FAISS, hnswlib, Qdrant) are excellent but opaque. When recall drops,
when an index gets corrupted, or when performance collapses on a particular data distribution,
you are debugging a black box. Building the algorithm from the paper forces you to understand
what actually matters: how layers get assigned, why greedy search converges, what happens when
you prune a neighbour list, and how entry-point selection shapes traversal.

## Design goals

| # | Goal | Measurable criterion |
|---|---|---|
| H1 | Correct HNSW | `recall@10 ≥ 0.95` on SIFT1M |
| H2 | Acceptable performance | Within **3x** of hnswlib (single thread, equal recall, both built `native`) |
| H3 | Memory efficiency | int8 quantization cuts **resident** vector data 4x, ≤ 1 point recall loss after rescoring |
| H4 | Durability | With `fsync=always`, no committed data lost after `kill -9`; index stays consistent |
| H5 | Filtered search | Filtered `recall ≥ 0.90` across 0.1%–100% selectivity |
| H6 | Reproducibility | One command regenerates every curve (fixed seed, pinned versions) |

Explicit non-goals: distributed architecture, disk-resident index (DiskANN-style), ACID
transactions, GPU indexing, AVX-512 (not available on the development hardware), and any claim
of production readiness.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Workspace, CI, `.fvecs`/`.ivecs` reader, `VectorStore` | 🚧 in progress |
| 1 | Distance metrics (scalar + AVX2), brute force, ground truth | ⬜ |
| 2 | HNSW core, `ef`/`M` sweeps, hnswlib comparison | ⬜ |
| 3 | Snapshot, WAL, crash recovery | ⬜ |
| — | **Benchmark showcase** — the milestone that makes this repo self-contained | ⬜ |
| 4 | Deletion (tombstones + compaction), metadata filtering | ⬜ |
| 5 | int8 scalar quantization + rescoring | ⬜ |
| 6 | HTTP API, Docker, semantic-search demo | ⬜ |

## Architecture

```
HTTP API (axum)                              — phase 6
├── Collection
│   ├── VectorStore   raw fp32 (Owned | mmap)
│   ├── QuantStore    u8 codes               — phase 5
│   ├── HnswIndex     graph                  — phase 2
│   ├── IdMap         ExternalId ↔ NodeId
│   ├── MetaStore     NodeId → attributes    — phase 4
│   ├── FilterIndex   roaring bitmaps        — phase 4
│   └── Tombstones    deleted NodeIds        — phase 4
├── Persistence       WAL + snapshot         — phase 3
└── Distance kernels  L2² / cosine / dot     — phase 1
    ├── scalar (reference, f64 accumulator)
    └── SIMD   (AVX2, runtime dispatch)
```

See [docs/DESIGN.md](docs/DESIGN.md) for the data model, on-disk formats, and the design
decisions behind them.

## Getting started

Requires a Linux environment (or WSL2) and a stable Rust toolchain.

```bash
git clone https://github.com/mustaffadnC/Anka.git && cd Anka
```

```bash
./scripts/download_datasets.sh
```

```bash
cargo test --workspace
```

```bash
cargo run -p anka-cli --release -- load sift1m
```

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — architecture, data model, file formats
- [docs/RESULTS.md](docs/RESULTS.md) — every measurement, with the command that produced it
- [docs/decisions/](docs/decisions/) — architecture decision records

## References

- Malkov & Yashunin (2016), *Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs*, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- Aumüller, Bernhardsson & Faithfull, *ANN-Benchmarks*,
  [arXiv:1807.05614](https://arxiv.org/abs/1807.05614)

## License

MIT — see [LICENSE](LICENSE).
