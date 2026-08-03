# Anka

[![CI](https://github.com/mustaffadnC/Anka/actions/workflows/ci.yml/badge.svg)](https://github.com/mustaffadnC/Anka/actions/workflows/ci.yml)

A vector search engine written from scratch in Rust: HNSW index, int8 scalar quantization,
metadata filtering, durability (WAL + snapshot), and an HTTP API.

> **Built with AI coding tools.** Much of the code here was scaffolded and iterated using Claude
> Code. The benchmark methodology — which datasets, which baseline, which parameters held equal,
> and what the roofline says the ceiling is — is mine, and so is every conclusion drawn from it.

**The point of this project is the measurement, not the implementation.** Plenty of
repositories say "I implemented HNSW". This one aims to say: *I implemented HNSW, plotted
recall/QPS Pareto curves on SIFT1M and GloVe-100, compared them against hnswlib under
identical parameters on identical hardware, and profiled where the difference comes from.*

> **Status: under construction.** Phases 0 and 1 are done: the distance kernels are verified
> against the published SIFT1M and GloVe-100 ground truth, and exact brute-force search is in
> place as the reference every later phase is measured against. The HNSW index itself is next,
> so there are no recall/QPS curves yet — and there will be none until they come from a script
> anyone can re-run. See [docs/RESULTS.md](docs/RESULTS.md) for what has actually been measured.
>
> One early result worth the click: brute-force scan over SIFT1M runs at **49.4 GB/s**, which is
> this machine's DDR5 ceiling. The AVX2 kernel is 12.9× faster than the reference *in cache* and
> only 1.20× faster on the full dataset — it is waiting on memory, not arithmetic. Which is the
> argument for building an index at all.

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
transactions, GPU indexing, an AVX-512 code path (see [docs/DESIGN.md](docs/DESIGN.md) §10), and
any claim of production readiness.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Workspace, CI, `.fvecs`/`.ivecs` readers, `VectorStore` | ✅ done |
| 1 | Distance metrics (scalar + AVX2), brute force, ground truth | ✅ done |
| 2 | HNSW core, `ef`/`M` sweeps, hnswlib comparison | 🚧 next |
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
