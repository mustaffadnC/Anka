# Anka — Results

Every number in this document comes from a command recorded next to it. Nothing here is an
estimate, an expectation, or a figure typed in by hand. Sections are filled in as phases land,
not at the end.

**Status:** phase 0 in progress. No measurements recorded yet.

---

## 0. Environment

Written automatically by `scripts/run_benchmarks.sh` into `docs/results/env-<date>.txt` and
summarised here for the run that produced the published numbers.

| Field | Value |
|---|---|
| CPU | *pending* |
| RAM | *pending* |
| OS / kernel / WSL version | *pending* |
| `rustc -V` | *pending* |
| `RUSTFLAGS` | *pending* |
| RNG seed | *pending* |
| Clock-variance handling | *pending* |
| Query threads | 1 |

### Datasets

| Dataset | Vectors | Dim | Metric | SHA256 verified |
|---|---|---|---|---|
| SIFT1M | 1 000 000 | 128 | L2² | *pending* |
| siftsmall | 10 000 | 128 | L2² | *pending* |
| GloVe-100 | 1 183 514 | 100 | cosine | *pending* |
| Synthetic | 100 000 | 768 | cosine | generated, seed *pending* |

---

## 1. Phase 0 — Loading and memory

| Dataset | Load time | Peak RSS | Notes |
|---|---|---|---|
| *pending* | | | |

---

## 2. Phase 1 — Distance kernels and ground truth

### SIMD vs scalar

| Metric | Dim | Scalar (ns/op) | SIMD (ns/op) | Speed-up |
|---|---|---|---|---|
| *pending* | | | | |

### Ground truth verification

| Check | Result |
|---|---|
| Scalar-generated GT vs official SIFT1M `.ivecs` | *pending* (target: 100%) |
| SIMD-generated GT vs scalar GT | *pending* (target: distance-equivalent) |
| SIMD ≡ scalar, relative tolerance 1e-6 (proptest) | *pending* |

---

## 3. Phase 2 — HNSW

### recall/QPS Pareto — SIFT1M (`M=16, ef_construction=200`)

| `ef` | recall@10 | QPS | p50 (µs) | p95 (µs) | p99 (µs) |
|---|---|---|---|---|---|
| *pending* | | | | | |

### recall/QPS Pareto — GloVe-100 (cosine)

| `ef` | recall@10 | QPS | p50 (µs) | p95 (µs) | p99 (µs) |
|---|---|---|---|---|---|
| *pending* | | | | | |

### `M` sweep

| `M` | recall@10 | Graph memory | Build time |
|---|---|---|---|
| *pending* | | | |

### hnswlib comparison

Same dataset, same `M` / `ef_construction` / `ef`, same machine, both built with native
optimisation, single-threaded, hnswlib called through its batch API with `num_threads=1`.

| `ef` | Anka recall | Anka QPS | hnswlib recall | hnswlib QPS | Ratio |
|---|---|---|---|---|---|
| *pending* | | | | | |

Where the difference comes from: *pending profile*.

### Ablations

| Change | recall@10 | Notes |
|---|---|---|
| Baseline | *pending* | |
| `SELECT_NEIGHBORS_HEURISTIC` disabled | *pending* | expected: large drop |
| `keepPrunedConnections` disabled | *pending* | also report degree distribution |

### Distance computations

Measured with `--features stats`. **These runs are not the QPS runs** — the counter sits in the
hot loop.

| `ef` | Distance computations / query |
|---|---|
| *pending* | |

---

## 4. Phase 3 — Persistence

| Check | Result |
|---|---|
| 1M index write → reload, results bit-identical | *pending* |
| WAL replay path, results bit-identical | *pending* |
| mmap load time vs full read | *pending* |
| `kill -9` during WAL write, `fsync=always` | *pending* |
| `kill -9` during snapshot write | *pending* |
| Torn WAL: truncated `len` / truncated payload / bad CRC / `seq` gap | *pending* |
| Corrupt snapshot: bad magic / unknown version / bad header CRC | *pending* |

**What the crash test proves:** recovery survives a torn log and an interrupted snapshot, and no
record acknowledged under `fsync=always` is lost to a process kill.
**What it does not prove:** power-loss durability. `kill -9` leaves the page cache intact, so it
cannot demonstrate that. No such claim is made.

---

## 5. Phase 4 — Deletion and filtering

### Deletion and compaction

| Check | Result |
|---|---|
| recall after delete + compaction returns to pre-delete level | *pending* |
| `ExternalId`s preserved across compaction | *pending* |
| Writes arriving during compaction are not lost | *pending* |

### Selectivity sweep

Selectivity = fraction of vectors **passing** the filter. Recall is measured against **filtered**
ground truth.

| Target selectivity | Measured | Strategy chosen | Filtered recall@10 | p50 (µs) | p95 (µs) |
|---|---|---|---|---|---|
| 0.1% | | | | | |
| 1% | | | | | |
| 5% | | | | | |
| 20% | | | | | |
| 50% | | | | | |
| 90% | | | | | |

### Strategy crossover

All three strategies measured at every selectivity level, so the adaptive choice can be shown to
be the right one. Figure: `figures/filter-crossover.png` — *pending*.

---

## 6. Phase 5 — Quantization

| Configuration | Resident vector data | On disk | recall@10 | QPS |
|---|---|---|---|---|
| fp32 | *pending* | | | |
| u8, no rescoring | *pending* (target: 4x less resident) | | | |
| u8 + rescoring | *pending* | | | |

| Check | Result |
|---|---|
| Integer kernel ≡ dequantize-then-fp32 (saturation test) | *pending* |
| `rescore_factor` sweep `[1, 2, 3, 5]` | *pending* |
| Global scale vs per-vector scale + correction terms | *pending* |
| u8 kernel speed-up over fp32 kernel | *pending* |

The 4x figure refers to **resident** vector data. Full-precision vectors remain in the memory
mapping because rescoring needs them; total footprint is reported separately.

---

## 7. Phase 6 — API

| Check | Result |
|---|---|
| `docker compose up` works | *pending* |
| Integration tests pass | *pending* |
| Search under load does not block `/health` | *pending* |
| `scripts/run_benchmarks.sh` reproduces everything in one command | *pending* |
