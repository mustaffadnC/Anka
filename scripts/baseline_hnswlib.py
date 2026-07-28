#!/usr/bin/env python3
"""hnswlib reference measurement, for comparison against `anka bench`.

The point is not to beat hnswlib. It has years of optimisation behind it and this project has
weeks. The point is to be able to explain where the difference comes from, which requires a
comparison that is actually fair:

* same M, ef_construction, ef, k, and the same dataset
* the batch `knn_query` API with one thread, so Python's per-call overhead is amortised over the
  whole batch instead of being measured as if it were search time
* warm-up drawn from the tail of the query set, measurement from the head, matching `anka bench`

One caveat this script cannot fix and therefore reports: a pip wheel of hnswlib is a generic
build, while Anka's numbers come from `-C target-cpu=native`. That handicaps hnswlib, so a
favourable comparison against a wheel means less than it appears. Build hnswlib from source with
`-O3 -march=native` before drawing conclusions, and say which one was used.

Usage:
    python3 scripts/baseline_hnswlib.py sift1m
    python3 scripts/baseline_hnswlib.py glove100 --ef 40,80,160,320,800
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

try:
    import hnswlib
except ImportError:
    sys.exit("hnswlib is not installed. Run ./scripts/setup-wsl.sh, or pip install hnswlib.")

# Mirrors the DatasetSpec table in anka-cli: directory, filename prefix, hnswlib space name.
DATASETS = {
    "siftsmall": ("siftsmall", "siftsmall", "l2"),
    "sift1m": ("sift", "sift", "l2"),
    "glove100": ("glove100", "glove100", "cosine"),
}


def read_fvecs(path: Path) -> np.ndarray:
    """Reads `.fvecs`: per record, an int32 dimension followed by that many float32s."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        sys.exit(f"{path}: file is empty")
    dim = int(raw[0])
    if raw.size % (dim + 1) != 0:
        sys.exit(f"{path}: {raw.size * 4} bytes is not a whole number of {dim}-dim records")
    return raw.reshape(-1, dim + 1)[:, 1:].view(np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        sys.exit(f"{path}: file is empty")
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:]


def percentile(sorted_us: np.ndarray, fraction: float) -> float:
    if sorted_us.size == 0:
        return 0.0
    rank = min(max(int(np.ceil(sorted_us.size * fraction)), 1), sorted_us.size)
    return float(sorted_us[rank - 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--datasets-dir", type=Path, default=None)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", default="10,20,40,80,160,320,512,800")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N base vectors")
    ap.add_argument("--queries", type=int, default=None)
    args = ap.parse_args()

    root = args.datasets_dir or Path(
        os.environ.get("ANKA_DATASETS", Path.home() / "anka-datasets")
    )
    directory, prefix, space = DATASETS[args.dataset]
    base_dir = root / directory
    if not base_dir.is_dir():
        sys.exit(f"{base_dir} does not exist — run ./scripts/download_datasets.sh {args.dataset}")

    base = read_fvecs(base_dir / f"{prefix}_base.fvecs")
    queries = read_fvecs(base_dir / f"{prefix}_query.fvecs")
    truth = read_ivecs(base_dir / f"{prefix}_groundtruth.ivecs")

    if args.limit is not None and args.limit < len(base):
        sys.exit(
            "--limit would invalidate the published ground truth, which describes the whole "
            "collection. Recomputing it here is out of scope; use `anka bench --limit` instead, "
            "which does recompute it."
        )
    if args.queries is not None:
        queries = queries[: args.queries]
        truth = truth[: args.queries]

    ef_values = [int(v) for v in args.ef.split(",") if v.strip()]
    count, dim = base.shape

    print(f"# hnswlib {getattr(hnswlib, '__version__', 'unknown')} on {platform.platform()}")
    print(f"# numpy {np.__version__}, python {platform.python_version()}")
    print("# NOTE: a pip wheel is a generic build; Anka is measured with target-cpu=native.")
    print("#       Build hnswlib from source with -O3 -march=native for a fair comparison.")
    print(f"# {args.dataset}: {count} base x {len(queries)} queries, dim {dim}, space {space}")
    print(f"# params: M={args.m} ef_construction={args.ef_construction} seed={args.seed}")

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(
        max_elements=count,
        ef_construction=args.ef_construction,
        M=args.m,
        random_seed=args.seed,
    )
    # Single-threaded throughout, so the comparison matches spec section 6 rule 1.
    index.set_num_threads(1)

    start = time.perf_counter()
    index.add_items(base, np.arange(count))
    build_seconds = time.perf_counter() - start
    print(f"# build: {build_seconds:.2f}s ({count / build_seconds:.0f} vectors/s, 1 thread)")

    warmup = min(args.warmup, len(queries) // 2)
    measured = len(queries) - warmup
    if measured <= 0:
        sys.exit("--warmup leaves no queries to measure")
    print(f"# {measured} queries measured, {warmup} used for warm-up")
    print()
    print(f"  {'ef':>5}  {'recall@k':>9}  {'QPS':>10}  {'p50':>9}  {'p95':>9}  {'p99':>9}")

    expected = truth[:measured, : args.k]
    for ef in ef_values:
        index.set_ef(max(ef, args.k))

        if warmup:
            index.knn_query(queries[len(queries) - warmup :], k=args.k)

        # Batch call: the loop runs inside hnswlib, so what is timed is search, not the
        # interpreter. Per-query latency is measured separately below for the percentiles.
        start = time.perf_counter()
        labels, _ = index.knn_query(queries[:measured], k=args.k)
        elapsed = time.perf_counter() - start

        hits = sum(len(np.intersect1d(labels[i], expected[i])) for i in range(measured))
        recall = hits / (measured * args.k)

        # Percentiles need per-query timings, and those necessarily include Python call overhead.
        # They are reported for shape, not to be compared against Anka's microsecond figures.
        latencies_us = np.empty(measured, dtype=np.float64)
        for i in range(measured):
            t0 = time.perf_counter()
            index.knn_query(queries[i : i + 1], k=args.k)
            latencies_us[i] = (time.perf_counter() - t0) * 1e6
        latencies_us.sort()

        print(
            f"  {ef:>5}  {recall:>9.4f}  {measured / elapsed:>10.1f}  "
            f"{percentile(latencies_us, 0.50):>8.1f}µs  "
            f"{percentile(latencies_us, 0.95):>8.1f}µs  "
            f"{percentile(latencies_us, 0.99):>8.1f}µs"
        )

    print()
    print("# QPS comes from the batch call. The percentiles come from single-query calls and")
    print("# include Python overhead, so they are not comparable to Anka's directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
