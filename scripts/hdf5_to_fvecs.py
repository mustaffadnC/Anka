#!/usr/bin/env python3
"""Convert an ann-benchmarks HDF5 dataset into the .fvecs/.ivecs formats.

ann-benchmarks distributes datasets as HDF5 with four members:

    train      (n, d) float32   base vectors
    test       (q, d) float32   query vectors
    neighbors  (q, k) int32     ground-truth neighbour ids
    distances  (q, k) float32   ground-truth distances (not used here)

anka-core reads only .fvecs/.ivecs, so conversion happens here rather than pulling an HDF5
C dependency into the Rust build to read one file once.

Formats, per record:
    .fvecs   int32 dim, then dim x float32
    .ivecs   int32 dim, then dim x int32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Written in chunks so a 1.18M x 100 dataset never needs a second full-size copy in memory.
CHUNK = 100_000


def write_fvecs(path: Path, data: h5py.Dataset) -> tuple[int, int]:
    n, d = data.shape
    with path.open("wb") as f:
        for start in range(0, n, CHUNK):
            block = np.ascontiguousarray(data[start : start + CHUNK], dtype=np.float32)
            out = np.empty((block.shape[0], d + 1), dtype=np.uint32)
            out[:, 0] = d
            # Reinterpret the float32 bit patterns as uint32 so the dimension prefix and the
            # payload can be written as one contiguous block per row.
            out[:, 1:] = block.view(np.uint32)
            out.tofile(f)
    return n, d


def write_ivecs(path: Path, data: h5py.Dataset) -> tuple[int, int]:
    n, d = data.shape
    with path.open("wb") as f:
        for start in range(0, n, CHUNK):
            block = np.ascontiguousarray(data[start : start + CHUNK], dtype=np.int32)
            out = np.empty((block.shape[0], d + 1), dtype=np.int32)
            out[:, 0] = d
            out[:, 1:] = block
            out.tofile(f)
    return n, d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", required=True, type=Path, help="input .hdf5 file")
    ap.add_argument("-o", "--outdir", required=True, type=Path, help="output directory")
    ap.add_argument("--prefix", required=True, help="output filename prefix, e.g. glove100")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as h5:
        missing = [k for k in ("train", "test", "neighbors") if k not in h5]
        if missing:
            print(
                f"error: {args.input} is missing {missing}; "
                f"found {sorted(h5.keys())}",
                file=sys.stderr,
            )
            return 1

        targets = [
            (f"{args.prefix}_base.fvecs", h5["train"], write_fvecs),
            (f"{args.prefix}_query.fvecs", h5["test"], write_fvecs),
            (f"{args.prefix}_groundtruth.ivecs", h5["neighbors"], write_ivecs),
        ]

        for name, dataset, writer in targets:
            path = args.outdir / name
            n, d = writer(path, dataset)
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"{name}: {n} x {d}  ({size_mb:.1f} MiB)")

        metric = h5.attrs.get("distance", "unknown")
        print(f"source metric attribute: {metric}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
