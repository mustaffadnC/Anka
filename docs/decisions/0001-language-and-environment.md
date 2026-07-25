# ADR 0001 — Language, environment, and licence

**Date:** 2026-07-26
**Status:** Accepted

## Context

Anka is a from-scratch ANN search engine whose central claim is measurement quality: recall/QPS
Pareto curves, a fair comparison against hnswlib, and a profile explaining the gap. Three
choices had to be settled before any code was written, because each one shapes the rest.

The development machine is Windows 11 on a Ryzen 5 7600X (Zen 4: AVX2, no AVX-512) with 32 GB of
RAM. At the start of the project it had no Rust toolchain, no WSL, no Python, no Docker, and a
Visual Studio install without the C++ workload.

## Decision 1 — Rust

**Chosen: Rust (edition 2024), with `std::arch` intrinsics for SIMD.**

`cargo test` and `cargo bench` come out of the box, and benchmark discipline is the point of the
project — time spent on a build system is time not spent measuring. The ownership model catches
data races in concurrent index access at compile time, which matters once compaction swaps an
index under live readers.

C++20 with CMake, vcpkg, Google Benchmark and Catch2 was the alternative and remains a valid one;
the design translates directly. It was not chosen because dependency and build management would
have consumed hours that add nothing to the measurements.

`std::simd` was rejected: it is nightly-only, and pinning to nightly for a project that will run
for months is a maintenance cost with no upside here. `std::arch` intrinsics with
`is_x86_feature_detected!` runtime dispatch are stable and give the same control.

## Decision 2 — WSL2 + Ubuntu as the development and measurement environment

**Chosen: WSL2 + Ubuntu. Source on `/mnt/c`, build output and datasets on ext4.**

The design leans on POSIX behaviour throughout: `fsync` on a file *and* on its parent directory
for atomic snapshot rename, `kill -9` crash tests, `/proc/self/status` for RSS, shell scripts for
dataset download and benchmarking, Docker for phase 6. Running on Linux means the development
environment and CI (`ubuntu-latest`) behave identically, so a green CI run means something.

Native Windows with MSVC was the alternative. It would have required installing the Visual Studio
C++ workload and then porting six or seven parts of the design — crash tests to
`TerminateProcess`, RSS to `GetProcessMemoryInfo`, shell scripts to PowerShell — while leaving the
development and CI environments divergent. Not worth it for a project this measurement-sensitive.

Two consequences are accepted rather than solved:

- **`perf` hardware counters do not work under WSL2** (the PMU is not virtualised). Cache
  behaviour is measured with `cachegrind` (simulated; useful as a ratio) or with AMD uProf against
  a Windows build. Whichever was used is stated next to the number.
- **`cpupower frequency-set` cannot work from inside a VM.** Clock variance is handled with the
  Windows power plan plus three repetitions and a median.

Layout follows from WSL2's filesystem performance: `/mnt/c` (9p/virtiofs) is markedly slower than
ext4, so source lives on `/mnt/c` where Windows editors can reach it, while `CARGO_TARGET_DIR`,
the datasets, and the temporary directory used by snapshot and WAL tests all live on ext4.

## Decision 3 — MIT licence

**Chosen: MIT only.**

The Rust ecosystem convention is dual MIT/Apache-2.0, and the Apache-2.0 patent grant is the
reason libraries adopt it. Anka is an application and a portfolio project, not a crate others are
expected to depend on, so a single permissive licence is enough. If `anka-core` is ever published
to crates.io, add Apache-2.0 then.

## Consequences

- Benchmarks are built with `RUSTFLAGS="-C target-cpu=native"`; CI is not, because the runner is a
  different machine and only correctness is checked there.
- AVX-512 is an explicit non-goal — the hardware cannot execute it, so it cannot be tested.
- The Python toolchain is required, not optional: dataset conversion (GloVe ships as HDF5) and the
  hnswlib baseline both depend on it.
- CI runs `fmt --check`, `clippy -D warnings`, and the test suite. A recall-regression job on
  `siftsmall` is added in phase 2.
