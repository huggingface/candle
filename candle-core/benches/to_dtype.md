# CPU dtype conversion: scope and measurements

The type-list refactor keeps `AsPrimitive` for generic scalar conversions. The
runtime optimization specializes contiguous F32-to-F16 conversion using half's
slice API. Other pairs and non-contiguous layouts retain `unary_map`.

For up to 1,024 elements, the implementation uses half's vector constructor.
Larger inputs are converted through a reusable 2 KiB initialized scratch buffer,
then appended to one preallocated result. This avoids zero-initializing the whole
result before overwriting it, and adds no unsafe code. Conversion remains O(N).

## Reproduction

`to_dtype.rs` times actual `Tensor::to_dtype` calls, including allocation and
destruction of the output. Input creation is outside timing. The contiguous
input has a nonzero offset; transposed and narrowed block views are also included.
The `blocks` view has half the size in its benchmark name, and its throughput is
computed from its actual element count.

Run these commands from the repository root. Cargo locates and runs the benchmark
executable automatically; there is no executable literally named
`path/to/benchmark`.

### Run the complete benchmark

```console
RUSTFLAGS="-C target-cpu=native" \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
cargo bench -p candle-core --bench to_dtype
```

This runs all 60 cases with Criterion's default settings. It is a convenient
local smoke run, but it does not reproduce the shorter, filtered measurement
protocol below. A candidate-only run does not establish a speedup.

### Run the 16 cases used in the report

Pass the filter and Criterion options after Cargo's `--` separator:

```console
RUSTFLAGS="-C target-cpu=native" \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
cargo bench -p candle-core --bench to_dtype -- \
  'F32_to_F16/contiguous/|/(contiguous|transposed|blocks)/1048576$' \
  --noplot --sample-size 40 --warm-up-time 0.2 \
  --measurement-time 0.7 --nresamples 5000 --save-baseline blocked-r1
```

Cargo already passes `--bench` to the executable; do not add another one after
the separator. `--no-run` only builds the executable and does not measure it.
The filter selects every contiguous F32-to-F16 size plus all pairs and layouts
at the size named 1,048,576 (16 distinct cases).

On Linux, CPU affinity is optional. Check the CPUs allowed for your shell with
`taskset -pc $$`. If CPU 2 is allowed, replace `cargo bench` in the command
above with `taskset -c 2 cargo bench`, keeping the environment assignments
before `taskset`. Otherwise select an allowed CPU or omit `taskset`.
CPU 2 was the choice for the reported VM, not a requirement of this benchmark.

### Compare two revisions

Run the same benchmark and manifest entry on both revisions with the same
Cargo.lock, compiler flags, filter, and Criterion settings. The historical
baseline commit below predates the benchmark: copy `to_dtype.rs` and its
`[[bench]]` manifest entry into that checkout before building.

Use separate Cargo target directories for the two checkouts (their default
`target` directories suffice if `CARGO_TARGET_DIR` is not shared). Generate
Cargo.lock once if necessary, reuse it in both checkouts, then add `--locked`
before Cargo's `--` separator for the comparative runs.

Run serially on the same logical CPU, with no concurrent builds. Use
`--save-baseline baseline-r1` for the reference and
`--save-baseline blocked-r1` for the candidate. Repeat in reverse order with
`blocked-r2` then `baseline-r2`. These names label results; they do not switch
the source revision. With separate target directories, compare the estimates
from each directory's `criterion` results. Do not infer a local speedup by
comparing one laptop run against the VM timings below.

## Local results, 2026-09-28

- Baseline: published type-list refactor `59b120c839e6c763048f6e6776206bff38441e85`.
- Candidate: the blocked contiguous F32-to-F16 path in this revision.
- AMD EPYC 9V74 virtual machine; affinity CPU 2. Frequency and other host workloads
  were not controlled. This is not a measurement on the user's laptop.
- Rust 1.98.1, LLVM 22.1.8, half 2.7.1, Criterion 0.8.2; native target, one release
  codegen unit. Both binaries used identical compiler settings and dependencies.
- Cargo.lock SHA-256: `76dc42f656d14930679e7313e66fe72a92d17980051f6a8de4d3090ea7787cc8`.
- Run order: baseline-r1, blocked-r1, blocked-r2, baseline-r2. Every run measured
  16 cases. The CSV records all 64 median estimates and their 95% confidence bounds.

F32-to-F16 contiguous calls; each cell gives round 1 / round 2, in microseconds:

| Elements | Baseline | Blocked | Baseline / blocked, paired rounds |
| ---: | ---: | ---: | ---: |
| 64 | 0.125 / 0.124 | 0.109 / 0.109 | 1.15x / 1.13x |
| 1,024 | 0.410 / 0.436 | 0.244 / 0.210 | 1.68x / 2.07x |
| 65,536 | 19.123 / 18.914 | 6.473 / 6.932 | 2.95x / 2.73x |
| 1,048,576 | 312.717 / 337.369 | 119.198 / 115.390 | 2.62x / 2.92x |
| 8,388,608 | 2,592.078 / 2,566.475 | 1,533.009 / 1,592.004 | 1.69x / 1.61x |

See `to_dtype-results.csv` for the controls (other pairs, transposes and blocks).
No acceleration is claimed for these unchanged paths. These measurements compare
against the published refactor, not a newly measured upstream main, and do not
establish an end-to-end model inference speedup or a guarantee for every CPU.

## Correctness

`cargo test -p candle-core --release --test dtype_tests --locked` passes all nine
tests with the same RUSTFLAGS and release codegen setting. Coverage includes the
100 supported type pairs across six layouts, every f16/bf16/FP8 identity pattern,
integer boundaries, unsupported formats, and 456,006 F32 inputs compared bitwise
against `f16::from_f32`. SIMD tails, scratch-block boundaries and nonzero offsets
are tested separately. Formatting and whitespace checks also pass.
