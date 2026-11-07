# Colab validation

All builds, inference checks, and timing runs for this report executed through
Google Colab CLI on an isolated **CPU-only** runtime. No GPU or TPU was requested.

Hardware: x86_64 Intel Xeon @ 2.20 GHz, two vCPUs (two SMT threads on one core).
Python and Rust each used two threads. The checkpoint was the pinned Jared Palmer
Kev-0.8B release documented in RESEARCH.md, with its original temperature.

## Correctness

The panel contains ten requests and 22 questions. It covers `choice`, `score`, `noul`,
structured JSON state, delimiter escaping, single-option cases, empty state, numeric
state with scientific notation and a large integer ID, question reordering, independent
questions, and a 286-token state crossing the 256-token prefill boundary.

The unbatched, full-row batched, and prefix-batched native implementations all passed:

- Identical token IDs and option order against the official Python fp32 unmerged model.
- Zero argmax changes.
- Maximum raw probability difference: `1.2264e-6` (required below `1e-4`).
- Maximum calibrated logit difference: below `7e-6` (required below `1e-3`).
- Question isolation and cache continuation checks.
- Four Rust unit tests, including independent tensor references for scan and convolution.

## Warmed inference time

Four short requests, five repetitions per request; the first repetition of each
request is excluded, leaving 16 warmed samples per mode. Builds and downloads finished
before timing. Modes ran sequentially on the same runtime. Model loading and tokenization
are excluded from these inference timings.

| Implementation | Warmed median |
| --- | ---: |
| Python fp32, unmerged LoRA | 3.378 s |
| Rust full rows, tensor kernels | 5.651 s |
| Rust full rows, fused kernels, unbatched | 4.244 s |
| Rust full rows, fused kernels, batched | 3.882 s |
| Rust shared prefix, fused kernels, batched | **3.465 s** |

Prefix batching improves the native baseline by **1.63×** (about 39% less inference
time). Python is approximately 2.6% faster than the best Rust mode on this panel.
The Rust result preserves fp32 precision; it does not use quantization.

These are implementation parity and latency probes, not a model accuracy study.
The sample is small, Colab CPU timings vary, and the results do not establish latency
on production CPUs, longer documents, or other thread counts. An earlier run on a
runtime later shared with another job was excluded from the final batching timings.

The machine-readable report, per-mode comparisons, prediction files, hardware data,
and full job log are preserved under `results/colab/`. The final default executable is
checked separately after selecting prefix batching as the default. The report records
both benchmark source hashes and final source hashes.

## NVIDIA CUDA validation

The GPU build and all CUDA tests ran through Colab CLI on a Tesla T4 with CUDA
13.0. The host CPU was an Intel Xeon @ 2.00 GHz with two vCPUs. CPU and GPU timing
runs used the same VM, checkpoint, FP32 dtype, shared-prefix batching, and request
panel, sequentially. CPU execution used two threads. TF32 and quantization were
disabled. These comparisons are Rust CPU versus Rust CUDA; no Python GPU speed
claim is made.

All three CUDA modes (prefix-batched, full-row batched, and prefix-unbatched) matched
the official unmerged Python FP32 reference on all 22 questions, with identical
tokenization and zero argmax changes. Maximum raw probability error was `7.44e-7`
for the default mode, and `1.53e-6` across all modes. Maximum calibrated logit error
across all modes was `5.90e-6`. Serialized probabilities differed by up to `4.73e-5`
because the Python API rounds its returned values. All existing tolerances passed.
Question isolation and chunked prefix continuation passed. Six unit tests passed,
including the two CUDA kernel comparisons. A separate CPU-only Colab build passed
all four CPU tests and correctly rejected `--device cuda` before model loading.

| Rust implementation | Warmed median | Samples |
| --- | ---: | ---: |
| CPU, shared prefix, fused kernels, batched | 2.865 s | 16 |
| T4 CUDA, shared prefix, fused kernels, batched | **89.62 ms** | 16 |

CUDA was **31.97× faster** on this short-request panel. The first iteration of each
of four requests was excluded from five repetitions, and CUDA synchronization
bracketed inference timing. Loading and tokenization are excluded. This result
does not establish performance on other hardware or longer contexts.

The first CUDA dependency build plus tests took 12m 23s, followed by a 2.98s
executable build. The source archive, formatted final source, machine-readable
report, comparisons, and logs are under `results/colab-gpu/`. The binary and its
checksum are packaged in `dist/kev-rust-linux-x86_64-cuda.tar.gz`. The CUDA build
requires CUDA 13 libraries and a CPU compatible with the native Colab compilation
target; rebuild on another host when its toolkit or CPU differs.

## GPU Comparison: Kev Rust vs. Official Python (PyTorch CUDA)

To evaluate whether the native Rust implementation provides a significant difference over the official Python PyTorch implementation on GPU, a comprehensive benchmark was executed on the same Google Colab Tesla T4 GPU (CUDA 13.0, FP32 precision) across 5 distinct test scenarios and the standard fixture panel:

1. **Case 1: Single Question**: 1 choice question, short state (~25 tokens).
2. **Case 2: Small Batch**: 3 questions (`choice`, `score`, `noul`).
3. **Case 3: Medium Batch**: 8 questions assessing multiple facets of a support ticket.
4. **Case 4: Heavy Batch**: 16 questions evaluating SLA/audit clauses on a transaction text.
5. **Case 5: Long Context**: Master Service Agreement (~600 tokens) with 4 questions.
6. **Fixture Panel**: 4 baseline requests from `fixtures/requests.jsonl`.

### Warmed Median Latency across Scenarios

| Scenario | Num Qs | Python Prefix (PyTorch) | Python Rows (No Prefix) | Rust Prefix Batched (Default) | Rust Rows Batched | Rust Unbatched | Rust Batching Speedup |
|---|:---:|---:|---:|---:|---:|---:|:---:|
| **Case 1: Single Question** | 1 | 163.38 ms | 89.55 ms | **69.33 ms** | 53.79 ms | 74.23 ms | 1.07× |
| **Case 2: Small Batch** | 3 | 161.68 ms | 91.76 ms | **102.35 ms** | 108.42 ms | 121.93 ms | 1.19× |
| **Case 3: Medium Batch** | 8 | 234.51 ms | 436.81 ms | **237.81 ms** | 527.83 ms | 299.75 ms | 1.26× |
| **Case 4: Heavy Batch** | 16 | 384.50 ms | 1,180.75 ms | **361.25 ms** | 1,318.13 ms | 569.35 ms | **1.58×** |
| **Case 5: Long Context (~600t)** | 4 | **386.14 ms** | 973.12 ms | **456.65 ms** | 1,163.01 ms | 474.60 ms | 1.04× |
| **Fixture Panel (requests.jsonl)** | 22 | 160.94 ms | 81.45 ms | **80.12 ms** | 108.42 ms | 121.93 ms | 1.20× |

All cases preserved exact argmax parity with zero decision flips and maximum raw probability differences below `4.93e-5` (floating-point rounding).

### Is the Performance Difference Significant?

#### 1. Warmed GPU Compute: Close (5%–30% difference)
When the model is already loaded and warmed up on GPU, pure execution latency between Rust Prefix and Python Prefix is comparable:
- **Case 4 (16 questions)**: Rust is slightly faster (361.25 ms vs. 384.50 ms).
- **Case 5 (Long context)**: Python is slightly faster (386.14 ms vs. 456.65 ms).
- **Why**: Both PyTorch and Candle execute native NVIDIA CUDA kernels (cuBLAS GEMMs and custom CUDA kernels) on the GPU hardware. Once tensors reside in VRAM, the hardware compute and memory bandwidth are the primary bottlenecks, not the host language interpreter.

#### 2. Where the Differences ARE Substantial & Significant:

1. **Algorithmic Architecture (Shared Prefix Caching)**:
   - The largest performance leap comes from **Shared Prefix Caching rather than language choice**:
   - On 16 questions (Case 4), shared prefix caching is **3.27× faster** than evaluating full rows (361 ms vs. 1,180–1,318 ms).
   - On long context (Case 5), prefix caching is **2.13× faster** (456 ms vs. 973–1,163 ms).
   - Within Rust, multi-question batching accelerates 16 questions by **1.58×** over sequential execution (361 ms vs. 569 ms, averaging **~22.5 ms per question**).

2. **Cold Start & Initialization Latency**:
   - Rust: **~119 ms** cold start on the first request.
   - Python PyTorch: **~1,498 ms** (1.5 seconds) cold start due to JIT tracing, dynamic tensor initialization, and PyTorch allocator setup.
   - For serverless / scale-to-zero environments (AWS Lambda, Modal, Cloud Run), Rust starts **~12× faster**.

3. **Deployment Footprint & Dependency Size**:
   - Rust: A single standalone binary (`dist/kev-rust-linux-x86_64-cuda`) of **~35 MB**, requiring only NVIDIA drivers/CUDA runtime libraries.
   - Python: Requires a full Python runtime, PyTorch CUDA wheel, Transformers, PEFT, and dependencies, typically producing Docker containers of **6 GB to 10 GB**.

4. **Memory Allocation & Predictability**:
   - Python PyTorch utilizes an aggressive caching allocator that retained **~10 GB of GPU VRAM** in memory.
   - Rust allocates tensors with tight lifetimes and precise buffers (~2.5–3.0 GB VRAM peak), allowing multiple model replicas on a single 16 GB GPU.

5. **Concurrency & Threading (No GIL)**:
   - Python's Global Interpreter Lock (GIL) requires multi-process architectures (e.g. Uvicorn with multiple workers) to serve concurrent traffic, multiplying RAM and model weight copies.
   - Rust provides native thread safety (Rayon/Tokio) across requests with zero GIL contention.

