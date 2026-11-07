# Kev-0.8B Rust inference

Target: Jared Palmer's released Kev-0.8B, not the older Qwen3 Kev-0.6B.
Inference runs entirely in Rust/Candle; Python is a checkpoint conversion and parity oracle.

## Sources and pins

- Kev source: `../kev`, commit `0a6d3fa` (full hash recorded in the final report).
- [Kev-0.8B model card](https://github.com/jaredpalmer/kev/blob/0a6d3fa/docs/model-cards/kev-0.8b.md).
- [Checkpoint loader](https://github.com/jaredpalmer/kev/blob/0a6d3fa/kev/checkpoint.py) and
  [encoder/pointer readout](https://github.com/jaredpalmer/kev/blob/0a6d3fa/kev/model.py).
- Checkpoint: `jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f`.
- Base: `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Rust backbone starts from [Candle PR #3396](https://github.com/huggingface/candle/pull/3396),
  `woodRock/candle@e9e4929f3e31b6183cc17e7be0596929c6725257`, under Candle's MIT/Apache-2.0 license.
- Python backbone oracle: installed Transformers Qwen3.5 implementation; package versions recorded in results.

The model card's abbreviated Hub revision differs from the revision resolved by `v1.0` on this checkout date.
The exporter pins the actual full revision, adapter SHA-256, base revision and effective temperature.

## Mechanism

The text backbone has 24 layers: 18 Gated DeltaNet layers and six gated full-attention layers.
Rank-16 LoRA (alpha 32) modifies 186 projections. The export merges `W + 2 * B @ A` in fp32
and discards vision, MTP, and vocabulary readout weights. The retained pointer head has two
biased linear projections, dimension 256. It scores closing option tokens against the final
decision token, then divides by `sqrt(256)` and checkpoint temperature before softmax.

Caller control-token strings must be escaped before tokenization. Each question runs as an
independent causal row, because DeltaNet recurrence cannot implement a block attention mask.
Text rotary embeddings use 25% of the 256-dimensional attention head, with split-half rotation.
The initial PR implementation rotated the whole head and reused DeltaNet caches only for
one-token decoding. Both assumptions must be corrected for Kev's multi-token question branches.

## Optimizations and comparison rule

1. Merge LoRA once during conversion, rather than executing adapters on every inference.
2. Compute the state prefix once, retaining attention KV, convolution history, and DeltaNet recurrent
   state. Clone immutable Candle tensors for each question; no question mutates another question's cache.
3. Fuse the CPU DeltaNet recurrence into one mutable fp32 state buffer per head, avoiding tensor
   allocations and small GEMMs per token. Parallelize independent heads with Rayon.
4. Use a direct short depthwise convolution on CPU rather than thousands of grouped convolution calls.
5. Chunk prefix prefill at 256 tokens to bound full-attention scratch space.
6. Batch up to eight independent question rows, capped by a 16,384-token row budget.
   Replicate prefix caches across the batch, keep attention queries contiguous for
   CPU matrix multiplication, and retain only option/decision marker hidden states.
7. Preserve arbitrary JSON integer IDs and Python's float text formatting before
   tokenization. `1e-7` must become `1e-07`, and a large integer must not be rounded to f64.

Compare against the official Python **fp32 unmerged** reference, not an independently reimplemented
Python model. Require identical token IDs and option order, zero argmax changes, and maximum raw
probability error below `1e-4`. Compare Rust full rows/tensor kernels against optimized prefix/fused
kernels separately. Fixtures cover all three primitives, structured state, escaped delimiters,
single-option cases, repeated requests, question isolation, and a state crossing a prefill chunk boundary.
Runtime tests must also compare the fused scan and short convolution against independent tensor operations.

Benchmark warmed runs with two threads on the same host, sequentially (no concurrent build/inference).
Record loading separately from model inference. These are latency/parity probes; they do not establish
task accuracy, calibration on a new dataset, or GPU performance. No training or temperature refitting.

## Scope

The implementation supports CPU and NVIDIA CUDA fp32, exposed as a Rust library and a JSONL command-line runner.
The Python exporter is a one-time build step; exported checkpoints need no Python to execute.
There is no HTTP server or quantized backend yet. CUDA builds and tests run remotely in Colab;
Metal and 64k-token performance have not been validated. A future quantized backend must be measured against
this fp32 implementation and the Python oracle before being offered as equivalent.

The isolated CPU Colab run is recorded in [BENCHMARK.md](BENCHMARK.md). Prefix batching
was the fastest native mode on the short-request panel and is now the default.

## CUDA execution

The `cuda` Cargo feature enables Candle's CUDA backend. The runner loads the same
merged fp32 checkpoint onto the selected GPU. CLI device selection is explicit,
with CPU remaining the default. FP32 reduced precision GEMMs (TF32) are disabled.
Device synchronization brackets inference timing to include all CUDA execution.

`cuda/kev.cu` contains the short depthwise convolution and gated DeltaNet scan.
The scan parallelizes independent heads and tiles of 32 value columns. Each warp
owns its shared-memory state tile, executes the token recurrence in order, and
writes fresh final state without mutating the initial cache. No recurrence tensor
is transferred to the host. Input tensor packing preserves strided and sliced
layouts; the final answer probabilities and logits are copied to the host for JSON.
CUDA compilation disables FMA contraction in these two custom kernels to follow
the existing CPU recurrence's multiplication/addition rounding more closely.

CUDA tests compare the fused scan against the CPU rule for both real head sizes
and partial value-column tiles, nonzero initial caches, and split continuation.
Convolution tests cover sliced, strided inputs. Model parity uses the saved output
of the official unmerged Python FP32 implementation for all 22 fixture questions,
with the existing probability/logit tolerances unchanged.
