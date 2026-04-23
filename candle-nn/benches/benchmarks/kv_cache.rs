// candle-nn/benches/benchmarks/kv_cache.rs
//
// Criterion benchmarks comparing ConcatKvCache vs PreallocKvCache.
//
// Run:
//   cargo bench -p candle-nn --bench bench_main -- kv_cache
//   cargo bench -p candle-nn --features cuda   --bench bench_main -- kv_cache
//   cargo bench -p candle-nn --features metal  --bench bench_main -- kv_cache

use super::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Tensor};
use candle_nn::kv_cache::{ConcatKvCache, PreallocKvCache};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group};

// ── Benchmark parameters ──────────────────────────────────────────────────────

const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 512;
const DTYPE: DType = DType::F32;

/// Token counts for the multi-append throughput benchmark.
const MULTI_APPEND_COUNTS: &[usize] = &[10, 50, 100, 500];

/// Number of conversation turns for the boundary benchmark.
const CONVERSATION_TURNS: usize = 50;
const TOKENS_PER_TURN: usize = 20;

// ── Helper ────────────────────────────────────────────────────────────────────

fn rand_kv(device: &Device) -> (Tensor, Tensor) {
    let k = Tensor::randn(0f32, 1.0, (1, NUM_KV_HEADS, 1, HEAD_DIM), device).unwrap();
    let v = Tensor::randn(0f32, 1.0, (1, NUM_KV_HEADS, 1, HEAD_DIM), device).unwrap();
    (k, v)
}

// ── Group 1: single append ────────────────────────────────────────────────────
//
// Measures the cost of one append() call from a cold cache.
// PreallocKvCache should be comparable to ConcatKvCache here (both write one
// token). The allocation difference becomes visible in multi-append below.

fn bench_single_append(c: &mut Criterion, device: &Device) {
    let mut group = c.benchmark_group("kv_cache/single_append");
    group.sample_size(200);

    group.bench_function(
        BenchmarkId::new("ConcatKvCache", device.bench_name("")),
        |b| {
            b.iter_batched(
                || {
                    let cache = ConcatKvCache::new(2);
                    let (k, v) = rand_kv(device);
                    (cache, k, v)
                },
                |(mut cache, k, v)| {
                    cache.append(&k, &v).unwrap();
                    device.sync().unwrap();
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("PreallocKvCache", device.bench_name("")),
        |b| {
            b.iter_batched(
                || {
                    let cache =
                        PreallocKvCache::new(NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, DTYPE, device)
                            .unwrap();
                    let (k, v) = rand_kv(device);
                    (cache, k, v)
                },
                |(mut cache, k, v)| {
                    cache.append(&k, &v).unwrap();
                    device.sync().unwrap();
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

// ── Group 2: multi-append throughput ─────────────────────────────────────────
//
// Measures end-to-end cost of appending N tokens sequentially.
// ConcatKvCache's alloc-per-token pattern shows up here as super-linear growth.
// PreallocKvCache should scale linearly because all allocations are same-sized.

fn bench_multi_append(c: &mut Criterion, device: &Device) {
    let mut group = c.benchmark_group("kv_cache/multi_append");
    group.sample_size(50);

    for &n_tokens in MULTI_APPEND_COUNTS {
        if n_tokens > MAX_SEQ_LEN {
            continue;
        }

        group.bench_with_input(
            BenchmarkId::new(
                format!("ConcatKvCache/{}", device.bench_name("")),
                n_tokens,
            ),
            &n_tokens,
            |b, &n| {
                b.iter_batched(
                    || ConcatKvCache::new(2),
                    |mut cache| {
                        for _ in 0..n {
                            let (k, v) = rand_kv(device);
                            cache.append(&k, &v).unwrap();
                        }
                        device.sync().unwrap();
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                format!("PreallocKvCache/{}", device.bench_name("")),
                n_tokens,
            ),
            &n_tokens,
            |b, &n| {
                b.iter_batched(
                    || {
                        PreallocKvCache::new(
                            NUM_KV_HEADS,
                            HEAD_DIM,
                            MAX_SEQ_LEN,
                            DTYPE,
                            device,
                        )
                        .unwrap()
                    },
                    |mut cache| {
                        for _ in 0..n {
                            let (k, v) = rand_kv(device);
                            cache.append(&k, &v).unwrap();
                        }
                        device.sync().unwrap();
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Group 3: conversation boundary reset ─────────────────────────────────────
//
// Simulates a multi-turn chat workload: TOKENS_PER_TURN appends followed by
// reset(), repeated CONVERSATION_TURNS times.
//
// ConcatKvCache: reset() sets k/v to None (free), then each new turn reallocates.
// PreallocKvCache: reset() is counter=0, no dealloc. Next turn reuses the buffer.
// This is the workload where PreallocKvCache's O(1) reset shines.

fn bench_conversation_boundary(c: &mut Criterion, device: &Device) {
    let mut group = c.benchmark_group("kv_cache/conversation_boundary");
    group.sample_size(30);

    group.bench_function(
        BenchmarkId::new("ConcatKvCache", device.bench_name("")),
        |b| {
            b.iter_batched(
                || ConcatKvCache::new(2),
                |mut cache| {
                    for _ in 0..CONVERSATION_TURNS {
                        for _ in 0..TOKENS_PER_TURN {
                            let (k, v) = rand_kv(device);
                            cache.append(&k, &v).unwrap();
                        }
                        cache.reset();
                    }
                    device.sync().unwrap();
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("PreallocKvCache", device.bench_name("")),
        |b| {
            b.iter_batched(
                || {
                    PreallocKvCache::new(NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, DTYPE, device)
                        .unwrap()
                },
                |mut cache| {
                    for _ in 0..CONVERSATION_TURNS {
                        for _ in 0..TOKENS_PER_TURN {
                            let (k, v) = rand_kv(device);
                            cache.append(&k, &v).unwrap();
                        }
                        cache.reset();
                    }
                    device.sync().unwrap();
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

// ── Group 4: time-to-first-token ─────────────────────────────────────────────
//
// Measures the cost of construction + first append (TTFT proxy).
// PreallocKvCache pays upfront at new(); ConcatKvCache pays nothing at new()
// but allocates on first append. This benchmark quantifies the tradeoff.

fn bench_time_to_first_token(c: &mut Criterion, device: &Device) {
    let mut group = c.benchmark_group("kv_cache/time_to_first_token");
    group.sample_size(200);

    group.bench_function(
        BenchmarkId::new("ConcatKvCache", device.bench_name("")),
        |b| {
            b.iter(|| {
                let mut cache = ConcatKvCache::new(2);
                let (k, v) = rand_kv(device);
                cache.append(&k, &v).unwrap();
                device.sync().unwrap();
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("PreallocKvCache", device.bench_name("")),
        |b| {
            b.iter(|| {
                let mut cache =
                    PreallocKvCache::new(NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, DTYPE, device)
                        .unwrap();
                let (k, v) = rand_kv(device);
                cache.append(&k, &v).unwrap();
                device.sync().unwrap();
            });
        },
    );

    group.finish();
}

// ── Dispatch over available devices ──────────────────────────────────────────

pub fn run_kv_cache_benches(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().expect("failed to initialise bench devices");
    for device in &handler.devices {
        bench_single_append(c, device);
        bench_multi_append(c, device);
        bench_conversation_boundary(c, device);
        bench_time_to_first_token(c, device);
    }
}

criterion_group!(benches, run_kv_cache_benches);
