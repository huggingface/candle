//! Correctness tests for the device-agnostic block-paged KV-cache attention.
//!
//! The paged implementation is validated against a straightforward dense
//! attention computed over the same keys/values, for MHA / GQA / MQA and
//! across multiple block sizes and contexts. The same cases are run on Metal
//! when the `metal` feature is enabled, plus a Metal-vs-CPU consistency check.

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::attention::paged::{paged_attention, reshape_and_cache, BlockAllocator};
use candle_nn::ops::softmax_last_dim;

const EPS: f32 = 1e-4;

/// Dense single-token attention reference for one sequence.
/// q: [num_heads, head_dim], k/v: [ctx, num_kv_heads, head_dim].
fn dense_decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    num_kv_heads: usize,
) -> Result<Tensor> {
    let (num_heads, head_dim) = q.dims2()?;
    let (ctx, _, _) = k.dims3()?;
    let group = num_heads / num_kv_heads;

    let expand = |t: &Tensor| -> Result<Tensor> {
        t.reshape((ctx, num_kv_heads, 1, head_dim))?
            .broadcast_as((ctx, num_kv_heads, group, head_dim))?
            .reshape((ctx, num_heads, head_dim))
    };
    let k = expand(k)?.transpose(0, 1)?.contiguous()?; // [H, ctx, D]
    let v = expand(v)?.transpose(0, 1)?.contiguous()?; // [H, ctx, D]
    let q = q.unsqueeze(1)?; // [H, 1, D]

    let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale as f64)?; // [H,1,ctx]
    let probs = softmax_last_dim(&scores)?;
    let out = probs.matmul(&v)?; // [H,1,D]
    out.squeeze(1)
}

/// Build a paged cache by laying each sequence's KV out across freshly
/// allocated blocks, returning the cache tensors plus the per-batch
/// block_tables / context_lens needed by `paged_attention`.
#[allow(clippy::type_complexity)]
fn build_paged_cache(
    keys: &[Tensor], // each [ctx_i, num_kv_heads, head_dim]
    values: &[Tensor],
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let mut allocator = BlockAllocator::new(num_blocks, block_size);
    let mut k_cache = Tensor::zeros(
        (num_blocks, block_size, num_kv_heads, head_dim),
        DType::F32,
        device,
    )?;
    let mut v_cache = k_cache.clone();

    let mut block_tables: Vec<Vec<u32>> = Vec::new();
    let mut context_lens: Vec<u32> = Vec::new();
    let mut max_blocks = 1usize;

    for (k, v) in keys.iter().zip(values.iter()) {
        let ctx = k.dim(0)?;
        let phys = allocator.allocate_for_tokens(ctx).expect("pool exhausted");
        let mut slots = Vec::with_capacity(ctx);
        for (b, &p) in phys.iter().enumerate() {
            let tokens_in_block = (ctx - b * block_size).min(block_size);
            for off in 0..tokens_in_block {
                slots.push((p * block_size + off) as u32);
            }
        }
        let slot_mapping = Tensor::from_vec(slots, ctx, device)?;
        let (kc, vc) = reshape_and_cache(k, v, &k_cache, &v_cache, &slot_mapping)?;
        k_cache = kc;
        v_cache = vc;

        max_blocks = max_blocks.max(phys.len());
        block_tables.push(phys.into_iter().map(|p| p as u32).collect());
        context_lens.push(ctx as u32);
    }

    // Pad block tables to a rectangular [num_seqs, max_blocks].
    for table in block_tables.iter_mut() {
        table.resize(max_blocks, 0);
    }
    let flat: Vec<u32> = block_tables.iter().flatten().copied().collect();
    let block_tables = Tensor::from_vec(flat, (keys.len(), max_blocks), device)?;
    let context_lens = Tensor::from_vec(context_lens, keys.len(), device)?;

    Ok((k_cache, v_cache, block_tables, context_lens))
}

fn run_case(
    num_kv_heads: usize,
    group: usize,
    head_dim: usize,
    block_size: usize,
    ctx_lens: &[usize],
) -> Result<()> {
    run_case_on(
        &Device::Cpu,
        num_kv_heads,
        group,
        head_dim,
        block_size,
        ctx_lens,
    )
}

fn run_case_on(
    device: &Device,
    num_kv_heads: usize,
    group: usize,
    head_dim: usize,
    block_size: usize,
    ctx_lens: &[usize],
) -> Result<()> {
    let num_heads = num_kv_heads * group;
    let num_seqs = ctx_lens.len();
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Deterministic pseudo-random inputs (no rng dependency).
    let gen = |n: usize, seed: f32| -> Result<Tensor> {
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.123 + seed).sin()) * 0.5)
            .collect();
        Tensor::from_vec(data, n, device)
    };

    let q = gen(num_seqs * num_heads * head_dim, 0.0)?.reshape((num_seqs, num_heads, head_dim))?;

    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (s, &ctx) in ctx_lens.iter().enumerate() {
        keys.push(
            gen(ctx * num_kv_heads * head_dim, 1.0 + s as f32)?.reshape((
                ctx,
                num_kv_heads,
                head_dim,
            ))?,
        );
        values.push(
            gen(ctx * num_kv_heads * head_dim, 7.0 + s as f32)?.reshape((
                ctx,
                num_kv_heads,
                head_dim,
            ))?,
        );
    }

    let total_blocks: usize = ctx_lens.iter().map(|c| c.div_ceil(block_size)).sum();
    let num_blocks = total_blocks + 2; // a little slack in the pool

    let (k_cache, v_cache, block_tables, context_lens) = build_paged_cache(
        &keys,
        &values,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device,
    )?;

    let out = paged_attention(
        &q,
        &k_cache,
        &v_cache,
        &block_tables,
        &context_lens,
        block_size,
        scale,
        None,
    )?;
    assert_eq!(out.dims(), &[num_seqs, num_heads, head_dim]);

    for s in 0..num_seqs {
        let expected = dense_decode(&q.i(s)?, &keys[s], &values[s], scale, num_kv_heads)?;
        let got = out.i(s)?;
        let diff = (got - expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < EPS,
            "seq {s}: max abs diff {diff} exceeds {EPS} (kv_heads={num_kv_heads}, group={group}, block_size={block_size}, ctx={})",
            ctx_lens[s]
        );
    }
    Ok(())
}

#[test]
fn paged_matches_dense_mha() -> Result<()> {
    // Multi-head (group = 1), contexts that do and do not fill whole blocks.
    run_case(4, 1, 8, 4, &[4, 7, 1, 16])
}

#[test]
fn paged_matches_dense_gqa() -> Result<()> {
    // Grouped-query attention: 2 KV heads shared by 8 query heads.
    run_case(2, 4, 16, 8, &[10, 3, 20])
}

#[test]
fn paged_matches_dense_mqa_block_size_one() -> Result<()> {
    // Multi-query attention with the degenerate block size of 1.
    run_case(1, 8, 8, 1, &[5, 9])
}

#[test]
fn alibi_bias_is_applied() -> Result<()> {
    // With a single sequence, ALiBi only adds a per-key constant before
    // softmax; check the paged path matches a hand-rolled dense ALiBi decode.
    let device = Device::Cpu;
    let (num_heads, head_dim, block_size, ctx) = (2usize, 8usize, 4usize, 7usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let gen = |n: usize, seed: f32| -> Result<Tensor> {
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.07 + seed).cos()) * 0.5)
            .collect();
        Tensor::from_vec(data, n, &device)
    };
    let q = gen(num_heads * head_dim, 0.0)?.reshape((1, num_heads, head_dim))?;
    let k = gen(ctx * num_heads * head_dim, 2.0)?.reshape((ctx, num_heads, head_dim))?;
    let v = gen(ctx * num_heads * head_dim, 5.0)?.reshape((ctx, num_heads, head_dim))?;
    let slopes = Tensor::from_vec(vec![0.5f32, 0.25], num_heads, &device)?;

    let num_blocks = ctx.div_ceil(block_size) + 1;
    let (k_cache, v_cache, block_tables, context_lens) = build_paged_cache(
        std::slice::from_ref(&k),
        std::slice::from_ref(&v),
        num_blocks,
        block_size,
        num_heads,
        head_dim,
        &device,
    )?;
    let out = paged_attention(
        &q,
        &k_cache,
        &v_cache,
        &block_tables,
        &context_lens,
        block_size,
        scale,
        Some(&slopes),
    )?;

    // Dense ALiBi reference for the single decode token at position ctx-1.
    let qh = q.i(0)?.unsqueeze(1)?; // [H,1,D]
    let kh = k.transpose(0, 1)?.contiguous()?; // [H,ctx,D]
    let vh = v.transpose(0, 1)?.contiguous()?; // [H,ctx,D]
    let mut scores = (qh.matmul(&kh.transpose(1, 2)?.contiguous()?)? * scale as f64)?; // [H,1,ctx]
    let slopes_v = slopes.to_vec1::<f32>()?;
    let mut bias: Vec<f32> = Vec::with_capacity(num_heads * ctx);
    for &slope in slopes_v.iter() {
        for j in 0..ctx {
            bias.push(-slope * (ctx - 1 - j) as f32);
        }
    }
    let bias = Tensor::from_vec(bias, (num_heads, 1, ctx), &device)?;
    scores = scores.add(&bias)?;
    let expected = softmax_last_dim(&scores)?.matmul(&vh)?.squeeze(1)?; // [H,D]

    let diff = (out.i(0)? - expected)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff < EPS, "alibi paged vs dense diff {diff} exceeds {EPS}");
    Ok(())
}

#[test]
fn empty_context_yields_zeros() -> Result<()> {
    let device = Device::Cpu;
    let (num_blocks, block_size, num_kv_heads, head_dim) = (4, 4, 2, 8);
    let num_heads = 2;
    let k_cache = Tensor::zeros(
        (num_blocks, block_size, num_kv_heads, head_dim),
        DType::F32,
        &device,
    )?;
    let v_cache = k_cache.clone();
    let q = Tensor::ones((1, num_heads, head_dim), DType::F32, &device)?;
    let block_tables = Tensor::zeros((1, 1), DType::U32, &device)?;
    let context_lens = Tensor::zeros(1, DType::U32, &device)?;
    let out = paged_attention(
        &q,
        &k_cache,
        &v_cache,
        &block_tables,
        &context_lens,
        block_size,
        0.5,
        None,
    )?;
    let sum = out.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(sum, 0.0);
    Ok(())
}

#[test]
fn block_allocator_pools_blocks() {
    let mut alloc = BlockAllocator::new(3, 8);
    assert_eq!(alloc.num_free(), 3);
    let a = alloc.allocate().unwrap();
    let b = alloc.allocate().unwrap();
    let c = alloc.allocate().unwrap();
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_eq!(alloc.allocate(), None);
    alloc.free(b);
    assert_eq!(alloc.num_free(), 1);
    assert_eq!(alloc.allocate(), Some(b));

    // Multi-block allocation rounds up by block size and fails atomically.
    let mut alloc = BlockAllocator::new(2, 4);
    assert_eq!(alloc.allocate_for_tokens(5).map(|v| v.len()), Some(2));
    assert_eq!(alloc.num_free(), 0);
    assert_eq!(alloc.allocate_for_tokens(1), None);
}

// ------------------------------------------------------------------------
// Metal path. Mirrors the CPU cases on an actual Metal device and adds a
// Metal-vs-CPU consistency check. Compiled out unless the `metal` feature
// is enabled. By default it skips gracefully when no Metal device is
// present (e.g. CPU-only dev machines); set `CANDLE_METAL_REQUIRED=1` (as
// the macOS CI does) to turn a missing device into a hard failure so a
// green run actually means the Metal path executed.
// ------------------------------------------------------------------------
#[cfg(feature = "metal")]
mod metal {
    use super::*;

    fn metal_device() -> Option<Device> {
        match Device::new_metal(0) {
            Ok(dev) => Some(dev),
            Err(e) => {
                if std::env::var("CANDLE_METAL_REQUIRED").is_ok() {
                    panic!("CANDLE_METAL_REQUIRED is set but no Metal device is available: {e}");
                }
                eprintln!("skipping: no Metal device ({e})");
                None
            }
        }
    }

    #[test]
    fn paged_matches_dense_metal() -> Result<()> {
        let Some(dev) = metal_device() else {
            return Ok(());
        };
        run_case_on(&dev, 4, 1, 8, 4, &[4, 7, 1, 16])?;
        run_case_on(&dev, 2, 4, 16, 8, &[10, 3, 20])?;
        run_case_on(&dev, 1, 8, 8, 1, &[5, 9])
    }

    #[test]
    fn paged_metal_matches_cpu() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("skipping: no Metal device");
            return Ok(());
        };
        let (num_kv_heads, group, head_dim, block_size) = (2usize, 4usize, 16usize, 8usize);
        let num_heads = num_kv_heads * group;
        let ctx_lens = [10usize, 3, 20];
        let num_seqs = ctx_lens.len();
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let gen = |dev: &Device, n: usize, seed: f32| -> Result<Tensor> {
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.123 + seed).sin()) * 0.5)
                .collect();
            Tensor::from_vec(data, n, dev)
        };
        let run = |dev: &Device| -> Result<Tensor> {
            let q = gen(dev, num_seqs * num_heads * head_dim, 0.0)?
                .reshape((num_seqs, num_heads, head_dim))?;
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for (s, &ctx) in ctx_lens.iter().enumerate() {
                keys.push(
                    gen(dev, ctx * num_kv_heads * head_dim, 1.0 + s as f32)?.reshape((
                        ctx,
                        num_kv_heads,
                        head_dim,
                    ))?,
                );
                values.push(
                    gen(dev, ctx * num_kv_heads * head_dim, 7.0 + s as f32)?.reshape((
                        ctx,
                        num_kv_heads,
                        head_dim,
                    ))?,
                );
            }
            let total_blocks: usize = ctx_lens.iter().map(|c| c.div_ceil(block_size)).sum();
            let (k_cache, v_cache, block_tables, context_lens) = build_paged_cache(
                &keys,
                &values,
                total_blocks + 2,
                block_size,
                num_kv_heads,
                head_dim,
                dev,
            )?;
            paged_attention(
                &q,
                &k_cache,
                &v_cache,
                &block_tables,
                &context_lens,
                block_size,
                scale,
                None,
            )
        };

        let cpu = run(&Device::Cpu)?;
        let metal = run(&dev)?.to_device(&Device::Cpu)?;
        let diff = (cpu - metal)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < EPS,
            "Metal vs CPU paged attention diff {diff} exceeds {EPS}"
        );
        Ok(())
    }
}
