//! Block-paged KV-cache attention (a.k.a. PagedAttention), device-agnostic.
//!
//! This is a portable, pure-`candle` implementation of the attention mechanism
//! popularised by vLLM, where the key/value cache is not a single contiguous
//! buffer per sequence but a pool of fixed-size *blocks*. Each sequence keeps a
//! *block table* mapping its logical block indices to physical blocks in the
//! shared pool, which makes it possible to:
//!
//! - share identical prefixes across requests (prefix caching),
//! - add or evict sequences without re-allocating the whole cache,
//! - keep memory fragmentation low through block pooling.
//!
//! The fused CUDA kernels added in [#3655] (`candle-flash-attn`) cover the GPU
//! decode/prefill path. This module is the complementary CPU/Metal (and CUDA
//! fallback) path requested in [#3656]: it is written entirely in terms of
//! `candle` tensor operations (`index_select` + dense attention), so it runs
//! unchanged on every backend and doubles as a correctness reference for the
//! fused kernels.
//!
//! The paged cache tensor layout matches the one used by the fused kernels in
//! `candle-flash-attn`:
//!
//! - `k_cache`, `v_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
//! - a token's *slot* in the flattened cache is `block_id * block_size + offset`
//! - `block_tables`: `[num_seqs, max_blocks_per_seq]` (`u32`), logical → physical
//! - `context_lens`: `[num_seqs]` (`u32`), number of valid KV tokens per sequence
//!
//! [#3655]: https://github.com/huggingface/candle/pull/3655
//! [#3656]: https://github.com/huggingface/candle/issues/3656

use crate::ops;
use candle::{DType, Device, IndexOp, Result, Tensor};

/// Decode-step paged attention.
///
/// Computes attention for a single query token per sequence against a paged
/// KV cache, matching the canonical PagedAttention *decode* signature.
///
/// **Shapes**
/// - `q`: `[num_seqs, num_heads, head_dim]`
/// - `k_cache`, `v_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
/// - `block_tables`: `[num_seqs, max_blocks_per_seq]` (`u32`)
/// - `context_lens`: `[num_seqs]` (`u32`)
///
/// **Output**: `[num_seqs, num_heads, head_dim]`
///
/// Grouped-query / multi-query attention is supported: `num_heads` must be a
/// multiple of `num_kv_heads`, and query head `h` attends to KV head
/// `h / (num_heads / num_kv_heads)` (the vLLM / GQA convention).
///
/// `alibi_slopes`, when provided, must have shape `[num_heads]` and adds the
/// standard ALiBi distance bias for the decode position.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    block_size: usize,
    scale: f32,
    alibi_slopes: Option<&Tensor>,
) -> Result<Tensor> {
    let device = q.device();
    let dtype = q.dtype();
    let (num_seqs, num_heads, head_dim) = q.dims3()?;
    let (num_blocks, cache_block_size, num_kv_heads, cache_head_dim) = k_cache.dims4()?;

    if cache_block_size != block_size {
        candle::bail!(
            "k_cache block dim ({cache_block_size}) does not match block_size ({block_size})"
        );
    }
    if cache_head_dim != head_dim {
        candle::bail!(
            "k_cache head_dim ({cache_head_dim}) does not match query head_dim ({head_dim})"
        );
    }
    if k_cache.dims() != v_cache.dims() {
        candle::bail!(
            "k_cache and v_cache must share the same shape ({:?} vs {:?})",
            k_cache.dims(),
            v_cache.dims()
        );
    }
    if num_heads % num_kv_heads != 0 {
        candle::bail!(
            "num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        );
    }
    let group = num_heads / num_kv_heads;

    let (bt_seqs, max_blocks) = block_tables.dims2()?;
    if bt_seqs != num_seqs {
        candle::bail!("block_tables has {bt_seqs} rows but q has {num_seqs} sequences");
    }
    if context_lens.dims1()? != num_seqs {
        candle::bail!(
            "context_lens has len {} but q has {num_seqs} sequences",
            context_lens.dims1()?
        );
    }

    // Flatten the block tables and context lengths to plain integers so we can
    // build per-sequence gathers. These are tiny (one entry per sequence /
    // block) and live on the host regardless of the compute device.
    let block_tables = block_tables.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let context_lens = context_lens.to_dtype(DType::U32)?.to_vec1::<u32>()?;

    let mut outputs = Vec::with_capacity(num_seqs);
    for seq in 0..num_seqs {
        let ctx_len = context_lens[seq] as usize;
        if ctx_len == 0 {
            outputs.push(Tensor::zeros((num_heads, head_dim), dtype, device)?);
            continue;
        }
        let needed_blocks = ctx_len.div_ceil(block_size);
        if needed_blocks > max_blocks {
            candle::bail!(
                "sequence {seq} needs {needed_blocks} blocks for context_len {ctx_len} but block_tables only has {max_blocks} columns"
            );
        }

        // Gather the physical blocks backing this sequence, then flatten the
        // (block, offset) dims into a contiguous time axis and trim to the
        // real context length.
        let block_ids: Vec<u32> = block_tables[seq][..needed_blocks].to_vec();
        for &b in &block_ids {
            if b as usize >= num_blocks {
                candle::bail!(
                    "sequence {seq} references physical block {b} but the cache only has {num_blocks} blocks"
                );
            }
        }
        let block_ids = Tensor::from_vec(block_ids, needed_blocks, device)?;

        // [needed_blocks, block_size, num_kv_heads, head_dim]
        let k_blocks = k_cache.index_select(&block_ids, 0)?;
        let v_blocks = v_cache.index_select(&block_ids, 0)?;
        // -> [needed_blocks * block_size, num_kv_heads, head_dim] -> trim
        let k_seq = k_blocks
            .reshape((needed_blocks * block_size, num_kv_heads, head_dim))?
            .narrow(0, 0, ctx_len)?;
        let v_seq = v_blocks
            .reshape((needed_blocks * block_size, num_kv_heads, head_dim))?
            .narrow(0, 0, ctx_len)?;

        // Expand KV heads to query heads for GQA/MQA, following h / group.
        let (k_seq, v_seq) = if group == 1 {
            (k_seq, v_seq)
        } else {
            let expand = |t: Tensor| -> Result<Tensor> {
                t.reshape((ctx_len, num_kv_heads, 1, head_dim))?
                    .broadcast_as((ctx_len, num_kv_heads, group, head_dim))?
                    .reshape((ctx_len, num_heads, head_dim))
            };
            (expand(k_seq)?, expand(v_seq)?)
        };

        // Attention for the single query token.
        // q_seq: [num_heads, 1, head_dim]
        let q_seq = q.i(seq)?.unsqueeze(1)?;
        // k/v: [num_heads, ctx_len, head_dim]
        let k_seq = k_seq.transpose(0, 1)?.contiguous()?;
        let v_seq = v_seq.transpose(0, 1)?.contiguous()?;

        let scores = q_seq.matmul(&k_seq.transpose(1, 2)?.contiguous()?)?; // [H, 1, ctx]
        let scale_t = Tensor::new(scale, device)?.to_dtype(scores.dtype())?;
        let mut scores = scores.broadcast_mul(&scale_t)?;

        if let Some(alibi_slopes) = alibi_slopes {
            let bias = alibi_bias(ctx_len, num_heads, alibi_slopes, device)?.to_dtype(dtype)?;
            scores = scores.add(&bias)?;
        }

        let probs = ops::softmax_last_dim(&scores)?; // [H, 1, ctx]
        let out = probs.matmul(&v_seq)?; // [H, 1, head_dim]
        outputs.push(out.squeeze(1)?); // [H, head_dim]
    }

    Tensor::stack(&outputs, 0)
}

/// ALiBi decode bias of shape `[num_heads, 1, ctx_len]`.
///
/// For the most recent query token (position `ctx_len - 1`), the bias applied
/// to key position `j` is `-slope_h * (ctx_len - 1 - j)`.
fn alibi_bias(
    ctx_len: usize,
    num_heads: usize,
    alibi_slopes: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let slopes = alibi_slopes.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if slopes.len() != num_heads {
        candle::bail!(
            "alibi_slopes has len {}, expected num_heads={num_heads}",
            slopes.len()
        );
    }
    let mut per_head = Vec::with_capacity(num_heads);
    for &slope in &slopes {
        let bias: Vec<f32> = (0..ctx_len)
            .map(|j| -slope * (ctx_len - 1 - j) as f32)
            .collect();
        per_head.push(Tensor::from_vec(bias, (1, ctx_len), device)?);
    }
    Tensor::stack(&per_head, 0) // [H, 1, ctx]
}

/// Write freshly computed keys/values into a paged KV cache.
///
/// This is the paged analogue of appending to a contiguous cache: each input
/// token is written to the slot given by `slot_mapping`, where a slot is
/// `block_id * block_size + offset_within_block` into the flattened cache.
///
/// **Shapes**
/// - `k`, `v`: `[num_tokens, num_kv_heads, head_dim]`
/// - `k_cache`, `v_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
/// - `slot_mapping`: `[num_tokens]` (`u32`)
///
/// Returns the updated `(k_cache, v_cache)`. The operation is functional (it
/// does not mutate the inputs in place), which keeps it portable across all
/// backends.
pub fn reshape_and_cache(
    k: &Tensor,
    v: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (num_blocks, block_size, num_kv_heads, head_dim) = k_cache.dims4()?;
    if k_cache.dims() != v_cache.dims() {
        candle::bail!(
            "k_cache and v_cache must share the same shape ({:?} vs {:?})",
            k_cache.dims(),
            v_cache.dims()
        );
    }
    let (num_tokens, k_heads, k_dim) = k.dims3()?;
    if k_heads != num_kv_heads || k_dim != head_dim {
        candle::bail!(
            "k has shape {:?} but cache expects [*, {num_kv_heads}, {head_dim}]",
            k.dims()
        );
    }
    if v.dims() != k.dims() {
        candle::bail!(
            "k and v must share the same shape ({:?} vs {:?})",
            k.dims(),
            v.dims()
        );
    }
    if slot_mapping.dims1()? != num_tokens {
        candle::bail!(
            "slot_mapping has len {} but there are {num_tokens} tokens",
            slot_mapping.dims1()?
        );
    }

    let num_slots = num_blocks * block_size;
    let kc = k_cache.reshape((num_slots, num_kv_heads, head_dim))?;
    let vc = v_cache.reshape((num_slots, num_kv_heads, head_dim))?;

    // Broadcast the per-token slot index across the (head, dim) axes so it
    // lines up with the source tensors for `scatter` along dim 0.
    let index = slot_mapping
        .to_dtype(DType::U32)?
        .reshape((num_tokens, 1, 1))?
        .broadcast_as((num_tokens, num_kv_heads, head_dim))?
        .contiguous()?;

    let kc = kc.scatter(&index, &k.contiguous()?, 0)?;
    let vc = vc.scatter(&index, &v.contiguous()?, 0)?;

    Ok((
        kc.reshape((num_blocks, block_size, num_kv_heads, head_dim))?,
        vc.reshape((num_blocks, block_size, num_kv_heads, head_dim))?,
    ))
}

/// A simple pool allocator for physical KV-cache blocks.
///
/// Blocks are handed out one at a time and returned to the free list when a
/// sequence is evicted, implementing the block-pooling memory strategy used for
/// high-throughput serving. Allocation and free are O(1).
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    block_size: usize,
    num_blocks: usize,
    free: Vec<usize>,
}

impl BlockAllocator {
    /// Create an allocator managing `num_blocks` physical blocks of
    /// `block_size` tokens each.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        // Hand out low block ids first; this keeps tables readable in tests.
        let free = (0..num_blocks).rev().collect();
        Self {
            block_size,
            num_blocks,
            free,
        }
    }

    /// Number of tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total number of physical blocks managed.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Number of blocks currently available.
    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    /// Allocate a single physical block, or `None` if the pool is exhausted.
    pub fn allocate(&mut self) -> Option<usize> {
        self.free.pop()
    }

    /// Allocate enough contiguous-in-logical-order blocks to hold `num_tokens`
    /// tokens, returning their physical ids. Returns `None` (and frees nothing)
    /// if the pool cannot satisfy the request.
    pub fn allocate_for_tokens(&mut self, num_tokens: usize) -> Option<Vec<usize>> {
        let needed = num_tokens.div_ceil(self.block_size);
        if self.free.len() < needed {
            return None;
        }
        Some((0..needed).map(|_| self.free.pop().unwrap()).collect())
    }

    /// Return a previously allocated block to the pool.
    pub fn free(&mut self, block: usize) {
        debug_assert!(
            block < self.num_blocks,
            "freeing out-of-range block {block}"
        );
        self.free.push(block);
    }

    /// Return a batch of blocks to the pool.
    pub fn free_all(&mut self, blocks: impl IntoIterator<Item = usize>) {
        for b in blocks {
            self.free(b);
        }
    }
}
