//! KV cache implementations for Qwen3-TTS autoregressive generation.

use candle::{DType, Device, Result, Tensor};

/// Simple concatenation-based KV cache.
pub struct KVCache {
    pub(crate) k: Option<Tensor>,
    pub(crate) v: Option<Tensor>,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn update_k(&mut self, k: &Tensor) -> Result<Tensor> {
        let k = if let Some(prev) = &self.k {
            Tensor::cat(&[prev, k], 2)?
        } else {
            k.clone()
        };
        self.k = Some(k.clone());
        Ok(k)
    }

    pub fn update_v(&mut self, v: &Tensor) -> Result<Tensor> {
        let v = if let Some(prev) = &self.v {
            Tensor::cat(&[prev, v], 2)?
        } else {
            v.clone()
        };
        self.v = Some(v.clone());
        Ok(v)
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Pre-allocated fixed-size KV cache using `slice_set` for in-place writes.
pub struct PreAllocKVCache {
    k_buf: Tensor,
    v_buf: Tensor,
    current_len: usize,
    max_seq: usize,
    num_heads: usize,
    head_dim: usize,
}

impl PreAllocKVCache {
    pub fn new(
        batch: usize,
        num_heads: usize,
        max_seq: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = (batch, num_heads, max_seq, head_dim);
        Ok(Self {
            k_buf: Tensor::zeros(shape, dtype, device)?,
            v_buf: Tensor::zeros(shape, dtype, device)?,
            current_len: 0,
            max_seq,
            num_heads,
            head_dim,
        })
    }

    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq = k.dim(2)?;
        let new_len = self.current_len + new_seq;
        if new_len > self.max_seq {
            candle::bail!(
                "KV cache overflow: current={} + new={} > max={}",
                self.current_len,
                new_seq,
                self.max_seq
            );
        }
        let k_c = k.contiguous()?;
        let v_c = v.contiguous()?;
        self.k_buf.slice_set(&k_c, 2, self.current_len)?;
        self.v_buf.slice_set(&v_c, 2, self.current_len)?;
        self.current_len = new_len;
        let k_view = self.k_buf.narrow(2, 0, new_len)?;
        let v_view = self.v_buf.narrow(2, 0, new_len)?;
        Ok((k_view, v_view))
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    pub fn len(&self) -> usize {
        self.current_len
    }

    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }
}

// ── CircularKVCache ───────────────────────────────────────────────────────────

/// Ring-buffer KV cache for sliding-window attention.
///
/// Stores at most `window` key/value vectors. When the buffer is full the
/// oldest slot is overwritten. The attention window therefore always covers
/// the last `window` tokens in absolute-position order.
///
/// # Absolute vs window positions
///
/// RoPE embeddings are applied **before** the cache, keyed on the *absolute*
/// token position (`offset` parameter). The cache stores those already-rotated
/// vectors, so attention sees the correct relative distances automatically.
///
/// When the buffer wraps, the stored vectors are no longer in contiguous
/// time order.  `update` re-assembles them into chronological order before
/// returning (oldest slot first) and reports `window_start` — the absolute
/// position of the oldest retained token — so callers can build the right
/// causal mask.
///
/// # Causal mask with circular cache
///
/// For a query at absolute position `q_pos` the mask must be:
/// - 0.0  for every key at absolute position ≤ `q_pos`
/// - −∞   for every key at absolute position > `q_pos`
///
/// Because the window only ever contains past+current tokens (we never store
/// future tokens), and queries are always the *newest* token, the mask is
/// **all zeros** once the buffer fills: every stored key is causally valid
/// for the current query. Before the buffer fills the standard causal mask
/// applies (same as the prefill path).
///
/// See `create_window_causal_mask` for the helper that builds this.
pub struct CircularKVCache {
    k_buf: Tensor,             // [B, H, window, D]
    v_buf: Tensor,
    write_pos: usize,          // next slot to write (wraps modulo window)
    filled: usize,             // number of valid entries, capped at window
    abs_start: usize,          // absolute position of slot 0 in the ring
    window: usize,
    num_heads: usize,
    head_dim: usize,
}

impl CircularKVCache {
    pub fn new(
        batch: usize,
        num_heads: usize,
        window: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = (batch, num_heads, window, head_dim);
        Ok(Self {
            k_buf: Tensor::zeros(shape, dtype, device)?,
            v_buf: Tensor::zeros(shape, dtype, device)?,
            write_pos: 0,
            filled: 0,
            abs_start: 0,
            window,
            num_heads,
            head_dim,
        })
    }

    /// Write one or more new tokens' K/V into the ring buffer.
    ///
    /// Supports `seq ≥ 1` — used for multi-token prefill as well as
    /// single-step autoregressive generation.
    ///
    /// Returns `(k_window, v_window, window_start_abs)` where:
    /// - `k_window` / `v_window` are `[B, H, filled, D]` in chronological order
    /// - `window_start_abs` is the absolute position of the oldest token in the window
    pub fn update(
        &mut self,
        k: &Tensor,   // [B, H, seq, D]
        v: &Tensor,
    ) -> Result<(Tensor, Tensor, usize)> {
        let seq = k.dim(2)?;
        // Write each token into the ring one at a time
        for t in 0..seq {
            let k_t = k.narrow(2, t, 1)?;
            let v_t = v.narrow(2, t, 1)?;
            let slot = self.write_pos % self.window;
            self.k_buf.slice_set(&k_t.contiguous()?, 2, slot)?;
            self.v_buf.slice_set(&v_t.contiguous()?, 2, slot)?;
            if self.filled == self.window {
                self.abs_start += 1;
            } else {
                self.filled += 1;
            }
            self.write_pos += 1;
        }
        let (k_win, v_win) = self.ordered_window()?;
        Ok((k_win, v_win, self.abs_start))
    }

    /// Absolute position of the oldest token currently in the window.
    pub fn window_start_abs(&self) -> usize {
        self.abs_start
    }

    /// Number of tokens currently stored in the cache.
    pub fn filled(&self) -> usize {
        self.filled
    }

    pub fn reset(&mut self) {
        self.write_pos = 0;
        self.filled = 0;
        self.abs_start = 0;
        // No need to zero the buffer — filled tracks validity
    }

    /// Return the stored K or V in chronological order (oldest first).
    ///
    /// The ring write pointer sits at `write_pos % window`.  The oldest
    /// entry is at `write_pos % window` when the buffer is full (it is the
    /// slot that will be overwritten *next*), or at slot 0 when not yet full.
    fn ordered_window(&self) -> Result<(Tensor, Tensor)> {
        if self.filled < self.window {
            // Buffer not yet full — slots 0..filled are in order
            let k = self.k_buf.narrow(2, 0, self.filled)?;
            let v = self.v_buf.narrow(2, 0, self.filled)?;
            Ok((k, v))
        } else {
            // Buffer full — oldest slot is at write_pos % window
            let oldest = self.write_pos % self.window;
            if oldest == 0 {
                // Convenient alignment — no split needed
                Ok((self.k_buf.clone(), self.v_buf.clone()))
            } else {
                // Concat [oldest..window, 0..oldest]
                let k = Tensor::cat(
                    &[
                        self.k_buf.narrow(2, oldest, self.window - oldest)?,
                        self.k_buf.narrow(2, 0, oldest)?,
                    ],
                    2,
                )?;
                let v = Tensor::cat(
                    &[
                        self.v_buf.narrow(2, oldest, self.window - oldest)?,
                        self.v_buf.narrow(2, 0, oldest)?,
                    ],
                    2,
                )?;
                Ok((k, v))
            }
        }
    }
}

// ── AnyKVCache ────────────────────────────────────────────────────────────────

/// Result returned by [`AnyKVCache::update`].
///
/// For `Concat` and `PreAlloc` caches the window always starts at absolute
/// position 0, so callers that only use those variants can ignore
/// `window_start` (it will always be 0 until the buffer fills) or use the
/// convenience `update_simple` which returns only `(k, v)`.
///
/// For `Circular` caches `window_start` is the absolute position of the
/// **oldest** token in the returned window — callers must use this to build
/// the correct causal mask.
pub struct KVUpdateResult {
    pub k: Tensor,
    pub v: Tensor,
    /// Absolute position of the first token in the returned K/V window.
    /// Always 0 for Concat/PreAlloc (full history retained).
    pub window_start: usize,
}

/// Unified KV cache: concat-based, pre-allocated, or circular/sliding-window.
pub enum AnyKVCache {
    Concat(KVCache),
    PreAlloc(PreAllocKVCache),
    Circular(CircularKVCache),
}

impl AnyKVCache {
    /// Update the cache with new K/V, returning the full window plus metadata.
    ///
    /// For `Concat`/`PreAlloc` `window_start` is always 0 (full history).
    /// For `Circular` `window_start` is the absolute position of the oldest
    /// retained token — the caller must use this to build the causal mask.
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<KVUpdateResult> {
        match self {
            AnyKVCache::Concat(c) => {
                let k = c.update_k(k)?;
                let v = c.update_v(v)?;
                Ok(KVUpdateResult { k, v, window_start: 0 })
            }
            AnyKVCache::PreAlloc(c) => {
                let (k, v) = c.update(k, v)?;
                Ok(KVUpdateResult { k, v, window_start: 0 })
            }
            AnyKVCache::Circular(c) => {
                let (k, v, window_start) = c.update(k, v)?;
                Ok(KVUpdateResult { k, v, window_start })
            }
        }
    }

    pub fn reset(&mut self) {
        match self {
            AnyKVCache::Concat(c) => c.reset(),
            AnyKVCache::PreAlloc(c) => c.reset(),
            AnyKVCache::Circular(c) => c.reset(),
        }
    }
}

// ── Causal mask helpers for windowed attention ────────────────────────────────

/// Build a causal mask for `q_seq` query tokens (starting at absolute position
/// `q_start_abs`) attending to a K/V window that spans absolute positions
/// `[window_start, window_start + window_len)`.
///
/// Returns a `[1, 1, q_seq, window_len]` F32 tensor:
/// - `0.0` for positions ≤ query position (causally valid)
/// - `f32::NEG_INFINITY` for positions > query position (masked)
///
/// For autoregressive generation steps (`q_seq = 1`) every stored K/V is
/// causally valid so the mask is all zeros.
pub fn create_window_causal_mask(
    q_start_abs: usize,
    q_seq: usize,
    window_start: usize,
    window_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![0.0f32; q_seq * window_len];
    for qi in 0..q_seq {
        let q_abs = q_start_abs + qi;
        for ki in 0..window_len {
            let k_abs = window_start + ki;
            if k_abs > q_abs {
                mask[qi * window_len + ki] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, q_seq, window_len), device)
}
