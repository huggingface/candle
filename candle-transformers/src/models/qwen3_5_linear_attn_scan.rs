//! Chunk-based GatedDeltaNet parallel scan for prefill.
//!
//! Replaces the O(T) sequential recurrence with the HuggingFace-proven
//! chunked WY (Woodbury) approach: O(T/chunk_size) sequential steps.
//!
//! Reference: transformers/models/qwen3_5/modeling_qwen3_5.py
//!            `torch_chunk_gated_delta_rule`
//!
//! # Performance TODOs
//!
//! 1. **Redundant `decay_mask` materialisation** — `diff.broadcast_mul(&tril_strict)?`
//!    allocates a full [b, n_h, C, S, S] tensor just to zero the upper triangle
//!    before `exp`, then re-multiplies by `tril_strict` after `exp`.  A fused
//!    masked-exp kernel (or computing only the lower triangle) would halve memory
//!    traffic for this step.
//!
//! 2. **Inter-chunk loop sequential matmuls** — the `for ci in 0..num_chunks` loop
//!    issues 3 independent matmuls per chunk sequentially.  Some could be batched
//!    across chunks (e.g. `v_prime = w @ state` before the loop if state were
//!    fixed), but causality requires sequential state updates.  Investigate whether
//!    CUDA graphs or explicit stream pipelining could hide latency.
//!
//! 3. **Inter-chunk state tensor re-allocation** — `s = (s.broadcast_mul(...) + state_update)?`
//!    allocates a new tensor on every chunk iteration.  For long sequences (many chunks)
//!    this adds up to `num_chunks` extra allocations of [b, n_h, hk, hv].  If candle
//!    exposes in-place `mul_` / `add_` ops, use them to update `s` without re-allocating.
//!
//! 4. **Defensive `contiguous()` copies** — many `narrow` + `contiguous` + `reshape`
//!    chains create intermediate copies.  Profile to identify which are load-bearing
//!    (CUDA non-contiguous reshape bug) vs unnecessary, and remove the latter.

use candle::{DType, Result, Tensor, D};

const CHUNK_SIZE: usize = 64;

/// Pure-Candle chunked GatedDeltaNet for prefill (t > 1).
///
/// Inputs are all F32, shapes:
///   q:     [b, t, n_heads, head_k_dim]
///   k:     [b, t, n_heads, head_k_dim]
///   v:     [b, t, n_heads, head_v_dim]
///   log_g: [b, t, n_heads]  (log of per-head decay, i.e. -a_exp * softplus(a + dt_bias))
///   beta:  [b, t, n_heads]
///   state: [b, n_heads, head_k_dim, head_v_dim]  (mutable, updated in-place)
///
/// `log_g` must be passed directly (not as `g.log()`) to avoid log(0) = -inf on
/// CUDA where subnormal float32 values are flushed to zero, which would produce
/// NaN through -inf - (-inf) in the decay_diff computation.
///
/// Returns: [b, t, n_heads, head_v_dim]
pub fn gated_delta_rule_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let device = q.device().clone();
    let (b, t, n_heads, head_k_dim) = q.dims4()?;
    let head_v_dim = v.dim(3)?;

    let chunk = CHUNK_SIZE;
    let num_chunks = t.div_ceil(chunk);
    let pad_t = num_chunks * chunk;
    let needs_pad = pad_t != t;
    // Flat batch size used to collapse [b, n_h, C] into one dim for 3D matmul.
    // CUDA gemm_config only handles ≤2 batch-prefix dims; 5D tensors with 3 batch
    // dims fall through to MatMulNonContiguous. Reshaping to [bhnc, S, d] avoids this.
    let bhnc = b * n_heads * num_chunks;

    // Reshape [b, t, n_h, d] -> [b, n_h, num_chunks, chunk, d] with padding
    let reshape_4d = |tensor: &Tensor, d: usize| -> Result<Tensor> {
        let padded = if needs_pad {
            let zeros = Tensor::zeros((b, pad_t - t, n_heads, d), DType::F32, &device)?;
            Tensor::cat(&[tensor, &zeros], 1)?
        } else {
            tensor.clone()
        };
        Ok(padded
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, n_heads, num_chunks, chunk, d))?)
    };

    let reshape_3d = |tensor: &Tensor| -> Result<Tensor> {
        let padded = if needs_pad {
            let zeros = Tensor::zeros((b, pad_t - t, n_heads), DType::F32, &device)?;
            Tensor::cat(&[tensor, &zeros], 1)?
        } else {
            tensor.clone()
        };
        Ok(padded
            .permute((0, 2, 1))?
            .contiguous()?
            .reshape((b, n_heads, num_chunks, chunk))?)
    };

    let q_c = reshape_4d(q, head_k_dim)?; // [b, n_h, C, S, hk]
    let k_c = reshape_4d(k, head_k_dim)?; // [b, n_h, C, S, hk]
    let v_c = reshape_4d(v, head_v_dim)?; // [b, n_h, C, S, hv]
    let log_g_c = reshape_3d(log_g)?; // [b, n_h, C, S]
    let beta_c = reshape_3d(beta)?; // [b, n_h, C, S]

    // ── Step 3a: Log-decay cumsum + decay mask ────────────────────────────
    // g_cumsum[i] = sum(log_g[0..i+1]) within each chunk
    let g_cumsum = log_g_c.cumsum(D::Minus1)?; // [b, n_h, C, S]

    // decay_mask[i,j] = exp(g_cumsum[i] - g_cumsum[j]) for i > j, else 0
    // tril_strict: strictly lower triangular (diagonal = 0) — position i must not
    // read its own write in the same step, so A[i,i] = 0 by definition.
    let tril = Tensor::tril2(chunk, DType::F32, &device)?; // [S, S]
    let tril_strict = (&tril - &Tensor::eye(chunk, DType::F32, &device)?)?; // diagonal zeroed
                                                                            // Materialize both expansion directions before sub: broadcast_sub on CUDA
                                                                            // doesn't work when both sides need different-dimension expansion simultaneously.
    let outer = (b, n_heads, num_chunks, chunk, chunk);
    let gc_i = g_cumsum
        .unsqueeze(D::Minus1)?
        .broadcast_as(outer)?
        .contiguous()?; // [b, n_h, C, S, S] — each row i repeated S times
    let gc_j = g_cumsum
        .unsqueeze(D::Minus2)?
        .broadcast_as(outer)?
        .contiguous()?; // [b, n_h, C, S, S] — each col j repeated S times
                        // Apply tril masks BEFORE exp to prevent float32 overflow in the upper-triangular region.
                        // For the upper tri (j > i), diff[i,j] = g_cumsum[i] - g_cumsum[j] > 0 (positive, since
                        // g_cumsum is non-increasing). For long sequences with strong decay, this can reach
                        // ~(chunk-1)*max_|log_g| which exceeds ln(f32::MAX)≈88, producing +inf after exp.
                        // Then tril*inf = 0*inf = NaN. Lower-triangular entries are always ≤ 0 so exp ≤ 1 — safe.
                        // Fix: zero upper-tri in diff first; exp(0)=1 in those slots; then re-zero with the mask.
    let diff = (&gc_i - &gc_j)?;
    let decay_mask = diff
        .broadcast_mul(&tril_strict)?
        .exp()?
        .broadcast_mul(&tril_strict)?;
    let decay_mask_full = diff.broadcast_mul(&tril)?.exp()?.broadcast_mul(&tril)?;

    // ── Step 3b: Weighted keys/values ─────────────────────────────────────
    let beta_unsq = beta_c.unsqueeze(D::Minus1)?; // [b, n_h, C, S, 1]
    let k_beta = k_c.broadcast_mul(&beta_unsq)?; // [b, n_h, C, S, hk]
    let v_beta = v_c.broadcast_mul(&beta_unsq)?; // [b, n_h, C, S, hv]

    // ── Step 3c: Lower-triangular A = -k_beta @ k^T * decay_mask ──────────
    // k_beta: [b, n_h, C, S, hk], k_c: [b, n_h, C, S, hk]
    // Reshape to 3D [bhnc, S, hk] so CUDA gemm_config sees exactly 1 batch dim.
    let kk = k_beta
        .reshape((bhnc, chunk, head_k_dim))?
        .broadcast_matmul(
            &k_c.reshape((bhnc, chunk, head_k_dim))?
                .transpose(D::Minus1, D::Minus2)?
                .contiguous()?,
        )?
        .reshape((b, n_heads, num_chunks, chunk, chunk))?;
    // kk: [b, n_h, C, S, S] — this is k_beta @ k^T
    // .contiguous() ensures a_mat has dense row-major strides before narrow+reshape
    // in the forward-substitution loop.  broadcast_mul may produce a strided view
    // on CUDA (like torch), so the explicit copy here is required for correctness.
    let a_mat: Tensor = kk.broadcast_mul(&decay_mask)?.neg()?.contiguous()?;

    // ── Step 3d: Solve (I + A) via forward substitution ───────────────────
    // attn starts as identity + lower-triangular part solved row by row.
    // Row 0: attn[0] = e_0 (identity, since A[0,:] = 0 for strictly lower tri)
    // Row i: attn[i,j] = A[i,j] + sum_k(attn[k,j] * A[i,k]) for k < i, j < i
    //
    // We build attn as [b, n_h, C, S, S], init with identity.
    let identity = Tensor::eye(chunk, DType::F32, &device)?.reshape((1, 1, 1, chunk, chunk))?;
    let identity = identity
        .broadcast_as((b, n_heads, num_chunks, chunk, chunk))?
        .contiguous()?;
    let attn = identity;

    // Forward substitution: solve (I − a_mat) * attn = I  →  attn = (I − a_mat)^{-1}
    //
    // Derivation: unrolling the recurrence gives u_j = rhs_j + Σ_{k<j} a_mat[j,k] u_k
    // i.e. (I − a_mat) u = rhs, so attn = (I − a_mat)^{-1}.
    //
    // Row i recurrence: attn[i, :] = e_i + sum_{k<i} a_mat[i, k] * attn[k, :]
    // (a_mat entries are negative, so this slightly shrinks each row away from the identity)
    //
    // Row 0 is already e_0 from the identity initialisation.
    for i in 1..chunk {
        // attn[0..i, :] — already-solved rows.
        // .contiguous() is required: narrow produces a strided view, and the
        // subsequent reshape to [bhnc, i, S] is only valid on a dense tensor.
        // On CUDA, reshaping a non-contiguous view reinterprets strides incorrectly.
        let attn_sub = attn.narrow(D::Minus2, 0, i)?.contiguous()?; // [b, n_h, C, i, S]

        // a_mat[i, 0:i] — row i of A, first i sub-diagonal entries.
        // Same reasoning: two narrows leave non-trivial strides; make contiguous
        // before the reshape merges the (b, n_h, C) prefix into bhnc.
        let a_sub = a_mat
            .narrow(D::Minus2, i, 1)? // row i:    [b, n_h, C, 1, S]
            .narrow(D::Minus1, 0, i)? // cols 0..i: [b, n_h, C, 1, i]
            .contiguous()?;

        // contrib = a_mat[i, 0:i] @ attn[0:i, :] — [b,n_h,C,1,i] @ [b,n_h,C,i,S] → [b,n_h,C,1,S]
        // Reshape to 3D: CUDA gemm_config only handles ≤2 batch-prefix dims.
        let contrib = a_sub
            .reshape((bhnc, 1, i))?
            .broadcast_matmul(&attn_sub.reshape((bhnc, i, chunk))?)?
            .reshape((b, n_heads, num_chunks, 1, chunk))?;

        // new_row = e_i + contrib  (row i of attn is still e_i: only rows 0..i-1 have been updated)
        // Solving (I − a_mat) X = I: X[i,:] = e_i + Σ_{k<i} a_mat[i,k] X[k,:]
        // (a_mat has negative entries, so this slightly shrinks each row)
        // same_storage check in slice_set is between attn and new_row — safe because + allocates new storage.
        let cur_row = attn.narrow(D::Minus2, i, 1)?; // [b, n_h, C, 1, S] = e_i (unmodified)
        let new_row = (&cur_row + &contrib)?.contiguous()?; // [b, n_h, C, 1, S]

        // Splice new_row into attn at position i. In-place via copy2d — avoids the
        // alloc + full-tensor copy of Tensor::cat and slice_scatter.
        attn.slice_set(&new_row, D::Minus2, i)?;
    }

    // ── Step 3e: Apply WY representation ──────────────────────────────────
    // value_new = attn @ v_beta — reshape to 3D for CUDA gemm_config
    let value_new = attn
        .reshape((bhnc, chunk, chunk))?
        .broadcast_matmul(&v_beta.reshape((bhnc, chunk, head_v_dim))?)?
        .reshape((b, n_heads, num_chunks, chunk, head_v_dim))?; // [b, n_h, C, S, hv]

    // w = attn @ (k_beta * exp(g_cumsum))
    let g_exp = g_cumsum.exp()?.unsqueeze(D::Minus1)?; // [b, n_h, C, S, 1]
    let k_beta_scaled = k_beta.broadcast_mul(&g_exp)?; // [b, n_h, C, S, hk]
                                                       // w_ci @ state gives the intra-chunk state correction.
    let w = attn
        .reshape((bhnc, chunk, chunk))?
        .broadcast_matmul(&k_beta_scaled.reshape((bhnc, chunk, head_k_dim))?)?
        .reshape((b, n_heads, num_chunks, chunk, head_k_dim))?; // [b, n_h, C, S, hk]

    // ── Step 4: Inter-chunk state propagation ─────────────────────────────
    // Flat 2D batch size: collapse [b, n_h] into one dim for 3D matmuls.
    // Mirrors the bhnc workaround above — CUDA gemm_config may fall through to
    // MatMulNonContiguous for 4D tensors with 2 batch-prefix dims, so we
    // reshape to explicit 3D to guarantee the fast batched-GEMM path.
    let bn = b * n_heads;
    let mut outputs = Vec::with_capacity(num_chunks);
    let mut s = state.clone(); // [b, n_heads, hk, hv]

    for ci in 0..num_chunks {
        // narrow(chunk-dim) + squeeze leaves non-contiguous strides.
        // Explicit .contiguous() is required before any reshape that merges
        // (b, n_h) into bn — on CUDA, a non-contiguous reshape reinterprets
        // strides and silently produces wrong values.
        let q_ci = q_c.narrow(2, ci, 1)?.squeeze(2)?.contiguous()?; // [b, n_h, S, hk]
        let k_ci = k_c.narrow(2, ci, 1)?.squeeze(2)?.contiguous()?; // [b, n_h, S, hk]
        let v_new_ci = value_new.narrow(2, ci, 1)?.squeeze(2)?.contiguous()?; // [b, n_h, S, hv]
        let w_ci = w.narrow(2, ci, 1)?.squeeze(2)?.contiguous()?; // [b, n_h, S, hk]
                                                                  // decay_ci_full: uses tril (diagonal=1) so position i reads its own write
        let decay_ci_full = decay_mask_full.narrow(2, ci, 1)?.squeeze(2)?.contiguous()?; // [b, n_h, S, S]
        let gc_ci = g_cumsum.narrow(2, ci, 1)?.squeeze(2)?.contiguous()?; // [b, n_h, S]

        // v_prime = w_ci @ s: [bn, S, hk] @ [bn, hk, hv] -> [bn, S, hv]
        let s3 = s.reshape((bn, head_k_dim, head_v_dim))?;
        let v_prime = w_ci
            .reshape((bn, chunk, head_k_dim))?
            .broadcast_matmul(&s3)?
            .reshape((b, n_heads, chunk, head_v_dim))?;

        // v_corrected = value_new - v_prime
        let v_corrected = (&v_new_ci - &v_prime)?; // [b, n_h, S, hv]

        // Intra-chunk attention: [bn, S, hk] @ [bn, hk, S] -> [bn, S, S]
        let a_intra = q_ci
            .reshape((bn, chunk, head_k_dim))?
            .broadcast_matmul(
                &k_ci
                    .reshape((bn, chunk, head_k_dim))?
                    .transpose(D::Minus1, D::Minus2)?
                    .contiguous()?,
            )?
            .reshape((b, n_heads, chunk, chunk))?
            .broadcast_mul(&decay_ci_full)?; // [b, n_h, S, S]

        // Inter-chunk: (q * exp(g_cumsum)) @ s: [bn, S, hk] @ [bn, hk, hv] -> [bn, S, hv]
        let gc_exp = gc_ci.exp()?.unsqueeze(D::Minus1)?; // [b, n_h, S, 1]
        let q_scaled = q_ci.broadcast_mul(&gc_exp)?; // [b, n_h, S, hk]
        let attn_inter = q_scaled
            .reshape((bn, chunk, head_k_dim))?
            .broadcast_matmul(&s3)?
            .reshape((b, n_heads, chunk, head_v_dim))?; // [b, n_h, S, hv]

        // output = attn_inter + a_intra @ v_corrected: [bn, S, S] @ [bn, S, hv]
        let out_ci = (attn_inter
            + a_intra
                .reshape((bn, chunk, chunk))?
                .broadcast_matmul(&v_corrected.reshape((bn, chunk, head_v_dim))?)?
                .reshape((b, n_heads, chunk, head_v_dim))?)?; // [b, n_h, S, hv]
        outputs.push(out_ci.unsqueeze(2)?); // [b, n_h, 1, S, hv]

        // State update: compute decay from chunk start to chunk end
        let g_end = gc_ci.narrow(D::Minus1, chunk - 1, 1)?.squeeze(D::Minus1)?; // [b, n_h]
        let g_end_exp = g_end.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // [b, n_h, 1, 1]

        // decay_to_end[j] = exp(g_cumsum[S-1] - g_cumsum[j])
        let gc_last = gc_ci.narrow(D::Minus1, chunk - 1, 1)?; // [b, n_h, 1]
        let decay_to_end = gc_last.broadcast_sub(&gc_ci)?.exp()?.unsqueeze(D::Minus1)?; // [b, n_h, S, 1]

        // state = state * exp(g_cumsum[-1]) + k_weighted^T @ v_corrected
        // [bn, hk, S] @ [bn, S, hv] -> [bn, hk, hv]
        let k_weighted = k_ci.broadcast_mul(&decay_to_end)?; // [b, n_h, S, hk]
        let state_update = k_weighted
            .reshape((bn, chunk, head_k_dim))?
            .transpose(D::Minus1, D::Minus2)?
            .contiguous()?
            .broadcast_matmul(&v_corrected.reshape((bn, chunk, head_v_dim))?)?
            .reshape((b, n_heads, head_k_dim, head_v_dim))?; // [b, n_h, hk, hv]
        s = (s.broadcast_mul(&g_end_exp)? + state_update)?;
    }

    // Save state
    *state = s.detach();

    // ── Step 5: Truncate padding + reshape output ─────────────────────────
    let out_all = Tensor::cat(&outputs, 2)?; // [b, n_h, C, S, hv]
                                             // Permute back to [b, C*S, n_h, hv] = [b, pad_t, n_h, hv]
    let out_perm = out_all
        .permute((0, 2, 3, 1, 4))?
        .contiguous()?
        .reshape((b, pad_t, n_heads, head_v_dim))?;
    // Truncate padding
    let out = out_perm.narrow(1, 0, t)?;
    Ok(out)
}

/// Reference sequential loop for t>1 (used in tests to verify chunked output).
/// Inputs: q/k/v [b, t, n_h, d], g/beta [b, t, n_h]. State: [b, n_h, hk, hv].
/// Returns [b, t, n_h, hv].
#[cfg(test)]
fn sequential_loop(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (_b, t, _n_heads, _) = q.dims4()?;
    let _head_v_dim = v.dim(3)?;
    // Permute to [b, n_h, t, d] so we can iterate over the time axis
    let q_p = q.permute((0, 2, 1, 3))?.contiguous()?; // [b, n_h, t, hk]
    let k_p = k.permute((0, 2, 1, 3))?.contiguous()?;
    let v_p = v.permute((0, 2, 1, 3))?.contiguous()?;
    let g_p = g.permute((0, 2, 1))?.contiguous()?; // [b, n_h, t]
    let b_p = beta.permute((0, 2, 1))?.contiguous()?;
    let mut outputs = Vec::with_capacity(t);
    for i in 0..t {
        let q_t = q_p.narrow(2, i, 1)?.squeeze(2)?; // [b, n_h, hk]
        let k_t = k_p.narrow(2, i, 1)?.squeeze(2)?;
        let v_t = v_p.narrow(2, i, 1)?.squeeze(2)?;
        let g_t = g_p.narrow(2, i, 1)?.squeeze(2)?; // [b, n_h]
        let beta_t = b_p.narrow(2, i, 1)?.squeeze(2)?;
        let out = sequential_step(&q_t, &k_t, &v_t, &g_t, &beta_t, state)?; // [b, n_h, hv]
        outputs.push(out.unsqueeze(2)?); // [b, n_h, 1, hv]
    }
    // cat along time → [b, n_h, t, hv], then permute to [b, t, n_h, hv]
    let out_all = Tensor::cat(&outputs, 2)?;
    out_all
        .permute((0, 2, 1, 3))?
        .contiguous()
        .map_err(Into::into)
}

/// Sequential single-step decode (t=1). Extracted from the original loop.
///
/// Inputs are all F32:
///   q_t:  [b, n_heads, head_k_dim]
///   k_t:  [b, n_heads, head_k_dim]
///   v_t:  [b, n_heads, head_v_dim]
///   g_t:  [b, n_heads]
///   beta_t: [b, n_heads]
///   state: [b, n_heads, head_k_dim, head_v_dim]
///
/// Returns: [b, n_heads, head_v_dim]
pub fn sequential_step(
    q_t: &Tensor,
    k_t: &Tensor,
    v_t: &Tensor,
    g_t: &Tensor,
    beta_t: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    // Decay
    *state = state.broadcast_mul(&g_t.unsqueeze(2)?.unsqueeze(3)?)?;

    // Read: kv_mem = (state * k_t[:,:,None,:]).sum(-2)
    let kv_mem = (state.broadcast_mul(&k_t.unsqueeze(3)?)?).sum(D::Minus2)?;

    // Delta
    let diff = (v_t - &kv_mem)?;
    let delta = diff.broadcast_mul(&beta_t.unsqueeze(2)?)?;

    // Write: state += k_t[:,:,:,None] * delta[:,:,None,:]
    *state = (&*state + k_t.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;

    // Read output
    let out = (state.broadcast_mul(&q_t.unsqueeze(3)?)?).sum(D::Minus2)?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = (a - b).unwrap().abs().unwrap();
        diff.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
    }

    fn run_chunked_vs_sequential(
        device: &Device,
        b: usize,
        t: usize,
        n_heads: usize,
        hk: usize,
        hv: usize,
    ) {
        let q = Tensor::randn(0f32, 0.1f32, (b, t, n_heads, hk), device).unwrap();
        let k = Tensor::randn(0f32, 0.1f32, (b, t, n_heads, hk), device).unwrap();
        let v = Tensor::randn(0f32, 1.0f32, (b, t, n_heads, hv), device).unwrap();
        // g in (0,1): sigmoid of randn. log_g = log(g) is always finite since sigmoid != 0.
        let g_raw = Tensor::randn(-2.0f32, 1.0f32, (b, t, n_heads), device).unwrap();
        let g = candle_nn::ops::sigmoid(&g_raw).unwrap();
        // log_g: for the model this is -a_exp*softplus (always negative, no underflow risk).
        // In tests, g.log() is safe since sigmoid is always > 0.
        let log_g = g.log().unwrap();
        let beta_raw = Tensor::randn(0f32, 1.0f32, (b, t, n_heads), device).unwrap();
        let beta = candle_nn::ops::sigmoid(&beta_raw).unwrap();

        let mut state_seq = Tensor::zeros((b, n_heads, hk, hv), DType::F32, device).unwrap();
        let mut state_chk = Tensor::zeros((b, n_heads, hk, hv), DType::F32, device).unwrap();

        let out_seq = sequential_loop(&q, &k, &v, &g, &beta, &mut state_seq).unwrap();
        let out_chk = gated_delta_rule_chunked(&q, &k, &v, &log_g, &beta, &mut state_chk).unwrap();

        let out_diff = max_abs_diff(&out_seq, &out_chk);
        let state_diff = max_abs_diff(&state_seq, &state_chk);

        println!("b={b} t={t} n_h={n_heads} hk={hk} hv={hv}: out_diff={out_diff:.6} state_diff={state_diff:.6}");

        assert!(
            out_diff < 1e-3,
            "b={b} t={t}: output mismatch: max diff = {out_diff}"
        );
        assert!(
            state_diff < 1e-3,
            "b={b} t={t}: state mismatch: max diff = {state_diff}"
        );
    }

    /// Run chunked scan with large-magnitude log_g (≈ −6.7 per step) to reproduce the
    /// float32 overflow bug: exp(g_cumsum[i] − g_cumsum[j]) for upper-triangular (i < j)
    /// can reach exp(~275) → +inf for t ≥ 15 tokens.  Before the fix, inf * 0 = NaN.
    fn run_chunked_large_decay(device: &Device, t: usize, n_heads: usize, hk: usize, hv: usize) {
        let b = 1usize;

        let q = Tensor::randn(0f32, 0.1f32, (b, t, n_heads, hk), device).unwrap();
        let k = Tensor::randn(0f32, 0.1f32, (b, t, n_heads, hk), device).unwrap();
        let v = Tensor::randn(0f32, 1.0f32, (b, t, n_heads, hv), device).unwrap();
        // Use log_g ≈ −6.72 to match observed Qwen3.5-0.8B values.
        // For t=42, the upper-triangular exponent reaches ~41*6.72≈275, overflowing f32.
        let log_g = Tensor::full(-6.72f32, (b, t, n_heads), device).unwrap();
        let beta = Tensor::full(0.65f32, (b, t, n_heads), device).unwrap();

        let g = log_g.exp().unwrap();
        let mut state_seq = Tensor::zeros((b, n_heads, hk, hv), DType::F32, device).unwrap();
        let mut state_chk = Tensor::zeros((b, n_heads, hk, hv), DType::F32, device).unwrap();

        let out_seq = sequential_loop(&q, &k, &v, &g, &beta, &mut state_seq).unwrap();
        let out_chk = gated_delta_rule_chunked(&q, &k, &v, &log_g, &beta, &mut state_chk).unwrap();

        // Verify no NaN first
        let has_nan = out_chk
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .any(|x| x.is_nan());
        assert!(
            !has_nan,
            "t={t}: NaN in chunked output (overflow bug not fixed)"
        );

        let out_diff = max_abs_diff(&out_seq, &out_chk);
        println!("large_decay t={t} n_h={n_heads}: out_diff={out_diff:.6}");
        assert!(
            out_diff < 1e-3,
            "t={t}: output mismatch: max diff = {out_diff}"
        );
    }

    #[test]
    fn chunked_large_decay_cpu() {
        let device = Device::Cpu;
        // t=14 (threshold: 13*6.72≈87 < 88 — just under overflow without fix)
        run_chunked_large_decay(&device, 14, 2, 4, 4);
        // t=15 would have overflowed without the fix (14*6.72≈94 > 88)
        run_chunked_large_decay(&device, 15, 2, 4, 4);
        // t=42: the actual failing case in production
        run_chunked_large_decay(&device, 42, 2, 4, 4);
        // realistic model dims
        run_chunked_large_decay(&device, 42, 16, 128, 128);
    }

    #[test]
    fn chunked_large_decay_cuda() {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        if matches!(device, Device::Cpu) {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        }
        run_chunked_large_decay(&device, 14, 2, 4, 4);
        run_chunked_large_decay(&device, 15, 2, 4, 4);
        run_chunked_large_decay(&device, 42, 2, 4, 4);
        run_chunked_large_decay(&device, 42, 16, 128, 128);
    }

    #[test]
    fn chunked_matches_sequential_cpu() {
        let device = Device::Cpu;
        // t < CHUNK_SIZE (padded): simulates typical first/second REPL turn
        run_chunked_vs_sequential(&device, 1, 29, 2, 4, 4);
        run_chunked_vs_sequential(&device, 1, 56, 2, 4, 4);
        // t == CHUNK_SIZE (no padding)
        run_chunked_vs_sequential(&device, 1, 64, 2, 4, 4);
        // t > CHUNK_SIZE (multiple chunks)
        run_chunked_vs_sequential(&device, 1, 128, 2, 4, 4);
        // realistic model dims (n_heads=16, hk=hv=128) — single chunk
        run_chunked_vs_sequential(&device, 1, 56, 16, 128, 128);
        // realistic model dims — multi-chunk (2 chunks)
        run_chunked_vs_sequential(&device, 1, 128, 16, 128, 128);
        // realistic model dims — 3 chunks
        run_chunked_vs_sequential(&device, 1, 192, 16, 128, 128);
        // batch > 1: validates identity broadcast and slice_set across batch dim
        run_chunked_vs_sequential(&device, 2, 56, 2, 4, 4);
        run_chunked_vs_sequential(&device, 2, 128, 2, 4, 4);
    }

    #[test]
    fn chunked_matches_sequential_cuda() {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        if matches!(device, Device::Cpu) {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        }
        run_chunked_vs_sequential(&device, 1, 29, 2, 4, 4);
        run_chunked_vs_sequential(&device, 1, 56, 2, 4, 4);
        run_chunked_vs_sequential(&device, 1, 128, 2, 4, 4);
        // single chunk, realistic dims
        run_chunked_vs_sequential(&device, 1, 56, 16, 128, 128);
        // multi-chunk, realistic dims — the case that fails in production
        run_chunked_vs_sequential(&device, 1, 128, 16, 128, 128);
        run_chunked_vs_sequential(&device, 1, 200, 16, 128, 128);
        // batch > 1: validates identity broadcast and slice_set across batch dim
        run_chunked_vs_sequential(&device, 2, 56, 2, 4, 4);
        run_chunked_vs_sequential(&device, 2, 128, 2, 4, 4);
    }

    /// Run a chunked prefill followed by sequential decode steps, comparing
    /// against a pure-sequential reference over all tokens.
    /// This tests that the state handed off from chunked prefill to
    /// sequential_step is correct — the exact pattern used in production
    /// (turn N prefill → token-by-token generation).
    fn run_prefill_then_decode(
        device: &Device,
        t_prefill: usize,
        t_decode: usize,
        n_heads: usize,
        hk: usize,
        hv: usize,
    ) {
        let b = 1usize;
        let all_t = t_prefill + t_decode;

        // Build all inputs on CPU then move to target device.
        let cpu = Device::Cpu;
        let q_all = Tensor::randn(0f32, 0.1f32, (b, all_t, n_heads, hk), &cpu)
            .unwrap()
            .to_device(device)
            .unwrap();
        let k_all = Tensor::randn(0f32, 0.1f32, (b, all_t, n_heads, hk), &cpu)
            .unwrap()
            .to_device(device)
            .unwrap();
        let v_all = Tensor::randn(0f32, 1.0f32, (b, all_t, n_heads, hv), &cpu)
            .unwrap()
            .to_device(device)
            .unwrap();
        let g_raw = Tensor::randn(-2.0f32, 1.0f32, (b, all_t, n_heads), &cpu).unwrap();
        let g_all = candle_nn::ops::sigmoid(&g_raw)
            .unwrap()
            .to_device(device)
            .unwrap();
        let log_g_all = g_all.log().unwrap(); // safe: sigmoid > 0 always
        let beta_raw = Tensor::randn(0f32, 1.0f32, (b, all_t, n_heads), &cpu).unwrap();
        let beta_all = candle_nn::ops::sigmoid(&beta_raw)
            .unwrap()
            .to_device(device)
            .unwrap();

        // Reference: pure sequential over all tokens.
        let mut state_ref = Tensor::zeros((b, n_heads, hk, hv), DType::F32, device).unwrap();
        let out_ref =
            sequential_loop(&q_all, &k_all, &v_all, &g_all, &beta_all, &mut state_ref).unwrap();

        // Test: chunked prefill then sequential decode.
        let mut state_test = Tensor::zeros((b, n_heads, hk, hv), DType::F32, device).unwrap();

        // ── Prefill ───────────────────────────────────────────────────────────
        let out_pre = gated_delta_rule_chunked(
            &q_all.narrow(1, 0, t_prefill).unwrap(),
            &k_all.narrow(1, 0, t_prefill).unwrap(),
            &v_all.narrow(1, 0, t_prefill).unwrap(),
            &log_g_all.narrow(1, 0, t_prefill).unwrap(),
            &beta_all.narrow(1, 0, t_prefill).unwrap(),
            &mut state_test,
        )
        .unwrap();

        let pre_diff = max_abs_diff(&out_ref.narrow(1, 0, t_prefill).unwrap(), &out_pre);

        // ── Decode steps ──────────────────────────────────────────────────────
        // Permute decode slice to [b, n_h, t_decode, d] for easy per-step narrow.
        let q_dec = q_all
            .narrow(1, t_prefill, t_decode)
            .unwrap()
            .permute((0, 2, 1, 3))
            .unwrap()
            .contiguous()
            .unwrap();
        let k_dec = k_all
            .narrow(1, t_prefill, t_decode)
            .unwrap()
            .permute((0, 2, 1, 3))
            .unwrap()
            .contiguous()
            .unwrap();
        let v_dec = v_all
            .narrow(1, t_prefill, t_decode)
            .unwrap()
            .permute((0, 2, 1, 3))
            .unwrap()
            .contiguous()
            .unwrap();
        let g_dec = g_all
            .narrow(1, t_prefill, t_decode)
            .unwrap()
            .permute((0, 2, 1))
            .unwrap()
            .contiguous()
            .unwrap();
        let beta_dec = beta_all
            .narrow(1, t_prefill, t_decode)
            .unwrap()
            .permute((0, 2, 1))
            .unwrap()
            .contiguous()
            .unwrap();

        let mut dec_outs = Vec::with_capacity(t_decode);
        for i in 0..t_decode {
            let out = sequential_step(
                &q_dec.narrow(2, i, 1).unwrap().squeeze(2).unwrap(),
                &k_dec.narrow(2, i, 1).unwrap().squeeze(2).unwrap(),
                &v_dec.narrow(2, i, 1).unwrap().squeeze(2).unwrap(),
                &g_dec.narrow(2, i, 1).unwrap().squeeze(2).unwrap(),
                &beta_dec.narrow(2, i, 1).unwrap().squeeze(2).unwrap(),
                &mut state_test,
            )
            .unwrap();
            dec_outs.push(out.unsqueeze(2).unwrap()); // [b, n_h, 1, hv]
        }
        // [b, n_h, t_decode, hv] → [b, t_decode, n_h, hv]
        let out_dec = Tensor::cat(&dec_outs, 2)
            .unwrap()
            .permute((0, 2, 1, 3))
            .unwrap();

        let dec_diff = max_abs_diff(&out_ref.narrow(1, t_prefill, t_decode).unwrap(), &out_dec);

        println!(
            "prefill_then_decode t_pre={t_prefill} t_dec={t_decode} n_h={n_heads} hk={hk}: \
             prefill_diff={pre_diff:.6} decode_diff={dec_diff:.6}"
        );

        assert!(pre_diff < 1e-3, "prefill mismatch: {pre_diff}");
        assert!(dec_diff < 1e-3, "decode mismatch after prefill: {dec_diff}");
    }

    /// Test candle's depthwise conv1d (groups = c) on CUDA vs CPU.
    /// This directly exercises LinearAttn::apply_conv1d_silu with the exact
    /// parameters used by Qwen3.5-0.8B (kernel=4, c=6144).
    /// If this test fails, the bug is in candle's CUDA conv1d kernel.
    #[allow(dead_code)]
    fn run_depthwise_conv1d(device: &Device, c: usize, kernel: usize, t: usize) {
        let cpu = Device::Cpu;
        let pad_len = kernel - 1;
        let total = pad_len + t;

        // Build weight [c, 1, kernel] and input [1, c, total] on CPU then move.
        let weight = Tensor::randn(0f32, 0.1f32, (c, 1, kernel), &cpu).unwrap();
        let inp = Tensor::randn(0f32, 1.0f32, (1usize, c, total), &cpu).unwrap();

        let out_cpu = inp.conv1d(&weight, 0, 1, 1, c).unwrap();

        let weight_d = weight.to_device(device).unwrap();
        let inp_d = inp.to_device(device).unwrap();
        let out_d = inp_d
            .conv1d(&weight_d, 0, 1, 1, c)
            .unwrap()
            .to_device(&cpu)
            .unwrap();

        let diff = max_abs_diff(&out_cpu, &out_d);
        println!("depthwise_conv1d c={c} kernel={kernel} t={t}: diff={diff:.6}");
        assert!(diff < 1e-3, "conv1d CUDA vs CPU mismatch at t={t}: {diff}");
    }

    #[test]
    fn depthwise_conv1d_cpu() {
        let device = Device::Cpu;
        // Qwen3.5-0.8B: n_h=16, hk=hv=128 → key_dim=2048, c=6144
        run_depthwise_conv1d(&device, 6144, 4, 30);
        run_depthwise_conv1d(&device, 6144, 4, 128);
        run_depthwise_conv1d(&device, 6144, 4, 200);
    }

    #[test]
    fn depthwise_conv1d_cuda() {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        if matches!(device, Device::Cpu) {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        }
        // Qwen3.5-0.8B exact params: kernel=4, c=6144
        run_depthwise_conv1d(&device, 6144, 4, 30); // short (turn 1)
        run_depthwise_conv1d(&device, 6144, 4, 80); // medium
        run_depthwise_conv1d(&device, 6144, 4, 128); // 2 chunks
        run_depthwise_conv1d(&device, 6144, 4, 200); // long (turn 2)
    }

    #[test]
    fn prefill_then_decode_cpu() {
        let device = Device::Cpu;
        run_prefill_then_decode(&device, 50, 20, 2, 4, 4);
        run_prefill_then_decode(&device, 80, 20, 16, 128, 128); // 2 chunks
        run_prefill_then_decode(&device, 128, 30, 16, 128, 128); // exactly 2 chunks
    }

    #[test]
    fn prefill_then_decode_cuda() {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        if matches!(device, Device::Cpu) {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        }
        run_prefill_then_decode(&device, 50, 20, 2, 4, 4);
        // single-chunk prefill, realistic dims
        run_prefill_then_decode(&device, 50, 20, 16, 128, 128);
        // multi-chunk prefill, realistic dims — matches production second turn
        run_prefill_then_decode(&device, 80, 20, 16, 128, 128);
        run_prefill_then_decode(&device, 128, 30, 16, 128, 128);
        run_prefill_then_decode(&device, 200, 30, 16, 128, 128);
    }

    /// Validate `slice_scatter` correctness on CUDA by running `gated_delta_rule_chunked`
    /// on both CPU and CUDA with identical inputs and requiring tight numerical agreement.
    ///
    /// The forward substitution loop is the primary user of `slice_scatter`.  Any device-
    /// specific bug (wrong strides, misaligned write, etc.) surfaces here before the error
    /// propagates through the rest of the attention computation and becomes hard to diagnose.
    fn run_forward_subst_cpu_vs_cuda(
        cuda: &Device,
        b: usize,
        t: usize,
        n_heads: usize,
        hk: usize,
        hv: usize,
    ) {
        let cpu = Device::Cpu;

        // Build inputs on CPU, then clone to CUDA.
        let q_cpu = Tensor::randn(0f32, 0.1f32, (b, t, n_heads, hk), &cpu).unwrap();
        let k_cpu = Tensor::randn(0f32, 0.1f32, (b, t, n_heads, hk), &cpu).unwrap();
        let v_cpu = Tensor::randn(0f32, 1.0f32, (b, t, n_heads, hv), &cpu).unwrap();
        let g_raw = Tensor::randn(-2.0f32, 1.0f32, (b, t, n_heads), &cpu).unwrap();
        let log_g_cpu = candle_nn::ops::sigmoid(&g_raw).unwrap().log().unwrap();
        let beta_cpu =
            candle_nn::ops::sigmoid(&Tensor::randn(0f32, 1.0f32, (b, t, n_heads), &cpu).unwrap())
                .unwrap();

        let q_gpu = q_cpu.to_device(cuda).unwrap();
        let k_gpu = k_cpu.to_device(cuda).unwrap();
        let v_gpu = v_cpu.to_device(cuda).unwrap();
        let log_g_gpu = log_g_cpu.to_device(cuda).unwrap();
        let beta_gpu = beta_cpu.to_device(cuda).unwrap();

        let mut state_cpu = Tensor::zeros((b, n_heads, hk, hv), DType::F32, &cpu).unwrap();
        let mut state_gpu = Tensor::zeros((b, n_heads, hk, hv), DType::F32, cuda).unwrap();

        let out_cpu = gated_delta_rule_chunked(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            &log_g_cpu,
            &beta_cpu,
            &mut state_cpu,
        )
        .unwrap();
        let out_gpu = gated_delta_rule_chunked(
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &log_g_gpu,
            &beta_gpu,
            &mut state_gpu,
        )
        .unwrap()
        .to_device(&cpu)
        .unwrap();

        // No NaN/Inf on GPU path.
        let flat: Vec<f32> = out_gpu.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|x| x.is_finite()),
            "t={t}: NaN or Inf in CUDA chunked output"
        );

        // Tight tolerance: same algorithm on both devices should agree to ~1e-5.
        let diff = max_abs_diff(&out_cpu, &out_gpu);
        println!(
            "forward_subst_cpu_vs_cuda b={b} t={t} n_h={n_heads} hk={hk} hv={hv}: diff={diff:.2e}"
        );
        assert!(
            diff < 5e-5,
            "b={b} t={t}: CPU vs CUDA output mismatch after slice_set: max diff = {diff}"
        );

        // State should also match.
        let state_gpu_cpu = state_gpu.to_device(&cpu).unwrap();
        let state_diff = max_abs_diff(&state_cpu, &state_gpu_cpu);
        assert!(
            state_diff < 5e-5,
            "b={b} t={t}: CPU vs CUDA state mismatch after slice_set: max diff = {state_diff}"
        );
    }

    /// CPU-vs-CUDA parity test for the forward substitution `slice_scatter` path.
    #[test]
    fn forward_subst_cpu_vs_cuda() {
        let cuda = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        if matches!(cuda, Device::Cpu) {
            eprintln!("CUDA not available, skipping forward_subst_cpu_vs_cuda");
            return;
        }
        // Single-chunk (t ≤ 64): exercises the full 63-iteration loop.
        run_forward_subst_cpu_vs_cuda(&cuda, 1, 30, 2, 4, 4);
        run_forward_subst_cpu_vs_cuda(&cuda, 1, 64, 2, 4, 4); // exactly one chunk, no padding
        run_forward_subst_cpu_vs_cuda(&cuda, 1, 50, 16, 128, 128); // realistic single-chunk
                                                                   // Multi-chunk: verifies correct behaviour across chunk boundaries.
        run_forward_subst_cpu_vs_cuda(&cuda, 1, 128, 16, 128, 128); // 2 chunks
        run_forward_subst_cpu_vs_cuda(&cuda, 1, 200, 16, 128, 128); // 3+ chunks
                                                                    // batch > 1: validates identity broadcast and slice_set across batch dim
        run_forward_subst_cpu_vs_cuda(&cuda, 2, 56, 2, 4, 4);
        run_forward_subst_cpu_vs_cuda(&cuda, 2, 128, 2, 4, 4);
    }
}
