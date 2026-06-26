//! Candle-native interleaved repacking of Q6_K weights for fast aarch64 GEMM/GEMV.
//!
//! Mirrors the Q4_K x8 idea (8-row-interleaved layout consumed by a dedicated SDOT
//! kernel) for the Q6_K residual tensors (attn_v / ffn_down that Q4_K_M keeps as
//! Q6_K). The packed `BlockQ6Kx8` is loaded from a baked GGUF (GgmlDType::Q6Kx8) and
//! driven by `gemm_q6kx8_q8k`; no runtime repack.
#![allow(clippy::needless_range_loop)]

use super::k_quants::{BlockQ6K, BlockQ8K, GgmlType, QK_K};
use half::f16;
use std::sync::{Arc, LazyLock};

// ---- shared interleave constants ----
pub(crate) const Q4KX8_ROWS: usize = 8;

// ---- prefill tile selection (shared with the Q4 path upstream) ----
#[derive(Clone, Copy, PartialEq)]
enum PrefillTile {
    Nc8mr4,
    Nc4mr4,
    Nc8mr2,
}
static PACKED_PREFILL_TILE: LazyLock<PrefillTile> = LazyLock::new(|| {
    match std::env::var("CANDLE_PACKED_PREFILL")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "nc8mr4" => PrefillTile::Nc8mr4,
        "nc8mr2" => PrefillTile::Nc8mr2,
        _ => PrefillTile::Nc4mr4,
    }
});

// ---- packed Q6_K weight block + repack ----
pub struct BlockQ6Kx8 {
    pub(crate) d: [f16; Q4KX8_ROWS],
    pub(crate) scales: [i8; Q4KX8_ROWS * (QK_K / 16)], // 128
    pub(crate) ql: [u8; Q4KX8_ROWS * (QK_K / 2)],      // 1024: [chunk(2)][row(8)][64]
    pub(crate) qh: [u8; Q4KX8_ROWS * (QK_K / 4)],      // 512:  [chunk(2)][row(8)][32]
}

impl BlockQ6Kx8 {
    fn zeroed() -> Self {
        Self {
            d: [f16::ZERO; Q4KX8_ROWS],
            scales: [0; Q4KX8_ROWS * (QK_K / 16)],
            ql: [0; Q4KX8_ROWS * (QK_K / 2)],
            qh: [0; Q4KX8_ROWS * (QK_K / 4)],
        }
    }
}

/// Repack `Q4KX8_ROWS` Q6_K weight rows (each `nb` super-blocks) into `nb`
/// interleaved `BlockQ6Kx8`. Scalar (load-time, not perf-critical); the i8 scales
/// copy directly and the 128B `ql`/64B `qh` split into two 64B/32B chunks.
pub(crate) fn repack_q6k_x8(rows: &[&[BlockQ6K]; Q4KX8_ROWS]) -> Vec<BlockQ6Kx8> {
    let nb = rows[0].len();
    debug_assert!(
        rows.iter().all(|r| r.len() == nb),
        "ragged rows in repack_q6k_x8"
    );
    const QLC: usize = QK_K / 4; // 64 ql bytes / 128-value chunk
    const QHC: usize = QK_K / 8; // 32 qh bytes / chunk
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let mut blk = BlockQ6Kx8::zeroed();
        for r in 0..Q4KX8_ROWS {
            let src = &rows[r][i];
            blk.d[r] = src.d;
            for s in 0..(QK_K / 16) {
                blk.scales[r * (QK_K / 16) + s] = src.scales[s];
            }
            for j in 0..2 {
                let qld = j * (Q4KX8_ROWS * QLC) + r * QLC;
                blk.ql[qld..qld + QLC].copy_from_slice(&src.ql[j * QLC..j * QLC + QLC]);
                let qhd = j * (Q4KX8_ROWS * QHC) + r * QHC;
                blk.qh[qhd..qhd + QHC].copy_from_slice(&src.qh[j * QHC..j * QHC + QHC]);
            }
        }
        out.push(blk);
    }
    out
}

/// Repack a full row-major Q6_K weight matrix into the interleaved `BlockQ6Kx8`
/// layout for all `n/8` channel groups. Mirror of `repack_q4k_weight`.
pub fn repack_q6k_weight(rows: &[BlockQ6K], n: usize, nb: usize) -> Vec<BlockQ6Kx8> {
    debug_assert_eq!(n % Q4KX8_ROWS, 0, "repack_q6k_weight: n {n} not /8");
    debug_assert_eq!(rows.len(), n * nb, "repack_q6k_weight: rows len mismatch");
    let mut packed = Vec::with_capacity((n / 8) * nb);
    for g in 0..n / 8 {
        let grp: [&[BlockQ6K]; Q4KX8_ROWS] =
            std::array::from_fn(|r| &rows[(g * 8 + r) * nb..(g * 8 + r + 1) * nb]);
        packed.extend(repack_q6k_x8(&grp));
    }
    packed
}

// ---- prepacked GEMM driver ----
pub(crate) fn matmul_q6kx8_prepacked(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    packed: &[BlockQ6Kx8],
    dst: &mut [f32],
) {
    use super::neon::gemm_q6kx8_q8k;
    let nb = k / QK_K;
    let groups = n / Q4KX8_ROWS;
    // Q6 has only the 8-channel kernel; widen the prefill tile to MR=4 (the nc8mr4
    // default) for the same channel+row reuse win, else MR=2.
    let mr4 = *PACKED_PREFILL_TILE == PrefillTile::Nc8mr4;

    thread_local! {
        static LHS_Q: std::cell::RefCell<Vec<BlockQ8K>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    LHS_Q.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < m * nb {
            scratch.resize(m * nb, BlockQ8K::zeros());
        }
        for a in 0..m {
            BlockQ8K::from_float(&lhs[a * k..(a + 1) * k], &mut scratch[a * nb..(a + 1) * nb]);
        }
        let lhs_q: &[BlockQ8K] = &scratch;

        struct DstPtr(*mut f32);
        unsafe impl Sync for DstPtr {}
        let dptr = DstPtr(dst.as_mut_ptr());
        let process = |g: usize| {
            let w = &packed[g * nb..(g + 1) * nb];
            let p = &dptr;
            let row = |r: usize| -> &[BlockQ8K] { &lhs_q[r * nb..(r + 1) * nb] };
            let mut r = 0;
            if mr4 {
                while r + 4 <= m {
                    let rows: [&[BlockQ8K]; 4] = std::array::from_fn(|a| row(r + a));
                    let mut t = [0f32; Q4KX8_ROWS * 4];
                    gemm_q6kx8_q8k::<4>(w, &rows, &mut t);
                    for c in 0..Q4KX8_ROWS {
                        for a in 0..4 {
                            unsafe { *p.0.add((r + a) * n + g * 8 + c) = t[c * 4 + a] };
                        }
                    }
                    r += 4;
                }
            }
            while r + 2 <= m {
                let rows: [&[BlockQ8K]; 2] = std::array::from_fn(|a| row(r + a));
                let mut t = [0f32; Q4KX8_ROWS * 2];
                gemm_q6kx8_q8k::<2>(w, &rows, &mut t);
                for c in 0..Q4KX8_ROWS {
                    for a in 0..2 {
                        unsafe { *p.0.add((r + a) * n + g * 8 + c) = t[c * 2 + a] };
                    }
                }
                r += 2;
            }
            while r < m {
                let rows: [&[BlockQ8K]; 1] = [row(r)];
                let mut t = [0f32; Q4KX8_ROWS];
                gemm_q6kx8_q8k::<1>(w, &rows, &mut t);
                for c in 0..Q4KX8_ROWS {
                    unsafe { *p.0.add(r * n + g * 8 + c) = t[c] };
                }
                r += 1;
            }
        };

        let pool = crate::utils::barrier_pool();
        let n_total = pool.n_workers() + 1;
        let gpt = groups.div_ceil(n_total);
        pool.execute(|tid| {
            let start = tid * gpt;
            if start < groups {
                let end = groups.min((tid + 1) * gpt);
                for g in start..end {
                    process(g);
                }
            }
        });
    });
}

// ===========================================================================
// Pre-packed Q6_Kx8 as a first-class quantized storage type (GgmlDType::Q6Kx8).
//
// The interleaved BlockQ6Kx8 layout is baked into a GGUF offline (see the
// gguf-requant `--pack` flag) and loaded as a SINGLE copy with no runtime
// repack. Owned or mmap-backed; the mmap path is zero-copy (GGUF tensors are
// 32-aligned, which covers BlockQ6Kx8's f16/u8 alignment of 2).
// ===========================================================================

// ---- packed storage type (GGUF-loaded) ----
// Backing store for a `PackedQ6Kx8`: either an owned `Vec` (built from raw bytes)
// or a view into a memory-mapped GGUF region (zero-copy, read-only).
enum PackedStoreQ6 {
    Owned(Vec<BlockQ6Kx8>),
    Mmap {
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        count: usize,
    },
}

// SAFETY: identical to PackedStore - the Mmap variant references an immutable,
// file-backed region kept alive by the Arc; BlockQ6Kx8 is a #[repr(C)] POD.
unsafe impl Send for PackedStoreQ6 {}
unsafe impl Sync for PackedStoreQ6 {}

/// A pre-packed Q6_K weight: `n` output channels (`n % 8 == 0`) stored as the
/// interleaved `BlockQ6Kx8` groups, ready for the SDOT GEMM with no repack.
pub struct PackedQ6Kx8 {
    store: PackedStoreQ6,
    n: usize,
}

impl PackedQ6Kx8 {
    #[inline]
    fn as_slice(&self) -> &[BlockQ6Kx8] {
        match &self.store {
            PackedStoreQ6::Owned(v) => v.as_slice(),
            PackedStoreQ6::Mmap {
                mmap,
                offset,
                count,
            } => {
                // SAFETY: offset/count/alignment checked in `from_mmap`; Arc keeps
                // the mapping alive for the lifetime of the returned slice.
                unsafe {
                    std::slice::from_raw_parts(
                        mmap.as_ptr().add(*offset) as *const BlockQ6Kx8,
                        *count,
                    )
                }
            }
        }
    }

    /// Build an owned `PackedQ6Kx8` by copying raw interleaved bytes. `raw` length
    /// must be a whole number of blocks.
    pub fn from_bytes(raw: &[u8], n: usize) -> Self {
        let bs = std::mem::size_of::<BlockQ6Kx8>();
        assert_eq!(
            raw.len() % bs,
            0,
            "PackedQ6Kx8::from_bytes: {} bytes not a multiple of block size {bs}",
            raw.len()
        );
        let count = raw.len() / bs;
        let mut v: Vec<BlockQ6Kx8> = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(raw.as_ptr(), v.as_mut_ptr() as *mut u8, raw.len());
            v.set_len(count);
        }
        Self {
            store: PackedStoreQ6::Owned(v),
            n,
        }
    }

    /// Build a zero-copy `PackedQ6Kx8` viewing `byte_len` bytes at `offset` in an
    /// mmap'd GGUF. Validates bounds and alignment.
    pub fn from_mmap(
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        byte_len: usize,
        n: usize,
    ) -> crate::Result<Self> {
        let bs = std::mem::size_of::<BlockQ6Kx8>();
        if offset + byte_len > mmap.len() {
            crate::bail!(
                "Q6Kx8 mmap region end {} exceeds map len {}",
                offset + byte_len,
                mmap.len()
            );
        }
        if !byte_len.is_multiple_of(bs) {
            crate::bail!("Q6Kx8 mmap byte_len {byte_len} not a multiple of block size {bs}");
        }
        let base = mmap.as_ptr() as usize + offset;
        if !base.is_multiple_of(std::mem::align_of::<BlockQ6Kx8>()) {
            crate::bail!(
                "Q6Kx8 mmap tensor at offset {offset} not aligned to {}",
                std::mem::align_of::<BlockQ6Kx8>()
            );
        }
        Ok(Self {
            store: PackedStoreQ6::Mmap {
                mmap,
                offset,
                count: byte_len / bs,
            },
            n,
        })
    }

    /// Dequantize the packed weight back to f32 of shape `[n, k]` (row-major).
    /// Mirrors `BlockQ6K::to_float` per channel: the i8 sub-block scales and the
    /// `ql`/`qh` quants come from the chunk-major, row-interleaved layout produced
    /// by `repack_q6k_x8`. Correctness-only (not perf).
    fn dequantize_to(&self, elem_count: usize, ys: &mut [f32]) {
        const QLC: usize = QK_K / 4; // 64 ql bytes / 128-value chunk
        const QHC: usize = QK_K / 8; // 32 qh bytes / chunk
        const NSC: usize = QK_K / 16; // 16 i8 scales / row
        let n = self.n;
        let k = elem_count / n;
        let nb = k / QK_K;
        let blocks = self.as_slice();
        debug_assert_eq!(blocks.len(), (n / Q4KX8_ROWS) * nb);
        for g in 0..n / Q4KX8_ROWS {
            for r in 0..Q4KX8_ROWS {
                let out_row = g * Q4KX8_ROWS + r;
                for i in 0..nb {
                    let blk = &blocks[g * nb + i];
                    let d = blk.d[r].to_f32();
                    let base = out_row * k + i * QK_K;
                    // Two 128-value chunks, exactly as BlockQ6K::to_float steps by 128.
                    for idx in 0..2 {
                        let sc = &blk.scales[r * NSC + 8 * idx..];
                        let ql = &blk.ql[idx * (Q4KX8_ROWS * QLC) + r * QLC..];
                        let qh = &blk.qh[idx * (Q4KX8_ROWS * QHC) + r * QHC..];
                        let cbase = base + idx * 128;
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                            ys[cbase + l] = d * sc[is] as f32 * q1 as f32;
                            ys[cbase + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                            ys[cbase + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                            ys[cbase + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                        }
                    }
                }
            }
        }
    }
}

impl super::QuantizedType for PackedQ6Kx8 {
    #[cfg(target_feature = "neon")]
    fn matmul_t(
        &self,
        mkn: (usize, usize, usize),
        lhs: &[f32],
        dst: &mut [f32],
    ) -> crate::Result<()> {
        matmul_q6kx8_prepacked(mkn, lhs, self.as_slice(), dst);
        Ok(())
    }

    #[cfg(not(target_feature = "neon"))]
    fn matmul_t(
        &self,
        _mkn: (usize, usize, usize),
        _lhs: &[f32],
        _dst: &mut [f32],
    ) -> crate::Result<()> {
        crate::bail!("Q6Kx8 packed matmul requires the neon target feature")
    }

    fn matmul_t_f16(
        &self,
        mkn: (usize, usize, usize),
        lhs: &[f16],
        dst: &mut [f16],
    ) -> crate::Result<()> {
        let lhs_f: Vec<f32> = lhs.iter().map(|x| x.to_f32()).collect();
        let mut dst_f = vec![0f32; dst.len()];
        self.matmul_t(mkn, &lhs_f, &mut dst_f)?;
        for (o, v) in dst.iter_mut().zip(dst_f.iter()) {
            *o = f16::from_f32(*v);
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> crate::Result<super::CpuStorage> {
        let mut ys = vec![0f32; elem_count];
        self.dequantize_to(elem_count, &mut ys);
        Ok(super::CpuStorage::F32(ys))
    }

    fn embedding(
        &self,
        _ids: &[u32],
        _rows: usize,
        _hidden: usize,
    ) -> crate::Result<super::CpuStorage> {
        // Q6Kx8 is 8-row interleaved, so individual rows can't be gathered. The
        // `--pack` tool never packs gather-used weights (token_embd stays Q6_K), so
        // this is only a defensive bail.
        crate::bail!("Q6Kx8 is matmul-only (8-row interleaved); embedding gather is unsupported")
    }

    fn storage_size_in_bytes(&self) -> usize {
        std::mem::size_of_val(self.as_slice())
    }

    fn size(&self) -> usize {
        self.storage_size_in_bytes()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr() as *const u8
    }

    fn block_size(&self) -> usize {
        QK_K
    }

    fn dtype(&self) -> super::GgmlDType {
        super::GgmlDType::Q6Kx8
    }

    fn from_float(&mut self, _xs: &[f32]) {
        unreachable!("Q6Kx8 is bake-only; quantize-to is not supported")
    }

    fn from_float_imatrix(&mut self, _xs: &[f32], _imatrix_weights: &[f32], _n_per_row: usize) {
        unreachable!("Q6Kx8 is bake-only; quantize-to is not supported")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::k_quants::GgmlType;

    fn lcg(s: &mut u64) -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        2.0 * ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    // gemm_q6kx8_q8k (packed Q6_K GEMV) must be BIT-IDENTICAL to the scalar
    // vec_dot_q6k_q8k applied per row - it mirrors the same integer ops/order.
    #[cfg(target_feature = "neon")]
    #[test]
    fn gemm_q6kx8_matches_vec_dot() {
        use crate::quantized::k_quants::{BlockQ6K, BlockQ8K};
        use crate::quantized::neon::{gemm_q6kx8_q8k, vec_dot_q6k_q8k};

        let nb = 4usize;
        let k = nb * QK_K;
        let mut st = 0x1357_9bdf_2468_ace0u64;
        let mut rows_q: Vec<Vec<BlockQ6K>> = Vec::new();
        for _ in 0..Q4KX8_ROWS {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ6K::zeros(); nb];
            BlockQ6K::from_float(&f, &mut q);
            rows_q.push(q);
        }
        let af: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
        let mut q8 = vec![BlockQ8K::zeros(); nb];
        BlockQ8K::from_float(&af, &mut q8);

        let reference: [f32; 8] = std::array::from_fn(|r| vec_dot_q6k_q8k(k, &rows_q[r], &q8));
        let refs: [&[BlockQ6K]; Q4KX8_ROWS] = std::array::from_fn(|r| rows_q[r].as_slice());
        let packed = repack_q6k_x8(&refs);
        let mut dst = [0f32; 8];
        gemm_q6kx8_q8k::<1>(&packed, &[q8.as_slice()], &mut dst);
        for c in 0..8 {
            assert_eq!(
                dst[c].to_bits(),
                reference[c].to_bits(),
                "channel {c}: packed {} vs ref {}",
                dst[c],
                reference[c]
            );
        }
    }

    // End-to-end bake check: row-major Q6_K -> repack -> raw bytes -> PackedQ6Kx8
    // must match the scalar Q6_K matmul bit-for-bit, and dequantize() must
    // reproduce the original Q6_K dequant.
    #[test]
    fn packed_q6kx8_from_bytes_matches_baseline() {
        use crate::quantized::k_quants::{matmul, BlockQ6K};
        use crate::quantized::QuantizedType;

        let k = 512usize; // 2 super-blocks
        let n = 24usize; // 3 groups of 8
        let nb = k / QK_K;
        let mut st = 0x51c6_ba9d_2244_8821u64;

        // Row-major Q6_K weight: channel r at r*nb.
        let mut rhs_t = vec![BlockQ6K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            BlockQ6K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }

        // Offline bake: repack -> raw bytes -> PackedQ6Kx8 (owned, from bytes).
        let packed = repack_q6k_weight(&rhs_t, n, nb);
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                packed.as_ptr() as *const u8,
                std::mem::size_of_val(packed.as_slice()),
            )
        };
        let pq = PackedQ6Kx8::from_bytes(raw, n);

        for &m in &[1usize, 4usize] {
            let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_pack = vec![0f32; m * n];
            matmul::<BlockQ6K>((m, k, n), &lhs, &rhs_t, &mut dst_base).unwrap();
            pq.matmul_t((m, k, n), &lhs, &mut dst_pack).unwrap();
            for i in 0..m * n {
                assert_eq!(
                    dst_base[i].to_bits(),
                    dst_pack[i].to_bits(),
                    "m={m} idx {i}: base {} packed {}",
                    dst_base[i],
                    dst_pack[i]
                );
            }
        }

        // dequantize() must reproduce the original Q6_K dequant (same formula).
        let mut want = vec![0f32; n * k];
        BlockQ6K::to_float(&rhs_t, &mut want);
        let got = match pq.dequantize(n * k).unwrap() {
            crate::CpuStorage::F32(v) => v,
            _ => panic!("expected f32"),
        };
        for i in 0..n * k {
            assert_eq!(want[i].to_bits(), got[i].to_bits(), "dequant idx {i}");
        }
    }
}
