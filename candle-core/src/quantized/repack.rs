//! Candle-native interleaved repacking of Q6_K weights for fast aarch64 GEMM/GEMV.
//!
//! Mirrors the Q4_K x8 idea (8-row-interleaved layout consumed by a dedicated SDOT
//! kernel) for the Q6_K residual tensors (attn_v / ffn_down that Q4_K_M keeps as
//! Q6_K). The packed `BlockQ6Kx8` is loaded from a baked GGUF (GgmlDType::Q6Kx8) and
//! driven by `gemm_q6kx8_q8k`; no runtime repack.
#![allow(clippy::needless_range_loop)]

#[cfg(target_feature = "dotprod")]
use super::k_quants::BlockQ4K;
use super::k_quants::{BlockQ6K, QK_K};
// Used only by the NEON-gated Q6 packed GEMM driver.
#[cfg(target_feature = "neon")]
use super::k_quants::{BlockQ8K, GgmlType};
use half::f16;
use std::sync::Arc;
#[cfg(target_feature = "neon")]
use std::sync::LazyLock;

// ---- shared interleave constants ----
pub(crate) const Q4KX8_ROWS: usize = 8;

// ---- prefill tile selection (NEON Q6 driver only) ----
#[cfg(target_feature = "neon")]
#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum PrefillTile {
    Nc8mr4,
    Nc4mr4,
    Nc8mr2,
}
#[cfg(target_feature = "neon")]
static PACKED_PREFILL_TILE: LazyLock<PrefillTile> = LazyLock::new(|| PrefillTile::Nc4mr4);

// On-disk Q6Kx8 GGUF block (from_bytes memcpy / from_mmap cast); repr(C) pins the
// field order so baked models can't mis-decode across rustc versions.
#[repr(C)]
pub struct BlockQ6Kx8 {
    pub(crate) d: [f16; Q4KX8_ROWS],
    pub(crate) scales: [i8; Q4KX8_ROWS * (QK_K / 16)], // 128
    pub(crate) ql: [u8; Q4KX8_ROWS * (QK_K / 2)],      // 1024: [chunk(2)][row(8)][64]
    pub(crate) qh: [u8; Q4KX8_ROWS * (QK_K / 4)],      // 512:  [chunk(2)][row(8)][32]
}
const _: () = assert!(std::mem::size_of::<BlockQ6Kx8>() == 1680);

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

// Repack 8 Q6_K weight rows into interleaved `BlockQ6Kx8` (load-time, scalar).
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

// Repack a full row-major Q6_K weight into `BlockQ6Kx8` for all n/8 channel groups.
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
// NEON-only (uses `neon::gemm_q6kx8_q8k`); `matmul_t` bails on the not(neon) arm.
#[cfg(target_feature = "neon")]
pub(crate) fn matmul_q6kx8_prepacked(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    packed: &[BlockQ6Kx8],
    dst: &mut [f32],
) {
    use super::neon::gemm_q6kx8_q8k;
    let nb = k / QK_K;
    let groups = n / Q4KX8_ROWS;
    // Q6 has only the 8-channel kernel. MR=2 (16 accumulators) is the default and
    // the N1-safe tile; MR=4 (32 acc) tends to spill the 32 NEON regs - the same
    // regression seen on Q4's nc8mr4 tile.
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

// Pre-packed Q6_K weight: n output channels (n % 8 == 0) as `BlockQ6Kx8` groups.
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

    // Owned copy from raw interleaved bytes (whole number of blocks).
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

    // Zero-copy view of `byte_len` bytes at `offset` in an mmap'd GGUF.
    pub fn from_mmap(
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        byte_len: usize,
        n: usize,
    ) -> crate::Result<Self> {
        let bs = std::mem::size_of::<BlockQ6Kx8>();
        // checked_add: a malformed offset near usize::MAX would otherwise wrap and
        // pass the bounds check, yielding a pointer outside the mapping.
        match offset.checked_add(byte_len) {
            Some(end) if end <= mmap.len() => {}
            _ => crate::bail!(
                "Q6Kx8 mmap region at offset {offset} (+{byte_len}) out of bounds for map len {}",
                mmap.len()
            ),
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

    // Dequantize back to row-major f32 [n, k]; mirrors `BlockQ6K::to_float`.
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
        use crate::quantized::k_quants::BlockQ6K;
        use crate::quantized::QuantizedType;

        let k = 512usize;
        let n = 24usize;
        let nb = k / QK_K;
        let mut st = 0x51c6_ba9d_2244_8821u64;

        let mut rhs_t = vec![BlockQ6K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            BlockQ6K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }

        // bake: repack -> raw bytes -> owned PackedQ6Kx8
        let packed = repack_q6k_weight(&rhs_t, n, nb);
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                packed.as_ptr() as *const u8,
                std::mem::size_of_val(packed.as_slice()),
            )
        };
        let pq = PackedQ6Kx8::from_bytes(raw, n);

        // matmul_t requires NEON; off-NEON it bails, so only assert it there.
        #[cfg(target_feature = "neon")]
        {
            use crate::quantized::k_quants::matmul;
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
        }

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

    // The lane=row 8x4 SDOT kernel (laneq-4 weights + q8_Kx4 4x4 activations) vs
    // the f32 ground truth: validates the laneq weight layout, the q8_Kx4 (4x4)
    // activation pack, the 6-bit scale unpack, and the lane-row tiling/fold. Runs on
    // any dotprod host (incl. M1), so this is a real check of the N1 kernel.
    #[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
    #[test]
    fn gemm_q4kx8_lanerow_matches_reference() {
        use crate::quantized::k_quants::BlockQ4K;
        use crate::quantized::neon::gemm_q4kx8_q8k_lanerow;

        let nb = 3usize;
        let k = nb * QK_K;
        let mut st = 0x9e37_79b9_7f4a_7c15u64;

        let mut wq: Vec<Vec<BlockQ4K>> = Vec::new();
        let mut wf: Vec<Vec<f32>> = Vec::new();
        for _ in 0..8 {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            let mut q = vec![BlockQ4K::zeros(); nb];
            BlockQ4K::from_float(&f, &mut q);
            let mut deq = vec![0f32; k];
            BlockQ4K::to_float(&q, &mut deq);
            wq.push(q);
            wf.push(deq);
        }
        let cols: [&[BlockQ4K]; 8] = std::array::from_fn(|c| wq[c].as_slice());
        let packed = repack_q4kx8_laneq4(&cols);

        let af: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..k).map(|_| lcg(&mut st)).collect())
            .collect();
        let arows: [&[f32]; 4] = std::array::from_fn(|r| af[r].as_slice());
        let q8 = quantize_mat_q8_k_4x4(&arows, nb);

        let mut dst = [0f32; 32];
        gemm_q4kx8_q8k_lanerow(&packed, &q8, &mut dst);

        // Reconstruct each row's q8 from the 4x4 interleave: element e of row r
        // lives at qs[j], j = (e/4)*16 + r*4 + (e%4).
        for r in 0..4 {
            for col in 0..8 {
                let mut sum = 0f64;
                for blk in 0..nb {
                    let qb = &q8[blk];
                    for e in 0..QK_K {
                        let j = (e / 4) * 16 + r * 4 + (e % 4);
                        let actd = qb.d[r] as f64 * qb.qs[j] as f64;
                        sum += wf[col][blk * QK_K + e] as f64 * actd;
                    }
                }
                let got = dst[r * 8 + col] as f64;
                let rel = (got - sum).abs() / sum.abs().max(1e-6);
                assert!(rel < 1e-2, "r{r} col{col}: got {got} ref {sum} rel {rel}");
            }
        }
    }

    // The lane=row prefill driver must match the baseline matmul end-to-end (full
    // dst, multiple 8-channel groups), for m a multiple of 4 and not (zero-padded
    // remainder). Validates tiling, scatter, and the parallel group partition.
    #[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
    #[test]
    fn matmul_q4kx8l_lanerow_matches_baseline() {
        use crate::quantized::k_quants::{matmul, BlockQ4K};

        let k = 512usize; // 2 super-blocks
        let n = 24usize; // 3 groups of 8
        let nb = k / QK_K;
        let mut st = 0xc0ff_ee00_1234_abcdu64;

        let mut rhs_t = vec![BlockQ4K::zeros(); n * nb];
        for r in 0..n {
            let f: Vec<f32> = (0..k).map(|_| lcg(&mut st)).collect();
            BlockQ4K::from_float(&f, &mut rhs_t[r * nb..(r + 1) * nb]);
        }
        let laneq = repack_q4k_weight_laneq4(&rhs_t, n, nb);

        for &m in &[4usize, 7usize, 16usize] {
            let lhs: Vec<f32> = (0..m * k).map(|_| lcg(&mut st)).collect();
            let mut dst_base = vec![0f32; m * n];
            let mut dst_lr = vec![0f32; m * n];
            matmul::<BlockQ4K>((m, k, n), &lhs, &rhs_t, &mut dst_base).unwrap();
            matmul_q4kx8l_lanerow((m, k, n), &lhs, &laneq, &mut dst_lr);
            let mut err = 0f64;
            let mut sig = 0f64;
            for i in 0..m * n {
                let d = dst_lr[i] as f64 - dst_base[i] as f64;
                err += d * d;
                sig += (dst_base[i] as f64) * (dst_base[i] as f64);
            }
            let rel = (err / sig.max(1e-12)).sqrt();
            assert!(rel < 2e-2, "m={m}: relative L2 error {rel} too high");
        }
    }
}

// Same 1152-byte layout as k_quants::BlockQ4Kx8 (only the qs interleave differs:
// bsi=4 vs 8); zerocopy derives let the per-QTensor `repacked_laneq` cache it as bytes.
#[cfg(target_feature = "dotprod")]
#[derive(
    Clone,
    Copy,
    zerocopy::FromBytes,
    zerocopy::IntoBytes,
    zerocopy::KnownLayout,
    zerocopy::Immutable,
)]
#[repr(C)]
pub struct BlockQ4Kx8L {
    pub(crate) d: [f16; 8],
    pub(crate) dmin: [f16; 8],
    pub(crate) scales: [u8; 96], // 6-bit scales+mins, repacked (8 cols)
    pub(crate) qs: [u8; 1024],   // 4-bit quants, 4-byte round-robin interleave
}
#[cfg(target_feature = "dotprod")]
const _: () = assert!(std::mem::size_of::<BlockQ4Kx8L>() == 1152);

#[cfg(target_feature = "dotprod")]
impl BlockQ4Kx8L {
    fn zeroed() -> Self {
        Self {
            d: [f16::ZERO; 8],
            dmin: [f16::ZERO; 8],
            scales: [0; 96],
            qs: [0; 1024],
        }
    }
}

// Port of llama's make_block_q4_Kx8 for one 8-column super-block; `bsi` is the qs
// interleave stride (4 for the lane=row kernel). Scales/mins are bsi-independent.
#[cfg(target_feature = "dotprod")]
fn make_q4kx8_laneq_bsi(cols: &[&[BlockQ4K]; 8], i: usize, bsi: usize) -> BlockQ4Kx8L {
    let mut out = BlockQ4Kx8L::zeroed();
    for c in 0..8 {
        out.d[c] = cols[c][i].d;
        out.dmin[c] = cols[c][i].dmin;
    }
    // qs: take `bsi` bytes at a time, round-robin across the 8 columns.
    // end = QK_K * 4 / bsi; src col = t%8, src off = (t/8)*bsi, dst = t*bsi.
    let end = QK_K * 4 / bsi;
    for t in 0..end {
        let src_id = t % 8;
        let src_off = (t / 8) * bsi;
        let dst_off = t * bsi;
        out.qs[dst_off..dst_off + bsi].copy_from_slice(&cols[src_id][i].qs[src_off..src_off + bsi]);
    }
    // scales: repack the 6-bit packed scales/mins of all 8 columns into 96 bytes.
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];
    for ii in 0..4 {
        for j in 0..8 {
            s[j] = cols[j][i].scales[ii] & 63;
            m[j] = cols[j][i].scales[ii + 4] & 63;
        }
        let b = ii * 12;
        out.scales[b] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[b + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[b + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[b + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[b + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[b + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[b + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[b + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[b + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[b + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[b + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[b + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    for ii in 0..4 {
        for j in 0..8 {
            s[j] = ((cols[j][i].scales[ii] & 192) >> 2) | (cols[j][i].scales[ii + 8] & 15);
            m[j] =
                ((cols[j][i].scales[ii + 4] & 192) >> 2) | ((cols[j][i].scales[ii + 8] & 240) >> 4);
        }
        let b = ii * 12 + 48;
        out.scales[b] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[b + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[b + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[b + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[b + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[b + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[b + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[b + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[b + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[b + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[b + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[b + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    out
}

// Repack 8 weight columns into `nb` interleave-4 laneq blocks (lane=row layout).
#[cfg(target_feature = "dotprod")]
pub(crate) fn repack_q4kx8_laneq4(cols: &[&[BlockQ4K]; 8]) -> Vec<BlockQ4Kx8L> {
    let nb = cols[0].len();
    (0..nb).map(|i| make_q4kx8_laneq_bsi(cols, i, 4)).collect()
}

// Four Q8_K activation rows interleaved for the lane=row GEMM (llama block_q8_Kx4).
#[cfg(target_feature = "dotprod")]
#[repr(C)]
#[derive(Clone)]
pub(crate) struct BlockQ8Kx4 {
    pub(crate) d: [f32; 4],
    pub(crate) qs: [i8; QK_K * 4],
    pub(crate) bsums: [i16; QK_K / 4],
}

#[cfg(target_feature = "dotprod")]
impl BlockQ8Kx4 {
    fn zeroed() -> Self {
        Self {
            d: [0.0; 4],
            qs: [0; QK_K * 4],
            bsums: [0; QK_K / 4],
        }
    }
}

// Quantize+interleave 4 activation rows into `out` (len nb), interleave-4 so the
// SDOT lane selects the row. Per-super-block independent. Port of llama's
// ggml_quantize_mat_q8_K_4x4_generic.
#[cfg(target_feature = "dotprod")]
pub(crate) fn quantize_mat_q8_k_4x4_into(rows: &[&[f32]; 4], out: &mut [BlockQ8Kx4]) {
    const BSI: usize = 4; // blck_size_interleave
    for (i, blk) in out.iter_mut().enumerate() {
        *blk = BlockQ8Kx4::zeroed();
        let mut srcv = [[0f32; QK_K]; 4];
        let mut iscale = [0f32; 4];
        for (row, &r) in rows.iter().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;
            for j in 0..QK_K {
                let v = r[i * QK_K + j];
                srcv[row][j] = v;
                if amax < v.abs() {
                    amax = v.abs();
                    max = v;
                }
            }
            iscale[row] = if amax != 0.0 { -127.0 / max } else { 0.0 };
            blk.d[row] = if amax != 0.0 { 1.0 / iscale[row] } else { 0.0 };
        }
        // Quants interleaved in 4-byte runs across the 4 rows; bsums grouped
        // 4-at-a-time per source super-block (the kernel's bias term).
        for j in 0..QK_K * 4 {
            let mut src_offset = (j / (4 * BSI)) * BSI;
            let src_id = (j % (4 * BSI)) / BSI;
            src_offset += j % BSI;
            let index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);
            let x0 = srcv[src_id][src_offset] * iscale[src_id];
            let q = x0.round() as i8;
            blk.qs[j] = q;
            blk.bsums[index] += q as i16;
        }
    }
}

// Vec-returning wrapper over `quantize_mat_q8_k_4x4_into` (test only).
#[cfg(all(test, target_feature = "dotprod"))]
pub(crate) fn quantize_mat_q8_k_4x4(rows: &[&[f32]; 4], nb: usize) -> Vec<BlockQ8Kx4> {
    let mut out = vec![BlockQ8Kx4::zeroed(); nb];
    quantize_mat_q8_k_4x4_into(rows, &mut out);
    out
}

// Repack a full Q4_K weight into interleave-4 laneq groups for all n/8 channels.
#[cfg(target_feature = "dotprod")]
pub(crate) fn repack_q4k_weight_laneq4(rows: &[BlockQ4K], n: usize, nb: usize) -> Vec<BlockQ4Kx8L> {
    debug_assert_eq!(n % Q4KX8_ROWS, 0, "repack_q4k_weight_laneq4: n {n} not /8");
    debug_assert_eq!(
        rows.len(),
        n * nb,
        "repack_q4k_weight_laneq4: rows len mismatch"
    );
    let mut packed = Vec::with_capacity((n / 8) * nb);
    for g in 0..n / 8 {
        let grp: [&[BlockQ4K]; Q4KX8_ROWS] =
            std::array::from_fn(|r| &rows[(g * 8 + r) * nb..(g * 8 + r + 1) * nb]);
        packed.extend(repack_q4kx8_laneq4(&grp));
    }
    packed
}

// Whether the lane=row Q4_K prefill path is enabled; the pack is cached per-QTensor.
#[cfg(target_feature = "dotprod")]
pub(crate) fn prefill_lanerow_enabled() -> bool {
    true
}

// Lane=row prefill GEMM: dst(m,n) = lhs(m,k) x W^T over interleave-4 laneq weights,
// 4-row tiles (last zero-padded), parallelized over channel groups on the pool.
#[cfg(all(target_feature = "neon", target_feature = "dotprod"))]
pub(crate) fn matmul_q4kx8l_lanerow(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    packed: &[BlockQ4Kx8L],
    dst: &mut [f32],
) {
    use super::neon::gemm_q4kx8_q8k_lanerow;
    let nb = k / QK_K;
    let groups = n / Q4KX8_ROWS;
    let row_tiles = m.div_ceil(4);

    // Quantize all m activation rows to Q8 (4x4 interleave) into 4-row tiles, the
    // last zero-padded. Parallelized over row-tiles on the barrier pool (each tile
    // is independent), so it does not serialize multi-thread prefill; identical
    // bytes to the serial path.
    let zeros = vec![0f32; k];
    let mut q8: Vec<BlockQ8Kx4> = vec![BlockQ8Kx4::zeroed(); row_tiles * nb];
    let tile_rows = |rt: usize| -> [&[f32]; 4] {
        std::array::from_fn(|a| {
            let r = rt * 4 + a;
            if r < m {
                &lhs[r * k..(r + 1) * k]
            } else {
                zeros.as_slice()
            }
        })
    };
    crate::utils::par_chunks_mut(&mut q8, nb, |rt, chunk| {
        quantize_mat_q8_k_4x4_into(&tile_rows(rt), chunk);
    });

    struct DstPtr(*mut f32);
    unsafe impl Sync for DstPtr {}
    let dptr = DstPtr(dst.as_mut_ptr());

    let process = |g: usize| {
        let w = &packed[g * nb..(g + 1) * nb];
        let p = &dptr;
        for rt in 0..row_tiles {
            let q8t = &q8[rt * nb..(rt + 1) * nb];
            let mut t = [0f32; 32];
            gemm_q4kx8_q8k_lanerow(w, q8t, &mut t);
            for a in 0..4 {
                let r = rt * 4 + a;
                if r >= m {
                    break;
                }
                for c in 0..Q4KX8_ROWS {
                    unsafe { *p.0.add(r * n + g * 8 + c) = t[a * 8 + c] };
                }
            }
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
}
