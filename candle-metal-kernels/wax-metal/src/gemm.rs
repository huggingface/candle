//! GEMM/GEMV backend routing.
//!
//! A compiler decision, not a DSL one: MPP's `matmul2d` owns the whole K loop and
//! threadgroup, so it replaces a loop nest rather than slotting into one.
//!
//! Thresholds here are MEASURED on the shape suite in `wax-metal benches`. Comment numbers are the
//! evidence; do not retune from first principles.
//!
//! [`route`] reports the clause that decided, so "this shape must hit MPP" is testable.
//! Split-K pays only when the reduce (`S*M*N`) is small against the matmul (`M*N*K`);
//! at 1024^3 with S=4 it is ~30% of runtime, which is why deep K gates it.

/// The problem to be lowered, independent of backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Bytes per operand element (2 for f16/bf16, 4 for f32).
    pub elem_bytes: usize,
    /// Row stride of the DESTINATION in elements. `n` means packed.
    ///
    /// Load-bearing for split-K, which is why it is here rather than left implicit: the
    /// reduce pass treats the `[S,M,N]` partial buffer as S flat planes of `M*N`
    /// contiguous elements, which is only valid when the output is packed. A non-packed
    /// destination through split-K would silently produce wrong results, so [`route`]
    /// refuses to choose split-K unless `ldc == n`.
    pub ldc: usize,
}

impl GemmSpec {
    /// A packed (`ldc == n`) problem.
    pub fn new(m: usize, n: usize, k: usize, elem_bytes: usize) -> Self {
        Self {
            m,
            n,
            k,
            elem_bytes,
            ldc: n,
        }
    }

    /// Same, with an explicit destination row stride.
    pub fn with_ldc(m: usize, n: usize, k: usize, elem_bytes: usize, ldc: usize) -> Self {
        Self {
            m,
            n,
            k,
            elem_bytes,
            ldc,
        }
    }

    /// Whether the destination is packed, which split-K's reduce requires.
    pub fn is_packed_out(&self) -> bool {
        self.ldc == self.n
    }

    /// A shape the SGMMA path would have to pad or clamp. MPP clamps its own slices,
    /// so raggedness is a reason to PREFER it.
    pub fn is_ragged(&self) -> bool {
        !self.m.is_multiple_of(8) || !self.n.is_multiple_of(8) || !self.k.is_multiple_of(8)
    }
}

/// The chosen backend and its tiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The SGMMA (`simdgroup_matrix_8x8`) tile loop. Tile choice is the SGMMA path's
    /// own concern, so none is carried here.
    Sgmma,
    /// MPP single pass, one `bm x bn` output tile per threadgroup.
    Mpp { bm: i32, bn: i32 },
    /// MPP split-K: `splits` threadgroups per output tile writing `[S,M,N]` f32
    /// partials, which a reduce pass then sums. Two dispatches, not one.
    ///
    /// NOT SELECTED BY [`route`] for M>1 - see the `split_k` note on this module. It IS
    /// selected for the M==1 vecmat regime, with its own measured tiles.
    MppSplitK { bm: i32, bn: i32, splits: i32 },
    /// `out[1,N] = x[1,K] @ y[K,N]`, one threadgroup of `sg` SIMD groups per 32-column
    /// chunk, reducing K across the groups in threadgroup memory. One dispatch.
    Vecmat { sg: i32 },
    /// `out[M,1] = a[M,K] @ v[K,1]`, one threadgroup per output row, `bk` lanes each.
    Gemv { bk: i32 },
}

impl Backend {
    pub fn is_mpp(&self) -> bool {
        matches!(self, Backend::Mpp { .. } | Backend::MppSplitK { .. })
    }

    /// Elements the DESTINATION buffer must hold for an `m x n` problem.
    ///
    /// Everything writes `m * n` except [`Backend::Gemv`]: a `[1, W]` store with `W < 32`
    /// still writes 32 lanes (see `tests/narrow_store.rs`), so that kernel broadcasts each
    /// row's dot product across `bk` lanes and its destination is `[M, bk]` with the
    /// answer in column 0. Allocating `m * n` for it overruns the buffer.
    ///
    /// This is a method rather than a comment because a caller that gets it wrong gets
    /// silent corruption - which is exactly what happened the first time the GEMV arm was
    /// driven through a plan instead of its own bespoke launcher.
    pub fn dst_elems(&self, m: usize, n: usize) -> usize {
        match self {
            Backend::Gemv { bk } => m * *bk as usize,
            _ => m * n,
        }
    }

    /// Row pitch of the destination in elements, for reading a result back.
    pub fn dst_pitch(&self, n: usize) -> usize {
        match self {
            Backend::Gemv { bk } => *bk as usize,
            _ => n,
        }
    }
}

/// The clause that decided. Named so tests can assert on the reasoning rather than
/// only on the outcome - two shapes can both land on SGMMA for quite different reasons,
/// and a retune that changes which clause fires is worth noticing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// `min(M,N) < 192`. MPP's tile computes `bn` columns whatever N is, so thin-N and
    /// GEMV starve it: 4096x8 is 2.5x slower on MPP, 1x4096 is 3x.
    ThinN,
    /// A threadgroup streams `(BM+BN)*K` elements and this exceeds the budget. The one
    /// dtype-dependent clause: fp32 K=11008 is 2.8 MB and MPP collapses to 8 TF
    /// (46 ms vs SGMMA's 31), while 2.1 MB at fp32 K=8192 is still a 10% MPP win.
    WorkingSetTooLarge,
    /// Aligned and below the work floor, where MPP's fixed setup (descriptor allocas,
    /// slice calls, cooperative-tensor init) does not amortise - 1.8x slower at 128^3.
    NotEnoughWork,
    /// Ragged, so MPP wins at ANY size: it clamps its own slices while SGMMA pays for
    /// pad slabs and a clamped core.
    Ragged,
    /// Enough total work to amortise MPP's setup; 3-8% faster from 1024^3 up.
    EnoughWork,
    /// fp32 with a large per-threadgroup stream. MPP does not collapse as at
    /// [`WorkingSetTooLarge`](Reason::WorkingSetTooLarge), it just loses to SGMMA, and only
    /// at fp32: 4096x11008x4096 is MPP 11.00 TF vs SGMMA 13.50; 2048x2048x8192 8.63 vs
    /// 12.81. Not the tile or thread count - SGMMA at MPP's own 32x32/1-group geometry
    /// still reads 13.47. Non-ragged only ([`Ragged`](Reason::Ragged) measured the
    /// opposite). Yields split-K where the shape allows (2048x2048x8192, relative throughput:
    /// MPP 0.785, SGMMA 0.842, split-K 1.115), SGMMA otherwise.
    Fp32WorkingSet,
    /// Working set too large for one pass, so K is split rather than falling back to
    /// SGMMA. 4096x4096x11008 fp32: MPP 49.07 ms, SGMMA 37.14, split-K 27.31
    /// (S=2 38.14, S=8 29.00).
    SplitWorkingSet,
    /// `M == 1`, the transposed GEMV `x[1,K] @ y[K,N]`. Three MEASURED regimes (ms, bf16):
    /// N <= 2048 takes the single-dispatch threadgroup reduction (1x1024x1024: 0.0058 vs
    /// 0.0065 MPP split-K); mid-N 2-byte takes MPP split-K at 8x64 (0.0419 vs 0.0451);
    /// wide N takes no MPP at any dtype (see [`MPP_MAX_BN`] and the wide clause in
    /// [`route`]).
    Vecmat,
    /// `N == 1`, the plain GEMV `a[M,K] @ v[K,1]`. One threadgroup per output row; MPP
    /// starves here for the [`ThinN`](Reason::ThinN) reason, squared.
    Gemv,
}

/// A routing decision, with the clause that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Routing {
    pub backend: Backend,
    pub reason: Reason,
}

/// SGMMA's K-block. The SGMMA arm walks K in blocks of this many elements.
pub const SGMMA_BK: usize = 8;

/// The SGMMA arm's `(BM, BN)` for `spec`.
///
/// Trades REUSE (`BM*BN/(BM+BN)` MACs per loaded element: 8 at 16x16, 16 at 32x32) against
/// GRID (`ceil(M/BM)*ceil(N/BN)` threadgroups). The limit is arithmetic intensity, not
/// occupancy - at 128x2048 a 16x16 tile still fills the GPU but loads 4x the data per MAC.
/// Preferring 32x32 in the `tiles32 <= 256` bucket took the prefill scorecard from 13
/// losses to 7 (geomean 1.0253 -> 1.0408).
/// `(BM, BN, BK)` for the aligned SGMMA kernel - [`sgmma_tile_aligned`] plus the K-block.
/// A tile the harnesses pick but the library does not know about is how an instrument
/// stops measuring the shipped implementation, so the narrow-N clause lives here.
pub fn sgmma_tile(spec: &GemmSpec) -> (usize, usize, usize) {
    // Narrow N wants a DEEP K tile with a short BM: with BK=32 each threadgroup reads A as
    // contiguous 32-column strips rather than 8, which is what these A-bandwidth-bound
    // shapes are limited by.
    if spec.m.min(spec.n) <= 64 && spec.k.is_multiple_of(32) {
        let bn = if spec.n <= 32 {
            spec.n.clamp(8, 16)
        } else {
            32
        };
        if spec.m.is_multiple_of(8) && spec.n.is_multiple_of(bn) {
            return (8, bn, 32);
        }
    }
    let (bm, bn) = sgmma_tile_aligned(spec);
    (bm, bn, SGMMA_BK)
}

/// Tile for the ALIGNED SGMMA kernel, which needs the tile to DIVIDE the problem, and
/// takes the first preference that does (8x8, the SGMMA fragment, always divides an
/// aligned shape).
pub fn sgmma_tile_aligned(spec: &GemmSpec) -> (usize, usize) {
    let thin = spec.m.min(spec.n);
    let tiles32 = spec.m.div_ceil(32) * spec.n.div_ceil(32);
    let prefs: &[(usize, usize)] = if thin <= 32 {
        &[(16, 16), (8, 16), (8, 8)]
    } else if thin <= 64 {
        &[(8, 32), (16, 16), (8, 8)]
    } else if tiles32 <= SMALL_TILE_MAX_TILES {
        // THE MEASURED CHANGE - was `[(16, 16), (16, 32), (32, 32), (8, 8)]`.
        &[(32, 32), (16, 32), (16, 16), (8, 8)]
    } else if tiles32 <= 8192 || spec.k < 256 {
        // ALSO MEASURED - was `[(16, 32), (32, 32), (16, 16), (8, 8)]`. Same mechanism as
        // the bucket above: 32x64 is 21 MACs per loaded element against 16x32's 11.
        // 32x64 beat 16x32 on every sample over three runs at 128x8192x2048 (fp32
        // 0.3264/0.3414/0.3328 -> 0.3231/0.3226/0.3151 ms, fp16 and bf16 likewise).
        // `thin >= 192` only reaches SGMMA via `Reason::Fp32WorkingSet`, and those big
        // deep-K cells want the squarer tile: at 2048x2048x8192 and 4096x11008x4096 a
        // 32x64 tile on one SIMD group reads 7.32/35.84 ms against 32x32's 7.32/34.32.
        if thin >= 192 {
            &[(32, 32), (32, 64), (16, 32), (16, 16), (8, 8)]
        } else if spec.k >= THIN_DEEP_K {
            // A THIN shape with a DEEP K wants the SMALLER tile. The comment above used to
            // claim no cell reached this bucket; that was read off labels a harness
            // produced with its own private tile rule, which never called this function.
            // Unified onto it, 15 cells landed here and 32x64 lost every one.
            &[(16, 32), (32, 64), (32, 32), (16, 16), (8, 8)]
        } else {
            &[(32, 64), (16, 32), (32, 32), (16, 16), (8, 8)]
        }
    } else {
        if thin >= 192 {
            &[(32, 32), (32, 64), (16, 32), (8, 8)]
        } else {
            &[(32, 64), (32, 32), (16, 32), (8, 8)]
        }
    };
    prefs
        .iter()
        .copied()
        .find(|&(bm, bn)| spec.m.is_multiple_of(bm) && spec.n.is_multiple_of(bn))
        .unwrap_or((8, 8))
}

/// Tile for the CLAMPED SGMMA kernel, which handles a ragged shape by clamping tile
/// origins back inside the matrix, so the tile need only FIT, shrunk to the 8x8 fragment.
///
/// The selector for the `nt` path that external callers invoke directly, bypassing
/// [`route`] - so it, not [`sgmma_tile_aligned`], is what a real prefill runs. Fitted on
/// that kernel; the aligned `nn` harness measures a different kernel and does not transfer.
///
/// Six tiles x 48 cells, stable subset, relative geomean f32/f16/bf16: 32x32 wins all three
/// (1.098/0.964/1.033), 32x64 is worst (0.769/0.850/0.773). NO DTYPE CARVE-OUT - one was
/// fitted to a cell that read 0.976 and 0.621 on two runs of the same binary.
pub fn sgmma_tile_clamped(spec: &GemmSpec) -> (usize, usize) {
    let thin = spec.m.min(spec.n);
    let (bm, bn) = if thin <= 32 {
        // Thin shapes are NOT covered by the sweep below (it starts at n=512) - these two
        // buckets are the original heuristic and stay until something measures them.
        (16, 16)
    } else if thin <= 64 {
        (8, 32)
    } else {
        (32, 32)
    };
    (bm.min(spec.m & !7).max(8), bn.min(spec.n & !7).max(8))
}

/// MPP's default tile. One SIMD group per threadgroup either way - 4 groups is 6x
/// slower - but the tile SHAPE depends on how many threadgroups the grid gets.
const MPP_TILE: (i32, i32) = (32, 32);
/// The low-occupancy tile. Half the rows per threadgroup, so twice the grid.
const MPP_TILE_SMALL: (i32, i32) = (16, 32);
/// Tile for genuinely small grids - 16x16 halves BN as well, doubling the grid again.
const MPP_TILE_TINY: (i32, i32) = (16, 16);

/// Tile counts at or below which the small tile wins, and the squareness floor that
/// keeps thin shapes out of the second clause. Both MEASURED - see [`mpp_tile`].
const SMALL_TILE_MAX_TILES: usize = 256;
const SQUARISH_MAX_TILES: usize = 1024;
const SQUARISH_MIN_TILES_PER_DIM: usize = 16;
/// Largest aligned fp32 square allowed past the thin-N floor - see the clause in [`route`].
const SMALL_SQUARE_MAX: usize = 256;
/// K at or above which a THIN shape (`min(M,N) < 192`) prefers the smaller SGMMA tile.
///
/// The cell that wants 32x64 in that bucket is K=2048; every cell that wants 16x32 is
/// K=4096. MEASURED - see the clause in [`sgmma_tile_aligned`].
const THIN_DEEP_K: usize = 4096;
/// Threadgroup-count floor for taking the full 32x32 MPP tile on a deep-K fp32 shape.
/// Below this the bigger tile's reuse does not pay for the grid it gives up.
const MPP_FULL_TILE_MIN_TILES: usize = 128;

/// MPP's tile for `spec`.
///
/// The grid is `ceil(M/BM) x ceil(N/BN)`; below a few hundred threadgroups a 16-row tile
/// (twice the grid, same work) wins. 16x32 against 32x32, two sweeps: 257^3 1.34-1.38x,
/// 511^3 1.02-1.15x, 1024^3 1.037-1.041x, 4096^3 0.93-0.98x (reuse wins again).
///
/// `SQUARISH_*` is narrow by measurement: THIN shapes at the same tile count are
/// neutral-to-worse (192x4096x4096 0.986-1.012), and it is 2-byte only - a 16-row tile
/// halves B-reuse, so at fp32 bandwidth binds first and ragged 1023^3 regressed 0.815x.
/// Both clauses are fit to few points; re-measure before widening either.
pub fn mpp_tile(spec: &GemmSpec) -> (i32, i32) {
    let (bm, bn) = MPP_TILE;
    let tm = spec.m.div_ceil(bm as usize);
    let tn = spec.n.div_ceil(bn as usize);
    let tiles = tm * tn;
    // ALIGNMENT, not size, is the discriminator for the small-square tile: 256^3 and
    // 257^3 have near-identical grids and opposite outcomes (16x16 regressed every ragged
    // neighbour: 257^3 0.95x, 511^3 0.76x). Deep-K fp32 wants the full tile's reuse even
    // on a small grid - 512x512x2048 fp32, relative: 16x16 0.824, 16x32 1.021, 32x32 1.081.
    // `tiles >= 128` keeps the shallow-K squares (which want the small tile) out.
    if spec.elem_bytes >= 4
        && !spec.is_ragged()
        && spec.k >= 1024
        && tiles >= MPP_FULL_TILE_MIN_TILES
    {
        return MPP_TILE;
    }
    if spec.elem_bytes >= 4 && !spec.is_ragged() && tiles <= SMALL_TILE_MAX_TILES {
        return MPP_TILE_TINY;
    }
    let squarish = tm.min(tn) >= SQUARISH_MIN_TILES_PER_DIM && spec.elem_bytes <= 2;
    if tiles <= SMALL_TILE_MAX_TILES || (tiles <= SQUARISH_MAX_TILES && squarish) {
        MPP_TILE_SMALL
    } else {
        MPP_TILE
    }
}

/// Per-threadgroup stream budget in bytes, above which MPP collapses.
const WORKING_SET_MAX: usize = 2_500_000;
/// fp32 stream budget above which our own SGMMA arm already beats MPP - far BELOW the
/// outright collapse at [`WORKING_SET_MAX`]. See [`Reason::Fp32WorkingSet`].
///
/// 1 MiB is `(32+32) * 4096 * 4`, i.e. fp32 K=4096 at the 32x32 tile, and it sits with 2x
/// margin on both sides of the measured crossover: K=2048 is 0.5 MB and MPP wins there,
/// K=4096 is exactly 1 MB and SGMMA wins.
const FP32_MPP_WORKING_SET_MAX: usize = 1 << 20;
/// Work floor for MPP at all.
const MPP_WORK_FLOOR: usize = 1 << 30;
/// K splits used when the working set forces a split. Measured optimum; 2 and 8 are both
/// worse (38.14 and 29.00 against 27.31 at 4096x4096x11008 fp32).
const SPLITS: i32 = 4;

/// The M==1 (vecmat) thresholds and tiles. All MEASURED - see [`Reason::Vecmat`].
/// One 32-column chunk is the unit of work, so below that there is nothing to launch.
const VECMAT_MIN_N: usize = 32;
/// At or below this N the two-pass split-K is launch-bound and the single dispatch wins.
const VECMAT_SMALL_MAX_N: usize = 2048;
/// Above this N the wide MPP tile beats the short one.
const VECMAT_WIDE_MIN_N: usize = 8192;
const VECMAT_MPP_TILE_MID: (i32, i32) = (8, 64);
/// Widest MPP tile our emitter produces correct results for at one SIMD group.
///
/// A `bn` of 256 silently returns garbage: 1x16384x4096 f16 reads a max relative error of
/// 2.95e0 against a CPU reference at (16, 256) and (8, 256), and 4.88e-4 at (16, 128),
/// (16, 64) and (8, 64). So it is the tile WIDTH, not `bm`, not K, and not the split
/// count. We emit one SIMD group per threadgroup (extra groups duplicate the work rather
/// than dividing it), and a 16x256 cooperative-tensor tile exceeds what one group holds.
/// A wider tile needs the emitter to PARTITION a tile across SIMD groups first.
///
/// It went unnoticed for months because the only harness covering this route timed without
/// checking a value; `tests/gemm_ragged.rs` now checks every routed regime against a CPU
/// reference.
pub const MPP_MAX_BN: i32 = 128;
/// Splits for the M==1 MPP path. Its grid is only `ceil(N/BN)` tiles, so unlike the
/// M>1 split-K (`SPLITS`) this exists to manufacture threadgroups, and wants far more.
const VECMAT_SPLITS: i32 = 16;
/// GEMV's lanes per row. `K` must be a multiple of it.
const GEMV_BK: i32 = 32;

/// Choose a backend for `spec`.
///
/// Validated offline against a full A/B of both backends over the suite: this rule
/// matched a perfect oracle to within 0.02 geomean.
pub fn route(spec: &GemmSpec) -> Routing {
    let (bm, bn) = mpp_tile(spec);
    let sgmma = |reason| Routing {
        backend: Backend::Sgmma,
        reason,
    };

    // The two GEMV regimes come FIRST, because both would otherwise be swallowed by the
    // `ThinN` clause below and sent to a tiled GEMM kernel that is entirely wrong for a
    // single row or column. See [`Reason::Vecmat`] for the measurements behind the
    // thresholds here.
    if spec.m == 1 {
        // Below a chunk's worth of columns there is not even one full threadgroup of
        // work; nothing in the GEMV family applies and the caller gets the tiled arm.
        if spec.n >= VECMAT_MIN_N {
            let small = spec.n <= VECMAT_SMALL_MAX_N;
            let wide = spec.n > VECMAT_WIDE_MIN_N;
            // WIDE N TAKES NO MPP. At M==1 a `bm`-row tile computes `bm` rows and keeps
            // one, so MPP pays 8-16x for discarded output. `tests/vecmat_tune.rs` swept 24
            // (bm, bn, splits) combinations against the threadgroup reduction with a
            // correctness gate; at 1x32000x4096 the reduction won every dtype (fp32 1.0376
            // ms vs best MPP 1.1849, bf16 0.5829 vs 0.5930, f16 0.5934 vs 0.5929).
            if !small && !wide && spec.elem_bytes == 2 {
                let (bm, bn) = VECMAT_MPP_TILE_MID;
                // The reduce reads the partials as flat planes, so the same packed-output
                // gate as every other split-K routing applies.
                if spec.n.is_multiple_of(bn as usize) && spec.is_packed_out() {
                    return Routing {
                        backend: Backend::MppSplitK {
                            bm,
                            bn,
                            splits: VECMAT_SPLITS,
                        },
                        reason: Reason::Vecmat,
                    };
                }
            }
            // 32 groups where the problem is small enough to be launch-bound; at wide N
            // the 2-byte dtypes want 16 and fp32 wants 8 (measured above).
            let sg = if small {
                32
            } else if wide && spec.elem_bytes == 2 {
                16
            } else {
                8
            };
            return Routing {
                backend: Backend::Vecmat { sg },
                reason: Reason::Vecmat,
            };
        }
    } else if spec.n == 1 {
        return Routing {
            backend: Backend::Gemv { bk: GEMV_BK },
            reason: Reason::Gemv,
        };
    }
    // MPP's tile computes `bn` columns whatever N is, so thin-N and GEMV starve it
    // (4096x8 is 2.5x slower, 1x4096 is 3x). Relaxing this floor to 32 was tried and
    // reverted: it won its targeted shapes but cost more elsewhere, netting 86 wins -> 79
    // and 2 losses -> 5. Do not re-land without a FULL sweep, not just the target cells.
    // Small aligned fp32 squares bypass the thin-N floor: [`mpp_tile`] measured 128^3 at
    // 1.23x and 256^3 at 1.24x on 16x16, but `ThinN` fired first so 128^3 could never
    // reach it. Kept as narrow as that measurement - square, aligned, fp32, <= 256 -
    // because the ragged neighbours regress (257^3 0.95x, 511^3 0.76x).
    let small_square =
        spec.m == spec.n && spec.m <= SMALL_SQUARE_MAX && spec.elem_bytes >= 4 && !spec.is_ragged();
    if spec.m.min(spec.n) < 192 && !small_square {
        return sgmma(Reason::ThinN);
    }
    // The working set is computed with the tile the router itself picks, so this stays
    // self-consistent if MPP_TILE ever changes.
    let ragged = spec.is_ragged();
    let work = spec.m * spec.n * spec.k;
    let working_set = (bm as usize + bn as usize) * spec.k * spec.elem_bytes;
    if working_set > WORKING_SET_MAX {
        // Too much stream for one pass. K is necessarily huge here, so the reduce
        // (S*M*N) is cheap against the matmul (M*N*K). `is_packed_out` is a CORRECTNESS
        // gate: the reduce treats partials as S flat planes of M*N, so a strided
        // destination would be summed wrong.
        if work >= MPP_WORK_FLOOR && spec.is_packed_out() {
            return Routing {
                backend: Backend::MppSplitK {
                    bm,
                    bn,
                    splits: SPLITS,
                },
                reason: Reason::SplitWorkingSet,
            };
        }
        return sgmma(Reason::WorkingSetTooLarge);
    }
    // Below the collapse point but still a large fp32 stream: MPP loses to SGMMA. Placed
    // AFTER the clause above on purpose - above `WORKING_SET_MAX` split-K is better than
    // both (27.31 ms vs SGMMA's 37.14 at 4096x4096x11008 fp32), so this must not preempt
    // it. See [`Reason::Fp32WorkingSet`] for the crossover measurements.
    if !ragged && spec.elem_bytes >= 4 && working_set >= FP32_MPP_WORKING_SET_MAX {
        // Prefer SPLITTING K. The stream per threadgroup is what hurts, and splitting
        // divides it while the reduce (S*M*N) stays cheap against a matmul of M*N*K at a
        // K this deep - the same argument as `Reason::SplitWorkingSet`, just reached
        // sooner. `is_packed_out` is the correctness gate (see there); below the work
        // floor the two dispatches do not amortise, and SGMMA is the fallback.
        if work >= MPP_WORK_FLOOR && spec.is_packed_out() {
            return Routing {
                backend: Backend::MppSplitK {
                    bm,
                    bn,
                    splits: SPLITS,
                },
                reason: Reason::Fp32WorkingSet,
            };
        }
        return sgmma(Reason::Fp32WorkingSet);
    }
    // The work floor exists because MPP's fixed setup does not amortise on small
    // problems - but that was measured with a 32x32 tile. At fp32 with a 16x16 tile MPP
    // wins small squares outright: 128^3 1.165x and 256^3 1.171x (worst-case) against
    // the SGMMA routing. At 2 bytes the same shapes are ~0.89-1.00x, so the floor stays
    // there; fp32 is where the architectural edge is (13.97 TF peak), at every size.
    if !ragged && work < MPP_WORK_FLOOR && spec.elem_bytes < 4 {
        return sgmma(Reason::NotEnoughWork);
    }

    // Single-pass MPP. Split-K is deliberately NOT considered here; see the module note.
    Routing {
        backend: Backend::Mpp { bm, bn },
        reason: if ragged {
            Reason::Ragged
        } else {
            Reason::EnoughWork
        },
    }
}

/// MPP's tile for callers that force MPP where [`route`] would not choose it.
pub fn default_mpp_tile_for(spec: &GemmSpec) -> (i32, i32) {
    mpp_tile(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(m: usize, n: usize, k: usize, eb: usize) -> Routing {
        route(&GemmSpec::new(m, n, k, eb))
    }

    /// Thin-N stays on SGMMA. A relaxation of this floor from 192 to 32 (with an 8x32
    /// tile) was tried and reverted - see the note in [`route`].
    ///
    /// M==1 and N==1 are NOT in this list any more: they used to land here because the
    /// library had no GEMV regime and the bench routed them outside it. They now have
    /// their own backends, pinned by the tests below.
    #[test]
    fn thin_n_stays_on_sgmma() {
        for (m, n) in [(4096, 8), (4096, 16), (4096, 32), (4096, 128)] {
            let got = r(m, n, 4096, 2);
            assert_eq!(got.backend, Backend::Sgmma, "{m}x{n} must not use MPP");
            assert_eq!(got.reason, Reason::ThinN);
        }
    }

    /// The three M==1 regimes, at the thresholds [`Reason::Vecmat`] records. A tiled GEMM
    /// arm must never appear here - that was the bug this routing exists to make
    /// impossible, since `min(M,N) < 192` would otherwise send a single row to SGMMA.
    #[test]
    fn vecmat_picks_its_three_regimes() {
        // N small: single dispatch, 32 SIMD groups, whatever the dtype.
        for eb in [2, 4] {
            assert_eq!(r(1, 2048, 4096, eb).backend, Backend::Vecmat { sg: 32 });
        }
        // N mid: MPP split-K at 8x64 for 2-byte, but fp32 keeps the 8-group reduction.
        assert_eq!(
            r(1, 4096, 4096, 2).backend,
            Backend::MppSplitK {
                bm: 8,
                bn: 64,
                splits: 16
            }
        );
        assert_eq!(r(1, 4096, 4096, 4).backend, Backend::Vecmat { sg: 8 });
        // N wide: NO MPP at any dtype - the threadgroup reduction beat all 24 swept
        // (bm, bn, splits) combinations. 2-byte wants 16 groups, fp32 wants 8.
        assert_eq!(r(1, 32000, 4096, 2).backend, Backend::Vecmat { sg: 16 });
        assert_eq!(r(1, 32000, 4096, 4).backend, Backend::Vecmat { sg: 8 });
        for got in [r(1, 2048, 4096, 2), r(1, 32000, 4096, 2)] {
            assert_eq!(got.reason, Reason::Vecmat);
        }
    }

    /// Split-K's packed-output gate applies to the vecmat MPP route too: the reduce reads
    /// the partials as flat planes, so a strided destination must fall back rather than
    /// silently sum the wrong elements. Only the MID regime can take MPP now, so that is
    /// where this has to be checked.
    #[test]
    fn vecmat_split_k_refuses_a_strided_destination() {
        let packed = GemmSpec::new(1, 4096, 4096, 2);
        assert!(matches!(route(&packed).backend, Backend::MppSplitK { .. }));
        let strided = GemmSpec::with_ldc(1, 4096, 4096, 2, 8192);
        assert_eq!(route(&strided).backend, Backend::Vecmat { sg: 8 });
    }

    /// No routing may emit a tile wider than [`MPP_MAX_BN`] - above it the emitter returns
    /// wrong values. Swept over every shape here that reaches an MPP backend at all.
    #[test]
    fn no_route_exceeds_the_mpp_tile_width_limit() {
        for &m in &[1usize, 7, 64, 331, 1024, 4096] {
            for &n in &[32usize, 64, 512, 4096, 16384, 32000, 128256] {
                for &k in &[512usize, 4096, 11008] {
                    for &eb in &[2usize, 4] {
                        let bn = match r(m, n, k, eb).backend {
                            Backend::Mpp { bn, .. } | Backend::MppSplitK { bn, .. } => bn,
                            _ => continue,
                        };
                        assert!(
                            bn <= MPP_MAX_BN,
                            "{m}x{n}x{k} eb={eb} routed to bn={bn}, above the {MPP_MAX_BN} limit"
                        );
                    }
                }
            }
        }
    }

    /// N==1 is its own backend, and N below one chunk falls through to the tiled arm
    /// rather than launching a vecmat with no work in it.
    #[test]
    fn gemv_and_sub_chunk_n() {
        let got = r(4096, 1, 4096, 2);
        assert_eq!(got.backend, Backend::Gemv { bk: 32 });
        assert_eq!(got.reason, Reason::Gemv);
        // M==1 with fewer than 32 columns has no full chunk; it must not claim Vecmat.
        assert_eq!(r(1, 16, 4096, 2).backend, Backend::Sgmma);
    }

    /// The one dtype-dependent clause. fp32 K=11008 streams 2.8 MB per threadgroup, too
    /// much for one pass - so it SPLITS K rather than falling back to SGMMA. Measured:
    /// 27.31 ms split vs 49.07 single-pass MPP vs 37.14 SGMMA.
    /// The same K at 2 bytes is 1.4 MB and stays single-pass.
    #[test]
    fn large_working_set_splits_k_instead_of_falling_back() {
        let fp32 = r(4096, 4096, 11008, 4);
        assert_eq!(
            fp32.backend,
            Backend::MppSplitK {
                bm: 32,
                bn: 32,
                splits: 4
            }
        );
        assert_eq!(fp32.reason, Reason::SplitWorkingSet);
        // 2-byte at the same K is under the budget and needs no split.
        let bf16 = r(4096, 4096, 11008, 2);
        assert_eq!(bf16.backend, Backend::Mpp { bm: 32, bn: 32 });
        // A huge working set BELOW the work floor has nowhere useful to go: splitting a
        // small problem reintroduces the reduce overhead that sank the few-tiles case.
        let small = r(192, 192, 20000, 4);
        assert_eq!(small.backend, Backend::Sgmma);
        assert_eq!(small.reason, Reason::WorkingSetTooLarge);
    }

    /// CORRECTNESS gate: split-K's reduce sums S flat planes of M*N, so a strided
    /// destination would be summed wrong. The router must never hand back a split-K
    /// decision the caller cannot legally execute.
    #[test]
    fn split_k_refused_on_non_packed_output() {
        let packed = route(&GemmSpec::new(4096, 4096, 11008, 4));
        assert!(matches!(packed.backend, Backend::MppSplitK { .. }));

        let strided = route(&GemmSpec::with_ldc(4096, 4096, 11008, 4, 4224));
        assert_eq!(strided.backend, Backend::Sgmma);
        assert_eq!(strided.reason, Reason::WorkingSetTooLarge);
    }

    /// Small aligned shapes stay on SGMMA (1.8x slower on MPP at 128^3), but a RAGGED
    /// shape goes to MPP at any size because MPP clamps its own slices.
    #[test]
    fn small_aligned_stays_but_ragged_goes() {
        let small = r(256, 256, 256, 2);
        assert_eq!(small.backend, Backend::Sgmma);
        assert_eq!(small.reason, Reason::NotEnoughWork);

        let ragged = r(257, 257, 257, 2);
        assert!(ragged.backend.is_mpp(), "ragged must use MPP at any size");
        assert_eq!(ragged.reason, Reason::Ragged);
    }

    /// REGRESSION: gating split-K on total work alone cost 3.5x here. 4096x4096x64 has
    /// ~1e9 work but 16384 tiles and a tiny working set, so splitting buys nothing and
    /// pays 268 MB of partials.
    #[test]
    fn attn_must_not_split_k() {
        let got = r(4096, 4096, 64, 2);
        assert_eq!(
            got.backend,
            Backend::Mpp { bm: 32, bn: 32 },
            "attn is 16384 tiles: single-pass MPP at the big tile"
        );
        assert_eq!(got.reason, Reason::EnoughWork);
    }

    /// REGRESSION: these three shapes used to route to split-K on measurements taken
    /// with an under-dispatched reduce. Corrected, split-K is ~20% SLOWER at all of
    /// them (1024^3 0.1932 vs 0.1599), so they must be single-pass MPP. See the module
    /// note before re-enabling split-K anywhere.
    #[test]
    fn few_tiles_do_not_split_k() {
        for (m, n, k) in [(1024, 1024, 1024), (1023, 1023, 1023), (511, 511, 511)] {
            let got = r(m, n, k, 2);
            assert!(
                matches!(got.backend, Backend::Mpp { .. }),
                "{m}x{n}x{k} must be single-pass MPP, got {:?}",
                got.backend
            );
        }
    }

    /// A big square with plenty of tiles stays single-pass.
    #[test]
    fn large_square_is_single_pass_mpp() {
        let got = r(4096, 4096, 4096, 2);
        assert_eq!(got.backend, Backend::Mpp { bm: 32, bn: 32 });
        assert_eq!(got.reason, Reason::EnoughWork);
    }

    /// The tile rule, pinned to the measurements in [`mpp_tile`].
    #[test]
    fn tile_follows_occupancy() {
        let t = |m, n, k, eb| mpp_tile(&GemmSpec::new(m, n, k, eb));
        // aligned small fp32 squares take 16x16 (128^3 1.23x, 256^3 1.24x)
        assert_eq!(t(128, 128, 128, 4), (16, 16));
        assert_eq!(t(256, 256, 256, 4), (16, 16));
        // ...but RAGGED neighbours must not: 16x16 regressed 257^3 fp32 0.95x,
        // 511^3 fp32 0.76x, 333x444x555 fp32 0.76x. Alignment is the discriminator.
        assert_eq!(t(257, 257, 257, 4), (16, 32));
        assert_eq!(t(511, 511, 511, 4), (16, 32));
        assert_eq!(t(333, 444, 555, 4), (16, 32));
        // 2-byte never takes 16x16
        assert_eq!(t(256, 256, 256, 2), (16, 32));
        // squarish low-occupancy at 2 bytes; fp32 regressed 0.815x at ragged 1023^3
        assert_eq!(t(1024, 1024, 1024, 2), (16, 32));
        assert_eq!(t(1024, 1024, 1024, 4), (32, 32));
        // plenty of threadgroups -> the big tile's reuse wins again
        assert_eq!(t(4096, 4096, 4096, 2), (32, 32));
        assert_eq!(t(2048, 2048, 2048, 2), (32, 32));
    }

    /// fp32 bypasses the work floor, and small ALIGNED fp32 squares then take a 16x16
    /// tile: 256^3 measured 1.24x against the SGMMA routing. At 2 bytes the same shapes
    /// are 0.89-1.00x and keep the floor. Same asymmetry as peak throughput - 13.97 TF
    /// fp32 - so fp32 is worth entering MPP for at every size.
    #[test]
    fn small_squares_are_dtype_split() {
        let f32_small = r(256, 256, 256, 4);
        assert_eq!(f32_small.backend, Backend::Mpp { bm: 16, bn: 16 });
        assert_eq!(f32_small.reason, Reason::EnoughWork);

        let bf16_small = r(256, 256, 256, 2);
        assert_eq!(bf16_small.backend, Backend::Sgmma);
        assert_eq!(bf16_small.reason, Reason::NotEnoughWork);

        // 128^3 fp32 NOW REACHES MPP TOO. This used to read `Reason::ThinN` with a note
        // that the four-condition clause was not worth it "for one cell we already win
        // 2.9x" - but that judgement was made while the suite had its own router, so
        // nobody could see this cell was the last LOSS on it (0.953). Now paid for.
        let f32_128 = r(128, 128, 128, 4);
        assert_eq!(f32_128.backend, Backend::Mpp { bm: 16, bn: 16 });

        // The floor it bypasses must still hold for everything that is not a small
        // aligned fp32 square: thin, ragged, and 2-byte all stay out.
        assert_eq!(r(128, 4096, 4096, 4).reason, Reason::ThinN);
        assert_eq!(r(128, 128, 128, 2).reason, Reason::ThinN);
        assert_eq!(r(129, 129, 129, 4).reason, Reason::ThinN);
        assert_eq!(r(384, 384, 384, 4).reason, Reason::EnoughWork);
    }
}

// ---------------------------------------------------------------------------
// The compiler-owned entry point
// ---------------------------------------------------------------------------

use crate::launch::LoweredKernel;
use crate::mpp::{MppDtype, MppGemm, lower_mpp_gemm};
use crate::profile::TargetProfile;

/// A GEMM to lower. One source of truth for the dtype, so `elem_bytes` cannot disagree
/// with the tag the MPP path emits.
#[derive(Debug, Clone)]
pub struct GemmRequest {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub dtype: MppDtype,
    /// Row strides in elements. `packed` sets them to k / n / n.
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    /// Entry-point name for the emitted kernel.
    pub entry: String,
    /// `B` is `[N, K]` and the product is `A @ B.t()`, the `Linear` weight layout.
    ///
    /// `ldb` is then B's row stride in that layout, i.e. `k` rather than `n`. MPP
    /// expresses this natively (`matmul2d_descriptor::transpose_right`) and MEASURED at
    /// 1.01-1.05x the `nn` time, geomean 1.024, so the routed backend's lead survives the
    /// layout. The SGMMA arm must be an `nt` kernel when this is set - it is caller-
    /// supplied, so `lower` cannot check it.
    pub trans_b: bool,
}

impl GemmRequest {
    /// A row-major packed `C[m,n] = A[m,k] @ B[k,n]`.
    pub fn packed(m: usize, n: usize, k: usize, dtype: MppDtype, entry: impl Into<String>) -> Self {
        Self {
            m,
            n,
            k,
            dtype,
            lda: k,
            ldb: n,
            ldc: n,
            entry: entry.into(),
            trans_b: false,
        }
    }

    /// A row-major packed `C[m,n] = A[m,k] @ B[n,k].t()`, the `Linear` layout.
    pub fn packed_nt(
        m: usize,
        n: usize,
        k: usize,
        dtype: MppDtype,
        entry: impl Into<String>,
    ) -> Self {
        Self {
            ldb: k,
            trans_b: true,
            ..Self::packed(m, n, k, dtype, entry)
        }
    }

    /// The routing input. `elem_bytes` is derived, never supplied.
    pub fn spec(&self) -> GemmSpec {
        GemmSpec::with_ldc(self.m, self.n, self.k, self.dtype.size() as usize, self.ldc)
    }
}

/// A lowered GEMM: the passes to run, in order.
///
/// The point of this type is that a caller can execute it WITHOUT knowing which backend
/// produced it. That is what makes the MPP-vs-SGMMA choice a compiler decision rather
/// than something every call site re-implements - see the module header.
#[derive(Debug, Clone)]
pub enum GemmPlan {
    /// One dispatch, writing the destination directly. Bind `[dst, a, b]`.
    Single(LoweredKernel),
    /// Two dispatches. Bind `[scratch, a, b]` for `partial`, then `[dst, scratch]` for
    /// `reduce`. `scratch` must be `partial.scratch_bytes` long and holds `splits` planes
    /// of `plane_elems` f32.
    SplitK {
        partial: LoweredKernel,
        reduce: LoweredKernel,
        splits: i32,
        plane_elems: usize,
    },
}

impl GemmPlan {
    /// Device scratch the caller must allocate, in bytes; 0 for a single-pass plan.
    pub fn scratch_bytes(&self) -> usize {
        match self {
            GemmPlan::Single(k) => k.scratch_bytes,
            GemmPlan::SplitK { partial, .. } => partial.scratch_bytes,
        }
    }

    /// The kernels in execution order.
    pub fn kernels(&self) -> Vec<&LoweredKernel> {
        match self {
            GemmPlan::Single(k) => vec![k],
            GemmPlan::SplitK {
                partial, reduce, ..
            } => vec![partial, reduce],
        }
    }
}

/// Kernels this crate cannot author, supplied by the caller.
///
/// `wax-metal` LOWERS `cutile_ir`; it does not write DSL. The SGMMA arm and the
/// split-K reduce are both `#[wax::entry]` kernels living in the caller's crate,
/// so the only honest options are to inject them or to move the DSL in here. Injecting
/// keeps the dependency direction intact, and the caller still cannot tell which arm got
/// used - it hands over both capabilities and receives one plan.
pub struct GemmArms<'a> {
    /// Build an SGMMA GEMM for this spec, or `None` if the caller has none.
    pub sgmma: &'a dyn Fn(&GemmSpec) -> Option<LoweredKernel>,
    /// Build the f32-partial reduce: `(plane_elems, splits) -> kernel`. Bound as
    /// `[dst, scratch]`.
    pub reduce: &'a dyn Fn(usize, i32) -> Option<LoweredKernel>,
    /// Build the M==1 threadgroup-reduction vecmat: `(spec, sg) -> kernel`. Bound as
    /// `[dst, x, y]`.
    pub vecmat: &'a dyn Fn(&GemmSpec, i32) -> Option<LoweredKernel>,
    /// Build the N==1 GEMV: `(spec, bk) -> kernel`. Bound as `[dst, a, v]`, like every
    /// other arm - every plan this module produces binds the destination first.
    pub gemv: &'a dyn Fn(&GemmSpec, i32) -> Option<LoweredKernel>,
}

/// Lower `req`, choosing the backend.
///
/// Returns the plan and the [`Routing`] that produced it - the routing is exposed for
/// diagnostics and tests ("this shape is supposed to hit MPP"), NOT because callers need
/// it to run the plan. `None` means the chosen backend needed an arm the caller did not
/// supply.
pub fn lower(
    req: &GemmRequest,
    profile: &TargetProfile,
    arms: &GemmArms<'_>,
) -> Option<(GemmPlan, Routing)> {
    let spec = req.spec();
    let routing = route(&spec);
    let plan = match routing.backend {
        Backend::Sgmma => GemmPlan::Single((arms.sgmma)(&spec)?),
        Backend::Mpp { bm, bn } => {
            GemmPlan::Single(lower_mpp_gemm(&mpp_cfg(req, bm, bn, 1, req.dtype), profile))
        }
        Backend::MppSplitK { bm, bn, splits } => {
            // Partials must be f32; `lower_mpp_gemm` reports the scratch size for them.
            let partial = lower_mpp_gemm(&mpp_cfg(req, bm, bn, splits, MppDtype::F32), profile);
            let plane_elems = req.m * req.n;
            let reduce = (arms.reduce)(plane_elems, splits)?;
            GemmPlan::SplitK {
                partial,
                reduce,
                splits,
                plane_elems,
            }
        }
        Backend::Vecmat { sg } => GemmPlan::Single((arms.vecmat)(&spec, sg)?),
        Backend::Gemv { bk } => GemmPlan::Single((arms.gemv)(&spec, bk)?),
    };
    Some((plan, routing))
}

fn mpp_cfg(req: &GemmRequest, bm: i32, bn: i32, splits: i32, out_dtype: MppDtype) -> MppGemm {
    MppGemm {
        entry: req.entry.clone(),
        dtype: req.dtype,
        m: req.m as i32,
        n: req.n as i32,
        k: req.k as i32,
        lda: req.lda as i32,
        ldb: req.ldb as i32,
        ldc: req.ldc as i32,
        trans_b: req.trans_b,
        bm,
        bn,
        // Measured: the work duplicates per SIMD group rather than dividing, so 1 is the
        // only sane value.
        simdgroups: 1,
        out_dtype,
        splits,
    }
}

#[cfg(test)]
mod sgmma_tile_tests {
    use super::*;

    /// The tile selectors were lifted out of two example files that each held a verbatim
    /// copy. This pins the port: every bucket must still produce what the copies did,
    /// EXCEPT the one bucket the 2026-08-11 measurement changed.
    #[test]
    fn port_preserves_the_original_buckets() {
        // (m, n, k, expected aligned, expected clamped)
        let cases = [
            // thin <= 32 and thin <= 64 buckets: unchanged in both selectors.
            (4096, 32, 4096, (16, 16), (16, 16)),
            (4096, 64, 4096, (8, 32), (8, 32)),
            // tiles32 > 8192, thin >= 192: the `Fp32WorkingSet` cells, which want the
            // squarer tile (4096^3 went 0.935 -> 1.007 relative on 32x32).
            (4096, 4096, 4096, (32, 32), (32, 32)),
            // THE SECOND CHANGED BUCKET (256 < tiles32 <= 8192), split by `thin`:
            // >= 192 is a big deep-K cell reached only via the fp32 clause -> 32x32;
            // < 192 is the thin case that always reached it -> 32x64. Clamped is
            // untouched in both.
            (512, 2048, 2048, (32, 32), (32, 32)),
            (128, 8192, 2048, (32, 64), (32, 32)),
            // THE CHANGED BUCKET (tiles32 <= 256): aligned now prefers 32x32, clamped
            // deliberately still returns 16x16 - it was never measured.
            (128, 2048, 2048, (32, 32), (32, 32)),
            (128, 128, 2048, (32, 32), (32, 32)),
            (512, 512, 512, (32, 32), (32, 32)),
        ];
        for (m, n, k, want_aligned, want_clamped) in cases {
            let spec = GemmSpec::new(m, n, k, 4);
            assert_eq!(
                sgmma_tile_aligned(&spec),
                want_aligned,
                "aligned {m}x{n}x{k}"
            );
            assert_eq!(
                sgmma_tile_clamped(&spec),
                want_clamped,
                "clamped {m}x{n}x{k}"
            );
        }
    }

    /// The aligned selector must never return a tile that does not divide the problem -
    /// the aligned kernel has no edge handling, so a non-dividing tile is wrong answers,
    /// not just slow ones.
    #[test]
    fn aligned_tile_always_divides() {
        for &(m, n) in &[
            (128, 128),
            (128, 2048),
            (512, 512),
            (2048, 2048),
            (4096, 4096),
        ] {
            for k in [512usize, 2048, 8192] {
                let (bm, bn) = sgmma_tile_aligned(&GemmSpec::new(m, n, k, 4));
                assert!(
                    m.is_multiple_of(bm) && n.is_multiple_of(bn),
                    "{m}x{n}x{k} -> {bm}x{bn}"
                );
            }
        }
    }

    /// The clamped selector must never exceed the problem, or the clamp cannot place a
    /// tile at all.
    #[test]
    fn clamped_tile_fits() {
        for &(m, n) in &[(8, 8), (24, 40), (100, 50), (331, 331), (4096, 8)] {
            let (bm, bn) = sgmma_tile_clamped(&GemmSpec::new(m, n, 512, 4));
            assert!(bm <= m.max(8) && bn <= n.max(8), "{m}x{n} -> {bm}x{bn}");
        }
    }
}

#[cfg(test)]
mod fp32_working_set_tests {
    use super::*;

    fn route4(m: usize, n: usize, k: usize) -> Routing {
        route(&GemmSpec::new(m, n, k, 4))
    }

    /// The cells the clause exists for: fp32, non-ragged, stream >= 1 MiB. All of these
    /// are packed and above the work floor, so they take SPLIT-K.
    #[test]
    fn large_fp32_streams_split_k() {
        for &(m, n, k) in &[
            (4096, 11008, 4096), // the suite's only loss
            (4096, 4096, 4096),
            (2048, 2048, 8192),
            (512, 2048, 8192),
        ] {
            let r = route4(m, n, k);
            assert!(
                matches!(r.backend, Backend::MppSplitK { .. }),
                "{m}x{n}x{k} backend: {:?}",
                r.backend
            );
            assert_eq!(r.reason, Reason::Fp32WorkingSet, "{m}x{n}x{k} reason");
        }
    }

    /// The clause falls back to SGMMA when the shape cannot legally or profitably split:
    /// a non-packed destination would be summed wrong by the reduce, and below the work
    /// floor two dispatches do not amortise.
    #[test]
    fn fp32_clause_falls_back_to_sgmma_when_it_cannot_split() {
        // Below MPP_WORK_FLOOR: 256x256x8192 is 5.4e8 < 2^30.
        let r = route4(256, 256, 8192);
        assert_eq!(r.reason, Reason::Fp32WorkingSet);
        assert_eq!(r.backend, Backend::Sgmma);
        // Non-packed destination, above the floor.
        let r = route(&GemmSpec::with_ldc(2048, 2048, 8192, 4, 4096));
        assert_eq!(r.reason, Reason::Fp32WorkingSet);
        assert_eq!(r.backend, Backend::Sgmma);
    }

    /// The other side of the crossover, where MPP measured FASTER. A regression here is
    /// the clause being too greedy, which is the failure mode that matters.
    #[test]
    fn small_fp32_streams_stay_on_mpp() {
        for &(m, n, k) in &[(2048, 2048, 2048), (512, 512, 2048), (1024, 1024, 1024)] {
            let r = route4(m, n, k);
            assert!(
                r.backend.is_mpp(),
                "{m}x{n}x{k} must stay on MPP, got {:?} via {:?}",
                r.backend,
                r.reason
            );
        }
    }

    /// 2-byte dtypes are untouched: the crossover was only ever measured at fp32, and at
    /// 2 bytes we are already at the ~15.3 TF hardware ceiling.
    #[test]
    fn half_precision_is_unaffected() {
        for &(m, n, k) in &[(4096, 11008, 4096), (2048, 2048, 8192)] {
            for eb in [2usize, 2usize].into_iter().take(1) {
                let r = route(&GemmSpec::new(m, n, k, eb));
                assert_ne!(
                    r.reason,
                    Reason::Fp32WorkingSet,
                    "{m}x{n}x{k} eb={eb} must not take the fp32 clause"
                );
            }
        }
    }

    /// Above `WORKING_SET_MAX` split-K is better than both, so the new clause must not
    /// steal those cells - it is deliberately ordered after.
    #[test]
    fn split_k_still_wins_the_biggest_streams() {
        let r = route4(4096, 4096, 11008);
        assert!(
            matches!(r.backend, Backend::MppSplitK { .. }),
            "expected split-K, got {:?} via {:?}",
            r.backend,
            r.reason
        );
        assert_eq!(r.reason, Reason::SplitWorkingSet);
    }

    /// Ragged is excluded - `Reason::Ragged` records MPP winning at any size there, and
    /// it was not re-measured.
    #[test]
    fn ragged_is_excluded() {
        let r = route(&GemmSpec::new(2044, 2044, 8188, 4));
        assert_ne!(r.reason, Reason::Fp32WorkingSet);
    }
}

#[cfg(test)]
mod mpp_deep_k_tile_tests {
    use super::*;

    /// 512x512x2048 fp32 is what this clause exists for: 16x16 read 0.824-0.831 against
    /// baseline, 32x32 reads 1.081.
    #[test]
    fn deep_k_fp32_takes_the_full_tile() {
        assert_eq!(mpp_tile(&GemmSpec::new(512, 512, 2048, 4)), MPP_TILE);
    }

    /// The shallow-K squares the small/tiny tiles were measured on must not move: 128^3
    /// and 256^3 measured 1.23x and 1.24x on 16x16, and 512^3 is below the K floor.
    #[test]
    fn shallow_k_squares_keep_the_tiny_tile() {
        for n in [128usize, 256, 512] {
            assert_eq!(
                mpp_tile(&GemmSpec::new(n, n, n, 4)),
                MPP_TILE_TINY,
                "{n}^3 fp32"
            );
        }
    }

    /// A deep-K shape too small to fill a 32x32 grid stays on the small tile - the
    /// `tiles >= 128` guard.
    #[test]
    fn deep_k_but_tiny_grid_is_excluded() {
        // 256x256 at 32x32 is 8x8 = 64 tiles, under the floor.
        assert_eq!(mpp_tile(&GemmSpec::new(256, 256, 2048, 4)), MPP_TILE_TINY);
    }

    /// 2-byte and ragged are untouched: neither was measured for this clause.
    #[test]
    fn half_and_ragged_unaffected() {
        // fp16 512x512x2048 would be MPP_TILE_SMALL via the squarish clause.
        assert_ne!(mpp_tile(&GemmSpec::new(512, 512, 2048, 2)), MPP_TILE);
        // Ragged fp32 skips the clause: 500x500x2048 is 256 tiles at 32x32, so without
        // the clause it takes the small tile - and must keep taking it. (516x516x2044
        // would be a useless test: at 289 tiles it reaches MPP_TILE through the fallback
        // anyway, so the assertion could not distinguish the two paths.)
        assert_eq!(mpp_tile(&GemmSpec::new(500, 500, 2048, 4)), MPP_TILE_SMALL);
    }
}

#[cfg(test)]
mod clamped_dtype_tests {
    use super::*;

    /// Aligned shapes of every dtype take the 32x32 tile the sweep chose.
    #[test]
    fn aligned_takes_the_square_tile() {
        for eb in [2usize, 4] {
            for &(m, n, k) in &[(2048, 2048, 8192), (512, 2048, 2048), (128, 8192, 2048)] {
                assert_eq!(
                    sgmma_tile_clamped(&GemmSpec::new(m, n, k, eb)),
                    (32, 32),
                    "{m}x{n}x{k} eb={eb}"
                );
            }
        }
    }

    /// Ragged shapes take the SAME tile as aligned ones. A 2-byte/ragged exception was
    /// tried and REMOVED: the cell justifying it (`331x2048x2048 bf16`, 16x16 0.98 vs
    /// 32x32 0.63) read 0.976 and 0.621 on two runs of the SAME binary - 36% apart, so it
    /// was noise, and the exception did not survive restricting the fit to cells that
    /// repeat within 5%. Pinned so it cannot creep back without a stability check.
    #[test]
    fn ragged_takes_the_same_tile_as_aligned() {
        for eb in [2usize, 4] {
            assert_eq!(
                sgmma_tile_clamped(&GemmSpec::new(331, 2048, 2048, eb)),
                (32, 32),
                "ragged eb={eb}"
            );
        }
    }

    /// The thin buckets are NOT covered by the sweep (it starts at n=512), so they must
    /// keep the original heuristic rather than drift onto the square tile.
    #[test]
    fn thin_shapes_are_untouched() {
        assert_eq!(
            sgmma_tile_clamped(&GemmSpec::new(4096, 32, 4096, 4)),
            (16, 16)
        );
        assert_eq!(
            sgmma_tile_clamped(&GemmSpec::new(4096, 64, 4096, 4)),
            (8, 32)
        );
    }

    /// The tile must still fit the problem, shrunk to the 8x8 fragment.
    #[test]
    fn still_fits() {
        for &(m, n) in &[(8usize, 8usize), (24, 40), (100, 50), (4096, 8)] {
            let (bm, bn) = sgmma_tile_clamped(&GemmSpec::new(m, n, 2048, 4));
            assert!(bm <= m.max(8) && bn <= n.max(8), "{m}x{n} -> {bm}x{bn}");
        }
    }
}

#[cfg(test)]
mod nt_request_tests {
    use super::*;

    /// `packed_nt` describes `A @ B.t()`: B's row stride becomes K, and the flag reaches
    /// the MPP config so the descriptor and every cooperative-destination helper carry
    /// `transpose_right` (see `mpp.rs` - setting only some of them is silent).
    #[test]
    fn nt_request_sets_stride_and_flag() {
        let r = GemmRequest::packed_nt(512, 2048, 4096, MppDtype::F32, "g");
        assert_eq!(r.ldb, 4096, "B is [N,K], so its row stride is K");
        assert!(r.trans_b);
        let c = mpp_cfg(&r, 32, 32, 1, MppDtype::F32);
        assert!(c.trans_b, "the flag must reach MppGemm");
        assert_eq!(c.ldb, 4096);
        // nn is unchanged.
        let r2 = GemmRequest::packed(512, 2048, 4096, MppDtype::F32, "g");
        assert_eq!(r2.ldb, 2048);
        assert!(!r2.trans_b);
        assert!(!mpp_cfg(&r2, 32, 32, 1, MppDtype::F32).trans_b);
    }

    /// Routing is a function of SHAPE, not layout: an nt request must take the same
    /// backend and clause as its nn twin, so the measured routing work carries over.
    #[test]
    fn nt_routes_identically_to_nn() {
        for &(m, n, k) in &[(2048, 2048, 8192), (2048, 2048, 2048), (512, 512, 2048)] {
            let nn = route(&GemmRequest::packed(m, n, k, MppDtype::F32, "g").spec());
            let nt = route(&GemmRequest::packed_nt(m, n, k, MppDtype::F32, "g").spec());
            assert_eq!(nn.backend, nt.backend, "{m}x{n}x{k} backend");
            assert_eq!(nn.reason, nt.reason, "{m}x{n}x{k} reason");
        }
    }
}
