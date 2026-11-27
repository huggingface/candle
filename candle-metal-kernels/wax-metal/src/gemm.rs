//! GEMM/GEMV routing.
//!
//! Compiler takes the wax-ir analysis and decides the optimal route from previously measured performance.
//! Any change to the decisions here must be backed by performance measurements.

use crate::profile::TargetProfile;
use wax_ir::passes::{AnalysisManager, gemm_analysis::GemmAnalysis};

/// The problem to be lowered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub elem_bytes: usize,
    /// True means `A @ B.t()` (standard `Linear` layer).
    pub trans_b: bool,
    /// Row stride of the destination in elements. `n` means packed.
    /// Split-k reduce uses `ldc` as its row stride.
    pub ldc: usize,
    /// Batch count.
    pub batch: usize,
    /// A batch stride.
    pub a_batch: usize,
    /// B batch stride.
    pub b_batch: usize,
}

impl GemmSpec {
    /// A packed (`ldc == n`) problem.
    pub fn new(m: usize, n: usize, k: usize, elem_bytes: usize) -> Self {
        Self {
            m,
            n,
            k,
            elem_bytes,
            trans_b: false,
            ldc: n,
            batch: 1,
            a_batch: 0,
            b_batch: 0,
        }
    }

    pub fn with_ldc(m: usize, n: usize, k: usize, elem_bytes: usize, ldc: usize) -> Self {
        Self {
            m,
            n,
            k,
            elem_bytes,
            trans_b: false,
            ldc,
            batch: 1,
            a_batch: 0,
            b_batch: 0,
        }
    }

    pub fn trans_b(self) -> Self {
        Self {
            trans_b: true,
            ..self
        }
    }

    /// A shape the SGMMA path would have to pad or clamp.
    pub fn is_ragged(&self) -> bool {
        !self.m.is_multiple_of(8) || !self.n.is_multiple_of(8) || !self.k.is_multiple_of(8)
    }
}

/// The reason for a chosen GEMM implementation. Lets us catch optimization changes at a finer
/// granularity.
///
/// `sgmma(Reason::ThinN)` reads "chose sgmma because of thin N".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// MPP tiles use `bn` columns regardless of what N is, so thin-N and GEMV starve it.
    /// (4096x8 is 2.5x slower on MPP, 1x4096 is 3x)
    ThinN,
    /// A thin M with a wide N. Not the same problem as a thin N. See [`MPP_THIN_M_TILE`].
    ThinM,
    /// `(BM+BN)*K` exceeds per-threadgroup stream budget.
    WorkingSetTooLarge,
    /// Not enough work to justify MPP fixed setup cost.
    NotEnoughWork,
    /// MPP wins on all ragged shapes.
    Ragged,
    /// Enough work to justify MPP fixed setup cost.
    EnoughWork,
    /// GEMM is batched.
    Batched,
    /// No SGMMA tile divides the shape, so the tile loop would run on 8x8 fragments.
    /// See [`MPP_COLLAPSED_TILE_WORK_FLOOR`].
    TileCollapsed,
    /// Large per-threadgroup stream in fp32. If unable to split-K then SGMMA wins over MPP.
    Fp32WorkingSet,
    /// Large working set. Split work on K and reduce.
    SplitWorkingSet,
    /// `C = A[1,K] @ B[K,N]`. `M == 1`.
    /// Small N with a deep K: split-k gemv_t. Medium or large N: single-dispatch gemv_t.
    GemvT,
    /// `C = A[M,K] @ B[K,1]`. `N == 1`. One threadgroup per output row.
    Gemv,
}

/// Routing decision with reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Routing {
    pub plan: Plan,
    pub reason: Reason,
}

/// Plan describing how to lower.
///
/// We either run our wax-ir GEMM optimization passes and lower normally, or we build a MPP kernel
/// for the specific problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// Apply optimization passes and lower normally.
    Lower { sg: i32, splits: i32 },
    /// Build a MPP cooperative-tensor kernel.
    Mpp { bm: i32, bn: i32, splits: i32 },
}

impl Plan {
    pub fn is_mpp(&self) -> bool {
        matches!(self, Plan::Mpp { .. })
    }
}

/// SGMMA walks K in blocks of this many elements.
pub const SGMMA_BK: usize = 8;

/// Calculates the [`GemmSpec`]'s `(BM, BN, BK)` for SGMMA.
///
/// Balances reuse (`BM*BN/(BM+BN)` MACs per loaded element against
/// the grid (`ceil(M/BM)*ceil(N/BN)` threadgroups).
/// Testing has found that the limit is arithmetic intensity, as opposed to occupancy.
/// For example at 128x2048 a 16x16 tile still fills the GPU, but loads 4x the data per MAC.
pub fn sgmma_tile(spec: &GemmSpec) -> (usize, usize, usize) {
    // Narrow-N pairs well with a deep K tile and short BM.
    if spec.m.min(spec.n) <= 64 && spec.k.is_multiple_of(32) {
        let bn = if spec.n <= 32 {
            spec.n.clamp(8, 16)
        } else {
            32
        };
        // At `BK=32` each threadgroup reads A as contiguous 32-column strips
        if spec.m.is_multiple_of(8) && spec.n.is_multiple_of(bn) {
            return (8, bn, 32);
        }
    }
    let (bm, bn) = sgmma_tile_aligned(spec);
    (bm, bn, SGMMA_BK)
}

/// Calculates the [`GemmSpec`]'s `(BM, BN)` tile for aligned SGMMA.
///
/// Choices are measured. Improvements require the same.
pub fn sgmma_tile_aligned(spec: &GemmSpec) -> (usize, usize) {
    let thin = spec.m.min(spec.n);
    let tiles32 = spec.m.div_ceil(32) * spec.n.div_ceil(32);
    let prefs: &[(usize, usize)] = if thin <= 32 {
        &[(16, 16), (8, 16), (8, 8)]
    } else if thin <= 64 {
        &[(8, 32), (16, 16), (8, 8)]
    } else if tiles32 <= SMALL_TILE_MAX_TILES {
        &[(32, 32), (16, 32), (16, 16), (8, 8)]
    } else if tiles32 <= 8192 || spec.k < 256 {
        if thin >= 192 {
            &[(32, 32), (32, 64), (16, 32), (16, 16), (8, 8)]
        } else {
            &[(16, 32), (32, 64), (32, 32), (16, 16), (8, 8)]
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

/// Calculates the [`GemmSpec`]'s `(BM, BN)` tile for clamped SGMMA.
///
/// SGMMA handles a ragged shape by clamping the tile inside the matrix, so the tile just needs
/// to fit. Shrunk to an 8x8 fragment.
pub fn sgmma_tile_clamped(spec: &GemmSpec) -> (usize, usize) {
    let thin = spec.m.min(spec.n);
    let (bm, bn) = if thin <= 32 {
        // TODO: Thin shapes have not been sweeped yet.
        (16, 16)
    } else if thin <= 64 {
        (8, 32)
    } else {
        (32, 32)
    };
    (bm.min(spec.m & !7).max(8), bn.min(spec.n & !7).max(8))
}

/// Default MPP tile. One SIMD group per threadgroup either way.
const MPP_TILE: (i32, i32) = (32, 32);
/// Low-occupancy tile. Half the rows per threadgroup, twice the grid.
const MPP_TILE_SMALL: (i32, i32) = (16, 32);
/// Tile for tiny grids. Half the rows/cols per threadgroup, 4x the grid.
const MPP_TILE_TINY: (i32, i32) = (16, 16);

/// Upper limit tile count where small tiles are beneficial.
const SMALL_TILE_MAX_TILES: usize = 256;
const SQUARISH_MAX_TILES: usize = 1024;
const SQUARISH_MIN_TILES_PER_DIM: usize = 16;
/// Largest aligned fp32 square allowed past the thin-N floor.
const SMALL_SQUARE_MAX: usize = 256;
/// Threadgroup-count lower limit for taking the full 32x32 MPP tile on a deep-K fp32 shape.
/// Below this `MPP_TILE_SMALL` or `MPP_TILE_TINY` is preferred.
const MPP_FULL_TILE_MIN_TILES: usize = 128;

/// Calculates the [`GemmSpec`]'s `(BM, BN)` for MPP GEMM.
///
/// The grid is `ceil(M/BM) * ceil(N/BN)` threadgroups. Fewer threadgroups (smaller gemm problem)
/// indicate that choosing a smaller tile will be more performant. For example 32x32 -> 16x32
/// gives twice the grid for the same amount of work.
/// This is not true for ragged and deep-k gemms in fp32, where tile reuse matters more.
pub fn mpp_tile(spec: &GemmSpec) -> (i32, i32) {
    let (bm, bn) = MPP_TILE;
    let tm = spec.m.div_ceil(bm as usize);
    let tn = spec.n.div_ceil(bn as usize);
    let tiles = tm * tn;

    let is_fp32 = spec.elem_bytes >= 4;
    let is_deep_k = spec.k >= 1024;
    let is_ragged = spec.is_ragged();

    let over_mpp_min_tiles_limit = tiles >= MPP_FULL_TILE_MIN_TILES;
    let below_small_tiles_limit = tiles <= SMALL_TILE_MAX_TILES;
    let below_squarish_tiles_limit = tiles <= SQUARISH_MAX_TILES;

    if is_fp32 && !is_ragged && is_deep_k && over_mpp_min_tiles_limit {
        return MPP_TILE;
    }
    if is_fp32 && !is_ragged && below_small_tiles_limit {
        return MPP_TILE_TINY;
    }

    let is_thin = tm.min(tn) < SQUARISH_MIN_TILES_PER_DIM;
    let is_squarish = !is_thin;
    let is_small_dtype = spec.elem_bytes <= 2;

    if below_small_tiles_limit || (below_squarish_tiles_limit && is_squarish && is_small_dtype) {
        MPP_TILE_SMALL
    } else {
        MPP_TILE
    }
}

/// Per-threadgroup stream budget in bytes. SGMMA or split-k.
const WORKING_SET_MAX: usize = 2_500_000;
/// fp32 stream budget. Above this SGMMA beats MPP.
const FP32_MPP_WORKING_SET_MAX: usize = 1 << 20;
/// Minimum amount of work for MPP to be beneficial.
const MPP_WORK_FLOOR: usize = 1 << 30;
/// Max batch count for batched MPP.
const MPP_BATCHED_MAX_BATCH: usize = 4;
/// Smallest per batch work worth giving batched MPP.
const MPP_BATCHED_WORK_FLOOR: usize = 128 << 20;
/// Work floor for a shape no SGMMA tile divides, which falls through [`sgmma_tile_aligned`] to
/// an 8x8 fragment.
const MPP_COLLAPSED_TILE_WORK_FLOOR: usize = 12 << 20;
/// K splits used when the working set size forces a split. Measured. 2 and 8 are both worse.
// TODO: sweep for both different split-k problems and splits.
const SPLITS: i32 = 4;
/// Tile count at which an f32 deep-K grid already fills the device, so split-K stops paying.
const FP32_SPLIT_MAX_TILES: usize = 512;

/// SIMD groups for the transposed `gemv_t` layout at a short K, where each group owns one
/// output column and the grid is `n / sg`.
const GEMV_T_NT_SG: i32 = 16;
/// The same, at a long K, where the column already has work and parallelism wins.
const GEMV_T_NT_LONG_SG: i32 = 2;
/// K at or below which a transposed `gemv_t` column is too short to fill a small threadgroup.
const GEMV_T_NT_SHORT_K: usize = 512;
/// Minimum N for `gemv_t` (`M == 1`). A 32-column chunk is required.
const GEMV_T_MIN_N: usize = 32;
/// Maximum N for split-k gemv_t. At or below single dispatch wins.
const GEMV_T_SMALL_MAX_N: usize = 2048;
/// Above this N the wide MPP tile beats the narrow one.
const GEMV_T_WIDE_MIN_N: usize = 8192;
/// Widest MPP tile our emitter produces correct results for at one SIMD group.
pub const MPP_MAX_BN: i32 = 128;
/// K splits for gemv_t.
const GEMV_T_SPLITS: i32 = 16;
/// At or below this N single dispatch wins.
// TODO: Figure out relation to `GEMV_T_SMALL_MAX_N`.
pub const GEMV_T_SPLITK_MAX_N: usize = 1024;
/// Minimum K for split-k gemv_t.
pub const GEMV_T_SPLITK_MIN_K: usize = 4096;
/// The target's SIMD width.
// TODO: Should not be stored here.
pub const SIMD_WIDTH: i64 = 32;

/// Below this, a dimension starves an MPP tile that uses `bn` columns regardless.
const MPP_THIN_MAX: usize = 192;
/// Smallest M worth giving MPP's short tile. Below it the as-written loop wins.
const MPP_THIN_M_MIN: usize = 32;
/// MPP's tile for a thin M.
const MPP_THIN_M_TILE: (i32, i32) = (16, 32);

/// Choose a GEMM impl based on the provided [`GemmSpec`].
pub fn route(spec: &GemmSpec) -> Routing {
    if spec.batch > 1 {
        if spec.m == 1 {
            return Routing {
                plan: Plan::Lower { sg: 0, splits: 1 },
                reason: Reason::Batched,
            };
        }
        let unbatched = GemmSpec {
            batch: 1,
            ..spec.clone()
        };
        // MMP is eligible for batched gemm, but only for narrow problems.
        let per_batch_work = spec.m * spec.n * spec.k;
        let mpp_band =
            spec.batch <= MPP_BATCHED_MAX_BATCH && per_batch_work >= MPP_BATCHED_WORK_FLOOR;
        let plan = match route_inner(&unbatched).plan {
            Plan::Mpp { bm, bn, .. } => Plan::Mpp { bm, bn, splits: 1 },
            Plan::Lower { .. } if mpp_band => {
                let (bm, bn) = mpp_tile(spec);
                Plan::Mpp { bm, bn, splits: 1 }
            }
            // Batching fills occupancy fast, hence `sg: 0`.
            Plan::Lower { .. } => Plan::Lower { sg: 0, splits: 1 },
        };
        return Routing {
            plan,
            reason: Reason::Batched,
        };
    }
    route_inner(spec)
}

fn route_inner(spec: &GemmSpec) -> Routing {
    let (bm, bn) = mpp_tile(spec);
    let sgmma = |reason| Routing {
        plan: Plan::Lower { sg: 0, splits: 1 },
        reason,
    };

    // GEMV and transposed GEMV handled first.
    if spec.m == 1 {
        // Ensure problem is large enough for the transposed GEMV impl
        if spec.n >= GEMV_T_MIN_N {
            let small_enough_n = spec.n <= GEMV_T_SPLITK_MAX_N;
            let large_enough_k = spec.k >= GEMV_T_SPLITK_MIN_K;
            // Split-k transposed Gemv. `ceil(N/32)` threadgroups do not fill the device,
            // and K has enough room for a split to be valuable.
            if small_enough_n && large_enough_k {
                let splits = (spec.n / 32).clamp(2, GEMV_T_SPLITS as usize) as i32;
                return Routing {
                    plan: Plan::Lower { sg: 32, splits },
                    reason: Reason::GemvT,
                };
            }

            let small = spec.n <= GEMV_T_SMALL_MAX_N;
            let wide = spec.n > GEMV_T_WIDE_MIN_N;
            let small_dtype = spec.elem_bytes <= 2;
            // Single dispatch gemv_t.
            // If the problem is small enough to be launch bound we want 32 groups.
            // At wide N and small dtypes (f16/bf16) we want 16 groups, and otherwise
            // (large or f32) we want 8.
            let sg = if small {
                32
            } else if wide && small_dtype {
                16
            } else {
                8
            };
            // Take B layout into account.
            let sg = if spec.trans_b {
                // Pick best simd group size for `gemv_t` `Nt`.
                if spec.k <= GEMV_T_NT_SHORT_K {
                    GEMV_T_NT_SG
                } else {
                    GEMV_T_NT_LONG_SG
                }
            } else {
                sg
            };
            return Routing {
                plan: Plan::Lower { sg, splits: 1 },
                reason: Reason::GemvT,
            };
        }
    } else if spec.n == 1 {
        // Standard GEMV
        return Routing {
            // TODO: Rare shape, but could impl a (non-transposed) wax gemv that broadcasts
            // each row's dot product across `bk` lanes and writes `[M, bk]` with the answer
            // in column 0.
            plan: Plan::Lower { sg: 0, splits: 1 },
            reason: Reason::Gemv,
        };
    }

    // Consider thin-M case
    if spec.m >= MPP_THIN_M_MIN && spec.m < MPP_THIN_MAX && spec.n >= MPP_THIN_MAX {
        let (bm, bn) = MPP_THIN_M_TILE;
        return Routing {
            plan: Plan::Mpp { bm, bn, splits: 1 },
            reason: Reason::ThinM,
        };
    }

    let is_aligned = !spec.is_ragged();
    let is_f32 = spec.elem_bytes == 4;
    let is_square = spec.m == spec.n;
    let small_aligned_f32_square = is_f32 && is_square && is_aligned && spec.m <= SMALL_SQUARE_MAX;
    let is_thin = spec.m.min(spec.n) < MPP_THIN_MAX;
    // MPP generally does not perform well on thin N problems, with the exception of small aligned
    // fp32 squares.
    if is_thin && !small_aligned_f32_square {
        return sgmma(Reason::ThinN);
    }

    let work = spec.m * spec.n * spec.k;
    let working_set = (bm as usize + bn as usize) * spec.k * spec.elem_bytes;
    if working_set > WORKING_SET_MAX {
        // Too much for one pass.
        // K will always be fairly large here (minimum 9766 based on the calculation above), so
        // reducing `S*M*N` is cheap compared to matmul of `M*N*K`.
        if work >= MPP_WORK_FLOOR {
            return Routing {
                plan: Plan::Mpp {
                    bm,
                    bn,
                    splits: SPLITS,
                },
                reason: Reason::SplitWorkingSet,
            };
        }
        return sgmma(Reason::WorkingSetTooLarge);
    }

    if is_aligned && is_f32 && working_set >= FP32_MPP_WORKING_SET_MAX {
        let tiles = spec.m.div_ceil(bm as usize) * spec.n.div_ceil(bn as usize);
        let splits = if tiles >= FP32_SPLIT_MAX_TILES {
            1
        } else {
            SPLITS
        };
        return Routing {
            plan: Plan::Mpp { bm, bn, splits },
            reason: Reason::Fp32WorkingSet,
        };
    }

    // A shape no tile divides gets the collapsed floor.
    let collapsed = sgmma_tile_aligned(spec) == (8, 8);
    let floor = if collapsed {
        MPP_COLLAPSED_TILE_WORK_FLOOR
    } else {
        MPP_WORK_FLOOR
    };
    // The setup cost of MPP is not worth it if the work is below the floor, with the exception
    // of small aligned f32 squares.
    if work < floor && is_aligned && !is_f32 {
        return sgmma(Reason::NotEnoughWork);
    }

    // Single-pass MPP.
    let rerouted = collapsed && is_aligned && !is_f32 && work < MPP_WORK_FLOOR;
    Routing {
        plan: Plan::Mpp { bm, bn, splits: 1 },
        reason: match (is_aligned, rerouted) {
            (false, _) => Reason::Ragged,
            (true, true) => Reason::TileCollapsed,
            (true, false) => Reason::EnoughWork,
        },
    }
}

/// Calculates the [`GemmSpec`]'s `(BM, BN, BK)` for the unsubstituted GEMM.
///
/// Same output as `sgmma_tile`, except when `M = 1`.
/// `K % BK != 0` is handled by `padding::Zero`. Surplus lanes read zero.
pub fn module_tile(spec: &GemmSpec) -> (usize, usize, usize) {
    if spec.m == 1 {
        return (1, 32, 1);
    }
    let (bm, bn, bk) = sgmma_tile(spec);
    // A K below one block takes `BK = 1`, which divides everything.
    if spec.k < bk {
        return (1, 32, 1);
    }
    (bm, bn, bk)
}

/// A lowered GEMM kernel
#[derive(Debug, Clone)]
pub struct GemmLowering {
    pub kernel: crate::LoweredKernel,
    /// The backend `route` asked for.
    pub wanted: Plan,
    /// Why it asked for that.
    pub reason: Reason,
    /// If lowering the plan fell back to the original GEMM this will a message describing why.
    pub fell_back: Option<&'static str>,
}

fn mpp_dtype_of(s: wax_ir::types::ScalarType) -> Option<crate::mpp::MppDtype> {
    use wax_ir::types::ScalarType as S;
    Some(match s {
        S::F32 => crate::mpp::MppDtype::F32,
        S::F16 => crate::mpp::MppDtype::F16,
        S::BF16 => crate::mpp::MppDtype::BF16,
        _ => return None,
    })
}

/// Lower the GEMM `module` for the problem `spec` describes.
///
/// `module` is the GEMM as written. Simple tile loop over `mma_f`.
///
/// Lowering has four stages:
/// 1. analyse the kernel.
/// 2. choose a backend.
/// 3. realise it.
/// 4. if lowering fails - fall back to the original kernel.
// TODO: the facts should be marked on the op by a `gemm` analysis pass and carried in the IR,
// rather than re-derived here. Tuning should remain here in the compiler.
pub fn lower_gemm(
    module: &wax_ir::Module,
    spec: &GemmSpec,
    p: &TargetProfile,
) -> Result<GemmLowering, crate::LowerError> {
    let routing = route(spec);

    let (kernel, mut fell_back) = match routing.plan {
        // MPP kernel substitution
        Plan::Mpp { bm, bn, splits } => {
            let mut r = recognise(module, spec)?;
            match realise_mpp(&r.module, r.func, &r.pattern, spec, p, (bm, bn, splits))? {
                Realised::Kernel { kernel, note } => (kernel, note),
                Realised::Declined(why) => (
                    lower(&mut r.module, r.func, &r.pattern, None, None, spec, p)?,
                    Some(why),
                ),
            }
        }
        // JIT optimize wax-ir kernel and lower.
        Plan::Lower { sg, splits } => {
            let mut r = realise(module, sg, splits, spec)?;
            let launch = r.rewritten.launch;
            let reduce = r.rewritten.reduce.as_ref();
            let kernel = lower(&mut r.module, r.func, &r.pattern, launch, reduce, spec, p)?;
            (kernel, r.rewritten.declined)
        }
    };

    // The `N == 1` route plans no realisation because none exists yet.
    if routing.reason == Reason::Gemv {
        fell_back = fell_back.or(Some(GEMV_MISSING));
    }

    Ok(GemmLowering {
        kernel,
        wanted: routing.plan,
        reason: routing.reason,
        fell_back,
    })
}

/// The recognised GEMM
type Pattern = wax_ir::passes::gemm_analysis::GemmPattern;

/// What a backend made of the kernel.
// A realised kernel carries its metallib, so the variants differ in size by design.
#[allow(clippy::large_enum_variant)]
enum Realised {
    /// Realised. `note` records a downgrade within the backend's own family.
    Kernel {
        kernel: crate::LoweredKernel,
        note: Option<&'static str>,
    },
    /// Not realised, with the reason to report as `fell_back`.
    Declined(&'static str),
}

const GEMV_MISSING: &str = "lowered as written. No N == 1 (gemv) realisation exists";
pub const NOT_A_GEMM: &str = "lowered as written. The kernel is not a recognised GEMM";
pub const SHAPE_DECLINED: &str = "lowered as written. GEMV specialisation declined this shape";
pub const SPLIT_INDIVISIBLE: &str =
    "K does not divide the SIMD group count per split. Single-pass GEMV chosen";

/// What the pipeline left on the IR.
#[derive(Debug)]
pub struct Rewritten {
    /// Launch geometry recorded in the IR.
    pub launch: Option<wax_ir::passes::Launch>,
    /// Secondary dispatch information of split-K. Implies [`Self::launch`].
    pub reduce: Option<wax_ir::passes::split_k::SplitKReduce>,
    /// Why a pass did not run. `None` means all passes ran.
    pub declined: Option<&'static str>,
}

impl Rewritten {
    /// Read off the IR what our relevant optimization passes recorded.
    fn observe(
        sg: i32,
        splits: i32,
        module: &wax_ir::Module,
        pattern: &Result<Pattern, &'static str>,
    ) -> Rewritten {
        use wax_ir::passes::{Launch, split_k::SplitKReduce};
        let nothing = Rewritten {
            launch: None,
            reduce: None,
            declined: None,
        };
        let Some(func) = module.entries().first().copied() else {
            return nothing;
        };
        // `sg == 0` implies the router did not request changes.
        let asked = sg >= 1;

        // Extract launch geometry from IR.
        let Some(launch) = Launch::of(&module.ctx, func) else {
            if !asked {
                return nothing;
            }
            let why = if pattern.is_err() {
                NOT_A_GEMM
            } else {
                SHAPE_DECLINED
            };
            return Rewritten {
                declined: Some(why),
                ..nothing
            };
        };
        let launch = Some(launch);
        if splits < 2 {
            return Rewritten { launch, ..nothing };
        }
        // A plan that asked to split either left a reduce or downgraded to one pass.
        match SplitKReduce::of(&module.ctx, module.module_op()) {
            Some(reduce) => Rewritten {
                launch,
                reduce: Some(reduce),
                declined: None,
            },
            None => Rewritten {
                launch,
                declined: Some(SPLIT_INDIVISIBLE),
                ..nothing
            },
        }
    }
}

/// The result of our pipeline. The updated module and what was decided.
pub struct Realisation {
    /// Rewritten copy. Original module untouched.
    pub module: wax_ir::Module,
    /// Entry function in [`Self::module`].
    pub func: wax_ir::OpId,
    /// GEMM pattern from IR analysis.
    pub pattern: Result<Pattern, &'static str>,
    /// Pipeline result.
    pub rewritten: Rewritten,
}

/// A copied module with the identified GEMM pattern.
struct Recognised {
    module: wax_ir::Module,
    func: wax_ir::OpId,
    pattern: Result<Pattern, &'static str>,
    analyses: AnalysisManager,
}

/// The spec must describe the tensors the module declares.
fn precondition(p: &Pattern, spec: &GemmSpec) -> Result<(), crate::LowerError> {
    use wax_ir::types::TensorViewType;

    if p.trans_b != spec.trans_b {
        return Err(crate::LowerError::Precondition(format!(
            "the module declares trans_b = {} and the spec declares {}",
            p.trans_b, spec.trans_b
        )));
    }
    let (m, n, k) = (spec.m as i64, spec.n as i64, spec.k as i64);
    let view = |shape: Vec<i64>, strides: Vec<i64>| TensorViewType {
        element_type: p.elem_t.clone(),
        shape,
        strides,
    };
    let implied = [
        (
            "A",
            if p.trans_a {
                view(vec![m, k], vec![1, p.lda])
            } else {
                view(vec![m, k], vec![p.lda, 1])
            },
        ),
        (
            "B",
            if p.trans_b {
                view(vec![k, n], vec![1, p.ldb])
            } else {
                view(vec![k, n], vec![p.ldb, 1])
            },
        ),
        ("C", view(vec![m, n], vec![p.ldc, 1])),
    ];
    for (name, tv) in implied {
        tv.verify_strides()
            .map_err(|e| crate::LowerError::Precondition(format!("{name}: {e}")))?;
    }
    Ok(())
}

fn recognise(module: &wax_ir::Module, spec: &GemmSpec) -> Result<Recognised, crate::LowerError> {
    // `working_copy` currently only supports a single entry fn.
    if module.entries().len() != 1 {
        return Err(crate::LowerError::MalformedIr(format!(
            "gemm lowering wants exactly one entry, got {}",
            module.entries().len()
        )));
    }
    let (owned, func) = wax_ir::passes::utils::working_copy(module)
        .ok_or_else(|| crate::LowerError::MalformedIr("module has no entry function".into()))?;
    let mut analyses = AnalysisManager::default();
    let pattern = analyses
        .get_analysis::<GemmAnalysis>(func, &owned.ctx)
        .map_err(|e| crate::LowerError::MalformedIr(format!("gemm analysis: {e:?}")))?
        .pattern
        .clone();
    if let Ok(p) = &pattern {
        precondition(p, spec)?;
    }
    Ok(Recognised {
        module: owned,
        func,
        pattern,
        analyses,
    })
}

/// Realise the module.
///
/// Recognises the GEMM pattern, applies optimization passes, extracts pass results from IR.
pub fn realise(
    module: &wax_ir::Module,
    sg: i32,
    splits: i32,
    spec: &GemmSpec,
) -> Result<Realisation, crate::LowerError> {
    use pliron::pass::{Pass, Passes};
    use wax_ir::passes::{gemm_analysis::GemmMarkPass, gemv_t, split_k};

    // Recognise
    let Recognised {
        module: mut owned,
        func,
        pattern,
        mut analyses,
    } = recognise(module, spec)?;

    // Apply passes
    let facts = gemv_t::Facts {
        m: spec.m as i64,
        n: spec.n as i64,
        sg,
        lanes: SIMD_WIDTH,
    };
    let mut on_entry = Passes::default();
    on_entry.add_pass(GemmMarkPass::default());
    on_entry.add_pass(gemv_t::GemvTPass::new(facts));
    on_entry.add_pass(split_k::SplitKPass::new(splits));
    let mut on_module = Passes::default();
    on_module.add_pass(split_k::SplitKReducePass::new(SIMD_WIDTH));

    let failed = |e| crate::LowerError::MalformedIr(format!("pass pipeline: {e:?}"));
    let module_op = pliron::op::Op::get_operation(&owned.module_op());
    on_entry
        .run(func, &mut owned.ctx, &mut analyses)
        .map_err(failed)?;
    on_module
        .run(module_op, &mut owned.ctx, &mut analyses)
        .map_err(failed)?;

    // Extract pass results from IR
    let rewritten = Rewritten::observe(sg, splits, &owned, &pattern);
    Ok(Realisation {
        module: owned,
        func,
        pattern,
        rewritten,
    })
}

/// Lower the optimized GEMM module
#[allow(clippy::too_many_arguments)]
fn lower(
    owned: &mut wax_ir::Module,
    func: wax_ir::OpId,
    pattern: &Result<Pattern, &'static str>,
    launch: Option<wax_ir::passes::Launch>,
    reduce: Option<&wax_ir::passes::split_k::SplitKReduce>,
    spec: &GemmSpec,
    profile: &TargetProfile,
) -> Result<crate::LoweredKernel, crate::LowerError> {
    // The two-pass form carries its own reduce dispatch and scratch, from the same metallib.
    let reduce_fn = reduce.and_then(|r| named_entry(owned, &r.entry));
    let (metallib, threads, reduce_threads) = match reduce_fn {
        None => {
            let l = crate::lower::lower_to_metallib_for(owned, profile)?;
            let primary = l.entry_threads(0);
            (l.metallib, primary, None)
        }
        Some(reduce_fn) => {
            let units = [entry_unit(owned, func)?, entry_unit(owned, reduce_fn)?];
            let l = crate::lower::lower_to_metallib_entries(owned, &units, profile)?;
            let (primary, secondary) = (l.entry_threads(0), l.entry_threads(1));
            (l.metallib, primary, secondary)
        }
    };
    let second = reduce
        .zip(reduce_fn)
        .map(|(r, refn)| crate::launch::SecondaryDispatch {
            entry: r.entry.clone(),
            launch: with_threads(
                crate::Launch::from_tile_blocks(r.block, &r.tile),
                reduce_threads,
            ),
            scalar_args: shape_scalars(&crate::lower::promoted_scalars_for(owned, refn), spec),
        });
    let grid = match launch {
        Some(l) => crate::Launch::from_tile_blocks(l.block, &l.tile),
        // `with_threads` updates the threadgroup if conversion has inferred threads, otherwise
        // we fall back to a single SIMD group.
        None => {
            let (bm, bn, _bk) = pattern
                .as_ref()
                .map(|p| (p.bm as usize, p.bn as usize, p.bk as usize))
                .unwrap_or_else(|_| module_tile(spec));
            let simd = SIMD_WIDTH as usize;
            crate::Launch::new((spec.m.div_ceil(bm), spec.n.div_ceil(bn), 1), (simd, 1, 1))
        }
    };
    let grid = if spec.batch > 1 {
        let (x, y, _) = grid.grid;
        crate::Launch::new((spec.batch, x, y), grid.threadgroup)
    } else {
        grid
    };
    let grid = with_threads(grid, threads);
    Ok(crate::LoweredKernel {
        metallib,
        entry: gemm_entry_name(owned, func),
        launch: grid,
        scratch_bytes: reduce.map(|r| r.scratch_elems * 4).unwrap_or(0),
        // Is a no-op if an optimization bakes the shape into the kernel, but is always required
        // for the original gemm kernel.
        scalar_args: shape_scalars(&crate::lower::promoted_scalars(owned), spec),
        second,
    })
}

/// Realises the MPP GEMM. `m` and `n` come from the spec, K loop and the strides from
/// the recognised kernel, and tile from the router.
fn realise_mpp(
    module: &wax_ir::Module,
    func: wax_ir::OpId,
    pattern: &Result<Pattern, &'static str>,
    spec: &GemmSpec,
    profile: &TargetProfile,
    (bm, bn, splits): (i32, i32, i32),
) -> Result<Realised, crate::LowerError> {
    let p = match pattern {
        Ok(p) => p,
        Err(why) => return Ok(Realised::Declined(why)),
    };
    // No convert in the epilogue means the f32 accumulator is the output.
    let out = match &p.epilogue {
        wax_ir::passes::gemm_analysis::Epilogue::None => p.accum_t.clone(),
        wax_ir::passes::gemm_analysis::Epilogue::Convert(to) => to.clone(),
    };
    let (Some(dtype), Some(out_dtype)) =
        (mpp_dtype_of(p.elem_t.clone()), mpp_dtype_of(out.clone()))
    else {
        return Ok(Realised::Declined("element type has no MPP equivalent"));
    };
    let entry = gemm_entry_name(module, func);
    let two_pass = splits > 1;
    if spec.batch > 1 && two_pass {
        return Ok(Realised::Declined("batch and split-k both want grid z"));
    }
    let cfg = crate::mpp::MppGemm {
        entry: entry.clone(),
        batch: spec.batch as i32,
        a_batch: spec.a_batch as i32,
        b_batch: spec.b_batch as i32,
        dtype,
        m: spec.m as i32,
        n: spec.n as i32,
        k: spec.k as i32,
        lda: p.lda as i32,
        ldb: p.ldb as i32,
        // First split-k step writes contiguous partials (`ldc == n`).
        // Second step, reduce, uses row stride `ldc`.
        ldc: if two_pass {
            spec.n as i32
        } else {
            p.ldc as i32
        },
        bm,
        bn,
        out_dtype: if two_pass {
            crate::mpp::MppDtype::F32
        } else {
            out_dtype
        },
        splits,
        trans_b: p.trans_b,
        trans_a: p.trans_a,
        simdgroups: 1,
    };
    if !two_pass {
        return Ok(Realised::Kernel {
            kernel: crate::mpp::lower_mpp_gemm_air(&cfg, profile)?,
            note: None,
        });
    }
    // One f32 `[m, n]` plane per split into the scratch. The shared reduce sums the planes into
    // the destination at row stride `ldc`, possibly narrowing to its type on the way.
    let reduce_entry = format!("{entry}__splitk_reduce");
    let mut reduce_module = wax_ir::Module::new("splitk_reduce");
    let reduce_module_op = reduce_module.module_op();
    let reduce = wax_ir::passes::split_k::build(
        &mut reduce_module.ctx,
        reduce_module_op,
        "splitk_reduce",
        &reduce_entry,
        &out,
        spec.m as i64,
        spec.n as i64,
        splits,
        spec.ldc as i64,
        SIMD_WIDTH,
    )
    .map_err(|e| crate::LowerError::MalformedIr(format!("split-k reduce: {e}")))?;

    let reduce_func = *reduce_module.entries().last().ok_or_else(|| {
        crate::LowerError::MalformedIr("split-k reduce: no entry function".into())
    })?;
    Ok(Realised::Kernel {
        kernel: crate::mpp::lower_mpp_gemm_splitk(
            &cfg,
            &mut reduce_module,
            reduce_func,
            &reduce_entry,
            crate::Launch::from_tile_blocks(reduce.block, &reduce.tile),
            profile,
        )?,
        note: None,
    })
}

/// Apply threads to launch threadgroup if present.
fn with_threads(launch: crate::Launch, threads: Option<usize>) -> crate::Launch {
    match threads {
        Some(t) => crate::Launch::new(launch.grid, (t, 1, 1)),
        None => launch,
    }
}

/// Binds the module's promoted shape scalars to the spec's dimensions.
fn shape_scalars(promoted: &[crate::lower::PromotedScalar], spec: &GemmSpec) -> Vec<i32> {
    promoted
        .iter()
        .map(|p| match (p.param, p.dim, spec.batch > 1) {
            (0, 0, true) => spec.batch as i32,
            (0, 1, true) => spec.m as i32,
            (0, 2, true) => spec.n as i32,
            (0, 0, false) => spec.m as i32,
            (0, 1, false) => spec.n as i32,
            _ => 0,
        })
        .collect()
}

/// Find function in module by name.
fn named_entry(module: &wax_ir::Module, name: &str) -> Option<wax_ir::OpId> {
    module
        .entries()
        .into_iter()
        .find(|f| wax_ir::build::symbol_name_in(&module.ctx, *f).as_deref() == Some(name))
}

/// An entry unit for a function a pass registered and named.
fn entry_unit(
    module: &wax_ir::Module,
    func: wax_ir::OpId,
) -> Result<crate::wax_to_air::EntryUnit, crate::LowerError> {
    let kernel_symbol = crate::wax_to_air::kernel_symbol(module, func).ok_or_else(|| {
        crate::LowerError::MalformedIr("entry function carries no kernel symbol".into())
    })?;
    let args = crate::lower::kernel_args_for_module(module, func).ok_or_else(|| {
        crate::LowerError::MalformedIr("entry function has no entry block".into())
    })?;
    Ok(crate::wax_to_air::EntryUnit {
        func,
        kernel_symbol,
        args,
    })
}

/// The entry symbol of the function being lowered.
fn gemm_entry_name(
    module: &wax_ir::Module,
    func: pliron::context::Ptr<pliron::operation::Operation>,
) -> String {
    wax_ir::build::symbol_name_in(&module.ctx, func).unwrap_or_else(|| "gemm".to_string())
}

#[cfg(test)]
mod module_tile_tests {
    use super::*;

    /// M = 1 (gemv_t) should always return `(1, 32, 1)`.
    #[test]
    fn gemv_t_guard() {
        for k in [1usize, 64, 2048, 8192] {
            assert_eq!(module_tile(&GemmSpec::new(1, 2, k, 3)), (1, 32, 1), "k={k}");
        }
    }

    #[test]
    fn should_equal_sgmma_selector() {
        for m in [2usize, 64, 2048, 8192] {
            let spec = GemmSpec::new(m, 2048, 4096, 4);
            assert_eq!(module_tile(&spec), sgmma_tile(&spec));
        }
    }

    #[test]
    fn indivisible_k_is_masked() {
        let spec = GemmSpec::new(128, 2048, 12, 2);
        assert_eq!(module_tile(&spec), sgmma_tile(&spec));
        // A K the deep block (32) would break is fine. `sgmma_tile` falls back to BK = 8.
        let spec = GemmSpec::new(128, 2048, 40, 2);
        let (bm, bn, _) = sgmma_tile(&spec);
        assert_eq!(module_tile(&spec), (bm, bn, 8), "40 % 8 == 0");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(m: usize, n: usize, k: usize, eb: usize) -> Routing {
        route(&GemmSpec::new(m, n, k, eb))
    }

    #[test]
    fn router_chooses_expected_sg() {
        // 4096 columns of f32 -> Not small. Not a 2-byte wide case. 8.
        let nn = GemmSpec::new(1, 4096, 2048, 4);
        assert_eq!(route(&nn).plan, Plan::Lower { sg: 8, splits: 1 });

        // Transposed long K: smallest count for grid `n / sg`.
        assert_eq!(route(&nn.trans_b()).plan, Plan::Lower { sg: 2, splits: 1 });

        // Transposed short K: column is too cheap to launch a small threadgroup for.
        // Use larger simdgroup count.
        let short_k = GemmSpec::new(1, 4096, 256, 2);
        assert_eq!(
            route(&short_k.trans_b()).plan,
            Plan::Lower { sg: 16, splits: 1 }
        );

        let small = GemmSpec::new(1, 2048, 2048, 2);
        let Plan::Lower { sg, .. } = route(&small).plan else {
            panic!("expected a realisation, got {:?}", route(&small).plan)
        };
        assert_eq!(sg, 32);
        assert_eq!(
            route(&small.trans_b()).plan,
            Plan::Lower { sg: 2, splits: 1 }
        );
    }

    /// Thin-N stays on SGMMA.
    #[test]
    fn tiny_indivisible_k_takes_the_bk1_tile() {
        for k in 1..=SGMMA_BK {
            let spec = GemmSpec::new(131072, 32, k, 2);
            let got = module_tile(&spec);
            // Either a block divides K, or the BK=1 tile takes it.
            assert!(
                k.is_multiple_of(got.2),
                "K={k} tile {got:?} drops part of the reduction"
            );
        }
    }

    #[test]
    fn large_indivisible_k_is_masked() {
        let spec = GemmSpec::new(2048, 2048, 1023, 2);
        assert_eq!(module_tile(&spec), sgmma_tile(&spec));
    }

    #[test]
    fn thin_n_stays_on_sgmma() {
        for (m, n) in [(4096, 8), (4096, 16), (4096, 32), (4096, 128)] {
            let got = r(m, n, 4096, 2);
            assert_eq!(
                got.plan,
                Plan::Lower { sg: 0, splits: 1 },
                "{m}x{n} must not use MPP"
            );
            assert_eq!(got.reason, Reason::ThinN);
        }
    }

    #[test]
    fn router_produces_expected_gemv_t_plans() {
        // N small: single dispatch, 32 SIMD groups, whatever the dtype.
        for eb in [2, 4] {
            assert_eq!(r(1, 2048, 4096, eb).plan, Plan::Lower { sg: 32, splits: 1 });
        }
        // N mid: the GEMV realisation at both dtypes.
        assert_eq!(r(1, 4096, 4096, 2).plan, Plan::Lower { sg: 8, splits: 1 });
        assert_eq!(r(1, 4096, 4096, 4).plan, Plan::Lower { sg: 8, splits: 1 });
        // N wide: no MPP at any dtype. Threadgroup reduction beats it.
        // f16/bf16 -> 16 groups, f32 -> 8.
        assert_eq!(r(1, 32000, 4096, 2).plan, Plan::Lower { sg: 16, splits: 1 });
        assert_eq!(r(1, 32000, 4096, 4).plan, Plan::Lower { sg: 8, splits: 1 });
        for got in [r(1, 2048, 4096, 2), r(1, 32000, 4096, 2)] {
            assert_eq!(got.reason, Reason::GemvT);
        }
    }

    #[test]
    fn gemv_t_split_k_supports_strided_dst() {
        let packed = GemmSpec::new(1, 512, 8192, 2);
        assert!(matches!(route(&packed).plan, Plan::Lower { .. }));
        let strided = GemmSpec::with_ldc(1, 512, 8192, 2, 8192);
        assert_eq!(route(&strided).plan, route(&packed).plan);
    }

    #[test]
    fn router_produces_expected_gemv_t_split_k_plans() {
        for eb in [2, 4] {
            assert_eq!(r(1, 512, 8192, eb).plan, Plan::Lower { sg: 32, splits: 16 },);
            assert_eq!(
                r(1, 1024, 8192, eb).plan,
                Plan::Lower { sg: 32, splits: 16 },
            );
        }
        // Below K floor second dispatch is not worth it.
        assert_eq!(r(1, 512, 2048, 2).plan, Plan::Lower { sg: 32, splits: 1 });
        // Above N ceiling single dispatch already saturates memory.
        assert_eq!(r(1, 2048, 8192, 2).plan, Plan::Lower { sg: 32, splits: 1 });
        // GemvT Split-K supports strided dst and produces expected splits.
        let strided = GemmSpec::with_ldc(1, 512, 8192, 2, 1024);
        assert_eq!(route(&strided).plan, Plan::Lower { sg: 32, splits: 16 });
    }

    #[test]
    fn router_adheres_to_mpp_tile_width_limit() {
        for &m in &[1usize, 7, 64, 331, 1024, 4096] {
            for &n in &[32usize, 64, 512, 4096, 16384, 32000, 128256] {
                for &k in &[512usize, 4096, 11008] {
                    for &eb in &[2usize, 4] {
                        let bn = match r(m, n, k, eb).plan {
                            Plan::Mpp { bn, .. } => bn,
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

    #[test]
    fn gemv_and_sub_chunk_n() {
        let got = r(4096, 1, 4096, 2);
        assert_eq!(got.plan, Plan::Lower { sg: 0, splits: 1 });
        assert_eq!(got.reason, Reason::Gemv);
        // `M = 1` with fewer than 32 columns has no full chunk. It must not claim GemvT.
        assert_eq!(r(1, 16, 4096, 2).plan, Plan::Lower { sg: 0, splits: 1 });
    }

    #[test]
    fn large_working_set_splits_k_instead_of_falling_back() {
        let fp32 = r(4096, 4096, 11008, 4);
        assert_eq!(
            fp32.plan,
            Plan::Mpp {
                bm: 32,
                bn: 32,
                splits: 4
            }
        );
        assert_eq!(fp32.reason, Reason::SplitWorkingSet);
        // f16/bf16 at the same K is under the budget and needs no split.
        let bf16 = r(4096, 4096, 11008, 2);
        assert_eq!(
            bf16.plan,
            Plan::Mpp {
                bm: 32,
                bn: 32,
                splits: 1
            }
        );
        // large working set but also below work floor -> fall back.
        let small = r(192, 192, 20000, 4);
        assert_eq!(small.plan, Plan::Lower { sg: 0, splits: 1 });
        assert_eq!(small.reason, Reason::WorkingSetTooLarge);
    }

    #[test]
    fn split_k_supports_strided_dst() {
        let packed = route(&GemmSpec::new(4096, 4096, 11008, 4));
        assert!(matches!(packed.plan, Plan::Mpp { .. }));

        let strided = route(&GemmSpec::with_ldc(4096, 4096, 11008, 4, 4224));
        assert_eq!(strided.plan, packed.plan);
        assert_eq!(strided.reason, packed.reason);
    }

    #[test]
    fn ragged_shapes_are_handled_correctly() {
        // Small aligned squares in f16/bf16 run on SGMMA.
        let small = r(256, 256, 256, 2);
        assert_eq!(small.plan, Plan::Lower { sg: 0, splits: 1 });
        assert_eq!(small.reason, Reason::NotEnoughWork);

        // The ragged neighbour shape does not.
        let ragged = r(257, 257, 257, 2);
        assert!(ragged.plan.is_mpp(), "ragged must use MPP at any size");
        assert_eq!(ragged.reason, Reason::Ragged);

        // Neither does small aligned squares in f32
        let small = r(256, 256, 256, 4);
        assert!(small.plan.is_mpp(), "small aligned f32 squares use MPP");
        assert_eq!(small.reason, Reason::EnoughWork);
    }

    #[test]
    fn no_split_k_on_tiny_working_set() {
        let got = r(4096, 4096, 64, 2);
        assert_eq!(
            got.plan,
            Plan::Mpp {
                bm: 32,
                bn: 32,
                splits: 1
            }
        );
        assert_eq!(got.reason, Reason::EnoughWork);
    }

    #[test]
    fn few_tiles_do_not_split_k() {
        for (m, n, k) in [(1024, 1024, 1024), (1023, 1023, 1024), (511, 511, 512)] {
            let got = r(m, n, k, 2);
            assert!(
                matches!(got.plan, Plan::Mpp { .. }),
                "{m}x{n}x{k} must be single-pass MPP, got {:?}",
                got.plan
            );
        }
    }

    #[test]
    fn large_square_is_single_pass_mpp() {
        let got = r(4096, 4096, 4096, 2);
        assert_eq!(
            got.plan,
            Plan::Mpp {
                bm: 32,
                bn: 32,
                splits: 1
            }
        );
        assert_eq!(got.reason, Reason::EnoughWork);
    }

    /// The tile rule, pinned to the measurements in [`mpp_tile`].
    #[test]
    fn tile_follows_occupancy() {
        let t = |m, n, k, eb| mpp_tile(&GemmSpec::new(m, n, k, eb));
        // aligned small fp32 squares take 16x16.
        assert_eq!(t(128, 128, 128, 4), (16, 16));
        assert_eq!(t(256, 256, 256, 4), (16, 16));
        // ragged neighbours do not.
        assert_eq!(t(257, 257, 257, 4), (16, 32));
        assert_eq!(t(511, 511, 511, 4), (16, 32));
        assert_eq!(t(333, 444, 555, 4), (16, 32));
        // f16/bf16 never takes 16x16
        assert_eq!(t(256, 256, 256, 2), (16, 32));
        // squarish low-occupancy at 2 bytes
        assert_eq!(t(1024, 1024, 1024, 2), (16, 32));
        assert_eq!(t(1024, 1024, 1024, 4), (32, 32));
        // plenty of threadgroups -> the big tile's reuse wins again
        assert_eq!(t(4096, 4096, 4096, 2), (32, 32));
        assert_eq!(t(2048, 2048, 2048, 2), (32, 32));
    }

    #[test]
    fn small_squares_are_dtype_split() {
        let f32_small = r(256, 256, 256, 4);
        assert_eq!(
            f32_small.plan,
            Plan::Mpp {
                bm: 16,
                bn: 16,
                splits: 1
            }
        );
        assert_eq!(f32_small.reason, Reason::EnoughWork);

        let bf16_small = r(256, 256, 256, 2);
        assert_eq!(bf16_small.plan, Plan::Lower { sg: 0, splits: 1 });
        assert_eq!(bf16_small.reason, Reason::NotEnoughWork);

        let f32_tiny = r(128, 128, 128, 4);
        assert_eq!(
            f32_tiny.plan,
            Plan::Mpp {
                bm: 16,
                bn: 16,
                splits: 1
            }
        );

        // The bypassed floor must hold for everything that is not a small aligned f32 square.
        assert_eq!(r(4096, 128, 4096, 4).reason, Reason::ThinN);
        assert_eq!(r(128, 4096, 4096, 4).reason, Reason::ThinM);
        assert_eq!(r(128, 128, 128, 2).reason, Reason::ThinN);
        assert_eq!(r(129, 129, 128, 4).reason, Reason::ThinN);
        assert_eq!(r(384, 384, 384, 4).reason, Reason::EnoughWork);
    }
}

#[cfg(test)]
mod sgmma_tile_tests {
    use super::*;

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

    #[test]
    fn large_fp32_uses_split_k() {
        for &(m, n, k) in &[
            (4096, 11008, 4096),
            (4096, 4096, 4096),
            (2048, 2048, 8192),
            (512, 2048, 8192),
        ] {
            let r = route4(m, n, k);
            assert!(
                matches!(r.plan, Plan::Mpp { .. }),
                "{m}x{n}x{k} backend: {:?}",
                r.plan
            );
            assert_eq!(r.reason, Reason::Fp32WorkingSet, "{m}x{n}x{k} reason");
        }
    }

    #[test]
    fn fp32_clause_splits_regardless_of_work() {
        // Small work, and split-K still wins.
        let r = route4(256, 256, 8192);
        assert_eq!(r.reason, Reason::Fp32WorkingSet);
        assert!(matches!(r.plan, Plan::Mpp { splits: SPLITS, .. }));

        // `is_aligned` -> `k % 8 == 0`, so `SPLITS` always divides K here.
        for &(m, n, k) in &[(192, 192, 8192), (384, 384, 4096), (224, 448, 8192)] {
            assert!(k % SPLITS as usize == 0, "{m}x{n}x{k}");
            assert!(
                matches!(route4(m, n, k).plan, Plan::Mpp { .. }),
                "{m}x{n}x{k}"
            );
        }

        // Strided dst splits correctly
        let r = route(&GemmSpec::with_ldc(2048, 2048, 8192, 4, 4096));
        assert_eq!(r.reason, Reason::Fp32WorkingSet);
        assert!(matches!(r.plan, Plan::Mpp { .. }));
    }

    #[test]
    fn small_fp32_streams_stay_on_mpp() {
        for &(m, n, k) in &[(2048, 2048, 2048), (512, 512, 2048), (1024, 1024, 1024)] {
            let r = route4(m, n, k);
            assert!(
                r.plan.is_mpp(),
                "{m}x{n}x{k} must stay on MPP, got {:?} via {:?}",
                r.plan,
                r.reason
            );
        }
    }

    #[test]
    fn split_k_still_wins_the_biggest_streams() {
        let r = route4(4096, 4096, 11008);
        assert!(
            matches!(r.plan, Plan::Mpp { .. }),
            "expected split-K, got {:?} via {:?}",
            r.plan,
            r.reason
        );
        assert_eq!(r.reason, Reason::SplitWorkingSet);
    }

    #[test]
    fn half_and_ragged_unaffected_by_fp32_clause() {
        // Half precision
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

        // Ragged
        let r = route(&GemmSpec::new(2044, 2044, 8188, 4));
        assert_ne!(r.reason, Reason::Fp32WorkingSet);
    }
}

#[cfg(test)]
mod mpp_deep_k_tile_tests {
    use super::*;

    #[test]
    fn deep_k_fp32_takes_the_full_tile() {
        assert_eq!(mpp_tile(&GemmSpec::new(512, 512, 2048, 4)), MPP_TILE);
    }

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

    #[test]
    fn deep_k_but_tiny_grid_is_excluded() {
        assert_eq!(mpp_tile(&GemmSpec::new(256, 256, 2048, 4)), MPP_TILE_TINY);
    }

    #[test]
    fn half_and_ragged_use_expected_mpp_tile() {
        assert_ne!(mpp_tile(&GemmSpec::new(512, 512, 2048, 2)), MPP_TILE);
        assert_eq!(mpp_tile(&GemmSpec::new(500, 500, 2048, 4)), MPP_TILE_SMALL);
    }
}

#[cfg(test)]
mod clamped_dtype_tests {
    use super::*;

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

    #[test]
    fn thin_shapes_are_unaffected() {
        assert_eq!(
            sgmma_tile_clamped(&GemmSpec::new(4096, 32, 4096, 4)),
            (16, 16)
        );
        assert_eq!(
            sgmma_tile_clamped(&GemmSpec::new(4096, 64, 4096, 4)),
            (8, 32)
        );
    }

    #[test]
    fn clamped_tile_fits_the_problem() {
        for &(m, n) in &[(8usize, 8usize), (24, 40), (100, 50), (4096, 8)] {
            let (bm, bn) = sgmma_tile_clamped(&GemmSpec::new(m, n, 2048, 4));
            assert!(bm <= m.max(8) && bn <= n.max(8), "{m}x{n} -> {bm}x{bn}");
        }
    }
}
