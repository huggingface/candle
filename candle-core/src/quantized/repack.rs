#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
use super::GgmlDType;
use super::QuantizedType;
use crate::Result;
#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
use std::sync::OnceLock;

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
use super::k_quants::{BlockQ4K, BlockQ4_0, QK4_0};
#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
use super::k_quants::{BlockQ5K, BlockQ6K, QK_K};
#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
use super::k_quants::{BlockQ8_0, QK8_0};

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
enum PackedKind {
    Q4_0x4,
    Q4Kx8,
    Q5Kx8,
    Q6Kx8,
    Q8_0x4,
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
pub(crate) enum PackedStorage {
    Q4_0x4(Vec<BlockQ4_0x4>),
    Q4Kx8(Vec<BlockQ4Kx8>),
    Q5Kx8(Vec<BlockQ5Kx8>),
    Q6Kx8(Vec<BlockQ6Kx8>),
    Q8_0x4(Vec<BlockQ8_0x4>),
}

#[allow(dead_code)]
pub(crate) struct PackedCache {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_feature = "dotprod", target_feature = "i8mm")
    ))]
    q4_0x4: OnceLock<PackedStorage>,
    #[cfg(all(
        target_arch = "aarch64",
        any(target_feature = "dotprod", target_feature = "i8mm")
    ))]
    q4kx8: OnceLock<PackedStorage>,
    #[cfg(all(
        target_arch = "aarch64",
        any(target_feature = "dotprod", target_feature = "i8mm")
    ))]
    q5kx8: OnceLock<PackedStorage>,
    #[cfg(all(
        target_arch = "aarch64",
        any(target_feature = "dotprod", target_feature = "i8mm")
    ))]
    q6kx8: OnceLock<PackedStorage>,
    #[cfg(all(
        target_arch = "aarch64",
        any(target_feature = "dotprod", target_feature = "i8mm")
    ))]
    q8_0x4: OnceLock<PackedStorage>,
}

impl PackedCache {
    pub(crate) fn new() -> Self {
        Self {
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            q4_0x4: OnceLock::new(),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            q4kx8: OnceLock::new(),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            q5kx8: OnceLock::new(),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            q6kx8: OnceLock::new(),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            q8_0x4: OnceLock::new(),
        }
    }

    #[cfg(all(
        target_arch = "aarch64",
        any(target_feature = "dotprod", target_feature = "i8mm")
    ))]
    fn get_or_init(
        &self,
        kind: PackedKind,
        init: impl FnOnce() -> PackedStorage,
    ) -> &PackedStorage {
        match kind {
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            PackedKind::Q4_0x4 => self.q4_0x4.get_or_init(init),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            PackedKind::Q4Kx8 => self.q4kx8.get_or_init(init),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            PackedKind::Q5Kx8 => self.q5kx8.get_or_init(init),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            PackedKind::Q6Kx8 => self.q6kx8.get_or_init(init),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            PackedKind::Q8_0x4 => self.q8_0x4.get_or_init(init),
        }
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
#[derive(
    Clone,
    Copy,
    zerocopy::FromBytes,
    zerocopy::IntoBytes,
    zerocopy::KnownLayout,
    zerocopy::Immutable,
)]
#[repr(C)]
pub(crate) struct BlockQ8_0x4 {
    pub(crate) d: [f16; 4],
    pub(crate) qs: [i8; QK8_0 * 4],
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
const _: () = assert!(std::mem::size_of::<BlockQ8_0x4>() == 136);

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
#[derive(
    Clone,
    Copy,
    zerocopy::FromBytes,
    zerocopy::IntoBytes,
    zerocopy::KnownLayout,
    zerocopy::Immutable,
)]
#[repr(C)]
pub(crate) struct BlockQ4_0x4 {
    pub(crate) d: [f16; 4],
    pub(crate) qs: [i8; QK4_0 * 2],
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
const _: () = assert!(std::mem::size_of::<BlockQ4_0x4>() == 72);

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
#[derive(
    Clone,
    Copy,
    zerocopy::FromBytes,
    zerocopy::IntoBytes,
    zerocopy::KnownLayout,
    zerocopy::Immutable,
)]
#[repr(C)]
pub(crate) struct BlockQ4Kx8 {
    pub(crate) d: [f16; 8],
    pub(crate) dmin: [f16; 8],
    pub(crate) scales: [u8; 96],
    pub(crate) qs: [u8; 1024],
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
const _: () = assert!(std::mem::size_of::<BlockQ4Kx8>() == 1152);

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
#[derive(
    Clone,
    Copy,
    zerocopy::FromBytes,
    zerocopy::IntoBytes,
    zerocopy::KnownLayout,
    zerocopy::Immutable,
)]
#[repr(C)]
pub(crate) struct BlockQ5Kx8 {
    pub(crate) d: [f16; 8],
    pub(crate) dmin: [f16; 8],
    pub(crate) scales: [u8; 96],
    pub(crate) qh: [u8; QK_K],
    pub(crate) qs: [u8; QK_K / 2 * 8],
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
const _: () = assert!(std::mem::size_of::<BlockQ5Kx8>() == 1408);

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
#[derive(
    Clone,
    Copy,
    zerocopy::FromBytes,
    zerocopy::IntoBytes,
    zerocopy::KnownLayout,
    zerocopy::Immutable,
)]
#[repr(C)]
pub(crate) struct BlockQ6Kx8 {
    pub(crate) d: [f16; 8],
    pub(crate) scales: [i8; QK_K / 16 * 8],
    pub(crate) ql: [u8; QK_K / 2 * 8],
    pub(crate) qh: [u8; QK_K / 4 * 8],
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
const _: () = assert!(std::mem::size_of::<BlockQ6Kx8>() == 1680);

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
#[derive(
    Clone,
    Copy,
    zerocopy::FromBytes,
    zerocopy::IntoBytes,
    zerocopy::KnownLayout,
    zerocopy::Immutable,
)]
#[repr(C)]
pub(crate) struct BlockQ8Kx4 {
    pub(crate) d: [f32; 4],
    pub(crate) qs: [i8; QK_K * 4],
    pub(crate) bsums: [i16; QK_K / 4],
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
const _: () = assert!(std::mem::size_of::<BlockQ8Kx4>() == 1168);

pub(crate) fn try_matmul_f32(
    storage: &dyn QuantizedType,
    packed: &PackedCache,
    mkn: (usize, usize, usize),
    lhs: &[f32],
    dst: &mut [f32],
) -> Result<bool> {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_feature = "dotprod", target_feature = "i8mm")
    ))]
    {
        let Some(kind) = PackedKind::select(storage.dtype(), mkn) else {
            return Ok(false);
        };
        let packed = packed.get_or_init(kind, || kind.pack(storage, mkn.2));
        kind.matmul(mkn, lhs, packed, dst)?;
        Ok(true)
    }

    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_feature = "dotprod", target_feature = "i8mm")
    )))]
    {
        let _ = (storage, packed, mkn, lhs, dst);
        Ok(false)
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
impl PackedKind {
    fn select(dtype: GgmlDType, (m, _k, n): (usize, usize, usize)) -> Option<Self> {
        match dtype {
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            GgmlDType::Q4_0 if n.is_multiple_of(4) && m == 1 => Some(Self::Q4_0x4),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            GgmlDType::Q4_0 if n.is_multiple_of(4) && m >= 4 && m.is_multiple_of(4) => {
                Some(Self::Q4_0x4)
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_feature = "i8mm",
                target_feature = "dotprod"
            ))]
            GgmlDType::Q4K
                if n.is_multiple_of(8) && (m == 1 || (m >= 4 && m.is_multiple_of(4))) =>
            {
                Some(Self::Q4Kx8)
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_feature = "i8mm",
                not(target_feature = "dotprod")
            ))]
            GgmlDType::Q4K if n.is_multiple_of(8) && m >= 4 && m.is_multiple_of(4) => {
                Some(Self::Q4Kx8)
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_feature = "dotprod",
                not(target_feature = "i8mm")
            ))]
            GgmlDType::Q4K if n.is_multiple_of(8) => Some(Self::Q4Kx8),
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            GgmlDType::Q5K if n.is_multiple_of(8) && m == 1 => Some(Self::Q5Kx8),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            GgmlDType::Q5K if n.is_multiple_of(8) && m >= 4 && m.is_multiple_of(4) => {
                Some(Self::Q5Kx8)
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            GgmlDType::Q6K if n.is_multiple_of(8) && m == 1 => Some(Self::Q6Kx8),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            GgmlDType::Q6K if n.is_multiple_of(8) && m >= 4 && m.is_multiple_of(4) => {
                Some(Self::Q6Kx8)
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            GgmlDType::Q8_0 if n.is_multiple_of(4) && m == 1 => Some(Self::Q8_0x4),
            #[cfg(all(target_arch = "aarch64", target_feature = "i8mm"))]
            GgmlDType::Q8_0 if n.is_multiple_of(4) && m >= 4 && m.is_multiple_of(4) => {
                Some(Self::Q8_0x4)
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_feature = "dotprod",
                not(target_feature = "i8mm")
            ))]
            GgmlDType::Q8_0 if n.is_multiple_of(4) && m >= 4 && m.is_multiple_of(4) => {
                Some(Self::Q8_0x4)
            }
            _ => None,
        }
    }

    fn pack(self, storage: &dyn QuantizedType, n: usize) -> PackedStorage {
        match self {
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            Self::Q4_0x4 => {
                let total_blocks =
                    storage.storage_size_in_bytes() / std::mem::size_of::<BlockQ4_0>();
                let blocks = unsafe {
                    std::slice::from_raw_parts(storage.as_ptr() as *const BlockQ4_0, total_blocks)
                };
                #[cfg(target_feature = "i8mm")]
                {
                    PackedStorage::Q4_0x4(pack_to_q4_0x4(blocks, n, 8))
                }
                #[cfg(not(target_feature = "i8mm"))]
                {
                    PackedStorage::Q4_0x4(pack_to_q4_0x4(blocks, n, 4))
                }
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            Self::Q4Kx8 => {
                let total_blocks =
                    storage.storage_size_in_bytes() / std::mem::size_of::<BlockQ4K>();
                let blocks = unsafe {
                    std::slice::from_raw_parts(storage.as_ptr() as *const BlockQ4K, total_blocks)
                };
                PackedStorage::Q4Kx8(pack_to_q4kx8(blocks, n))
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            Self::Q5Kx8 => {
                let total_blocks =
                    storage.storage_size_in_bytes() / std::mem::size_of::<BlockQ5K>();
                let blocks = unsafe {
                    std::slice::from_raw_parts(storage.as_ptr() as *const BlockQ5K, total_blocks)
                };
                #[cfg(target_feature = "i8mm")]
                {
                    PackedStorage::Q5Kx8(pack_to_q5kx8(blocks, n, 8))
                }
                #[cfg(not(target_feature = "i8mm"))]
                {
                    PackedStorage::Q5Kx8(pack_to_q5kx8(blocks, n, 4))
                }
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            Self::Q6Kx8 => {
                let total_blocks =
                    storage.storage_size_in_bytes() / std::mem::size_of::<BlockQ6K>();
                let blocks = unsafe {
                    std::slice::from_raw_parts(storage.as_ptr() as *const BlockQ6K, total_blocks)
                };
                #[cfg(target_feature = "i8mm")]
                {
                    PackedStorage::Q6Kx8(pack_to_q6kx8(blocks, n, 8))
                }
                #[cfg(not(target_feature = "i8mm"))]
                {
                    PackedStorage::Q6Kx8(pack_to_q6kx8(blocks, n, 4))
                }
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            Self::Q8_0x4 => {
                let total_blocks =
                    storage.storage_size_in_bytes() / std::mem::size_of::<BlockQ8_0>();
                let blocks = unsafe {
                    std::slice::from_raw_parts(storage.as_ptr() as *const BlockQ8_0, total_blocks)
                };
                #[cfg(target_feature = "i8mm")]
                {
                    PackedStorage::Q8_0x4(pack_to_q8_0x4(blocks, n, 8))
                }
                #[cfg(all(target_feature = "dotprod", not(target_feature = "i8mm")))]
                {
                    PackedStorage::Q8_0x4(pack_to_q8_0x4(blocks, n, 4))
                }
            }
        }
    }

    fn matmul(
        self,
        mkn: (usize, usize, usize),
        lhs: &[f32],
        packed: &PackedStorage,
        dst: &mut [f32],
    ) -> Result<()> {
        match (self, packed) {
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            (Self::Q4_0x4, PackedStorage::Q4_0x4(packed)) => {
                #[cfg(target_feature = "i8mm")]
                {
                    super::neon::matmul_q4_0_x4_i8mm(mkn, lhs, packed, dst)
                }
                #[cfg(not(target_feature = "i8mm"))]
                {
                    super::neon::matmul_q4_0_x4(mkn, lhs, packed, dst)
                }
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            (Self::Q4Kx8, PackedStorage::Q4Kx8(packed)) => {
                #[cfg(all(target_feature = "i8mm", target_feature = "dotprod"))]
                {
                    if mkn.0 == 1 {
                        super::k_quants::matmul_q4k_x8(mkn, lhs, packed, dst)
                    } else {
                        super::neon::matmul_q4k_x8_i8mm(mkn, lhs, packed, dst)
                    }
                }
                #[cfg(all(target_feature = "i8mm", not(target_feature = "dotprod")))]
                {
                    super::neon::matmul_q4k_x8_i8mm(mkn, lhs, packed, dst)
                }
                #[cfg(all(target_feature = "dotprod", not(target_feature = "i8mm")))]
                {
                    super::k_quants::matmul_q4k_x8(mkn, lhs, packed, dst)
                }
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            (Self::Q5Kx8, PackedStorage::Q5Kx8(packed)) => {
                super::neon::matmul_q5k_x8(mkn, lhs, packed, dst)
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_feature = "dotprod", target_feature = "i8mm")
            ))]
            (Self::Q6Kx8, PackedStorage::Q6Kx8(packed)) => {
                super::neon::matmul_q6k_x8(mkn, lhs, packed, dst)
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "i8mm"))]
            (Self::Q8_0x4, PackedStorage::Q8_0x4(packed)) => {
                super::neon::matmul_q8_0_x4_i8mm(mkn, lhs, packed, dst)
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_feature = "dotprod",
                not(target_feature = "i8mm")
            ))]
            (Self::Q8_0x4, PackedStorage::Q8_0x4(packed)) => {
                super::neon::matmul_q8_0_x4(mkn, lhs, packed, dst)
            }
            #[allow(unreachable_patterns)]
            _ => crate::bail!("packed storage kind mismatch"),
        }
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
fn pack_to_q4kx8(blocks: &[BlockQ4K], n: usize) -> Vec<BlockQ4Kx8> {
    debug_assert!(n.is_multiple_of(8));
    debug_assert_eq!(blocks.len() % n, 0);
    let k_blocks = blocks.len() / n;
    let n_groups = n / 8;
    let mut packed = Vec::with_capacity(n_groups * k_blocks);
    for g in 0..n_groups {
        for b in 0..k_blocks {
            let src: [&BlockQ4K; 8] = std::array::from_fn(|i| &blocks[(g * 8 + i) * k_blocks + b]);
            let mut p = BlockQ4Kx8 {
                d: [f16::ZERO; 8],
                dmin: [f16::ZERO; 8],
                scales: [0; 96],
                qs: [0; 1024],
            };
            for (i, s) in src.iter().enumerate() {
                p.d[i] = s.d;
                p.dmin[i] = s.dmin;
            }
            for i in 0..128usize {
                let src_id = i % 8;
                let src_offset = (i / 8) * 8;
                let dst_offset = i * 8;
                p.qs[dst_offset..dst_offset + 8]
                    .copy_from_slice(&src[src_id].qs[src_offset..src_offset + 8]);
            }
            pack_k_scale_min_x8(&src.map(|s| &s.scales), &mut p.scales);
            packed.push(p);
        }
    }
    packed
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
fn pack_to_q5kx8(blocks: &[BlockQ5K], n: usize, block_len: usize) -> Vec<BlockQ5Kx8> {
    debug_assert!(n.is_multiple_of(8));
    debug_assert!((QK_K / 2).is_multiple_of(block_len));
    debug_assert!((QK_K / 8).is_multiple_of(block_len));
    debug_assert_eq!(blocks.len() % n, 0);
    let k_blocks = blocks.len() / n;
    let n_groups = n / 8;
    let mut packed = Vec::with_capacity(n_groups * k_blocks);
    for g in 0..n_groups {
        for b in 0..k_blocks {
            let src: [&BlockQ5K; 8] = std::array::from_fn(|i| &blocks[(g * 8 + i) * k_blocks + b]);
            let mut p = BlockQ5Kx8 {
                d: [f16::ZERO; 8],
                dmin: [f16::ZERO; 8],
                scales: [0; 96],
                qh: [0; QK_K],
                qs: [0; QK_K / 2 * 8],
            };
            for (i, s) in src.iter().enumerate() {
                p.d[i] = s.d;
                p.dmin[i] = s.dmin;
            }
            for i in 0..(QK_K / 2 * 8) / block_len {
                let src_id = i % 8;
                let src_offset = (i / 8) * block_len;
                let dst_offset = i * block_len;
                p.qs[dst_offset..dst_offset + block_len]
                    .copy_from_slice(&src[src_id].qs[src_offset..src_offset + block_len]);
            }
            for i in 0..(QK_K / 8 * 8) / block_len {
                let src_id = i % 8;
                let src_offset = (i / 8) * block_len;
                let dst_offset = i * block_len;
                p.qh[dst_offset..dst_offset + block_len]
                    .copy_from_slice(&src[src_id].qh[src_offset..src_offset + block_len]);
            }
            pack_k_scale_min_x8(&src.map(|s| &s.scales), &mut p.scales);
            packed.push(p);
        }
    }
    packed
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
fn pack_to_q6kx8(blocks: &[BlockQ6K], n: usize, block_len: usize) -> Vec<BlockQ6Kx8> {
    debug_assert!(n.is_multiple_of(8));
    debug_assert!((QK_K / 2).is_multiple_of(block_len));
    debug_assert!((QK_K / 4).is_multiple_of(block_len));
    debug_assert_eq!(blocks.len() % n, 0);
    let k_blocks = blocks.len() / n;
    let n_groups = n / 8;
    let mut packed = Vec::with_capacity(n_groups * k_blocks);
    for g in 0..n_groups {
        for b in 0..k_blocks {
            let src: [&BlockQ6K; 8] = std::array::from_fn(|i| &blocks[(g * 8 + i) * k_blocks + b]);
            let mut p = BlockQ6Kx8 {
                d: [f16::ZERO; 8],
                scales: [0; QK_K / 16 * 8],
                ql: [0; QK_K / 2 * 8],
                qh: [0; QK_K / 4 * 8],
            };
            for (i, s) in src.iter().enumerate() {
                p.d[i] = s.d;
            }
            for i in 0..(QK_K / 2 * 8) / block_len {
                let src_id = i % 8;
                let src_offset = (i / 8) * block_len;
                let dst_offset = i * block_len;
                p.ql[dst_offset..dst_offset + block_len]
                    .copy_from_slice(&src[src_id].ql[src_offset..src_offset + block_len]);
            }
            for i in 0..(QK_K / 4 * 8) / block_len {
                let src_id = i % 8;
                let src_offset = (i / 8) * block_len;
                let dst_offset = i * block_len;
                p.qh[dst_offset..dst_offset + block_len]
                    .copy_from_slice(&src[src_id].qh[src_offset..src_offset + block_len]);
            }
            for (col, s) in src.iter().enumerate() {
                for j in 0..QK_K / 16 {
                    p.scales[j * 8 + col] = s.scales[j];
                }
            }
            packed.push(p);
        }
    }
    packed
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
fn pack_k_scale_min_x8(src: &[&[u8; 12]; 8], dst: &mut [u8; 96]) {
    for i in 0..4usize {
        let mut s = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            s[j] = src[j][i] & 63;
            m[j] = src[j][i + 4] & 63;
        }
        let b12 = i * 12;
        dst[b12] = (s[0] & 63) + ((s[4] & 48) << 2);
        dst[b12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        dst[b12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        dst[b12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        dst[b12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        dst[b12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        dst[b12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        dst[b12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        dst[b12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        dst[b12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        dst[b12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        dst[b12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    for i in 0..4usize {
        let mut s = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            s[j] = ((src[j][i] & 192) >> 2) | (src[j][i + 8] & 15);
            m[j] = ((src[j][i + 4] & 192) >> 2) | ((src[j][i + 8] & 240) >> 4);
        }
        let b12 = i * 12 + 48;
        dst[b12] = (s[0] & 63) + ((s[4] & 48) << 2);
        dst[b12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        dst[b12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        dst[b12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        dst[b12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        dst[b12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        dst[b12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        dst[b12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        dst[b12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        dst[b12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        dst[b12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        dst[b12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
fn pack_to_q4_0x4(blocks: &[BlockQ4_0], n: usize, block_len: usize) -> Vec<BlockQ4_0x4> {
    debug_assert!(n.is_multiple_of(4));
    debug_assert!((QK4_0 / 2).is_multiple_of(block_len));
    debug_assert_eq!(blocks.len() % n, 0);
    let k_blocks = blocks.len() / n;
    let n_groups = n / 4;
    let mut packed = Vec::with_capacity(n_groups * k_blocks);
    for g in 0..n_groups {
        for b in 0..k_blocks {
            let src: [&BlockQ4_0; 4] = std::array::from_fn(|i| &blocks[(g * 4 + i) * k_blocks + b]);
            let mut p = BlockQ4_0x4 {
                d: [f16::ZERO; 4],
                qs: [0; QK4_0 * 2],
            };
            for (i, s) in src.iter().enumerate() {
                p.d[i] = s.d;
            }
            for i in 0..(QK4_0 / 2) / block_len {
                let offset = i * block_len;
                let dst_offset = i * block_len * 4;
                for (j, s) in src.iter().enumerate() {
                    let dst_offset = dst_offset + j * block_len;
                    for k in 0..block_len {
                        p.qs[dst_offset + k] = (s.qs[offset + k] ^ 0x88) as i8;
                    }
                }
            }
            packed.push(p);
        }
    }
    packed
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_feature = "dotprod", target_feature = "i8mm")
))]
fn pack_to_q8_0x4(blocks: &[BlockQ8_0], n: usize, block_len: usize) -> Vec<BlockQ8_0x4> {
    debug_assert!(n.is_multiple_of(4));
    debug_assert!(QK8_0.is_multiple_of(block_len));
    debug_assert_eq!(blocks.len() % n, 0);
    let k_blocks = blocks.len() / n;
    let n_groups = n / 4;
    let mut packed = Vec::with_capacity(n_groups * k_blocks);
    for g in 0..n_groups {
        for b in 0..k_blocks {
            let src: [&BlockQ8_0; 4] = std::array::from_fn(|i| &blocks[(g * 4 + i) * k_blocks + b]);
            let mut p = BlockQ8_0x4 {
                d: [f16::ZERO; 4],
                qs: [0; QK8_0 * 4],
            };
            for (i, s) in src.iter().enumerate() {
                p.d[i] = s.d;
            }
            for i in 0..QK8_0 / block_len {
                let offset = i * block_len;
                let dst_offset = i * block_len * 4;
                for (j, s) in src.iter().enumerate() {
                    let dst_offset = dst_offset + j * block_len;
                    p.qs[dst_offset..dst_offset + block_len]
                        .copy_from_slice(&s.qs[offset..offset + block_len]);
                }
            }
            packed.push(p);
        }
    }
    packed
}
