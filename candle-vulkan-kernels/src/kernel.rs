//! Lazy compute-pipeline cache for the Vulkan backend.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::device::Device;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderStages;

use crate::err::VulkanKernelError;
use crate::source::Source;

/// Rows of `w` handled per workgroup by the `gemv_t` shader; the
/// dispatch grid must divide by this.
pub const GEMV_T_TILE_N: usize = 4;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelName {
    AffineF32,
    ReduceSumF32,
    ReduceMaxF32,
    GemmF32,
    GemmF16,
    GemmF64,
    ConstSetF32,
    UnaryLogF32,
    UnaryAbsF32,
    UnaryRecipF32,
    UnarySqrF32,
    UnaryGeluF32,
    UnaryGeluErfF32,
    UnaryErfF32,
    UnaryReluF32,
    UnaryTanhF32,
    UnaryFloorF32,
    UnaryCeilF32,
    UnaryRoundF32,
    UnarySignF32,
    UnaryPowF32,
    UnaryEluF32,
    BinaryMaximumF32,
    BinaryMinimumF32,
    BinaryAddF32,
    BinarySubF32,
    BinaryMulF32,
    BinaryDivF32,
    UnaryExpF32,
    UnarySiluF32,
    UnarySqrtF32,
    UnarySinF32,
    UnaryCosF32,
    UnaryNegF32,
    CmpEqF32,
    CmpNeF32,
    CmpLtF32,
    CmpLeF32,
    CmpGtF32,
    CmpGeF32,
    BinaryAddBf16,
    BinarySubBf16,
    BinaryMulBf16,
    BinaryDivBf16,
    BinaryMaximumBf16,
    BinaryMinimumBf16,
    CmpEqBf16,
    CmpNeBf16,
    CmpLtBf16,
    CmpLeBf16,
    CmpGtBf16,
    CmpGeBf16,
    BinaryAddF8e4m3,
    BinarySubF8e4m3,
    BinaryMulF8e4m3,
    BinaryDivF8e4m3,
    BinaryMaximumF8e4m3,
    BinaryMinimumF8e4m3,
    CmpEqF8e4m3,
    CmpNeF8e4m3,
    CmpLtF8e4m3,
    CmpLeF8e4m3,
    CmpGtF8e4m3,
    CmpGeF8e4m3,
    UnaryLogBf16,
    UnaryAbsBf16,
    UnaryRecipBf16,
    UnarySqrBf16,
    UnaryGeluBf16,
    UnaryGeluErfBf16,
    UnaryErfBf16,
    UnaryReluBf16,
    UnaryTanhBf16,
    UnaryFloorBf16,
    UnaryCeilBf16,
    UnaryRoundBf16,
    UnarySignBf16,
    UnaryExpBf16,
    UnarySiluBf16,
    UnarySqrtBf16,
    UnarySinBf16,
    UnaryCosBf16,
    UnaryNegBf16,
    UnaryLogF8e4m3,
    UnaryAbsF8e4m3,
    UnaryRecipF8e4m3,
    UnarySqrF8e4m3,
    UnaryGeluF8e4m3,
    UnaryGeluErfF8e4m3,
    UnaryErfF8e4m3,
    UnaryReluF8e4m3,
    UnaryTanhF8e4m3,
    UnaryFloorF8e4m3,
    UnaryCeilF8e4m3,
    UnaryRoundF8e4m3,
    UnarySignF8e4m3,
    UnaryExpF8e4m3,
    UnarySiluF8e4m3,
    UnarySqrtF8e4m3,
    UnarySinF8e4m3,
    UnaryCosF8e4m3,
    UnaryNegF8e4m3,
    WhereF32,
    WhereBf16,
    WhereF8e4m3,
    AffineBf16,
    AffineF8e4m3,
    UnaryPowBf16,
    UnaryEluBf16,
    UnaryPowF8e4m3,
    UnaryEluF8e4m3,
    AffineF16,
    AffineF64,
    UnaryLogF16,
    UnaryAbsF16,
    UnaryRecipF16,
    UnarySqrF16,
    UnaryGeluF16,
    UnaryGeluErfF16,
    UnaryErfF16,
    UnaryReluF16,
    UnaryTanhF16,
    UnaryFloorF16,
    UnaryCeilF16,
    UnaryRoundF16,
    UnarySignF16,
    UnaryPowF16,
    UnaryEluF16,
    UnaryExpF16,
    UnarySiluF16,
    UnarySqrtF16,
    UnarySinF16,
    UnaryCosF16,
    UnaryNegF16,
    UnaryLogF64,
    UnaryAbsF64,
    UnaryRecipF64,
    UnarySqrF64,
    UnaryGeluF64,
    UnaryGeluErfF64,
    UnaryErfF64,
    UnaryReluF64,
    UnaryTanhF64,
    UnaryFloorF64,
    UnaryCeilF64,
    UnaryRoundF64,
    UnarySignF64,
    UnaryPowF64,
    UnaryEluF64,
    UnaryExpF64,
    UnarySiluF64,
    UnarySqrtF64,
    UnarySinF64,
    UnaryCosF64,
    UnaryNegF64,
    BinaryAddF16,
    BinarySubF16,
    BinaryMulF16,
    BinaryDivF16,
    BinaryMaximumF16,
    BinaryMinimumF16,
    CmpEqF16,
    CmpNeF16,
    CmpLtF16,
    CmpLeF16,
    CmpGtF16,
    CmpGeF16,
    BinaryAddF64,
    BinarySubF64,
    BinaryMulF64,
    BinaryDivF64,
    BinaryMaximumF64,
    BinaryMinimumF64,
    CmpEqF64,
    CmpNeF64,
    CmpLtF64,
    CmpLeF64,
    CmpGtF64,
    CmpGeF64,
    BinaryAddU8,
    BinarySubU8,
    BinaryMulU8,
    BinaryDivU8,
    BinaryMaximumU8,
    BinaryMinimumU8,
    BinaryAddU32,
    BinarySubU32,
    BinaryMulU32,
    BinaryDivU32,
    BinaryMaximumU32,
    BinaryMinimumU32,
    BinaryAddI16,
    BinarySubI16,
    BinaryMulI16,
    BinaryDivI16,
    BinaryMaximumI16,
    BinaryMinimumI16,
    BinaryAddI32,
    BinarySubI32,
    BinaryMulI32,
    BinaryDivI32,
    BinaryMaximumI32,
    BinaryMinimumI32,
    BinaryAddI64,
    BinarySubI64,
    BinaryMulI64,
    BinaryDivI64,
    BinaryMaximumI64,
    BinaryMinimumI64,
    CmpEqU8,
    CmpNeU8,
    CmpLtU8,
    CmpLeU8,
    CmpGtU8,
    CmpGeU8,
    CmpEqU32,
    CmpNeU32,
    CmpLtU32,
    CmpLeU32,
    CmpGtU32,
    CmpGeU32,
    CmpEqI16,
    CmpNeI16,
    CmpLtI16,
    CmpLeI16,
    CmpGtI16,
    CmpGeI16,
    CmpEqI32,
    CmpNeI32,
    CmpLtI32,
    CmpLeI32,
    CmpGtI32,
    CmpGeI32,
    CmpEqI64,
    CmpNeI64,
    CmpLtI64,
    CmpLeI64,
    CmpGtI64,
    CmpGeI64,
    WhereF16,
    WhereF64,
    Copy2dF32,
    Copy2dF16,
    Copy2dF64,
    GatherIdxF32,
    GatherRowsF32,
    IndexSelectF32,
    GatherIdxF16,
    GatherRowsF16,
    IndexSelectF16,
    GatherIdxF64,
    GatherRowsF64,
    IndexSelectF64,
    ReduceMinF32,
    ReduceArgMinF32,
    ReduceArgMaxF32,
    ReduceSumF16,
    ReduceMaxF16,
    ReduceMinF16,
    ReduceArgMinF16,
    ReduceArgMaxF16,
    ReduceSumF64,
    ReduceMaxF64,
    ReduceMinF64,
    ReduceArgMinF64,
    ReduceArgMaxF64,
    AvgPool2dF32,
    MaxPool2dF32,
    UpsampleNearest1dF32,
    UpsampleNearest2dF32,
    UpsampleBilinear2dF32,
    AvgPool2dF16,
    MaxPool2dF16,
    UpsampleNearest1dF16,
    UpsampleNearest2dF16,
    UpsampleBilinear2dF16,
    AvgPool2dF64,
    MaxPool2dF64,
    UpsampleNearest1dF64,
    UpsampleNearest2dF64,
    UpsampleBilinear2dF64,
    Conv1dF32,
    Conv2dF32,
    ConvTranspose1dF32,
    ConvTranspose2dF32,
    Conv1dF16,
    Conv2dF16,
    ConvTranspose1dF16,
    ConvTranspose2dF16,
    Conv1dF64,
    Conv2dF64,
    ConvTranspose1dF64,
    ConvTranspose2dF64,
    ScatterF32,
    ScatterF16,
    ScatterF64,
    TestFillF16,
    TestFillU8,
    TestFillI16,
    TestFillBf16Native,
    TestFillBf16Emulated,
    TestFillU32,
    TestFillI32,
    TestFillF32,
    TestFillI64,
    TestFillF64,
    TestFillF4,
    TestFillF6e2m3,
    TestFillF6e3m2,
    TestFillF8e4m3,
    TestFillF8e8m0,
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        match self {
            Self::AffineF32 => "main",
            Self::ReduceSumF32 => "main_reduce_sum",
            Self::ReduceMaxF32 => "main_reduce_max",
            Self::GemmF32 => "main",
            Self::GemmF16 => "main",
            Self::GemmF64 => "main",
            Self::ConstSetF32 => "main",
            Self::UnaryLogF32 => "main_log",
            Self::UnaryAbsF32 => "main_abs",
            Self::UnaryRecipF32 => "main_recip",
            Self::UnarySqrF32 => "main_sqr",
            Self::UnaryGeluF32 => "main_gelu",
            Self::UnaryGeluErfF32 => "main_gelu_erf",
            Self::UnaryErfF32 => "main_erf",
            Self::UnaryReluF32 => "main_relu",
            Self::UnaryTanhF32 => "main_tanh",
            Self::UnaryFloorF32 => "main_floor",
            Self::UnaryCeilF32 => "main_ceil",
            Self::UnaryRoundF32 => "main_round",
            Self::UnarySignF32 => "main_sign",
            Self::UnaryPowF32 => "main_pow",
            Self::UnaryEluF32 => "main_elu",
            Self::BinaryMaximumF32 => "main_maximum",
            Self::BinaryMinimumF32 => "main_minimum",
            Self::BinaryAddF32 => "main_add",
            Self::BinarySubF32 => "main_sub",
            Self::BinaryMulF32 => "main_mul",
            Self::BinaryDivF32 => "main_div",
            Self::UnaryExpF32 => "main_exp",
            Self::UnarySiluF32 => "main_silu",
            Self::UnarySqrtF32 => "main_sqrt",
            Self::UnarySinF32 => "main_sin",
            Self::UnaryCosF32 => "main_cos",
            Self::UnaryNegF32 => "main_neg",
            Self::CmpEqF32 => "main_eq",
            Self::CmpNeF32 => "main_ne",
            Self::CmpLtF32 => "main_lt",
            Self::CmpLeF32 => "main_le",
            Self::CmpGtF32 => "main_gt",
            Self::CmpGeF32 => "main_ge",
            Self::BinaryAddBf16 => "main_add",
            Self::BinarySubBf16 => "main_sub",
            Self::BinaryMulBf16 => "main_mul",
            Self::BinaryDivBf16 => "main_div",
            Self::BinaryMaximumBf16 => "main_maximum",
            Self::BinaryMinimumBf16 => "main_minimum",
            Self::CmpEqBf16 => "main_eq",
            Self::CmpNeBf16 => "main_ne",
            Self::CmpLtBf16 => "main_lt",
            Self::CmpLeBf16 => "main_le",
            Self::CmpGtBf16 => "main_gt",
            Self::CmpGeBf16 => "main_ge",
            Self::BinaryAddF8e4m3 => "main_add",
            Self::BinarySubF8e4m3 => "main_sub",
            Self::BinaryMulF8e4m3 => "main_mul",
            Self::BinaryDivF8e4m3 => "main_div",
            Self::BinaryMaximumF8e4m3 => "main_maximum",
            Self::BinaryMinimumF8e4m3 => "main_minimum",
            Self::CmpEqF8e4m3 => "main_eq",
            Self::CmpNeF8e4m3 => "main_ne",
            Self::CmpLtF8e4m3 => "main_lt",
            Self::CmpLeF8e4m3 => "main_le",
            Self::CmpGtF8e4m3 => "main_gt",
            Self::CmpGeF8e4m3 => "main_ge",
            Self::UnaryLogBf16 => "main_log",
            Self::UnaryLogF8e4m3 => "main_log",
            Self::UnaryAbsBf16 => "main_abs",
            Self::UnaryAbsF8e4m3 => "main_abs",
            Self::UnaryRecipBf16 => "main_recip",
            Self::UnaryRecipF8e4m3 => "main_recip",
            Self::UnarySqrBf16 => "main_sqr",
            Self::UnarySqrF8e4m3 => "main_sqr",
            Self::UnaryGeluBf16 => "main_gelu",
            Self::UnaryGeluF8e4m3 => "main_gelu",
            Self::UnaryGeluErfBf16 => "main_gelu_erf",
            Self::UnaryGeluErfF8e4m3 => "main_gelu_erf",
            Self::UnaryErfBf16 => "main_erf",
            Self::UnaryErfF8e4m3 => "main_erf",
            Self::UnaryReluBf16 => "main_relu",
            Self::UnaryReluF8e4m3 => "main_relu",
            Self::UnaryTanhBf16 => "main_tanh",
            Self::UnaryTanhF8e4m3 => "main_tanh",
            Self::UnaryFloorBf16 => "main_floor",
            Self::UnaryFloorF8e4m3 => "main_floor",
            Self::UnaryCeilBf16 => "main_ceil",
            Self::UnaryCeilF8e4m3 => "main_ceil",
            Self::UnaryRoundBf16 => "main_round",
            Self::UnaryRoundF8e4m3 => "main_round",
            Self::UnarySignBf16 => "main_sign",
            Self::UnarySignF8e4m3 => "main_sign",
            Self::UnaryExpBf16 => "main_exp",
            Self::UnaryExpF8e4m3 => "main_exp",
            Self::UnarySiluBf16 => "main_silu",
            Self::UnarySiluF8e4m3 => "main_silu",
            Self::UnarySqrtBf16 => "main_sqrt",
            Self::UnarySqrtF8e4m3 => "main_sqrt",
            Self::UnarySinBf16 => "main_sin",
            Self::UnarySinF8e4m3 => "main_sin",
            Self::UnaryCosBf16 => "main_cos",
            Self::UnaryCosF8e4m3 => "main_cos",
            Self::UnaryNegBf16 => "main_neg",
            Self::UnaryNegF8e4m3 => "main_neg",
            Self::WhereF32 => "main",
            Self::WhereBf16 => "main",
            Self::WhereF8e4m3 => "main",
            Self::AffineBf16 => "main",
            Self::AffineF8e4m3 => "main",
            Self::UnaryPowBf16 => "main_pow",
            Self::UnaryEluBf16 => "main_elu",
            Self::UnaryPowF8e4m3 => "main_pow",
            Self::UnaryEluF8e4m3 => "main_elu",
            Self::AffineF16 => "main",
            Self::AffineF64 => "main",
            Self::UnaryLogF16 => "main_log",
            Self::UnaryAbsF16 => "main_abs",
            Self::UnaryRecipF16 => "main_recip",
            Self::UnarySqrF16 => "main_sqr",
            Self::UnaryGeluF16 => "main_gelu",
            Self::UnaryGeluErfF16 => "main_gelu_erf",
            Self::UnaryErfF16 => "main_erf",
            Self::UnaryReluF16 => "main_relu",
            Self::UnaryTanhF16 => "main_tanh",
            Self::UnaryFloorF16 => "main_floor",
            Self::UnaryCeilF16 => "main_ceil",
            Self::UnaryRoundF16 => "main_round",
            Self::UnarySignF16 => "main_sign",
            Self::UnaryPowF16 => "main_pow",
            Self::UnaryEluF16 => "main_elu",
            Self::UnaryExpF16 => "main_exp",
            Self::UnarySiluF16 => "main_silu",
            Self::UnarySqrtF16 => "main_sqrt",
            Self::UnarySinF16 => "main_sin",
            Self::UnaryCosF16 => "main_cos",
            Self::UnaryNegF16 => "main_neg",
            Self::UnaryLogF64 => "main_log",
            Self::UnaryAbsF64 => "main_abs",
            Self::UnaryRecipF64 => "main_recip",
            Self::UnarySqrF64 => "main_sqr",
            Self::UnaryGeluF64 => "main_gelu",
            Self::UnaryGeluErfF64 => "main_gelu_erf",
            Self::UnaryErfF64 => "main_erf",
            Self::UnaryReluF64 => "main_relu",
            Self::UnaryTanhF64 => "main_tanh",
            Self::UnaryFloorF64 => "main_floor",
            Self::UnaryCeilF64 => "main_ceil",
            Self::UnaryRoundF64 => "main_round",
            Self::UnarySignF64 => "main_sign",
            Self::UnaryPowF64 => "main_pow",
            Self::UnaryEluF64 => "main_elu",
            Self::UnaryExpF64 => "main_exp",
            Self::UnarySiluF64 => "main_silu",
            Self::UnarySqrtF64 => "main_sqrt",
            Self::UnarySinF64 => "main_sin",
            Self::UnaryCosF64 => "main_cos",
            Self::UnaryNegF64 => "main_neg",
            Self::BinaryAddF16 => "main_add",
            Self::BinarySubF16 => "main_sub",
            Self::BinaryMulF16 => "main_mul",
            Self::BinaryDivF16 => "main_div",
            Self::BinaryMaximumF16 => "main_maximum",
            Self::BinaryMinimumF16 => "main_minimum",
            Self::CmpEqF16 => "main_eq",
            Self::CmpNeF16 => "main_ne",
            Self::CmpLtF16 => "main_lt",
            Self::CmpLeF16 => "main_le",
            Self::CmpGtF16 => "main_gt",
            Self::CmpGeF16 => "main_ge",
            Self::BinaryAddF64 => "main_add",
            Self::BinarySubF64 => "main_sub",
            Self::BinaryMulF64 => "main_mul",
            Self::BinaryDivF64 => "main_div",
            Self::BinaryMaximumF64 => "main_maximum",
            Self::BinaryMinimumF64 => "main_minimum",
            Self::CmpEqF64 => "main_eq",
            Self::CmpNeF64 => "main_ne",
            Self::CmpLtF64 => "main_lt",
            Self::CmpLeF64 => "main_le",
            Self::CmpGtF64 => "main_gt",
            Self::CmpGeF64 => "main_ge",
            Self::BinaryAddU8 => "main_add",
            Self::BinarySubU8 => "main_sub",
            Self::BinaryMulU8 => "main_mul",
            Self::BinaryDivU8 => "main_div",
            Self::BinaryMaximumU8 => "main_maximum",
            Self::BinaryMinimumU8 => "main_minimum",
            Self::BinaryAddU32 => "main_add",
            Self::BinarySubU32 => "main_sub",
            Self::BinaryMulU32 => "main_mul",
            Self::BinaryDivU32 => "main_div",
            Self::BinaryMaximumU32 => "main_maximum",
            Self::BinaryMinimumU32 => "main_minimum",
            Self::BinaryAddI16 => "main_add",
            Self::BinarySubI16 => "main_sub",
            Self::BinaryMulI16 => "main_mul",
            Self::BinaryDivI16 => "main_div",
            Self::BinaryMaximumI16 => "main_maximum",
            Self::BinaryMinimumI16 => "main_minimum",
            Self::BinaryAddI32 => "main_add",
            Self::BinarySubI32 => "main_sub",
            Self::BinaryMulI32 => "main_mul",
            Self::BinaryDivI32 => "main_div",
            Self::BinaryMaximumI32 => "main_maximum",
            Self::BinaryMinimumI32 => "main_minimum",
            Self::BinaryAddI64 => "main_add",
            Self::BinarySubI64 => "main_sub",
            Self::BinaryMulI64 => "main_mul",
            Self::BinaryDivI64 => "main_div",
            Self::BinaryMaximumI64 => "main_maximum",
            Self::BinaryMinimumI64 => "main_minimum",
            Self::CmpEqU8 => "main_eq",
            Self::CmpNeU8 => "main_ne",
            Self::CmpLtU8 => "main_lt",
            Self::CmpLeU8 => "main_le",
            Self::CmpGtU8 => "main_gt",
            Self::CmpGeU8 => "main_ge",
            Self::CmpEqU32 => "main_eq",
            Self::CmpNeU32 => "main_ne",
            Self::CmpLtU32 => "main_lt",
            Self::CmpLeU32 => "main_le",
            Self::CmpGtU32 => "main_gt",
            Self::CmpGeU32 => "main_ge",
            Self::CmpEqI16 => "main_eq",
            Self::CmpNeI16 => "main_ne",
            Self::CmpLtI16 => "main_lt",
            Self::CmpLeI16 => "main_le",
            Self::CmpGtI16 => "main_gt",
            Self::CmpGeI16 => "main_ge",
            Self::CmpEqI32 => "main_eq",
            Self::CmpNeI32 => "main_ne",
            Self::CmpLtI32 => "main_lt",
            Self::CmpLeI32 => "main_le",
            Self::CmpGtI32 => "main_gt",
            Self::CmpGeI32 => "main_ge",
            Self::CmpEqI64 => "main_eq",
            Self::CmpNeI64 => "main_ne",
            Self::CmpLtI64 => "main_lt",
            Self::CmpLeI64 => "main_le",
            Self::CmpGtI64 => "main_gt",
            Self::CmpGeI64 => "main_ge",
            Self::WhereF16 => "main",
            Self::WhereF64 => "main",
            Self::Copy2dF32 => "main",
            Self::Copy2dF16 => "main",
            Self::Copy2dF64 => "main",
            Self::GatherIdxF32 => "main",
            Self::GatherRowsF32 => "main_gather_rows",
            Self::IndexSelectF32 => "main_index_select",
            Self::GatherIdxF16 => "main",
            Self::GatherRowsF16 => "main_gather_rows",
            Self::IndexSelectF16 => "main_index_select",
            Self::GatherIdxF64 => "main",
            Self::GatherRowsF64 => "main_gather_rows",
            Self::IndexSelectF64 => "main_index_select",
            Self::ReduceMinF32 => "main_reduce_min",
            Self::ReduceArgMinF32 => "main_reduce_argmin",
            Self::ReduceArgMaxF32 => "main_reduce_argmax",
            Self::ReduceSumF16 => "main_reduce_sum",
            Self::ReduceMaxF16 => "main_reduce_max",
            Self::ReduceMinF16 => "main_reduce_min",
            Self::ReduceArgMinF16 => "main_reduce_argmin",
            Self::ReduceArgMaxF16 => "main_reduce_argmax",
            Self::ReduceSumF64 => "main_reduce_sum",
            Self::ReduceMaxF64 => "main_reduce_max",
            Self::ReduceMinF64 => "main_reduce_min",
            Self::ReduceArgMinF64 => "main_reduce_argmin",
            Self::ReduceArgMaxF64 => "main_reduce_argmax",
            Self::AvgPool2dF32 => "main_avg_pool2d",
            Self::MaxPool2dF32 => "main_max_pool2d",
            Self::UpsampleNearest1dF32 => "main_upsample_nearest1d",
            Self::UpsampleNearest2dF32 => "main_upsample_nearest2d",
            Self::UpsampleBilinear2dF32 => "main_upsample_bilinear2d",
            Self::AvgPool2dF16 => "main_avg_pool2d",
            Self::MaxPool2dF16 => "main_max_pool2d",
            Self::UpsampleNearest1dF16 => "main_upsample_nearest1d",
            Self::UpsampleNearest2dF16 => "main_upsample_nearest2d",
            Self::UpsampleBilinear2dF16 => "main_upsample_bilinear2d",
            Self::AvgPool2dF64 => "main_avg_pool2d",
            Self::MaxPool2dF64 => "main_max_pool2d",
            Self::UpsampleNearest1dF64 => "main_upsample_nearest1d",
            Self::UpsampleNearest2dF64 => "main_upsample_nearest2d",
            Self::UpsampleBilinear2dF64 => "main_upsample_bilinear2d",
            Self::Conv1dF32 => "main_conv1d",
            Self::Conv2dF32 => "main_conv2d",
            Self::ConvTranspose1dF32 => "main_conv_transpose1d",
            Self::ConvTranspose2dF32 => "main_conv_transpose2d",
            Self::Conv1dF16 => "main_conv1d",
            Self::Conv2dF16 => "main_conv2d",
            Self::ConvTranspose1dF16 => "main_conv_transpose1d",
            Self::ConvTranspose2dF16 => "main_conv_transpose2d",
            Self::Conv1dF64 => "main_conv1d",
            Self::Conv2dF64 => "main_conv2d",
            Self::ConvTranspose1dF64 => "main_conv_transpose1d",
            Self::ConvTranspose2dF64 => "main_conv_transpose2d",
            Self::ScatterF32 => "main",
            Self::ScatterF16 => "main",
            Self::ScatterF64 => "main",
            Self::TestFillF16 => "main",
            Self::TestFillU8
            | Self::TestFillI16
            | Self::TestFillBf16Native
            | Self::TestFillBf16Emulated
            | Self::TestFillU32
            | Self::TestFillI32
            | Self::TestFillF32
            | Self::TestFillI64
            | Self::TestFillF4
            | Self::TestFillF6e2m3
            | Self::TestFillF6e3m2
            | Self::TestFillF8e4m3
            | Self::TestFillF8e8m0
            | Self::TestFillF64 => "main",
        }
    }
}

fn pipeline_error(e: vulkano::VulkanError) -> VulkanKernelError {
    VulkanKernelError::Pipeline(e.to_string())
}

/// A cached pipeline together with the layout needed to dispatch it.
#[derive(Clone)]
pub struct PipelineEntry {
    pub pipeline: Arc<ComputePipeline>,
    pub layout: Arc<PipelineLayout>,
    pub set_layout: Arc<DescriptorSetLayout>,
}

pub type Pipelines = HashMap<(Source, KernelName), PipelineEntry>;

pub struct Kernels {
    device: Arc<Device>,
    dss_alloc: Arc<StandardDescriptorSetAllocator>,
    pipelines: RwLock<Pipelines>,
}

impl Kernels {
    pub fn new(device: Arc<Device>, dss_alloc: Arc<StandardDescriptorSetAllocator>) -> Self {
        Self {
            device,
            dss_alloc,
            pipelines: RwLock::new(Pipelines::new()),
        }
    }

    pub fn dss_alloc(&self) -> &Arc<StandardDescriptorSetAllocator> {
        &self.dss_alloc
    }

    /// The full pipeline entry (pipeline + layouts), building on first use.
    pub fn load_entry(
        &self,
        source: Source,
        name: KernelName,
    ) -> Result<PipelineEntry, VulkanKernelError> {
        if std::env::var("CANDLE_VULKAN_TRACE").is_ok() {
            trace_count(source, name);
        }
        self.load_entry_inner(source, name)
    }

    /// The full pipeline entry (pipeline + layouts), building on first use.
    fn load_entry_inner(
        &self,
        source: Source,
        name: KernelName,
    ) -> Result<PipelineEntry, VulkanKernelError> {
        {
            let guard = self.pipelines.read();
            if let Some(entry) = guard.get(&(source, name)) {
                return Ok(entry.clone());
            }
        }
        let mut guard = self.pipelines.write();
        if let Some(entry) = guard.get(&(source, name)) {
            return Ok(entry.clone());
        }
        let entry = self.build_pipeline(source, name)?;
        guard.insert((source, name), entry.clone());
        Ok(entry)
    }

    fn build_pipeline(
        &self,
        source: Source,
        name: KernelName,
    ) -> Result<PipelineEntry, VulkanKernelError> {
        let words = source.spv_words();
        let create_info = vulkano::shader::ShaderModuleCreateInfo::new(&words);
        let shader = unsafe {
            vulkano::shader::ShaderModule::new(&self.device, &create_info)
        }
        .map_err(pipeline_error)?;

        let entry_point = shader
            .entry_point(name.as_ref())
            .ok_or(VulkanKernelError::EntryPoint)?;

        let bindings: Vec<_> = (0..descriptor_bindings(name))
            .map(|i| {
                let mut b =
                    DescriptorSetLayoutBinding::new(DescriptorType::StorageBuffer);
                b.binding = i;
                b.stages = ShaderStages::COMPUTE;
                b
            })
            .collect();
        let create_info = DescriptorSetLayoutCreateInfo {
            bindings: &bindings,
            ..Default::default()
        };
        let set_layout = DescriptorSetLayout::new(&self.device, &create_info)
            .map_err(pipeline_error)?;

        let set_layouts = [&set_layout];
        let push_constant_ranges: Vec<PushConstantRange> = {
            let size = source.push_constant_size(name);
            if size > 0 {
                vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size,
                }]
            } else {
                vec![]
            }
        };
        let create_info = PipelineLayoutCreateInfo {
            set_layouts: &set_layouts,
            push_constant_ranges: &push_constant_ranges,
            ..Default::default()
        };
        let layout = PipelineLayout::new(&self.device, &create_info)
            .map_err(pipeline_error)?;
        let stage = PipelineShaderStageCreateInfo::new(&entry_point);
        let create_info = ComputePipelineCreateInfo::new(stage, &layout);
        let pipeline = ComputePipeline::new(
            &self.device,
            None,
            &create_info,
        )
        .map_err(pipeline_error)?;

        Ok(PipelineEntry {
            pipeline,
            layout,
            set_layout,
        })
    }
}

/// Number of storage-buffer bindings in the kernel's descriptor set. The
/// elementwise WGSL module is shared by all its entry points, so each of
/// them sees in/rhs/out at bindings 0/1/2 and needs the 3-binding layout;
/// the other kernels use their input (+rhs) and output at bindings 0..N-1.
fn descriptor_bindings(name: KernelName) -> u32 {
    match name {
        KernelName::AffineF32
        | KernelName::AffineBf16
        | KernelName::AffineF8e4m3 => 3,
        KernelName::GemmF32 | KernelName::GemmF16 | KernelName::GemmF64 => 4,
        KernelName::ConstSetF32
        | KernelName::UnaryLogF32
        | KernelName::UnaryAbsF32
        | KernelName::UnaryRecipF32
        | KernelName::UnarySqrF32
        | KernelName::UnaryGeluF32
        | KernelName::UnaryGeluErfF32
        | KernelName::UnaryErfF32
        | KernelName::UnaryReluF32
        | KernelName::UnaryTanhF32
        | KernelName::UnaryFloorF32
        | KernelName::UnaryCeilF32
        | KernelName::UnaryRoundF32
        | KernelName::UnarySignF32
        | KernelName::UnaryPowF32
        | KernelName::UnaryEluF32
        | KernelName::UnaryPowBf16
        | KernelName::UnaryEluBf16
        | KernelName::UnaryPowF8e4m3
        | KernelName::UnaryEluF8e4m3
        | KernelName::UnaryExpF32
        | KernelName::UnarySiluF32
        | KernelName::UnarySqrtF32
        | KernelName::UnarySinF32
        | KernelName::UnaryCosF32
        | KernelName::UnaryNegF32
        | KernelName::UnaryLogBf16
        | KernelName::UnaryAbsBf16
        | KernelName::UnaryRecipBf16
        | KernelName::UnarySqrBf16
        | KernelName::UnaryGeluBf16
        | KernelName::UnaryGeluErfBf16
        | KernelName::UnaryErfBf16
        | KernelName::UnaryReluBf16
        | KernelName::UnaryTanhBf16
        | KernelName::UnaryFloorBf16
        | KernelName::UnaryCeilBf16
        | KernelName::UnaryRoundBf16
        | KernelName::UnarySignBf16
        | KernelName::UnaryExpBf16
        | KernelName::UnarySiluBf16
        | KernelName::UnarySqrtBf16
        | KernelName::UnarySinBf16
        | KernelName::UnaryCosBf16
        | KernelName::UnaryNegBf16
        | KernelName::UnaryLogF8e4m3
        | KernelName::UnaryAbsF8e4m3
        | KernelName::UnaryRecipF8e4m3
        | KernelName::UnarySqrF8e4m3
        | KernelName::UnaryGeluF8e4m3
        | KernelName::UnaryGeluErfF8e4m3
        | KernelName::UnaryErfF8e4m3
        | KernelName::UnaryReluF8e4m3
        | KernelName::UnaryTanhF8e4m3
        | KernelName::UnaryFloorF8e4m3
        | KernelName::UnaryCeilF8e4m3
        | KernelName::UnaryRoundF8e4m3
        | KernelName::UnarySignF8e4m3
        | KernelName::UnaryExpF8e4m3
        | KernelName::UnarySiluF8e4m3
        | KernelName::UnarySqrtF8e4m3
        | KernelName::UnarySinF8e4m3
        | KernelName::UnaryCosF8e4m3
        | KernelName::UnaryNegF8e4m3
        | KernelName::AffineF16
        | KernelName::AffineF64
        | KernelName::UnaryLogF16
        | KernelName::UnaryAbsF16
        | KernelName::UnaryRecipF16
        | KernelName::UnarySqrF16
        | KernelName::UnaryGeluF16
        | KernelName::UnaryGeluErfF16
        | KernelName::UnaryErfF16
        | KernelName::UnaryReluF16
        | KernelName::UnaryTanhF16
        | KernelName::UnaryFloorF16
        | KernelName::UnaryCeilF16
        | KernelName::UnaryRoundF16
        | KernelName::UnarySignF16
        | KernelName::UnaryPowF16
        | KernelName::UnaryEluF16
        | KernelName::UnaryExpF16
        | KernelName::UnarySiluF16
        | KernelName::UnarySqrtF16
        | KernelName::UnarySinF16
        | KernelName::UnaryCosF16
        | KernelName::UnaryNegF16
        | KernelName::UnaryLogF64
        | KernelName::UnaryAbsF64
        | KernelName::UnaryRecipF64
        | KernelName::UnarySqrF64
        | KernelName::UnaryGeluF64
        | KernelName::UnaryGeluErfF64
        | KernelName::UnaryErfF64
        | KernelName::UnaryReluF64
        | KernelName::UnaryTanhF64
        | KernelName::UnaryFloorF64
        | KernelName::UnaryCeilF64
        | KernelName::UnaryRoundF64
        | KernelName::UnarySignF64
        | KernelName::UnaryPowF64
        | KernelName::UnaryEluF64
        | KernelName::UnaryExpF64
        | KernelName::UnarySiluF64
        | KernelName::UnarySqrtF64
        | KernelName::UnarySinF64
        | KernelName::UnaryCosF64
        | KernelName::UnaryNegF64 => 3,
        KernelName::BinaryMaximumF32
        | KernelName::BinaryMinimumF32
        | KernelName::BinaryAddF32
        | KernelName::BinarySubF32
        | KernelName::BinaryMulF32
        | KernelName::BinaryDivF32
        | KernelName::CmpEqF32
        | KernelName::CmpNeF32
        | KernelName::CmpLtF32
        | KernelName::CmpLeF32
        | KernelName::CmpGtF32
        | KernelName::CmpGeF32
        | KernelName::BinaryAddBf16
        | KernelName::BinarySubBf16
        | KernelName::BinaryMulBf16
        | KernelName::BinaryDivBf16
        | KernelName::BinaryMaximumBf16
        | KernelName::BinaryMinimumBf16
        | KernelName::CmpEqBf16
        | KernelName::CmpNeBf16
        | KernelName::CmpLtBf16
        | KernelName::CmpLeBf16
        | KernelName::CmpGtBf16
        | KernelName::CmpGeBf16
        | KernelName::BinaryAddF8e4m3
        | KernelName::BinarySubF8e4m3
        | KernelName::BinaryMulF8e4m3
        | KernelName::BinaryDivF8e4m3
        | KernelName::BinaryMaximumF8e4m3
        | KernelName::BinaryMinimumF8e4m3
        | KernelName::CmpEqF8e4m3
        | KernelName::CmpNeF8e4m3
        | KernelName::CmpLtF8e4m3
        | KernelName::CmpLeF8e4m3
        | KernelName::CmpGtF8e4m3
        | KernelName::CmpGeF8e4m3
        | KernelName::BinaryAddF16
        | KernelName::BinarySubF16
        | KernelName::BinaryMulF16
        | KernelName::BinaryDivF16
        | KernelName::BinaryMaximumF16
        | KernelName::BinaryMinimumF16
        | KernelName::CmpEqF16
        | KernelName::CmpNeF16
        | KernelName::CmpLtF16
        | KernelName::CmpLeF16
        | KernelName::CmpGtF16
        | KernelName::CmpGeF16
        | KernelName::BinaryAddF64
        | KernelName::BinarySubF64
        | KernelName::BinaryMulF64
        | KernelName::BinaryDivF64
        | KernelName::BinaryMaximumF64
        | KernelName::BinaryMinimumF64
        | KernelName::CmpEqF64
        | KernelName::CmpNeF64
        | KernelName::CmpLtF64
        | KernelName::CmpLeF64
        | KernelName::CmpGtF64
        | KernelName::CmpGeF64
        | KernelName::BinaryAddU8
        | KernelName::BinarySubU8
        | KernelName::BinaryMulU8
        | KernelName::BinaryDivU8
        | KernelName::BinaryMaximumU8
        | KernelName::BinaryMinimumU8
        | KernelName::BinaryAddU32
        | KernelName::BinarySubU32
        | KernelName::BinaryMulU32
        | KernelName::BinaryDivU32
        | KernelName::BinaryMaximumU32
        | KernelName::BinaryMinimumU32
        | KernelName::BinaryAddI16
        | KernelName::BinarySubI16
        | KernelName::BinaryMulI16
        | KernelName::BinaryDivI16
        | KernelName::BinaryMaximumI16
        | KernelName::BinaryMinimumI16
        | KernelName::BinaryAddI32
        | KernelName::BinarySubI32
        | KernelName::BinaryMulI32
        | KernelName::BinaryDivI32
        | KernelName::BinaryMaximumI32
        | KernelName::BinaryMinimumI32
        | KernelName::BinaryAddI64
        | KernelName::BinarySubI64
        | KernelName::BinaryMulI64
        | KernelName::BinaryDivI64
        | KernelName::BinaryMaximumI64
        | KernelName::BinaryMinimumI64
        | KernelName::CmpEqU8
        | KernelName::CmpNeU8
        | KernelName::CmpLtU8
        | KernelName::CmpLeU8
        | KernelName::CmpGtU8
        | KernelName::CmpGeU8
        | KernelName::CmpEqU32
        | KernelName::CmpNeU32
        | KernelName::CmpLtU32
        | KernelName::CmpLeU32
        | KernelName::CmpGtU32
        | KernelName::CmpGeU32
        | KernelName::CmpEqI16
        | KernelName::CmpNeI16
        | KernelName::CmpLtI16
        | KernelName::CmpLeI16
        | KernelName::CmpGtI16
        | KernelName::CmpGeI16
        | KernelName::CmpEqI32
        | KernelName::CmpNeI32
        | KernelName::CmpLtI32
        | KernelName::CmpLeI32
        | KernelName::CmpGtI32
        | KernelName::CmpGeI32
        | KernelName::CmpEqI64
        | KernelName::CmpNeI64
        | KernelName::CmpLtI64
        | KernelName::CmpLeI64
        | KernelName::CmpGtI64
        | KernelName::CmpGeI64
        => 4,
        KernelName::WhereF32
        | KernelName::WhereBf16
        | KernelName::WhereF8e4m3
        | KernelName::WhereF16
        | KernelName::WhereF64 => 5,
        KernelName::Copy2dF32
        | KernelName::Copy2dF16
        | KernelName::Copy2dF64
        => 3,
        KernelName::GatherIdxF32 => 4,
        KernelName::GatherRowsF32 => 4,
        KernelName::IndexSelectF32
        | KernelName::GatherIdxF16
        | KernelName::GatherRowsF16
        | KernelName::IndexSelectF16
        | KernelName::GatherIdxF64
        | KernelName::GatherRowsF64
        | KernelName::IndexSelectF64
        => 4,
        KernelName::ReduceMinF32
        | KernelName::ReduceArgMinF32
        | KernelName::ReduceArgMaxF32
        | KernelName::ReduceSumF32
        | KernelName::ReduceMaxF32
        | KernelName::ReduceSumF16
        | KernelName::ReduceMaxF16
        | KernelName::ReduceMinF16
        | KernelName::ReduceArgMinF16
        | KernelName::ReduceArgMaxF16
        | KernelName::ReduceSumF64
        | KernelName::ReduceMaxF64
        | KernelName::ReduceMinF64
        | KernelName::ReduceArgMinF64
        | KernelName::ReduceArgMaxF64
        | KernelName::AvgPool2dF32
        | KernelName::MaxPool2dF32
        | KernelName::UpsampleNearest1dF32
        | KernelName::UpsampleNearest2dF32
        | KernelName::UpsampleBilinear2dF32
        | KernelName::AvgPool2dF16
        | KernelName::MaxPool2dF16
        | KernelName::UpsampleNearest1dF16
        | KernelName::UpsampleNearest2dF16
        | KernelName::UpsampleBilinear2dF16
        | KernelName::AvgPool2dF64
        | KernelName::MaxPool2dF64
        | KernelName::UpsampleNearest1dF64
        | KernelName::UpsampleNearest2dF64
        | KernelName::UpsampleBilinear2dF64
        => 4,
        | KernelName::Conv1dF32
        | KernelName::Conv2dF32
        | KernelName::ConvTranspose1dF32
        | KernelName::ConvTranspose2dF32
        | KernelName::Conv1dF16
        | KernelName::Conv2dF16
        | KernelName::ConvTranspose1dF16
        | KernelName::ConvTranspose2dF16
        | KernelName::Conv1dF64
        | KernelName::Conv2dF64
        | KernelName::ConvTranspose1dF64
        | KernelName::ConvTranspose2dF64
        => 4,
        KernelName::ScatterF32
        | KernelName::ScatterF16
        | KernelName::ScatterF64
        => 4,
        KernelName::TestFillF16 => 1,
        KernelName::TestFillU8
        | KernelName::TestFillI16
        | KernelName::TestFillBf16Native
        | KernelName::TestFillBf16Emulated
        | KernelName::TestFillU32
        | KernelName::TestFillI32
        | KernelName::TestFillF32
        | KernelName::TestFillI64
        | KernelName::TestFillF4
        | KernelName::TestFillF6e2m3
        | KernelName::TestFillF6e3m2
        | KernelName::TestFillF8e4m3
        | KernelName::TestFillF8e8m0
        | KernelName::TestFillF64 => 1,
    }
}

use std::sync::Mutex;
static TRACE: Mutex<Option<std::collections::HashMap<(Source, KernelName), u64>>> =
    Mutex::new(None);
fn trace_count(source: Source, name: KernelName) {
    let mut g = TRACE.lock().unwrap();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    *m.entry((source, name)).or_insert(0) += 1;
}
/// Drain the per-kernel dispatch counters (CANDLE_VULKAN_TRACE only).
pub fn trace_counts() -> Vec<((Source, KernelName), u64)> {
    let mut g = TRACE.lock().unwrap();
    let m = g.take().unwrap_or_default();
    m.into_iter().collect()
}
pub fn trace_reset() {
    *TRACE.lock().unwrap() = None;
}
