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
    GatherF32,
    CopyF32,
    ElemAddF32,
    ElemSubF32,
    ElemMulF32,
    ElemDivF32,
    ElemSigmoidF32,
    ElemSiluF32,
    ElemExpF32,
    ElemSqrtF32,
    ElemSinF32,
    ElemCosF32,
    ElemNegF32,
    GemmF32,
    GemvTF32,
    RmsNormF32,
    SoftmaxLastDimF32,
    RopeF32,
    GemvF32,
    Q4kQmatvecF32,
    Q4kDequantF32,
    Q6kDequantF32,
    Q80DequantF32,
    Q50DequantF32,
    Q5KDequantF32,
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
    CmpEqF32,
    CmpNeF32,
    CmpLtF32,
    CmpLeF32,
    CmpGtF32,
    CmpGeF32,
    WhereF32,
    Copy2dF32,
    GatherIdxF32,
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
            Self::ReduceSumF32 => "main",
            Self::ReduceMaxF32 => "main",
            Self::GatherF32 => "main",
            Self::CopyF32 => "main",
            Self::ElemAddF32 => "main_add",
            Self::ElemSubF32 => "main_sub",
            Self::ElemMulF32 => "main_mul",
            Self::ElemDivF32 => "main_div",
            Self::ElemSigmoidF32 => "main_sigmoid",
            Self::ElemSiluF32 => "main_silu",
            Self::ElemExpF32 => "main_exp",
            Self::ElemSqrtF32 => "main_sqrt",
            Self::ElemSinF32 => "main_sin",
            Self::ElemCosF32 => "main_cos",
            Self::ElemNegF32 => "main_neg",
            Self::GemmF32 => "main_gemm",
            Self::RmsNormF32 => "main",
            Self::SoftmaxLastDimF32 => "main",
            Self::RopeF32 => "main",
            Self::GemvF32 => "main",
            Self::GemvTF32 => "main",
            Self::Q4kQmatvecF32 => "main_qmatvec",
            Self::Q4kDequantF32 => "main_dequant",
            Self::Q6kDequantF32 => "main_q6k_dequant",
            Self::Q80DequantF32 => "main_q80_dequant",
            Self::Q50DequantF32 => "main_q50_dequant",
            Self::Q5KDequantF32 => "main_q5k_dequant",
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
            Self::CmpEqF32 => "main_eq",
            Self::CmpNeF32 => "main_ne",
            Self::CmpLtF32 => "main_lt",
            Self::CmpLeF32 => "main_le",
            Self::CmpGtF32 => "main_gt",
            Self::CmpGeF32 => "main_ge",
            Self::WhereF32 => "main",
            Self::Copy2dF32 => "main",
            Self::GatherIdxF32 => "main",
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
        | KernelName::ReduceSumF32
        | KernelName::ReduceMaxF32
        | KernelName::CopyF32
        | KernelName::SoftmaxLastDimF32 => 2,
        KernelName::GatherF32
        | KernelName::ElemAddF32
        | KernelName::ElemSubF32
        | KernelName::ElemMulF32
        | KernelName::ElemDivF32
        | KernelName::ElemSigmoidF32
        | KernelName::ElemSiluF32
        | KernelName::ElemExpF32
        | KernelName::ElemSqrtF32
        | KernelName::ElemSinF32
        | KernelName::ElemCosF32
        | KernelName::ElemNegF32
        | KernelName::GemmF32
        | KernelName::RmsNormF32 => 3,
        KernelName::RopeF32 => 4,
        KernelName::GemvF32
        | KernelName::GemvTF32
        | KernelName::Q4kQmatvecF32
        | KernelName::Q4kDequantF32
        | KernelName::Q6kDequantF32
        | KernelName::Q80DequantF32
        | KernelName::Q50DequantF32
        | KernelName::Q5KDequantF32 => 3,
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
        | KernelName::UnaryEluF32 => 3,
        KernelName::BinaryMaximumF32
        | KernelName::BinaryMinimumF32
        | KernelName::CmpEqF32
        | KernelName::CmpNeF32
        | KernelName::CmpLtF32
        | KernelName::CmpLeF32
        | KernelName::CmpGtF32
        | KernelName::CmpGeF32 => 4,
        KernelName::WhereF32 => 5,
        KernelName::Copy2dF32 => 3,
        KernelName::GatherIdxF32 => 4,
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
