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
use vulkano::pipeline::layout::{
    PipelineLayoutCreateInfo, PushConstantRange,
};
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::{ShaderModule, ShaderStages};

use crate::err::VulkanKernelError;
use crate::source::Source;

/// Number of storage-buffer bindings shared by the scaffold kernels (input + output).
const DESCRIPTOR_BINDINGS: u32 = 2;
/// Bytes of push constants shared by the scaffold kernels (`uint + 2 * float`).
const PUSH_CONSTANT_SIZE: u32 = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelName {
    AffineF32,
    ReduceSumF32,
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        match self {
            Self::AffineF32 => "main",
            Self::ReduceSumF32 => "main",
        }
    }
}

/// A cached pipeline together with the layout needed to dispatch it.
#[derive(Clone)]
pub struct PipelineEntry {
    pub pipeline: Arc<ComputePipeline>,
    pub layout: Arc<PipelineLayout>,
    pub set_layout: Arc<DescriptorSetLayout>,
}

pub type Pipelines = HashMap<(Source, KernelName), PipelineEntry>;

#[derive(Debug)]
pub struct Kernels {
    device: Arc<Device>,
    dss_alloc: Arc<StandardDescriptorSetAllocator>,
    pipelines: RwLock<Pipelines>,
}

impl Kernels {
    pub fn new(
        device: Arc<Device>,
        dss_alloc: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
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

    fn build_pipeline(&self, source: Source, name: KernelName) -> Result<PipelineEntry, VulkanKernelError> {
        let words = source.spv_words();
        let shader = unsafe {
            vulkano::shader::ShaderModule::new(
                self.device.clone(),
                vulkano::shader::ShaderModuleCreateInfo::new(&words),
            )
        }
        .map_err(|e| VulkanKernelError::Pipeline(e.to_string()))?;

        let entry_point = shader
            .entry_point(name.as_ref())
            .ok_or(VulkanKernelError::EntryPoint)?;

        let binding = {
            let mut b = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
            b.stages = ShaderStages::COMPUTE;
            b
        };
        let set_layout = DescriptorSetLayout::new(
            self.device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            },
        )
        .map_err(|e| VulkanKernelError::Pipeline(e.to_string()))?;

        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![set_layout.clone()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: PUSH_CONSTANT_SIZE,
                }],
                ..Default::default()
            },
        )
        .map_err(|e| VulkanKernelError::Pipeline(e.to_string()))?;

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout.clone()),
        )
        .map_err(|e| VulkanKernelError::Pipeline(e.to_string()))?;

        Ok(PipelineEntry {
            pipeline,
            layout,
            set_layout,
        })
    }
}
