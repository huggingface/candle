//! A pipeline that performs general-purpose operations.
//!
//! A compute pipeline takes buffers and/or images as both inputs and outputs. It operates
//! "standalone", with no additional infrastructure such as render passes or vertex input. Compute
//! pipelines can be used by themselves for performing work on the Vulkan device, but they can also
//! assist graphics operations by precalculating or post-processing the operations from another
//! kind of pipeline. While it theoretically possible to perform graphics operations entirely in a
//! compute pipeline, a graphics pipeline is better suited to that task.
//!
//! A compute pipeline is relatively simple to create, requiring only a pipeline layout and a
//! single shader, the *compute shader*. The compute shader is the actual program that performs the
//! work. Once created, you can execute a compute pipeline by *binding* it in a command buffer,
//! binding any descriptor sets and/or push constants that the pipeline needs, and then issuing a
//! `dispatch` command on the command buffer.

use super::{
    PipelineCreateFlags, PipelineShaderStageCreateInfo, PipelineShaderStageCreateInfoExtensionsVk,
    PipelineShaderStageCreateInfoFields1Vk, PipelineShaderStageCreateInfoFields2Vk,
};
use crate::{
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    pipeline::{cache::PipelineCache, layout::PipelineLayout},
    shader::{spirv::ExecutionModel, DescriptorBindingRequirements},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use foldhash::HashMap;
use std::{fmt::Debug, mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// A pipeline object that describes to the Vulkan implementation how it should perform compute
/// operations.
///
/// The template parameter contains the descriptor set to use with this pipeline.
///
/// Pass an optional `Arc` to a `PipelineCache` to enable pipeline caching. The Vulkan
/// implementation will handle the `PipelineCache` and check if it is available.
/// Check the documentation of the `PipelineCache` for more information.
#[derive(Debug)]
pub struct ComputePipeline {
    handle: vk::Pipeline,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: PipelineCreateFlags,
    layout: DeviceOwnedDebugWrapper<Arc<PipelineLayout>>,

    descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    num_used_descriptor_sets: u32,
}

impl ComputePipeline {
    /// Creates a new `ComputePipeline`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        cache: Option<&Arc<PipelineCache>>,
        create_info: &ComputePipelineCreateInfo<'_>,
    ) -> Result<Arc<ComputePipeline>, VulkanError> {
        match Self::try_new(device, cache, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `ComputePipeline`.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        cache: Option<&Arc<PipelineCache>>,
        create_info: &ComputePipelineCreateInfo<'_>,
    ) -> Result<Arc<ComputePipeline>, Validated<VulkanError>> {
        Self::validate_new(device, cache.map(|c| &**c), create_info)?;

        Ok(unsafe { Self::new_unchecked(device, cache, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        cache: Option<&PipelineCache>,
        create_info: &ComputePipelineCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkCreateComputePipelines-pipelineCache-parent
        if let Some(cache) = cache {
            assert_eq!(device, cache.device().as_ref());
        }
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        cache: Option<&Arc<PipelineCache>>,
        create_info: &ComputePipelineCreateInfo<'_>,
    ) -> Result<Arc<ComputePipeline>, VulkanError> {
        let create_info_fields2_vk = create_info.to_vk_fields2();
        let create_info_fields1_vk = create_info.to_vk_fields1(&create_info_fields2_vk);
        let mut create_info_extensions_vk = create_info.to_vk_extensions();
        let create_info_vk =
            create_info.to_vk(&create_info_fields1_vk, &mut create_info_extensions_vk);

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_compute_pipelines)(
                    device.handle(),
                    cache.as_ref().map_or_else(Default::default, |c| c.handle()),
                    1,
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle(device, handle, create_info) })
    }

    /// Creates a new `ComputePipeline` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::Pipeline,
        create_info: &ComputePipelineCreateInfo<'_>,
    ) -> Arc<ComputePipeline> {
        let &ComputePipelineCreateInfo {
            flags,
            ref stage,
            layout,
            base_pipeline: _,
            _ne: _,
        } = create_info;

        let descriptor_binding_requirements: HashMap<_, _> = stage
            .entry_point
            .info()
            .descriptor_binding_requirements
            .iter()
            .map(|(&loc, reqs)| (loc, reqs.clone()))
            .collect();
        let num_used_descriptor_sets = descriptor_binding_requirements
            .keys()
            .map(|loc| loc.0)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);

        Arc::new(ComputePipeline {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),

            flags,
            layout: DeviceOwnedDebugWrapper(layout.clone()),

            descriptor_binding_requirements,
            num_used_descriptor_sets,
        })
    }

    /// Returns the `Device` that the pipeline was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the flags that the pipeline was created with.
    #[inline]
    pub fn flags(&self) -> PipelineCreateFlags {
        self.flags
    }

    /// Returns the pipeline layout used in this pipeline.
    #[inline]
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }

    /// Returns the number of descriptor sets actually accessed by this pipeline. This may be less
    /// than the number of sets in the pipeline layout.
    #[inline]
    pub fn num_used_descriptor_sets(&self) -> u32 {
        self.num_used_descriptor_sets
    }

    /// Returns a reference to the descriptor binding requirements for this pipeline.
    #[inline]
    pub fn descriptor_binding_requirements(
        &self,
    ) -> &HashMap<(u32, u32), DescriptorBindingRequirements> {
        &self.descriptor_binding_requirements
    }
}

impl_id_counter!(ComputePipeline);

unsafe impl VulkanObject for ComputePipeline {
    type Handle = vk::Pipeline;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for ComputePipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device()
    }
}

impl Drop for ComputePipeline {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.destroy_pipeline)(self.device.handle(), self.handle, ptr::null()) };
    }
}

/// Parameters to create a new `ComputePipeline`.
#[derive(Clone, Debug)]
pub struct ComputePipelineCreateInfo<'a> {
    /// Additional properties of the pipeline.
    ///
    /// The default value is empty.
    pub flags: PipelineCreateFlags,

    /// The compute shader stage to use.
    ///
    /// There is no default value.
    pub stage: PipelineShaderStageCreateInfo<'a>,

    /// The pipeline layout to use.
    ///
    /// There is no default value.
    pub layout: &'a Arc<PipelineLayout>,

    /// The pipeline to use as a base when creating this pipeline.
    ///
    /// If this is `Some`, then `flags` must contain [`PipelineCreateFlags::DERIVATIVE`],
    /// and the `flags` of the provided pipeline must contain
    /// [`PipelineCreateFlags::ALLOW_DERIVATIVES`].
    ///
    /// The default value is `None`.
    pub base_pipeline: Option<&'a Arc<ComputePipeline>>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> ComputePipelineCreateInfo<'a> {
    /// Returns a default `ComputePipelineCreateInfo` with the provided `stage` and `layout`.
    #[inline]
    pub const fn new(
        stage: PipelineShaderStageCreateInfo<'a>,
        layout: &'a Arc<PipelineLayout>,
    ) -> Self {
        Self {
            flags: PipelineCreateFlags::empty(),
            stage,
            layout,
            base_pipeline: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref stage,
            layout,
            base_pipeline,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkComputePipelineCreateInfo-flags-parameter"])
        })?;

        stage
            .validate(device)
            .map_err(|err| err.add_context("stage"))?;

        if flags.intersects(PipelineCreateFlags::DERIVATIVE) {
            let base_pipeline = base_pipeline.as_ref().ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "`flags` contains `PipelineCreateFlags::DERIVATIVE`, but \
                        `base_pipeline` is `None`"
                        .into(),
                    vuids: &["VUID-VkComputePipelineCreateInfo-flags-07984"],
                    ..Default::default()
                })
            })?;

            if !base_pipeline
                .flags()
                .intersects(PipelineCreateFlags::ALLOW_DERIVATIVES)
            {
                return Err(Box::new(ValidationError {
                    context: "base_pipeline.flags()".into(),
                    problem: "does not contain `PipelineCreateFlags::ALLOW_DERIVATIVES`".into(),
                    vuids: &["VUID-vkCreateComputePipelines-flags-00696"],
                    ..Default::default()
                }));
            }
        } else if base_pipeline.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`flags` does not contain `PipelineCreateFlags::DERIVATIVE`, but \
                    `base_pipeline` is `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        let PipelineShaderStageCreateInfo {
            flags: _,
            entry_point,
            required_subgroup_size: _,
            _ne: _,
        } = stage;

        let entry_point_info = entry_point.info();

        if !matches!(entry_point_info.execution_model, ExecutionModel::GLCompute) {
            return Err(Box::new(ValidationError {
                context: "stage.entry_point".into(),
                problem: "is not a `ShaderStage::Compute` entry point".into(),
                vuids: &["VUID-VkComputePipelineCreateInfo-stage-00701"],
                ..Default::default()
            }));
        }

        // TODO: Make sure that all VUIDs are indeed checked.
        layout
            .ensure_compatible_with_shader(
                entry_point_info
                    .descriptor_binding_requirements
                    .iter()
                    .map(|(k, v)| (*k, v)),
                entry_point_info.push_constant_requirements.as_ref(),
            )
            .map_err(|err| {
                Box::new(ValidationError {
                    context: "stage.entry_point".into(),
                    vuids: &[
                        "VUID-VkComputePipelineCreateInfo-layout-07987",
                        "VUID-VkComputePipelineCreateInfo-layout-07988",
                        "VUID-VkComputePipelineCreateInfo-layout-07990",
                        "VUID-VkComputePipelineCreateInfo-layout-07991",
                    ],
                    ..ValidationError::from_error(err)
                })
            })?;

        // TODO:
        // VUID-VkComputePipelineCreateInfo-stage-00702
        // VUID-VkComputePipelineCreateInfo-layout-01687

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a ComputePipelineCreateInfoFields1Vk<'_>,
        extensions_vk: &'a mut ComputePipelineCreateInfoExtensionsVk,
    ) -> vk::ComputePipelineCreateInfo<'a> {
        let &Self {
            flags,
            ref stage,
            layout,
            base_pipeline,
            _ne: _,
        } = self;
        let ComputePipelineCreateInfoFields1Vk { stage_fields1_vk } = fields1_vk;
        let ComputePipelineCreateInfoExtensionsVk {
            stage_extensions_vk,
        } = extensions_vk;

        let stage_vk = stage.to_vk(stage_fields1_vk, stage_extensions_vk);

        vk::ComputePipelineCreateInfo::default()
            .flags(flags.into())
            .stage(stage_vk)
            .layout(layout.handle())
            .base_pipeline_handle(
                base_pipeline
                    .as_ref()
                    .map_or(vk::Pipeline::null(), VulkanObject::handle),
            )
            .base_pipeline_index(-1)
    }

    pub(crate) fn to_vk_extensions(&self) -> ComputePipelineCreateInfoExtensionsVk {
        let stage_extensions_vk = self.stage.to_vk_extensions();

        ComputePipelineCreateInfoExtensionsVk {
            stage_extensions_vk,
        }
    }

    pub(crate) fn to_vk_fields1(
        &self,
        fields2_vk: &'a ComputePipelineCreateInfoFields2Vk,
    ) -> ComputePipelineCreateInfoFields1Vk<'a> {
        let ComputePipelineCreateInfoFields2Vk { stage_fields2_vk } = fields2_vk;

        let stage_fields1_vk = self.stage.to_vk_fields1(stage_fields2_vk);

        ComputePipelineCreateInfoFields1Vk { stage_fields1_vk }
    }

    pub(crate) fn to_vk_fields2(&self) -> ComputePipelineCreateInfoFields2Vk {
        let stage_fields2_vk = self.stage.to_vk_fields2();

        ComputePipelineCreateInfoFields2Vk { stage_fields2_vk }
    }
}

pub(crate) struct ComputePipelineCreateInfoExtensionsVk {
    pub(crate) stage_extensions_vk: PipelineShaderStageCreateInfoExtensionsVk,
}

pub(crate) struct ComputePipelineCreateInfoFields1Vk<'a> {
    pub(crate) stage_fields1_vk: PipelineShaderStageCreateInfoFields1Vk<'a>,
}

pub(crate) struct ComputePipelineCreateInfoFields2Vk {
    pub(crate) stage_fields2_vk: PipelineShaderStageCreateInfoFields2Vk,
}

#[cfg(test)]
mod tests {
    use crate::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage},
        command_buffer::{
            allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator, DescriptorBufferInfo, DescriptorSet,
            WriteDescriptorSet,
        },
        memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
        pipeline::{
            compute::ComputePipelineCreateInfo, ComputePipeline, PipelineBindPoint, PipelineLayout,
            PipelineShaderStageCreateInfo,
        },
        shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
        sync::{now, GpuFuture},
    };
    use std::{slice, sync::Arc};

    // TODO: test for basic creation
    // TODO: test for pipeline layout error

    #[test]
    fn specialization_constants() {
        // This test checks whether specialization constants work.
        // It executes a single compute shader (one invocation) that writes the value of a spec.
        // constant to a buffer. The buffer content is then checked for the right value.

        let (device, queue) = gfx_dev_and_queue!();

        let cs = {
            /*
            #version 450

            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            layout(constant_id = 83) const int VALUE = 0xdeadbeef;

            layout(set = 0, binding = 0) buffer Output {
                int write;
            } write;

            void main() {
                write.write = VALUE;
            }
            */
            const MODULE: [u32; 120] = [
                119734787, 65536, 524289, 14, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1, 1,
                196611, 2, 450, 262149, 4, 1852399981, 0, 262149, 7, 1886680399, 29813, 327686, 7,
                0, 1953067639, 101, 262149, 9, 1953067639, 101, 262149, 11, 1431060822, 69, 327752,
                7, 0, 35, 0, 196679, 7, 3, 262215, 9, 34, 0, 262215, 9, 33, 0, 262215, 11, 1, 83,
                131091, 2, 196641, 3, 2, 262165, 6, 32, 1, 196638, 7, 6, 262176, 8, 2, 7, 262203,
                8, 9, 2, 262187, 6, 10, 0, 262194, 6, 11, 3735928559, 262176, 12, 2, 6, 327734, 2,
                4, 0, 3, 131320, 5, 327745, 12, 13, 9, 10, 196670, 13, 11, 65789, 65592,
            ];
            let module =
                unsafe { ShaderModule::new(&device, &ShaderModuleCreateInfo::new(&MODULE)) }
                    .unwrap();
            module
                .specialize(&[(83, 0x12345678i32.into())])
                .entry_point("main")
                .unwrap()
        };

        let pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = PipelineLayout::from_stages(&device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                &device,
                None,
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let data_buffer = Buffer::from_data(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            0,
        )
        .unwrap();

        let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &device,
            &Default::default(),
        ));
        let set = DescriptorSet::new(
            &ds_allocator,
            &pipeline.layout().set_layouts()[0],
            &[WriteDescriptorSet::buffer(
                0,
                &DescriptorBufferInfo {
                    buffer: Some(data_buffer.buffer()),
                    ..Default::default()
                },
            )],
            &[],
        )
        .unwrap();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let mut cbb = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cbb.bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap();
        unsafe { cbb.dispatch([1, 1, 1]) }.unwrap();

        let cb = cbb.build().unwrap();

        let future = now(device)
            .then_execute(queue, cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let data_buffer_content = data_buffer.read().unwrap();
        assert_eq!(*data_buffer_content, 0x12345678);
    }

    #[test]
    fn required_subgroup_size() {
        // This test checks whether required_subgroup_size works.
        // It executes a single compute shader (one invocation) that writes the subgroup size
        // to a buffer. The buffer content is then checked for the right value.

        let (device, queue) = gfx_dev_and_queue!(subgroup_size_control);

        if !device
            .physical_device()
            .properties()
            .required_subgroup_size_stages
            .unwrap_or_default()
            .intersects(ShaderStages::COMPUTE)
        {
            return;
        }

        let cs = {
            /*
            #version 450

            #extension GL_KHR_shader_subgroup_basic: enable

            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Output {
                uint write;
            } write;

            void main() {
                if (gl_GlobalInvocationID.x == 0) {
                    write.write = gl_SubgroupSize;
                }
            }
            */
            const MODULE: [u32; 246] = [
                119734787, 66304, 851979, 30, 0, 131089, 1, 131089, 61, 393227, 1, 1280527431,
                1685353262, 808793134, 0, 196622, 0, 1, 458767, 5, 4, 1852399981, 0, 9, 23, 393232,
                4, 17, 128, 1, 1, 196611, 2, 450, 655364, 1197427783, 1279741775, 1885560645,
                1953718128, 1600482425, 1701734764, 1919509599, 1769235301, 25974, 524292,
                1197427783, 1279741775, 1852399429, 1685417059, 1768185701, 1952671090, 6649449,
                589828, 1264536647, 1935626824, 1701077352, 1970495346, 1869768546, 1650421877,
                1667855201, 0, 262149, 4, 1852399981, 0, 524293, 9, 1197436007, 1633841004,
                1986939244, 1952539503, 1231974249, 68, 262149, 18, 1886680399, 29813, 327686, 18,
                0, 1953067639, 101, 262149, 20, 1953067639, 101, 393221, 23, 1398762599,
                1919378037, 1399879023, 6650473, 262215, 9, 11, 28, 327752, 18, 0, 35, 0, 196679,
                18, 2, 262215, 20, 34, 0, 262215, 20, 33, 0, 196679, 23, 0, 262215, 23, 11, 36,
                196679, 24, 0, 262215, 29, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0,
                262167, 7, 6, 3, 262176, 8, 1, 7, 262203, 8, 9, 1, 262187, 6, 10, 0, 262176, 11, 1,
                6, 131092, 14, 196638, 18, 6, 262176, 19, 12, 18, 262203, 19, 20, 12, 262165, 21,
                32, 1, 262187, 21, 22, 0, 262203, 11, 23, 1, 262176, 25, 12, 6, 262187, 6, 27, 128,
                262187, 6, 28, 1, 393260, 7, 29, 27, 28, 28, 327734, 2, 4, 0, 3, 131320, 5, 327745,
                11, 12, 9, 10, 262205, 6, 13, 12, 327850, 14, 15, 13, 10, 196855, 17, 0, 262394,
                15, 16, 17, 131320, 16, 262205, 6, 24, 23, 327745, 25, 26, 20, 22, 196670, 26, 24,
                131321, 17, 131320, 17, 65789, 65592,
            ];
            let module =
                unsafe { ShaderModule::new(&device, &ShaderModuleCreateInfo::new(&MODULE)) }
                    .unwrap();
            module.entry_point("main").unwrap()
        };

        let properties = device.physical_device().properties();
        let subgroup_size = properties.min_subgroup_size.unwrap_or(1);

        let pipeline = {
            let stage = PipelineShaderStageCreateInfo {
                required_subgroup_size: Some(subgroup_size),
                ..PipelineShaderStageCreateInfo::new(&cs)
            };
            let layout = PipelineLayout::from_stages(&device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                &device,
                None,
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let data_buffer = Buffer::from_data(
            &memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            0,
        )
        .unwrap();

        let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &device,
            &Default::default(),
        ));
        let set = DescriptorSet::new(
            &ds_allocator,
            &pipeline.layout().set_layouts()[0],
            &[WriteDescriptorSet::buffer(
                0,
                &DescriptorBufferInfo {
                    buffer: Some(data_buffer.buffer()),
                    ..Default::default()
                },
            )],
            &[],
        )
        .unwrap();

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let mut cbb = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cbb.bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap();
        unsafe { cbb.dispatch([128, 1, 1]) }.unwrap();

        let cb = cbb.build().unwrap();

        let future = now(device)
            .then_execute(queue, cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let data_buffer_content = data_buffer.read().unwrap();
        assert_eq!(*data_buffer_content, subgroup_size);
    }
}
