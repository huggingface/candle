//! Ray tracing pipeline functionality for GPU-accelerated ray tracing.
//!
//! # Overview
//!
//! Ray tracing pipelines enable high-performance ray tracing by defining a set of shader stages
//! that handle ray generation, intersection testing, and shading calculations. The pipeline
//! consists of different shader stages organized into shader groups.
//!
//! # Shader types
//!
//! ## Ray generation shader
//!
//! - Entry point for ray tracing.
//! - Generates and traces primary rays.
//! - Controls the overall ray tracing process.
//!
//! ## Intersection shaders
//!
//! - **Built-in triangle intersection** handles standard triangle geometry intersection.
//! - **Custom intersection** implements custom geometry intersection testing.
//!
//! ## Hit shaders
//!
//! - **Closest hit** executes when a ray finds its closest intersection.
//! - **Any hit** is an optional shader that runs on every potential intersection.
//!
//! ## Miss shader
//!
//! - Executes when a ray doesn't intersect any geometry.
//! - Typically handles environment mapping or background colors.
//!
//! ## Callable shader
//!
//! - Utility shader that can be called from other shader stages.
//! - Enables code reuse across different shader stages.
//!
//! # Pipeline organization
//!
//! Shaders are organized into groups:
//! - **General groups** contain ray generation, miss, or callable shaders.
//! - **Triangle hit groups** contain closest-hit and optional any-hit shaders.
//! - **Procedural hit groups** contain intersection, closest-hit, and optional any-hit shaders.
//!
//! The ray tracing pipeline uses a Shader Binding Table (SBT) to organize and access these shader
//! groups during execution.

use super::{
    cache::PipelineCache, DynamicState, PipelineCreateFlags, PipelineLayout,
    PipelineShaderStageCreateInfo, PipelineShaderStageCreateInfoExtensionsVk,
    PipelineShaderStageCreateInfoFields1Vk, PipelineShaderStageCreateInfoFields2Vk,
};
use crate::{
    buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    memory::{
        allocator::{
            align_up, AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter,
        },
        DeviceAlignment,
    },
    self_referential::self_referential,
    shader::{spirv::ExecutionModel, DescriptorBindingRequirements, EntryPoint},
    DeviceSize, StridedDeviceAddressRegion, Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use foldhash::HashMap;
use smallvec::SmallVec;
use std::{collections::hash_map::Entry, mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// Defines how the implementation should perform ray tracing operations.
///
/// This object uses the `VK_KHR_ray_tracing_pipeline` extension.
#[derive(Debug)]
pub struct RayTracingPipeline {
    handle: vk::Pipeline,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: PipelineCreateFlags,
    layout: DeviceOwnedDebugWrapper<Arc<PipelineLayout>>,

    descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    num_used_descriptor_sets: u32,

    groups: SmallVec<[RayTracingShaderGroupCreateInfo; 5]>,
    stages: OwnedPipelineShaderStages,
}

self_referential! {
    mod owned_pipeline_shader_stages {
        struct OwnedPipelineShaderStages {
            inner: SmallVec<[PipelineShaderStageCreateInfo<'_>; 5]>,
            entry_points: Vec<EntryPoint>,
        }
    }
}

impl RayTracingPipeline {
    /// Creates a new `RayTracingPipeline`, panicking on a validation error.
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
        create_info: &RayTracingPipelineCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        match Self::try_new(device, cache, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `RayTracingPipeline`.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        cache: Option<&Arc<PipelineCache>>,
        create_info: &RayTracingPipelineCreateInfo<'_>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_new(device, cache.map(|c| &**c), create_info)?;

        unsafe { Ok(Self::new_unchecked(device, cache, create_info)?) }
    }

    fn validate_new(
        device: &Arc<Device>,
        cache: Option<&PipelineCache>,
        create_info: &RayTracingPipelineCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if let Some(cache) = cache {
            assert_eq!(device, cache.device());
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
        create_info: &RayTracingPipelineCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        let handle = {
            let fields3_vk = create_info.to_vk_fields3();
            let fields2_vk = create_info.to_vk_fields2(&fields3_vk);
            let mut fields1_extensions_vk = create_info.to_vk_fields1_extensions();
            let fields1_vk = create_info.to_vk_fields1(&fields2_vk, &mut fields1_extensions_vk);
            let create_infos_vk = create_info.to_vk(&fields1_vk);

            let fns = device.fns();
            let mut output = MaybeUninit::uninit();

            unsafe {
                (fns.khr_ray_tracing_pipeline
                    .create_ray_tracing_pipelines_khr)(
                    device.handle(),
                    vk::DeferredOperationKHR::null(), // TODO: RayTracing: deferred_operation
                    cache.map_or(vk::PipelineCache::null(), |c| c.handle()),
                    1,
                    &create_infos_vk,
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

    /// Creates a new `RayTracingPipeline` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::Pipeline,
        create_info: &RayTracingPipelineCreateInfo<'_>,
    ) -> Arc<Self> {
        let &RayTracingPipelineCreateInfo {
            flags,
            stages,
            groups,
            layout,
            ..
        } = create_info;

        let mut descriptor_binding_requirements: HashMap<
            (u32, u32),
            DescriptorBindingRequirements,
        > = HashMap::default();

        for stage in stages {
            for (&loc, reqs) in stage
                .entry_point
                .info()
                .descriptor_binding_requirements
                .iter()
            {
                match descriptor_binding_requirements.entry(loc) {
                    Entry::Occupied(entry) => {
                        entry.into_mut().merge(reqs).expect(
                            "could not produce an intersection of the shader descriptor \
                            requirements",
                        );
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(reqs.clone());
                    }
                }
            }
        }

        let num_used_descriptor_sets = descriptor_binding_requirements
            .keys()
            .map(|loc| loc.0)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);

        let entry_points = stages
            .iter()
            .map(|stage| stage.entry_point.clone())
            .collect();
        let stages = OwnedPipelineShaderStages::new(entry_points, |entry_points| {
            stages
                .iter()
                .zip(entry_points)
                .map(|(stage, entry_point)| PipelineShaderStageCreateInfo {
                    entry_point,
                    _ne: crate::NE,
                    ..*stage
                })
                .collect()
        });

        Arc::new(Self {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),

            flags,
            layout: DeviceOwnedDebugWrapper(layout.clone()),

            descriptor_binding_requirements,
            num_used_descriptor_sets,

            groups: groups.iter().cloned().collect(),
            stages,
        })
    }

    /// Returns the shader groups that the pipeline was created with.
    #[inline]
    pub fn groups(&self) -> &[RayTracingShaderGroupCreateInfo] {
        &self.groups
    }

    /// Returns the shader stages that the pipeline was created with.
    #[inline]
    pub fn stages(&self) -> &[PipelineShaderStageCreateInfo<'_>] {
        self.stages.as_ref()
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

    /// Retrieves the opaque handles of shaders in the ray tracing pipeline, panicking on a
    /// validation error.
    ///
    /// Handles for `group_count` groups are retrieved, starting at `first_group`. The group
    /// indices correspond to the [`groups`] the pipeline was created with.
    ///
    /// This is a shortcut for `try_group_handles().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_group_handles`] returns a [`ValidationError`].
    ///
    /// [`groups`]: Self::groups
    /// [`try_group_handles`]: Self::try_group_handles
    #[track_caller]
    pub fn group_handles(
        &self,
        first_group: u32,
        group_count: u32,
    ) -> Result<ShaderGroupHandlesData, VulkanError> {
        match self.try_group_handles(first_group, group_count) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Retrieves the opaque handles of shaders in the ray tracing pipeline.
    ///
    /// Handles for `group_count` groups are retrieved, starting at `first_group`. The group
    /// indices correspond to the [`groups`] the pipeline was created with.
    ///
    /// [`groups`]: Self::groups
    pub fn try_group_handles(
        &self,
        first_group: u32,
        group_count: u32,
    ) -> Result<ShaderGroupHandlesData, Validated<VulkanError>> {
        self.validate_group_handles(first_group, group_count)?;

        Ok(unsafe { self.group_handles_unchecked(first_group, group_count) }?)
    }

    fn validate_group_handles(
        &self,
        first_group: u32,
        group_count: u32,
    ) -> Result<(), Box<ValidationError>> {
        if (first_group + group_count) as usize > self.groups().len() {
            Err(Box::new(ValidationError {
                problem: "the sum of `first_group` and `group_count` must be less than or equal \
                    to the number of shader groups in the pipeline"
                    .into(),
                vuids: &["VUID-vkGetRayTracingShaderGroupHandlesKHR-firstGroup-02419"],
                ..Default::default()
            }))?
        }

        // TODO: VUID-vkGetRayTracingShaderGroupHandlesKHR-pipeline-07828

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn group_handles_unchecked(
        &self,
        first_group: u32,
        group_count: u32,
    ) -> Result<ShaderGroupHandlesData, VulkanError> {
        let handle_size = self
            .device()
            .physical_device()
            .properties()
            .shader_group_handle_size
            .unwrap();

        let mut data = vec![0u8; (handle_size * group_count) as usize];
        let fns = self.device().fns();
        unsafe {
            (fns.khr_ray_tracing_pipeline
                .get_ray_tracing_shader_group_handles_khr)(
                self.device().handle(),
                self.handle(),
                first_group,
                group_count,
                data.len(),
                data.as_mut_ptr().cast(),
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(ShaderGroupHandlesData { data, handle_size })
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

impl_id_counter!(RayTracingPipeline);

unsafe impl VulkanObject for RayTracingPipeline {
    type Handle = vk::Pipeline;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for RayTracingPipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device()
    }
}

impl Drop for RayTracingPipeline {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_pipeline)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

/// Parameters to create a new `RayTracingPipeline`.
#[derive(Clone, Debug)]
pub struct RayTracingPipelineCreateInfo<'a> {
    /// Additional properties of the pipeline.
    ///
    /// The default value is empty.
    pub flags: PipelineCreateFlags,

    /// The ray tracing shader stages to use.
    ///
    /// The default value is empty, which must be overridden.
    pub stages: &'a [PipelineShaderStageCreateInfo<'a>],

    /// The shader groups to use. They reference the shader stages in `stages`.
    ///
    /// The default value is empty, which must be overridden.
    pub groups: &'a [RayTracingShaderGroupCreateInfo],

    /// The maximum recursion depth of the pipeline.
    ///
    /// The default value is `1`.
    pub max_pipeline_ray_recursion_depth: u32,

    /// The dynamic state to use.
    ///
    /// May only contain `DynamicState::RayTracingPipelineStackSize`.
    ///
    /// The default value is empty.
    pub dynamic_state: &'a [DynamicState],

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
    pub base_pipeline: Option<&'a Arc<RayTracingPipeline>>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> RayTracingPipelineCreateInfo<'a> {
    /// Returns a default `RayTracingPipelineCreateInfo` with the provided `layout`.
    #[inline]
    pub const fn new(layout: &'a Arc<PipelineLayout>) -> Self {
        Self {
            flags: PipelineCreateFlags::empty(),
            stages: &[],
            groups: &[],
            max_pipeline_ray_recursion_depth: 1,
            dynamic_state: &[],
            layout,
            base_pipeline: None,
            _ne: crate::NE,
        }
    }

    fn validate(&self, device: &Arc<Device>) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            stages,
            groups,
            layout,
            base_pipeline,
            dynamic_state,
            max_pipeline_ray_recursion_depth,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkRayTracingPipelineCreateInfoKHR-flags-parameter"])
        })?;

        if flags.intersects(PipelineCreateFlags::DERIVATIVE) {
            let base_pipeline = base_pipeline.as_ref().ok_or_else(|| {
                Box::new(ValidationError {
                    context: "flags".into(),
                    problem: "contains `PipelineCreateFlags::DERIVATIVE`, but `base_pipeline` is \
                        `None`"
                        .into(),
                    vuids: &["VUID-VkRayTracingPipelineCreateInfoKHR-flags-07984"],
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
                    vuids: &["VUID-vkCreateRayTracingPipelinesKHR-flags-03416"],
                    ..Default::default()
                }));
            }
        } else if base_pipeline.is_some() {
            return Err(Box::new(ValidationError {
                context: "flags".into(),
                problem: "does not contain `PipelineCreateFlags::DERIVATIVE`, but `base_pipeline` \
                    is `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        if stages.is_empty() {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkRayTracingPipelineCreateInfoKHR-pLibraryInfo-07999"],
                ..Default::default()
            }));
        }

        let has_raygen = stages.iter().any(|stage| {
            stage.entry_point.info().execution_model == ExecutionModel::RayGenerationKHR
        });
        if !has_raygen {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "does not contain a `RayGeneration` shader".into(),
                vuids: &["VUID-VkRayTracingPipelineCreateInfoKHR-stage-03425"],
                ..Default::default()
            }));
        }

        for (stage_index, stage) in stages.iter().enumerate() {
            stage.validate(device).map_err(|err| {
                err.add_context(format!("stages[{}]", stage_index))
                    .set_vuids(&["VUID-VkRayTracingPipelineCreateInfoKHR-pStages-parameter"])
            })?;

            let entry_point_info = stage.entry_point.info();

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
                        context: format!("stages[{}].entry_point", stage_index).into(),
                        vuids: &[
                            "VUID-VkRayTracingPipelineCreateInfoKHR-layout-07987",
                            "VUID-VkRayTracingPipelineCreateInfoKHR-layout-07988",
                            "VUID-VkRayTracingPipelineCreateInfoKHR-layout-07990",
                            "VUID-VkRayTracingPipelineCreateInfoKHR-layout-07991",
                        ],
                        ..ValidationError::from_error(err)
                    })
                })?;
        }

        if groups.is_empty() {
            return Err(Box::new(ValidationError {
                context: "groups".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkRayTracingPipelineCreateInfoKHR-flags-08700"],
                ..Default::default()
            }));
        }

        for group in groups {
            group.validate(stages).map_err(|err| {
                err.add_context("groups")
                    .set_vuids(&["VUID-VkRayTracingPipelineCreateInfoKHR-pGroups-parameter"])
            })?;
        }

        // TODO: Enable
        // if dynamic_state
        //     .iter()
        //     .any(|&state| state != DynamicState::RayTracingPipelineStackSize)
        // {
        //     return Err(Box::new(ValidationError {
        //         problem:
        //             format!("`dynamic_state` contains a dynamic state other than
        // RayTracingPipelineStackSize: {:?}", dynamic_state).into(),         vuids:
        // &["VUID-VkRayTracingPipelineCreateInfoKHR-pDynamicStates-03602"],
        //         ..Default::default()
        //     }));
        // }

        if !dynamic_state.is_empty() {
            todo!("dynamic state for ray tracing pipelines is not yet supported");
        }

        let max_ray_recursion_depth = device
            .physical_device()
            .properties()
            .max_ray_recursion_depth
            .unwrap();

        if max_pipeline_ray_recursion_depth > max_ray_recursion_depth {
            return Err(Box::new(ValidationError {
                context: "max_pipeline_ray_recursion_depth".into(),
                problem: "is greater than the `max_ray_recursion_depth` device property".into(),
                vuids: &[
                    "VUID-VkRayTracingPipelineCreateInfoKHR-maxPipelineRayRecursionDepth-03589",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a RayTracingPipelineCreateInfoFields1Vk<'_>,
    ) -> vk::RayTracingPipelineCreateInfoKHR<'a> {
        let &Self {
            flags,
            max_pipeline_ray_recursion_depth,
            layout,
            base_pipeline,
            ..
        } = self;

        let RayTracingPipelineCreateInfoFields1Vk {
            stages_vk,
            groups_vk,
            dynamic_state_vk,
        } = fields1_vk;

        let mut val_vk = vk::RayTracingPipelineCreateInfoKHR::default()
            .flags(flags.into())
            .stages(stages_vk)
            .groups(groups_vk)
            .layout(layout.handle())
            .max_pipeline_ray_recursion_depth(max_pipeline_ray_recursion_depth)
            .base_pipeline_handle(
                base_pipeline
                    .as_ref()
                    .map_or(vk::Pipeline::null(), |p| p.handle()),
            )
            .base_pipeline_index(-1);

        if let Some(dynamic_state_vk) = dynamic_state_vk {
            val_vk = val_vk.dynamic_state(dynamic_state_vk);
        }

        val_vk
    }

    pub(crate) fn to_vk_fields1(
        &self,
        fields2_vk: &'a RayTracingPipelineCreateInfoFields2Vk<'_>,
        extensions_vk: &'a mut RayTracingPipelineCreateInfoFields1ExtensionsVk,
    ) -> RayTracingPipelineCreateInfoFields1Vk<'a> {
        let &Self { stages, groups, .. } = self;
        let RayTracingPipelineCreateInfoFields2Vk {
            stages_fields1_vk,
            dynamic_states_vk,
        } = fields2_vk;
        let RayTracingPipelineCreateInfoFields1ExtensionsVk {
            stages_extensions_vk,
        } = extensions_vk;

        let stages_vk: SmallVec<[_; 5]> = stages
            .iter()
            .zip(stages_fields1_vk)
            .zip(stages_extensions_vk)
            .map(|((stage, fields1), fields1_extensions_vk)| {
                stage.to_vk(fields1, fields1_extensions_vk)
            })
            .collect();

        let groups_vk = groups
            .iter()
            .map(RayTracingShaderGroupCreateInfo::to_vk)
            .collect();

        let dynamic_state_vk = (!dynamic_states_vk.is_empty()).then(|| {
            vk::PipelineDynamicStateCreateInfo::default()
                .flags(vk::PipelineDynamicStateCreateFlags::empty())
                .dynamic_states(dynamic_states_vk)
        });

        RayTracingPipelineCreateInfoFields1Vk {
            stages_vk,
            groups_vk,
            dynamic_state_vk,
        }
    }

    pub(crate) fn to_vk_fields1_extensions(
        &self,
    ) -> RayTracingPipelineCreateInfoFields1ExtensionsVk {
        let &Self { stages, .. } = self;

        let stages_extensions_vk = stages
            .iter()
            .map(|stage| stage.to_vk_extensions())
            .collect();

        RayTracingPipelineCreateInfoFields1ExtensionsVk {
            stages_extensions_vk,
        }
    }

    pub(crate) fn to_vk_fields2(
        &self,
        fields3_vk: &'a RayTracingPipelineCreateInfoFields3Vk,
    ) -> RayTracingPipelineCreateInfoFields2Vk<'a> {
        let &Self {
            stages,
            dynamic_state,
            ..
        } = self;

        let stages_fields1_vk = stages
            .iter()
            .zip(fields3_vk.stages_fields2_vk.iter())
            .map(|(stage, fields3)| stage.to_vk_fields1(fields3))
            .collect();

        let dynamic_states_vk = dynamic_state.iter().copied().map(Into::into).collect();

        RayTracingPipelineCreateInfoFields2Vk {
            stages_fields1_vk,
            dynamic_states_vk,
        }
    }

    pub(crate) fn to_vk_fields3(&self) -> RayTracingPipelineCreateInfoFields3Vk {
        let &Self { stages, .. } = self;

        let stages_fields2_vk = stages.iter().map(|stage| stage.to_vk_fields2()).collect();

        RayTracingPipelineCreateInfoFields3Vk { stages_fields2_vk }
    }
}

/// Enum representing different types of Ray Tracing Shader Groups.
///
/// Contains the index of the shader to use for each type of shader group. The index corresponds to
/// the position of the shader in the [`stages`] field of the [`RayTracingPipelineCreateInfo`].
///
/// [`stages`]: RayTracingPipelineCreateInfo::stages
#[derive(Debug, Clone)]
pub enum RayTracingShaderGroupCreateInfo {
    /// General shader group type, typically used for ray generation and miss shaders.
    ///
    /// Contains a single shader stage that can be:
    /// - Ray generation shader
    /// - Miss shader
    /// - Callable shader
    General {
        /// Index of the general shader stage.
        general_shader: u32,
    },

    /// Procedural hit shader group type, used for custom intersection testing.
    ///
    /// Used when implementing custom intersection shapes or volumes. Requires an intersection
    /// shader and can optionally include closest hit and any hit shaders.
    ProceduralHit {
        /// Optional index of the closest hit shader stage.
        closest_hit_shader: Option<u32>,
        /// Optional index of the any hit shader stage.
        any_hit_shader: Option<u32>,
        /// Index of the intersection shader stage.
        intersection_shader: u32,
    },

    /// Triangle hit shader group type, used for built-in triangle intersection.
    ///
    /// Used for standard triangle geometry intersection testing. Can optionally include closest
    /// hit and any hit shaders.
    TrianglesHit {
        /// Optional index of the closest hit shader stage.
        closest_hit_shader: Option<u32>,
        /// Optional index of the any hit shader stage.
        any_hit_shader: Option<u32>,
    },
}

impl RayTracingShaderGroupCreateInfo {
    fn validate(
        &self,
        stages: &[PipelineShaderStageCreateInfo<'_>],
    ) -> Result<(), Box<ValidationError>> {
        let get_shader_type =
            |shader: u32| stages[shader as usize].entry_point.info().execution_model;

        match self {
            RayTracingShaderGroupCreateInfo::General { general_shader } => {
                match get_shader_type(*general_shader) {
                    ExecutionModel::RayGenerationKHR
                    | ExecutionModel::MissKHR
                    | ExecutionModel::CallableKHR => Ok(()),
                    _ => Err(Box::new(ValidationError {
                        problem: "general shader in `GENERAL` group must be a `RayGeneration`, \
                            `Miss`, or `Callable` shader"
                            .into(),
                        vuids: &["VUID-VkRayTracingShaderGroupCreateInfoKHR-type-03474"],
                        ..Default::default()
                    })),
                }?;
            }
            RayTracingShaderGroupCreateInfo::ProceduralHit {
                intersection_shader,
                any_hit_shader,
                closest_hit_shader,
            } => {
                if get_shader_type(*intersection_shader) != ExecutionModel::IntersectionKHR {
                    return Err(Box::new(ValidationError {
                        problem: "intersection shader in `PROCEDURAL_HIT_GROUP` must be an \
                            `Intersection` shader"
                            .into(),
                        vuids: &["VUID-VkRayTracingShaderGroupCreateInfoKHR-type-03476"],
                        ..Default::default()
                    }));
                }

                if let Some(any_hit_shader) = any_hit_shader {
                    if get_shader_type(*any_hit_shader) != ExecutionModel::AnyHitKHR {
                        return Err(Box::new(ValidationError {
                            problem: "any hit shader must be an `AnyHit` shader".into(),
                            vuids: &[
                                "VUID-VkRayTracingShaderGroupCreateInfoKHR-anyHitShader-03479",
                            ],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(closest_hit_shader) = closest_hit_shader {
                    if get_shader_type(*closest_hit_shader) != ExecutionModel::ClosestHitKHR {
                        return Err(Box::new(ValidationError {
                            problem: "closest hit shader must be a `ClosestHit` shader".into(),
                            vuids: &[
                                "VUID-VkRayTracingShaderGroupCreateInfoKHR-closestHitShader-03478",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
            RayTracingShaderGroupCreateInfo::TrianglesHit {
                any_hit_shader,
                closest_hit_shader,
            } => {
                if let Some(any_hit_shader) = any_hit_shader {
                    if get_shader_type(*any_hit_shader) != ExecutionModel::AnyHitKHR {
                        return Err(Box::new(ValidationError {
                            problem: "any hit shader must be an `AnyHit` shader".into(),
                            vuids: &[
                                "VUID-VkRayTracingShaderGroupCreateInfoKHR-anyHitShader-03479",
                            ],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(closest_hit_shader) = closest_hit_shader {
                    if get_shader_type(*closest_hit_shader) != ExecutionModel::ClosestHitKHR {
                        return Err(Box::new(ValidationError {
                            problem: "closest hit shader must be a `ClosestHit` shader".into(),
                            vuids: &[
                                "VUID-VkRayTracingShaderGroupCreateInfoKHR-closestHitShader-03478",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::RayTracingShaderGroupCreateInfoKHR<'static> {
        match self {
            RayTracingShaderGroupCreateInfo::General { general_shader } => {
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(*general_shader)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR)
            }
            RayTracingShaderGroupCreateInfo::ProceduralHit {
                closest_hit_shader,
                any_hit_shader,
                intersection_shader,
            } => vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(closest_hit_shader.unwrap_or(vk::SHADER_UNUSED_KHR))
                .any_hit_shader(any_hit_shader.unwrap_or(vk::SHADER_UNUSED_KHR))
                .intersection_shader(*intersection_shader),
            RayTracingShaderGroupCreateInfo::TrianglesHit {
                closest_hit_shader,
                any_hit_shader,
            } => vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(closest_hit_shader.unwrap_or(vk::SHADER_UNUSED_KHR))
                .any_hit_shader(any_hit_shader.unwrap_or(vk::SHADER_UNUSED_KHR))
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        }
    }
}

pub(crate) struct RayTracingPipelineCreateInfoFields1Vk<'a> {
    pub(crate) stages_vk: SmallVec<[vk::PipelineShaderStageCreateInfo<'a>; 5]>,
    pub(crate) groups_vk: SmallVec<[vk::RayTracingShaderGroupCreateInfoKHR<'static>; 5]>,
    pub(crate) dynamic_state_vk: Option<vk::PipelineDynamicStateCreateInfo<'a>>,
}

pub(crate) struct RayTracingPipelineCreateInfoFields1ExtensionsVk {
    pub(crate) stages_extensions_vk: SmallVec<[PipelineShaderStageCreateInfoExtensionsVk; 5]>,
}

pub(crate) struct RayTracingPipelineCreateInfoFields2Vk<'a> {
    pub(crate) stages_fields1_vk: SmallVec<[PipelineShaderStageCreateInfoFields1Vk<'a>; 5]>,
    pub(crate) dynamic_states_vk: SmallVec<[vk::DynamicState; 4]>,
}

pub(crate) struct RayTracingPipelineCreateInfoFields3Vk {
    pub(crate) stages_fields2_vk: SmallVec<[PipelineShaderStageCreateInfoFields2Vk; 5]>,
}

/// Holds the data returned by [`RayTracingPipeline::group_handles`].
#[derive(Clone, Debug)]
pub struct ShaderGroupHandlesData {
    data: Vec<u8>,
    handle_size: u32,
}

impl ShaderGroupHandlesData {
    /// Returns the opaque handle data as one blob.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns the shader group handle size of the device.
    #[inline]
    pub fn handle_size(&self) -> u32 {
        self.handle_size
    }

    /// Returns an iterator over the data of each group handle.
    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> + use<'_> {
        self.data().chunks_exact(self.handle_size as usize)
    }
}

/// An object that holds the strided addresses of the shader groups in a shader binding table.
#[derive(Debug, Clone)]
pub struct ShaderBindingTableAddresses {
    /// The address of the ray generation shader group handle.
    pub raygen: StridedDeviceAddressRegion,
    /// The address of the miss shader group handles.
    pub miss: StridedDeviceAddressRegion,
    /// The address of the hit shader group handles.
    pub hit: StridedDeviceAddressRegion,
    /// The address of the callable shader group handles.
    pub callable: StridedDeviceAddressRegion,
}

/// An object that holds the shader binding table buffer and its addresses.
#[derive(Debug, Clone)]
pub struct ShaderBindingTable {
    addresses: ShaderBindingTableAddresses,
    _buffer: Subbuffer<[u8]>,
}

impl ShaderBindingTable {
    /// Automatically creates a shader binding table from a ray tracing pipeline, panicking on a
    /// validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[track_caller]
    pub fn new(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        ray_tracing_pipeline: &RayTracingPipeline,
    ) -> Result<Self, VulkanError> {
        Self::new_inner(allocator.clone().as_dyn(), ray_tracing_pipeline)
    }

    #[track_caller]
    fn new_inner(
        allocator: Arc<dyn MemoryAllocator>,
        ray_tracing_pipeline: &RayTracingPipeline,
    ) -> Result<Self, VulkanError> {
        match Self::try_new_inner(allocator, ray_tracing_pipeline) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Automatically creates a shader binding table from a ray tracing pipeline.
    pub fn try_new(
        allocator: &Arc<impl MemoryAllocator + ?Sized>,
        ray_tracing_pipeline: &RayTracingPipeline,
    ) -> Result<Self, Validated<VulkanError>> {
        Self::try_new_inner(allocator.clone().as_dyn(), ray_tracing_pipeline)
    }

    fn try_new_inner(
        allocator: Arc<dyn MemoryAllocator>,
        ray_tracing_pipeline: &RayTracingPipeline,
    ) -> Result<Self, Validated<VulkanError>> {
        // VUID-vkCmdTraceRaysKHR-size-04023
        // There should be exactly one raygen shader group.
        let mut raygen_shader_handle = None;
        let mut miss_shader_handles = Vec::new();
        let mut hit_shader_handles = Vec::new();
        let mut callable_shader_handles = Vec::new();

        let handle_data =
            ray_tracing_pipeline.group_handles(0, ray_tracing_pipeline.groups().len() as u32)?;
        let mut handle_iter = handle_data.iter();

        for group in ray_tracing_pipeline.groups() {
            let handle = handle_iter.next().unwrap();
            match group {
                RayTracingShaderGroupCreateInfo::General { general_shader } => {
                    match ray_tracing_pipeline.stages()[*general_shader as usize]
                        .entry_point
                        .info()
                        .execution_model
                    {
                        ExecutionModel::RayGenerationKHR => {
                            raygen_shader_handle = Some(handle);
                        }
                        ExecutionModel::MissKHR => {
                            miss_shader_handles.push(handle);
                        }
                        ExecutionModel::CallableKHR => {
                            callable_shader_handles.push(handle);
                        }
                        _ => {
                            panic!("Unexpected shader type in general shader group");
                        }
                    }
                }
                RayTracingShaderGroupCreateInfo::ProceduralHit { .. }
                | RayTracingShaderGroupCreateInfo::TrianglesHit { .. } => {
                    hit_shader_handles.push(handle);
                }
            }
        }

        let raygen_shader_handle = raygen_shader_handle.expect("no raygen shader group found");

        let properties = ray_tracing_pipeline.device().physical_device().properties();
        let handle_size_aligned = align_up(
            handle_data.handle_size() as u64,
            DeviceAlignment::new(properties.shader_group_handle_alignment.unwrap() as u64).unwrap(),
        );

        let shader_group_base_alignment =
            DeviceAlignment::new(properties.shader_group_base_alignment.unwrap() as u64).unwrap();

        let raygen_stride = align_up(handle_size_aligned, shader_group_base_alignment);

        let mut raygen = StridedDeviceAddressRegion {
            stride: raygen_stride,
            size: raygen_stride,
            device_address: 0,
        };
        let mut miss = StridedDeviceAddressRegion {
            stride: handle_size_aligned,
            size: align_up(
                handle_size_aligned * miss_shader_handles.len() as u64,
                shader_group_base_alignment,
            ),
            device_address: 0,
        };
        let mut hit = StridedDeviceAddressRegion {
            stride: handle_size_aligned,
            size: align_up(
                handle_size_aligned * hit_shader_handles.len() as u64,
                shader_group_base_alignment,
            ),
            device_address: 0,
        };
        let mut callable = StridedDeviceAddressRegion {
            stride: handle_size_aligned,
            size: align_up(
                handle_size_aligned * callable_shader_handles.len() as u64,
                shader_group_base_alignment,
            ),
            device_address: 0,
        };

        let sbt_buffer = new_bytes_buffer_with_alignment(
            allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::SHADER_BINDING_TABLE,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            raygen.size + miss.size + hit.size + callable.size,
            shader_group_base_alignment,
        )
        .expect("todo: raytracing: better error type for buffer errors");

        raygen.device_address = sbt_buffer.buffer().device_address().get();
        miss.device_address = raygen.device_address + raygen.size;
        hit.device_address = miss.device_address + miss.size;
        callable.device_address = hit.device_address + hit.size;

        {
            let mut sbt_buffer_write = sbt_buffer.write().unwrap();

            let handle_size = handle_data.handle_size() as usize;
            sbt_buffer_write[..handle_size].copy_from_slice(raygen_shader_handle);
            let mut offset = raygen.size as usize;
            for handle in miss_shader_handles {
                sbt_buffer_write[offset..offset + handle_size].copy_from_slice(handle);
                offset += miss.stride as usize;
            }
            offset = (raygen.size + miss.size) as usize;
            for handle in hit_shader_handles {
                sbt_buffer_write[offset..offset + handle_size].copy_from_slice(handle);
                offset += hit.stride as usize;
            }
            offset = (raygen.size + miss.size + hit.size) as usize;
            for handle in callable_shader_handles {
                sbt_buffer_write[offset..offset + handle_size].copy_from_slice(handle);
                offset += callable.stride as usize;
            }
        }

        Ok(Self {
            addresses: ShaderBindingTableAddresses {
                raygen,
                miss,
                hit,
                callable,
            },
            _buffer: sbt_buffer,
        })
    }

    /// Returns the addresses of the shader groups in the shader binding table.
    #[inline]
    pub fn addresses(&self) -> &ShaderBindingTableAddresses {
        &self.addresses
    }
}

fn new_bytes_buffer_with_alignment(
    allocator: Arc<dyn MemoryAllocator>,
    create_info: &BufferCreateInfo<'_>,
    allocation_info: &AllocationCreateInfo<'_>,
    size: DeviceSize,
    alignment: DeviceAlignment,
) -> Result<Subbuffer<[u8]>, Validated<AllocateBufferError>> {
    let layout = DeviceLayout::from_size_alignment(size, alignment.as_devicesize()).unwrap();
    let buffer = Subbuffer::new(Buffer::try_new_inner(
        allocator,
        create_info,
        allocation_info,
        layout,
    )?);
    Ok(buffer)
}
