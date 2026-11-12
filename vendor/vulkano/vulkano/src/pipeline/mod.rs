//! Describes a processing operation that will execute on the Vulkan device.
//!
//! In Vulkan, before you can add a draw or a compute command to a command buffer you have to
//! create a *pipeline object* that describes this command.
//!
//! When you create a pipeline object, the implementation will usually generate some GPU machine
//! code that will execute the operation (similar to a compiler that generates an executable for
//! the CPU). Consequently it is a CPU-intensive operation that should be performed at
//! initialization or during a loading screen.
//!
//! The state and shaders of a pipeline are compiled into zero or more device-specific
//! *executables*. If the [`pipeline_executable_info`] feature is enabled, then the properties of
//! these executables can be queried with [`Pipeline::executable_properties`]. Their compile time
//! statistics and internal representations can be queried with
//! [`Pipeline::executable_statistics`] and [`Pipeline::executable_internal_representations`],
//! respectively, provided that the pipeline was created with the matching
//! [`PipelineCreateFlags::CAPTURE_STATISTICS`] or
//! [`PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS`] flag. This is intended for use by
//! debugging and performance tools.
//!
//! [`pipeline_executable_info`]: crate::device::DeviceFeatures::pipeline_executable_info

pub use self::{
    compute::ComputePipeline, graphics::GraphicsPipeline, layout::PipelineLayout,
    ray_tracing::RayTracingPipeline, shader::*,
};
use crate::{
    device::{Device, DeviceOwned},
    macros::{vulkan_bitflags, vulkan_enum},
    shader::{DescriptorBindingRequirements, ShaderStages},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use foldhash::HashMap;
use std::{ffi::CStr, ptr, sync::Arc};

pub mod cache;
pub mod compute;
pub mod graphics;
pub mod layout;
pub mod ray_tracing;
pub(crate) mod shader;

/// An enum of the different pipeline types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Pipeline<'a> {
    Compute(&'a Arc<ComputePipeline>),
    Graphics(&'a Arc<GraphicsPipeline>),
    RayTracing(&'a Arc<RayTracingPipeline>),
}

impl Pipeline<'_> {
    /// Returns the bind point of this pipeline.
    #[inline]
    pub fn bind_point(&self) -> PipelineBindPoint {
        match self {
            Pipeline::Compute(_) => PipelineBindPoint::Compute,
            Pipeline::Graphics(_) => PipelineBindPoint::Graphics,
            Pipeline::RayTracing(_) => PipelineBindPoint::RayTracing,
        }
    }

    /// Returns the flags that the pipeline was created with.
    #[inline]
    pub fn flags(&self) -> PipelineCreateFlags {
        match self {
            Pipeline::Compute(compute_pipeline) => compute_pipeline.flags(),
            Pipeline::Graphics(graphics_pipeline) => graphics_pipeline.flags(),
            Pipeline::RayTracing(ray_tracing_pipeline) => ray_tracing_pipeline.flags(),
        }
    }

    /// Returns the pipeline layout used in this pipeline.
    #[inline]
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        match self {
            Pipeline::Compute(compute_pipeline) => compute_pipeline.layout(),
            Pipeline::Graphics(graphics_pipeline) => graphics_pipeline.layout(),
            Pipeline::RayTracing(ray_tracing_pipeline) => ray_tracing_pipeline.layout(),
        }
    }

    /// Returns the number of descriptor sets actually accessed by this pipeline. This may be less
    /// than the number of sets in the pipeline layout.
    #[inline]
    pub fn num_used_descriptor_sets(&self) -> u32 {
        match self {
            Pipeline::Compute(compute_pipeline) => compute_pipeline.num_used_descriptor_sets(),
            Pipeline::Graphics(graphics_pipeline) => graphics_pipeline.num_used_descriptor_sets(),
            Pipeline::RayTracing(ray_tracing_pipeline) => {
                ray_tracing_pipeline.num_used_descriptor_sets()
            }
        }
    }

    /// Returns a reference to the descriptor binding requirements for this pipeline.
    #[inline]
    pub fn descriptor_binding_requirements(
        &self,
    ) -> &HashMap<(u32, u32), DescriptorBindingRequirements> {
        match self {
            Pipeline::Compute(compute_pipeline) => {
                compute_pipeline.descriptor_binding_requirements()
            }
            Pipeline::Graphics(graphics_pipeline) => {
                graphics_pipeline.descriptor_binding_requirements()
            }
            Pipeline::RayTracing(ray_tracing_pipeline) => {
                ray_tracing_pipeline.descriptor_binding_requirements()
            }
        }
    }

    /// Retrieves the properties of the executables that the pipeline was compiled into, panicking
    /// on a validation error.
    ///
    /// The [`pipeline_executable_info`] feature must be enabled on the device.
    ///
    /// This is a shortcut for `try_executable_properties().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_executable_properties`] returns a [`ValidationError`].
    ///
    /// [`pipeline_executable_info`]: crate::device::DeviceFeatures::pipeline_executable_info
    /// [`try_executable_properties`]: Self::try_executable_properties
    #[inline]
    #[track_caller]
    pub fn executable_properties(&self) -> Result<Vec<PipelineExecutableProperties>, VulkanError> {
        match self.try_executable_properties() {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Retrieves the properties of the executables that the pipeline was compiled into.
    ///
    /// The [`pipeline_executable_info`] feature must be enabled on the device.
    ///
    /// [`pipeline_executable_info`]: crate::device::DeviceFeatures::pipeline_executable_info
    #[inline]
    pub fn try_executable_properties(
        &self,
    ) -> Result<Vec<PipelineExecutableProperties>, Validated<VulkanError>> {
        self.validate_executable_properties()?;

        Ok(unsafe { self.executable_properties_unchecked() }?)
    }

    fn validate_executable_properties(&self) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_features().pipeline_executable_info {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "pipeline_executable_info",
                )])]),
                vuids: &["VUID-vkGetPipelineExecutablePropertiesKHR-pipelineExecutableInfo-03270"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn executable_properties_unchecked(
        &self,
    ) -> Result<Vec<PipelineExecutableProperties>, VulkanError> {
        let device = self.device();
        let pipeline_info_vk = vk::PipelineInfoKHR::default().pipeline(self.handle());
        let fns = device.fns();

        let properties_vk = loop {
            let mut count = 0;
            unsafe {
                (fns.khr_pipeline_executable_properties
                    .get_pipeline_executable_properties_khr)(
                    device.handle(),
                    &pipeline_info_vk,
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut properties_vk =
                vec![vk::PipelineExecutablePropertiesKHR::default(); count as usize];
            let result = unsafe {
                (fns.khr_pipeline_executable_properties
                    .get_pipeline_executable_properties_khr)(
                    device.handle(),
                    &pipeline_info_vk,
                    &mut count,
                    properties_vk.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { properties_vk.set_len(count as usize) };
                    break properties_vk;
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        Ok(properties_vk
            .iter()
            .map(PipelineExecutableProperties::from_vk)
            .collect())
    }

    /// Retrieves the compile time statistics of one of the executables that the pipeline was
    /// compiled into, panicking on a validation error.
    ///
    /// The [`pipeline_executable_info`] feature must be enabled on the device, and the pipeline
    /// must have been created with the [`PipelineCreateFlags::CAPTURE_STATISTICS`] flag.
    ///
    /// `executable_index` is an index into the executables returned by
    /// [`executable_properties`], and must be less than the number of those executables.
    ///
    /// This is a shortcut for `try_executable_statistics().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_executable_statistics`] returns a [`ValidationError`].
    ///
    /// [`pipeline_executable_info`]: crate::device::DeviceFeatures::pipeline_executable_info
    /// [`executable_properties`]: Self::executable_properties
    /// [`try_executable_statistics`]: Self::try_executable_statistics
    #[inline]
    #[track_caller]
    pub fn executable_statistics(
        &self,
        executable_index: u32,
    ) -> Result<Vec<PipelineExecutableStatistic>, VulkanError> {
        match self.try_executable_statistics(executable_index) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Retrieves the compile time statistics of one of the executables that the pipeline was
    /// compiled into.
    ///
    /// The [`pipeline_executable_info`] feature must be enabled on the device, and the pipeline
    /// must have been created with the [`PipelineCreateFlags::CAPTURE_STATISTICS`] flag.
    ///
    /// `executable_index` is an index into the executables returned by
    /// [`executable_properties`], and must be less than the number of those executables.
    ///
    /// [`pipeline_executable_info`]: crate::device::DeviceFeatures::pipeline_executable_info
    /// [`executable_properties`]: Self::executable_properties
    #[inline]
    pub fn try_executable_statistics(
        &self,
        executable_index: u32,
    ) -> Result<Vec<PipelineExecutableStatistic>, Validated<VulkanError>> {
        self.validate_executable_statistics()?;

        Ok(unsafe { self.executable_statistics_unchecked(executable_index) }?)
    }

    fn validate_executable_statistics(&self) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_features().pipeline_executable_info {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "pipeline_executable_info",
                )])]),
                vuids: &["VUID-vkGetPipelineExecutableStatisticsKHR-pipelineExecutableInfo-03272"],
                ..Default::default()
            }));
        }

        if !self
            .flags()
            .intersects(PipelineCreateFlags::CAPTURE_STATISTICS)
        {
            return Err(Box::new(ValidationError {
                context: "self.flags()".into(),
                problem: "does not contain `PipelineCreateFlags::CAPTURE_STATISTICS`".into(),
                vuids: &["VUID-vkGetPipelineExecutableStatisticsKHR-pipeline-03274"],
                ..Default::default()
            }));
        }

        // TODO: VUID-VkPipelineExecutableInfoKHR-executableIndex-03275

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn executable_statistics_unchecked(
        &self,
        executable_index: u32,
    ) -> Result<Vec<PipelineExecutableStatistic>, VulkanError> {
        let device = self.device();
        let executable_info_vk = vk::PipelineExecutableInfoKHR::default()
            .pipeline(self.handle())
            .executable_index(executable_index);
        let fns = device.fns();

        let statistics_vk = loop {
            let mut count = 0;
            unsafe {
                (fns.khr_pipeline_executable_properties
                    .get_pipeline_executable_statistics_khr)(
                    device.handle(),
                    &executable_info_vk,
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut statistics_vk =
                vec![vk::PipelineExecutableStatisticKHR::default(); count as usize];
            let result = unsafe {
                (fns.khr_pipeline_executable_properties
                    .get_pipeline_executable_statistics_khr)(
                    device.handle(),
                    &executable_info_vk,
                    &mut count,
                    statistics_vk.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { statistics_vk.set_len(count as usize) };
                    break statistics_vk;
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        Ok(statistics_vk
            .iter()
            .filter_map(PipelineExecutableStatistic::from_vk)
            .collect())
    }

    /// Retrieves the internal representations of one of the executables that the pipeline was
    /// compiled into, panicking on a validation error.
    ///
    /// The implementation should order the internal representations in the order in which they
    /// occur in the compiled pipeline, with the final shader assembly (if any) last.
    ///
    /// The [`pipeline_executable_info`] feature must be enabled on the device, and the pipeline
    /// must have been created with the
    /// [`PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS`] flag.
    ///
    /// `executable_index` is an index into the executables returned by
    /// [`executable_properties`], and must be less than the number of those executables.
    ///
    /// This is a shortcut for
    /// `try_executable_internal_representations().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_executable_internal_representations`] returns a [`ValidationError`].
    ///
    /// [`pipeline_executable_info`]: crate::device::DeviceFeatures::pipeline_executable_info
    /// [`executable_properties`]: Self::executable_properties
    /// [`try_executable_internal_representations`]: Self::try_executable_internal_representations
    #[inline]
    #[track_caller]
    pub fn executable_internal_representations(
        &self,
        executable_index: u32,
    ) -> Result<Vec<PipelineExecutableInternalRepresentation>, VulkanError> {
        match self.try_executable_internal_representations(executable_index) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Retrieves the internal representations of one of the executables that the pipeline was
    /// compiled into.
    ///
    /// The implementation should order the internal representations in the order in which they
    /// occur in the compiled pipeline, with the final shader assembly (if any) last.
    ///
    /// The [`pipeline_executable_info`] feature must be enabled on the device, and the pipeline
    /// must have been created with the
    /// [`PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS`] flag.
    ///
    /// `executable_index` is an index into the executables returned by
    /// [`executable_properties`], and must be less than the number of those executables.
    ///
    /// [`pipeline_executable_info`]: crate::device::DeviceFeatures::pipeline_executable_info
    /// [`executable_properties`]: Self::executable_properties
    #[inline]
    pub fn try_executable_internal_representations(
        &self,
        executable_index: u32,
    ) -> Result<Vec<PipelineExecutableInternalRepresentation>, Validated<VulkanError>> {
        self.validate_executable_internal_representations()?;

        Ok(unsafe { self.executable_internal_representations_unchecked(executable_index) }?)
    }

    fn validate_executable_internal_representations(&self) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_features().pipeline_executable_info {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "pipeline_executable_info",
                )])]),
                vuids: &[
                    "VUID-vkGetPipelineExecutableInternalRepresentationsKHR-pipelineExecutableInfo-03276",
                ],
                ..Default::default()
            }));
        }

        if !self
            .flags()
            .intersects(PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS)
        {
            return Err(Box::new(ValidationError {
                context: "self.flags()".into(),
                problem: "does not contain \
                    `PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS`"
                    .into(),
                vuids: &["VUID-vkGetPipelineExecutableInternalRepresentationsKHR-pipeline-03278"],
                ..Default::default()
            }));
        }

        // TODO: VUID-VkPipelineExecutableInfoKHR-executableIndex-03275

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn executable_internal_representations_unchecked(
        &self,
        executable_index: u32,
    ) -> Result<Vec<PipelineExecutableInternalRepresentation>, VulkanError> {
        let device = self.device();
        let executable_info_vk = vk::PipelineExecutableInfoKHR::default()
            .pipeline(self.handle())
            .executable_index(executable_index);
        let fns = device.fns();

        let (internal_representations_vk, data) = loop {
            let mut count = 0;
            unsafe {
                (fns.khr_pipeline_executable_properties
                    .get_pipeline_executable_internal_representations_khr)(
                    device.handle(),
                    &executable_info_vk,
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut internal_representations_vk =
                vec![vk::PipelineExecutableInternalRepresentationKHR::default(); count as usize];
            let result = unsafe {
                (fns.khr_pipeline_executable_properties
                    .get_pipeline_executable_internal_representations_khr)(
                    device.handle(),
                    &executable_info_vk,
                    &mut count,
                    internal_representations_vk.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { internal_representations_vk.set_len(count as usize) };
                }
                vk::Result::INCOMPLETE => continue,
                err => return Err(VulkanError::from(err)),
            }

            // Retrieve the data itself, into buffers sized by the `data_size` values from above.
            let mut data: Vec<Vec<u8>> = internal_representations_vk
                .iter()
                .map(|val_vk| Vec::with_capacity(val_vk.data_size))
                .collect();

            for (val_vk, data_vk) in internal_representations_vk.iter_mut().zip(&mut data) {
                val_vk.p_data = data_vk.as_mut_ptr().cast();
            }

            let mut count = internal_representations_vk.len() as u32;
            let result = unsafe {
                (fns.khr_pipeline_executable_properties
                    .get_pipeline_executable_internal_representations_khr)(
                    device.handle(),
                    &executable_info_vk,
                    &mut count,
                    internal_representations_vk.as_mut_ptr(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { internal_representations_vk.set_len(count as usize) };
                    unsafe { data.set_len(count as usize) };
                    break (internal_representations_vk, data);
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        Ok(internal_representations_vk
            .iter()
            .zip(data)
            .map(|(val_vk, mut data)| {
                unsafe { data.set_len(val_vk.data_size) };

                PipelineExecutableInternalRepresentation::from_vk(val_vk, data)
            })
            .collect())
    }
}

impl<'a> From<&'a Arc<ComputePipeline>> for Pipeline<'a> {
    #[inline]
    fn from(compute_pipeline: &'a Arc<ComputePipeline>) -> Self {
        Pipeline::Compute(compute_pipeline)
    }
}

impl<'a> From<&'a Arc<GraphicsPipeline>> for Pipeline<'a> {
    #[inline]
    fn from(graphics_pipeline: &'a Arc<GraphicsPipeline>) -> Self {
        Pipeline::Graphics(graphics_pipeline)
    }
}

impl<'a> From<&'a Arc<RayTracingPipeline>> for Pipeline<'a> {
    #[inline]
    fn from(ray_tracing_pipeline: &'a Arc<RayTracingPipeline>) -> Self {
        Pipeline::RayTracing(ray_tracing_pipeline)
    }
}

unsafe impl VulkanObject for Pipeline<'_> {
    type Handle = vk::Pipeline;

    #[inline]
    fn handle(&self) -> Self::Handle {
        match self {
            Pipeline::Compute(compute_pipeline) => compute_pipeline.handle(),
            Pipeline::Graphics(graphics_pipeline) => graphics_pipeline.handle(),
            Pipeline::RayTracing(ray_tracing_pipeline) => ray_tracing_pipeline.handle(),
        }
    }
}

unsafe impl DeviceOwned for Pipeline<'_> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        match self {
            Pipeline::Compute(compute_pipeline) => compute_pipeline.device(),
            Pipeline::Graphics(graphics_pipeline) => graphics_pipeline.device(),
            Pipeline::RayTracing(ray_tracing_pipeline) => ray_tracing_pipeline.device(),
        }
    }
}

/// The properties of one of the executables that a pipeline was compiled into.
///
/// This is returned by [`Pipeline::executable_properties`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PipelineExecutableProperties {
    /// The shader stages that were principally used as inputs to compile the executable. There is
    /// no guaranteed mapping between shader stages and executables, so this is a best effort hint
    /// only; [`name`] and [`description`] describe the executable more accurately.
    ///
    /// [`name`]: Self::name
    /// [`description`]: Self::description
    pub stages: ShaderStages,

    /// The name of the executable.
    pub name: String,

    /// A description of the executable.
    pub description: String,

    /// The subgroup size with which the executable is dispatched.
    pub subgroup_size: u32,
}

impl PipelineExecutableProperties {
    pub(crate) fn from_vk(val_vk: &vk::PipelineExecutablePropertiesKHR<'_>) -> Self {
        let &vk::PipelineExecutablePropertiesKHR {
            stages,
            subgroup_size,
            ..
        } = val_vk;

        Self {
            stages: stages.into(),
            name: unsafe { crate::c_str_to_string_unchecked(val_vk.name_as_c_str().unwrap()) },
            description: unsafe {
                crate::c_str_to_string_unchecked(val_vk.description_as_c_str().unwrap())
            },
            subgroup_size,
        }
    }
}

/// A statistic that was generated by the compilation process of a pipeline executable.
///
/// This is returned by [`Pipeline::executable_statistics`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PipelineExecutableStatistic {
    /// The name of the statistic.
    pub name: String,

    /// A description of the statistic.
    pub description: String,

    /// The value of the statistic.
    pub value: PipelineExecutableStatisticValue,
}

impl PipelineExecutableStatistic {
    pub(crate) fn from_vk(val_vk: &vk::PipelineExecutableStatisticKHR<'_>) -> Option<Self> {
        let &vk::PipelineExecutableStatisticKHR { format, value, .. } = val_vk;

        Some(Self {
            name: unsafe { crate::c_str_to_string_unchecked(val_vk.name_as_c_str().unwrap()) },
            description: unsafe {
                crate::c_str_to_string_unchecked(val_vk.description_as_c_str().unwrap())
            },
            value: PipelineExecutableStatisticValue::from_vk(format, value)?,
        })
    }
}

/// The value of a [`PipelineExecutableStatistic`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PipelineExecutableStatisticValue {
    /// The statistic is a boolean value.
    Bool32(bool),

    /// The statistic is a signed 64-bit integer.
    Int64(i64),

    /// The statistic is an unsigned 64-bit integer.
    Uint64(u64),

    /// The statistic is a 64-bit floating-point value.
    Float64(f64),
}

impl PipelineExecutableStatisticValue {
    pub(crate) fn from_vk(
        format_vk: vk::PipelineExecutableStatisticFormatKHR,
        val_vk: vk::PipelineExecutableStatisticValueKHR,
    ) -> Option<Self> {
        // SAFETY: In each of the arms below, `format_vk` specifies that the field of the union
        // being read is the active one.
        match format_vk {
            vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                Some(Self::Bool32(unsafe { val_vk.b32 } != vk::FALSE))
            }
            vk::PipelineExecutableStatisticFormatKHR::INT64 => {
                Some(Self::Int64(unsafe { val_vk.i64 }))
            }
            vk::PipelineExecutableStatisticFormatKHR::UINT64 => {
                Some(Self::Uint64(unsafe { val_vk.u64 }))
            }
            vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => {
                Some(Self::Float64(unsafe { val_vk.f64 }))
            }
            _ => None,
        }
    }
}

/// An internal representation that was generated by the compilation process of a pipeline
/// executable.
///
/// This may be the final shader assembly, a binary form of the compiled shader, or the shader
/// compiler's internal representation at an intermediate compile step.
///
/// This is returned by [`Pipeline::executable_internal_representations`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PipelineExecutableInternalRepresentation {
    /// The name of the internal representation.
    pub name: String,

    /// A description of the internal representation.
    pub description: String,

    /// The internal representation itself.
    pub data: PipelineExecutableInternalRepresentationData,
}

impl PipelineExecutableInternalRepresentation {
    pub(crate) fn from_vk(
        val_vk: &vk::PipelineExecutableInternalRepresentationKHR<'_>,
        data: Vec<u8>,
    ) -> Self {
        let &vk::PipelineExecutableInternalRepresentationKHR { is_text, .. } = val_vk;

        Self {
            name: unsafe { crate::c_str_to_string_unchecked(val_vk.name_as_c_str().unwrap()) },
            description: unsafe {
                crate::c_str_to_string_unchecked(val_vk.description_as_c_str().unwrap())
            },
            data: if is_text != vk::FALSE {
                PipelineExecutableInternalRepresentationData::Text(unsafe {
                    crate::c_str_to_string_unchecked(CStr::from_bytes_until_nul(&data).unwrap())
                })
            } else {
                PipelineExecutableInternalRepresentationData::Binary(data)
            },
        }
    }
}

/// The data of a [`PipelineExecutableInternalRepresentation`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineExecutableInternalRepresentationData {
    /// The internal representation is text.
    Text(String),

    /// The internal representation is opaque data.
    Binary(Vec<u8>),
}

vulkan_enum! {
    #[non_exhaustive]

    /// The type of a pipeline.
    ///
    /// When binding a pipeline or descriptor sets in a command buffer, the state for each bind point
    /// is independent from the others. This means that it is possible, for example, to bind a graphics
    /// pipeline without disturbing any bound compute pipeline. Likewise, binding descriptor sets for
    /// the `Compute` bind point does not affect sets that were bound to the `Graphics` bind point.
    PipelineBindPoint = PipelineBindPoint(i32);

    // TODO: document
    Compute = COMPUTE,

    // TODO: document
    Graphics = GRAPHICS,


    // TODO: document
    RayTracing = RAY_TRACING_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /* TODO: enable
    // TODO: document
    SubpassShading = SUBPASS_SHADING_HUAWEI
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(huawei_subpass_shading)]),
    ]),*/
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a pipeline.
    PipelineCreateFlags = PipelineCreateFlags(u32);

    /// The pipeline will not be optimized.
    DISABLE_OPTIMIZATION = DISABLE_OPTIMIZATION,

    /// Derivative pipelines can be created using this pipeline as a base.
    ALLOW_DERIVATIVES = ALLOW_DERIVATIVES,

    /// Create the pipeline by deriving from a base pipeline.
    DERIVATIVE = DERIVATIVE,

    /* TODO: enable
    // TODO: document
    VIEW_INDEX_FROM_DEVICE_INDEX = VIEW_INDEX_FROM_DEVICE_INDEX
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DISPATCH_BASE = DISPATCH_BASE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    FAIL_ON_PIPELINE_COMPILE_REQUIRED = FAIL_ON_PIPELINE_COMPILE_REQUIRED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_pipeline_creation_cache_control)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    EARLY_RETURN_ON_FAILURE = EARLY_RETURN_ON_FAILURE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_pipeline_creation_cache_control)]),
    ]),
    */

    /* TODO: enable
    // TODO: document
    RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT = RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_KHR {
        // Provided by VK_KHR_dynamic_rendering with VK_KHR_fragment_shading_rate
    },*/

    /* TODO: enable
    // TODO: document
    RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT = RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_EXT {
        // Provided by VK_KHR_dynamic_rendering with VK_EXT_fragment_density_map
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_ANY_HIT_SHADERS = RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS = RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_MISS_SHADERS = RAY_TRACING_NO_NULL_MISS_SHADERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_INTERSECTION_SHADERS = RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SKIP_TRIANGLES = RAY_TRACING_SKIP_TRIANGLES_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SKIP_AABBS = RAY_TRACING_SKIP_AABBS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY = RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DEFER_COMPILE = DEFER_COMPILE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),*/

    /// The shader compiler should capture statistics for the pipeline executables that the compile
    /// process produces, which can later be retrieved by calling
    /// [`Pipeline::executable_statistics`].
    ///
    /// Enabling this flag must not affect the final compiled pipeline, but may disable pipeline
    /// caching or otherwise affect pipeline creation time.
    CAPTURE_STATISTICS = CAPTURE_STATISTICS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_pipeline_executable_properties)]),
    ]),

    /// The shader compiler should capture the internal representations of the pipeline executables
    /// that the compile process produces, which can later be retrieved by calling
    /// [`Pipeline::executable_internal_representations`].
    ///
    /// Enabling this flag must not affect the final compiled pipeline, but may disable pipeline
    /// caching or otherwise affect pipeline creation time.
    CAPTURE_INTERNAL_REPRESENTATIONS = CAPTURE_INTERNAL_REPRESENTATIONS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_pipeline_executable_properties)]),
    ]),

    /* TODO: enable
    // TODO: document
    INDIRECT_BINDABLE = INDIRECT_BINDABLE_NV{
        device_extensions: [nv_device_generated_commands],
    },*/

    /* TODO: enable
    // TODO: document
    LIBRARY = LIBRARY_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_pipeline_library)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DESCRIPTOR_BUFFER = DESCRIPTOR_BUFFER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RETAIN_LINK_TIME_OPTIMIZATION_INFO = RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_graphics_pipeline_library)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    LINK_TIME_OPTIMIZATION = LINK_TIME_OPTIMIZATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_graphics_pipeline_library)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_ALLOW_MOTION = RAY_TRACING_ALLOW_MOTION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing_motion_blur)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    COLOR_ATTACHMENT_FEEDBACK_LOOP = COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_attachment_feedback_loop_layout)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP = DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_attachment_feedback_loop_layout)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_OPACITY_MICROMAP = RAY_TRACING_OPACITY_MICROMAP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_DISPLACEMENT_MICROMAP = RAY_TRACING_DISPLACEMENT_MICROMAP_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_displacement_micromap)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    NO_PROTECTED_ACCESS = NO_PROTECTED_ACCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_pipeline_protected_access)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PROTECTED_ACCESS_ONLY = PROTECTED_ACCESS_ONLY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_pipeline_protected_access)]),
    ]),*/
}

vulkan_enum! {
    #[non_exhaustive]

    /// A particular state value within a pipeline that can be dynamically set by a command buffer.
    ///
    /// Whenever a particular state is set to be dynamic while creating the pipeline,
    /// the corresponding predefined value in the pipeline's create info is ignored, unless
    /// specified otherwise here.
    ///
    /// If the dynamic state is used to enable/disable a certain functionality,
    /// and the value in the create info is an `Option`
    /// (for example, [`DynamicState::DepthTestEnable`] and [`DepthStencilState::depth`]),
    /// then that `Option` must be `Some` when creating the pipeline,
    /// in order to provide settings to use when the functionality is enabled.
    ///
    /// [`DepthStencilState::depth`]: (crate::pipeline::graphics::depth_stencil::DepthStencilState::depth)
    DynamicState = DynamicState(i32);

    /// The elements, but not the count, of
    /// [`ViewportState::viewports`](crate::pipeline::graphics::viewport::ViewportState::viewports).
    ///
    /// Set with
    /// [`set_viewport`](crate::command_buffer::AutoCommandBufferBuilder::set_viewport).
    Viewport = VIEWPORT,

    /// The elements, but not the count, of
    /// [`ViewportState::scissors`](crate::pipeline::graphics::viewport::ViewportState::scissors).
    ///
    /// Set with
    /// [`set_scissor`](crate::command_buffer::AutoCommandBufferBuilder::set_scissor).
    Scissor = SCISSOR,

    /// The value of
    /// [`RasterizationState::line_width`](crate::pipeline::graphics::rasterization::RasterizationState::line_width).
    ///
    /// Set with
    /// [`set_line_width`](crate::command_buffer::AutoCommandBufferBuilder::set_line_width).
    LineWidth = LINE_WIDTH,

    /// The value of
    /// [`RasterizationState::depth_bias`](crate::pipeline::graphics::rasterization::RasterizationState::depth_bias).
    ///
    /// Set with
    /// [`set_depth_bias`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_bias).
    DepthBias = DEPTH_BIAS,

    /// The value of
    /// [`ColorBlendState::blend_constants`](graphics::color_blend::ColorBlendState::blend_constants).
    ///
    /// Set with
    /// [`set_blend_constants`](crate::command_buffer::AutoCommandBufferBuilder::set_blend_constants).
    BlendConstants = BLEND_CONSTANTS,

    /// The value, but not the `Option` variant, of
    /// [`DepthStencilState::depth_bounds`](crate::pipeline::graphics::depth_stencil::DepthStencilState::depth_bounds).
    ///
    /// Set with
    /// [`set_depth_bounds`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_bounds).
    DepthBounds = DEPTH_BOUNDS,

    /// The value of
    /// [`StencilOpState::compare_mask`](crate::pipeline::graphics::depth_stencil::StencilOpState::compare_mask)
    /// for both the front and back face.
    ///
    /// Set with
    /// [`set_stencil_compare_mask`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_compare_mask).
    StencilCompareMask = STENCIL_COMPARE_MASK,

    /// The value of
    /// [`StencilOpState::write_mask`](crate::pipeline::graphics::depth_stencil::StencilOpState::write_mask)
    /// for both the front and back face.
    ///
    /// Set with
    /// [`set_stencil_write_mask`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_write_mask).
    StencilWriteMask = STENCIL_WRITE_MASK,

    /// The value of
    /// [`StencilOpState::reference`](crate::pipeline::graphics::depth_stencil::StencilOpState::reference)
    /// for both the front and back face.
    ///
    /// Set with
    /// [`set_stencil_reference`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_reference).
    StencilReference = STENCIL_REFERENCE,

    /// The value of
    /// [`RasterizationState::cull_mode`](graphics::rasterization::RasterizationState::cull_mode).
    ///
    /// Set with
    /// [`set_cull_mode`](crate::command_buffer::AutoCommandBufferBuilder::set_cull_mode).
    CullMode = CULL_MODE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`RasterizationState::front_face`](graphics::rasterization::RasterizationState::front_face).
    ///
    /// Set with
    /// [`set_front_face`](crate::command_buffer::AutoCommandBufferBuilder::set_front_face).
    FrontFace = FRONT_FACE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`InputAssemblyState::topology`](graphics::input_assembly::InputAssemblyState::topology).
    ///
    /// Set with
    /// [`set_primitive_topology`](crate::command_buffer::AutoCommandBufferBuilder::set_primitive_topology).
    PrimitiveTopology = PRIMITIVE_TOPOLOGY
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// Both the elements and the count of
    /// [`ViewportState::viewports`](crate::pipeline::graphics::viewport::ViewportState::viewports).
    ///
    /// Set with
    /// [`set_viewport_with_count`](crate::command_buffer::AutoCommandBufferBuilder::set_viewport_with_count).
    ViewportWithCount = VIEWPORT_WITH_COUNT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// Both the elements and the count of
    /// [`ViewportState::scissors`](crate::pipeline::graphics::viewport::ViewportState::scissors).
    ///
    /// Set with
    /// [`set_scissor_with_count`](crate::command_buffer::AutoCommandBufferBuilder::set_scissor_with_count).
    ScissorWithCount = SCISSOR_WITH_COUNT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /* TODO: enable
    // TODO: document
    VertexInputBindingStride = VERTEX_INPUT_BINDING_STRIDE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),*/

    /// The `Option` variant of
    /// [`DepthStencilState::depth`](crate::pipeline::graphics::depth_stencil::DepthStencilState::depth).
    ///
    /// Set with
    /// [`set_depth_test_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_test_enable).
    DepthTestEnable = DEPTH_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`DepthState::write_enable`](crate::pipeline::graphics::depth_stencil::DepthState::write_enable).
    ///
    /// Set with
    /// [`set_depth_write_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_write_enable).
    DepthWriteEnable = DEPTH_WRITE_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`DepthState::compare_op`](crate::pipeline::graphics::depth_stencil::DepthState::compare_op).
    ///
    /// Set with
    /// [`set_depth_compare_op`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_compare_op).
    DepthCompareOp = DEPTH_COMPARE_OP
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The `Option` variant of
    /// [`DepthStencilState::depth_bounds`](crate::pipeline::graphics::depth_stencil::DepthStencilState::depth_bounds).
    ///
    /// Set with
    /// [`set_depth_bounds_test_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_bounds_test_enable).
    DepthBoundsTestEnable = DEPTH_BOUNDS_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The `Option` variant of
    /// [`DepthStencilState::stencil`](crate::pipeline::graphics::depth_stencil::DepthStencilState::stencil).
    ///
    /// Set with
    /// [`set_stencil_test_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_test_enable).
    StencilTestEnable = STENCIL_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`StencilOpState::ops`](crate::pipeline::graphics::depth_stencil::StencilOpState::ops)
    /// for both the front and back face.
    ///
    /// Set with
    /// [`set_stencil_op`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_op).
    StencilOp = STENCIL_OP
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`RasterizationState::rasterizer_discard_enable`](crate::pipeline::graphics::rasterization::RasterizationState::rasterizer_discard_enable).
    ///
    /// Set with
    /// [`set_rasterizer_discard_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_rasterizer_discard_enable).
    RasterizerDiscardEnable = RASTERIZER_DISCARD_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The `Option` variant of
    /// [`RasterizationState::depth_bias`](crate::pipeline::graphics::rasterization::RasterizationState::depth_bias).
    ///
    /// Set with
    /// [`set_depth_bias_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_bias_enable).
    DepthBiasEnable = DEPTH_BIAS_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The value of
    /// [`InputAssemblyState::primitive_restart_enable`](graphics::input_assembly::InputAssemblyState::primitive_restart_enable).
    ///
    /// Set with
    /// [`set_primitive_restart_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_primitive_restart_enable).
    PrimitiveRestartEnable = PRIMITIVE_RESTART_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /* TODO: enable
    // TODO: document
    ViewportWScaling = VIEWPORT_W_SCALING_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_clip_space_w_scaling)]),
    ]), */

    /// The elements, but not count, of
    /// [`DiscardRectangleState::rectangles`](crate::pipeline::graphics::discard_rectangle::DiscardRectangleState::rectangles).
    ///
    /// Set with
    /// [`set_discard_rectangle`](crate::command_buffer::AutoCommandBufferBuilder::set_discard_rectangle).
    DiscardRectangle = DISCARD_RECTANGLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_discard_rectangles)]),
    ]),

    /* TODO: enable
    // TODO: document
    SampleLocations = SAMPLE_LOCATIONS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_sample_locations)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RayTracingPipelineStackSize = RAY_TRACING_PIPELINE_STACK_SIZE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ViewportShadingRatePalette = VIEWPORT_SHADING_RATE_PALETTE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_shading_rate_image)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ViewportCoarseSampleOrder = VIEWPORT_COARSE_SAMPLE_ORDER_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_shading_rate_image)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ExclusiveScissor = EXCLUSIVE_SCISSOR_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_scissor_exclusive)]),
    ]), */

    /// The value of
    /// [`FragmentShadingRateState`](crate::pipeline::graphics::fragment_shading_rate::FragmentShadingRateState).
    ///
    /// Set with
    /// [`set_fragment_shading_rate`](crate::command_buffer::AutoCommandBufferBuilder::set_fragment_shading_rate).
    FragmentShadingRate = FRAGMENT_SHADING_RATE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),

    /// The value of
    /// [`RasterizationState::line_stipple`](crate::pipeline::graphics::rasterization::RasterizationState::line_stipple).
    ///
    /// Set with
    /// [`set_line_stipple`](crate::command_buffer::AutoCommandBufferBuilder::set_line_stipple).
    LineStipple = LINE_STIPPLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_line_rasterization)]),
    ]),

    /// The `Option` variant of
    /// [`GraphicsPipelineCreateInfo::vertex_input_state`](crate::pipeline::graphics::GraphicsPipelineCreateInfo::vertex_input_state).
    ///
    /// Set with
    /// [`set_vertex_input`](crate::command_buffer::AutoCommandBufferBuilder::set_vertex_input).
    VertexInput = VERTEX_INPUT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_vertex_input_dynamic_state)]),
    ]),

    /// The value of
    /// [`TessellationState::patch_control_points`](graphics::tessellation::TessellationState::patch_control_points).
    ///
    /// Set with
    /// [`set_patch_control_points`](crate::command_buffer::AutoCommandBufferBuilder::set_patch_control_points).
    PatchControlPoints = PATCH_CONTROL_POINTS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The value of
    /// [`ColorBlendState::logic_op`](graphics::color_blend::ColorBlendState::logic_op).
    ///
    /// Set with
    /// [`set_logic_op`](crate::command_buffer::AutoCommandBufferBuilder::set_logic_op).
    LogicOp = LOGIC_OP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The value of
    /// [`ColorBlendAttachmentState::color_write_enable`](crate::pipeline::graphics::color_blend::ColorBlendAttachmentState::color_write_enable)
    /// for every attachment.
    ///
    /// Set with
    /// [`set_color_write_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_color_write_enable).
    ColorWriteEnable = COLOR_WRITE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_color_write_enable)]),
    ]),

    /* TODO: enable
    // TODO: document
    TessellationDomainOrigin = TESSELLATION_DOMAIN_ORIGIN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    DepthClampEnable = DEPTH_CLAMP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    PolygonMode = POLYGON_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RasterizationSamples = RASTERIZATION_SAMPLES_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    SampleMask = SAMPLE_MASK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    AlphaToCoverageEnable = ALPHA_TO_COVERAGE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    AlphaToOneEnable = ALPHA_TO_ONE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    LogicOpEnable = LOGIC_OP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ColorBlendEnable = COLOR_BLEND_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ColorBlendEquation = COLOR_BLEND_EQUATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ColorWriteMask = COLOR_WRITE_MASK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RasterizationStream = RASTERIZATION_STREAM_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /// The value of
    /// [`ConservativeRasterizationState::mode`](crate::pipeline::graphics::rasterization::RasterizationConservativeState::mode)
    ///
    /// Set with
    /// [`set_conservative_rasterization_mode`](crate::command_buffer::AutoCommandBufferBuilder::set_conservative_rasterization_mode).
    ConservativeRasterizationMode = CONSERVATIVE_RASTERIZATION_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    /// The value of
    /// [`ConservativeRasterizationState::overestimation_size`](crate::pipeline::graphics::rasterization::RasterizationConservativeState::overestimation_size)
    ///
    /// Set with
    /// [`set_extra_primitive_overestimation_size`](crate::command_buffer::AutoCommandBufferBuilder::set_extra_primitive_overestimation_size).
    ExtraPrimitiveOverestimationSize = EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    /* TODO: enable
    // TODO: document
    DepthClipEnable = DEPTH_CLIP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    SampleLocationsEnable = SAMPLE_LOCATIONS_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ColorBlendAdvanced = COLOR_BLEND_ADVANCED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ProvokingVertexMode = PROVOKING_VERTEX_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    LineRasterizationMode = LINE_RASTERIZATION_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    LineStippleEnable = LINE_STIPPLE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    DepthClipNegativeOneToOne = DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ViewportWScalingEnable = VIEWPORT_W_SCALING_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ViewportSwizzle = VIEWPORT_SWIZZLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageToColorEnable = COVERAGE_TO_COLOR_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageToColorLocation = COVERAGE_TO_COLOR_LOCATION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageModulationMode = COVERAGE_MODULATION_MODE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageModulationTableEnable = COVERAGE_MODULATION_TABLE_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageModulationTable = COVERAGE_MODULATION_TABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ShadingRateImageEnable = SHADING_RATE_IMAGE_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RepresentativeFragmentTestEnable = REPRESENTATIVE_FRAGMENT_TEST_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageReductionMode = COVERAGE_REDUCTION_MODE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */
}

#[cfg(test)]
mod tests {
    use crate::{
        device::Device,
        pipeline::{
            compute::ComputePipelineCreateInfo, ComputePipeline, Pipeline, PipelineCreateFlags,
            PipelineLayout, PipelineShaderStageCreateInfo,
        },
        shader::{ShaderModule, ShaderModuleCreateInfo},
    };
    use std::{slice, sync::Arc};

    #[test]
    fn executable_queries_require_capture_flags() {
        let (device, _queue) =
            gfx_dev_and_queue!(pipeline_executable_info; khr_pipeline_executable_properties);

        let compute_pipeline = new_compute_pipeline(&device, PipelineCreateFlags::empty());
        let pipeline = Pipeline::from(&compute_pipeline);

        assert!(pipeline.try_executable_statistics(0).is_err());
        assert!(pipeline.try_executable_internal_representations(0).is_err());
    }

    #[test]
    fn executable_queries_succeed() {
        let (device, _queue) =
            gfx_dev_and_queue!(pipeline_executable_info; khr_pipeline_executable_properties);

        let compute_pipeline = new_compute_pipeline(
            &device,
            PipelineCreateFlags::CAPTURE_STATISTICS
                | PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS,
        );
        let pipeline = Pipeline::from(&compute_pipeline);

        // A pipeline is compiled into zero or more executables, and an implementation is allowed
        // to provide no statistics or internal representations for any of them. The only thing
        // that can be asserted is that each query succeeds for every executable that is reported.
        let properties = pipeline.executable_properties().unwrap();

        for index in 0..properties.len() as u32 {
            pipeline.executable_statistics(index).unwrap();
            pipeline.executable_internal_representations(index).unwrap();
        }
    }

    fn new_compute_pipeline(
        device: &Arc<Device>,
        flags: PipelineCreateFlags,
    ) -> Arc<ComputePipeline> {
        let cs = {
            /*
             * #version 450
             * void main() {
             * }
             */
            const MODULE: [u32; 48] = [
                119734787, 65536, 524298, 6, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1, 1,
                196611, 2, 450, 262149, 4, 1852399981, 0, 131091, 2, 196641, 3, 2, 327734, 2, 4, 0,
                3, 131320, 5, 65789, 65592,
            ];
            let module =
                unsafe { ShaderModule::new(device, &ShaderModuleCreateInfo::new(&MODULE)) }
                    .unwrap();
            module.entry_point("main").unwrap()
        };

        let stage = PipelineShaderStageCreateInfo::new(&cs);
        let layout = PipelineLayout::from_stages(device, slice::from_ref(&stage)).unwrap();

        ComputePipeline::new(
            device,
            None,
            &ComputePipelineCreateInfo {
                flags,
                ..ComputePipelineCreateInfo::new(stage, &layout)
            },
        )
        .unwrap()
    }
}
