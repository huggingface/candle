//! The layout of descriptor sets and push constants used by a pipeline.
//!
//! # Overview
//!
//! The layout itself only *describes* the descriptors and push constants, and does not contain
//! their content itself. Instead, you can think of it as a `struct` definition that states which
//! members there are, what types they have, and in what order.
//! One could imagine a Rust definition somewhat like this:
//!
//! ```text
//! #[repr(C)]
//! struct MyPipelineLayout {
//!     push_constants: Pc,
//!     descriptor_set0: Ds0,
//!     descriptor_set1: Ds1,
//!     descriptor_set2: Ds2,
//!     descriptor_set3: Ds3,
//! }
//! ```
//!
//! Of course, a pipeline layout is created at runtime, unlike a Rust type.
//!
//! # Layout compatibility
//!
//! When binding descriptor sets or setting push constants, you must provide a pipeline layout.
//! This layout is used to decide where in memory Vulkan should write the new data. The
//! descriptor sets and push constants can later be read by dispatch or draw calls, but only if
//! the bound pipeline being used for the command has a layout that is *compatible* with the layout
//! that was used to bind the resources.
//!
//! *Compatible* means that the pipeline layout must be the same object, or a different layout in
//! which the push constant ranges and descriptor set layouts were be identically defined.
//! However, Vulkan allows for partial compatibility as well. In the `struct` analogy used above,
//! one could imagine that using a different definition would leave some members with the same
//! offset and size within the struct as in the old definition, while others are no longer
//! positioned correctly. For example, if a new, incompatible type were used for `Ds1`, then the
//! `descriptor_set1`, `descriptor_set2` and `descriptor_set3` members would no longer be correct,
//! but `descriptor_set0` and `push_constants` would remain accessible in the new layout.
//! Because of this behavior, the following rules apply to compatibility between the layouts used
//! in subsequent descriptor set binding calls:
//!
//! - An incompatible definition of `Pc` invalidates all bound descriptor sets.
//! - An incompatible definition of `DsN` invalidates all bound descriptor sets *N* and higher.
//! - If *N* is the highest set being assigned in a bind command, and it and all lower sets have
//!   compatible definitions, including the push constants, then descriptor sets above *N* remain
//!   valid.
//!
//! [`AutoCommandBufferBuilder`] keeps track of this state and will automatically remove descriptor
//! sets that have been invalidated by incompatible layouts in subsequent binding commands.
//!
//! # Creating pipeline layouts
//!
//! A pipeline layout is a Vulkan object type, represented in Vulkano with the `PipelineLayout`
//! type. Each pipeline that you create holds a pipeline layout object.
//!
//! [`AutoCommandBufferBuilder`]: crate::command_buffer::AutoCommandBufferBuilder

use super::PipelineShaderStageCreateInfo;
use crate::{
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags,
        DescriptorSetLayoutCreateInfo, DescriptorType,
    },
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper, DeviceProperties},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags},
    shader::{DescriptorBindingRequirements, ShaderStage, ShaderStages},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use foldhash::HashMap;
use smallvec::SmallVec;
use std::{
    array, cmp::max, collections::hash_map::Entry, fmt::Write, mem::MaybeUninit, num::NonZero, ptr,
    sync::Arc,
};

/// Describes the layout of descriptor sets and push constants that are made available to shaders.
#[derive(Debug)]
pub struct PipelineLayout {
    handle: vk::PipelineLayout,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: PipelineLayoutCreateFlags,
    set_layouts: Vec<DeviceOwnedDebugWrapper<Arc<DescriptorSetLayout>>>,
    push_constant_ranges: Vec<PushConstantRange>,

    push_constant_ranges_disjoint: Vec<PushConstantRange>,
}

impl PipelineLayout {
    /// Creates a new `PipelineLayout`, panicking on a validation error.
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
        device: &Arc<Device>,
        create_info: &PipelineLayoutCreateInfo<'_>,
    ) -> Result<Arc<PipelineLayout>, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `PipelineLayout`.
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &PipelineLayoutCreateInfo<'_>,
    ) -> Result<Arc<PipelineLayout>, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &PipelineLayoutCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkCreatePipelineLayout-pCreateInfo-parameter
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &PipelineLayoutCreateInfo<'_>,
    ) -> Result<Arc<PipelineLayout>, VulkanError> {
        let create_info_fields1_vk = create_info.to_vk_fields1();
        let create_info_vk = create_info.to_vk(&create_info_fields1_vk);

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_pipeline_layout)(
                    device.handle(),
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

    /// Creates a new `PipelineLayout` from the union of the requirements of each shader stage in
    /// `stages`, panicking on a validation error.
    ///
    /// This is intended for quick prototyping or for single use layouts that do not have any
    /// bindings in common with other shaders. For the general case, it is strongly recommended to
    /// create pipeline layouts manually:
    /// - When multiple pipelines share the same layout object, it is faster than if they have
    ///   different objects, even if the objects both contain identical bindings. It is also faster
    ///   (though a little bit less), if multiple pipeline layout objects share common descriptor
    ///   set objects.
    /// - Pipeline layouts only need to be a superset of what the shaders use; they don't have to
    ///   match exactly. Creating a manual pipeline layout therefore allows you to specify layouts
    ///   that are applicable for many shaders, as long as each one uses a subset. This allows
    ///   further sharing.
    /// - Creating a manual pipeline layout makes your code more robust against changes in the
    ///   shader, in particular regarding whether a particular binding in the shader is used or not
    ///   (see also the limitations below).
    ///
    /// This is a shortcut for `try_from_stages().map_err(Validated::unwrap)`.
    ///
    /// # Limitations
    ///
    /// Only bindings that are [statically used] are included in the descriptor binding
    /// requirements, and therefore are included in the descriptor set layout. If the use of a
    /// binding depends on input variables to the shader (buffers, images, push constants etc.)
    /// then the shader reflection is unable to know that the binding is in use, and it will not be
    /// included in the pipeline layout.
    ///
    /// Note that this corresponds to the `shader_*_array_dynamic_indexing` device features.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_from_stages`] returns a [`ValidationError`].
    ///
    /// [statically used]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#shaders-staticuse
    /// [`try_from_stages`]: Self::try_from_stages
    #[track_caller]
    pub fn from_stages(
        device: &Arc<Device>,
        stages: &[PipelineShaderStageCreateInfo<'_>],
    ) -> Result<Arc<PipelineLayout>, VulkanError> {
        match Self::try_from_stages(device, stages) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `PipelineLayout` from the union of the requirements of each shader stage in
    /// `stages`.
    ///
    /// This is intended for quick prototyping or for single use layouts that do not have any
    /// bindings in common with other shaders. For the general case, it is strongly recommended to
    /// create pipeline layouts manually:
    /// - When multiple pipelines share the same layout object, it is faster than if they have
    ///   different objects, even if the objects both contain identical bindings. It is also faster
    ///   (though a little bit less), if multiple pipeline layout objects share common descriptor
    ///   set objects.
    /// - Pipeline layouts only need to be a superset of what the shaders use; they don't have to
    ///   match exactly. Creating a manual pipeline layout therefore allows you to specify layouts
    ///   that are applicable for many shaders, as long as each one uses a subset. This allows
    ///   further sharing.
    /// - Creating a manual pipeline layout makes your code more robust against changes in the
    ///   shader, in particular regarding whether a particular binding in the shader is used or not
    ///   (see also the limitations below).
    ///
    /// # Limitations
    ///
    /// Only bindings that are [statically used] are included in the descriptor binding
    /// requirements, and therefore are included in the descriptor set layout. If the use of a
    /// binding depends on input variables to the shader (buffers, images, push constants etc.)
    /// then the shader reflection is unable to know that the binding is in use, and it will not be
    /// included in the pipeline layout.
    ///
    /// Note that this corresponds to the `shader_*_array_dynamic_indexing` device features.
    ///
    /// [statically used]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#shaders-staticuse
    pub fn try_from_stages(
        device: &Arc<Device>,
        stages: &[PipelineShaderStageCreateInfo<'_>],
    ) -> Result<Arc<PipelineLayout>, Validated<VulkanError>> {
        // Produce `DescriptorBindingRequirements` for each binding by iterating over all shaders
        // and adding the requirements of each.
        let mut descriptor_binding_requirements =
            HashMap::<_, DescriptorBindingRequirements>::default();
        let mut max_set_num = 0;

        for stage in stages {
            let entry_point_info = stage.entry_point.info();

            for (&(set_num, binding_num), reqs) in &entry_point_info.descriptor_binding_requirements
            {
                max_set_num = max(max_set_num, set_num);

                match descriptor_binding_requirements.entry((set_num, binding_num)) {
                    Entry::Occupied(entry) => {
                        // Previous shaders already added requirements, so we merge requirements of
                        // the current shader into the requirements of the previous ones.
                        // TODO: return an error here instead of panicking?
                        entry.into_mut().merge(reqs).expect(
                            "could not produce an intersection of the shader descriptor \
                            requirements",
                        );
                    }
                    Entry::Vacant(entry) => {
                        // No previous shader had this descriptor yet, so we just insert the
                        // requirements.
                        entry.insert(reqs.clone());
                    }
                }
            }
        }

        // Convert the descriptor binding requirements.
        let mut set_layouts_bindings =
            vec![Vec::<DescriptorSetLayoutBinding<'static>>::new(); max_set_num as usize + 1];

        for ((set_num, binding_num), reqs) in descriptor_binding_requirements {
            let bindings = &mut set_layouts_bindings[set_num as usize];
            let (Ok(index) | Err(index)) =
                bindings.binary_search_by_key(&binding_num, |binding| binding.binding);
            bindings.insert(
                index,
                DescriptorSetLayoutBinding {
                    binding: binding_num,
                    descriptor_count: reqs.descriptor_count.unwrap_or(0),
                    stages: reqs.stages,
                    ..DescriptorSetLayoutBinding::new(reqs.descriptor_types[0])
                },
            );
        }

        let set_layouts = set_layouts_bindings
            .iter()
            .map(|bindings| {
                DescriptorSetLayout::new(
                    device,
                    &DescriptorSetLayoutCreateInfo {
                        bindings,
                        ..Default::default()
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::try_new(
            device,
            &PipelineLayoutCreateInfo {
                set_layouts: &set_layouts.iter().collect::<Vec<_>>(),
                push_constant_ranges: &push_constant_ranges_from_stages(stages),
                ..Default::default()
            },
        )
    }

    /// Creates a new `PipelineLayout` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::PipelineLayout,
        create_info: &PipelineLayoutCreateInfo<'_>,
    ) -> Arc<PipelineLayout> {
        let &PipelineLayoutCreateInfo {
            flags,
            set_layouts,
            push_constant_ranges,
            _ne: _,
        } = create_info;

        let mut push_constant_ranges = push_constant_ranges.to_owned();

        // Sort the ranges for the purpose of comparing for equality.
        // The stage mask is guaranteed to be unique, so it's a suitable sorting key.
        push_constant_ranges.sort_unstable_by_key(|range| {
            (
                range.offset,
                range.size,
                vk::ShaderStageFlags::from(range.stages),
            )
        });

        let mut push_constant_ranges_disjoint: Vec<PushConstantRange> =
            Vec::with_capacity(push_constant_ranges.len());

        if !push_constant_ranges.is_empty() {
            let mut min_offset = push_constant_ranges[0].offset;
            loop {
                let mut max_offset = u32::MAX;
                let mut stages = ShaderStages::empty();

                for range in push_constant_ranges.iter() {
                    // new start (begin next time from it)
                    if range.offset > min_offset {
                        max_offset = max_offset.min(range.offset);
                        break;
                    } else if range.offset + range.size > min_offset {
                        // inside the range, include the stage
                        // use the minimum of the end of all ranges that are overlapping
                        max_offset = max_offset.min(range.offset + range.size);
                        stages |= range.stages;
                    }
                }

                if !stages.is_empty() {
                    push_constant_ranges_disjoint.push(PushConstantRange {
                        stages,
                        offset: min_offset,
                        size: max_offset - min_offset,
                    });
                }

                if max_offset == u32::MAX {
                    break;
                }

                // prepare for next range
                min_offset = max_offset;
            }
        }

        Arc::new(PipelineLayout {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),
            flags,
            set_layouts: set_layouts
                .iter()
                .copied()
                .cloned()
                .map(DeviceOwnedDebugWrapper)
                .collect(),
            push_constant_ranges,
            push_constant_ranges_disjoint,
        })
    }

    /// Returns the flags that the pipeline layout was created with.
    #[inline]
    pub fn flags(&self) -> PipelineLayoutCreateFlags {
        self.flags
    }

    /// Returns the descriptor set layouts this pipeline layout was created from.
    #[inline]
    pub fn set_layouts(&self) -> &[Arc<DescriptorSetLayout>] {
        DeviceOwnedDebugWrapper::cast_slice_inner(&self.set_layouts)
    }

    /// Returns a slice containing the push constant ranges this pipeline layout was created from.
    ///
    /// The ranges are guaranteed to be sorted deterministically by offset, size, then stages.
    /// This means that two slices containing the same elements will always have the same order.
    #[inline]
    pub fn push_constant_ranges(&self) -> &[PushConstantRange] {
        &self.push_constant_ranges
    }

    /// Returns a slice containing the push constant ranges in with all disjoint stages.
    ///
    /// For example, if we have these `push_constant_ranges`:
    /// - `offset=0, size=4, stages=vertex`
    /// - `offset=0, size=12, stages=fragment`
    ///
    /// The returned value will be:
    /// - `offset=0, size=4, stages=vertex|fragment`
    /// - `offset=4, size=8, stages=fragment`
    ///
    /// The ranges are guaranteed to be sorted deterministically by offset, and
    /// guaranteed to be disjoint, meaning that there is no overlap between the ranges.
    #[doc(hidden)]
    #[inline]
    pub fn push_constant_ranges_disjoint(&self) -> &[PushConstantRange] {
        &self.push_constant_ranges_disjoint
    }

    /// Returns whether `self` is compatible with `other` for the given number of sets.
    #[inline]
    pub fn is_compatible_with(&self, other: &PipelineLayout, num_sets: u32) -> bool {
        let num_sets = num_sets as usize;
        assert!(num_sets <= self.set_layouts.len());

        if self == other {
            return true;
        }

        if self.push_constant_ranges != other.push_constant_ranges {
            return false;
        }

        let other_sets = match other.set_layouts.get(0..num_sets) {
            Some(x) => x,
            None => return false,
        };

        self.set_layouts
            .iter()
            .zip(other_sets)
            .all(|(self_set_layout, other_set_layout)| {
                self_set_layout.is_compatible_with(other_set_layout)
            })
    }

    /// Makes sure that `self` is a superset of the provided descriptor set layouts and push
    /// constant ranges. Returns an `Err` if this is not the case.
    pub(crate) fn ensure_compatible_with_shader<'a>(
        &self,
        descriptor_requirements: impl IntoIterator<
            Item = ((u32, u32), &'a DescriptorBindingRequirements),
        >,
        push_constant_range: Option<&PushConstantRange>,
    ) -> Result<(), Box<ValidationError>> {
        for ((set_num, binding_num), reqs) in descriptor_requirements.into_iter() {
            let layout_binding = self
                .set_layouts
                .get(set_num as usize)
                .and_then(|set_layout| set_layout.binding(binding_num));

            let Some(layout_binding) = layout_binding else {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the requirements for descriptor set {} binding {} were not met: \
                        no such binding exists in the pipeline layout",
                        set_num, binding_num,
                    )
                    .into(),
                    ..Default::default()
                }));
            };

            if let Err(error) = layout_binding.ensure_compatible_with_shader(reqs) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the requirements for descriptor set {} binding {} were not met: {}",
                        set_num, binding_num, error,
                    )
                    .into(),
                    ..Default::default()
                }));
            }
        }

        if let Some(range) = push_constant_range {
            for own_range in self.push_constant_ranges.iter() {
                if range.stages.intersects(own_range.stages) &&       // check if it shares any stages
                    (range.offset < own_range.offset || // our range must start before and end after the given range
                        own_range.offset + own_range.size < range.offset + range.size)
                {
                    return Err(Box::new(ValidationError {
                        problem: "the required push constant range is larger than the \
                            push constant range in the pipeline layout"
                            .into(),
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }
}

impl Drop for PipelineLayout {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe {
            (fns.v1_0.destroy_pipeline_layout)(self.device.handle(), self.handle, ptr::null())
        };
    }
}

unsafe impl VulkanObject for PipelineLayout {
    type Handle = vk::PipelineLayout;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for PipelineLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(PipelineLayout);

/// Parameters to create a new `PipelineLayout`.
#[derive(Clone, Debug)]
pub struct PipelineLayoutCreateInfo<'a> {
    /// Additional properties of the pipeline layout.
    ///
    /// The default value is empty.
    pub flags: PipelineLayoutCreateFlags,

    /// The descriptor set layouts that should be part of the pipeline layout.
    ///
    /// They are provided in order of set number.
    ///
    /// The default value is empty.
    pub set_layouts: &'a [&'a Arc<DescriptorSetLayout>],

    /// The ranges of push constants that the pipeline will access.
    ///
    /// A shader stage can only appear in one element of the list, but it is possible to combine
    /// ranges for multiple shader stages if they are the same.
    ///
    /// The default value is empty.
    pub push_constant_ranges: &'a [PushConstantRange],

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for PipelineLayoutCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> PipelineLayoutCreateInfo<'a> {
    /// Returns a default `PipelineLayoutCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: PipelineLayoutCreateFlags::empty(),
            set_layouts: &[],
            push_constant_ranges: &[],
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let properties = device.physical_device().properties();

        let &Self {
            flags,
            set_layouts,
            push_constant_ranges,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkPipelineLayoutCreateInfo-flags-parameter"])
        })?;

        if set_layouts.len() > properties.max_bound_descriptor_sets as usize {
            return Err(Box::new(ValidationError {
                context: "set_layouts".into(),
                problem: "the length exceeds the `max_bound_descriptor_sets` limit".into(),
                vuids: &["VUID-VkPipelineLayoutCreateInfo-setLayoutCount-00286"],
                ..Default::default()
            }));
        }

        struct DescriptorLimit {
            descriptor_types: &'static [DescriptorType],
            get_limit_all: fn(&DeviceProperties) -> Option<u32>,
            limit_name_all: &'static str,
            vuids_all: &'static [&'static str],
            get_limit_not_uab: fn(&DeviceProperties) -> u32,
            limit_name_not_uab: &'static str,
            vuids_not_uab: &'static [&'static str],
        }

        const PER_STAGE_DESCRIPTOR_LIMITS: [DescriptorLimit; 8] = [
            DescriptorLimit {
                descriptor_types: &[
                    DescriptorType::Sampler,
                    DescriptorType::CombinedImageSampler,
                ],
                get_limit_all: |p| p.max_per_stage_descriptor_update_after_bind_samplers,
                limit_name_all: "max_per_stage_descriptor_update_after_bind_samplers",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03022"],
                get_limit_not_uab: |p| p.max_per_stage_descriptor_samplers,
                limit_name_not_uab: "max_per_stage_descriptor_samplers",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03016"],
            },
            DescriptorLimit {
                descriptor_types: &[
                    DescriptorType::UniformBuffer,
                    DescriptorType::UniformBufferDynamic,
                ],
                get_limit_all: |p| p.max_per_stage_descriptor_update_after_bind_uniform_buffers,
                limit_name_all: "max_per_stage_descriptor_update_after_bind_uniform_buffers",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03023"],
                get_limit_not_uab: |p| p.max_per_stage_descriptor_uniform_buffers,
                limit_name_not_uab: "max_per_stage_descriptor_uniform_buffers",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03017"],
            },
            DescriptorLimit {
                descriptor_types: &[
                    DescriptorType::StorageBuffer,
                    DescriptorType::StorageBufferDynamic,
                ],
                get_limit_all: |p| p.max_per_stage_descriptor_update_after_bind_storage_buffers,
                limit_name_all: "max_per_stage_descriptor_update_after_bind_storage_buffers",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03024"],
                get_limit_not_uab: |p| p.max_per_stage_descriptor_storage_buffers,
                limit_name_not_uab: "max_per_stage_descriptor_storage_buffers",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03018"],
            },
            DescriptorLimit {
                descriptor_types: &[
                    DescriptorType::CombinedImageSampler,
                    DescriptorType::SampledImage,
                    DescriptorType::UniformTexelBuffer,
                ],
                get_limit_all: |p| p.max_per_stage_descriptor_update_after_bind_sampled_images,
                limit_name_all: "max_per_stage_descriptor_update_after_bind_sampled_images",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03025"],
                get_limit_not_uab: |p| p.max_per_stage_descriptor_sampled_images,
                limit_name_not_uab: "max_per_stage_descriptor_sampled_images",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-06939"],
            },
            DescriptorLimit {
                descriptor_types: &[
                    DescriptorType::StorageImage,
                    DescriptorType::StorageTexelBuffer,
                ],
                get_limit_all: |p| p.max_per_stage_descriptor_update_after_bind_storage_images,
                limit_name_all: "max_per_stage_descriptor_update_after_bind_storage_images",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03026"],
                get_limit_not_uab: |p| p.max_per_stage_descriptor_storage_images,
                limit_name_not_uab: "max_per_stage_descriptor_storage_images",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03020"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::InputAttachment],
                get_limit_all: |p| p.max_per_stage_descriptor_update_after_bind_input_attachments,
                limit_name_all: "max_per_stage_descriptor_update_after_bind_input_attachments",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03027"],
                get_limit_not_uab: |p| p.max_per_stage_descriptor_input_attachments,
                limit_name_not_uab: "max_per_stage_descriptor_input_attachments",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03021"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::InlineUniformBlock],
                get_limit_all: |p| {
                    p.max_per_stage_descriptor_update_after_bind_inline_uniform_blocks
                },
                limit_name_all: "max_per_stage_descriptor_update_after_bind_inline_uniform_blocks",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-02215"],
                get_limit_not_uab: |p| {
                    p.max_per_stage_descriptor_inline_uniform_blocks
                        .unwrap_or(0)
                },
                limit_name_not_uab: "max_per_stage_descriptor_inline_uniform_blocks",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-02214"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::AccelerationStructure],
                get_limit_all: |p| {
                    p.max_per_stage_descriptor_update_after_bind_acceleration_structures
                },
                limit_name_all:
                    "max_per_stage_descriptor_update_after_bind_acceleration_structures",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03572"],
                get_limit_not_uab: |p| {
                    p.max_per_stage_descriptor_acceleration_structures
                        .unwrap_or(0)
                },
                limit_name_not_uab: "max_per_stage_descriptor_acceleration_structures",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03571"],
            },
        ];

        const TOTAL_DESCRIPTOR_LIMITS: [DescriptorLimit; 10] = [
            DescriptorLimit {
                descriptor_types: &[
                    DescriptorType::Sampler,
                    DescriptorType::CombinedImageSampler,
                ],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_samplers,
                limit_name_all: "max_descriptor_set_update_after_bind_samplers",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-03036"],
                get_limit_not_uab: |p| p.max_descriptor_set_samplers,
                limit_name_not_uab: "max_descriptor_set_samplers",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03028"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::UniformBuffer],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_uniform_buffers,
                limit_name_all: "max_descriptor_set_update_after_bind_uniform_buffers",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-03037"],
                get_limit_not_uab: |p| p.max_descriptor_set_uniform_buffers,
                limit_name_not_uab: "max_descriptor_set_uniform_buffers",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03029"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::UniformBufferDynamic],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_uniform_buffers_dynamic,
                limit_name_all: "max_descriptor_set_update_after_bind_uniform_buffers_dynamic",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-03038"],
                get_limit_not_uab: |p| p.max_descriptor_set_uniform_buffers_dynamic,
                limit_name_not_uab: "max_descriptor_set_uniform_buffers_dynamic",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03030"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::StorageBuffer],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_storage_buffers,
                limit_name_all: "max_descriptor_set_update_after_bind_storage_buffers",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-03039"],
                get_limit_not_uab: |p| p.max_descriptor_set_storage_buffers,
                limit_name_not_uab: "max_descriptor_set_storage_buffers",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03031"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::StorageBufferDynamic],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_storage_buffers_dynamic,
                limit_name_all: "max_descriptor_set_update_after_bind_storage_buffers_dynamic",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-03040"],
                get_limit_not_uab: |p| p.max_descriptor_set_storage_buffers_dynamic,
                limit_name_not_uab: "max_descriptor_set_storage_buffers_dynamic",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03032"],
            },
            DescriptorLimit {
                descriptor_types: &[
                    DescriptorType::CombinedImageSampler,
                    DescriptorType::SampledImage,
                    DescriptorType::UniformTexelBuffer,
                ],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_sampled_images,
                limit_name_all: "max_descriptor_set_update_after_bind_sampled_images",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-03041"],
                get_limit_not_uab: |p| p.max_descriptor_set_sampled_images,
                limit_name_not_uab: "max_descriptor_set_sampled_images",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03033"],
            },
            DescriptorLimit {
                descriptor_types: &[
                    DescriptorType::StorageImage,
                    DescriptorType::StorageTexelBuffer,
                ],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_storage_images,
                limit_name_all: "max_descriptor_set_update_after_bind_storage_images",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-03042"],
                get_limit_not_uab: |p| p.max_descriptor_set_storage_images,
                limit_name_not_uab: "max_descriptor_set_storage_images",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03034"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::InputAttachment],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_input_attachments,
                limit_name_all: "max_descriptor_set_update_after_bind_input_attachments",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-03043"],
                get_limit_not_uab: |p| p.max_descriptor_set_input_attachments,
                limit_name_not_uab: "max_descriptor_set_input_attachments",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03035"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::InlineUniformBlock],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_inline_uniform_blocks,
                limit_name_all: "max_descriptor_set_update_after_bind_inline_uniform_blocks",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-02217"],
                get_limit_not_uab: |p| p.max_descriptor_set_inline_uniform_blocks.unwrap_or(0),
                limit_name_not_uab: "max_descriptor_set_inline_uniform_blocks",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-02216"],
            },
            DescriptorLimit {
                descriptor_types: &[DescriptorType::AccelerationStructure],
                get_limit_all: |p| p.max_descriptor_set_update_after_bind_acceleration_structures,
                limit_name_all: "max_descriptor_set_update_after_bind_acceleration_structures",
                vuids_all: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03574"],
                get_limit_not_uab: |p| p.max_descriptor_set_acceleration_structures.unwrap_or(0),
                limit_name_not_uab: "max_descriptor_set_acceleration_structures",
                vuids_not_uab: &["VUID-VkPipelineLayoutCreateInfo-descriptorType-03573"],
            },
        ];

        let mut per_stage_descriptors_all: [HashMap<ShaderStage, u32>;
            PER_STAGE_DESCRIPTOR_LIMITS.len()] = array::from_fn(|_| HashMap::default());
        let mut per_stage_descriptors_not_uab: [HashMap<ShaderStage, u32>;
            PER_STAGE_DESCRIPTOR_LIMITS.len()] = array::from_fn(|_| HashMap::default());
        let mut total_descriptors_all = [0; TOTAL_DESCRIPTOR_LIMITS.len()];
        let mut total_descriptors_not_uab = [0; TOTAL_DESCRIPTOR_LIMITS.len()];
        let mut has_push_descriptor_set = false;

        for set_layout in set_layouts {
            assert_eq!(device, set_layout.device().as_ref());

            if set_layout
                .flags()
                .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
            {
                if has_push_descriptor_set {
                    return Err(Box::new(ValidationError {
                        context: "set_layouts".into(),
                        problem: "contains more than one descriptor set layout whose flags \
                            include `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR`"
                            .into(),
                        vuids: &["VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293"],
                        ..Default::default()
                    }));
                }

                has_push_descriptor_set = true;
            }

            let is_not_uab = !set_layout
                .flags()
                .intersects(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);

            for layout_binding in set_layout.bindings() {
                let &DescriptorSetLayoutBinding {
                    binding: _,
                    binding_flags: _,
                    descriptor_type,
                    descriptor_count,
                    stages,
                    immutable_samplers: _,
                    _ne: _,
                } = layout_binding;

                for ((limit, count_all), count_not_uab) in PER_STAGE_DESCRIPTOR_LIMITS
                    .iter()
                    .zip(&mut per_stage_descriptors_all)
                    .zip(&mut per_stage_descriptors_not_uab)
                {
                    if limit.descriptor_types.contains(&descriptor_type) {
                        for stage in stages {
                            *count_all.entry(stage).or_default() += descriptor_count;

                            if is_not_uab {
                                *count_not_uab.entry(stage).or_default() += descriptor_count;
                            }
                        }
                    }
                }

                for ((limit, count_all), count_not_uab) in TOTAL_DESCRIPTOR_LIMITS
                    .iter()
                    .zip(&mut total_descriptors_all)
                    .zip(&mut total_descriptors_not_uab)
                {
                    if limit.descriptor_types.contains(&descriptor_type) {
                        *count_all += descriptor_count;

                        if is_not_uab {
                            *count_not_uab += descriptor_count;
                        }
                    }
                }
            }
        }

        for ((limit, count_all), count_not_uab) in PER_STAGE_DESCRIPTOR_LIMITS
            .iter()
            .zip(per_stage_descriptors_all)
            .zip(per_stage_descriptors_not_uab)
        {
            let limit_not_uab = (limit.get_limit_not_uab)(properties);

            if let Some((max_stage, max_count)) = count_not_uab.into_iter().max_by_key(|(_, c)| *c)
            {
                if max_count > limit_not_uab {
                    return Err(Box::new(ValidationError {
                        context: "set_layouts".into(),
                        problem: format!(
                            "the combined number of {} descriptors, \
                            belonging to descriptor set layouts without the \
                            `DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL` flag, \
                            accessible to the `ShaderStage::{:?}` stage, \
                            exceeds the `{}` limit",
                            limit.descriptor_types[1..].iter().fold(
                                format!("`DescriptorType::{:?}`", limit.descriptor_types[0]),
                                |mut s, dt| {
                                    write!(s, " + `DescriptorType::{:?}`", dt).unwrap();
                                    s
                                }
                            ),
                            max_stage,
                            limit.limit_name_not_uab,
                        )
                        .into(),
                        vuids: limit.vuids_not_uab,
                        ..Default::default()
                    }));
                }
            }

            let limit_all = match (limit.get_limit_all)(properties) {
                Some(x) => x,
                None => continue,
            };

            if let Some((max_stage, max_count)) = count_all.into_iter().max_by_key(|(_, c)| *c) {
                if max_count > limit_all {
                    return Err(Box::new(ValidationError {
                        context: "set_layouts".into(),
                        problem: format!(
                            "the combined number of {} descriptors, \
                            accessible to the `ShaderStage::{:?}` stage, \
                            exceeds the `{}` limit",
                            limit.descriptor_types[1..].iter().fold(
                                format!("`DescriptorType::{:?}`", limit.descriptor_types[0]),
                                |mut s, dt| {
                                    write!(s, " + `DescriptorType::{:?}`", dt).unwrap();
                                    s
                                }
                            ),
                            max_stage,
                            limit.limit_name_all,
                        )
                        .into(),
                        vuids: limit.vuids_all,
                        ..Default::default()
                    }));
                }
            }
        }

        for ((limit, count_all), count_not_uab) in TOTAL_DESCRIPTOR_LIMITS
            .iter()
            .zip(total_descriptors_all)
            .zip(total_descriptors_not_uab)
        {
            let limit_not_uab = (limit.get_limit_not_uab)(properties);

            if count_not_uab > limit_not_uab {
                return Err(Box::new(ValidationError {
                    context: "set_layouts".into(),
                    problem: format!(
                        "the combined number of {} descriptors, \
                        belonging to descriptor set layouts without the \
                        `DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL` flag, \
                        accessible across all shader stages, \
                        exceeds the `{}` limit",
                        limit.descriptor_types[1..].iter().fold(
                            format!("`DescriptorType::{:?}`", limit.descriptor_types[0]),
                            |mut s, dt| {
                                write!(s, " + `DescriptorType::{:?}`", dt).unwrap();
                                s
                            }
                        ),
                        limit.limit_name_not_uab,
                    )
                    .into(),
                    vuids: limit.vuids_not_uab,
                    ..Default::default()
                }));
            }

            let limit_all = match (limit.get_limit_all)(properties) {
                Some(x) => x,
                None => continue,
            };

            if count_all > limit_all {
                return Err(Box::new(ValidationError {
                    context: "set_layouts".into(),
                    problem: format!(
                        "the combined number of {} descriptors, \
                        accessible across all shader stages, \
                        exceeds the `{}` limit",
                        limit.descriptor_types[1..].iter().fold(
                            format!("`DescriptorType::{:?}`", limit.descriptor_types[0]),
                            |mut s, dt| {
                                write!(s, " + `DescriptorType::{:?}`", dt).unwrap();
                                s
                            }
                        ),
                        limit.limit_name_all,
                    )
                    .into(),
                    vuids: limit.vuids_all,
                    ..Default::default()
                }));
            }
        }

        let mut seen_stages = ShaderStages::empty();

        for (range_index, range) in push_constant_ranges.iter().enumerate() {
            range
                .validate(device)
                .map_err(|err| err.add_context(format!("push_constant_ranges[{}]", range_index)))?;

            let &PushConstantRange {
                stages,
                offset: _,
                size: _,
            } = range;

            if seen_stages.intersects(stages) {
                return Err(Box::new(ValidationError {
                    context: "push_constant_ranges".into(),
                    problem: "contains more than one range with the same stage".into(),
                    vuids: &["VUID-VkPipelineLayoutCreateInfo-pPushConstantRanges-00292"],
                    ..Default::default()
                }));
            }

            seen_stages |= stages;
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a PipelineLayoutCreateInfoFields1Vk,
    ) -> vk::PipelineLayoutCreateInfo<'a> {
        let &Self {
            flags,
            set_layouts: _,
            push_constant_ranges: _,
            _ne: _,
        } = self;
        let PipelineLayoutCreateInfoFields1Vk {
            set_layouts_vk,
            push_constant_ranges_vk,
        } = fields1_vk;

        vk::PipelineLayoutCreateInfo::default()
            .flags(flags.into())
            .set_layouts(set_layouts_vk)
            .push_constant_ranges(push_constant_ranges_vk)
    }

    pub(crate) fn to_vk_fields1(&self) -> PipelineLayoutCreateInfoFields1Vk {
        let &Self {
            set_layouts,
            push_constant_ranges,
            ..
        } = self;

        let set_layouts_vk = set_layouts.iter().map(|l| l.handle()).collect();
        let push_constant_ranges_vk = push_constant_ranges
            .iter()
            .map(|range| range.to_vk())
            .collect();

        PipelineLayoutCreateInfoFields1Vk {
            set_layouts_vk,
            push_constant_ranges_vk,
        }
    }
}

pub(crate) struct PipelineLayoutCreateInfoFields1Vk {
    pub(crate) set_layouts_vk: SmallVec<[vk::DescriptorSetLayout; 4]>,
    pub(crate) push_constant_ranges_vk: SmallVec<[vk::PushConstantRange; 4]>,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a pipeline layout.
    PipelineLayoutCreateFlags = PipelineLayoutCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    INDEPENDENT_SETS = INDEPENDENT_SETS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_graphics_pipeline_library)]),
    ]), */
}

/// Description of a range of the push constants of a pipeline layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PushConstantRange {
    /// The stages which can access this range. A stage can access at most one push constant range.
    ///
    /// The default value is [`ShaderStages::empty()`], which must be overridden.
    pub stages: ShaderStages,

    /// Offset in bytes from the start of the push constants to this range.
    ///
    /// The value must be a multiple of 4.
    ///
    /// The default value is `0`.
    pub offset: u32,

    /// Size in bytes of the range.
    ///
    /// The value must be a multiple of 4, and not 0.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: u32,
}

impl Default for PushConstantRange {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl PushConstantRange {
    /// Returns a default `PushConstantRange`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            stages: ShaderStages::empty(),
            offset: 0,
            size: 0,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            stages,
            offset,
            size,
        } = self;

        stages.validate_device(device).map_err(|err| {
            err.add_context("stages")
                .set_vuids(&["VUID-VkPushConstantRange-stageFlags-parameter"])
        })?;

        if stages.is_empty() {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkPushConstantRange-stageFlags-requiredbitmask"],
                ..Default::default()
            }));
        }

        let max_push_constants_size = device
            .physical_device()
            .properties()
            .max_push_constants_size;

        if offset >= max_push_constants_size {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not less than the `max_push_constants_size` limit".into(),
                vuids: &["VUID-VkPushConstantRange-offset-00294"],
                ..Default::default()
            }));
        }

        if offset % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-VkPushConstantRange-offset-00295"],
                ..Default::default()
            }));
        }

        if size == 0 {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkPushConstantRange-size-00296"],
                ..Default::default()
            }));
        }

        if size % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-VkPushConstantRange-size-00297"],
                ..Default::default()
            }));
        }

        if size > max_push_constants_size - offset {
            return Err(Box::new(ValidationError {
                problem: "`size` is greater than `max_push_constants_size` limit minus `offset`"
                    .into(),
                vuids: &["VUID-VkPushConstantRange-size-00298"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_vk(&self) -> vk::PushConstantRange {
        let &Self {
            stages,
            offset,
            size,
        } = self;

        vk::PushConstantRange {
            stage_flags: stages.into(),
            offset,
            size,
        }
    }
}

/// Creates a set of push constant ranges from the union of the requirements of `stages`.
///
/// This can be used to simplify the creation of a [`PipelineLayout`].
pub fn push_constant_ranges_from_stages(
    stages: &[PipelineShaderStageCreateInfo<'_>],
) -> Vec<PushConstantRange> {
    let mut push_constant_ranges = Vec::<PushConstantRange>::new();

    for stage in stages {
        let entry_point_info = stage.entry_point.info();

        if let Some(range) = &entry_point_info.push_constant_requirements {
            if let Some(existing_range) = push_constant_ranges.iter_mut().find(|existing_range| {
                existing_range.offset == range.offset && existing_range.size == range.size
            }) {
                // If this range was already used before, add our stage to it.
                existing_range.stages |= range.stages;
            } else {
                // If this range is new, insert it.
                push_constant_ranges.push(*range);
            }
        }
    }

    push_constant_ranges
}

#[cfg(test)]
mod tests {
    use super::PipelineLayout;
    use crate::{
        pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange},
        shader::ShaderStages,
    };

    #[test]
    fn push_constant_ranges_disjoint() {
        let test_cases = [
            // input:
            // - `0..12`, stage=fragment
            // - `0..40`, stage=vertex
            //
            // output:
            // - `0..12`, stage=fragment|vertex
            // - `12..40`, stage=vertex
            (
                &[
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT,
                        offset: 0,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 0,
                        size: 40,
                    },
                ][..],
                &[
                    PushConstantRange {
                        stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                        offset: 0,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 12,
                        size: 28,
                    },
                ][..],
            ),
            // input:
            // - `0..12`, stage=fragment
            // - `4..40`, stage=vertex
            //
            // output:
            // - `0..4`, stage=fragment
            // - `4..12`, stage=fragment|vertex
            // - `12..40`, stage=vertex
            (
                &[
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT,
                        offset: 0,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 4,
                        size: 36,
                    },
                ][..],
                &[
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT,
                        offset: 0,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT | ShaderStages::VERTEX,
                        offset: 4,
                        size: 8,
                    },
                    PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 12,
                        size: 28,
                    },
                ][..],
            ),
            // input:
            // - `0..12`, stage=fragment
            // - `8..20`, stage=compute
            // - `4..16`, stage=vertex
            // - `8..32`, stage=tess_ctl
            //
            // output:
            // - `0..4`, stage=fragment
            // - `4..8`, stage=fragment|vertex
            // - `8..16`, stage=fragment|vertex|compute|tess_ctl
            // - `16..20`, stage=compute|tess_ctl
            // - `20..32` stage=tess_ctl
            (
                &[
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT,
                        offset: 0,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        offset: 8,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 4,
                        size: 12,
                    },
                    PushConstantRange {
                        stages: ShaderStages::TESSELLATION_CONTROL,
                        offset: 8,
                        size: 24,
                    },
                ][..],
                &[
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT,
                        offset: 0,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT | ShaderStages::VERTEX,
                        offset: 4,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages::VERTEX
                            | ShaderStages::FRAGMENT
                            | ShaderStages::COMPUTE
                            | ShaderStages::TESSELLATION_CONTROL,
                        offset: 8,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages::VERTEX
                            | ShaderStages::COMPUTE
                            | ShaderStages::TESSELLATION_CONTROL,
                        offset: 12,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages::COMPUTE | ShaderStages::TESSELLATION_CONTROL,
                        offset: 16,
                        size: 4,
                    },
                    PushConstantRange {
                        stages: ShaderStages::TESSELLATION_CONTROL,
                        offset: 20,
                        size: 12,
                    },
                ][..],
            ),
            // input:
            // - `0..8`, stage=vertex
            // - `16..32`, stage=fragment
            //
            // output:
            // - `0..8`, stage=vertex
            // - `16..32`, stage=fragment
            (
                &[
                    PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 0,
                        size: 8,
                    },
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT,
                        offset: 16,
                        size: 16,
                    },
                ][..],
                &[
                    PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 0,
                        size: 8,
                    },
                    PushConstantRange {
                        stages: ShaderStages::FRAGMENT,
                        offset: 16,
                        size: 16,
                    },
                ][..],
            ),
        ];

        let (device, _) = gfx_dev_and_queue!();

        for (input, expected) in test_cases {
            let layout = PipelineLayout::new(
                &device,
                &PipelineLayoutCreateInfo {
                    push_constant_ranges: input,
                    ..Default::default()
                },
            )
            .unwrap();

            assert_eq!(layout.push_constant_ranges_disjoint.as_slice(), expected);
        }
    }
}

/* TODO: restore
#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::Arc;
    use descriptor::descriptor::ShaderStages;
    use descriptor::descriptor_set::DescriptorSetLayout;
    use descriptor::pipeline_layout::sys::PipelineLayout;
    use descriptor::pipeline_layout::sys::PipelineLayoutCreationError;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = PipelineLayout::new(&device, iter::empty(), iter::empty()).unwrap();
    }

    #[test]
    fn wrong_device_panic() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let set = match DescriptorSetLayout::raw(device1, iter::empty()) {
            Ok(s) => Arc::new(s),
            Err(_) => return
        };

        assert_should_panic!({
            let _ = PipelineLayout::new(&device2, Some(&set), iter::empty());
        });
    }

    #[test]
    fn invalid_push_constant_stages() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 8, ShaderStages::empty());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }

    #[test]
    fn invalid_push_constant_size1() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 0, ShaderStages::all_graphics());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }

    #[test]
    fn invalid_push_constant_size2() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 11, ShaderStages::all_graphics());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }
}
*/
