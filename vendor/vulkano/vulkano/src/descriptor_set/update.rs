use super::{
    layout::{DescriptorSetLayout, DescriptorType},
    sys::RawDescriptorSet,
    DescriptorSet,
};
use crate::{
    acceleration_structure::{AccelerationStructure, AccelerationStructureType},
    buffer::{view::BufferView, Buffer, BufferUsage},
    descriptor_set::{
        layout::{DescriptorBindingFlags, DescriptorSetLayoutCreateFlags},
        pool::DescriptorPoolCreateFlags,
    },
    device::DeviceOwned,
    image::{
        sampler::Sampler,
        view::{ImageView, ImageViewType},
        ImageAspects, ImageCreateFlags, ImageLayout, ImageType, ImageUsage,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use smallvec::SmallVec;
use std::{slice, sync::Arc};

/// Represents a single write operation to the binding of a descriptor set.
///
/// `WriteDescriptorSet` specifies the binding number and target array index, and includes one or
/// more resources of a given type that need to be written to that location. Two constructors are
/// provided for each resource type:
/// - The basic constructor variant writes a single element to array index 0. It is intended for
///   non-arrayed bindings, where `descriptor_count` in the descriptor set layout is 1.
/// - The `_array` variant writes several elements and allows specifying the target array index. At
///   least one element must be provided; a panic results if the provided iterator is empty.
#[derive(Clone, Debug)]
pub struct WriteDescriptorSet<'a> {
    pub(crate) dst_binding: u32,
    pub(crate) dst_array_element: u32,
    pub(crate) elements: WriteDescriptorSetElements<'a>,
}

impl<'a> WriteDescriptorSet<'a> {
    /// Write a single image to array element 0.
    ///
    /// If the descriptor type is `Sampler` and immutable samplers are used, you must not write any
    /// descriptors. Meaning, you should leave out the `WriteDescriptorSet` for the binding. That
    /// is unless this is for a push descriptor set.
    #[inline]
    pub fn image(dst_binding: u32, image_info: &'a DescriptorImageInfo<'a>) -> Self {
        Self::image_array(dst_binding, 0, slice::from_ref(image_info))
    }

    /// Write a number of consecutive image elements.
    ///
    /// If the descriptor type is `Sampler` and immutable samplers are used, you must not write any
    /// descriptors. Meaning, you should leave out the `WriteDescriptorSet` for the binding. That
    /// is unless this is for a push descriptor set.
    ///
    /// # Panics
    ///
    /// - Panics if `image_infos` is empty.
    #[inline]
    pub fn image_array(
        dst_binding: u32,
        dst_array_element: u32,
        image_infos: &'a [DescriptorImageInfo<'a>],
    ) -> Self {
        assert!(!image_infos.is_empty());

        Self {
            dst_binding,
            dst_array_element,
            elements: WriteDescriptorSetElements::Image(image_infos),
        }
    }

    /// Write a single buffer to array element 0.
    #[inline]
    pub fn buffer(dst_binding: u32, buffer_info: &'a DescriptorBufferInfo<'a>) -> Self {
        Self::buffer_array(dst_binding, 0, slice::from_ref(buffer_info))
    }

    /// Write a number of consecutive buffer elements.
    ///
    /// # Panics
    ///
    /// - Panics if `buffer_infos` is empty.
    #[inline]
    pub fn buffer_array(
        dst_binding: u32,
        dst_array_element: u32,
        buffer_infos: &'a [DescriptorBufferInfo<'a>],
    ) -> Self {
        assert!(!buffer_infos.is_empty());

        Self {
            dst_binding,
            dst_array_element,
            elements: WriteDescriptorSetElements::Buffer(buffer_infos),
        }
    }

    /// Write a single buffer view to array element 0.
    ///
    /// If the buffer view is `None`, the [`null_descriptor`] feature must be enabled on the
    /// device.
    ///
    /// [`null_descriptor`]: crate::device::DeviceFeatures::null_descriptor
    #[inline]
    pub fn buffer_view(dst_binding: u32, buffer_view: &'a Option<&'a Arc<BufferView>>) -> Self {
        Self::buffer_view_array(dst_binding, 0, slice::from_ref(buffer_view))
    }

    /// Write a number of consecutive buffer view elements.
    ///
    /// If a buffer view is `None`, the [`null_descriptor`] feature must be enabled on the device.
    ///
    /// # Panics
    ///
    /// - Panics if `buffer_views` is empty.
    ///
    /// [`null_descriptor`]: crate::device::DeviceFeatures::null_descriptor
    #[inline]
    pub fn buffer_view_array(
        dst_binding: u32,
        dst_array_element: u32,
        buffer_views: &'a [Option<&'a Arc<BufferView>>],
    ) -> Self {
        assert!(!buffer_views.is_empty());

        Self {
            dst_binding,
            dst_array_element,
            elements: WriteDescriptorSetElements::BufferView(buffer_views),
        }
    }

    /// Write data to an inline uniform block.
    ///
    /// `offset` and the length of `data` must be a multiple of 4.
    pub fn inline_uniform_block(dst_binding: u32, dst_offset: u32, data: &'a [u8]) -> Self {
        Self {
            dst_binding,
            dst_array_element: dst_offset,
            elements: WriteDescriptorSetElements::InlineUniformBlock(data),
        }
    }

    /// Write a single acceleration structure to array element 0.
    ///
    /// If the acceleration structure is `None`, the [`null_descriptor`] feature must be enabled on
    /// the device.
    ///
    /// [`null_descriptor`]: crate::device::DeviceFeatures::null_descriptor
    #[inline]
    pub fn acceleration_structure(
        dst_binding: u32,
        acceleration_structure: &'a Option<&'a Arc<AccelerationStructure>>,
    ) -> Self {
        Self::acceleration_structure_array(dst_binding, 0, slice::from_ref(acceleration_structure))
    }

    /// Write a number of consecutive acceleration structure elements.
    ///
    /// If an acceleration structure is `None`, the [`null_descriptor`] feature must be enabled on
    /// the device.
    ///
    /// # Panics
    ///
    /// - Panics if `acceleration_structures` is empty.
    ///
    /// [`null_descriptor`]: crate::device::DeviceFeatures::null_descriptor
    pub fn acceleration_structure_array(
        dst_binding: u32,
        dst_array_element: u32,
        acceleration_structures: &'a [Option<&'a Arc<AccelerationStructure>>],
    ) -> Self {
        assert!(!acceleration_structures.is_empty());

        Self {
            dst_binding,
            dst_array_element,
            elements: WriteDescriptorSetElements::AccelerationStructure(acceleration_structures),
        }
    }

    /// Returns the binding number that is updated by this descriptor write.
    #[inline]
    pub fn dst_binding(&self) -> u32 {
        self.dst_binding
    }

    /// Returns the first array element in the binding that is updated by this descriptor write.
    #[inline]
    pub fn dst_array_element(&self) -> u32 {
        self.dst_array_element
    }

    /// Returns the elements held by this descriptor write.
    #[inline]
    pub fn elements(&self) -> WriteDescriptorSetElements<'a> {
        self.elements
    }

    pub(crate) fn validate(
        &self,
        layout: &DescriptorSetLayout,
        variable_descriptor_count: u32,
    ) -> Result<(), Box<ValidationError>> {
        fn provided_element_type(elements: WriteDescriptorSetElements<'_>) -> &'static str {
            match elements {
                WriteDescriptorSetElements::Buffer(_) => "buffer",
                WriteDescriptorSetElements::BufferView(_) => "buffer_view",
                WriteDescriptorSetElements::Image(_) => "image",
                WriteDescriptorSetElements::InlineUniformBlock(_) => "inline_uniform_block",
                WriteDescriptorSetElements::AccelerationStructure(_) => "acceleration_structure",
            }
        }

        let &Self {
            dst_binding,
            dst_array_element,
            elements,
        } = self;

        let device = layout.device();

        let Some(layout_binding) = layout.binding(dst_binding) else {
            return Err(Box::new(ValidationError {
                context: "dst_binding".into(),
                problem: "does not exist in the descriptor set layout".into(),
                vuids: &["VUID-VkWriteDescriptorSet-dstBinding-00315"],
                ..Default::default()
            }));
        };

        let max_descriptor_count = if layout_binding
            .binding_flags
            .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
        {
            variable_descriptor_count
        } else {
            layout_binding.descriptor_count
        };

        let array_element_count = elements.len();
        debug_assert!(array_element_count != 0);

        let validate_image_view =
            |image_view: &ImageView, index: usize| -> Result<(), Box<ValidationError>> {
                if image_view.image().image_type() == ImageType::Dim3d {
                    if image_view.view_type() == ImageViewType::Dim2dArray {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}]", index).into(),
                            problem: "the image view's type is `ImageViewType::Dim2dArray`, and \
                                was created from a 3D image"
                                .into(),
                            vuids: &["VUID-VkDescriptorImageInfo-imageView-00343"],
                            ..Default::default()
                        }));
                    } else if image_view.view_type() == ImageViewType::Dim2d {
                        if !image_view
                            .image()
                            .flags()
                            .intersects(ImageCreateFlags::DIM2D_VIEW_COMPATIBLE)
                        {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "the image view's type is `ImageViewType::Dim2d`, and \
                                    was created from a 3D image, but the image's flags do not \
                                    contain `ImageCreateFlags::DIM2D_VIEW_COMPATIBLE`"
                                    .into(),
                                vuids: &["VUID-VkDescriptorImageInfo-imageView-07796"],
                                ..Default::default()
                            }));
                        }

                        match layout_binding.descriptor_type {
                            DescriptorType::StorageImage => {
                                if !device.enabled_features().image2_d_view_of3_d {
                                    return Err(Box::new(ValidationError {
                                        context: format!("elements[{}]", index).into(),
                                        problem: format!(
                                            "the descriptor type is `DescriptorType::{:?}`, and \
                                            the image view's type is `ImageViewType::Dim2d`,
                                            and was created from a 3D image",
                                            layout_binding.descriptor_type,
                                        )
                                        .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature("image2_d_view_of3_d"),
                                        ])]),
                                        vuids: &["VUID-VkDescriptorImageInfo-descriptorType-06713"],
                                    }));
                                }
                            }
                            DescriptorType::SampledImage | DescriptorType::CombinedImageSampler => {
                                if !device.enabled_features().sampler2_d_view_of3_d {
                                    return Err(Box::new(ValidationError {
                                        context: format!("elements[{}]", index).into(),
                                        problem: format!(
                                            "the descriptor type is `DescriptorType::{:?}`, and \
                                            the image view's type is `ImageViewType::Dim2d`,
                                            and was created from a 3D image",
                                            layout_binding.descriptor_type,
                                        )
                                        .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature("sampler2_d_view_of3_d"),
                                        ])]),
                                        vuids: &["VUID-VkDescriptorImageInfo-descriptorType-06714"],
                                    }));
                                }
                            }
                            _ => {
                                return Err(Box::new(ValidationError {
                                    context: format!("elements[{}]", index).into(),
                                    problem: "the descriptor type is not \
                                        `DescriptorType::StorageImage`, \
                                        `DescriptorType::SampledImage` or \
                                        `DescriptorType::CombinedImageSampler`,\
                                        and the image view's type is `ImageViewType::Dim2D`, \
                                        and was created from a 3D image"
                                        .into(),
                                    vuids: &["VUID-VkDescriptorImageInfo-imageView-07795"],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }

                if image_view
                    .subresource_range()
                    .aspects
                    .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                {
                    return Err(Box::new(ValidationError {
                        context: format!("elements[{}]", index).into(),
                        problem: "the image view's aspects include both a depth and a \
                            stencil component"
                            .into(),
                        vuids: &["VUID-VkDescriptorImageInfo-imageView-01976"],
                        ..Default::default()
                    }));
                }

                Ok(())
            };

        let default_image_layout = layout_binding.descriptor_type.default_image_layout();

        match layout_binding.descriptor_type {
            DescriptorType::Sampler => {
                let elements = if let WriteDescriptorSetElements::Image(elements) = elements {
                    elements
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `sampler` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                };

                if layout_binding.immutable_samplers.is_empty() {
                    for (index, image_info) in elements.iter().enumerate() {
                        let Some(sampler) = image_info.sampler else {
                            // For the `Sampler` descriptor type without immutable samplers,
                            // samplers must always be provided (can't be `None` even with the null
                            // descriptor feature).
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].sampler", index).into(),
                                problem: "is `None`, but samplers must be provided for \
                                    `DescriptorType::Sampler` when the descriptor set layout \
                                    was not created with immutable samplers"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00325"],
                                ..Default::default()
                            }));
                        };

                        assert_eq!(device, sampler.device());

                        if sampler.sampler_ycbcr_conversion().is_some() {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].sampler", index).into(),
                                problem: "the descriptor type is not \
                                    `DescriptorType::CombinedImageSampler`, and the sampler has a \
                                    sampler YCbCr conversion"
                                    .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }

                        if device.enabled_extensions().khr_portability_subset
                            && !device.enabled_features().mutable_comparison_samplers
                            && sampler.compare().is_some()
                        {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].sampler", index).into(),
                                problem: "this device is a portability subset device, and \
                                    the sampler has depth comparison enabled"
                                    .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::DeviceFeature("mutable_comparison_samplers"),
                                ])]),
                                vuids: &[
                                    "VUID-VkDescriptorImageInfo-mutableComparisonSamplers-04450",
                                ],
                            }));
                        }
                    }
                } else if layout
                    .flags()
                    .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
                {
                    for (index, image_info) in elements.iter().enumerate() {
                        if image_info.sampler.is_some() {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].sampler", index).into(),
                                problem: format!(
                                    "is `Some`, but descriptor set binding {} requires `None` for \
                                    push descriptors",
                                    dst_binding,
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }
                } else {
                    // For regular descriptors with immutable samplers, no element must be written.
                    return Err(Box::new(ValidationError {
                        context: "dst_binding".into(),
                        problem: "no descriptors must be written to this descriptor set binding"
                            .into(),
                        vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02752"],
                        ..Default::default()
                    }));
                }
            }
            DescriptorType::CombinedImageSampler => {
                if layout_binding.immutable_samplers.is_empty() {
                    let elements = if let WriteDescriptorSetElements::Image(elements) = elements {
                        elements
                    } else {
                        return Err(Box::new(ValidationError {
                            context: "elements".into(),
                            problem: format!(
                                "contains `{}` elements, but descriptor set binding {} \
                                requires `image` elements",
                                provided_element_type(elements),
                                dst_binding,
                            )
                            .into(),
                            // vuids?
                            ..Default::default()
                        }));
                    };

                    for (index, image_info) in elements.iter().enumerate() {
                        let &DescriptorImageInfo {
                            sampler,
                            image_view,
                            mut image_layout,
                        } = image_info;

                        // For the `CombinedImageSampler` descriptor type without immutable
                        // samplers, the sampler must always be provided (can't be `None` even with
                        // the null descriptor feature).
                        let Some(sampler) = sampler else {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].sampler", index).into(),
                                problem: "is `None`, but samplers must be provided for \
                                    `DescriptorType::CombinedImageSampler` when the descriptor \
                                    set layout was not created with immutable samplers"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00325"],
                                ..Default::default()
                            }));
                        };

                        let Some(image_view) = image_view else {
                            if !device.enabled_features().null_descriptor {
                                return Err(Box::new(ValidationError {
                                    context: format!("elements[{}].image_view", index).into(),
                                    problem: "is `None`, but the `null_descriptor` feature is not \
                                        enabled on the device"
                                        .into(),
                                    vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02997"],
                                    ..Default::default()
                                }));
                            }

                            continue;
                        };

                        if image_layout == ImageLayout::Undefined {
                            image_layout = default_image_layout;
                        }

                        assert_eq!(device, image_view.device());
                        assert_eq!(device, sampler.device());

                        validate_image_view(image_view.as_ref(), index)?;

                        if !image_view.usage().intersects(ImageUsage::SAMPLED) {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].image_view", index).into(),
                                problem: "the descriptor type is \
                                    `DescriptorType::SampledImage` or \
                                    `DescriptorType::CombinedImageSampler`, and the image was not \
                                    created with the `ImageUsage::SAMPLED` usage"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00337"],
                                ..Default::default()
                            }));
                        }

                        if !matches!(
                            image_layout,
                            ImageLayout::DepthStencilReadOnlyOptimal
                                | ImageLayout::ShaderReadOnlyOptimal
                                | ImageLayout::General
                                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                                | ImageLayout::DepthReadOnlyOptimal
                                | ImageLayout::StencilReadOnlyOptimal,
                        ) {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].image_layout", index).into(),
                                problem: "the descriptor type is \
                                    `DescriptorType::CombinedImageSampler`, and the image layout \
                                    is not valid with this type"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-04150"],
                                ..Default::default()
                            }));
                        }

                        if device.enabled_extensions().khr_portability_subset
                            && !device.enabled_features().mutable_comparison_samplers
                            && sampler.compare().is_some()
                        {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].sampler", index).into(),
                                problem: "this device is a portability subset device, and \
                                    the sampler has depth comparison enabled"
                                    .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::DeviceFeature("mutable_comparison_samplers"),
                                ])]),
                                vuids: &[
                                    "VUID-VkDescriptorImageInfo-mutableComparisonSamplers-04450",
                                ],
                            }));
                        }

                        if image_view.sampler_ycbcr_conversion().is_some() {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].image_view", index).into(),
                                problem: "the image view has a sampler YCbCr conversion, and the \
                                    descriptor set layout was not created with immutable samplers"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02738"],
                                ..Default::default()
                            }));
                        }

                        if sampler.sampler_ycbcr_conversion().is_some() {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].sampler", index).into(),
                                problem: "the sampler has a sampler YCbCr conversion".into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }

                        sampler
                            .check_can_sample(image_view.as_ref())
                            .map_err(|err| err.add_context(format!("elements[{}]", index)))?;
                    }
                } else {
                    let elements = if let WriteDescriptorSetElements::Image(elements) = elements {
                        elements
                    } else {
                        return Err(Box::new(ValidationError {
                            context: "elements".into(),
                            problem: format!(
                                "contains `{}` elements, but descriptor set binding {} \
                                requires `image` elements",
                                provided_element_type(elements),
                                dst_binding,
                            )
                            .into(),
                            ..Default::default()
                        }));
                    };

                    let immutable_samplers = &layout_binding.immutable_samplers
                        [dst_array_element as usize..][..array_element_count as usize];

                    for (index, (image_info, sampler)) in
                        elements.iter().zip(immutable_samplers).enumerate()
                    {
                        let &DescriptorImageInfo {
                            sampler: image_info_sampler,
                            image_view,
                            mut image_layout,
                        } = image_info;

                        // For the `CombinedImageSampler` descriptor type with immutable samplers,
                        // the `sampler` field should be `None`.
                        if image_info_sampler.is_some() {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].sampler", index).into(),
                                problem: "is `Some`, but the descriptor set layout was created \
                                    with immutable samplers"
                                    .into(),
                                // No VUIDs because Vulkan ignores the field in this case.
                                ..Default::default()
                            }));
                        }

                        let Some(image_view) = image_view else {
                            if !device.enabled_features().null_descriptor {
                                return Err(Box::new(ValidationError {
                                    context: format!("elements[{}].image_view", index).into(),
                                    problem: "is `None`, but the `null_descriptor` feature is not \
                                        enabled on the device"
                                        .into(),
                                    vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02997"],
                                    ..Default::default()
                                }));
                            }

                            continue;
                        };

                        if image_layout == ImageLayout::Undefined {
                            image_layout = default_image_layout;
                        }

                        assert_eq!(device, image_view.device());

                        validate_image_view(image_view.as_ref(), index)?;

                        if !image_view.usage().intersects(ImageUsage::SAMPLED) {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].image_view", index).into(),
                                problem: "the descriptor type is \
                                    `DescriptorType::SampledImage` or \
                                    `DescriptorType::CombinedImageSampler`, and the image was not \
                                    created with the `ImageUsage::SAMPLED` usage"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00337"],
                                ..Default::default()
                            }));
                        }

                        if !matches!(
                            image_layout,
                            ImageLayout::DepthStencilReadOnlyOptimal
                                | ImageLayout::ShaderReadOnlyOptimal
                                | ImageLayout::General
                                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                                | ImageLayout::DepthReadOnlyOptimal
                                | ImageLayout::StencilReadOnlyOptimal,
                        ) {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].image_layout", index).into(),
                                problem: "the descriptor type is \
                                    `DescriptorType::CombinedImageSampler`, and the image layout \
                                    is not valid with this type"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-04150"],
                                ..Default::default()
                            }));
                        }

                        sampler
                            .check_can_sample(image_view.as_ref())
                            .map_err(|err| err.add_context(format!("elements[{}]", index)))?;
                    }
                }
            }
            DescriptorType::SampledImage => {
                let elements = if let WriteDescriptorSetElements::Image(elements) = elements {
                    elements
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `image` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                };

                for (index, image_info) in elements.iter().enumerate() {
                    let &DescriptorImageInfo {
                        sampler,
                        image_view,
                        mut image_layout,
                    } = image_info;

                    // For the `SampledImage` descriptor type, the `sampler` field should be
                    // `None`.
                    if sampler.is_some() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].sampler", index).into(),
                            problem: "is `Some`, but the descriptor type is \
                                `DescriptorType::SampledImage`"
                                .into(),
                            // No VUIDs because Vulkan ignores the field in this case.
                            ..Default::default()
                        }));
                    }

                    let Some(image_view) = image_view else {
                        if !device.enabled_features().null_descriptor {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].image_view", index).into(),
                                problem: "is `None`, but the `null_descriptor` feature is not \
                                    enabled on the device"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02997"],
                                ..Default::default()
                            }));
                        }

                        continue;
                    };

                    if image_layout == ImageLayout::Undefined {
                        image_layout = default_image_layout;
                    }

                    assert_eq!(device, image_view.device());

                    validate_image_view(image_view.as_ref(), index)?;

                    if !image_view.usage().intersects(ImageUsage::SAMPLED) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "the descriptor type is \
                                `DescriptorType::SampledImage` or \
                                `DescriptorType::CombinedImageSampler`, and the image was not \
                                created with the `ImageUsage::SAMPLED` usage"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00337"],
                            ..Default::default()
                        }));
                    }

                    if !matches!(
                        image_layout,
                        ImageLayout::DepthStencilReadOnlyOptimal
                            | ImageLayout::ShaderReadOnlyOptimal
                            | ImageLayout::General
                            | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                            | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                            | ImageLayout::DepthReadOnlyOptimal
                            | ImageLayout::StencilReadOnlyOptimal,
                    ) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_layout", index).into(),
                            problem: "the descriptor type is \
                                `DescriptorType::SampledImage`, and the image layout is \
                                not valid with this type"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-04149"],
                            ..Default::default()
                        }));
                    }

                    if image_view.sampler_ycbcr_conversion().is_some() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "the descriptor type is `DescriptorType::SampledImage`, and \
                                the image view has a sampler YCbCr conversion"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-01946"],
                            ..Default::default()
                        }));
                    }
                }
            }
            DescriptorType::StorageImage => {
                let elements = if let WriteDescriptorSetElements::Image(elements) = elements {
                    elements
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `image` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                };

                for (index, image_info) in elements.iter().enumerate() {
                    let &DescriptorImageInfo {
                        sampler,
                        image_view,
                        mut image_layout,
                    } = image_info;

                    // For the `StorageImage` descriptor type, the `sampler` field should be
                    // `None`.
                    if sampler.is_some() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].sampler", index).into(),
                            problem: "is `Some`, but the descriptor type is \
                                `DescriptorType::StorageImage`"
                                .into(),
                            // No VUIDs because Vulkan ignores the field in this case.
                            ..Default::default()
                        }));
                    }

                    let Some(image_view) = image_view else {
                        if !device.enabled_features().null_descriptor {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].image_view", index).into(),
                                problem: "is `None`, but the `null_descriptor` feature is not \
                                    enabled on the device"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02997"],
                                ..Default::default()
                            }));
                        }

                        continue;
                    };

                    if image_layout == ImageLayout::Undefined {
                        image_layout = default_image_layout;
                    }

                    assert_eq!(device, image_view.device());

                    validate_image_view(image_view.as_ref(), index)?;

                    if !image_view.usage().intersects(ImageUsage::STORAGE) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "the descriptor type is \
                                `DescriptorType::StorageImage`, and the image was not \
                                created with the `ImageUsage::STORAGE` usage"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00339"],
                            ..Default::default()
                        }));
                    }

                    if !matches!(image_layout, ImageLayout::General) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_layout", index).into(),
                            problem: "the descriptor type is \
                                `DescriptorType::StorageImage`, and the image layout is \
                                not valid with this type"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-04152"],
                            ..Default::default()
                        }));
                    }

                    if !image_view.component_mapping().is_identity() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "the descriptor type is `DescriptorType::StorageImage` or \
                                `DescriptorType::InputAttachment`, and the image view is not \
                                identity swizzled"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00336"],
                            ..Default::default()
                        }));
                    }

                    if image_view.sampler_ycbcr_conversion().is_some() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "the image view has a sampler YCbCr conversion".into(),
                            // vuids?
                            ..Default::default()
                        }));
                    }
                }
            }
            DescriptorType::UniformTexelBuffer => {
                let elements = if let WriteDescriptorSetElements::BufferView(elements) = elements {
                    elements
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `buffer_view` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                };

                for (index, buffer_view) in elements.iter().enumerate() {
                    let Some(buffer_view) = buffer_view else {
                        if !device.enabled_features().null_descriptor {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "is `None`, but the `null_descriptor` feature is not \
                                    enabled on the device"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02995"],
                                ..Default::default()
                            }));
                        }

                        continue;
                    };

                    assert_eq!(device, buffer_view.device());

                    if !buffer_view
                        .buffer()
                        .usage()
                        .intersects(BufferUsage::UNIFORM_TEXEL_BUFFER)
                    {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}]", index).into(),
                            problem: "the descriptor type is \
                                `DescriptorType::UniformTexelBuffer`, and the buffer was not \
                                created with the `BufferUsage::UNIFORM_TEXEL_BUFFER` usage"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-08765"],
                            ..Default::default()
                        }));
                    }
                }
            }
            DescriptorType::StorageTexelBuffer => {
                let elements = if let WriteDescriptorSetElements::BufferView(elements) = elements {
                    elements
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `buffer_view` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                };

                for (index, buffer_view) in elements.iter().enumerate() {
                    let Some(buffer_view) = buffer_view else {
                        if !device.enabled_features().null_descriptor {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "is `None`, but the `null_descriptor` feature is not \
                                    enabled on the device"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02995"],
                                ..Default::default()
                            }));
                        }

                        continue;
                    };

                    assert_eq!(device, buffer_view.device());

                    // TODO: storage_texel_buffer_atomic
                    if !buffer_view
                        .buffer()
                        .usage()
                        .intersects(BufferUsage::STORAGE_TEXEL_BUFFER)
                    {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}]", index).into(),
                            problem: "the descriptor type is \
                                `DescriptorType::StorageTexelBuffer`, and the buffer was not \
                                created with the `BufferUsage::STORAGE_TEXEL_BUFFER` usage"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-08766"],
                            ..Default::default()
                        }));
                    }
                }
            }
            DescriptorType::UniformBuffer | DescriptorType::UniformBufferDynamic => {
                let elements = if let WriteDescriptorSetElements::Buffer(elements) = elements {
                    elements
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `buffer` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                };

                for (index, buffer_info) in elements.iter().enumerate() {
                    let &DescriptorBufferInfo {
                        buffer,
                        offset,
                        range,
                    } = buffer_info;

                    let Some(buffer) = buffer else {
                        if !device.enabled_features().null_descriptor {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].buffer", index).into(),
                                problem: "is `None`, but the `null_descriptor` feature is not \
                                    enabled on the device"
                                    .into(),
                                vuids: &["VUID-VkDescriptorBufferInfo-buffer-02998"],
                                ..Default::default()
                            }));
                        }

                        if offset != 0 {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "`buffer` is `None`, but `offset` is not zero".into(),
                                vuids: &["VUID-VkDescriptorBufferInfo-buffer-02999"],
                                ..Default::default()
                            }));
                        }

                        if range.is_some() {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "`buffer` is `None`, but `range` is not `None`".into(),
                                vuids: &["VUID-VkDescriptorBufferInfo-buffer-02999"],
                                ..Default::default()
                            }));
                        }

                        continue;
                    };

                    assert_eq!(device, buffer.device());

                    if !buffer.usage().intersects(BufferUsage::UNIFORM_BUFFER) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}]", index).into(),
                            problem: "the descriptor type is `DescriptorType::UniformBuffer` or \
                                `DescriptorType::UniformBufferDynamic`, and the buffer was not \
                                created with the `BufferUsage::UNIFORM_BUFFER` usage"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00330"],
                            ..Default::default()
                        }));
                    }

                    assert_ne!(range, Some(0));

                    if offset >= buffer.size() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}]", index).into(),
                            problem: "`offset` is not less than the size of the buffer".into(),
                            vuids: &["VUID-VkDescriptorBufferInfo-offset-00340"],
                            ..Default::default()
                        }));
                    }

                    if let Some(range) = range {
                        if range > buffer.size() - offset {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "`offset + range` is greater than the size of the buffer"
                                    .into(),
                                vuids: &["VUID-VkDescriptorBufferInfo-range-00342"],
                                ..Default::default()
                            }));
                        }
                    }
                }
            }
            DescriptorType::StorageBuffer | DescriptorType::StorageBufferDynamic => {
                let elements = if let WriteDescriptorSetElements::Buffer(elements) = elements {
                    elements
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `buffer` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                };

                for (index, buffer_info) in elements.iter().enumerate() {
                    let &DescriptorBufferInfo {
                        buffer,
                        offset,
                        range,
                    } = buffer_info;

                    let Some(buffer) = buffer else {
                        if !device.enabled_features().null_descriptor {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}].buffer", index).into(),
                                problem: "is `None`, but the `null_descriptor` feature is not \
                                    enabled on the device"
                                    .into(),
                                vuids: &["VUID-VkDescriptorBufferInfo-buffer-02998"],
                                ..Default::default()
                            }));
                        }

                        if offset != 0 {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "`buffer` is `None`, but `offset` is not zero".into(),
                                vuids: &["VUID-VkDescriptorBufferInfo-buffer-02999"],
                                ..Default::default()
                            }));
                        }

                        if range.is_some() {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "`buffer` is `None`, but `range` is not `None`".into(),
                                vuids: &["VUID-VkDescriptorBufferInfo-buffer-02999"],
                                ..Default::default()
                            }));
                        }

                        continue;
                    };

                    assert_eq!(device, buffer.device());

                    if !buffer.usage().intersects(BufferUsage::STORAGE_BUFFER) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}]", index).into(),
                            problem: "the descriptor type is `DescriptorType::StorageBuffer` or \
                                `DescriptorType::StorageBufferDynamic`, and the buffer was not \
                                created with the `BufferUsage::STORAGE_BUFFER` usage"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00331"],
                            ..Default::default()
                        }));
                    }

                    assert_ne!(range, Some(0));

                    if offset >= buffer.size() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}]", index).into(),
                            problem: "`offset` is not less than the size of the buffer".into(),
                            vuids: &["VUID-VkDescriptorBufferInfo-offset-00340"],
                            ..Default::default()
                        }));
                    }

                    if let Some(range) = range {
                        if range > buffer.size() - offset {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "`offset + range` is greater than the size of the buffer"
                                    .into(),
                                vuids: &["VUID-VkDescriptorBufferInfo-range-00342"],
                                ..Default::default()
                            }));
                        }
                    }
                }
            }
            DescriptorType::InputAttachment => {
                let elements = if let WriteDescriptorSetElements::Image(elements) = elements {
                    elements
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `image` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                };

                for (index, image_info) in elements.iter().enumerate() {
                    let &DescriptorImageInfo {
                        sampler,
                        image_view,
                        mut image_layout,
                    } = image_info;

                    // For the `InputAttachment` descriptor type, the `sampler` field should be
                    // `None`.
                    if sampler.is_some() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].sampler", index).into(),
                            problem: "is `Some`, but the descriptor type is \
                                `DescriptorType::InputAttachment`"
                                .into(),
                            // No VUIDs because Vulkan ignores the field in this case.
                            ..Default::default()
                        }));
                    }

                    // Input attachments must never be `None` (even with the null descriptor
                    // feature).
                    let Some(image_view) = image_view else {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "is `None`, but input attachment image views must not be \
                                `None`"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-07683"],
                            ..Default::default()
                        }));
                    };

                    if image_layout == ImageLayout::Undefined {
                        image_layout = default_image_layout;
                    }

                    assert_eq!(device, image_view.device());

                    validate_image_view(image_view.as_ref(), index)?;

                    if !image_view.usage().intersects(ImageUsage::INPUT_ATTACHMENT) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "the descriptor type is \
                                `DescriptorType::InputAttachment`, and the image was not \
                                created with the `ImageUsage::INPUT_ATTACHMENT` usage"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00338"],
                            ..Default::default()
                        }));
                    }

                    if !matches!(
                        image_layout,
                        ImageLayout::DepthStencilReadOnlyOptimal
                            | ImageLayout::ShaderReadOnlyOptimal
                            | ImageLayout::General
                            | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                            | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                            | ImageLayout::DepthReadOnlyOptimal
                            | ImageLayout::StencilReadOnlyOptimal,
                    ) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_layout", index).into(),
                            problem: "the descriptor type is \
                                `DescriptorType::InputAttachment`, and the image layout is \
                                not valid with this type"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-04151"],
                            ..Default::default()
                        }));
                    }

                    if !image_view.component_mapping().is_identity() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "the descriptor type is `DescriptorType::StorageImage` or \
                                `DescriptorType::InputAttachment`, and the image view is not \
                                identity swizzled"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSet-descriptorType-00336"],
                            ..Default::default()
                        }));
                    }

                    if image_view.sampler_ycbcr_conversion().is_some() {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}].image_view", index).into(),
                            problem: "the descriptor type is `DescriptorType::InputAttachment`, \
                                and the image view has a sampler YCbCr conversion"
                                .into(),
                            // vuids?
                            ..Default::default()
                        }));
                    }
                }
            }
            DescriptorType::InlineUniformBlock => {
                let data = if let WriteDescriptorSetElements::InlineUniformBlock(data) = elements {
                    data
                } else {
                    return Err(Box::new(ValidationError {
                        context: "elements".into(),
                        problem: format!(
                            "contains `{}` elements, but descriptor set binding {} \
                            requires `inline_uniform_block` elements",
                            provided_element_type(elements),
                            dst_binding,
                        )
                        .into(),
                        ..Default::default()
                    }));
                };

                if data.is_empty() {
                    return Err(Box::new(ValidationError {
                        context: "data".into(),
                        problem: "is empty".into(),
                        vuids: &[
                            "VUID-VkWriteDescriptorSetInlineUniformBlock-dataSize-arraylength",
                            "VUID-VkWriteDescriptorSet-descriptorCount-arraylength",
                        ],
                        ..Default::default()
                    }));
                }

                if data.len() % 4 != 0 {
                    return Err(Box::new(ValidationError {
                        context: "data".into(),
                        problem: "the length is not a multiple of 4".into(),
                        vuids: &[
                            "VUID-VkWriteDescriptorSetInlineUniformBlock-dataSize-02222",
                            "VUID-VkWriteDescriptorSet-descriptorType-02220",
                        ],
                        ..Default::default()
                    }));
                }

                if dst_array_element % 4 != 0 {
                    return Err(Box::new(ValidationError {
                        context: "offset".into(),
                        problem: "is not a multiple of 4".into(),
                        vuids: &["VUID-VkWriteDescriptorSet-descriptorType-02219"],
                        ..Default::default()
                    }));
                }
            }
            DescriptorType::AccelerationStructure => {
                let elements =
                    if let WriteDescriptorSetElements::AccelerationStructure(elements) = elements {
                        elements
                    } else {
                        return Err(Box::new(ValidationError {
                            context: "elements".into(),
                            problem: format!(
                                "contains `{}` elements, but descriptor set binding {} \
                                requires `acceleration_structure` elements",
                                provided_element_type(elements),
                                dst_binding,
                            )
                            .into(),
                            ..Default::default()
                        }));
                    };

                for (index, acceleration_structure) in elements.iter().enumerate() {
                    let Some(acceleration_structure) = acceleration_structure else {
                        if !device.enabled_features().null_descriptor {
                            return Err(Box::new(ValidationError {
                                context: format!("elements[{}]", index).into(),
                                problem: "is `None`, but the `null_descriptor` feature is not \
                                    enabled for the device"
                                    .into(),
                                vuids: &["VUID-VkWriteDescriptorSetAccelerationStructureKHR-\
                                    pAccelerationStructures-03580"],
                                ..Default::default()
                            }));
                        }

                        continue;
                    };

                    assert_eq!(device, acceleration_structure.device());

                    if !matches!(
                        acceleration_structure.ty(),
                        AccelerationStructureType::TopLevel | AccelerationStructureType::Generic
                    ) {
                        return Err(Box::new(ValidationError {
                            context: format!("elements[{}]", index).into(),
                            problem: "the acceleration structure's type is not \
                                `AccelerationStructureType::TopLevel` or \
                                `AccelerationStructureType::Generic`"
                                .into(),
                            vuids: &["VUID-VkWriteDescriptorSetAccelerationStructureKHR-\
                                pAccelerationStructures-03579"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if dst_array_element + array_element_count > max_descriptor_count {
            return Err(Box::new(ValidationError {
                problem: "`dst_array_element` + the number of provided elements is greater than \
                    the number of descriptors in the descriptor set binding"
                    .into(),
                vuids: &["VUID-VkWriteDescriptorSet-dstArrayElement-00321"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        dst_set: vk::DescriptorSet,
        descriptor_type: DescriptorType,
        fields1_vk: &'a WriteDescriptorSetFields1<'a>,
        extensions_vk: &'a mut WriteDescriptorSetExtensionsVk<'_>,
    ) -> vk::WriteDescriptorSet<'a> {
        let &Self {
            dst_binding,
            dst_array_element,
            elements: _,
        } = self;
        let WriteDescriptorSetFields1 {
            descriptor_infos_vk,
        } = fields1_vk;

        let mut val_vk = vk::WriteDescriptorSet::default()
            .dst_set(dst_set)
            .dst_binding(dst_binding)
            .dst_array_element(dst_array_element)
            .descriptor_type(descriptor_type.into());

        match descriptor_infos_vk {
            DescriptorInfosVk::Image(info) => val_vk = val_vk.image_info(info),
            DescriptorInfosVk::Buffer(info) => val_vk = val_vk.buffer_info(info),
            DescriptorInfosVk::BufferView(info) => val_vk = val_vk.texel_buffer_view(info),
            _ => (),
        }

        let WriteDescriptorSetExtensionsVk {
            descriptor_type_extension_vk,
        } = extensions_vk;

        if let Some(descriptor_type_extension_vk) = descriptor_type_extension_vk {
            match descriptor_type_extension_vk {
                DescriptorTypeExtensionVk::AccelerationStructure(next) => {
                    val_vk = val_vk
                        .descriptor_count(next.acceleration_structure_count)
                        .push(next)
                }
                DescriptorTypeExtensionVk::InlineUniformBlock(next) => {
                    val_vk = val_vk.descriptor_count(next.data_size).push(next)
                }
            }
        }

        debug_assert!(val_vk.descriptor_count != 0);
        val_vk
    }

    pub(crate) fn to_vk_extensions(
        &self,
        fields1_vk: &'a WriteDescriptorSetFields1<'a>,
    ) -> WriteDescriptorSetExtensionsVk<'a> {
        let WriteDescriptorSetFields1 {
            descriptor_infos_vk,
        } = fields1_vk;

        let descriptor_type_extension_vk = match descriptor_infos_vk {
            DescriptorInfosVk::Image(_)
            | DescriptorInfosVk::Buffer(_)
            | DescriptorInfosVk::BufferView(_) => None,
            DescriptorInfosVk::AccelerationStructure(info) => {
                Some(DescriptorTypeExtensionVk::AccelerationStructure(
                    vk::WriteDescriptorSetAccelerationStructureKHR::default()
                        .acceleration_structures(info),
                ))
            }
            DescriptorInfosVk::InlineUniformBlock(data) => {
                Some(DescriptorTypeExtensionVk::InlineUniformBlock(
                    vk::WriteDescriptorSetInlineUniformBlock::default().data(data),
                ))
            }
        };

        WriteDescriptorSetExtensionsVk {
            descriptor_type_extension_vk,
        }
    }

    pub(crate) fn to_vk_fields1(
        &self,
        default_image_layout: ImageLayout,
    ) -> WriteDescriptorSetFields1<'a> {
        let descriptor_infos_vk = match self.elements {
            WriteDescriptorSetElements::Image(elements) => DescriptorInfosVk::Image(
                elements
                    .iter()
                    .map(|element| element.to_vk(default_image_layout))
                    .collect(),
            ),
            WriteDescriptorSetElements::Buffer(elements) => DescriptorInfosVk::Buffer(
                elements.iter().map(DescriptorBufferInfo::to_vk).collect(),
            ),
            WriteDescriptorSetElements::BufferView(elements) => DescriptorInfosVk::BufferView(
                elements
                    .iter()
                    .map(|element| {
                        element
                            .as_ref()
                            .map_or(vk::BufferView::null(), VulkanObject::handle)
                    })
                    .collect(),
            ),
            WriteDescriptorSetElements::InlineUniformBlock(data) => {
                DescriptorInfosVk::InlineUniformBlock(data)
            }
            WriteDescriptorSetElements::AccelerationStructure(elements) => {
                DescriptorInfosVk::AccelerationStructure(
                    elements
                        .iter()
                        .map(|element| {
                            element
                                .as_ref()
                                .map_or(vk::AccelerationStructureKHR::null(), VulkanObject::handle)
                        })
                        .collect(),
                )
            }
        };

        WriteDescriptorSetFields1 {
            descriptor_infos_vk,
        }
    }

    pub(crate) fn to_owned(&self) -> OwnedWriteDescriptorSet {
        OwnedWriteDescriptorSet {
            dst_binding: self.dst_binding,
            dst_array_element: self.dst_array_element,
            elements: self.elements.to_owned(),
        }
    }
}

pub(crate) struct WriteDescriptorSetExtensionsVk<'a> {
    pub(crate) descriptor_type_extension_vk: Option<DescriptorTypeExtensionVk<'a>>,
}

pub(crate) enum DescriptorTypeExtensionVk<'a> {
    AccelerationStructure(vk::WriteDescriptorSetAccelerationStructureKHR<'a>),
    InlineUniformBlock(vk::WriteDescriptorSetInlineUniformBlock<'a>),
}

pub(crate) struct WriteDescriptorSetFields1<'a> {
    pub(crate) descriptor_infos_vk: DescriptorInfosVk<'a>,
}

pub(crate) enum DescriptorInfosVk<'a> {
    Image(SmallVec<[vk::DescriptorImageInfo; 1]>),
    Buffer(SmallVec<[vk::DescriptorBufferInfo; 1]>),
    BufferView(SmallVec<[vk::BufferView; 1]>),
    InlineUniformBlock(&'a [u8]),
    AccelerationStructure(SmallVec<[vk::AccelerationStructureKHR; 1]>),
}

pub(crate) struct OwnedWriteDescriptorSet {
    pub(crate) dst_binding: u32,
    pub(crate) dst_array_element: u32,
    pub(crate) elements: OwnedWriteDescriptorSetElements,
}

/// The elements held by a `WriteDescriptorSet`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum WriteDescriptorSetElements<'a> {
    Image(&'a [DescriptorImageInfo<'a>]),
    Buffer(&'a [DescriptorBufferInfo<'a>]),
    BufferView(&'a [Option<&'a Arc<BufferView>>]),
    InlineUniformBlock(&'a [u8]),
    AccelerationStructure(&'a [Option<&'a Arc<AccelerationStructure>>]),
}

impl WriteDescriptorSetElements<'_> {
    /// Returns the number of elements.
    #[inline]
    pub const fn len(&self) -> u32 {
        match self {
            Self::Image(elements) => elements.len() as u32,
            Self::Buffer(elements) => elements.len() as u32,
            Self::BufferView(elements) => elements.len() as u32,
            Self::InlineUniformBlock(data) => data.len() as u32,
            Self::AccelerationStructure(elements) => elements.len() as u32,
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_owned(&self) -> OwnedWriteDescriptorSetElements {
        match *self {
            WriteDescriptorSetElements::Image(elements) => OwnedWriteDescriptorSetElements::Image(
                elements.iter().map(DescriptorImageInfo::to_owned).collect(),
            ),
            WriteDescriptorSetElements::Buffer(elements) => {
                OwnedWriteDescriptorSetElements::Buffer(
                    elements
                        .iter()
                        .map(DescriptorBufferInfo::to_owned)
                        .collect(),
                )
            }
            WriteDescriptorSetElements::BufferView(elements) => {
                OwnedWriteDescriptorSetElements::BufferView(
                    elements.iter().map(|element| element.cloned()).collect(),
                )
            }
            WriteDescriptorSetElements::InlineUniformBlock(data) => {
                OwnedWriteDescriptorSetElements::InlineUniformBlock(data.into())
            }
            WriteDescriptorSetElements::AccelerationStructure(elements) => {
                OwnedWriteDescriptorSetElements::AccelerationStructure(
                    elements.iter().map(|element| element.cloned()).collect(),
                )
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum OwnedWriteDescriptorSetElements {
    Image(SmallVec<[OwnedDescriptorImageInfo; 1]>),
    Buffer(SmallVec<[OwnedDescriptorBufferInfo; 1]>),
    BufferView(SmallVec<[Option<Arc<BufferView>>; 1]>),
    InlineUniformBlock(Vec<u8>),
    AccelerationStructure(SmallVec<[Option<Arc<AccelerationStructure>>; 1]>),
}

/// Parameters to write a buffer reference to a descriptor.
#[derive(Clone, Debug)]
pub struct DescriptorBufferInfo<'a> {
    /// The buffer to write to the descriptor.
    ///
    /// If set to `None`, the [`null_descriptor`] feature must be enabled on the device, `offset`
    /// must be `0`, and `range` must be `None`.
    ///
    /// The default value is `None`.
    ///
    /// [`null_descriptor`]: crate::device::DeviceFeatures::null_descriptor
    pub buffer: Option<&'a Arc<Buffer>>,

    /// The byte offset from `buffer` that will be made available to the shader. Must not be
    /// outside of the range of `buffer`.
    ///
    /// For dynamic buffer bindings, `offset` specifies the offset that is to be bound if the
    /// dynamic offset were zero. When binding the descriptor set, the effective value of `offset`
    /// shifts forward by the offset that was provided. For example, if `offset` is specified as
    /// `8` when writing the descriptor set, and then when binding the descriptor set the offset
    /// `16` is used, then the offset from `buffer` that will actually be bound is `24`.
    ///
    /// The default value is `0`.
    pub offset: DeviceSize,

    /// The byte size that will be made available to the shader.
    ///
    /// If set to `None`, the size until the end of the buffer will be made available.
    ///
    /// The default value is `None`.
    pub range: Option<DeviceSize>,
}

impl Default for DescriptorBufferInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DescriptorBufferInfo<'_> {
    /// Returns a default `DescriptorBufferInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            buffer: None,
            offset: 0,
            range: None,
        }
    }

    pub(crate) fn to_vk(&self) -> vk::DescriptorBufferInfo {
        let &Self {
            buffer,
            offset,
            range,
        } = self;

        vk::DescriptorBufferInfo {
            buffer: buffer.map_or(vk::Buffer::null(), VulkanObject::handle),
            offset,
            range: range.unwrap_or(vk::WHOLE_SIZE),
        }
    }

    pub(crate) fn to_owned(&self) -> OwnedDescriptorBufferInfo {
        OwnedDescriptorBufferInfo {
            buffer: self.buffer.cloned(),
            offset: self.offset,
            range: self.range,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OwnedDescriptorBufferInfo {
    pub(crate) buffer: Option<Arc<Buffer>>,
    pub(crate) offset: DeviceSize,
    pub(crate) range: Option<DeviceSize>,
}

impl OwnedDescriptorBufferInfo {
    pub(crate) fn as_ref(&self) -> DescriptorBufferInfo<'_> {
        DescriptorBufferInfo {
            buffer: self.buffer.as_ref(),
            offset: self.offset,
            range: self.range,
        }
    }
}

/// Parameters to write an image reference to a descriptor.
#[derive(Clone, Debug)]
pub struct DescriptorImageInfo<'a> {
    /// The sampler to write to the descriptor.
    ///
    /// If the descriptor type is `Sampler` or `CombinedImageSampler` and immutable samplers aren't
    /// used, this must be `Some`. It must be `None` if immutable samplers are used or if the
    /// descriptor type is `SampledImage` or `InputAttachment`.
    ///
    /// The default value is `None`.
    pub sampler: Option<&'a Arc<Sampler>>,

    /// The image view to write to the descriptor.
    ///
    /// If the descriptor type is `Sampler`, this must be `None`. It must be `Some` if the
    /// descriptor type is `CombinedImageSampler`, `SampledImage` or `StorageImage`, otherwise the
    /// [`null_descriptor`] feature must be enabled on the device. It must be `Some` if the
    /// descriptor type is `InputAttachment` even if the `null_descriptor` feature is enabled.
    ///
    /// The default value is `None`.
    ///
    /// [`null_descriptor`]: crate::device::DeviceFeatures::null_descriptor
    pub image_view: Option<&'a Arc<ImageView>>,

    /// The layout that the image is expected to be in when it's accessed in the shader.
    ///
    /// Only certain layouts are allowed, depending on the type of descriptor.
    ///
    /// For `SampledImage`, `CombinedImageSampler` and `InputAttachment`:
    /// - `General`
    /// - `ShaderReadOnlyOptimal`
    /// - `DepthStencilReadOnlyOptimal`
    /// - `DepthReadOnlyStencilAttachmentOptimal`
    /// - `DepthAttachmentStencilReadOnlyOptimal`
    ///
    /// For `StorageImage`:
    /// - `General`
    ///
    /// If the `Undefined` layout is provided, then it will be automatically replaced with
    /// `General` for `StorageImage` descriptors, and with `ShaderReadOnlyOptimal` for any other
    /// descriptor type.
    ///
    /// The default value is `ImageLayout::Undefined`.
    pub image_layout: ImageLayout,
}

impl Default for DescriptorImageInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DescriptorImageInfo<'_> {
    /// Returns a default `DescriptorImageInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            sampler: None,
            image_view: None,
            image_layout: ImageLayout::Undefined,
        }
    }

    pub(crate) fn to_vk(&self, default_image_layout: ImageLayout) -> vk::DescriptorImageInfo {
        let &Self {
            image_view,
            sampler,
            image_layout,
        } = self;

        vk::DescriptorImageInfo {
            sampler: sampler.map_or(vk::Sampler::null(), VulkanObject::handle),
            image_view: image_view.map_or(vk::ImageView::null(), VulkanObject::handle),
            image_layout: if image_layout == ImageLayout::Undefined {
                default_image_layout.into()
            } else {
                image_layout.into()
            },
        }
    }

    pub(crate) fn to_owned(&self) -> OwnedDescriptorImageInfo {
        OwnedDescriptorImageInfo {
            sampler: self.sampler.cloned(),
            image_view: self.image_view.cloned(),
            image_layout: self.image_layout,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OwnedDescriptorImageInfo {
    pub(crate) sampler: Option<Arc<Sampler>>,
    pub(crate) image_view: Option<Arc<ImageView>>,
    pub(crate) image_layout: ImageLayout,
}

impl OwnedDescriptorImageInfo {
    pub(crate) fn as_ref(&self) -> DescriptorImageInfo<'_> {
        DescriptorImageInfo {
            sampler: self.sampler.as_ref(),
            image_view: self.image_view.as_ref(),
            image_layout: self.image_layout,
        }
    }
}

/// Represents a single copy operation to the binding of a descriptor set.
#[derive(Clone)]
pub struct CopyDescriptorSet<'a> {
    /// The source descriptor set to copy from.
    ///
    /// There is no default value.
    pub src_set: &'a Arc<DescriptorSet>,

    /// The binding number in the source descriptor set to copy from.
    ///
    /// The default value is 0.
    pub src_binding: u32,

    /// The first array element in the source descriptor set to copy from.
    ///
    /// The default value is 0.
    pub src_array_element: u32,

    /// The binding number in the destination descriptor set to copy into.
    ///
    /// The default value is 0.
    pub dst_binding: u32,

    /// The first array element in the destination descriptor set to copy into.
    ///
    /// The default value is 0.
    pub dst_array_element: u32,

    /// The number of descriptors (array elements) to copy.
    ///
    /// The default value is 1.
    pub descriptor_count: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> CopyDescriptorSet<'a> {
    /// Returns a default `CopyDescriptorSet` with the provided `src_set`.
    #[inline]
    pub const fn new(src_set: &'a Arc<DescriptorSet>) -> Self {
        Self {
            src_set,
            src_binding: 0,
            src_array_element: 0,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, dst_set: &RawDescriptorSet) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_set,
            src_binding,
            src_array_element: src_first_array_element,
            dst_binding,
            dst_array_element: dst_first_array_element,
            descriptor_count,
            _ne,
        } = self;

        // VUID-VkCopyDescriptorSet-commonparent
        assert_eq!(src_set.device(), dst_set.device());

        match (
            src_set
                .layout()
                .flags()
                .intersects(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
            dst_set
                .layout()
                .flags()
                .intersects(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "`src_set.layout().flags()` contains \
                        `DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL`, but \
                        `dst_set.layout().flags()` does not also contain it"
                        .into(),
                    vuids: &["VUID-VkCopyDescriptorSet-srcSet-01918"],
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "`src_set.layout().flags()` does not contain \
                        `DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL`, but \
                        `dst_set.layout().flags()` does contain it"
                        .into(),
                    vuids: &["VUID-VkCopyDescriptorSet-srcSet-04885"],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        match (
            src_set
                .pool()
                .flags()
                .intersects(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
            dst_set
                .pool()
                .flags()
                .intersects(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
        ) {
            (true, false) => {
                return Err(Box::new(ValidationError {
                    problem: "`src_set.pool().flags()` contains \
                        `DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`, but \
                        `dst_set.pool().flags()` does not also contain it"
                        .into(),
                    vuids: &["VUID-VkCopyDescriptorSet-srcSet-01920"],
                    ..Default::default()
                }));
            }
            (false, true) => {
                return Err(Box::new(ValidationError {
                    problem: "`src_set.pool().flags()` does not contain \
                        `DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`, but \
                        `dst_set.pool().flags()` does contain it"
                        .into(),
                    vuids: &["VUID-VkCopyDescriptorSet-srcSet-04887"],
                    ..Default::default()
                }));
            }
            _ => (),
        }

        let Some(src_layout_binding) = src_set.layout().binding(src_binding) else {
            return Err(Box::new(ValidationError {
                problem: "`src_binding` does not exist in the descriptor set layout of `src_set`"
                    .into(),
                vuids: &["VUID-VkCopyDescriptorSet-srcBinding-00345"],
                ..Default::default()
            }));
        };

        let src_max_descriptor_count = if src_layout_binding
            .binding_flags
            .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
        {
            src_set.variable_descriptor_count()
        } else {
            src_layout_binding.descriptor_count
        };

        if src_first_array_element + descriptor_count > src_max_descriptor_count {
            return Err(Box::new(ValidationError {
                problem: "`src_first_array_element` + `descriptor_count` is greater than \
                    the number of descriptors in `src_set`'s descriptor set binding"
                    .into(),
                vuids: &["VUID-VkCopyDescriptorSet-srcArrayElement-00346"],
                ..Default::default()
            }));
        }

        let Some(dst_layout_binding) = dst_set.layout().binding(dst_binding) else {
            return Err(Box::new(ValidationError {
                problem: "`dst_binding` does not exist in the descriptor set layout of `dst_set`"
                    .into(),
                vuids: &["VUID-VkCopyDescriptorSet-dstBinding-00347"],
                ..Default::default()
            }));
        };

        let dst_max_descriptor_count = if dst_layout_binding
            .binding_flags
            .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
        {
            dst_set.variable_descriptor_count()
        } else {
            dst_layout_binding.descriptor_count
        };

        if dst_first_array_element + descriptor_count > dst_max_descriptor_count {
            return Err(Box::new(ValidationError {
                problem: "`dst_first_array_element` + `descriptor_count` is greater than \
                    the number of descriptors in `dst_set`'s descriptor set binding"
                    .into(),
                vuids: &["VUID-VkCopyDescriptorSet-dstArrayElement-00348"],
                ..Default::default()
            }));
        }

        if src_layout_binding.descriptor_type != dst_layout_binding.descriptor_type {
            return Err(Box::new(ValidationError {
                problem: "the descriptor type of `src_binding` within `src_set` does not equal \
                    the descriptor type of `dst_binding` within `dst_set`"
                    .into(),
                vuids: &["VUID-VkCopyDescriptorSet-dstBinding-02632"],
                ..Default::default()
            }));
        }

        if dst_layout_binding.descriptor_type == DescriptorType::Sampler
            && !dst_layout_binding.immutable_samplers.is_empty()
        {
            return Err(Box::new(ValidationError {
                problem: "the descriptor type of `dst_binding` within `dst_set` is \
                    `DescriptorType::Sampler`, and the layout was created with immutable samplers \
                    for `dst_binding`"
                    .into(),
                vuids: &["VUID-VkCopyDescriptorSet-dstBinding-02753"],
                ..Default::default()
            }));
        }

        if dst_layout_binding.descriptor_type == DescriptorType::InlineUniformBlock {
            if src_first_array_element % 4 != 0 {
                return Err(Box::new(ValidationError {
                    problem: "the descriptor type of `src_binding` within `src_set` is \
                        `DescriptorType::InlineUniformBlock`, and `src_first_array_element` is \
                        not a multiple of 4"
                        .into(),
                    vuids: &["VUID-VkCopyDescriptorSet-srcBinding-02223"],
                    ..Default::default()
                }));
            }

            if dst_first_array_element % 4 != 0 {
                return Err(Box::new(ValidationError {
                    problem: "the descriptor type of `dst_binding` within `dst_set` is \
                        `DescriptorType::InlineUniformBlock`, and `dst_first_array_element` is \
                        not a multiple of 4"
                        .into(),
                    vuids: &["VUID-VkCopyDescriptorSet-dstBinding-02224"],
                    ..Default::default()
                }));
            }

            if descriptor_count % 4 != 0 {
                return Err(Box::new(ValidationError {
                    problem: "the descriptor type of `dst_binding` within `dst_set` is \
                        `DescriptorType::InlineUniformBlock`, and `descriptor_count` is \
                        not a multiple of 4"
                        .into(),
                    vuids: &["VUID-VkCopyDescriptorSet-srcBinding-02225"],
                    ..Default::default()
                }));
            }
        }

        // VUID-VkCopyDescriptorSet-srcSet-00349
        // Ensured as long as copies can only occur during descriptor set construction.

        Ok(())
    }

    pub(crate) fn to_vk(&self, dst_set: vk::DescriptorSet) -> vk::CopyDescriptorSet<'static> {
        let &Self {
            src_set,
            src_binding,
            src_array_element: src_first_array_element,
            dst_binding,
            dst_array_element: dst_first_array_element,
            descriptor_count,
            _ne: _,
        } = self;

        vk::CopyDescriptorSet::default()
            .src_set(src_set.handle())
            .src_binding(src_binding)
            .src_array_element(src_first_array_element)
            .dst_set(dst_set)
            .dst_binding(dst_binding)
            .dst_array_element(dst_first_array_element)
            .descriptor_count(descriptor_count)
    }
}

/// Invalidates descriptors within a descriptor set. Doesn't actually call into Vulkan and only
/// invalidates the descriptors inside vulkano's resource tracking. Invalidated descriptors are
/// equivalent to uninitialized descriptors, in that binding a descriptor set to a particular
/// pipeline requires all shader-accessible descriptors to be valid.
///
/// The intended use-case is an update-after-bind or bindless system, where entries in an arrayed
/// binding have to be invalidated so that the backing resource will be freed, and not stay forever
/// referenced until overridden by some update.
pub struct InvalidateDescriptorSet {
    /// The binding number in the descriptor set to invalidate.
    ///
    /// The default value is 0.
    pub binding: u32,

    /// The first array element in the descriptor set to invalidate.
    ///
    /// The default value is 0.
    pub first_array_element: u32,

    /// The number of descriptors (array elements) to invalidate.
    ///
    /// The default value is 1.
    pub descriptor_count: u32,

    pub _ne: crate::NonExhaustive<'static>,
}

impl InvalidateDescriptorSet {
    pub fn invalidate(binding: u32) -> Self {
        Self::invalidate_array(binding, 0, 1)
    }

    pub fn invalidate_array(binding: u32, first_array_element: u32, descriptor_count: u32) -> Self {
        Self {
            binding,
            first_array_element,
            descriptor_count,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(
        &self,
        layout: &DescriptorSetLayout,
        variable_descriptor_count: u32,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            binding,
            first_array_element,
            descriptor_count,
            ..
        } = self;

        let Some(layout_binding) = layout.binding(binding) else {
            return Err(Box::new(ValidationError {
                context: "binding".into(),
                problem: "does not exist in the descriptor set layout".into(),
                vuids: &["VUID-VkWriteDescriptorSet-dstBinding-00315"],
                ..Default::default()
            }));
        };

        debug_assert!(descriptor_count != 0);
        let max_descriptor_count = if layout_binding
            .binding_flags
            .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
        {
            variable_descriptor_count
        } else {
            layout_binding.descriptor_count
        };

        // VUID-VkWriteDescriptorSet-dstArrayElement-00321
        if first_array_element + descriptor_count > max_descriptor_count {
            return Err(Box::new(ValidationError {
                problem: "`first_array_element` + the number of provided elements is greater than \
                    the number of descriptors in the descriptor set binding"
                    .into(),
                vuids: &["VUID-VkWriteDescriptorSet-dstArrayElement-00321"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}
