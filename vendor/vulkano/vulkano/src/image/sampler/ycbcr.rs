//! Conversion from sampled YCbCr image data to RGB shader data.
//!
//! A sampler YCbCr conversion is an object that assists a sampler when converting from YCbCr
//! formats and/or YCbCr texel input data. It is used to read frames of video data within a shader,
//! possibly to apply it as texture on a rendered primitive. Sampler YCbCr conversion can only be
//! used with certain formats, and conversely, some formats require the use of a sampler YCbCr
//! conversion to be sampled at all.
//!
//! A sampler YCbCr conversion can only be used with a combined image sampler descriptor in a
//! descriptor set. The conversion must be attached on both the image view and sampler in the
//! descriptor, and the sampler must be included in the descriptor set layout as an immutable
//! sampler.
//!
//! # Examples
//!
//! ```
//! use vulkano::{
//!     descriptor_set::{
//!         layout::{
//!             DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
//!             DescriptorType,
//!         },
//!         DescriptorImageInfo, DescriptorSet, WriteDescriptorSet,
//!     },
//!     format::Format,
//!     image::{
//!         sampler::{
//!             ycbcr::{
//!                 SamplerYcbcrConversion, SamplerYcbcrConversionCreateInfo,
//!                 SamplerYcbcrModelConversion,
//!             },
//!             Sampler, SamplerCreateInfo,
//!         },
//!         view::{ImageView, ImageViewCreateInfo},
//!         Image, ImageCreateFlags, ImageCreateInfo, ImageType, ImageUsage,
//!     },
//!     memory::allocator::AllocationCreateInfo,
//!     shader::ShaderStage,
//! };
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
//! # let descriptor_set_allocator: std::sync::Arc<vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator> = return;
//! #
//! let conversion = SamplerYcbcrConversion::new(
//!     &device,
//!     &SamplerYcbcrConversionCreateInfo {
//!         format: Format::G8_B8_R8_3PLANE_420_UNORM,
//!         ycbcr_model: SamplerYcbcrModelConversion::YcbcrIdentity,
//!         ..Default::default()
//!     },
//! )
//! .unwrap();
//!
//! let sampler = Sampler::new(
//!     &device,
//!     &SamplerCreateInfo {
//!         sampler_ycbcr_conversion: Some(&conversion),
//!         ..Default::default()
//!     },
//! )
//! .unwrap();
//!
//! let descriptor_set_layout = DescriptorSetLayout::new(
//!     &device,
//!     &DescriptorSetLayoutCreateInfo {
//!         bindings: &[DescriptorSetLayoutBinding {
//!             binding: 0,
//!             stages: ShaderStage::Fragment.into(),
//!             immutable_samplers: &[&sampler],
//!             ..DescriptorSetLayoutBinding::new(DescriptorType::CombinedImageSampler)
//!         }],
//!         ..Default::default()
//!     },
//! )
//! .unwrap();
//!
//! let image = Image::new(
//!     &memory_allocator,
//!     &ImageCreateInfo {
//!         image_type: ImageType::Dim2d,
//!         format: Format::G8_B8_R8_3PLANE_420_UNORM,
//!         extent: [1920, 1080, 1],
//!         usage: ImageUsage::SAMPLED,
//!         ..Default::default()
//!     },
//!     &AllocationCreateInfo::default(),
//! )
//! .unwrap();
//!
//! let image_view = ImageView::new(
//!     &image,
//!     &ImageViewCreateInfo {
//!         sampler_ycbcr_conversion: Some(&conversion),
//!         ..ImageViewCreateInfo::from_image(&image)
//!     },
//! )
//! .unwrap();
//!
//! let descriptor_set = DescriptorSet::new(
//!     &descriptor_set_allocator,
//!     &descriptor_set_layout,
//!     &[WriteDescriptorSet::image(
//!         0,
//!         &DescriptorImageInfo {
//!             image_view: Some(&image_view),
//!             ..Default::default()
//!         },
//!     )],
//!     &[],
//! )
//! .unwrap();
//! ```

use crate::{
    device::{Device, DeviceOwned},
    format::{ChromaSampling, Format, FormatFeatures, NumericFormat},
    image::sampler::{ComponentMapping, ComponentSwizzle, Filter},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_enum},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ash::vk;
use std::{mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// Describes how sampled image data should converted from a YCbCr representation to an RGB one.
#[derive(Debug)]
pub struct SamplerYcbcrConversion {
    handle: vk::SamplerYcbcrConversion,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    format: Format,
    ycbcr_model: SamplerYcbcrModelConversion,
    ycbcr_range: SamplerYcbcrRange,
    component_mapping: ComponentMapping,
    chroma_offset: [ChromaLocation; 2],
    chroma_filter: Filter,
    force_explicit_reconstruction: bool,
}

impl SamplerYcbcrConversion {
    /// Creates a new `SamplerYcbcrConversion`, panicking on a validation error.
    ///
    /// The [`sampler_ycbcr_conversion`] feature must be enabled on the device.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`sampler_ycbcr_conversion`]: crate::device::DeviceFeatures::sampler_ycbcr_conversion
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        create_info: &SamplerYcbcrConversionCreateInfo<'_>,
    ) -> Result<Arc<SamplerYcbcrConversion>, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `SamplerYcbcrConversion`.
    ///
    /// The [`sampler_ycbcr_conversion`] feature must be enabled on the device.
    ///
    /// [`sampler_ycbcr_conversion`]: crate::device::DeviceFeatures::sampler_ycbcr_conversion
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &SamplerYcbcrConversionCreateInfo<'_>,
    ) -> Result<Arc<SamplerYcbcrConversion>, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &SamplerYcbcrConversionCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !device.enabled_features().sampler_ycbcr_conversion {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "sampler_ycbcr_conversion",
                )])]),
                vuids: &["VUID-vkCreateSamplerYcbcrConversion-None-01648"],
                ..Default::default()
            }));
        }

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &SamplerYcbcrConversionCreateInfo<'_>,
    ) -> Result<Arc<SamplerYcbcrConversion>, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let handle = {
            let fns = device.fns();
            let create_sampler_ycbcr_conversion = if device.api_version() >= Version::V1_1 {
                fns.v1_1.create_sampler_ycbcr_conversion
            } else {
                fns.khr_sampler_ycbcr_conversion
                    .create_sampler_ycbcr_conversion_khr
            };

            let mut output = MaybeUninit::uninit();
            unsafe {
                create_sampler_ycbcr_conversion(
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

    /// Creates a new `SamplerYcbcrConversion` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::SamplerYcbcrConversion,
        create_info: &SamplerYcbcrConversionCreateInfo<'_>,
    ) -> Arc<SamplerYcbcrConversion> {
        let &SamplerYcbcrConversionCreateInfo {
            format,
            ycbcr_model,
            ycbcr_range,
            component_mapping,
            chroma_offset,
            chroma_filter,
            force_explicit_reconstruction,
            _ne: _,
        } = create_info;

        Arc::new(SamplerYcbcrConversion {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),
            format,
            ycbcr_model,
            ycbcr_range,
            component_mapping,
            chroma_offset,
            chroma_filter,
            force_explicit_reconstruction,
        })
    }

    /// Returns the chroma filter used by the conversion.
    #[inline]
    pub fn chroma_filter(&self) -> Filter {
        self.chroma_filter
    }

    /// Returns the chroma offsets used by the conversion.
    #[inline]
    pub fn chroma_offset(&self) -> [ChromaLocation; 2] {
        self.chroma_offset
    }

    /// Returns the component mapping of the conversion.
    #[inline]
    pub fn component_mapping(&self) -> ComponentMapping {
        self.component_mapping
    }

    /// Returns whether the conversion has forced explicit reconstruction to be enabled.
    #[inline]
    pub fn force_explicit_reconstruction(&self) -> bool {
        self.force_explicit_reconstruction
    }

    /// Returns the format that the conversion was created for.
    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the YCbCr model of the conversion.
    #[inline]
    pub fn ycbcr_model(&self) -> SamplerYcbcrModelConversion {
        self.ycbcr_model
    }

    /// Returns the YCbCr range of the conversion.
    #[inline]
    pub fn ycbcr_range(&self) -> SamplerYcbcrRange {
        self.ycbcr_range
    }

    /// Returns whether `self` is equal or identically defined to `other`.
    #[inline]
    pub fn is_identical(&self, other: &SamplerYcbcrConversion) -> bool {
        self.handle == other.handle || {
            let &Self {
                handle: _,
                device: _,
                id: _,
                format,
                ycbcr_model,
                ycbcr_range,
                component_mapping,
                chroma_offset,
                chroma_filter,
                force_explicit_reconstruction,
            } = self;

            format == other.format
                && ycbcr_model == other.ycbcr_model
                && ycbcr_range == other.ycbcr_range
                && component_mapping == other.component_mapping
                && chroma_offset == other.chroma_offset
                && chroma_filter == other.chroma_filter
                && force_explicit_reconstruction == other.force_explicit_reconstruction
        }
    }
}

impl Drop for SamplerYcbcrConversion {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        let destroy_sampler_ycbcr_conversion = if self.device.api_version() >= Version::V1_1 {
            fns.v1_1.destroy_sampler_ycbcr_conversion
        } else {
            fns.khr_sampler_ycbcr_conversion
                .destroy_sampler_ycbcr_conversion_khr
        };

        unsafe { destroy_sampler_ycbcr_conversion(self.device.handle(), self.handle, ptr::null()) };
    }
}

unsafe impl VulkanObject for SamplerYcbcrConversion {
    type Handle = vk::SamplerYcbcrConversion;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for SamplerYcbcrConversion {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(SamplerYcbcrConversion);

/// Parameters to create a new `SamplerYcbcrConversion`.
#[derive(Clone, Debug)]
pub struct SamplerYcbcrConversionCreateInfo<'a> {
    /// The image view format that this conversion will read data from. The conversion cannot be
    /// used with image views of any other format.
    ///
    /// The format must support YCbCr conversions, meaning that its `FormatFeatures` must support
    /// at least one of `cosited_chroma_samples` or `midpoint_chroma_samples`.
    ///
    /// If this is set to a format that has chroma subsampling (contains `422` or `420` in the
    /// name) then `component_mapping` is restricted as follows:
    /// - `g` must be identity swizzled.
    /// - `a` must be identity swizzled or `Zero` or `One`.
    /// - `r` and `b` must be identity swizzled or mapped to each other.
    ///
    /// Compatibility notice: currently, this value must be `Some`, but future additions may allow
    /// `None` as a valid value as well.
    ///
    /// The default value is `Format::UNDEFINED`.
    pub format: Format,

    /// The conversion between the input color model and the output RGB color model.
    ///
    /// If this is not set to `RgbIdentity`, then the `r`, `g` and `b` components of
    /// `component_mapping` must not be `Zero` or `One`, and the component being read must exist in
    /// `format` (must be represented as a nonzero number of bits).
    ///
    /// The default value is [`RgbIdentity`](SamplerYcbcrModelConversion::RgbIdentity).
    pub ycbcr_model: SamplerYcbcrModelConversion,

    /// If `ycbcr_model` is not `RgbIdentity`, specifies the range expansion of the input values
    /// that should be used.
    ///
    /// If this is set to `ItuNarrow`, then the `r`, `g` and `b` components of `component_mapping`
    /// must each map to a component of `format` that is represented with at least 8 bits.
    ///
    /// The default value is [`ItuFull`](SamplerYcbcrRange::ItuFull).
    pub ycbcr_range: SamplerYcbcrRange,

    /// The mapping to apply to the components of the input format, before color model conversion
    /// and range expansion.
    ///
    /// The default value is [`ComponentMapping::identity()`].
    pub component_mapping: ComponentMapping,

    /// For formats with chroma subsampling and a `Linear` filter, specifies the sampled location
    /// for the subsampled components, in the x and y direction.
    ///
    /// The value is ignored if the filter is `Nearest` or the corresponding axis is not chroma
    /// subsampled. If the value is not ignored, the format must support the chosen mode.
    ///
    /// The default value is [`CositedEven`](ChromaLocation::CositedEven) for both axes.
    pub chroma_offset: [ChromaLocation; 2],

    /// For formats with chroma subsampling, specifies the filter used for reconstructing the
    /// chroma components to full resolution.
    ///
    /// The `Cubic` filter is not supported. If `Linear` is used, the format must support it.
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    pub chroma_filter: Filter,

    /// Forces explicit reconstruction if the implementation does not use it by default. The format
    /// must support it. See
    /// [the spec](https://registry.khronos.org/vulkan/specs/1.2-extensions/html/chap16.html#textures-chroma-reconstruction)
    /// for more information.
    ///
    /// The default value is `false`.
    pub force_explicit_reconstruction: bool,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SamplerYcbcrConversionCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SamplerYcbcrConversionCreateInfo<'_> {
    /// Returns a default `SamplerYcbcrConversionCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            format: Format::UNDEFINED,
            ycbcr_model: SamplerYcbcrModelConversion::RgbIdentity,
            ycbcr_range: SamplerYcbcrRange::ItuFull,
            component_mapping: ComponentMapping::identity(),
            chroma_offset: [ChromaLocation::CositedEven; 2],
            chroma_filter: Filter::Nearest,
            force_explicit_reconstruction: false,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            format,
            ycbcr_model,
            ycbcr_range,
            component_mapping,
            chroma_offset,
            chroma_filter,
            force_explicit_reconstruction,
            _ne: _,
        } = self;

        format.validate_device(device).map_err(|err| {
            err.add_context("format")
                .set_vuids(&["VUID-VkSamplerYcbcrConversionCreateInfo-format-parameter"])
        })?;

        ycbcr_model.validate_device(device).map_err(|err| {
            err.add_context("ycbcr_model")
                .set_vuids(&["VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrModel-parameter"])
        })?;

        ycbcr_range.validate_device(device).map_err(|err| {
            err.add_context("ycbcr_range")
                .set_vuids(&["VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrRange-parameter"])
        })?;

        component_mapping
            .validate(device)
            .map_err(|err| err.add_context("component_mapping"))?;

        for (index, offset) in chroma_offset.into_iter().enumerate() {
            offset.validate_device(device).map_err(|err| {
                err.add_context(format!("chroma_offset[{}]", index))
                    .set_vuids(&[
                        "VUID-VkSamplerYcbcrConversionCreateInfo-xChromaOffset-parameter",
                        "VUID-VkSamplerYcbcrConversionCreateInfo-yChromaOffset-parameter",
                    ])
            })?;
        }

        chroma_filter.validate_device(device).map_err(|err| {
            err.add_context("chroma_filter")
                .set_vuids(&["VUID-VkSamplerYcbcrConversionCreateInfo-chromaFilter-parameter"])
        })?;

        if !(format.numeric_format_color() == Some(NumericFormat::UNORM)) {
            return Err(Box::new(ValidationError {
                context: "format".into(),
                problem: "the numeric type is not `UNORM`".into(),
                vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-format-04061"],
                ..Default::default()
            }));
        }

        // Use unchecked, because all validation has been done above.
        let format_properties =
            unsafe { device.physical_device().format_properties_unchecked(format) };
        let potential_format_features = format_properties.potential_format_features();

        if !potential_format_features.intersects(
            FormatFeatures::MIDPOINT_CHROMA_SAMPLES | FormatFeatures::COSITED_CHROMA_SAMPLES,
        ) {
            return Err(Box::new(ValidationError {
                context: "format".into(),
                problem: "the potential format features do not contain \
                    `FormatFeatures::MIDPOINT_CHROMA_SAMPLES` or \
                    `FormatFeatures::COSITED_CHROMA_SAMPLES`"
                    .into(),
                vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-format-01650"],
                ..Default::default()
            }));
        }

        if let Some(chroma_sampling @ (ChromaSampling::Mode422 | ChromaSampling::Mode420)) =
            format.ycbcr_chroma_sampling()
        {
            match chroma_sampling {
                ChromaSampling::Mode420 => {
                    if chroma_offset.contains(&ChromaLocation::CositedEven)
                        && !potential_format_features
                            .intersects(FormatFeatures::COSITED_CHROMA_SAMPLES)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`format` has both horizontal and vertical chroma \
                                    subsampling, and \
                                    its potential format features do not \
                                    contain `FormatFeatures::COSITED_CHROMA_SAMPLES`, but \
                                    `chroma_offset[0]` or `chroma_offset[1]` are \
                                    `ChromaLocation::CositedEven`"
                                .into(),
                            vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-xChromaOffset-01651"],
                            ..Default::default()
                        }));
                    }

                    if chroma_offset.contains(&ChromaLocation::Midpoint)
                        && !potential_format_features
                            .intersects(FormatFeatures::MIDPOINT_CHROMA_SAMPLES)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`format` has both horizontal and vertical chroma \
                                    subsampling, and \
                                    its potential format features do not \
                                    contain `FormatFeatures::MIDPOINT_CHROMA_SAMPLES`, but \
                                    `chroma_offset[0]` or `chroma_offset[1]` are \
                                    `ChromaLocation::Midpoint`"
                                .into(),
                            vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-xChromaOffset-01652"],
                            ..Default::default()
                        }));
                    }
                }
                ChromaSampling::Mode422 => {
                    if chroma_offset[0] == ChromaLocation::CositedEven
                        && !potential_format_features
                            .intersects(FormatFeatures::COSITED_CHROMA_SAMPLES)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`format` has horizontal chroma subsampling, and \
                                    its potential format features do not \
                                    contain `FormatFeatures::COSITED_CHROMA_SAMPLES`, but \
                                    `chroma_offset[0]` is \
                                    `ChromaLocation::CositedEven`"
                                .into(),
                            vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-xChromaOffset-01651"],
                            ..Default::default()
                        }));
                    }

                    if chroma_offset[0] == ChromaLocation::Midpoint
                        && !potential_format_features
                            .intersects(FormatFeatures::MIDPOINT_CHROMA_SAMPLES)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`format` has horizontal chroma subsampling, and \
                                    its potential format features do not \
                                    contain `FormatFeatures::MIDPOINT_CHROMA_SAMPLES`, but \
                                    `chroma_offset[0]` is \
                                    `ChromaLocation::Midpoint`"
                                .into(),
                            vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-xChromaOffset-01652"],
                            ..Default::default()
                        }));
                    }
                }
                _ => unreachable!(),
            }

            if !component_mapping.g_is_identity() {
                return Err(Box::new(ValidationError {
                    problem: "`format` has chroma subsampling, but \
                        `component_mapping.g` is not identity swizzled"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02581"],
                    ..Default::default()
                }));
            }

            if !(component_mapping.a_is_identity()
                || matches!(
                    component_mapping.a,
                    ComponentSwizzle::One | ComponentSwizzle::Zero
                ))
            {
                return Err(Box::new(ValidationError {
                    context: "component_mapping.a".into(),
                    problem: "`format` has chroma subsampling, but \
                        `component_mapping.a` is not identity swizzled, or \
                        `ComponentSwizzle::One` or `ComponentSwizzle::Zero`"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02582"],
                    ..Default::default()
                }));
            }

            if !(component_mapping.r_is_identity()
                || matches!(component_mapping.r, ComponentSwizzle::Blue))
            {
                return Err(Box::new(ValidationError {
                    problem: "`format` has chroma subsampling, but \
                        `component_mapping.r` is not identity swizzled, or \
                        `ComponentSwizzle::Blue`"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02583"],
                    ..Default::default()
                }));
            }

            if !(component_mapping.b_is_identity()
                || matches!(component_mapping.b, ComponentSwizzle::Red))
            {
                return Err(Box::new(ValidationError {
                    problem: "`format` has chroma subsampling, but \
                        `component_mapping.b` is not identity swizzled, or \
                        `ComponentSwizzle::Red`"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02584"],
                    ..Default::default()
                }));
            }

            match (
                component_mapping.r_is_identity(),
                component_mapping.b_is_identity(),
            ) {
                (true, false) => {
                    return Err(Box::new(ValidationError {
                        problem: "`format` has chroma subsampling, and \
                            `component_mapping.r` is identity swizzled, but \
                            `component_mapping.b` is not identity swizzled"
                            .into(),
                        vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02585"],
                        ..Default::default()
                    }));
                }
                (false, true) => {
                    return Err(Box::new(ValidationError {
                        problem: "`format` has chroma subsampling, and \
                            `component_mapping.b` is identity swizzled, but \
                            `component_mapping.r` is not identity swizzled"
                            .into(),
                        vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02585"],
                        ..Default::default()
                    }));
                }
                _ => (),
            }
        }

        let components_bits = {
            let bits = format.components();
            component_mapping
                .component_map()
                .map(move |i| i.map(|i| bits[i]))
        };

        // VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrModel-01655
        if ycbcr_model != SamplerYcbcrModelConversion::RgbIdentity {
            if components_bits[0].is_none_or(|bits| bits == 0) {
                return Err(Box::new(ValidationError {
                    problem: "`ycbcr_model` is not `SamplerYcbcrModelConversion::RgbIdentity`, \
                        and `component_mapping.r` does not map to a component that exists in \
                        `format`"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02585"],
                    ..Default::default()
                }));
            }

            if components_bits[1].is_none_or(|bits| bits == 0) {
                return Err(Box::new(ValidationError {
                    problem: "`ycbcr_model` is not `SamplerYcbcrModelConversion::RgbIdentity`, \
                        and `component_mapping.g` does not map to a component that exists in \
                        `format`"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02585"],
                    ..Default::default()
                }));
            }

            if components_bits[2].is_none_or(|bits| bits == 0) {
                return Err(Box::new(ValidationError {
                    problem: "`ycbcr_model` is not `SamplerYcbcrModelConversion::RgbIdentity`, \
                        and `component_mapping.b` does not map to a component that exists in \
                        `format`"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02585"],
                    ..Default::default()
                }));
            }

            if components_bits[3].is_none_or(|bits| bits == 0) {
                return Err(Box::new(ValidationError {
                    problem: "`ycbcr_model` is not `SamplerYcbcrModelConversion::RgbIdentity`, \
                        and `component_mapping.a` does not map to a component that exists in \
                        `format`"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-components-02585"],
                    ..Default::default()
                }));
            }
        }

        if ycbcr_range == SamplerYcbcrRange::ItuNarrow {
            // TODO: Spec doesn't say how many bits `Zero` and `One` are considered to have, so
            // just skip them for now.

            if components_bits[0].is_some_and(|bits| bits < 8) {
                return Err(Box::new(ValidationError {
                    problem: "`ycbcr_range` is `SamplerYcbcrRange::ItuNarrow`, and \
                        `component_mapping.r` maps to a component in `format` with less than \
                        8 bits"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrRange-02748"],
                    ..Default::default()
                }));
            }

            if components_bits[1].is_some_and(|bits| bits < 8) {
                return Err(Box::new(ValidationError {
                    problem: "`ycbcr_range` is `SamplerYcbcrRange::ItuNarrow`, and \
                        `component_mapping.g` maps to a component in `format` with less than \
                        8 bits"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrRange-02748"],
                    ..Default::default()
                }));
            }

            if components_bits[2].is_some_and(|bits| bits < 8) {
                return Err(Box::new(ValidationError {
                    problem: "`ycbcr_range` is `SamplerYcbcrRange::ItuNarrow`, and \
                        `component_mapping.b` maps to a component in `format` with less than \
                        8 bits"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrRange-02748"],
                    ..Default::default()
                }));
            }

            if components_bits[3].is_some_and(|bits| bits < 8) {
                return Err(Box::new(ValidationError {
                    problem: "`ycbcr_range` is `SamplerYcbcrRange::ItuNarrow`, and \
                        `component_mapping.a` maps to a component in `format` with less than \
                        8 bits"
                        .into(),
                    vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-ycbcrRange-02748"],
                    ..Default::default()
                }));
            }
        }

        if force_explicit_reconstruction
            && !potential_format_features.intersects(FormatFeatures::
                SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE)
        {
            return Err(Box::new(ValidationError {
                problem: "`force_explicit_reconstruction` is `true`, but the \
                    potential format features of `format` do not include `FormatFeatures::\
                    SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE`"
                    .into(),
                vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-forceExplicitReconstruction-01656"],
                ..Default::default()
            }));
        }

        match chroma_filter {
            Filter::Nearest => (),
            Filter::Linear => {
                if !potential_format_features
                    .intersects(FormatFeatures::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER)
                {
                    return Err(Box::new(ValidationError {
                        problem: "`chroma_filter` is `Filter::Linear`, but the \
                            potential format features of `format` do not include `FormatFeatures::\
                            SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER`"
                            .into(),
                        vuids: &["VUID-VkSamplerYcbcrConversionCreateInfo-chromaFilter-01657"],
                        ..Default::default()
                    }));
                }
            }
            Filter::Cubic => {
                return Err(Box::new(ValidationError {
                    context: "chroma_filter".into(),
                    problem: "is `Filter::Cubic`".into(),
                    // vuids?
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::SamplerYcbcrConversionCreateInfo<'static> {
        let &Self {
            format,
            ycbcr_model,
            ycbcr_range,
            component_mapping,
            chroma_offset,
            chroma_filter,
            force_explicit_reconstruction,
            _ne: _,
        } = self;

        vk::SamplerYcbcrConversionCreateInfo::default()
            .format(format.into())
            .ycbcr_model(ycbcr_model.into())
            .ycbcr_range(ycbcr_range.into())
            .components(component_mapping.to_vk())
            .x_chroma_offset(chroma_offset[0].into())
            .y_chroma_offset(chroma_offset[1].into())
            .chroma_filter(chroma_filter.into())
            .force_explicit_reconstruction(force_explicit_reconstruction)
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The conversion between the color model of the source image and the color model of the shader.
    SamplerYcbcrModelConversion = SamplerYcbcrModelConversion(i32);

    /// The input values are already in the shader's model, and are passed through unmodified.
    RgbIdentity = RGB_IDENTITY,

    /// The input values are only range expanded, no other modifications are done.
    YcbcrIdentity = YCBCR_IDENTITY,

    /// The input values are converted according to the
    /// [ITU-R BT.709](https://en.wikipedia.org/wiki/Rec._709) standard.
    Ycbcr709 = YCBCR_709,

    /// The input values are converted according to the
    /// [ITU-R BT.601](https://en.wikipedia.org/wiki/Rec._601) standard.
    Ycbcr601 = YCBCR_601,

    /// The input values are converted according to the
    /// [ITU-R BT.2020](https://en.wikipedia.org/wiki/Rec._2020) standard.
    Ycbcr2020 = YCBCR_2020,
}

vulkan_enum! {
    #[non_exhaustive]

    /// How the numeric range of the input data is converted.
    SamplerYcbcrRange = SamplerYcbcrRange(i32);

    /// The input values cover the full numeric range, and are interpreted according to the ITU
    /// "full range" rules.
    ItuFull = ITU_FULL,

    /// The input values cover only a subset of the numeric range, with the remainder reserved as
    /// headroom/footroom. The values are interpreted according to the ITU "narrow range" rules.
    ItuNarrow = ITU_NARROW,
}

vulkan_enum! {
    #[non_exhaustive]

    /// For formats with chroma subsampling, the location where the chroma components are sampled,
    /// relative to the luma component.
    ChromaLocation = ChromaLocation(i32);

    /// The chroma components are sampled at the even luma coordinate.
    CositedEven = COSITED_EVEN,

    /// The chroma components are sampled at the midpoint between the even luma coordinate and
    /// the next higher odd luma coordinate.
    Midpoint = MIDPOINT,
}

#[cfg(test)]
mod tests {
    use super::SamplerYcbcrConversion;
    use crate::{Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError};

    #[test]
    fn feature_not_enabled() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = SamplerYcbcrConversion::try_new(&device, &Default::default());

        assert!(matches!(
            r,
            Err(Validated::ValidationError(err)) if matches!(
                *err,
                ValidationError {
                    requires_one_of: RequiresOneOf([RequiresAllOf([Requires::DeviceFeature(
                        "sampler_ycbcr_conversion",
                    )])]),
                    ..
                },
            ),
        ));
    }
}
