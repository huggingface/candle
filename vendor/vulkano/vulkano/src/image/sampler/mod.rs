//! How to retrieve data from a sampled image within a shader.
//!
//! When you retrieve data from a sampled image, you have to pass the coordinates of the pixel you
//! want to retrieve. The implementation then performs various calculations, and these operations
//! are what the `Sampler` object describes.
//!
//! # Level of detail
//!
//! The level-of-detail (LOD) is a floating-point value that expresses a sense of how much texture
//! detail is visible to the viewer. It is used in texture filtering and mipmapping calculations.
//!
//! LOD is calculated through one or more steps to produce a final value. The base LOD is
//! determined by one of two ways:
//! - Implicitly, by letting Vulkan calculate it automatically, based on factors such as number of
//!   pixels, distance and viewing angle. This is done using an `ImplicitLod` SPIR-V sampling
//!   operation, which corresponds to the `texture*` functions not suffixed with `Lod` in GLSL.
//! - Explicitly, specified in the shader. This is done using an `ExplicitLod` SPIR-V sampling
//!   operation, which corresponds to the `texture*Lod` functions in GLSL.
//!
//! It is possible to provide a *bias* to the base LOD value, which is simply added to it.
//! An LOD bias can be provided both in the sampler object and as part of the sampling operation in
//! the shader, and are combined by addition to produce the final bias value, which is then added
//! to the base LOD.
//!
//! Once LOD bias has been applied, the resulting value may be *clamped* to a minimum and maximum
//! value to provide the final LOD. A maximum may be specified by the sampler, while a minimum
//! can be specified by the sampler or the shader sampling operation.
//!
//! # Texel filtering
//!
//! Texel filtering operations determine how the color value to be sampled from each mipmap is
//! calculated. The filtering mode can be set independently for different signs of the LOD value:
//! - Negative or zero: **magnification**. The rendered object is closer to the viewer, and each
//!   pixel in the texture corresponds to exactly one or more than one framebuffer pixel.
//! - Positive: **minification**. The rendered object is further from the viewer, and each pixel in
//!   the texture corresponds to less than one framebuffer pixel.

use self::ycbcr::SamplerYcbcrConversion;
use crate::{
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    format::{FormatFeatures, NumericType},
    image::{
        view::{ImageView, ImageViewType},
        ImageAspects,
    },
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_enum},
    pipeline::graphics::depth_stencil::CompareOp,
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use std::{mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

pub mod ycbcr;

/// Describes how to retrieve data from a sampled image within a shader.
///
/// # Examples
///
/// A simple sampler for most usages:
///
/// ```
/// use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// #
/// let _sampler = Sampler::new(
///     &device,
///     &SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
/// );
/// ```
///
/// More detailed sampler creation:
///
/// ```
/// use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// #
/// let _sampler = Sampler::new(
///     &device,
///     &SamplerCreateInfo {
///         mag_filter: Filter::Linear,
///         min_filter: Filter::Linear,
///         address_mode: [SamplerAddressMode::Repeat; 3],
///         mip_lod_bias: 1.0,
///         max_lod: None,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// ```
#[derive(Debug)]
pub struct Sampler {
    handle: vk::Sampler,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    address_mode: [SamplerAddressMode; 3],
    anisotropy: Option<f32>,
    border_color: Option<BorderColor>,
    compare: Option<CompareOp>,
    min_lod: f32,
    max_lod: Option<f32>,
    mag_filter: Filter,
    min_filter: Filter,
    mip_lod_bias: f32,
    mipmap_mode: SamplerMipmapMode,
    reduction_mode: SamplerReductionMode,
    sampler_ycbcr_conversion: Option<DeviceOwnedDebugWrapper<Arc<SamplerYcbcrConversion>>>,
    unnormalized_coordinates: bool,
}

impl Sampler {
    /// Creates a new `Sampler`, panicking on a validation error.
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
        create_info: &SamplerCreateInfo<'_>,
    ) -> Result<Arc<Sampler>, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `Sampler`.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &SamplerCreateInfo<'_>,
    ) -> Result<Arc<Sampler>, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &SamplerCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &SamplerCreateInfo<'_>,
    ) -> Result<Arc<Sampler>, VulkanError> {
        let mut create_info_extensions_vk = create_info.to_vk_extensions();
        let create_info_vk = create_info.to_vk(&mut create_info_extensions_vk);

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_sampler)(
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

    /// Creates a new `Sampler` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::Sampler,
        create_info: &SamplerCreateInfo<'_>,
    ) -> Arc<Sampler> {
        let &SamplerCreateInfo {
            mag_filter,
            min_filter,
            mipmap_mode,
            address_mode,
            mip_lod_bias,
            anisotropy,
            compare,
            min_lod,
            max_lod,
            border_color,
            unnormalized_coordinates,
            reduction_mode,
            sampler_ycbcr_conversion,
            _ne: _,
        } = create_info;

        Arc::new(Sampler {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),
            address_mode,
            anisotropy,
            border_color: address_mode
                .into_iter()
                .any(|mode| mode == SamplerAddressMode::ClampToBorder)
                .then_some(border_color),
            compare,
            min_lod,
            max_lod,
            mag_filter,
            min_filter,
            mip_lod_bias,
            mipmap_mode,
            reduction_mode,
            sampler_ycbcr_conversion: sampler_ycbcr_conversion
                .cloned()
                .map(DeviceOwnedDebugWrapper),
            unnormalized_coordinates,
        })
    }

    /// Checks whether this sampler is compatible with `image_view`.
    pub(crate) fn check_can_sample(
        &self,
        image_view: &ImageView,
    ) -> Result<(), Box<ValidationError>> {
        /*
            NOTE: Most of these checks come from the Instruction/Sampler/Image View Validation
            section, and are not strictly VUIDs.
            https://registry.khronos.org/vulkan/specs/1.2-extensions/html/chap16.html#textures-input-validation
        */

        if self.compare.is_some() {
            // VUID-vkCmdDispatch-None-06479
            if !image_view
                .format_features()
                .intersects(FormatFeatures::SAMPLED_IMAGE_DEPTH_COMPARISON)
            {
                return Err(Box::new(ValidationError {
                    problem: "the sampler has depth comparison enabled, and \
                        the image view's format features do not include \
                        `FormatFeatures::SAMPLED_IMAGE_DEPTH_COMPARISON`"
                        .into(),
                    ..Default::default()
                }));
            }

            // The SPIR-V instruction is one of the OpImage*Dref* instructions, the image
            // view format is one of the depth/stencil formats, and the image view aspect
            // is not VK_IMAGE_ASPECT_DEPTH_BIT.
            if !image_view
                .subresource_range()
                .aspects
                .intersects(ImageAspects::DEPTH)
            {
                return Err(Box::new(ValidationError {
                    problem: "the sampler has depth comparison enabled, and \
                        the image view's aspects do not include `ImageAspects::DEPTH`"
                        .into(),
                    ..Default::default()
                }));
            }
        } else {
            if !image_view
                .format_features()
                .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR)
            {
                // VUID-vkCmdDispatch-magFilter-04553
                if self.mag_filter == Filter::Linear || self.min_filter == Filter::Linear {
                    return Err(Box::new(ValidationError {
                        problem: "the sampler's `mag_filter` or `min_filter` is `Filter::Linear`, \
                            and the image view's format features do not include \
                            `FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR`"
                            .into(),
                        ..Default::default()
                    }));
                }

                // VUID-vkCmdDispatch-mipmapMode-04770
                if self.mipmap_mode == SamplerMipmapMode::Linear {
                    return Err(Box::new(ValidationError {
                        problem: "the sampler's `mipmap_mode` is `SamplerMipmapMpde::Linear`, and \
                            the image view's format features do not include \
                            `FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR`"
                            .into(),
                        ..Default::default()
                    }));
                }
            }
        }

        if self.mag_filter == Filter::Cubic || self.min_filter == Filter::Cubic {
            // VUID-vkCmdDispatch-None-02692
            if !image_view
                .format_features()
                .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_CUBIC)
            {
                return Err(Box::new(ValidationError {
                    problem: "the sampler's `mag_filter` or `min_filter` is `Filter::Cubic`, and \
                        the image view's format features do not include \
                        `FormatFeatures::SAMPLED_IMAGE_FILTER_CUBIC`"
                        .into(),
                    ..Default::default()
                }));
            }

            // VUID-vkCmdDispatch-filterCubic-02694
            if !image_view.filter_cubic() {
                return Err(Box::new(ValidationError {
                    problem: "the sampler's `mag_filter` or `min_filter` is Filter::Cubic, and \
                        the image view does not support this, as returned by \
                        `PhysicalDevice::image_format_properties`"
                        .into(),
                    ..Default::default()
                }));
            }

            // VUID-vkCmdDispatch-filterCubicMinmax-02695
            if matches!(
                self.reduction_mode,
                SamplerReductionMode::Min | SamplerReductionMode::Max
            ) && !image_view.filter_cubic_minmax()
            {
                return Err(Box::new(ValidationError {
                    problem: "the sampler's `mag_filter` or `min_filter` is `Filter::Cubic`, and \
                        the its `reduction_mode` is `SamplerReductionMode::Min` or \
                        `SamplerReductionMode::Max`, and \
                        the image view does not support this, as returned by \
                        `PhysicalDevice::image_format_properties`"
                        .into(),
                    ..Default::default()
                }));
            }
        }

        if let Some(border_color) = self.border_color {
            let aspects = image_view.subresource_range().aspects;
            let view_numeric_type = NumericType::from(
                if aspects.intersects(
                    ImageAspects::COLOR
                        | ImageAspects::PLANE_0
                        | ImageAspects::PLANE_1
                        | ImageAspects::PLANE_2,
                ) {
                    image_view.format().numeric_format_color().unwrap()
                } else if aspects.intersects(ImageAspects::DEPTH) {
                    image_view.format().numeric_format_depth().unwrap()
                } else if aspects.intersects(ImageAspects::STENCIL) {
                    image_view.format().numeric_format_stencil().unwrap()
                } else {
                    // Per `ImageViewBuilder::aspects` and
                    // VUID-VkDescriptorImageInfo-imageView-01976
                    unreachable!()
                },
            );

            match border_color {
                BorderColor::IntTransparentBlack
                | BorderColor::IntOpaqueBlack
                | BorderColor::IntOpaqueWhite => {
                    // The sampler borderColor is an integer type and the image view
                    // format is not one of the VkFormat integer types or a stencil
                    // component of a depth/stencil format.
                    if !matches!(view_numeric_type, NumericType::Int | NumericType::Uint) {
                        return Err(Box::new(ValidationError {
                            problem: "the sampler has an integer border color, and \
                                the image view does not have an integer format"
                                .into(),
                            ..Default::default()
                        }));
                    }
                }
                BorderColor::FloatTransparentBlack
                | BorderColor::FloatOpaqueBlack
                | BorderColor::FloatOpaqueWhite => {
                    // The sampler borderColor is a float type and the image view
                    // format is not one of the VkFormat float types or a depth
                    // component of a depth/stencil format.
                    if !matches!(view_numeric_type, NumericType::Float) {
                        return Err(Box::new(ValidationError {
                            problem: "the sampler has an floating-point border color, and \
                                the image view does not have a floating-point format"
                                .into(),
                            ..Default::default()
                        }));
                    }
                }
            }

            // The sampler borderColor is one of the opaque black colors
            // (VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK or VK_BORDER_COLOR_INT_OPAQUE_BLACK)
            // and the image view VkComponentSwizzle for any of the VkComponentMapping
            // components is not the identity swizzle, and
            // VkPhysicalDeviceBorderColorSwizzleFeaturesEXT::borderColorSwizzleFromImage
            // feature is not enabled, and
            // VkSamplerBorderColorComponentMappingCreateInfoEXT is not specified.
            if matches!(
                border_color,
                BorderColor::FloatOpaqueBlack | BorderColor::IntOpaqueBlack
            ) && !image_view.component_mapping().is_identity()
            {
                return Err(Box::new(ValidationError {
                    problem: "the sampler has an opaque black border color, and \
                        the image view is not identity swizzled"
                        .into(),
                    ..Default::default()
                }));
            }
        }

        // The sampler unnormalizedCoordinates is VK_TRUE and any of the limitations of
        // unnormalized coordinates are violated.
        // https://registry.khronos.org/vulkan/specs/1.2-extensions/html/chap13.html#samplers-unnormalizedCoordinates
        if self.unnormalized_coordinates {
            // The viewType must be either VK_IMAGE_VIEW_TYPE_1D or
            // VK_IMAGE_VIEW_TYPE_2D.
            // VUID-vkCmdDispatch-None-02702
            if !matches!(
                image_view.view_type(),
                ImageViewType::Dim1d | ImageViewType::Dim2d
            ) {
                return Err(Box::new(ValidationError {
                    problem: "the sampler uses unnormalized coordinates, and \
                        the image view's type is not `ImageViewtype::Dim1d` or \
                        `ImageViewType::Dim2d`"
                        .into(),
                    ..Default::default()
                }));
            }

            // The image view must have a single layer and a single mip level.
            if image_view.subresource_range_level_count() != 1 {
                return Err(Box::new(ValidationError {
                    problem: "the sampler uses unnormalized coordinates, and \
                        the image view has more than one mip level"
                        .into(),
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    /// Returns the address modes for the u, v and w coordinates.
    #[inline]
    pub fn address_mode(&self) -> [SamplerAddressMode; 3] {
        self.address_mode
    }

    /// Returns the anisotropy mode.
    #[inline]
    pub fn anisotropy(&self) -> Option<f32> {
        self.anisotropy
    }

    /// Returns the border color if one is used by this sampler.
    #[inline]
    pub fn border_color(&self) -> Option<BorderColor> {
        self.border_color
    }

    /// Returns the compare operation if the sampler is a compare-mode sampler.
    #[inline]
    pub fn compare(&self) -> Option<CompareOp> {
        self.compare
    }

    /// Returns the minimum LOD.
    #[inline]
    pub fn min_lod(&self) -> f32 {
        self.min_lod
    }

    /// Returns the maximum LOD.
    #[inline]
    pub fn max_lod(&self) -> Option<f32> {
        self.max_lod
    }

    /// Returns the magnification filter.
    #[inline]
    pub fn mag_filter(&self) -> Filter {
        self.mag_filter
    }

    /// Returns the minification filter.
    #[inline]
    pub fn min_filter(&self) -> Filter {
        self.min_filter
    }

    /// Returns the mip LOD bias.
    #[inline]
    pub fn mip_lod_bias(&self) -> f32 {
        self.mip_lod_bias
    }

    /// Returns the mipmap mode.
    #[inline]
    pub fn mipmap_mode(&self) -> SamplerMipmapMode {
        self.mipmap_mode
    }

    /// Returns the reduction mode.
    #[inline]
    pub fn reduction_mode(&self) -> SamplerReductionMode {
        self.reduction_mode
    }

    /// Returns a reference to the sampler YCbCr conversion of this sampler, if any.
    #[inline]
    pub fn sampler_ycbcr_conversion(&self) -> Option<&Arc<SamplerYcbcrConversion>> {
        self.sampler_ycbcr_conversion.as_deref()
    }

    /// Returns true if the sampler uses unnormalized coordinates.
    #[inline]
    pub fn unnormalized_coordinates(&self) -> bool {
        self.unnormalized_coordinates
    }
}

impl Drop for Sampler {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.destroy_sampler)(self.device.handle(), self.handle, ptr::null()) };
    }
}

unsafe impl VulkanObject for Sampler {
    type Handle = vk::Sampler;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Sampler {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Sampler);

/// Parameters to create a new `Sampler`.
#[derive(Clone, Debug)]
pub struct SamplerCreateInfo<'a> {
    /// How the sampled value of a single mipmap should be calculated,
    /// when magnification is applied (LOD <= 0.0).
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    pub mag_filter: Filter,

    /// How the sampled value of a single mipmap should be calculated,
    /// when minification is applied (LOD > 0.0).
    ///
    /// The default value is [`Nearest`](Filter::Nearest).
    pub min_filter: Filter,

    /// How the final sampled value should be calculated from the samples of individual
    /// mipmaps.
    ///
    /// The default value is [`Nearest`](SamplerMipmapMode::Nearest).
    pub mipmap_mode: SamplerMipmapMode,

    /// How out-of-range texture coordinates should be treated, for the `u`, `v` and `w` texture
    /// coordinate indices respectively.
    ///
    /// The default value is [`ClampToEdge`](SamplerAddressMode::ClampToEdge).
    pub address_mode: [SamplerAddressMode; 3],

    /// The bias value to be added to the base LOD before clamping.
    ///
    /// The absolute value of the provided value must not exceed the
    /// [`max_sampler_lod_bias`](crate::device::DeviceProperties::max_sampler_lod_bias) limit of
    /// the device.
    ///
    /// On [portability
    /// subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `mip_lod_bias` is not `0.0`, the
    /// [`sampler_mip_lod_bias`](crate::device::DeviceFeatures::sampler_mip_lod_bias)
    /// feature must be enabled on the device.
    ///
    /// The default value is `0.0`.
    pub mip_lod_bias: f32,

    /// Whether anisotropic texel filtering is enabled (`Some`), and the maximum anisotropy value
    /// to use if it is enabled.
    ///
    /// Anisotropic filtering is a special filtering mode that takes into account the differences
    /// in scaling between the horizontal and vertical framebuffer axes.
    ///
    /// If set to `Some`, the
    /// [`sampler_anisotropy`](crate::device::DeviceFeatures::sampler_anisotropy) feature must
    /// be enabled on the device, the provided maximum value must not exceed the
    /// [`max_sampler_anisotropy`](crate::device::DeviceProperties::max_sampler_anisotropy) limit,
    /// and the [`Cubic`](Filter::Cubic) filter must not be used.
    ///
    /// The default value is `None`.
    pub anisotropy: Option<f32>,

    /// Whether depth comparison is enabled (`Some`), and the comparison operator to use if it is
    /// enabled.
    ///
    /// Depth comparison is an alternative mode for samplers that can be used in combination with
    /// image views specifying the depth aspect. Instead of returning a value that is sampled from
    /// the image directly, a comparison operation is applied between the sampled value and a
    /// reference value that is specified as part of the operation. The result is binary: 1.0 if
    /// the operation returns `true`, 0.0 if it returns `false`.
    ///
    /// If set to `Some`, the `reduction_mode` must be set to
    /// [`WeightedAverage`](SamplerReductionMode::WeightedAverage).
    ///
    /// On [portability
    /// subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if the sampler is going to be used as a mutable sampler (written to descriptor
    /// sets rather than being an immutable part of a descriptor set layout), the
    /// [`mutable_comparison_samplers`](crate::device::DeviceFeatures::mutable_comparison_samplers)
    /// feature must be enabled on the device.
    ///
    /// The default value is `None`.
    pub compare: Option<CompareOp>,

    /// The minimum that LOD values must be clamped to.
    ///
    /// The default value is `0.0`.
    pub min_lod: f32,

    /// The maximum that LOD values must be clamped to.
    ///
    /// If set to `None`, it is unbounded.
    ///
    /// The default value is `None`.
    pub max_lod: Option<f32>,

    /// The border color to use if `address_mode` is set to
    /// [`ClampToBorder`](SamplerAddressMode::ClampToBorder).
    ///
    /// The default value is [`FloatTransparentBlack`](BorderColor::FloatTransparentBlack).
    pub border_color: BorderColor,

    /// Whether unnormalized texture coordinates are enabled.
    ///
    /// When a sampler is set to use unnormalized coordinates as input, the texture coordinates are
    /// not scaled by the size of the image, and therefore range up to the size of the image rather
    /// than 1.0. Enabling this comes with several restrictions:
    /// - `min_filter` and `mag_filter` must be equal.
    /// - `mipmap_mode` must be [`Nearest`](SamplerMipmapMode::Nearest).
    /// - The `lod` range must be `0.0..=0.0`.
    /// - `address_mode` for u and v must be either
    ///   [`ClampToEdge`](`SamplerAddressMode::ClampToEdge`) or
    ///   [`ClampToBorder`](`SamplerAddressMode::ClampToBorder`).
    /// - Anisotropy and depth comparison must be disabled.
    ///
    /// Some restrictions also apply to the image view being sampled:
    /// - The view type must be [`Dim1d`](ImageViewType::Dim1d) or [`Dim2d`](ImageViewType::Dim2d).
    ///   Arrayed types are not allowed.
    /// - It must have a single mipmap level.
    ///
    /// Finally, restrictions apply to the sampling operations that can be used in a shader:
    /// - Only explicit LOD operations are allowed, implicit LOD operations are not.
    /// - Sampling with projection is not allowed.
    /// - Sampling with an LOD bias is not allowed.
    /// - Sampling with an offset is not allowed.
    ///
    /// The default value is `false`.
    pub unnormalized_coordinates: bool,

    /// How the value sampled from a mipmap should be calculated from the selected
    /// pixels, for the `Linear` and `Cubic` filters.
    ///
    /// The default value is [`WeightedAverage`](SamplerReductionMode::WeightedAverage).
    pub reduction_mode: SamplerReductionMode,

    /// Adds a sampler YCbCr conversion to the sampler.
    ///
    /// If set to `Some`, several restrictions apply:
    /// - If the `format` of `conversion` does not support
    ///   `sampled_image_ycbcr_conversion_separate_reconstruction_filter`, then `mag_filter` and
    ///   `min_filter` must be equal to the `chroma_filter` of `conversion`.
    /// - `address_mode` for u, v and w must be [`ClampToEdge`](`SamplerAddressMode::ClampToEdge`).
    /// - Anisotropy and unnormalized coordinates must be disabled.
    /// - The `reduction_mode` must be [`WeightedAverage`](SamplerReductionMode::WeightedAverage).
    ///
    /// In addition, the sampler must only be used as an immutable sampler within a descriptor set
    /// layout, and only in a combined image sampler descriptor.
    ///
    /// The default value is `None`.
    pub sampler_ycbcr_conversion: Option<&'a Arc<SamplerYcbcrConversion>>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for SamplerCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> SamplerCreateInfo<'a> {
    /// Returns a default `SamplerCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mipmap_mode: SamplerMipmapMode::Nearest,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            mip_lod_bias: 0.0,
            anisotropy: None,
            compare: None,
            min_lod: 0.0,
            max_lod: None,
            border_color: BorderColor::FloatTransparentBlack,
            unnormalized_coordinates: false,
            reduction_mode: SamplerReductionMode::WeightedAverage,
            sampler_ycbcr_conversion: None,
            _ne: crate::NE,
        }
    }

    /// Shortcut for creating a sampler with linear sampling, linear mipmaps, and with the repeat
    /// mode for borders.
    #[inline]
    pub fn simple_repeat_linear() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mipmap_mode: SamplerMipmapMode::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            max_lod: None,
            ..Default::default()
        }
    }

    /// Shortcut for creating a sampler with linear sampling, that only uses the main level of
    /// images, and with the repeat mode for borders.
    #[inline]
    pub fn simple_repeat_linear_no_mipmap() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            max_lod: Some(1.0),
            ..Default::default()
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            mag_filter,
            min_filter,
            mipmap_mode,
            address_mode,
            mip_lod_bias,
            anisotropy,
            compare,
            min_lod,
            max_lod,
            border_color,
            unnormalized_coordinates,
            reduction_mode,
            sampler_ycbcr_conversion,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        mag_filter.validate_device(device).map_err(|err| {
            err.add_context("mag_filter")
                .set_vuids(&["VUID-VkSamplerCreateInfo-magFilter-parameter"])
        })?;

        min_filter.validate_device(device).map_err(|err| {
            err.add_context("min_filter")
                .set_vuids(&["VUID-VkSamplerCreateInfo-minFilter-parameter"])
        })?;

        mipmap_mode.validate_device(device).map_err(|err| {
            err.add_context("mipmap_mode")
                .set_vuids(&["VUID-VkSamplerCreateInfo-mipmapMode-parameter"])
        })?;

        for (index, mode) in address_mode.into_iter().enumerate() {
            mode.validate_device(device).map_err(|err| {
                err.add_context(format!("address_mode[{}]", index))
                    .set_vuids(&[
                        "VUID-VkSamplerCreateInfo-addressModeU-parameter",
                        "VUID-VkSamplerCreateInfo-addressModeV-parameter",
                        "VUID-VkSamplerCreateInfo-addressModeW-parameter",
                    ])
            })?;
        }

        if address_mode.contains(&SamplerAddressMode::ClampToBorder) {
            border_color.validate_device(device).map_err(|err| {
                err.add_context("border_color")
                    .set_vuids(&["VUID-VkSamplerCreateInfo-addressModeU-01078"])
            })?;
        }

        reduction_mode.validate_device(device).map_err(|err| {
            err.add_context("reduction_mode")
                .set_vuids(&["VUID-VkSamplerReductionModeCreateInfo-reductionMode-parameter"])
        })?;

        if address_mode.contains(&SamplerAddressMode::MirrorClampToEdge) {
            if !(device.enabled_features().sampler_mirror_clamp_to_edge
                || device.enabled_extensions().khr_sampler_mirror_clamp_to_edge)
            {
                return Err(Box::new(ValidationError {
                    context: "address_mode".into(),
                    problem: "contains `SamplerAddressMode::MirrorClampToEdge`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature("sampler_mirror_clamp_to_edge")]),
                        RequiresAllOf(&[Requires::DeviceExtension(
                            "khr_sampler_mirror_clamp_to_edge",
                        )]),
                    ]),
                    ..Default::default()
                }));
            }
        }

        if max_lod.is_some_and(|max_lod| min_lod > max_lod) {
            return Err(Box::new(ValidationError {
                problem: "`min_lod` is greater than `max_lod`".into(),
                vuids: &["VUID-VkSamplerCreateInfo-maxLod-01973"],
                ..Default::default()
            }));
        }

        if mip_lod_bias.abs() > properties.max_sampler_lod_bias {
            return Err(Box::new(ValidationError {
                context: "mip_lod_bias".into(),
                problem: "the absolute value is greater than the `max_sampler_lod_bias` limit"
                    .into(),
                vuids: &["VUID-VkSamplerCreateInfo-mipLodBias-01069"],
                ..Default::default()
            }));
        }

        if device.enabled_extensions().khr_portability_subset
            && !device.enabled_features().sampler_mip_lod_bias
            && mip_lod_bias != 0.0
        {
            return Err(Box::new(ValidationError {
                problem: "this device is a portability subset device, and \
                    `mip_lod_bias` is not zero"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "sampler_mip_lod_bias",
                )])]),
                vuids: &["VUID-VkSamplerCreateInfo-samplerMipLodBias-04467"],
                ..Default::default()
            }));
        }

        if let Some(max_anisotropy) = anisotropy {
            if !device.enabled_features().sampler_anisotropy {
                return Err(Box::new(ValidationError {
                    context: "anisotropy".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "sampler_anisotropy",
                    )])]),
                    vuids: &["VUID-VkSamplerCreateInfo-anisotropyEnable-01070"],
                }));
            }

            if max_anisotropy < 1.0 {
                return Err(Box::new(ValidationError {
                    context: "anisotropy".into(),
                    problem: "is less than 1.0".into(),
                    vuids: &["VUID-VkSamplerCreateInfo-anisotropyEnable-01071"],
                    ..Default::default()
                }));
            }

            if max_anisotropy > properties.max_sampler_anisotropy {
                return Err(Box::new(ValidationError {
                    context: "anisotropy".into(),
                    problem: "is greater than the `max_sampler_anisotropy` limit".into(),
                    vuids: &["VUID-VkSamplerCreateInfo-anisotropyEnable-01071"],
                    ..Default::default()
                }));
            }

            if [mag_filter, min_filter].contains(&Filter::Cubic) {
                return Err(Box::new(ValidationError {
                    problem: "`anisotropy` is `Some`, but `mag_filter` or `min_filter` is \
                        `Filter::Cubic`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-magFilter-01081"],
                    ..Default::default()
                }));
            }
        }

        if let Some(compare_op) = compare {
            compare_op.validate_device(device).map_err(|err| {
                err.add_context("compare")
                    .set_vuids(&["VUID-VkSamplerCreateInfo-compareEnable-01080"])
            })?;

            if reduction_mode != SamplerReductionMode::WeightedAverage {
                return Err(Box::new(ValidationError {
                    problem: "`compare` is `Some`, but `reduction_mode` is not \
                        `SamplerReductionMode::WeightedAverage`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-compareEnable-01423"],
                    ..Default::default()
                }));
            }
        }

        if unnormalized_coordinates {
            if min_filter != mag_filter {
                return Err(Box::new(ValidationError {
                    problem: "`unnormalized_coordinates` is `true`, but \
                        `min_filter` and `mag_filter` are not equal"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01072"],
                    ..Default::default()
                }));
            }

            if mipmap_mode != SamplerMipmapMode::Nearest {
                return Err(Box::new(ValidationError {
                    problem: "`unnormalized_coordinates` is `true`, but \
                        `mipmap_mode` is not `SamplerMipmapMode::Nearest`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01073"],
                    ..Default::default()
                }));
            }

            if !(min_lod == 0.0 && max_lod == Some(0.0)) {
                return Err(Box::new(ValidationError {
                    problem: "`unnormalized_coordinates` is `true`, but `min_lod` and `max_lod` \
                        are not `0.0`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01074"],
                    ..Default::default()
                }));
            }

            if address_mode[0..2].iter().any(|mode| {
                !matches!(
                    mode,
                    SamplerAddressMode::ClampToEdge | SamplerAddressMode::ClampToBorder
                )
            }) {
                return Err(Box::new(ValidationError {
                    problem: "`unnormalized_coordinates` is `true`, but \
                        `address_mode[0]` or `address_mode[1]` are not \
                        `SamplerAddressMode::ClampToEdge` or `SamplerAddressMode::ClampToBorder`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01075"],
                    ..Default::default()
                }));
            }

            if anisotropy.is_some() {
                return Err(Box::new(ValidationError {
                    problem: "`unnormalized_coordinates` is `true`, but `anisotropy` is `Some`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01076"],
                    ..Default::default()
                }));
            }

            if compare.is_some() {
                return Err(Box::new(ValidationError {
                    problem: "`unnormalized_coordinates` is `true`, but `compare` is `Some`".into(),
                    vuids: &["VUID-VkSamplerCreateInfo-unnormalizedCoordinates-0107"],
                    ..Default::default()
                }));
            }
        }

        if reduction_mode != SamplerReductionMode::WeightedAverage {
            if !(device.enabled_features().sampler_filter_minmax
                || device.enabled_extensions().ext_sampler_filter_minmax)
            {
                return Err(Box::new(ValidationError {
                    context: "reduction_mode".into(),
                    problem: "is not `SamplerReductionMode::WeightedAverage`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature("sampler_filter_minmax")]),
                        RequiresAllOf(&[Requires::DeviceExtension("ext_sampler_filter_minmax")]),
                    ]),
                    ..Default::default()
                }));
            }
        }

        // Don't need to check features because you can't create a conversion object without the
        // feature anyway.
        if let Some(sampler_ycbcr_conversion) = sampler_ycbcr_conversion {
            assert_eq!(device, sampler_ycbcr_conversion.device().as_ref());

            // Use unchecked, because all validation has been done by the SamplerYcbcrConversion.
            let format_properties = unsafe {
                device
                    .physical_device()
                    .format_properties_unchecked(sampler_ycbcr_conversion.format())
            };
            let potential_format_features = format_properties.potential_format_features();

            if !potential_format_features.intersects(
                FormatFeatures::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER,
            ) && !(mag_filter == sampler_ycbcr_conversion.chroma_filter()
                && min_filter == sampler_ycbcr_conversion.chroma_filter())
            {
                return Err(Box::new(ValidationError {
                    problem: "`sampler_ycbcr_conversion` is `Some`, and \
                        the potential format features of `sampler_ycbcr_conversion.format()` \
                        do not include `FormatFeatures::\
                        SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER`, but \
                        `mag_filter` and `min_filter` are not both equal to \
                        `sampler_ycbcr_conversion.chroma_filter()`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-minFilter-01645"],
                    ..Default::default()
                }));
            }

            if address_mode
                .into_iter()
                .any(|mode| !matches!(mode, SamplerAddressMode::ClampToEdge))
            {
                return Err(Box::new(ValidationError {
                    problem: "`sampler_ycbcr_conversion` is `Some`, but \
                        not all elements of `address_mode` are \
                        `SamplerAddressMode::ClampToEdge`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-addressModeU-01646"],
                    ..Default::default()
                }));
            }

            if anisotropy.is_some() {
                return Err(Box::new(ValidationError {
                    problem: "`sampler_ycbcr_conversion` is `Some`, but \
                        `anisotropy` is `Some`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-addressModeU-01646"],
                    ..Default::default()
                }));
            }

            if unnormalized_coordinates {
                return Err(Box::new(ValidationError {
                    problem: "`sampler_ycbcr_conversion` is `Some`, but \
                        `unnormalized_coordinates` is `true`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-addressModeU-01646"],
                    ..Default::default()
                }));
            }

            if reduction_mode != SamplerReductionMode::WeightedAverage {
                return Err(Box::new(ValidationError {
                    problem: "`sampler_ycbcr_conversion` is `Some`, but \
                        `reduction_mode` is not `SamplerReductionMode::WeightedAverage`"
                        .into(),
                    vuids: &["VUID-VkSamplerCreateInfo-None-01647"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        extensions_vk: &'a mut SamplerCreateInfoExtensionsVk,
    ) -> vk::SamplerCreateInfo<'a> {
        let &Self {
            mag_filter,
            min_filter,
            mipmap_mode,
            address_mode,
            mip_lod_bias,
            anisotropy,
            compare,
            min_lod,
            max_lod,
            border_color,
            unnormalized_coordinates,
            reduction_mode: _,
            sampler_ycbcr_conversion: _,
            _ne: _,
        } = self;

        let (anisotropy_enable_vk, max_anisotropy_vk) = if let Some(max_anisotropy) = anisotropy {
            (true, max_anisotropy)
        } else {
            (false, 1.0)
        };

        let (compare_enable_vk, compare_op_vk) = if let Some(compare_op) = compare {
            (true, compare_op)
        } else {
            (false, CompareOp::Never)
        };

        let mut val_vk = vk::SamplerCreateInfo::default()
            .flags(vk::SamplerCreateFlags::empty())
            .mag_filter(mag_filter.into())
            .min_filter(min_filter.into())
            .mipmap_mode(mipmap_mode.into())
            .address_mode_u(address_mode[0].into())
            .address_mode_v(address_mode[1].into())
            .address_mode_w(address_mode[2].into())
            .mip_lod_bias(mip_lod_bias)
            .anisotropy_enable(anisotropy_enable_vk)
            .max_anisotropy(max_anisotropy_vk)
            .compare_enable(compare_enable_vk)
            .compare_op(compare_op_vk.into())
            .min_lod(min_lod)
            .max_lod(max_lod.unwrap_or(vk::LOD_CLAMP_NONE))
            .border_color(border_color.into())
            .unnormalized_coordinates(unnormalized_coordinates);

        let SamplerCreateInfoExtensionsVk {
            reduction_mode_vk,
            ycbcr_conversion_vk,
        } = extensions_vk;

        if let Some(next) = reduction_mode_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = ycbcr_conversion_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions(&self) -> SamplerCreateInfoExtensionsVk {
        let &Self {
            reduction_mode,
            sampler_ycbcr_conversion,
            ..
        } = self;

        let reduction_mode_vk =
            (reduction_mode != SamplerReductionMode::WeightedAverage).then(|| {
                vk::SamplerReductionModeCreateInfo::default().reduction_mode(reduction_mode.into())
            });

        let ycbcr_conversion_vk =
            sampler_ycbcr_conversion
                .as_ref()
                .map(|sampler_ycbcr_conversion| {
                    vk::SamplerYcbcrConversionInfo::default()
                        .conversion(sampler_ycbcr_conversion.handle())
                });

        SamplerCreateInfoExtensionsVk {
            reduction_mode_vk,
            ycbcr_conversion_vk,
        }
    }
}

pub(crate) struct SamplerCreateInfoExtensionsVk {
    pub(crate) reduction_mode_vk: Option<vk::SamplerReductionModeCreateInfo<'static>>,
    pub(crate) ycbcr_conversion_vk: Option<vk::SamplerYcbcrConversionInfo<'static>>,
}

/// A mapping between components of a source format and components read by a shader.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ComponentMapping {
    /// First component.
    pub r: ComponentSwizzle,
    /// Second component.
    pub g: ComponentSwizzle,
    /// Third component.
    pub b: ComponentSwizzle,
    /// Fourth component.
    pub a: ComponentSwizzle,
}

impl ComponentMapping {
    /// Creates a `ComponentMapping` with all components identity swizzled.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            r: ComponentSwizzle::Identity,
            g: ComponentSwizzle::Identity,
            b: ComponentSwizzle::Identity,
            a: ComponentSwizzle::Identity,
        }
    }

    /// Returns `true` if all components are identity swizzled,
    /// meaning that all the members are `Identity` or the name of that member.
    ///
    /// Certain operations require views that are identity swizzled, and will return an error
    /// otherwise. For example, attaching a view to a framebuffer is only possible if the view is
    /// identity swizzled.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.r_is_identity() && self.g_is_identity() && self.b_is_identity() && self.a_is_identity()
    }

    /// Returns `true` if the red component mapping is identity swizzled.
    #[inline]
    pub fn r_is_identity(&self) -> bool {
        matches!(self.r, ComponentSwizzle::Identity | ComponentSwizzle::Red)
    }

    /// Returns `true` if the green component mapping is identity swizzled.
    #[inline]
    pub fn g_is_identity(&self) -> bool {
        matches!(self.g, ComponentSwizzle::Identity | ComponentSwizzle::Green)
    }

    /// Returns `true` if the blue component mapping is identity swizzled.
    #[inline]
    pub fn b_is_identity(&self) -> bool {
        matches!(self.b, ComponentSwizzle::Identity | ComponentSwizzle::Blue)
    }

    /// Returns `true` if the alpha component mapping is identity swizzled.
    #[inline]
    pub fn a_is_identity(&self) -> bool {
        matches!(self.a, ComponentSwizzle::Identity | ComponentSwizzle::Alpha)
    }

    /// Returns the component indices that each component reads from. The index is `None` if the
    /// component has a fixed value and is not read from anywhere (`Zero` or `One`).
    #[inline]
    pub fn component_map(&self) -> [Option<usize>; 4] {
        [
            match self.r {
                ComponentSwizzle::Identity => Some(0),
                ComponentSwizzle::Zero => None,
                ComponentSwizzle::One => None,
                ComponentSwizzle::Red => Some(0),
                ComponentSwizzle::Green => Some(1),
                ComponentSwizzle::Blue => Some(2),
                ComponentSwizzle::Alpha => Some(3),
            },
            match self.g {
                ComponentSwizzle::Identity => Some(1),
                ComponentSwizzle::Zero => None,
                ComponentSwizzle::One => None,
                ComponentSwizzle::Red => Some(0),
                ComponentSwizzle::Green => Some(1),
                ComponentSwizzle::Blue => Some(2),
                ComponentSwizzle::Alpha => Some(3),
            },
            match self.b {
                ComponentSwizzle::Identity => Some(2),
                ComponentSwizzle::Zero => None,
                ComponentSwizzle::One => None,
                ComponentSwizzle::Red => Some(0),
                ComponentSwizzle::Green => Some(1),
                ComponentSwizzle::Blue => Some(2),
                ComponentSwizzle::Alpha => Some(3),
            },
            match self.a {
                ComponentSwizzle::Identity => Some(3),
                ComponentSwizzle::Zero => None,
                ComponentSwizzle::One => None,
                ComponentSwizzle::Red => Some(0),
                ComponentSwizzle::Green => Some(1),
                ComponentSwizzle::Blue => Some(2),
                ComponentSwizzle::Alpha => Some(3),
            },
        ]
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self { r, g, b, a } = self;

        r.validate_device(device).map_err(|err| {
            err.add_context("r")
                .set_vuids(&["VUID-VkComponentMapping-r-parameter"])
        })?;

        g.validate_device(device).map_err(|err| {
            err.add_context("g")
                .set_vuids(&["VUID-VkComponentMapping-g-parameter"])
        })?;

        b.validate_device(device).map_err(|err| {
            err.add_context("b")
                .set_vuids(&["VUID-VkComponentMapping-b-parameter"])
        })?;

        a.validate_device(device).map_err(|err| {
            err.add_context("a")
                .set_vuids(&["VUID-VkComponentMapping-a-parameter"])
        })?;

        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_vk(&self) -> vk::ComponentMapping {
        let &Self { r, g, b, a } = self;

        vk::ComponentMapping {
            r: r.into(),
            g: g.into(),
            b: b.into(),
            a: a.into(),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes the value that an individual component must return when being accessed.
    ComponentSwizzle = ComponentSwizzle(i32);

    /// Returns the value that this component should normally have.
    ///
    /// This is the `Default` value.
    Identity = IDENTITY,

    /// Always return zero.
    Zero = ZERO,

    /// Always return one.
    One = ONE,

    /// Returns the value of the first component.
    Red = R,

    /// Returns the value of the second component.
    Green = G,

    /// Returns the value of the third component.
    Blue = B,

    /// Returns the value of the fourth component.
    Alpha = A,
}

impl Default for ComponentSwizzle {
    #[inline]
    fn default() -> ComponentSwizzle {
        ComponentSwizzle::Identity
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes how the color of each pixel should be determined.
    Filter = Filter(i32);

    /// The pixel whose center is nearest to the requested coordinates is taken from the source
    /// and its value is returned as-is.
    Nearest = NEAREST,

    /// The 8/4/2 pixels (depending on view dimensionality) whose center surround the requested
    /// coordinates are taken, then their values are combined according to the chosen
    /// `reduction_mode`.
    Linear = LINEAR,

    /// The 64/16/4 pixels (depending on the view dimensionality) whose center surround the
    /// requested coordinates are taken, then their values are combined according to the chosen
    /// `reduction_mode`.
    ///
    /// The [`ext_filter_cubic`](crate::device::DeviceExtensions::ext_filter_cubic) extension must
    /// be enabled on the device, and anisotropy must be disabled. Sampled image views must have
    /// a type of [`Dim2d`](ImageViewType::Dim2d).
    Cubic = CUBIC_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_filter_cubic)]),
        RequiresAllOf([DeviceExtension(img_filter_cubic)]),
    ]),
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes which mipmap from the source to use.
    SamplerMipmapMode = SamplerMipmapMode(i32);

    /// Use the mipmap whose dimensions are the nearest to the dimensions of the destination.
    Nearest = NEAREST,

    /// Take the mipmap whose dimensions are no greater than that of the destination together
    /// with the next higher level mipmap, calculate the value for both, and interpolate them.
    Linear = LINEAR,
}

vulkan_enum! {
    #[non_exhaustive]

    /// How the sampler should behave when it needs to access a pixel that is out of range of the
    /// texture.
    SamplerAddressMode = SamplerAddressMode(i32);

    /// Repeat the texture. In other words, the pixel at coordinate `x + 1.0` is the same as the
    /// one at coordinate `x`.
    Repeat = REPEAT,

    /// Repeat the texture but mirror it at every repetition. In other words, the pixel at
    /// coordinate `x + 1.0` is the same as the one at coordinate `1.0 - x`.
    MirroredRepeat = MIRRORED_REPEAT,

    /// The coordinates are clamped to the valid range. Coordinates below 0.0 have the same value
    /// as coordinate 0.0. Coordinates over 1.0 have the same value as coordinate 1.0.
    ClampToEdge = CLAMP_TO_EDGE,

    /// Any pixel out of range is colored using the color selected with the `border_color` on the
    /// `SamplerBuilder`.
    ///
    /// When this mode is chosen, the numeric type of the image view's format must match the border
    /// color. When using a floating-point border color, the sampler can only be used with
    /// floating-point or depth image views. When using an integer border color, the sampler can
    /// only be used with integer or stencil image views. In addition to this, you can't use an
    /// opaque black border color with an image view that uses component swizzling.
    ClampToBorder = CLAMP_TO_BORDER,

    /// Similar to `MirroredRepeat`, except that coordinates are clamped to the range
    /// `[-1.0, 1.0]`.
    ///
    /// The [`sampler_mirror_clamp_to_edge`](crate::device::DeviceFeatures::sampler_mirror_clamp_to_edge)
    /// feature or the
    /// [`khr_sampler_mirror_clamp_to_edge`](crate::device::DeviceExtensions::khr_sampler_mirror_clamp_to_edge)
    /// extension must be enabled on the device.
    MirrorClampToEdge = MIRROR_CLAMP_TO_EDGE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(khr_sampler_mirror_clamp_to_edge)]),
    ]),
}

vulkan_enum! {
    #[non_exhaustive]

    /// The color to use for the border of an image.
    ///
    /// Only relevant if you use `ClampToBorder`.
    ///
    /// Using a border color restricts the sampler to either floating-point images or integer images.
    BorderColor = BorderColor(i32);

    /// The value `(0.0, 0.0, 0.0, 0.0)`. Can only be used with floating-point images.
    FloatTransparentBlack = FLOAT_TRANSPARENT_BLACK,

    /// The value `(0, 0, 0, 0)`. Can only be used with integer images.
    IntTransparentBlack = INT_TRANSPARENT_BLACK,

    /// The value `(0.0, 0.0, 0.0, 1.0)`. Can only be used with floating-point identity-swizzled
    /// images.
    FloatOpaqueBlack = FLOAT_OPAQUE_BLACK,

    /// The value `(0, 0, 0, 1)`. Can only be used with integer identity-swizzled images.
    IntOpaqueBlack = INT_OPAQUE_BLACK,

    /// The value `(1.0, 1.0, 1.0, 1.0)`. Can only be used with floating-point images.
    FloatOpaqueWhite = FLOAT_OPAQUE_WHITE,

    /// The value `(1, 1, 1, 1)`. Can only be used with integer images.
    IntOpaqueWhite = INT_OPAQUE_WHITE,

    /* TODO: enable
    // TODO: document
    FloatCustom = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_custom_border_color)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    IntCustom = INT_CUSTOM_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_custom_border_color)]),
    ]),*/
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes how the value sampled from a mipmap should be calculated from the selected
    /// pixels, for the `Linear` and `Cubic` filters.
    SamplerReductionMode = SamplerReductionMode(i32);

    /// Calculates a weighted average of the selected pixels. For `Linear` filtering the pixels
    /// are evenly weighted, for `Cubic` filtering they use Catmull-Rom weights.
    WeightedAverage = WEIGHTED_AVERAGE,

    /// Calculates the minimum of the selected pixels.
    ///
    /// The [`sampler_filter_minmax`](crate::device::DeviceFeatures::sampler_filter_minmax)
    /// feature or the
    /// [`ext_sampler_filter_minmax`](crate::device::DeviceExtensions::ext_sampler_filter_minmax)
    /// extension must be enabled on the device.
    Min = MIN,

    /// Calculates the maximum of the selected pixels.
    ///
    /// The [`sampler_filter_minmax`](crate::device::DeviceFeatures::sampler_filter_minmax)
    /// feature or the
    /// [`ext_sampler_filter_minmax`](crate::device::DeviceExtensions::ext_sampler_filter_minmax)
    /// extension must be enabled on the device.
    Max = MAX,
}

#[cfg(test)]
mod tests {
    use crate::{
        image::sampler::{
            Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerReductionMode,
        },
        pipeline::graphics::depth_stencil::CompareOp,
        Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError,
    };

    #[test]
    fn create_regular() {
        let (device, _queue) = gfx_dev_and_queue!();

        let s = Sampler::new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                max_lod: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.compare().is_none());
        assert!(!s.unnormalized_coordinates());
    }

    #[test]
    fn create_compare() {
        let (device, _queue) = gfx_dev_and_queue!();

        let s = Sampler::new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                compare: Some(CompareOp::Less),
                max_lod: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.compare().is_some());
        assert!(!s.unnormalized_coordinates());
    }

    #[test]
    fn create_unnormalized() {
        let (device, _queue) = gfx_dev_and_queue!();

        let s = Sampler::new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                max_lod: Some(0.0),
                unnormalized_coordinates: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.compare().is_none());
        assert!(s.unnormalized_coordinates());
    }

    #[test]
    fn simple_repeat_linear() {
        let (device, _queue) = gfx_dev_and_queue!();
        let _ = Sampler::new(&device, &SamplerCreateInfo::simple_repeat_linear());
    }

    #[test]
    fn simple_repeat_linear_no_mipmap() {
        let (device, _queue) = gfx_dev_and_queue!();
        let _ = Sampler::new(
            &device,
            &SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
        );
    }

    #[test]
    fn min_lod_inferior() {
        let (device, _queue) = gfx_dev_and_queue!();

        Sampler::try_new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                min_lod: 5.0,
                max_lod: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap_err();
    }

    #[test]
    fn max_anisotropy() {
        let (device, _queue) = gfx_dev_and_queue!();

        Sampler::try_new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                anisotropy: Some(0.5),
                max_lod: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap_err();
    }

    #[test]
    fn anisotropy_feature() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = Sampler::try_new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                anisotropy: Some(2.0),
                max_lod: Some(2.0),
                ..Default::default()
            },
        );

        assert!(matches!(r, Err(Validated::ValidationError(_))));
    }

    #[test]
    fn anisotropy_limit() {
        let (device, _queue) = gfx_dev_and_queue!(sampler_anisotropy);

        let r = Sampler::try_new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 1.0,
                anisotropy: Some(100000000.0),
                max_lod: Some(2.0),
                ..Default::default()
            },
        );

        assert!(matches!(r, Err(Validated::ValidationError(_))));
    }

    #[test]
    fn mip_lod_bias_limit() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = Sampler::try_new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 100000000.0,
                max_lod: Some(2.0),
                ..Default::default()
            },
        );

        assert!(matches!(r, Err(Validated::ValidationError(_))));
    }

    #[test]
    fn sampler_mirror_clamp_to_edge_extension() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = Sampler::try_new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::MirrorClampToEdge; 3],
                mip_lod_bias: 1.0,
                max_lod: Some(2.0),
                ..Default::default()
            },
        );

        assert!(matches!(
            r,
            Err(Validated::ValidationError(err)) if matches!(
                *err,
                ValidationError {
                    requires_one_of: RequiresOneOf([
                        RequiresAllOf([Requires::DeviceFeature(
                            "sampler_mirror_clamp_to_edge",
                        )]),
                        RequiresAllOf([Requires::DeviceExtension(
                            "khr_sampler_mirror_clamp_to_edge",
                        )]),
                    ]),
                    ..
                },
            ),
        ));
    }

    #[test]
    fn sampler_filter_minmax_extension() {
        let (device, _queue) = gfx_dev_and_queue!();

        let r = Sampler::try_new(
            &device,
            &SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                reduction_mode: SamplerReductionMode::Min,
                ..Default::default()
            },
        );

        assert!(matches!(
            r,
            Err(Validated::ValidationError(err)) if matches!(
                *err,
                ValidationError {
                    requires_one_of: RequiresOneOf([
                        RequiresAllOf([Requires::DeviceFeature("sampler_filter_minmax")]),
                        RequiresAllOf([Requires::DeviceExtension("ext_sampler_filter_minmax")]),
                    ]),
                    ..
                },
            ),
        ));
    }
}
