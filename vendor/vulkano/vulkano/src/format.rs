//! All the formats supported by Vulkan.
//!
//! A format is mostly used to describe the texel data of an image. However, formats also show up
//! in a few other places, most notably to describe the format of vertex buffers.
//!
//! # Format support
//!
//! Not all formats are supported by every device. Those that devices do support may only be
//! supported for certain use cases. It is an error to use a format where it is not supported, but
//! you can query a device beforehand for its support by calling `format_properties` on the
//! physical device. You can use this to select a usable format from one or more suitable
//! alternatives. Some formats are required to be always supported for a particular usage. These
//! are listed in the [tables in the Vulkan specification](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap43.html#features-required-format-support).
//!
//! # Special format types
//!
//! ## Depth/stencil formats
//!
//! Depth/stencil formats can be identified by the `D` and `S` components in their names. They are
//! used primarily as the format for framebuffer attachments, for the purposes of depth and stencil
//! testing.
//!
//! Some formats have only a depth or stencil component, while others combine both. The two
//! components are represented as separate *aspects*, which means that they can be accessed
//! individually as separate images. These pseudo-images have the same resolution, but different
//! bit depth and numeric representation.
//!
//! Depth/stencil formats deviate from the others in a few more ways. Their data representation is
//! considered opaque, meaning that they do not have a fixed layout in memory nor a fixed size per
//! texel. They also have special limitations in several operations such as copying; a
//! depth/stencil format is not compatible with any other format, only with itself.
//!
//! ## Block-compressed formats
//!
//! A block-compressed format uses compression to encode a larger block of texels into a smaller
//! number of bytes. Individual texels are no longer represented in memory, only the block as a
//! whole. An image must consist of a whole number of blocks, so the extent of an image must be
//! a whole multiple of the block extent. Vulkan supports several different compression schemes,
//! represented in Vulkano by the `CompressionType` enum.
//!
//! Overall, block-compressed formats do not behave significantly differently from regular formats.
//! They are mostly restricted in terms of compatibility. Because of the compression, the notion of
//! bits per component does not apply, so the `components` method will only return whether a
//! component is present or not.
//!
//! ## YCbCr formats
//!
//! YCbCr, also known as YUV, is an alternative image representation with three components:
//! Y for luminance or *luma* (overall brightness) and two color or *chroma* components Cb and Cr
//! encoding the blueness and redness respectively. YCbCr formats are primarily used in video
//! applications. In Vulkan, the formats used to encode YCbCr data use the green channel to
//! represent the luma component, while the blue and red components hold the chroma.
//!
//! To use most YCbCr formats in an [image view](crate::image::view), a
//! [sampler YCbCr conversion](crate::image::sampler::ycbcr) object must be created, and attached
//! to both the image view and the sampler.
//! To query whether a format requires the conversion, you can call `ycbcr_chroma_sampling` on a
//! format. As a rule, any format with `444`, `422`, `420`, `3PACK` or `4PACK` in the name
//! requires it.
//!
//! Many YCbCr formats make use of **chroma subsampling**. This is a technique whereby the two
//! chroma components are encoded using a lower resolution than the luma component. The human eye
//! is less sensitive to color detail than to detail in brightness, so this allows more detail to
//! be encoded in less data. Chroma subsampling is indicated with one of three numbered suffixes in
//! a format name:
//! - `444` indicates a YCbCr format without chroma subsampling. All components have the same
//!   resolution.
//! - `422` indicates horizontal chroma subsampling. The horizontal resolution of the chroma
//!   components is halved, so a single value is shared within a 2x1 block of texels.
//! - `420` indicates horizontal and vertical chroma subsampling. Both dimensions of the chroma
//!   components are halved, so a single value is shared within a 2x2 block of texels.
//!
//! Most YCbCr formats, including all of the `444` and `420` formats, are **multi-planar**. Instead
//! of storing the components of a single texel together in memory, the components are separated
//! into *planes*, which act like independent images. In 3-plane formats, the planes hold the Y,
//! Cb and Cr components respectively, while in 2-plane formats, Cb and Cr are combined into a
//! two-component plane. Where chroma subsampling is applied, plane 0 has the full resolution,
//! while planes 1 and 2 have reduced resolution. Effectively, they are standalone images with half
//! the resolution of the original.
//!
//! The texels of multi-planar images cannot be accessed individually, for example to copy or blit,
//! since the components of each texel are split across the planes. Instead, you must access each
//! plane as an individual *aspect* of the image. A single-plane aspect of a multi-planar image
//! behaves as a regular image, and even has its own format, which can be queried with the `plane`
//! method on a format.

use crate::{
    device::{physical::PhysicalDevice, Device},
    image::{ImageAspects, ImageTiling},
    macros::vulkan_bitflags,
    shader::spirv::ImageFormat,
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version,
};
use ash::vk;
use ash::vk::TaggedStructure;

use std::marker::PhantomData;

include!(crate::autogen_output!("formats.rs"));

impl Format {
    /// Returns whether the format can be used with a storage image, without specifying
    /// the format in the shader, if the
    /// [`shader_storage_image_read_without_format`](crate::device::DeviceFeatures::shader_storage_image_read_without_format)
    /// and/or
    /// [`shader_storage_image_write_without_format`](crate::device::DeviceFeatures::shader_storage_image_write_without_format)
    /// features are enabled on the device.
    #[inline]
    pub fn shader_storage_image_without_format(self) -> bool {
        matches!(
            self,
            Format::R8G8B8A8_UNORM
                | Format::R8G8B8A8_SNORM
                | Format::R8G8B8A8_UINT
                | Format::R8G8B8A8_SINT
                | Format::R32_UINT
                | Format::R32_SINT
                | Format::R32_SFLOAT
                | Format::R32G32_UINT
                | Format::R32G32_SINT
                | Format::R32G32_SFLOAT
                | Format::R32G32B32A32_UINT
                | Format::R32G32B32A32_SINT
                | Format::R32G32B32A32_SFLOAT
                | Format::R16G16B16A16_UINT
                | Format::R16G16B16A16_SINT
                | Format::R16G16B16A16_SFLOAT
                | Format::R16G16_SFLOAT
                | Format::B10G11R11_UFLOAT_PACK32
                | Format::R16_SFLOAT
                | Format::R16G16B16A16_UNORM
                | Format::A2B10G10R10_UNORM_PACK32
                | Format::R16G16_UNORM
                | Format::R8G8_UNORM
                | Format::R16_UNORM
                | Format::R8_UNORM
                | Format::R16G16B16A16_SNORM
                | Format::R16G16_SNORM
                | Format::R8G8_SNORM
                | Format::R16_SNORM
                | Format::R8_SNORM
                | Format::R16G16_SINT
                | Format::R8G8_SINT
                | Format::R16_SINT
                | Format::R8_SINT
                | Format::A2B10G10R10_UINT_PACK32
                | Format::R16G16_UINT
                | Format::R8G8_UINT
                | Format::R16_UINT
                | Format::R8_UINT
        )
    }
}

impl From<Format> for vk::Format {
    #[inline]
    fn from(val: Format) -> Self {
        vk::Format::from_raw(val as i32)
    }
}

// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap46.html#spirvenv-image-formats
impl From<ImageFormat> for Option<Format> {
    #[inline]
    fn from(val: ImageFormat) -> Self {
        match val {
            ImageFormat::Unknown => None,
            ImageFormat::Rgba32f => Some(Format::R32G32B32A32_SFLOAT),
            ImageFormat::Rgba16f => Some(Format::R16G16B16A16_SFLOAT),
            ImageFormat::R32f => Some(Format::R32_SFLOAT),
            ImageFormat::Rgba8 => Some(Format::R8G8B8A8_UNORM),
            ImageFormat::Rgba8Snorm => Some(Format::R8G8B8A8_SNORM),
            ImageFormat::Rg32f => Some(Format::R32G32_SFLOAT),
            ImageFormat::Rg16f => Some(Format::R16G16_SFLOAT),
            ImageFormat::R11fG11fB10f => Some(Format::B10G11R11_UFLOAT_PACK32),
            ImageFormat::R16f => Some(Format::R16_SFLOAT),
            ImageFormat::Rgba16 => Some(Format::R16G16B16A16_UNORM),
            ImageFormat::Rgb10A2 => Some(Format::A2B10G10R10_UNORM_PACK32),
            ImageFormat::Rg16 => Some(Format::R16G16_UNORM),
            ImageFormat::Rg8 => Some(Format::R8G8_UNORM),
            ImageFormat::R16 => Some(Format::R16_UNORM),
            ImageFormat::R8 => Some(Format::R8_UNORM),
            ImageFormat::Rgba16Snorm => Some(Format::R16G16B16A16_SNORM),
            ImageFormat::Rg16Snorm => Some(Format::R16G16_SNORM),
            ImageFormat::Rg8Snorm => Some(Format::R8G8_SNORM),
            ImageFormat::R16Snorm => Some(Format::R16_SNORM),
            ImageFormat::R8Snorm => Some(Format::R8_SNORM),
            ImageFormat::Rgba32i => Some(Format::R32G32B32A32_SINT),
            ImageFormat::Rgba16i => Some(Format::R16G16B16A16_SINT),
            ImageFormat::Rgba8i => Some(Format::R8G8B8A8_SINT),
            ImageFormat::R32i => Some(Format::R32_SINT),
            ImageFormat::Rg32i => Some(Format::R32G32_SINT),
            ImageFormat::Rg16i => Some(Format::R16G16_SINT),
            ImageFormat::Rg8i => Some(Format::R8G8_SINT),
            ImageFormat::R16i => Some(Format::R16_SINT),
            ImageFormat::R8i => Some(Format::R8_SINT),
            ImageFormat::Rgba32ui => Some(Format::R32G32B32A32_UINT),
            ImageFormat::Rgba16ui => Some(Format::R16G16B16A16_UINT),
            ImageFormat::Rgba8ui => Some(Format::R8G8B8A8_UINT),
            ImageFormat::R32ui => Some(Format::R32_UINT),
            ImageFormat::Rgb10a2ui => Some(Format::A2B10G10R10_UINT_PACK32),
            ImageFormat::Rg32ui => Some(Format::R32G32_UINT),
            ImageFormat::Rg16ui => Some(Format::R16G16_UINT),
            ImageFormat::Rg8ui => Some(Format::R8G8_UINT),
            ImageFormat::R16ui => Some(Format::R16_UINT),
            ImageFormat::R8ui => Some(Format::R8_UINT),
            ImageFormat::R64ui => Some(Format::R64_UINT),
            ImageFormat::R64i => Some(Format::R64_SINT),
        }
    }
}

/// The block compression scheme used in a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum CompressionType {
    /// Adaptive Scalable Texture Compression, low dynamic range.
    ASTC_LDR,
    /// Adaptive Scalable Texture Compression, high dynamic range.
    ASTC_HDR,
    /// S3TC Block Compression.
    BC,
    /// Ericsson Texture Compression 2.
    ETC2,
    /// ETC2 Alpha Compression.
    EAC,
    /// PowerVR Texture Compression.
    PVRTC,
}

/// For YCbCr formats, the type of chroma sampling used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaSampling {
    /// The chroma components are represented at the same resolution as the luma component.
    Mode444,
    /// The chroma components have half the horizontal resolution as the luma component.
    Mode422,
    /// The chroma components have half the horizontal and vertical resolution as the luma
    /// component.
    Mode420,
}

impl ChromaSampling {
    #[inline]
    pub fn subsampled_extent(self, mut extent: [u32; 3]) -> [u32; 3] {
        match self {
            ChromaSampling::Mode444 => (),
            ChromaSampling::Mode422 => {
                debug_assert!(extent[0].is_multiple_of(2));
                extent[0] /= 2;
            }
            ChromaSampling::Mode420 => {
                debug_assert!(extent[0].is_multiple_of(2) && extent[1].is_multiple_of(2));
                extent[0] /= 2;
                extent[1] /= 2;
            }
        }

        extent
    }
}

/// The numeric format in memory of the components of a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericFormat {
    /// Signed floating-point number.
    SFLOAT,
    /// Unsigned floating-point number.
    UFLOAT,
    /// Signed integer.
    SINT,
    /// Unsigned integer.
    UINT,
    /// Signed integer that represents a normalized floating-point value in the range \[-1,1].
    SNORM,
    /// Unsigned integer that represents a normalized floating-point value in the range \[0,1].
    UNORM,
    /// Signed integer that is converted to a floating-point value directly.
    SSCALED,
    /// Unsigned integer that is converted to a floating-point value directly.
    USCALED,
    /// Unsigned integer where R, G, B components represent a normalized floating-point value in
    /// the sRGB color space, while the A component is a simple normalized value as in `UNORM`.
    SRGB,
    /// Signed fixed-point value (e.g. `VK_FORMAT_R16G16_SFIXED5_NV`).
    SFIXED5,
    /// Boolean value (e.g. `VK_FORMAT_R8_BOOL_ARM`).
    BOOL,
}

impl NumericFormat {
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap47.html#formats-numericformat
    pub const fn numeric_type(self) -> NumericType {
        match self {
            NumericFormat::SFLOAT
            | NumericFormat::UFLOAT
            | NumericFormat::SNORM
            | NumericFormat::UNORM
            | NumericFormat::SSCALED
            | NumericFormat::USCALED
            | NumericFormat::SRGB => NumericType::Float,
            NumericFormat::SINT | NumericFormat::SFIXED5 => NumericType::Int,
            NumericFormat::UINT | NumericFormat::BOOL => NumericType::Uint,
        }
    }
}

/// The numeric base type of a scalar value, in a format, a shader, or elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericType {
    Float,
    Int,
    Uint,
}

impl From<NumericFormat> for NumericType {
    #[inline]
    fn from(val: NumericFormat) -> Self {
        val.numeric_type()
    }
}

/// An opaque type that represents a format compatibility class.
///
/// Two formats are compatible if their compatibility classes compare equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FormatCompatibility(pub(crate) &'static FormatCompatibilityInner);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub(crate) enum FormatCompatibilityInner {
    Undefined,
    Class_8bit,
    Class_16bit,
    Class_24bit,
    Class_32bit,
    Class_48bit,
    Class_64bit,
    Class_96bit,
    Class_128bit,
    Class_192bit,
    Class_256bit,
    Class_D16,
    Class_D24,
    Class_D32,
    Class_S8,
    Class_D16S8,
    Class_D24S8,
    Class_D32S8,
    Class_64bit_R10G10B10A10,
    Class_64bit_R12G12B12A12,
    Class_64bit_R14G14B14A14,
    Class_BC1_RGB,
    Class_BC1_RGBA,
    Class_BC2,
    Class_BC3,
    Class_BC4,
    Class_BC5,
    Class_BC6H,
    Class_BC7,
    Class_ETC2_RGB,
    Class_ETC2_RGBA,
    Class_ETC2_EAC_RGBA,
    Class_EAC_R,
    Class_EAC_RG,
    Class_ASTC_3x3x3,
    Class_ASTC_4x3x3,
    Class_ASTC_4x4,
    Class_ASTC_4x4x3,
    Class_ASTC_4x4x4,
    Class_ASTC_5x4,
    Class_ASTC_5x4x4,
    Class_ASTC_5x5,
    Class_ASTC_5x5x4,
    Class_ASTC_5x5x5,
    Class_ASTC_6x5,
    Class_ASTC_6x5x5,
    Class_ASTC_6x6,
    Class_ASTC_6x6x5,
    Class_ASTC_6x6x6,
    Class_ASTC_8x5,
    Class_ASTC_8x6,
    Class_ASTC_8x8,
    Class_ASTC_10x5,
    Class_ASTC_10x6,
    Class_ASTC_10x8,
    Class_ASTC_10x10,
    Class_ASTC_12x10,
    Class_ASTC_12x12,
    Class_PVRTC1_2BPP,
    Class_PVRTC1_4BPP,
    Class_PVRTC2_2BPP,
    Class_PVRTC2_4BPP,
    Class_32bit_G8B8G8R8,
    Class_32bit_B8G8R8G8,
    Class_64bit_G10B10G10R10,
    Class_64bit_B10G10R10G10,
    Class_64bit_G12B12G12R12,
    Class_64bit_B12G12R12G12,
    Class_64bit_G16B16G16R16,
    Class_64bit_B16G16R16G16,
    Class_8bit_3plane_420,
    Class_8bit_2plane_420,
    Class_10bit_3plane_420,
    Class_10bit_2plane_420,
    Class_12bit_3plane_420,
    Class_12bit_2plane_420,
    Class_16bit_3plane_420,
    Class_16bit_2plane_420,
    Class_14bit_2plane_420,
    Class_8bit_3plane_422,
    Class_8bit_2plane_422,
    Class_10bit_3plane_422,
    Class_10bit_2plane_422,
    Class_12bit_3plane_422,
    Class_12bit_2plane_422,
    Class_14bit_2plane_422,
    Class_16bit_3plane_422,
    Class_16bit_2plane_422,
    Class_8bit_3plane_444,
    Class_10bit_3plane_444,
    Class_12bit_3plane_444,
    Class_16bit_3plane_444,
    Class_8bit_2plane_444,
    Class_10bit_2plane_444,
    Class_12bit_2plane_444,
    Class_16bit_2plane_444,
    Class_8bit_alpha,
}

/// Describes a uniform value that will be used to fill an image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearValue {
    /// Value for floating-point attachments, including `UNORM`, `SNORM`, `SFLOAT`.
    Float([f32; 4]),

    /// Value for integer attachments, including `SINT`.
    Int([i32; 4]),

    /// Value for unsigned integer attachments, including `UINT`.
    Uint([u32; 4]),

    /// Value for depth attachments.
    Depth(f32),

    /// Value for stencil attachments.
    Stencil(u32),

    /// Value for depth and stencil attachments.
    DepthStencil((f32, u32)),
}

impl ClearValue {
    pub(crate) fn clear_value_type(&self) -> ClearValueType {
        match self {
            ClearValue::Float(_) => ClearValueType::Float,
            ClearValue::Int(_) => ClearValueType::Int,
            ClearValue::Uint(_) => ClearValueType::Uint,
            ClearValue::Depth(_) => ClearValueType::Depth,
            ClearValue::Stencil(_) => ClearValueType::Stencil,
            ClearValue::DepthStencil(_) => ClearValueType::DepthStencil,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        if let ClearValue::Depth(depth) | ClearValue::DepthStencil((depth, _)) = self {
            if !(0.0..=1.0).contains(depth)
                && !device.enabled_extensions().ext_depth_range_unrestricted
            {
                return Err(Box::new(ValidationError {
                    problem: "is `ClearValue::Depth` or `ClearValue::DepthStencil`, and \
                        the depth value is not between 0.0 and 1.0 inclusive"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                        "ext_depth_range_unrestricted",
                    )])]),
                    vuids: &["VUID-VkClearDepthStencilValue-depth-00022"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn to_vk(&self) -> vk::ClearValue {
        match *self {
            ClearValue::Float(val) => vk::ClearValue {
                color: vk::ClearColorValue { float32: val },
            },
            ClearValue::Int(val) => vk::ClearValue {
                color: vk::ClearColorValue { int32: val },
            },
            ClearValue::Uint(val) => vk::ClearValue {
                color: vk::ClearColorValue { uint32: val },
            },
            ClearValue::Depth(depth) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
            },
            ClearValue::Stencil(stencil) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil,
                },
            },
            ClearValue::DepthStencil((depth, stencil)) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}

impl From<ClearColorValue> for ClearValue {
    #[inline]
    fn from(val: ClearColorValue) -> Self {
        match val {
            ClearColorValue::Float(val) => Self::Float(val),
            ClearColorValue::Int(val) => Self::Int(val),
            ClearColorValue::Uint(val) => Self::Uint(val),
        }
    }
}

impl From<[f32; 1]> for ClearValue {
    #[inline]
    fn from(val: [f32; 1]) -> Self {
        Self::Float([val[0], 0.0, 0.0, 1.0])
    }
}

impl From<[f32; 2]> for ClearValue {
    #[inline]
    fn from(val: [f32; 2]) -> Self {
        Self::Float([val[0], val[1], 0.0, 1.0])
    }
}

impl From<[f32; 3]> for ClearValue {
    #[inline]
    fn from(val: [f32; 3]) -> Self {
        Self::Float([val[0], val[1], val[2], 1.0])
    }
}

impl From<[f32; 4]> for ClearValue {
    #[inline]
    fn from(val: [f32; 4]) -> Self {
        Self::Float(val)
    }
}

impl From<[u32; 1]> for ClearValue {
    #[inline]
    fn from(val: [u32; 1]) -> Self {
        Self::Uint([val[0], 0, 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 2]> for ClearValue {
    #[inline]
    fn from(val: [u32; 2]) -> Self {
        Self::Uint([val[0], val[1], 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 3]> for ClearValue {
    #[inline]
    fn from(val: [u32; 3]) -> Self {
        Self::Uint([val[0], val[1], val[2], 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[u32; 4]> for ClearValue {
    #[inline]
    fn from(val: [u32; 4]) -> Self {
        Self::Uint(val)
    }
}

impl From<[i32; 1]> for ClearValue {
    #[inline]
    fn from(val: [i32; 1]) -> Self {
        Self::Int([val[0], 0, 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 2]> for ClearValue {
    #[inline]
    fn from(val: [i32; 2]) -> Self {
        Self::Int([val[0], val[1], 0, 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 3]> for ClearValue {
    #[inline]
    fn from(val: [i32; 3]) -> Self {
        Self::Int([val[0], val[1], val[2], 0]) // TODO: is alpha value 0 correct?
    }
}

impl From<[i32; 4]> for ClearValue {
    #[inline]
    fn from(val: [i32; 4]) -> Self {
        Self::Int(val)
    }
}

impl From<f32> for ClearValue {
    #[inline]
    fn from(val: f32) -> Self {
        Self::Depth(val)
    }
}

impl From<u32> for ClearValue {
    #[inline]
    fn from(val: u32) -> Self {
        Self::Stencil(val)
    }
}

impl From<(f32, u32)> for ClearValue {
    #[inline]
    fn from(val: (f32, u32)) -> Self {
        Self::DepthStencil(val)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ClearValueType {
    Float,
    Int,
    Uint,
    Depth,
    Stencil,
    DepthStencil,
}

/// A value that will be used to clear a color image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearColorValue {
    /// Value for formats with a numeric type that is not `SINT` or `UINT`.
    Float([f32; 4]),
    /// Value for formats with a numeric type of `SINT`.
    Int([i32; 4]),
    /// Value for formats with a numeric type of `UINT`.
    Uint([u32; 4]),
}

impl ClearColorValue {
    /// Returns the numeric type of the clear value.
    pub fn numeric_type(&self) -> NumericType {
        match self {
            ClearColorValue::Float(_) => NumericType::Float,
            ClearColorValue::Int(_) => NumericType::Int,
            ClearColorValue::Uint(_) => NumericType::Uint,
        }
    }

    #[allow(clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn to_vk(&self) -> vk::ClearColorValue {
        match *self {
            ClearColorValue::Float(float32) => vk::ClearColorValue { float32 },
            ClearColorValue::Int(int32) => vk::ClearColorValue { int32 },
            ClearColorValue::Uint(uint32) => vk::ClearColorValue { uint32 },
        }
    }
}

impl From<[f32; 1]> for ClearColorValue {
    #[inline]
    fn from(val: [f32; 1]) -> Self {
        Self::Float([val[0], 0.0, 0.0, 1.0])
    }
}

impl From<[f32; 2]> for ClearColorValue {
    #[inline]
    fn from(val: [f32; 2]) -> Self {
        Self::Float([val[0], val[1], 0.0, 1.0])
    }
}

impl From<[f32; 3]> for ClearColorValue {
    #[inline]
    fn from(val: [f32; 3]) -> Self {
        Self::Float([val[0], val[1], val[2], 1.0])
    }
}

impl From<[f32; 4]> for ClearColorValue {
    #[inline]
    fn from(val: [f32; 4]) -> Self {
        Self::Float(val)
    }
}

impl From<[i32; 1]> for ClearColorValue {
    #[inline]
    fn from(val: [i32; 1]) -> Self {
        Self::Int([val[0], 0, 0, 1])
    }
}

impl From<[i32; 2]> for ClearColorValue {
    #[inline]
    fn from(val: [i32; 2]) -> Self {
        Self::Int([val[0], val[1], 0, 1])
    }
}

impl From<[i32; 3]> for ClearColorValue {
    #[inline]
    fn from(val: [i32; 3]) -> Self {
        Self::Int([val[0], val[1], val[2], 1])
    }
}

impl From<[i32; 4]> for ClearColorValue {
    #[inline]
    fn from(val: [i32; 4]) -> Self {
        Self::Int(val)
    }
}

impl From<[u32; 1]> for ClearColorValue {
    #[inline]
    fn from(val: [u32; 1]) -> Self {
        Self::Uint([val[0], 0, 0, 1])
    }
}

impl From<[u32; 2]> for ClearColorValue {
    #[inline]
    fn from(val: [u32; 2]) -> Self {
        Self::Uint([val[0], val[1], 0, 1])
    }
}

impl From<[u32; 3]> for ClearColorValue {
    #[inline]
    fn from(val: [u32; 3]) -> Self {
        Self::Uint([val[0], val[1], val[2], 1])
    }
}

impl From<[u32; 4]> for ClearColorValue {
    #[inline]
    fn from(val: [u32; 4]) -> Self {
        Self::Uint(val)
    }
}

/// A value that will be used to clear a depth/stencil image.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClearDepthStencilValue {
    /// Value for the depth component.
    pub depth: f32,
    /// Value for the stencil component.
    pub stencil: u32,
}

impl ClearDepthStencilValue {
    #[allow(clippy::trivially_copy_pass_by_ref, clippy::wrong_self_convention)]
    #[doc(hidden)]
    pub fn to_vk(&self) -> vk::ClearDepthStencilValue {
        let &Self { depth, stencil } = self;

        vk::ClearDepthStencilValue { depth, stencil }
    }
}

impl From<f32> for ClearDepthStencilValue {
    #[inline]
    fn from(depth: f32) -> Self {
        Self { depth, stencil: 0 }
    }
}

impl From<u32> for ClearDepthStencilValue {
    #[inline]
    fn from(stencil: u32) -> Self {
        Self {
            depth: 0.0,
            stencil,
        }
    }
}

impl From<(f32, u32)> for ClearDepthStencilValue {
    #[inline]
    fn from((depth, stencil): (f32, u32)) -> Self {
        Self { depth, stencil }
    }
}

/// The properties of a format that are supported by a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FormatProperties {
    /// Features available for images with linear tiling.
    pub linear_tiling_features: FormatFeatures,

    /// Features available for images with optimal tiling.
    pub optimal_tiling_features: FormatFeatures,

    /// Features available for buffers.
    pub buffer_features: FormatFeatures,

    /// The properties of the format when combined with a Linux DRM format modifier.
    ///
    /// This will be empty if the [`ext_image_drm_format_modifier`] extension is not supported
    /// by the physical device.
    ///
    /// [`ext_image_drm_format_modifier`]: crate::device::DeviceExtensions::ext_image_drm_format_modifier
    pub drm_format_modifier_properties: Vec<DrmFormatModifierProperties>,
}

impl FormatProperties {
    /// Returns the format features for the specified tiling.
    pub fn format_features(
        &self,
        tiling: ImageTiling,
        drm_format_modifiers: &[u64],
    ) -> FormatFeatures {
        match tiling {
            ImageTiling::Linear => self.linear_tiling_features,
            ImageTiling::Optimal => self.optimal_tiling_features,
            ImageTiling::DrmFormatModifier => self
                .drm_format_modifier_properties
                .iter()
                .filter(|properties| drm_format_modifiers.contains(&properties.drm_format_modifier))
                .fold(FormatFeatures::empty(), |total, properties| {
                    total | properties.drm_format_modifier_tiling_features
                }),
        }
    }

    /// Returns the potential format features, following the definition of
    /// <https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap47.html#potential-format-features>.
    #[inline]
    pub fn potential_format_features(&self) -> FormatFeatures {
        self.linear_tiling_features | self.optimal_tiling_features
    }

    pub(crate) fn to_mut_vk2<'a>(
        extensions_vk: &'a mut FormatProperties2ExtensionsVk<'_>,
    ) -> vk::FormatProperties2<'a> {
        let mut val_vk = vk::FormatProperties2::default();

        let FormatProperties2ExtensionsVk {
            drm_format_modifier_properties_list_vk,
            drm_format_modifier_properties_list2_vk,
            format_properties3_vk,
        } = extensions_vk;

        if let Some(next) = drm_format_modifier_properties_list_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = drm_format_modifier_properties_list2_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = format_properties3_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_mut_vk2_extensions<'a>(
        fields1_vk: &'a mut FormatProperties2Fields1Vk,
        physical_device: &PhysicalDevice,
    ) -> FormatProperties2ExtensionsVk<'a> {
        let FormatProperties2Fields1Vk {
            drm_format_modifier_properties_vk,
            drm_format_modifier_properties2_vk,
        } = fields1_vk;

        let mut val_vk = FormatProperties2ExtensionsVk {
            drm_format_modifier_properties_list_vk: None,
            drm_format_modifier_properties_list2_vk: None,
            format_properties3_vk: None,
        };

        if physical_device.api_version() >= Version::V1_3
            || physical_device
                .supported_extensions()
                .khr_format_feature_flags2
        {
            val_vk.format_properties3_vk = Some(vk::FormatProperties3::default());
            val_vk.drm_format_modifier_properties_list2_vk = drm_format_modifier_properties2_vk
                .as_mut()
                .map(|properties_vk| {
                    vk::DrmFormatModifierPropertiesList2EXT::default()
                        .drm_format_modifier_properties(properties_vk)
                });
        } else {
            val_vk.drm_format_modifier_properties_list_vk = drm_format_modifier_properties_vk
                .as_mut()
                .map(|properties_vk| {
                    vk::DrmFormatModifierPropertiesListEXT::default()
                        .drm_format_modifier_properties(properties_vk)
                });
        }

        val_vk
    }

    pub(crate) fn to_mut_vk2_extensions_query_count(
        physical_device: &PhysicalDevice,
    ) -> Option<FormatProperties2ExtensionsVk<'static>> {
        let must_query_drm_format_modifier_count = physical_device
            .supported_extensions()
            .ext_image_drm_format_modifier;
        let must_query_count = must_query_drm_format_modifier_count;

        must_query_count.then(|| {
            let mut val_vk = FormatProperties2ExtensionsVk {
                drm_format_modifier_properties_list_vk: None,
                drm_format_modifier_properties_list2_vk: None,
                format_properties3_vk: None,
            };

            if must_query_drm_format_modifier_count {
                if physical_device.api_version() >= Version::V1_3
                    || physical_device
                        .supported_extensions()
                        .khr_format_feature_flags2
                {
                    val_vk.drm_format_modifier_properties_list2_vk =
                        Some(vk::DrmFormatModifierPropertiesList2EXT::default());
                } else {
                    val_vk.drm_format_modifier_properties_list_vk =
                        Some(vk::DrmFormatModifierPropertiesListEXT::default());
                }
            }

            val_vk
        })
    }

    pub(crate) fn to_mut_vk2_fields1(
        extensions_vk: Option<FormatProperties2ExtensionsVk<'_>>,
    ) -> FormatProperties2Fields1Vk {
        let mut val_vk = FormatProperties2Fields1Vk {
            drm_format_modifier_properties_vk: None,
            drm_format_modifier_properties2_vk: None,
        };

        if let Some(extensions_vk) = extensions_vk {
            let FormatProperties2ExtensionsVk {
                drm_format_modifier_properties_list_vk,
                drm_format_modifier_properties_list2_vk,
                format_properties3_vk: _,
            } = extensions_vk;

            val_vk.drm_format_modifier_properties_vk = drm_format_modifier_properties_list_vk
                .as_ref()
                .map(|list_vk| {
                    vec![
                        vk::DrmFormatModifierPropertiesEXT::default();
                        list_vk.drm_format_modifier_count as usize
                    ]
                });

            val_vk.drm_format_modifier_properties2_vk = drm_format_modifier_properties_list2_vk
                .as_ref()
                .map(|list_vk| {
                    vec![
                        vk::DrmFormatModifierProperties2EXT::default();
                        list_vk.drm_format_modifier_count as usize
                    ]
                });
        }

        val_vk
    }

    pub(crate) fn from_vk2(
        val_vk: &vk::FormatProperties2<'_>,
        fields1_vk: &FormatProperties2Fields1Vk,
        extensions_vk: &FormatProperties2ExtensionsVk<'_>,
    ) -> Self {
        let FormatProperties2Fields1Vk {
            drm_format_modifier_properties_vk,
            drm_format_modifier_properties2_vk,
        } = fields1_vk;
        let FormatProperties2ExtensionsVk {
            drm_format_modifier_properties_list_vk,
            drm_format_modifier_properties_list2_vk,
            format_properties3_vk,
        } = extensions_vk;

        let mut properties = format_properties3_vk.as_ref().map_or_else(
            || {
                let &vk::FormatProperties2 {
                    format_properties:
                        vk::FormatProperties {
                            linear_tiling_features,
                            optimal_tiling_features,
                            buffer_features,
                        },
                    ..
                } = val_vk;

                Self {
                    linear_tiling_features: linear_tiling_features.into(),
                    optimal_tiling_features: optimal_tiling_features.into(),
                    buffer_features: buffer_features.into(),
                    drm_format_modifier_properties: Vec::new(),
                }
            },
            |val_vk| {
                let &vk::FormatProperties3 {
                    linear_tiling_features,
                    optimal_tiling_features,
                    buffer_features,
                    ..
                } = val_vk;

                Self {
                    linear_tiling_features: linear_tiling_features.into(),
                    optimal_tiling_features: optimal_tiling_features.into(),
                    buffer_features: buffer_features.into(),
                    drm_format_modifier_properties: Vec::new(),
                }
            },
        );

        if let Some((list_vk, properties_vk)) = drm_format_modifier_properties_list2_vk
            .as_ref()
            .zip(drm_format_modifier_properties2_vk.as_ref())
        {
            let &vk::DrmFormatModifierPropertiesList2EXT {
                drm_format_modifier_count,
                ..
            } = list_vk;

            properties = Self {
                drm_format_modifier_properties: properties_vk[..drm_format_modifier_count as usize]
                    .iter()
                    .map(DrmFormatModifierProperties::from_vk2)
                    .collect(),
                ..properties
            };
        } else if let Some((list_vk, properties_vk)) = drm_format_modifier_properties_list_vk
            .as_ref()
            .zip(drm_format_modifier_properties_vk.as_ref())
        {
            let &vk::DrmFormatModifierPropertiesListEXT {
                drm_format_modifier_count,
                ..
            } = list_vk;

            properties = Self {
                drm_format_modifier_properties: properties_vk[..drm_format_modifier_count as usize]
                    .iter()
                    .map(DrmFormatModifierProperties::from_vk)
                    .collect(),
                ..properties
            };
        }

        properties
    }
}

pub(crate) struct FormatProperties2ExtensionsVk<'a> {
    pub(crate) drm_format_modifier_properties_list_vk:
        Option<vk::DrmFormatModifierPropertiesListEXT<'a>>,
    pub(crate) drm_format_modifier_properties_list2_vk:
        Option<vk::DrmFormatModifierPropertiesList2EXT<'a>>,
    pub(crate) format_properties3_vk: Option<vk::FormatProperties3<'static>>,
}

impl FormatProperties2ExtensionsVk<'_> {
    pub(crate) fn unborrow(self) -> FormatProperties2ExtensionsVk<'static> {
        let Self {
            drm_format_modifier_properties_list_vk,
            drm_format_modifier_properties_list2_vk,
            format_properties3_vk,
        } = self;

        let drm_format_modifier_properties_list_vk =
            drm_format_modifier_properties_list_vk.map(|val_vk| {
                vk::DrmFormatModifierPropertiesListEXT {
                    _marker: PhantomData,
                    ..val_vk
                }
            });

        let drm_format_modifier_properties_list2_vk =
            drm_format_modifier_properties_list2_vk.map(|val_vk| {
                vk::DrmFormatModifierPropertiesList2EXT {
                    _marker: PhantomData,
                    ..val_vk
                }
            });

        FormatProperties2ExtensionsVk {
            drm_format_modifier_properties_list_vk,
            drm_format_modifier_properties_list2_vk,
            format_properties3_vk,
        }
    }
}

pub(crate) struct FormatProperties2Fields1Vk {
    pub(crate) drm_format_modifier_properties_vk: Option<Vec<vk::DrmFormatModifierPropertiesEXT>>,
    pub(crate) drm_format_modifier_properties2_vk: Option<Vec<vk::DrmFormatModifierProperties2EXT>>,
}

/// The properties of a format when combined with a Linux DRM format modifier.
#[derive(Clone, Debug)]
pub struct DrmFormatModifierProperties {
    /// The DRM format modifier that the properties apply to.
    pub drm_format_modifier: u64,

    /// The number of memory planes that an image will have if it is created with
    /// `drm_format_modifier` and the queried format.
    pub drm_format_modifier_plane_count: u32,

    /// The format features of the queried format when combined with `drm_format_modifier`.
    pub drm_format_modifier_tiling_features: FormatFeatures,
}

impl DrmFormatModifierProperties {
    pub(crate) fn from_vk2(val_vk: &vk::DrmFormatModifierProperties2EXT) -> Self {
        let &vk::DrmFormatModifierProperties2EXT {
            drm_format_modifier,
            drm_format_modifier_plane_count,
            drm_format_modifier_tiling_features,
        } = val_vk;

        Self {
            drm_format_modifier,
            drm_format_modifier_plane_count,
            drm_format_modifier_tiling_features: drm_format_modifier_tiling_features.into(),
        }
    }

    pub(crate) fn from_vk(val_vk: &vk::DrmFormatModifierPropertiesEXT) -> Self {
        let &vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier,
            drm_format_modifier_plane_count,
            drm_format_modifier_tiling_features,
        } = val_vk;

        Self {
            drm_format_modifier,
            drm_format_modifier_plane_count,
            drm_format_modifier_tiling_features: drm_format_modifier_tiling_features.into(),
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// The features supported by a device for an image or buffer with a particular format.
    FormatFeatures = FormatFeatureFlags2(u64);

    /* Image usage  */

    /// Can be used with a sampled image descriptor.
    SAMPLED_IMAGE = SAMPLED_IMAGE,

    /// Can be used with a storage image descriptor.
    STORAGE_IMAGE = STORAGE_IMAGE,

    /// Can be used with a storage image descriptor with atomic operations in a shader.
    STORAGE_IMAGE_ATOMIC = STORAGE_IMAGE_ATOMIC,

    /// Can be used with a storage image descriptor for reading, without specifying a format on the
    /// image view.
    STORAGE_READ_WITHOUT_FORMAT = STORAGE_READ_WITHOUT_FORMAT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_format_feature_flags2)]),
    ]),

    /// Can be used with a storage image descriptor for writing, without specifying a format on the
    /// image view.
    STORAGE_WRITE_WITHOUT_FORMAT = STORAGE_WRITE_WITHOUT_FORMAT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_format_feature_flags2)]),
    ]),

    /// Can be used with a color attachment in a framebuffer, or with an input attachment
    /// descriptor.
    COLOR_ATTACHMENT = COLOR_ATTACHMENT,

    /// Can be used with a color attachment in a framebuffer with blending, or with an input
    /// attachment descriptor.
    COLOR_ATTACHMENT_BLEND = COLOR_ATTACHMENT_BLEND,

    /// Can be used with a depth/stencil attachment in a framebuffer, or with an input attachment
    /// descriptor.
    DEPTH_STENCIL_ATTACHMENT = DEPTH_STENCIL_ATTACHMENT,

    /// Can be used with a fragment density map attachment in a framebuffer.
    FRAGMENT_DENSITY_MAP = FRAGMENT_DENSITY_MAP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_fragment_density_map)]),
    ]),

    /// Can be used with a fragment shading rate attachment in a framebuffer.
    FRAGMENT_SHADING_RATE_ATTACHMENT = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),

    /// Can be used with the source image in a transfer (copy) operation.
    TRANSFER_SRC = TRANSFER_SRC
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance1)]),
    ]),

    /// Can be used with the destination image in a transfer (copy) operation.
    TRANSFER_DST = TRANSFER_DST
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance1)]),
    ]),

    /// Can be used with the source image in a blit operation.
    BLIT_SRC = BLIT_SRC,

    /// Can be used with the destination image in a blit operation.
    BLIT_DST = BLIT_DST,

    /* Sampling  */

    /// Can be used with samplers or as a blit source, using the
    /// [`Linear`](crate::image::sampler::Filter::Linear) filter.
    SAMPLED_IMAGE_FILTER_LINEAR = SAMPLED_IMAGE_FILTER_LINEAR,

    /// Can be used with samplers or as a blit source, using the
    /// [`Cubic`](crate::image::sampler::Filter::Cubic) filter.
    SAMPLED_IMAGE_FILTER_CUBIC = SAMPLED_IMAGE_FILTER_CUBIC_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_filter_cubic)]),
        RequiresAllOf([DeviceExtension(img_filter_cubic)]),
    ]),

    /// Can be used with samplers using a reduction mode of
    /// [`Min`](crate::image::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::image::sampler::SamplerReductionMode::Max).
    SAMPLED_IMAGE_FILTER_MINMAX = SAMPLED_IMAGE_FILTER_MINMAX
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(ext_sampler_filter_minmax)]),
    ]),

    /// Can be used with sampler YCbCr conversions using a chroma offset of
    /// [`Midpoint`](crate::image::sampler::ycbcr::ChromaLocation::Midpoint).
    MIDPOINT_CHROMA_SAMPLES = MIDPOINT_CHROMA_SAMPLES
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// Can be used with sampler YCbCr conversions using a chroma offset of
    /// [`CositedEven`](crate::image::sampler::ycbcr::ChromaLocation::CositedEven).
    COSITED_CHROMA_SAMPLES = COSITED_CHROMA_SAMPLES
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// Can be used with sampler YCbCr conversions using the
    /// [`Linear`](crate::image::sampler::Filter::Linear) chroma filter.
    SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER = SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// Can be used with sampler YCbCr conversions whose chroma filter differs from the filters of
    /// the base sampler.
    SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER = SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// When used with a sampler YCbCr conversion, the implementation will always perform
    /// explicit chroma reconstruction.
    SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT = SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// Can be used with sampler YCbCr conversions with forced explicit reconstruction.
    SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE = SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// Can be used with samplers using depth comparison.
    SAMPLED_IMAGE_DEPTH_COMPARISON = SAMPLED_IMAGE_DEPTH_COMPARISON
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_format_feature_flags2)]),
    ]),

    /* Video */

    /// Can be used with the output image of a video decode operation.
    VIDEO_DECODE_OUTPUT = VIDEO_DECODE_OUTPUT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),

    /// Can be used with the DPB image of a video decode operation.
    VIDEO_DECODE_DPB = VIDEO_DECODE_DPB_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),

    /// Can be used with the input image of a video encode operation.
    VIDEO_ENCODE_INPUT = VIDEO_ENCODE_INPUT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),

    /// Can be used with the DPB image of a video encode operation.
    VIDEO_ENCODE_DPB = VIDEO_ENCODE_DPB_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),

    /* Misc image features */

    /// For multi-planar formats, can be used with images created with the [`DISJOINT`] flag.
    ///
    /// [`DISJOINT`]: crate::image::ImageCreateFlags::DISJOINT
    DISJOINT = DISJOINT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    // TODO: document
    LINEAR_COLOR_ATTACHMENT = LINEAR_COLOR_ATTACHMENT_NV
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(nv_linear_color_attachment),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_format_feature_flags2),
            DeviceExtension(nv_linear_color_attachment),
        ]),
    ]),

    // TODO: document
    WEIGHT_IMAGE = WEIGHT_IMAGE_QCOM
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(qcom_image_processing),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_format_feature_flags2),
            DeviceExtension(qcom_image_processing),
        ]),
    ]),

    // TODO: document
    WEIGHT_SAMPLED_IMAGE = WEIGHT_SAMPLED_IMAGE_QCOM
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(qcom_image_processing),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_format_feature_flags2),
            DeviceExtension(qcom_image_processing),
        ]),
    ]),

    // TODO: document
    BLOCK_MATCHING = BLOCK_MATCHING_QCOM
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(qcom_image_processing),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_format_feature_flags2),
            DeviceExtension(qcom_image_processing),
        ]),
    ]),

    // TODO: document
    BOX_FILTER_SAMPLED = BOX_FILTER_SAMPLED_QCOM
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(qcom_image_processing),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_format_feature_flags2),
            DeviceExtension(qcom_image_processing),
        ]),
    ]),

    // TODO: document
    OPTICAL_FLOW_IMAGE = OPTICAL_FLOW_IMAGE_NV
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(nv_optical_flow),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_format_feature_flags2),
            DeviceExtension(nv_optical_flow),
        ]),
    ]),

    // TODO: document
    OPTICAL_FLOW_VECTOR = OPTICAL_FLOW_VECTOR_NV
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(nv_optical_flow),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_format_feature_flags2),
            DeviceExtension(nv_optical_flow),
        ]),
    ]),

    // TODO: document
    OPTICAL_FLOW_COST = OPTICAL_FLOW_COST_NV
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(nv_optical_flow),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_format_feature_flags2),
            DeviceExtension(nv_optical_flow),
        ]),
    ]),

    /* Buffer usage  */

    /// Can be used with a uniform texel buffer descriptor.
    UNIFORM_TEXEL_BUFFER = UNIFORM_TEXEL_BUFFER,

    /// Can be used with a storage texel buffer descriptor.
    STORAGE_TEXEL_BUFFER = STORAGE_TEXEL_BUFFER,

    /// Can be used with a storage texel buffer descriptor with atomic operations in a shader.
    STORAGE_TEXEL_BUFFER_ATOMIC = STORAGE_TEXEL_BUFFER_ATOMIC,

    /// Can be used as the format of a vertex attribute in the vertex input state of a graphics
    /// pipeline.
    VERTEX_BUFFER = VERTEX_BUFFER,

    /// Can be used as the vertex format when building an acceleration structure.
    ACCELERATION_STRUCTURE_VERTEX_BUFFER = ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
    ]),
}

impl From<vk::FormatFeatureFlags> for FormatFeatures {
    #[inline]
    fn from(val: vk::FormatFeatureFlags) -> Self {
        Self::from(vk::FormatFeatureFlags2::from_raw(val.as_raw() as u64))
    }
}
