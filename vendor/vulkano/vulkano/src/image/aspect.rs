use crate::macros::vulkan_bitflags_enum;

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`ImageAspect`] values.
    ImageAspects,

    /// An individual data type within an image.
    ///
    /// Most images have only the [`Color`] aspect, but some may have others.
    ///
    /// [`Color`]: ImageAspect::Color
    ImageAspect,

    = ImageAspectFlags(u32);

    /// The single aspect of images with a color [format], or the combined aspect of all planes of
    /// images with a multi-planar format.
    ///
    /// [format]: crate::format::Format
    COLOR, Color = COLOR,

    /// The single aspect of images with a depth [format], or one of the two aspects of images
    /// with a combined depth/stencil format.
    ///
    /// [format]: crate::format::Format
    DEPTH, Depth = DEPTH,

    /// The single aspect of images with a stencil [format], or one of the two aspects of images
    /// with a combined depth/stencil format.
    ///
    /// [format]: crate::format::Format
    STENCIL, Stencil = STENCIL,

    /// An aspect used with sparse memory on some implementations, to hold implementation-defined
    /// metadata of an image.
    METADATA, Metadata = METADATA,

    /// The first plane of an image with a multi-planar [format], holding the green color component.
    ///
    /// [format]: crate::format::Format
    PLANE_0, Plane0 = PLANE_0
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// The second plane of an image with a multi-planar [format], holding the blue color component
    /// if the format has three planes, and a combination of blue and red if the format has two
    /// planes.
    ///
    /// [format]: crate::format::Format
    PLANE_1, Plane1 = PLANE_1
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// The third plane of an image with a multi-planar [format], holding the red color component.
    PLANE_2, Plane2 = PLANE_2
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_sampler_ycbcr_conversion)]),
    ]),

    /// The first memory plane of images created through the [`ext_image_drm_format_modifier`]
    /// extension.
    ///
    /// [`ext_image_drm_format_modifier`]: crate::device::DeviceExtensions::ext_image_drm_format_modifier
    MEMORY_PLANE_0, MemoryPlane0 = MEMORY_PLANE_0_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_image_drm_format_modifier)]),
    ]),

    /// The second memory plane of images created through the [`ext_image_drm_format_modifier`]
    /// extension.
    ///
    /// [`ext_image_drm_format_modifier`]: crate::device::DeviceExtensions::ext_image_drm_format_modifier
    MEMORY_PLANE_1, MemoryPlane1 = MEMORY_PLANE_1_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_image_drm_format_modifier)]),
    ]),

    /// The third memory plane of images created through the [`ext_image_drm_format_modifier`]
    /// extension.
    ///
    /// [`ext_image_drm_format_modifier`]: crate::device::DeviceExtensions::ext_image_drm_format_modifier
    MEMORY_PLANE_2, MemoryPlane2 = MEMORY_PLANE_2_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_image_drm_format_modifier)]),
    ]),

    /// The fourth memory plane of images created through the [`ext_image_drm_format_modifier`]
    /// extension.
    ///
    /// [`ext_image_drm_format_modifier`]: crate::device::DeviceExtensions::ext_image_drm_format_modifier
    MEMORY_PLANE_3, MemoryPlane3 = MEMORY_PLANE_3_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_image_drm_format_modifier)]),
    ]),
}
