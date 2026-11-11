use super::ImageAspect;
use crate::macros::vulkan_enum;

vulkan_enum! {
    #[non_exhaustive]

    /// In-memory layout of the pixel data of an image.
    ///
    /// The pixel data of a Vulkan image is arranged in a particular way, which is called its
    /// *layout*. Each image subresource (mipmap level and array layer) in an image can have a
    /// different layout, but usually the whole image has its data in the same layout. Layouts are
    /// abstract in the sense that the user does not know the specific details of each layout; the
    /// device driver is free to implement each layout in the way it sees fit.
    ///
    /// The layout of a newly created image is either `Undefined` or `Preinitialized`. Every
    /// operation that can be performed on an image is only possible with specific layouts, so
    /// before the operation is performed, the user must perform a *layout transition* on the
    /// image. This rearranges the pixel data from one layout into another. Layout transitions are
    /// performed as part of pipeline barriers in a command buffer.
    ///
    /// The `General` layout is compatible with any operation, so layout transitions are never
    /// needed. However, the other layouts, while more restricted, are usually better optimized for
    /// a particular type of operation than `General`, so they are usually preferred.
    ///
    /// Vulkan does not keep track of layouts itself, so it is the responsibility of the user to
    /// keep track of this information. When performing a layout transition, the previous layout
    /// must be specified as well. Some operations allow for different layouts, but require the
    /// user to specify which one. Vulkano helps with this by providing sensible defaults,
    /// automatically tracking the layout of each image when creating a command buffer, and adding
    /// layout transitions where needed.
    ImageLayout = ImageLayout(i32);

    /// The layout of the data is unknown, and the image is treated as containing no valid data.
    /// Transitioning from `Undefined` will discard any existing pixel data.
    Undefined = UNDEFINED,

    /// A general-purpose layout that can be used for any operation. Some operations may only allow
    /// `General`, such as storage images, but many have a more specific layout that is better
    /// optimized for that purpose.
    General = GENERAL,

    /// For a color image used as a color or resolve attachment in a framebuffer. Images that are
    /// transitioned into this layout must have the `color_attachment` usage enabled.
    ColorAttachmentOptimal = COLOR_ATTACHMENT_OPTIMAL,

    /// A combination of `DepthAttachmentOptimal` for the depth aspect of the image,
    /// and `StencilAttachmentOptimal` for the stencil aspect of the image.
    DepthStencilAttachmentOptimal = DEPTH_STENCIL_ATTACHMENT_OPTIMAL,

    /// A combination of `DepthReadOnlyOptimal` for the depth aspect of the image,
    /// and `StencilReadOnlyOptimal` for the stencil aspect of the image.
    DepthStencilReadOnlyOptimal = DEPTH_STENCIL_READ_ONLY_OPTIMAL,

    /// For a color image used as a (combined) sampled image or input attachment in a shader.
    /// Images that are transitioned into this layout must have the `sampled` or `input_attachment`
    /// usages enabled.
    ShaderReadOnlyOptimal = SHADER_READ_ONLY_OPTIMAL,

    /// For operations that transfer data from an image (copy, blit).
    TransferSrcOptimal = TRANSFER_SRC_OPTIMAL,

    /// For operations that transfer data to an image (copy, blit, clear).
    TransferDstOptimal = TRANSFER_DST_OPTIMAL,

    /// When creating an image, this specifies that the initial data is going to be directly
    /// written to from the CPU. Unlike `Undefined`, the image is assumed to contain valid data when
    /// transitioning from this layout. However, this only works right when the image has linear
    /// tiling, optimal tiling gives undefined results.
    Preinitialized = PREINITIALIZED,

    /// A combination of `DepthReadOnlyOptimal` for the depth aspect of the image,
    /// and `StencilAttachmentOptimal` for the stencil aspect of the image.
    DepthReadOnlyStencilAttachmentOptimal = DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance2)]),
    ]),

    /// A combination of `DepthAttachmentOptimal` for the depth aspect of the image,
    /// and `StencilReadOnlyOptimal` for the stencil aspect of the image.
    DepthAttachmentStencilReadOnlyOptimal = DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_maintenance2)]),
    ]),

    /// For a depth image used as a depth attachment in a framebuffer.
    DepthAttachmentOptimal = DEPTH_ATTACHMENT_OPTIMAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(khr_separate_depth_stencil_layouts)]),
    ]),

    /// For a depth image used as a read-only depth attachment in a framebuffer, or
    /// as a (combined) sampled image or input attachment in a shader.
    DepthReadOnlyOptimal = DEPTH_READ_ONLY_OPTIMAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(khr_separate_depth_stencil_layouts)]),
    ]),

    /// For a stencil image used as a stencil attachment in a framebuffer.
    StencilAttachmentOptimal = STENCIL_ATTACHMENT_OPTIMAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(khr_separate_depth_stencil_layouts)]),
    ]),

    /// For a stencil image used as a read-only stencil attachment in a framebuffer, or
    /// as a (combined) sampled image or input attachment in a shader.
    StencilReadOnlyOptimal = STENCIL_READ_ONLY_OPTIMAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(khr_separate_depth_stencil_layouts)]),
    ]),

    /* TODO: enable
    // TODO: document
    ReadOnlyOptimal = READ_ONLY_OPTIMAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    AttachmentOptimal = ATTACHMENT_OPTIMAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),*/

    /// The layout of images that are held in a swapchain. Images are in this layout when they are
    /// acquired from the swapchain, and must be transitioned back into this layout before
    /// presenting them.
    PresentSrc = PRESENT_SRC_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_swapchain)]),
    ]),

    /* TODO: enable
    // TODO: document
    VideoDecodeDst = VIDEO_DECODE_DST_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VideoDecodeSrc = VIDEO_DECODE_SRC_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VideoDecodeDpb = VIDEO_DECODE_DPB_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SharedPresent = SHARED_PRESENT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_shared_presentable_image)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    FragmentDensityMapOptimal = FRAGMENT_DENSITY_MAP_OPTIMAL_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_fragment_density_map)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    FragmentShadingRateAttachmentOptimal = FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VideoEncodeDst = VIDEO_ENCODE_DST_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VideoEncodeSrc = VIDEO_ENCODE_SRC_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VideoEncodeDpb = VIDEO_ENCODE_DPB_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    AttachmentFeedbackLoopOptimal = ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_attachment_feedback_loop_layout)]),
    ]),*/
}

impl Default for ImageLayout {
    #[inline]
    fn default() -> Self {
        ImageLayout::Undefined
    }
}

impl ImageLayout {
    /// If the layout can be used for `aspect`, returns whether `aspect` can be written to if an
    /// image is in that layout.
    pub fn is_writable(self, aspect: ImageAspect) -> bool {
        match aspect {
            ImageAspect::Color
            | ImageAspect::Plane0
            | ImageAspect::Plane1
            | ImageAspect::Plane2 => match self {
                ImageLayout::General
                | ImageLayout::ColorAttachmentOptimal
                | ImageLayout::TransferDstOptimal => true,
                ImageLayout::Undefined
                | ImageLayout::DepthStencilAttachmentOptimal
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::ShaderReadOnlyOptimal
                | ImageLayout::TransferSrcOptimal
                | ImageLayout::Preinitialized
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                | ImageLayout::DepthAttachmentOptimal
                | ImageLayout::DepthReadOnlyOptimal
                | ImageLayout::StencilAttachmentOptimal
                | ImageLayout::StencilReadOnlyOptimal
                | ImageLayout::PresentSrc => false,
            },
            ImageAspect::Depth => match self {
                ImageLayout::General
                | ImageLayout::DepthStencilAttachmentOptimal
                | ImageLayout::TransferDstOptimal
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                | ImageLayout::DepthAttachmentOptimal => true,

                ImageLayout::Undefined
                | ImageLayout::ColorAttachmentOptimal
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::ShaderReadOnlyOptimal
                | ImageLayout::TransferSrcOptimal
                | ImageLayout::Preinitialized
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::DepthReadOnlyOptimal
                | ImageLayout::StencilAttachmentOptimal
                | ImageLayout::StencilReadOnlyOptimal
                | ImageLayout::PresentSrc => false,
            },
            ImageAspect::Stencil => match self {
                ImageLayout::General
                | ImageLayout::DepthStencilAttachmentOptimal
                | ImageLayout::TransferDstOptimal
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::StencilAttachmentOptimal => true,

                ImageLayout::Undefined
                | ImageLayout::ColorAttachmentOptimal
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::ShaderReadOnlyOptimal
                | ImageLayout::TransferSrcOptimal
                | ImageLayout::Preinitialized
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                | ImageLayout::DepthAttachmentOptimal
                | ImageLayout::DepthReadOnlyOptimal
                | ImageLayout::StencilReadOnlyOptimal
                | ImageLayout::PresentSrc => false,
            },
            ImageAspect::Metadata
            | ImageAspect::MemoryPlane0
            | ImageAspect::MemoryPlane1
            | ImageAspect::MemoryPlane2
            | ImageAspect::MemoryPlane3 => false,
        }
    }
}
