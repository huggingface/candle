use crate::macros::vulkan_bitflags;

vulkan_bitflags! {
    #[non_exhaustive]

    /// Describes how an image is going to be used. This is **not** just an optimization.
    ///
    /// If you try to use an image in a way that you didn't declare, an error will occur.
    ImageUsage = ImageUsageFlags(u32);

    /// The image can be used as a source for transfer, blit, resolve and clear commands.
    TRANSFER_SRC = TRANSFER_SRC,

    /// The image can be used as a destination for transfer, blit, resolve and clear commands.
    TRANSFER_DST = TRANSFER_DST,

    /// The image can be used as a sampled image in a shader.
    SAMPLED = SAMPLED,

    /// The image can be used as a storage image in a shader.
    STORAGE = STORAGE,

    /// The image can be used as a color attachment in a render pass/framebuffer.
    COLOR_ATTACHMENT = COLOR_ATTACHMENT,

    /// The image can be used as a depth/stencil attachment in a render pass/framebuffer.
    DEPTH_STENCIL_ATTACHMENT = DEPTH_STENCIL_ATTACHMENT,

    /// The image will be used as an attachment, and will only ever be used temporarily.
    /// As soon as you leave a render pass, the content of transient images becomes undefined.
    ///
    /// This is a hint to the Vulkan implementation that it may not need allocate any memory for
    /// this image if the image can live entirely in some cache.
    ///
    /// If `transient_attachment` is true, then only `color_attachment`, `depth_stencil_attachment`
    /// and `input_attachment` can be true as well. The rest must be false or an error will be
    /// returned when creating the image.
    TRANSIENT_ATTACHMENT = TRANSIENT_ATTACHMENT,

    /// The image can be used as an input attachment in a render pass/framebuffer.
    INPUT_ATTACHMENT = INPUT_ATTACHMENT,

    /* TODO: enable
    // TODO: document
    VIDEO_DECODE_DST = VIDEO_DECODE_DST_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VIDEO_DECODE_SRC = VIDEO_DECODE_SRC_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VIDEO_DECODE_DPB = VIDEO_DECODE_DPB_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    FRAGMENT_DENSITY_MAP = FRAGMENT_DENSITY_MAP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_fragment_density_map)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    FRAGMENT_SHADING_RATE_ATTACHMENT = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VIDEO_ENCODE_DST = VIDEO_ENCODE_DST_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VIDEO_ENCODE_SRC = VIDEO_ENCODE_SRC_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VIDEO_ENCODE_DPB = VIDEO_ENCODE_DPB_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    ATTACHMENT_FEEDBACK_LOOP = ATTACHMENT_FEEDBACK_LOOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_attachment_feedback_loop_layout)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    INVOCATION_MASK = INVOCATION_MASK_HUAWEI
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(huawei_invocation_mask)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SAMPLE_WEIGHT = SAMPLE_WEIGHT_QCOM
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(qcom_image_processing)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SAMPLE_BLOCK_MATCH = SAMPLE_BLOCK_MATCH_QCOM
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(qcom_image_processing)]),
    ]),*/
}
