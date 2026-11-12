use crate::macros::vulkan_bitflags;

vulkan_bitflags! {
    #[non_exhaustive]

    /// Describes how a buffer is going to be used. This is **not** just an optimization.
    ///
    /// If you try to use a buffer in a way that you didn't declare, an error will be returned.
    BufferUsage = BufferUsageFlags(u32);

    /// The buffer can be used as a source for transfer, blit, resolve and clear commands.
    TRANSFER_SRC = TRANSFER_SRC,

    /// The buffer can be used as a destination for transfer, blit, resolve and clear commands.
    TRANSFER_DST = TRANSFER_DST,

    /// The buffer can be used as a uniform texel buffer in a descriptor set.
    UNIFORM_TEXEL_BUFFER = UNIFORM_TEXEL_BUFFER,

    /// The buffer can be used as a storage texel buffer in a descriptor set.
    STORAGE_TEXEL_BUFFER = STORAGE_TEXEL_BUFFER,

    /// The buffer can be used as a uniform buffer in a descriptor set.
    UNIFORM_BUFFER = UNIFORM_BUFFER,

    /// The buffer can be used as a storage buffer in a descriptor set.
    STORAGE_BUFFER = STORAGE_BUFFER,

    /// The buffer can be used as an index buffer.
    INDEX_BUFFER = INDEX_BUFFER,

    /// The buffer can be used as a vertex or instance buffer.
    VERTEX_BUFFER = VERTEX_BUFFER,

    /// The buffer can be used as an indirect buffer.
    INDIRECT_BUFFER = INDIRECT_BUFFER,

    /// The buffer's device address can be retrieved.
    ///
    /// A buffer created with this usage can only be bound to device memory allocated with the
    /// [`MemoryAllocateFlags::DEVICE_ADDRESS`] flag, unless the [`ext_buffer_device_address`]
    /// extension is enabled on the device.
    ///
    /// [`MemoryAllocateFlags::DEVICE_ADDRESS`]: crate::memory::MemoryAllocateFlags::DEVICE_ADDRESS
    /// [`ext_buffer_device_address`]: crate::device::DeviceExtensions::ext_buffer_device_address
    SHADER_DEVICE_ADDRESS = SHADER_DEVICE_ADDRESS
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(khr_buffer_device_address)]),
        RequiresAllOf([DeviceExtension(ext_buffer_device_address)]),
    ]),

    /* TODO: enable
    // TODO: document
    VIDEO_DECODE_SRC = VIDEO_DECODE_SRC_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    VIDEO_DECODE_DST = VIDEO_DECODE_DST_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    TRANSFORM_FEEDBACK_BUFFER = TRANSFORM_FEEDBACK_BUFFER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    TRANSFORM_FEEDBACK_COUNTER_BUFFER = TRANSFORM_FEEDBACK_COUNTER_BUFFER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    CONDITIONAL_RENDERING = CONDITIONAL_RENDERING_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_conditional_rendering)]),
    ]),*/

    /// The buffer can be used as input data for an acceleration structure build operation.
    ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY = ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
    ]),

    /// An acceleration structure can be created from the buffer.
    ACCELERATION_STRUCTURE_STORAGE = ACCELERATION_STRUCTURE_STORAGE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
    ]),


    // TODO: document
    SHADER_BINDING_TABLE = SHADER_BINDING_TABLE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

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
    SAMPLER_DESCRIPTOR_BUFFER = SAMPLER_DESCRIPTOR_BUFFER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RESOURCE_DESCRIPTOR_BUFFER = RESOURCE_DESCRIPTOR_BUFFER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER = PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    MICROMAP_BUILD_INPUT_READ_ONLY = MICROMAP_BUILD_INPUT_READ_ONLY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    MICROMAP_STORAGE = MICROMAP_STORAGE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]),*/
}
