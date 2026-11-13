use super::definitions::*;
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCacheCreateFlagBits.html>"]
pub struct PipelineCacheCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineCacheCreateFlags, Flags);
impl PipelineCacheCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkQueueFlagBits.html>"]
pub struct QueueFlags(pub(crate) Flags);
vk_bitflags_wrapped!(QueueFlags, Flags);
impl QueueFlags {
    #[doc = "Queue supports graphics operations"]
    pub const GRAPHICS: Self = Self(0b1);
    #[doc = "Queue supports compute operations"]
    pub const COMPUTE: Self = Self(0b10);
    #[doc = "Queue supports transfer operations"]
    pub const TRANSFER: Self = Self(0b100);
    #[doc = "Queue supports sparse resource memory management operations"]
    pub const SPARSE_BINDING: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkCullModeFlagBits.html>"]
pub struct CullModeFlags(pub(crate) Flags);
vk_bitflags_wrapped!(CullModeFlags, Flags);
impl CullModeFlags {
    pub const NONE: Self = Self(0);
    pub const FRONT: Self = Self(0b1);
    pub const BACK: Self = Self(0b10);
    pub const FRONT_AND_BACK: Self = Self(0x00000003);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderPassCreateFlagBits.html>"]
pub struct RenderPassCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(RenderPassCreateFlags, Flags);
impl RenderPassCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceQueueCreateFlagBits.html>"]
pub struct DeviceQueueCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(DeviceQueueCreateFlags, Flags);
impl DeviceQueueCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryPropertyFlagBits.html>"]
pub struct MemoryPropertyFlags(pub(crate) Flags);
vk_bitflags_wrapped!(MemoryPropertyFlags, Flags);
impl MemoryPropertyFlags {
    #[doc = "If otherwise stated, then allocate memory on device"]
    pub const DEVICE_LOCAL: Self = Self(0b1);
    #[doc = "Memory is mappable by host"]
    pub const HOST_VISIBLE: Self = Self(0b10);
    #[doc = "Memory will have i/o coherency. If not set, application may need to use vkFlushMappedMemoryRanges and vkInvalidateMappedMemoryRanges to flush/invalidate host cache"]
    pub const HOST_COHERENT: Self = Self(0b100);
    #[doc = "Memory will be cached by the host"]
    pub const HOST_CACHED: Self = Self(0b1000);
    #[doc = "Memory may be allocated by the driver when it is required"]
    pub const LAZILY_ALLOCATED: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryHeapFlagBits.html>"]
pub struct MemoryHeapFlags(pub(crate) Flags);
vk_bitflags_wrapped!(MemoryHeapFlags, Flags);
impl MemoryHeapFlags {
    #[doc = "If set, heap represents device memory"]
    pub const DEVICE_LOCAL: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkAccessFlagBits.html>"]
pub struct AccessFlags(pub(crate) Flags);
vk_bitflags_wrapped!(AccessFlags, Flags);
impl AccessFlags {
    #[doc = "Controls coherency of indirect command reads"]
    pub const INDIRECT_COMMAND_READ: Self = Self(0b1);
    #[doc = "Controls coherency of index reads"]
    pub const INDEX_READ: Self = Self(0b10);
    #[doc = "Controls coherency of vertex attribute reads"]
    pub const VERTEX_ATTRIBUTE_READ: Self = Self(0b100);
    #[doc = "Controls coherency of uniform buffer reads"]
    pub const UNIFORM_READ: Self = Self(0b1000);
    #[doc = "Controls coherency of input attachment reads"]
    pub const INPUT_ATTACHMENT_READ: Self = Self(0b1_0000);
    #[doc = "Controls coherency of shader reads"]
    pub const SHADER_READ: Self = Self(0b10_0000);
    #[doc = "Controls coherency of shader writes"]
    pub const SHADER_WRITE: Self = Self(0b100_0000);
    #[doc = "Controls coherency of color attachment reads"]
    pub const COLOR_ATTACHMENT_READ: Self = Self(0b1000_0000);
    #[doc = "Controls coherency of color attachment writes"]
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(0b1_0000_0000);
    #[doc = "Controls coherency of depth/stencil attachment reads"]
    pub const DEPTH_STENCIL_ATTACHMENT_READ: Self = Self(0b10_0000_0000);
    #[doc = "Controls coherency of depth/stencil attachment writes"]
    pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Self = Self(0b100_0000_0000);
    #[doc = "Controls coherency of transfer reads"]
    pub const TRANSFER_READ: Self = Self(0b1000_0000_0000);
    #[doc = "Controls coherency of transfer writes"]
    pub const TRANSFER_WRITE: Self = Self(0b1_0000_0000_0000);
    #[doc = "Controls coherency of host reads"]
    pub const HOST_READ: Self = Self(0b10_0000_0000_0000);
    #[doc = "Controls coherency of host writes"]
    pub const HOST_WRITE: Self = Self(0b100_0000_0000_0000);
    #[doc = "Controls coherency of memory reads"]
    pub const MEMORY_READ: Self = Self(0b1000_0000_0000_0000);
    #[doc = "Controls coherency of memory writes"]
    pub const MEMORY_WRITE: Self = Self(0b1_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferUsageFlagBits.html>"]
pub struct BufferUsageFlags(pub(crate) Flags);
vk_bitflags_wrapped!(BufferUsageFlags, Flags);
impl BufferUsageFlags {
    #[doc = "Can be used as a source of transfer operations"]
    pub const TRANSFER_SRC: Self = Self(0b1);
    #[doc = "Can be used as a destination of transfer operations"]
    pub const TRANSFER_DST: Self = Self(0b10);
    #[doc = "Can be used as TBO"]
    pub const UNIFORM_TEXEL_BUFFER: Self = Self(0b100);
    #[doc = "Can be used as IBO"]
    pub const STORAGE_TEXEL_BUFFER: Self = Self(0b1000);
    #[doc = "Can be used as UBO"]
    pub const UNIFORM_BUFFER: Self = Self(0b1_0000);
    #[doc = "Can be used as SSBO"]
    pub const STORAGE_BUFFER: Self = Self(0b10_0000);
    #[doc = "Can be used as source of fixed-function index fetch (index buffer)"]
    pub const INDEX_BUFFER: Self = Self(0b100_0000);
    #[doc = "Can be used as source of fixed-function vertex fetch (VBO)"]
    pub const VERTEX_BUFFER: Self = Self(0b1000_0000);
    #[doc = "Can be the source of indirect parameters (e.g. indirect buffer, parameter buffer)"]
    pub const INDIRECT_BUFFER: Self = Self(0b1_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferUsageFlagBits2.html>"]
pub struct BufferUsageFlags2(pub(crate) Flags64);
vk_bitflags_wrapped!(BufferUsageFlags2, Flags64);
impl BufferUsageFlags2 {
    pub const TRANSFER_SRC: Self = Self(0b1);
    pub const TRANSFER_DST: Self = Self(0b10);
    pub const UNIFORM_TEXEL_BUFFER: Self = Self(0b100);
    pub const STORAGE_TEXEL_BUFFER: Self = Self(0b1000);
    pub const UNIFORM_BUFFER: Self = Self(0b1_0000);
    pub const STORAGE_BUFFER: Self = Self(0b10_0000);
    pub const INDEX_BUFFER: Self = Self(0b100_0000);
    pub const VERTEX_BUFFER: Self = Self(0b1000_0000);
    pub const INDIRECT_BUFFER: Self = Self(0b1_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferCreateFlagBits.html>"]
pub struct BufferCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(BufferCreateFlags, Flags);
impl BufferCreateFlags {
    #[doc = "Buffer should support sparse backing"]
    pub const SPARSE_BINDING: Self = Self(0b1);
    #[doc = "Buffer should support sparse backing with partial residency"]
    pub const SPARSE_RESIDENCY: Self = Self(0b10);
    #[doc = "Buffer should support constant data access to physical memory ranges mapped into multiple locations of sparse buffers"]
    pub const SPARSE_ALIASED: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderStageFlagBits.html>"]
pub struct ShaderStageFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ShaderStageFlags, Flags);
impl ShaderStageFlags {
    pub const VERTEX: Self = Self(0b1);
    pub const TESSELLATION_CONTROL: Self = Self(0b10);
    pub const TESSELLATION_EVALUATION: Self = Self(0b100);
    pub const GEOMETRY: Self = Self(0b1000);
    pub const FRAGMENT: Self = Self(0b1_0000);
    pub const COMPUTE: Self = Self(0b10_0000);
    pub const ALL_GRAPHICS: Self = Self(0x0000001f);
    pub const ALL: Self = Self(0x7fffffff);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkImageUsageFlagBits.html>"]
pub struct ImageUsageFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ImageUsageFlags, Flags);
impl ImageUsageFlags {
    #[doc = "Can be used as a source of transfer operations"]
    pub const TRANSFER_SRC: Self = Self(0b1);
    #[doc = "Can be used as a destination of transfer operations"]
    pub const TRANSFER_DST: Self = Self(0b10);
    #[doc = "Can be sampled from (SAMPLED_IMAGE and COMBINED_IMAGE_SAMPLER descriptor types)"]
    pub const SAMPLED: Self = Self(0b100);
    #[doc = "Can be used as storage image (STORAGE_IMAGE descriptor type)"]
    pub const STORAGE: Self = Self(0b1000);
    #[doc = "Can be used as framebuffer color attachment"]
    pub const COLOR_ATTACHMENT: Self = Self(0b1_0000);
    #[doc = "Can be used as framebuffer depth/stencil attachment"]
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(0b10_0000);
    #[doc = "Image data not needed outside of rendering"]
    pub const TRANSIENT_ATTACHMENT: Self = Self(0b100_0000);
    #[doc = "Can be used as framebuffer input attachment"]
    pub const INPUT_ATTACHMENT: Self = Self(0b1000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCreateFlagBits.html>"]
pub struct ImageCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ImageCreateFlags, Flags);
impl ImageCreateFlags {
    #[doc = "Image should support sparse backing"]
    pub const SPARSE_BINDING: Self = Self(0b1);
    #[doc = "Image should support sparse backing with partial residency"]
    pub const SPARSE_RESIDENCY: Self = Self(0b10);
    #[doc = "Image should support constant data access to physical memory ranges mapped into multiple locations of sparse images"]
    pub const SPARSE_ALIASED: Self = Self(0b100);
    #[doc = "Allows image views to have different format than the base image"]
    pub const MUTABLE_FORMAT: Self = Self(0b1000);
    #[doc = "Allows creating image views with cube type from the created image"]
    pub const CUBE_COMPATIBLE: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkImageViewCreateFlagBits.html>"]
pub struct ImageViewCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ImageViewCreateFlags, Flags);
impl ImageViewCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSamplerCreateFlagBits.html>"]
pub struct SamplerCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SamplerCreateFlags, Flags);
impl SamplerCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreateFlagBits.html>"]
pub struct PipelineCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineCreateFlags, Flags);
impl PipelineCreateFlags {
    pub const DISABLE_OPTIMIZATION: Self = Self(0b1);
    pub const ALLOW_DERIVATIVES: Self = Self(0b10);
    pub const DERIVATIVE: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreateFlagBits2.html>"]
pub struct PipelineCreateFlags2(pub(crate) Flags64);
vk_bitflags_wrapped!(PipelineCreateFlags2, Flags64);
impl PipelineCreateFlags2 {
    pub const DISABLE_OPTIMIZATION: Self = Self(0b1);
    pub const ALLOW_DERIVATIVES: Self = Self(0b10);
    pub const DERIVATIVE: Self = Self(0b100);
    pub const VIEW_INDEX_FROM_DEVICE_INDEX: Self = Self(0b1000);
    pub const DISPATCH_BASE: Self = Self(0b1_0000);
    pub const FAIL_ON_PIPELINE_COMPILE_REQUIRED: Self = Self(0b1_0000_0000);
    pub const EARLY_RETURN_ON_FAILURE: Self = Self(0b10_0000_0000);
    pub const NO_PROTECTED_ACCESS: Self = Self(0b1000_0000_0000_0000_0000_0000_0000);
    pub const PROTECTED_ACCESS_ONLY: Self = Self(0b100_0000_0000_0000_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateFlagBits.html>"]
pub struct PipelineShaderStageCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineShaderStageCreateFlags, Flags);
impl PipelineShaderStageCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkColorComponentFlagBits.html>"]
pub struct ColorComponentFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ColorComponentFlags, Flags);
impl ColorComponentFlags {
    pub const R: Self = Self(0b1);
    pub const G: Self = Self(0b10);
    pub const B: Self = Self(0b100);
    pub const A: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkFenceCreateFlagBits.html>"]
pub struct FenceCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(FenceCreateFlags, Flags);
impl FenceCreateFlags {
    pub const SIGNALED: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphoreCreateFlagBits.html>"]
pub struct SemaphoreCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SemaphoreCreateFlags, Flags);
impl SemaphoreCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkFormatFeatureFlagBits.html>"]
pub struct FormatFeatureFlags(pub(crate) Flags);
vk_bitflags_wrapped!(FormatFeatureFlags, Flags);
impl FormatFeatureFlags {
    #[doc = "Format can be used for sampled images (SAMPLED_IMAGE and COMBINED_IMAGE_SAMPLER descriptor types)"]
    pub const SAMPLED_IMAGE: Self = Self(0b1);
    #[doc = "Format can be used for storage images (STORAGE_IMAGE descriptor type)"]
    pub const STORAGE_IMAGE: Self = Self(0b10);
    #[doc = "Format supports atomic operations in case it is used for storage images"]
    pub const STORAGE_IMAGE_ATOMIC: Self = Self(0b100);
    #[doc = "Format can be used for uniform texel buffers (TBOs)"]
    pub const UNIFORM_TEXEL_BUFFER: Self = Self(0b1000);
    #[doc = "Format can be used for storage texel buffers (IBOs)"]
    pub const STORAGE_TEXEL_BUFFER: Self = Self(0b1_0000);
    #[doc = "Format supports atomic operations in case it is used for storage texel buffers"]
    pub const STORAGE_TEXEL_BUFFER_ATOMIC: Self = Self(0b10_0000);
    #[doc = "Format can be used for vertex buffers (VBOs)"]
    pub const VERTEX_BUFFER: Self = Self(0b100_0000);
    #[doc = "Format can be used for color attachment images"]
    pub const COLOR_ATTACHMENT: Self = Self(0b1000_0000);
    #[doc = "Format supports blending in case it is used for color attachment images"]
    pub const COLOR_ATTACHMENT_BLEND: Self = Self(0b1_0000_0000);
    #[doc = "Format can be used for depth/stencil attachment images"]
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(0b10_0000_0000);
    #[doc = "Format can be used as the source image of blits with vkCmdBlitImage"]
    pub const BLIT_SRC: Self = Self(0b100_0000_0000);
    #[doc = "Format can be used as the destination image of blits with vkCmdBlitImage"]
    pub const BLIT_DST: Self = Self(0b1000_0000_0000);
    #[doc = "Format can be filtered with VK_FILTER_LINEAR when being sampled"]
    pub const SAMPLED_IMAGE_FILTER_LINEAR: Self = Self(0b1_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkQueryControlFlagBits.html>"]
pub struct QueryControlFlags(pub(crate) Flags);
vk_bitflags_wrapped!(QueryControlFlags, Flags);
impl QueryControlFlags {
    #[doc = "Require precise results to be collected by the query"]
    pub const PRECISE: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkQueryResultFlagBits.html>"]
pub struct QueryResultFlags(pub(crate) Flags);
vk_bitflags_wrapped!(QueryResultFlags, Flags);
impl QueryResultFlags {
    #[doc = "Results of the queries are written to the destination buffer as 64-bit values"]
    pub const TYPE_64: Self = Self(0b1);
    #[doc = "Results of the queries are waited on before proceeding with the result copy"]
    pub const WAIT: Self = Self(0b10);
    #[doc = "Besides the results of the query, the availability of the results is also written"]
    pub const WITH_AVAILABILITY: Self = Self(0b100);
    #[doc = "Copy the partial results of the query even if the final results are not available"]
    pub const PARTIAL: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandBufferUsageFlagBits.html>"]
pub struct CommandBufferUsageFlags(pub(crate) Flags);
vk_bitflags_wrapped!(CommandBufferUsageFlags, Flags);
impl CommandBufferUsageFlags {
    pub const ONE_TIME_SUBMIT: Self = Self(0b1);
    pub const RENDER_PASS_CONTINUE: Self = Self(0b10);
    #[doc = "Command buffer may be submitted/executed more than once simultaneously"]
    pub const SIMULTANEOUS_USE: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkQueryPipelineStatisticFlagBits.html>"]
pub struct QueryPipelineStatisticFlags(pub(crate) Flags);
vk_bitflags_wrapped!(QueryPipelineStatisticFlags, Flags);
impl QueryPipelineStatisticFlags {
    #[doc = "Optional"]
    pub const INPUT_ASSEMBLY_VERTICES: Self = Self(0b1);
    #[doc = "Optional"]
    pub const INPUT_ASSEMBLY_PRIMITIVES: Self = Self(0b10);
    #[doc = "Optional"]
    pub const VERTEX_SHADER_INVOCATIONS: Self = Self(0b100);
    #[doc = "Optional"]
    pub const GEOMETRY_SHADER_INVOCATIONS: Self = Self(0b1000);
    #[doc = "Optional"]
    pub const GEOMETRY_SHADER_PRIMITIVES: Self = Self(0b1_0000);
    #[doc = "Optional"]
    pub const CLIPPING_INVOCATIONS: Self = Self(0b10_0000);
    #[doc = "Optional"]
    pub const CLIPPING_PRIMITIVES: Self = Self(0b100_0000);
    #[doc = "Optional"]
    pub const FRAGMENT_SHADER_INVOCATIONS: Self = Self(0b1000_0000);
    #[doc = "Optional"]
    pub const TESSELLATION_CONTROL_SHADER_PATCHES: Self = Self(0b1_0000_0000);
    #[doc = "Optional"]
    pub const TESSELLATION_EVALUATION_SHADER_INVOCATIONS: Self = Self(0b10_0000_0000);
    #[doc = "Optional"]
    pub const COMPUTE_SHADER_INVOCATIONS: Self = Self(0b100_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryMapFlagBits.html>"]
pub struct MemoryMapFlags(pub(crate) Flags);
vk_bitflags_wrapped!(MemoryMapFlags, Flags);
impl MemoryMapFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkImageAspectFlagBits.html>"]
pub struct ImageAspectFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ImageAspectFlags, Flags);
impl ImageAspectFlags {
    pub const COLOR: Self = Self(0b1);
    pub const DEPTH: Self = Self(0b10);
    pub const STENCIL: Self = Self(0b100);
    pub const METADATA: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSparseImageFormatFlagBits.html>"]
pub struct SparseImageFormatFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SparseImageFormatFlags, Flags);
impl SparseImageFormatFlags {
    #[doc = "Image uses a single mip tail region for all array layers"]
    pub const SINGLE_MIPTAIL: Self = Self(0b1);
    #[doc = "Image requires mip level dimensions to be an integer multiple of the sparse image block dimensions for non-tail mip levels."]
    pub const ALIGNED_MIP_SIZE: Self = Self(0b10);
    #[doc = "Image uses a non-standard sparse image block dimensions"]
    pub const NONSTANDARD_BLOCK_SIZE: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSparseMemoryBindFlagBits.html>"]
pub struct SparseMemoryBindFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SparseMemoryBindFlags, Flags);
impl SparseMemoryBindFlags {
    #[doc = "Operation binds resource metadata to memory"]
    pub const METADATA: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineStageFlagBits.html>"]
pub struct PipelineStageFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineStageFlags, Flags);
impl PipelineStageFlags {
    #[doc = "Before subsequent commands are processed"]
    pub const TOP_OF_PIPE: Self = Self(0b1);
    #[doc = "Draw/DispatchIndirect command fetch"]
    pub const DRAW_INDIRECT: Self = Self(0b10);
    #[doc = "Vertex/index fetch"]
    pub const VERTEX_INPUT: Self = Self(0b100);
    #[doc = "Vertex shading"]
    pub const VERTEX_SHADER: Self = Self(0b1000);
    #[doc = "Tessellation control shading"]
    pub const TESSELLATION_CONTROL_SHADER: Self = Self(0b1_0000);
    #[doc = "Tessellation evaluation shading"]
    pub const TESSELLATION_EVALUATION_SHADER: Self = Self(0b10_0000);
    #[doc = "Geometry shading"]
    pub const GEOMETRY_SHADER: Self = Self(0b100_0000);
    #[doc = "Fragment shading"]
    pub const FRAGMENT_SHADER: Self = Self(0b1000_0000);
    #[doc = "Early fragment (depth and stencil) tests"]
    pub const EARLY_FRAGMENT_TESTS: Self = Self(0b1_0000_0000);
    #[doc = "Late fragment (depth and stencil) tests"]
    pub const LATE_FRAGMENT_TESTS: Self = Self(0b10_0000_0000);
    #[doc = "Color attachment writes"]
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(0b100_0000_0000);
    #[doc = "Compute shading"]
    pub const COMPUTE_SHADER: Self = Self(0b1000_0000_0000);
    #[doc = "Transfer/copy operations"]
    pub const TRANSFER: Self = Self(0b1_0000_0000_0000);
    #[doc = "After previous commands have completed"]
    pub const BOTTOM_OF_PIPE: Self = Self(0b10_0000_0000_0000);
    #[doc = "Indicates host (CPU) is a source/sink of the dependency"]
    pub const HOST: Self = Self(0b100_0000_0000_0000);
    #[doc = "All stages of the graphics pipeline"]
    pub const ALL_GRAPHICS: Self = Self(0b1000_0000_0000_0000);
    #[doc = "All stages supported on the queue"]
    pub const ALL_COMMANDS: Self = Self(0b1_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandPoolCreateFlagBits.html>"]
pub struct CommandPoolCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(CommandPoolCreateFlags, Flags);
impl CommandPoolCreateFlags {
    #[doc = "Command buffers have a short lifetime"]
    pub const TRANSIENT: Self = Self(0b1);
    #[doc = "Command buffers may release their memory individually"]
    pub const RESET_COMMAND_BUFFER: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandPoolResetFlagBits.html>"]
pub struct CommandPoolResetFlags(pub(crate) Flags);
vk_bitflags_wrapped!(CommandPoolResetFlags, Flags);
impl CommandPoolResetFlags {
    #[doc = "Release resources owned by the pool"]
    pub const RELEASE_RESOURCES: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandBufferResetFlagBits.html>"]
pub struct CommandBufferResetFlags(pub(crate) Flags);
vk_bitflags_wrapped!(CommandBufferResetFlags, Flags);
impl CommandBufferResetFlags {
    #[doc = "Release resources owned by the buffer"]
    pub const RELEASE_RESOURCES: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSampleCountFlagBits.html>"]
pub struct SampleCountFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SampleCountFlags, Flags);
impl SampleCountFlags {
    #[doc = "Sample count 1 supported"]
    pub const TYPE_1: Self = Self(0b1);
    #[doc = "Sample count 2 supported"]
    pub const TYPE_2: Self = Self(0b10);
    #[doc = "Sample count 4 supported"]
    pub const TYPE_4: Self = Self(0b100);
    #[doc = "Sample count 8 supported"]
    pub const TYPE_8: Self = Self(0b1000);
    #[doc = "Sample count 16 supported"]
    pub const TYPE_16: Self = Self(0b1_0000);
    #[doc = "Sample count 32 supported"]
    pub const TYPE_32: Self = Self(0b10_0000);
    #[doc = "Sample count 64 supported"]
    pub const TYPE_64: Self = Self(0b100_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkAttachmentDescriptionFlagBits.html>"]
pub struct AttachmentDescriptionFlags(pub(crate) Flags);
vk_bitflags_wrapped!(AttachmentDescriptionFlags, Flags);
impl AttachmentDescriptionFlags {
    #[doc = "The attachment may alias physical memory of another attachment in the same render pass"]
    pub const MAY_ALIAS: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkStencilFaceFlagBits.html>"]
pub struct StencilFaceFlags(pub(crate) Flags);
vk_bitflags_wrapped!(StencilFaceFlags, Flags);
impl StencilFaceFlags {
    #[doc = "Front face"]
    pub const FRONT: Self = Self(0b1);
    #[doc = "Back face"]
    pub const BACK: Self = Self(0b10);
    #[doc = "Front and back faces"]
    pub const FRONT_AND_BACK: Self = Self(0x00000003);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorPoolCreateFlagBits.html>"]
pub struct DescriptorPoolCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(DescriptorPoolCreateFlags, Flags);
impl DescriptorPoolCreateFlags {
    #[doc = "Descriptor sets may be freed individually"]
    pub const FREE_DESCRIPTOR_SET: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDependencyFlagBits.html>"]
pub struct DependencyFlags(pub(crate) Flags);
vk_bitflags_wrapped!(DependencyFlags, Flags);
impl DependencyFlags {
    #[doc = "Dependency is per pixel region "]
    pub const BY_REGION: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphoreWaitFlagBits.html>"]
pub struct SemaphoreWaitFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SemaphoreWaitFlags, Flags);
impl SemaphoreWaitFlags {
    pub const ANY: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDisplayPlaneAlphaFlagBitsKHR.html>"]
pub struct DisplayPlaneAlphaFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(DisplayPlaneAlphaFlagsKHR, Flags);
impl DisplayPlaneAlphaFlagsKHR {
    pub const OPAQUE: Self = Self(0b1);
    pub const GLOBAL: Self = Self(0b10);
    pub const PER_PIXEL: Self = Self(0b100);
    pub const PER_PIXEL_PREMULTIPLIED: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkCompositeAlphaFlagBitsKHR.html>"]
pub struct CompositeAlphaFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(CompositeAlphaFlagsKHR, Flags);
impl CompositeAlphaFlagsKHR {
    pub const OPAQUE: Self = Self(0b1);
    pub const PRE_MULTIPLIED: Self = Self(0b10);
    pub const POST_MULTIPLIED: Self = Self(0b100);
    pub const INHERIT: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSurfaceTransformFlagBitsKHR.html>"]
pub struct SurfaceTransformFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(SurfaceTransformFlagsKHR, Flags);
impl SurfaceTransformFlagsKHR {
    pub const IDENTITY: Self = Self(0b1);
    pub const ROTATE_90: Self = Self(0b10);
    pub const ROTATE_180: Self = Self(0b100);
    pub const ROTATE_270: Self = Self(0b1000);
    pub const HORIZONTAL_MIRROR: Self = Self(0b1_0000);
    pub const HORIZONTAL_MIRROR_ROTATE_90: Self = Self(0b10_0000);
    pub const HORIZONTAL_MIRROR_ROTATE_180: Self = Self(0b100_0000);
    pub const HORIZONTAL_MIRROR_ROTATE_270: Self = Self(0b1000_0000);
    pub const INHERIT: Self = Self(0b1_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainImageUsageFlagBitsANDROID.html>"]
pub struct SwapchainImageUsageFlagsANDROID(pub(crate) Flags);
vk_bitflags_wrapped!(SwapchainImageUsageFlagsANDROID, Flags);
impl SwapchainImageUsageFlagsANDROID {
    pub const SHARED: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDebugReportFlagBitsEXT.html>"]
pub struct DebugReportFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(DebugReportFlagsEXT, Flags);
impl DebugReportFlagsEXT {
    pub const INFORMATION: Self = Self(0b1);
    pub const WARNING: Self = Self(0b10);
    pub const PERFORMANCE_WARNING: Self = Self(0b100);
    pub const ERROR: Self = Self(0b1000);
    pub const DEBUG: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalMemoryHandleTypeFlagBitsNV.html>"]
pub struct ExternalMemoryHandleTypeFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(ExternalMemoryHandleTypeFlagsNV, Flags);
impl ExternalMemoryHandleTypeFlagsNV {
    pub const OPAQUE_WIN32: Self = Self(0b1);
    pub const OPAQUE_WIN32_KMT: Self = Self(0b10);
    pub const D3D11_IMAGE: Self = Self(0b100);
    pub const D3D11_IMAGE_KMT: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalMemoryFeatureFlagBitsNV.html>"]
pub struct ExternalMemoryFeatureFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(ExternalMemoryFeatureFlagsNV, Flags);
impl ExternalMemoryFeatureFlagsNV {
    pub const DEDICATED_ONLY: Self = Self(0b1);
    pub const EXPORTABLE: Self = Self(0b10);
    pub const IMPORTABLE: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkClusterAccelerationStructureIndexFormatFlagBitsNV.html>"]
pub struct ClusterAccelerationStructureIndexFormatFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(ClusterAccelerationStructureIndexFormatFlagsNV, Flags);
impl ClusterAccelerationStructureIndexFormatFlagsNV {
    pub const TYPE_8BIT: Self = Self(0b1);
    pub const TYPE_16BIT: Self = Self(0b10);
    pub const TYPE_32BIT: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkClusterAccelerationStructureClusterFlagBitsNV.html>"]
pub struct ClusterAccelerationStructureClusterFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(ClusterAccelerationStructureClusterFlagsNV, Flags);
impl ClusterAccelerationStructureClusterFlagsNV {
    pub const ALLOW_DISABLE_OPACITY_MICROMAPS: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkClusterAccelerationStructureGeometryFlagBitsNV.html>"]
pub struct ClusterAccelerationStructureGeometryFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(ClusterAccelerationStructureGeometryFlagsNV, Flags);
impl ClusterAccelerationStructureGeometryFlagsNV {
    pub const CULL_DISABLE: Self = Self(0b1);
    pub const NO_DUPLICATE_ANYHIT_INVOCATION: Self = Self(0b10);
    pub const OPAQUE: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkClusterAccelerationStructureAddressResolutionFlagBitsNV.html>"]
pub struct ClusterAccelerationStructureAddressResolutionFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(ClusterAccelerationStructureAddressResolutionFlagsNV, Flags);
impl ClusterAccelerationStructureAddressResolutionFlagsNV {
    pub const NONE: Self = Self(0);
    pub const INDIRECTED_DST_IMPLICIT_DATA: Self = Self(0b1);
    pub const INDIRECTED_SCRATCH_DATA: Self = Self(0b10);
    pub const INDIRECTED_DST_ADDRESS_ARRAY: Self = Self(0b100);
    pub const INDIRECTED_DST_SIZES_ARRAY: Self = Self(0b1000);
    pub const INDIRECTED_SRC_INFOS_ARRAY: Self = Self(0b1_0000);
    pub const INDIRECTED_SRC_INFOS_COUNT: Self = Self(0b10_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSubgroupFeatureFlagBits.html>"]
pub struct SubgroupFeatureFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SubgroupFeatureFlags, Flags);
impl SubgroupFeatureFlags {
    #[doc = "Basic subgroup operations"]
    pub const BASIC: Self = Self(0b1);
    #[doc = "Vote subgroup operations"]
    pub const VOTE: Self = Self(0b10);
    #[doc = "Arithmetic subgroup operations"]
    pub const ARITHMETIC: Self = Self(0b100);
    #[doc = "Ballot subgroup operations"]
    pub const BALLOT: Self = Self(0b1000);
    #[doc = "Shuffle subgroup operations"]
    pub const SHUFFLE: Self = Self(0b1_0000);
    #[doc = "Shuffle relative subgroup operations"]
    pub const SHUFFLE_RELATIVE: Self = Self(0b10_0000);
    #[doc = "Clustered subgroup operations"]
    pub const CLUSTERED: Self = Self(0b100_0000);
    #[doc = "Quad subgroup operations"]
    pub const QUAD: Self = Self(0b1000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkIndirectCommandsLayoutUsageFlagBitsNV.html>"]
pub struct IndirectCommandsLayoutUsageFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(IndirectCommandsLayoutUsageFlagsNV, Flags);
impl IndirectCommandsLayoutUsageFlagsNV {
    pub const EXPLICIT_PREPROCESS: Self = Self(0b1);
    pub const INDEXED_SEQUENCES: Self = Self(0b10);
    pub const UNORDERED_SEQUENCES: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkIndirectStateFlagBitsNV.html>"]
pub struct IndirectStateFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(IndirectStateFlagsNV, Flags);
impl IndirectStateFlagsNV {
    pub const FLAG_FRONTFACE: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPrivateDataSlotCreateFlagBits.html>"]
pub struct PrivateDataSlotCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PrivateDataSlotCreateFlags, Flags);
impl PrivateDataSlotCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorSetLayoutCreateFlagBits.html>"]
pub struct DescriptorSetLayoutCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(DescriptorSetLayoutCreateFlags, Flags);
impl DescriptorSetLayoutCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalMemoryHandleTypeFlagBits.html>"]
pub struct ExternalMemoryHandleTypeFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ExternalMemoryHandleTypeFlags, Flags);
impl ExternalMemoryHandleTypeFlags {
    pub const OPAQUE_FD: Self = Self(0b1);
    pub const OPAQUE_WIN32: Self = Self(0b10);
    pub const OPAQUE_WIN32_KMT: Self = Self(0b100);
    pub const D3D11_TEXTURE: Self = Self(0b1000);
    pub const D3D11_TEXTURE_KMT: Self = Self(0b1_0000);
    pub const D3D12_HEAP: Self = Self(0b10_0000);
    pub const D3D12_RESOURCE: Self = Self(0b100_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalMemoryFeatureFlagBits.html>"]
pub struct ExternalMemoryFeatureFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ExternalMemoryFeatureFlags, Flags);
impl ExternalMemoryFeatureFlags {
    pub const DEDICATED_ONLY: Self = Self(0b1);
    pub const EXPORTABLE: Self = Self(0b10);
    pub const IMPORTABLE: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalSemaphoreHandleTypeFlagBits.html>"]
pub struct ExternalSemaphoreHandleTypeFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ExternalSemaphoreHandleTypeFlags, Flags);
impl ExternalSemaphoreHandleTypeFlags {
    pub const OPAQUE_FD: Self = Self(0b1);
    pub const OPAQUE_WIN32: Self = Self(0b10);
    pub const OPAQUE_WIN32_KMT: Self = Self(0b100);
    pub const D3D12_FENCE: Self = Self(0b1000);
    pub const D3D11_FENCE: Self = Self::D3D12_FENCE;
    pub const SYNC_FD: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalSemaphoreFeatureFlagBits.html>"]
pub struct ExternalSemaphoreFeatureFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ExternalSemaphoreFeatureFlags, Flags);
impl ExternalSemaphoreFeatureFlags {
    pub const EXPORTABLE: Self = Self(0b1);
    pub const IMPORTABLE: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphoreImportFlagBits.html>"]
pub struct SemaphoreImportFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SemaphoreImportFlags, Flags);
impl SemaphoreImportFlags {
    pub const TEMPORARY: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalFenceHandleTypeFlagBits.html>"]
pub struct ExternalFenceHandleTypeFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ExternalFenceHandleTypeFlags, Flags);
impl ExternalFenceHandleTypeFlags {
    pub const OPAQUE_FD: Self = Self(0b1);
    pub const OPAQUE_WIN32: Self = Self(0b10);
    pub const OPAQUE_WIN32_KMT: Self = Self(0b100);
    pub const SYNC_FD: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalFenceFeatureFlagBits.html>"]
pub struct ExternalFenceFeatureFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ExternalFenceFeatureFlags, Flags);
impl ExternalFenceFeatureFlags {
    pub const EXPORTABLE: Self = Self(0b1);
    pub const IMPORTABLE: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkFenceImportFlagBits.html>"]
pub struct FenceImportFlags(pub(crate) Flags);
vk_bitflags_wrapped!(FenceImportFlags, Flags);
impl FenceImportFlags {
    pub const TEMPORARY: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSurfaceCounterFlagBitsEXT.html>"]
pub struct SurfaceCounterFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(SurfaceCounterFlagsEXT, Flags);
impl SurfaceCounterFlagsEXT {
    pub const VBLANK: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPeerMemoryFeatureFlagBits.html>"]
pub struct PeerMemoryFeatureFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PeerMemoryFeatureFlags, Flags);
impl PeerMemoryFeatureFlags {
    #[doc = "Can read with vkCmdCopy commands"]
    pub const COPY_SRC: Self = Self(0b1);
    #[doc = "Can write with vkCmdCopy commands"]
    pub const COPY_DST: Self = Self(0b10);
    #[doc = "Can read with any access type/command"]
    pub const GENERIC_SRC: Self = Self(0b100);
    #[doc = "Can write with and access type/command"]
    pub const GENERIC_DST: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryAllocateFlagBits.html>"]
pub struct MemoryAllocateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(MemoryAllocateFlags, Flags);
impl MemoryAllocateFlags {
    #[doc = "Force allocation on specific devices"]
    pub const DEVICE_MASK: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceGroupPresentModeFlagBitsKHR.html>"]
pub struct DeviceGroupPresentModeFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(DeviceGroupPresentModeFlagsKHR, Flags);
impl DeviceGroupPresentModeFlagsKHR {
    #[doc = "Present from local memory"]
    pub const LOCAL: Self = Self(0b1);
    #[doc = "Present from remote memory"]
    pub const REMOTE: Self = Self(0b10);
    #[doc = "Present sum of local and/or remote memory"]
    pub const SUM: Self = Self(0b100);
    #[doc = "Each physical device presents from local memory"]
    pub const LOCAL_MULTI_DEVICE: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainCreateFlagBitsKHR.html>"]
pub struct SwapchainCreateFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(SwapchainCreateFlagsKHR, Flags);
impl SwapchainCreateFlagsKHR {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSubpassDescriptionFlagBits.html>"]
pub struct SubpassDescriptionFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SubpassDescriptionFlags, Flags);
impl SubpassDescriptionFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDebugUtilsMessageSeverityFlagBitsEXT.html>"]
pub struct DebugUtilsMessageSeverityFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(DebugUtilsMessageSeverityFlagsEXT, Flags);
impl DebugUtilsMessageSeverityFlagsEXT {
    pub const VERBOSE: Self = Self(0b1);
    pub const INFO: Self = Self(0b1_0000);
    pub const WARNING: Self = Self(0b1_0000_0000);
    pub const ERROR: Self = Self(0b1_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDebugUtilsMessageTypeFlagBitsEXT.html>"]
pub struct DebugUtilsMessageTypeFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(DebugUtilsMessageTypeFlagsEXT, Flags);
impl DebugUtilsMessageTypeFlagsEXT {
    pub const GENERAL: Self = Self(0b1);
    pub const VALIDATION: Self = Self(0b10);
    pub const PERFORMANCE: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorBindingFlagBits.html>"]
pub struct DescriptorBindingFlags(pub(crate) Flags);
vk_bitflags_wrapped!(DescriptorBindingFlags, Flags);
impl DescriptorBindingFlags {
    pub const UPDATE_AFTER_BIND: Self = Self(0b1);
    pub const UPDATE_UNUSED_WHILE_PENDING: Self = Self(0b10);
    pub const PARTIALLY_BOUND: Self = Self(0b100);
    pub const VARIABLE_DESCRIPTOR_COUNT: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkConditionalRenderingFlagBitsEXT.html>"]
pub struct ConditionalRenderingFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(ConditionalRenderingFlagsEXT, Flags);
impl ConditionalRenderingFlagsEXT {
    pub const INVERTED: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkResolveModeFlagBits.html>"]
pub struct ResolveModeFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ResolveModeFlags, Flags);
impl ResolveModeFlags {
    pub const NONE: Self = Self(0);
    pub const SAMPLE_ZERO: Self = Self(0b1);
    pub const AVERAGE: Self = Self(0b10);
    pub const MIN: Self = Self(0b100);
    pub const MAX: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkGeometryInstanceFlagBitsKHR.html>"]
pub struct GeometryInstanceFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(GeometryInstanceFlagsKHR, Flags);
impl GeometryInstanceFlagsKHR {
    pub const TRIANGLE_FACING_CULL_DISABLE: Self = Self(0b1);
    pub const TRIANGLE_FLIP_FACING: Self = Self(0b10);
    pub const FORCE_OPAQUE: Self = Self(0b100);
    pub const FORCE_NO_OPAQUE: Self = Self(0b1000);
    pub const TRIANGLE_FRONT_COUNTERCLOCKWISE: Self = Self::TRIANGLE_FLIP_FACING;
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkGeometryFlagBitsKHR.html>"]
pub struct GeometryFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(GeometryFlagsKHR, Flags);
impl GeometryFlagsKHR {
    pub const OPAQUE: Self = Self(0b1);
    pub const NO_DUPLICATE_ANY_HIT_INVOCATION: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkBuildAccelerationStructureFlagBitsKHR.html>"]
pub struct BuildAccelerationStructureFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(BuildAccelerationStructureFlagsKHR, Flags);
impl BuildAccelerationStructureFlagsKHR {
    pub const ALLOW_UPDATE: Self = Self(0b1);
    pub const ALLOW_COMPACTION: Self = Self(0b10);
    pub const PREFER_FAST_TRACE: Self = Self(0b100);
    pub const PREFER_FAST_BUILD: Self = Self(0b1000);
    pub const LOW_MEMORY: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkAccelerationStructureCreateFlagBitsKHR.html>"]
pub struct AccelerationStructureCreateFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(AccelerationStructureCreateFlagsKHR, Flags);
impl AccelerationStructureCreateFlagsKHR {
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkFramebufferCreateFlagBits.html>"]
pub struct FramebufferCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(FramebufferCreateFlags, Flags);
impl FramebufferCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkQueryPoolCreateFlagBits.html>"]
pub struct QueryPoolCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(QueryPoolCreateFlags, Flags);
impl QueryPoolCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceDiagnosticsConfigFlagBitsNV.html>"]
pub struct DeviceDiagnosticsConfigFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(DeviceDiagnosticsConfigFlagsNV, Flags);
impl DeviceDiagnosticsConfigFlagsNV {
    pub const ENABLE_SHADER_DEBUG_INFO: Self = Self(0b1);
    pub const ENABLE_RESOURCE_TRACKING: Self = Self(0b10);
    pub const ENABLE_AUTOMATIC_CHECKPOINTS: Self = Self(0b100);
    pub const ENABLE_SHADER_ERROR_REPORTING: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreationFeedbackFlagBits.html>"]
pub struct PipelineCreationFeedbackFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineCreationFeedbackFlags, Flags);
impl PipelineCreationFeedbackFlags {
    pub const VALID: Self = Self(0b1);
    pub const APPLICATION_PIPELINE_CACHE_HIT: Self = Self(0b10);
    pub const BASE_PIPELINE_ACCELERATION: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryDecompressionMethodFlagBitsEXT.html>"]
pub struct MemoryDecompressionMethodFlagsEXT(pub(crate) Flags64);
vk_bitflags_wrapped!(MemoryDecompressionMethodFlagsEXT, Flags64);
impl MemoryDecompressionMethodFlagsEXT {
    pub const GDEFLATE_1_0: Self = Self(0b1);
    pub const GDEFLATE_1_0_NV: Self = Self::GDEFLATE_1_0;
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPerformanceCounterDescriptionFlagBitsKHR.html>"]
pub struct PerformanceCounterDescriptionFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(PerformanceCounterDescriptionFlagsKHR, Flags);
impl PerformanceCounterDescriptionFlagsKHR {
    pub const PERFORMANCE_IMPACTING: Self = Self(0b1);
    pub const CONCURRENTLY_IMPACTED: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkAcquireProfilingLockFlagBitsKHR.html>"]
pub struct AcquireProfilingLockFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(AcquireProfilingLockFlagsKHR, Flags);
impl AcquireProfilingLockFlagsKHR {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderCorePropertiesFlagBitsAMD.html>"]
pub struct ShaderCorePropertiesFlagsAMD(pub(crate) Flags);
vk_bitflags_wrapped!(ShaderCorePropertiesFlagsAMD, Flags);
impl ShaderCorePropertiesFlagsAMD {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModuleCreateFlagBits.html>"]
pub struct ShaderModuleCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ShaderModuleCreateFlags, Flags);
impl ShaderModuleCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCompilerControlFlagBitsAMD.html>"]
pub struct PipelineCompilerControlFlagsAMD(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineCompilerControlFlagsAMD, Flags);
impl PipelineCompilerControlFlagsAMD {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkToolPurposeFlagBits.html>"]
pub struct ToolPurposeFlags(pub(crate) Flags);
vk_bitflags_wrapped!(ToolPurposeFlags, Flags);
impl ToolPurposeFlags {
    pub const VALIDATION: Self = Self(0b1);
    pub const PROFILING: Self = Self(0b10);
    pub const TRACING: Self = Self(0b100);
    pub const ADDITIONAL_FEATURES: Self = Self(0b1000);
    pub const MODIFYING_FEATURES: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkAccessFlagBits2.html>"]
pub struct AccessFlags2(pub(crate) Flags64);
vk_bitflags_wrapped!(AccessFlags2, Flags64);
impl AccessFlags2 {
    pub const NONE: Self = Self(0);
    pub const INDIRECT_COMMAND_READ: Self = Self(0b1);
    pub const INDEX_READ: Self = Self(0b10);
    pub const VERTEX_ATTRIBUTE_READ: Self = Self(0b100);
    pub const UNIFORM_READ: Self = Self(0b1000);
    pub const INPUT_ATTACHMENT_READ: Self = Self(0b1_0000);
    pub const SHADER_READ: Self = Self(0b10_0000);
    pub const SHADER_WRITE: Self = Self(0b100_0000);
    pub const COLOR_ATTACHMENT_READ: Self = Self(0b1000_0000);
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(0b1_0000_0000);
    pub const DEPTH_STENCIL_ATTACHMENT_READ: Self = Self(0b10_0000_0000);
    pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Self = Self(0b100_0000_0000);
    pub const TRANSFER_READ: Self = Self(0b1000_0000_0000);
    pub const TRANSFER_WRITE: Self = Self(0b1_0000_0000_0000);
    pub const HOST_READ: Self = Self(0b10_0000_0000_0000);
    pub const HOST_WRITE: Self = Self(0b100_0000_0000_0000);
    pub const MEMORY_READ: Self = Self(0b1000_0000_0000_0000);
    pub const MEMORY_WRITE: Self = Self(0b1_0000_0000_0000_0000);
    pub const SHADER_SAMPLED_READ: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const SHADER_STORAGE_READ: Self = Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const SHADER_STORAGE_WRITE: Self = Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineStageFlagBits2.html>"]
pub struct PipelineStageFlags2(pub(crate) Flags64);
vk_bitflags_wrapped!(PipelineStageFlags2, Flags64);
impl PipelineStageFlags2 {
    pub const NONE: Self = Self(0);
    pub const TOP_OF_PIPE: Self = Self(0b1);
    pub const DRAW_INDIRECT: Self = Self(0b10);
    pub const VERTEX_INPUT: Self = Self(0b100);
    pub const VERTEX_SHADER: Self = Self(0b1000);
    pub const TESSELLATION_CONTROL_SHADER: Self = Self(0b1_0000);
    pub const TESSELLATION_EVALUATION_SHADER: Self = Self(0b10_0000);
    pub const GEOMETRY_SHADER: Self = Self(0b100_0000);
    pub const FRAGMENT_SHADER: Self = Self(0b1000_0000);
    pub const EARLY_FRAGMENT_TESTS: Self = Self(0b1_0000_0000);
    pub const LATE_FRAGMENT_TESTS: Self = Self(0b10_0000_0000);
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(0b100_0000_0000);
    pub const COMPUTE_SHADER: Self = Self(0b1000_0000_0000);
    pub const ALL_TRANSFER: Self = Self(0b1_0000_0000_0000);
    pub const TRANSFER: Self = Self::ALL_TRANSFER;
    pub const BOTTOM_OF_PIPE: Self = Self(0b10_0000_0000_0000);
    pub const HOST: Self = Self(0b100_0000_0000_0000);
    pub const ALL_GRAPHICS: Self = Self(0b1000_0000_0000_0000);
    pub const ALL_COMMANDS: Self = Self(0b1_0000_0000_0000_0000);
    pub const COPY: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const RESOLVE: Self = Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const BLIT: Self = Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const CLEAR: Self = Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const INDEX_INPUT: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const VERTEX_ATTRIBUTE_INPUT: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const PRE_RASTERIZATION_SHADERS: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSubmitFlagBits.html>"]
pub struct SubmitFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SubmitFlags, Flags);
impl SubmitFlags {
    pub const PROTECTED: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkEventCreateFlagBits.html>"]
pub struct EventCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(EventCreateFlags, Flags);
impl EventCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineLayoutCreateFlagBits.html>"]
pub struct PipelineLayoutCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineLayoutCreateFlags, Flags);
impl PipelineLayoutCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineColorBlendStateCreateFlagBits.html>"]
pub struct PipelineColorBlendStateCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineColorBlendStateCreateFlags, Flags);
impl PipelineColorBlendStateCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineDepthStencilStateCreateFlagBits.html>"]
pub struct PipelineDepthStencilStateCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(PipelineDepthStencilStateCreateFlags, Flags);
impl PipelineDepthStencilStateCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkGraphicsPipelineLibraryFlagBitsEXT.html>"]
pub struct GraphicsPipelineLibraryFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(GraphicsPipelineLibraryFlagsEXT, Flags);
impl GraphicsPipelineLibraryFlagsEXT {
    pub const VERTEX_INPUT_INTERFACE: Self = Self(0b1);
    pub const PRE_RASTERIZATION_SHADERS: Self = Self(0b10);
    pub const FRAGMENT_SHADER: Self = Self(0b100);
    pub const FRAGMENT_OUTPUT_INTERFACE: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderingAttachmentFlagBitsKHR.html>"]
pub struct RenderingAttachmentFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(RenderingAttachmentFlagsKHR, Flags);
impl RenderingAttachmentFlagsKHR {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkResolveImageFlagBitsKHR.html>"]
pub struct ResolveImageFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(ResolveImageFlagsKHR, Flags);
impl ResolveImageFlagsKHR {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceAddressBindingFlagBitsEXT.html>"]
pub struct DeviceAddressBindingFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(DeviceAddressBindingFlagsEXT, Flags);
impl DeviceAddressBindingFlagsEXT {
    pub const INTERNAL_OBJECT: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkFrameBoundaryFlagBitsEXT.html>"]
pub struct FrameBoundaryFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(FrameBoundaryFlagsEXT, Flags);
impl FrameBoundaryFlagsEXT {
    pub const FRAME_END: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentScalingFlagBitsKHR.html>"]
pub struct PresentScalingFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(PresentScalingFlagsKHR, Flags);
impl PresentScalingFlagsKHR {
    pub const ONE_TO_ONE: Self = Self(0b1);
    pub const ONE_TO_ONE_EXT: Self = Self::ONE_TO_ONE;
    pub const ASPECT_RATIO_STRETCH: Self = Self(0b10);
    pub const ASPECT_RATIO_STRETCH_EXT: Self = Self::ASPECT_RATIO_STRETCH;
    pub const STRETCH: Self = Self(0b100);
    pub const STRETCH_EXT: Self = Self::STRETCH;
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentGravityFlagBitsKHR.html>"]
pub struct PresentGravityFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(PresentGravityFlagsKHR, Flags);
impl PresentGravityFlagsKHR {
    pub const MIN: Self = Self(0b1);
    pub const MIN_EXT: Self = Self::MIN;
    pub const MAX: Self = Self(0b10);
    pub const MAX_EXT: Self = Self::MAX;
    pub const CENTERED: Self = Self(0b100);
    pub const CENTERED_EXT: Self = Self::CENTERED;
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSchedulingControlsFlagBitsARM.html>"]
pub struct PhysicalDeviceSchedulingControlsFlagsARM(pub(crate) Flags64);
vk_bitflags_wrapped!(PhysicalDeviceSchedulingControlsFlagsARM, Flags64);
impl PhysicalDeviceSchedulingControlsFlagsARM {
    pub const SHADER_CORE_COUNT: Self = Self(0b1);
    pub const DISPATCH_PARAMETERS: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentStageFlagBitsEXT.html>"]
pub struct PresentStageFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(PresentStageFlagsEXT, Flags);
impl PresentStageFlagsEXT {
    pub const QUEUE_OPERATIONS_END: Self = Self(0b1);
    pub const REQUEST_DEQUEUED: Self = Self(0b10);
    pub const IMAGE_FIRST_PIXEL_OUT: Self = Self(0b100);
    pub const IMAGE_FIRST_PIXEL_VISIBLE: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPastPresentationTimingFlagBitsEXT.html>"]
pub struct PastPresentationTimingFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(PastPresentationTimingFlagsEXT, Flags);
impl PastPresentationTimingFlagsEXT {
    pub const ALLOW_PARTIAL_RESULTS: Self = Self(0b1);
    pub const ALLOW_OUT_OF_ORDER_RESULTS: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentTimingInfoFlagBitsEXT.html>"]
pub struct PresentTimingInfoFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(PresentTimingInfoFlagsEXT, Flags);
impl PresentTimingInfoFlagsEXT {
    pub const PRESENT_AT_RELATIVE_TIME: Self = Self(0b1);
    pub const PRESENT_AT_NEAREST_REFRESH_CYCLE: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoCodecOperationFlagBitsKHR.html>"]
pub struct VideoCodecOperationFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoCodecOperationFlagsKHR, Flags);
impl VideoCodecOperationFlagsKHR {
    pub const NONE: Self = Self(0);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoChromaSubsamplingFlagBitsKHR.html>"]
pub struct VideoChromaSubsamplingFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoChromaSubsamplingFlagsKHR, Flags);
impl VideoChromaSubsamplingFlagsKHR {
    pub const INVALID: Self = Self(0);
    pub const MONOCHROME: Self = Self(0b1);
    pub const TYPE_420: Self = Self(0b10);
    pub const TYPE_422: Self = Self(0b100);
    pub const TYPE_444: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoComponentBitDepthFlagBitsKHR.html>"]
pub struct VideoComponentBitDepthFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoComponentBitDepthFlagsKHR, Flags);
impl VideoComponentBitDepthFlagsKHR {
    pub const INVALID: Self = Self(0);
    pub const TYPE_8: Self = Self(0b1);
    pub const TYPE_10: Self = Self(0b100);
    pub const TYPE_12: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoCapabilityFlagBitsKHR.html>"]
pub struct VideoCapabilityFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoCapabilityFlagsKHR, Flags);
impl VideoCapabilityFlagsKHR {
    pub const PROTECTED_CONTENT: Self = Self(0b1);
    pub const SEPARATE_REFERENCE_IMAGES: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoSessionCreateFlagBitsKHR.html>"]
pub struct VideoSessionCreateFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoSessionCreateFlagsKHR, Flags);
impl VideoSessionCreateFlagsKHR {
    pub const PROTECTED_CONTENT: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoSessionParametersCreateFlagBitsKHR.html>"]
pub struct VideoSessionParametersCreateFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoSessionParametersCreateFlagsKHR, Flags);
impl VideoSessionParametersCreateFlagsKHR {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoDecodeH264PictureLayoutFlagBitsKHR.html>"]
pub struct VideoDecodeH264PictureLayoutFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoDecodeH264PictureLayoutFlagsKHR, Flags);
impl VideoDecodeH264PictureLayoutFlagsKHR {
    pub const PROGRESSIVE: Self = Self(0);
    pub const INTERLACED_INTERLEAVED_LINES: Self = Self(0b1);
    pub const INTERLACED_SEPARATE_PLANES: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoCodingControlFlagBitsKHR.html>"]
pub struct VideoCodingControlFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoCodingControlFlagsKHR, Flags);
impl VideoCodingControlFlagsKHR {
    pub const RESET: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoDecodeUsageFlagBitsKHR.html>"]
pub struct VideoDecodeUsageFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoDecodeUsageFlagsKHR, Flags);
impl VideoDecodeUsageFlagsKHR {
    pub const DEFAULT: Self = Self(0);
    pub const TRANSCODING: Self = Self(0b1);
    pub const OFFLINE: Self = Self(0b10);
    pub const STREAMING: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoDecodeCapabilityFlagBitsKHR.html>"]
pub struct VideoDecodeCapabilityFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoDecodeCapabilityFlagsKHR, Flags);
impl VideoDecodeCapabilityFlagsKHR {
    pub const DPB_AND_OUTPUT_COINCIDE: Self = Self(0b1);
    pub const DPB_AND_OUTPUT_DISTINCT: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeFlagBitsKHR.html>"]
pub struct VideoEncodeFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeFlagsKHR, Flags);
impl VideoEncodeFlagsKHR {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeUsageFlagBitsKHR.html>"]
pub struct VideoEncodeUsageFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeUsageFlagsKHR, Flags);
impl VideoEncodeUsageFlagsKHR {
    pub const DEFAULT: Self = Self(0);
    pub const TRANSCODING: Self = Self(0b1);
    pub const STREAMING: Self = Self(0b10);
    pub const RECORDING: Self = Self(0b100);
    pub const CONFERENCING: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeContentFlagBitsKHR.html>"]
pub struct VideoEncodeContentFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeContentFlagsKHR, Flags);
impl VideoEncodeContentFlagsKHR {
    pub const DEFAULT: Self = Self(0);
    pub const CAMERA: Self = Self(0b1);
    pub const DESKTOP: Self = Self(0b10);
    pub const RENDERED: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeCapabilityFlagBitsKHR.html>"]
pub struct VideoEncodeCapabilityFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeCapabilityFlagsKHR, Flags);
impl VideoEncodeCapabilityFlagsKHR {
    pub const PRECEDING_EXTERNALLY_ENCODED_BYTES: Self = Self(0b1);
    pub const INSUFFICIENTSTREAM_BUFFER_RANGE_DETECTION: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeFeedbackFlagBitsKHR.html>"]
pub struct VideoEncodeFeedbackFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeFeedbackFlagsKHR, Flags);
impl VideoEncodeFeedbackFlagsKHR {
    pub const BITSTREAM_BUFFER_OFFSET: Self = Self(0b1);
    pub const BITSTREAM_BYTES_WRITTEN: Self = Self(0b10);
    pub const BITSTREAM_HAS_OVERRIDES: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeRateControlModeFlagBitsKHR.html>"]
pub struct VideoEncodeRateControlModeFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeRateControlModeFlagsKHR, Flags);
impl VideoEncodeRateControlModeFlagsKHR {
    pub const DEFAULT: Self = Self(0);
    pub const DISABLED: Self = Self(0b1);
    pub const CBR: Self = Self(0b10);
    pub const VBR: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeIntraRefreshModeFlagBitsKHR.html>"]
pub struct VideoEncodeIntraRefreshModeFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeIntraRefreshModeFlagsKHR, Flags);
impl VideoEncodeIntraRefreshModeFlagsKHR {
    pub const NONE: Self = Self(0);
    pub const PER_PICTURE_PARTITION: Self = Self(0b1);
    pub const BLOCK_BASED: Self = Self(0b10);
    pub const BLOCK_ROW_BASED: Self = Self(0b100);
    pub const BLOCK_COLUMN_BASED: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeH264CapabilityFlagBitsKHR.html>"]
pub struct VideoEncodeH264CapabilityFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeH264CapabilityFlagsKHR, Flags);
impl VideoEncodeH264CapabilityFlagsKHR {
    pub const HRD_COMPLIANCE: Self = Self(0b1);
    pub const PREDICTION_WEIGHT_TABLE_GENERATED: Self = Self(0b10);
    pub const ROW_UNALIGNED_SLICE: Self = Self(0b100);
    pub const DIFFERENT_SLICE_TYPE: Self = Self(0b1000);
    pub const B_FRAME_IN_L0_LIST: Self = Self(0b1_0000);
    pub const B_FRAME_IN_L1_LIST: Self = Self(0b10_0000);
    pub const PER_PICTURE_TYPE_MIN_MAX_QP: Self = Self(0b100_0000);
    pub const PER_SLICE_CONSTANT_QP: Self = Self(0b1000_0000);
    pub const GENERATE_PREFIX_NALU: Self = Self(0b1_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeH264StdFlagBitsKHR.html>"]
pub struct VideoEncodeH264StdFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeH264StdFlagsKHR, Flags);
impl VideoEncodeH264StdFlagsKHR {
    pub const SEPARATE_COLOR_PLANE_FLAG_SET: Self = Self(0b1);
    pub const QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET: Self = Self(0b10);
    pub const SCALING_MATRIX_PRESENT_FLAG_SET: Self = Self(0b100);
    pub const CHROMA_QP_INDEX_OFFSET: Self = Self(0b1000);
    pub const SECOND_CHROMA_QP_INDEX_OFFSET: Self = Self(0b1_0000);
    pub const PIC_INIT_QP_MINUS26: Self = Self(0b10_0000);
    pub const WEIGHTED_PRED_FLAG_SET: Self = Self(0b100_0000);
    pub const WEIGHTED_BIPRED_IDC_EXPLICIT: Self = Self(0b1000_0000);
    pub const WEIGHTED_BIPRED_IDC_IMPLICIT: Self = Self(0b1_0000_0000);
    pub const TRANSFORM_8X8_MODE_FLAG_SET: Self = Self(0b10_0000_0000);
    pub const DIRECT_SPATIAL_MV_PRED_FLAG_UNSET: Self = Self(0b100_0000_0000);
    pub const ENTROPY_CODING_MODE_FLAG_UNSET: Self = Self(0b1000_0000_0000);
    pub const ENTROPY_CODING_MODE_FLAG_SET: Self = Self(0b1_0000_0000_0000);
    pub const DIRECT_8X8_INFERENCE_FLAG_UNSET: Self = Self(0b10_0000_0000_0000);
    pub const CONSTRAINED_INTRA_PRED_FLAG_SET: Self = Self(0b100_0000_0000_0000);
    pub const DEBLOCKING_FILTER_DISABLED: Self = Self(0b1000_0000_0000_0000);
    pub const DEBLOCKING_FILTER_ENABLED: Self = Self(0b1_0000_0000_0000_0000);
    pub const DEBLOCKING_FILTER_PARTIAL: Self = Self(0b10_0000_0000_0000_0000);
    pub const SLICE_QP_DELTA: Self = Self(0b1000_0000_0000_0000_0000);
    pub const DIFFERENT_SLICE_QP_DELTA: Self = Self(0b1_0000_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeH264RateControlFlagBitsKHR.html>"]
pub struct VideoEncodeH264RateControlFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeH264RateControlFlagsKHR, Flags);
impl VideoEncodeH264RateControlFlagsKHR {
    pub const ATTEMPT_HRD_COMPLIANCE: Self = Self(0b1);
    pub const REGULAR_GOP: Self = Self(0b10);
    pub const REFERENCE_PATTERN_FLAT: Self = Self(0b100);
    pub const REFERENCE_PATTERN_DYADIC: Self = Self(0b1000);
    pub const TEMPORAL_LAYER_PATTERN_DYADIC: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkHostImageCopyFlagBits.html>"]
pub struct HostImageCopyFlags(pub(crate) Flags);
vk_bitflags_wrapped!(HostImageCopyFlags, Flags);
impl HostImageCopyFlags {
    pub const MEMCPY: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkPartitionedAccelerationStructureInstanceFlagBitsNV.html>"]
pub struct PartitionedAccelerationStructureInstanceFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(PartitionedAccelerationStructureInstanceFlagsNV, Flags);
impl PartitionedAccelerationStructureInstanceFlagsNV {
    pub const FLAG_TRIANGLE_FACING_CULL_DISABLE: Self = Self(0b1);
    pub const FLAG_TRIANGLE_FLIP_FACING: Self = Self(0b10);
    pub const FLAG_FORCE_OPAQUE: Self = Self(0b100);
    pub const FLAG_FORCE_NO_OPAQUE: Self = Self(0b1000);
    pub const FLAG_ENABLE_EXPLICIT_BOUNDING_BOX: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkImageFormatConstraintsFlagBitsFUCHSIA.html>"]
pub struct ImageFormatConstraintsFlagsFUCHSIA(pub(crate) Flags);
vk_bitflags_wrapped!(ImageFormatConstraintsFlagsFUCHSIA, Flags);
impl ImageFormatConstraintsFlagsFUCHSIA {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkImageConstraintsInfoFlagBitsFUCHSIA.html>"]
pub struct ImageConstraintsInfoFlagsFUCHSIA(pub(crate) Flags);
vk_bitflags_wrapped!(ImageConstraintsInfoFlagsFUCHSIA, Flags);
impl ImageConstraintsInfoFlagsFUCHSIA {
    pub const CPU_READ_RARELY: Self = Self(0b1);
    pub const CPU_READ_OFTEN: Self = Self(0b10);
    pub const CPU_WRITE_RARELY: Self = Self(0b100);
    pub const CPU_WRITE_OFTEN: Self = Self(0b1000);
    pub const PROTECTED_OPTIONAL: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkFormatFeatureFlagBits2.html>"]
pub struct FormatFeatureFlags2(pub(crate) Flags64);
vk_bitflags_wrapped!(FormatFeatureFlags2, Flags64);
impl FormatFeatureFlags2 {
    pub const SAMPLED_IMAGE: Self = Self(0b1);
    pub const STORAGE_IMAGE: Self = Self(0b10);
    pub const STORAGE_IMAGE_ATOMIC: Self = Self(0b100);
    pub const UNIFORM_TEXEL_BUFFER: Self = Self(0b1000);
    pub const STORAGE_TEXEL_BUFFER: Self = Self(0b1_0000);
    pub const STORAGE_TEXEL_BUFFER_ATOMIC: Self = Self(0b10_0000);
    pub const VERTEX_BUFFER: Self = Self(0b100_0000);
    pub const COLOR_ATTACHMENT: Self = Self(0b1000_0000);
    pub const COLOR_ATTACHMENT_BLEND: Self = Self(0b1_0000_0000);
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(0b10_0000_0000);
    pub const BLIT_SRC: Self = Self(0b100_0000_0000);
    pub const BLIT_DST: Self = Self(0b1000_0000_0000);
    pub const SAMPLED_IMAGE_FILTER_LINEAR: Self = Self(0b1_0000_0000_0000);
    pub const TRANSFER_SRC: Self = Self(0b100_0000_0000_0000);
    pub const TRANSFER_DST: Self = Self(0b1000_0000_0000_0000);
    pub const SAMPLED_IMAGE_FILTER_MINMAX: Self = Self(0b1_0000_0000_0000_0000);
    pub const MIDPOINT_CHROMA_SAMPLES: Self = Self(0b10_0000_0000_0000_0000);
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER: Self = Self(0b100_0000_0000_0000_0000);
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER: Self =
        Self(0b1000_0000_0000_0000_0000);
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT: Self =
        Self(0b1_0000_0000_0000_0000_0000);
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE: Self =
        Self(0b10_0000_0000_0000_0000_0000);
    pub const DISJOINT: Self = Self(0b100_0000_0000_0000_0000_0000);
    pub const COSITED_CHROMA_SAMPLES: Self = Self(0b1000_0000_0000_0000_0000_0000);
    pub const STORAGE_READ_WITHOUT_FORMAT: Self = Self(0b1000_0000_0000_0000_0000_0000_0000_0000);
    pub const STORAGE_WRITE_WITHOUT_FORMAT: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const SAMPLED_IMAGE_DEPTH_COMPARISON: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderingFlagBits.html>"]
pub struct RenderingFlags(pub(crate) Flags);
vk_bitflags_wrapped!(RenderingFlags, Flags);
impl RenderingFlags {
    pub const CONTENTS_SECONDARY_COMMAND_BUFFERS: Self = Self(0b1);
    pub const SUSPENDING: Self = Self(0b10);
    pub const RESUMING: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeH265CapabilityFlagBitsKHR.html>"]
pub struct VideoEncodeH265CapabilityFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeH265CapabilityFlagsKHR, Flags);
impl VideoEncodeH265CapabilityFlagsKHR {
    pub const HRD_COMPLIANCE: Self = Self(0b1);
    pub const PREDICTION_WEIGHT_TABLE_GENERATED: Self = Self(0b10);
    pub const ROW_UNALIGNED_SLICE_SEGMENT: Self = Self(0b100);
    pub const DIFFERENT_SLICE_SEGMENT_TYPE: Self = Self(0b1000);
    pub const B_FRAME_IN_L0_LIST: Self = Self(0b1_0000);
    pub const B_FRAME_IN_L1_LIST: Self = Self(0b10_0000);
    pub const PER_PICTURE_TYPE_MIN_MAX_QP: Self = Self(0b100_0000);
    pub const PER_SLICE_SEGMENT_CONSTANT_QP: Self = Self(0b1000_0000);
    pub const MULTIPLE_TILES_PER_SLICE_SEGMENT: Self = Self(0b1_0000_0000);
    pub const MULTIPLE_SLICE_SEGMENTS_PER_TILE: Self = Self(0b10_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeH265StdFlagBitsKHR.html>"]
pub struct VideoEncodeH265StdFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeH265StdFlagsKHR, Flags);
impl VideoEncodeH265StdFlagsKHR {
    pub const SEPARATE_COLOR_PLANE_FLAG_SET: Self = Self(0b1);
    pub const SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET: Self = Self(0b10);
    pub const SCALING_LIST_DATA_PRESENT_FLAG_SET: Self = Self(0b100);
    pub const PCM_ENABLED_FLAG_SET: Self = Self(0b1000);
    pub const SPS_TEMPORAL_MVP_ENABLED_FLAG_SET: Self = Self(0b1_0000);
    pub const INIT_QP_MINUS26: Self = Self(0b10_0000);
    pub const WEIGHTED_PRED_FLAG_SET: Self = Self(0b100_0000);
    pub const WEIGHTED_BIPRED_FLAG_SET: Self = Self(0b1000_0000);
    pub const LOG2_PARALLEL_MERGE_LEVEL_MINUS2: Self = Self(0b1_0000_0000);
    pub const SIGN_DATA_HIDING_ENABLED_FLAG_SET: Self = Self(0b10_0000_0000);
    pub const TRANSFORM_SKIP_ENABLED_FLAG_SET: Self = Self(0b100_0000_0000);
    pub const TRANSFORM_SKIP_ENABLED_FLAG_UNSET: Self = Self(0b1000_0000_0000);
    pub const PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG_SET: Self = Self(0b1_0000_0000_0000);
    pub const TRANSQUANT_BYPASS_ENABLED_FLAG_SET: Self = Self(0b10_0000_0000_0000);
    pub const CONSTRAINED_INTRA_PRED_FLAG_SET: Self = Self(0b100_0000_0000_0000);
    pub const ENTROPY_CODING_SYNC_ENABLED_FLAG_SET: Self = Self(0b1000_0000_0000_0000);
    pub const DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG_SET: Self = Self(0b1_0000_0000_0000_0000);
    pub const DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG_SET: Self = Self(0b10_0000_0000_0000_0000);
    pub const DEPENDENT_SLICE_SEGMENT_FLAG_SET: Self = Self(0b100_0000_0000_0000_0000);
    pub const SLICE_QP_DELTA: Self = Self(0b1000_0000_0000_0000_0000);
    pub const DIFFERENT_SLICE_QP_DELTA: Self = Self(0b1_0000_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeH265RateControlFlagBitsKHR.html>"]
pub struct VideoEncodeH265RateControlFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeH265RateControlFlagsKHR, Flags);
impl VideoEncodeH265RateControlFlagsKHR {
    pub const ATTEMPT_HRD_COMPLIANCE: Self = Self(0b1);
    pub const REGULAR_GOP: Self = Self(0b10);
    pub const REFERENCE_PATTERN_FLAT: Self = Self(0b100);
    pub const REFERENCE_PATTERN_DYADIC: Self = Self(0b1000);
    pub const TEMPORAL_SUB_LAYER_PATTERN_DYADIC: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeH265CtbSizeFlagBitsKHR.html>"]
pub struct VideoEncodeH265CtbSizeFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeH265CtbSizeFlagsKHR, Flags);
impl VideoEncodeH265CtbSizeFlagsKHR {
    pub const TYPE_16: Self = Self(0b1);
    pub const TYPE_32: Self = Self(0b10);
    pub const TYPE_64: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeH265TransformBlockSizeFlagBitsKHR.html>"]
pub struct VideoEncodeH265TransformBlockSizeFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeH265TransformBlockSizeFlagsKHR, Flags);
impl VideoEncodeH265TransformBlockSizeFlagsKHR {
    pub const TYPE_4: Self = Self(0b1);
    pub const TYPE_8: Self = Self(0b10);
    pub const TYPE_16: Self = Self(0b100);
    pub const TYPE_32: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeAV1CapabilityFlagBitsKHR.html>"]
pub struct VideoEncodeAV1CapabilityFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeAV1CapabilityFlagsKHR, Flags);
impl VideoEncodeAV1CapabilityFlagsKHR {
    pub const PER_RATE_CONTROL_GROUP_MIN_MAX_Q_INDEX: Self = Self(0b1);
    pub const GENERATE_OBU_EXTENSION_HEADER: Self = Self(0b10);
    pub const PRIMARY_REFERENCE_CDF_ONLY: Self = Self(0b100);
    pub const FRAME_SIZE_OVERRIDE: Self = Self(0b1000);
    pub const MOTION_VECTOR_SCALING: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeAV1StdFlagBitsKHR.html>"]
pub struct VideoEncodeAV1StdFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeAV1StdFlagsKHR, Flags);
impl VideoEncodeAV1StdFlagsKHR {
    pub const UNIFORM_TILE_SPACING_FLAG_SET: Self = Self(0b1);
    pub const SKIP_MODE_PRESENT_UNSET: Self = Self(0b10);
    pub const PRIMARY_REF_FRAME: Self = Self(0b100);
    pub const DELTA_Q: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeAV1RateControlFlagBitsKHR.html>"]
pub struct VideoEncodeAV1RateControlFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeAV1RateControlFlagsKHR, Flags);
impl VideoEncodeAV1RateControlFlagsKHR {
    pub const REGULAR_GOP: Self = Self(0b1);
    pub const TEMPORAL_LAYER_PATTERN_DYADIC: Self = Self(0b10);
    pub const REFERENCE_PATTERN_FLAT: Self = Self(0b100);
    pub const REFERENCE_PATTERN_DYADIC: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeAV1SuperblockSizeFlagBitsKHR.html>"]
pub struct VideoEncodeAV1SuperblockSizeFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeAV1SuperblockSizeFlagsKHR, Flags);
impl VideoEncodeAV1SuperblockSizeFlagsKHR {
    pub const TYPE_64: Self = Self(0b1);
    pub const TYPE_128: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkExportMetalObjectTypeFlagBitsEXT.html>"]
pub struct ExportMetalObjectTypeFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(ExportMetalObjectTypeFlagsEXT, Flags);
impl ExportMetalObjectTypeFlagsEXT {
    pub const METAL_DEVICE: Self = Self(0b1);
    pub const METAL_COMMAND_QUEUE: Self = Self(0b10);
    pub const METAL_BUFFER: Self = Self(0b100);
    pub const METAL_TEXTURE: Self = Self(0b1000);
    pub const METAL_IOSURFACE: Self = Self(0b1_0000);
    pub const METAL_SHARED_EVENT: Self = Self(0b10_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkInstanceCreateFlagBits.html>"]
pub struct InstanceCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(InstanceCreateFlags, Flags);
impl InstanceCreateFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCompressionFlagBitsEXT.html>"]
pub struct ImageCompressionFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(ImageCompressionFlagsEXT, Flags);
impl ImageCompressionFlagsEXT {
    pub const DEFAULT: Self = Self(0);
    pub const FIXED_RATE_DEFAULT: Self = Self(0b1);
    pub const FIXED_RATE_EXPLICIT: Self = Self(0b10);
    pub const DISABLED: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCompressionFixedRateFlagBitsEXT.html>"]
pub struct ImageCompressionFixedRateFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(ImageCompressionFixedRateFlagsEXT, Flags);
impl ImageCompressionFixedRateFlagsEXT {
    pub const NONE: Self = Self(0);
    pub const TYPE_1BPC: Self = Self(0b1);
    pub const TYPE_2BPC: Self = Self(0b10);
    pub const TYPE_3BPC: Self = Self(0b100);
    pub const TYPE_4BPC: Self = Self(0b1000);
    pub const TYPE_5BPC: Self = Self(0b1_0000);
    pub const TYPE_6BPC: Self = Self(0b10_0000);
    pub const TYPE_7BPC: Self = Self(0b100_0000);
    pub const TYPE_8BPC: Self = Self(0b1000_0000);
    pub const TYPE_9BPC: Self = Self(0b1_0000_0000);
    pub const TYPE_10BPC: Self = Self(0b10_0000_0000);
    pub const TYPE_11BPC: Self = Self(0b100_0000_0000);
    pub const TYPE_12BPC: Self = Self(0b1000_0000_0000);
    pub const TYPE_13BPC: Self = Self(0b1_0000_0000_0000);
    pub const TYPE_14BPC: Self = Self(0b10_0000_0000_0000);
    pub const TYPE_15BPC: Self = Self(0b100_0000_0000_0000);
    pub const TYPE_16BPC: Self = Self(0b1000_0000_0000_0000);
    pub const TYPE_17BPC: Self = Self(0b1_0000_0000_0000_0000);
    pub const TYPE_18BPC: Self = Self(0b10_0000_0000_0000_0000);
    pub const TYPE_19BPC: Self = Self(0b100_0000_0000_0000_0000);
    pub const TYPE_20BPC: Self = Self(0b1000_0000_0000_0000_0000);
    pub const TYPE_21BPC: Self = Self(0b1_0000_0000_0000_0000_0000);
    pub const TYPE_22BPC: Self = Self(0b10_0000_0000_0000_0000_0000);
    pub const TYPE_23BPC: Self = Self(0b100_0000_0000_0000_0000_0000);
    pub const TYPE_24BPC: Self = Self(0b1000_0000_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkOpticalFlowGridSizeFlagBitsNV.html>"]
pub struct OpticalFlowGridSizeFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(OpticalFlowGridSizeFlagsNV, Flags);
impl OpticalFlowGridSizeFlagsNV {
    pub const UNKNOWN: Self = Self(0);
    pub const TYPE_1X1: Self = Self(0b1);
    pub const TYPE_2X2: Self = Self(0b10);
    pub const TYPE_4X4: Self = Self(0b100);
    pub const TYPE_8X8: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkOpticalFlowUsageFlagBitsNV.html>"]
pub struct OpticalFlowUsageFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(OpticalFlowUsageFlagsNV, Flags);
impl OpticalFlowUsageFlagsNV {
    pub const UNKNOWN: Self = Self(0);
    pub const INPUT: Self = Self(0b1);
    pub const OUTPUT: Self = Self(0b10);
    pub const HINT: Self = Self(0b100);
    pub const COST: Self = Self(0b1000);
    pub const GLOBAL_FLOW: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkOpticalFlowSessionCreateFlagBitsNV.html>"]
pub struct OpticalFlowSessionCreateFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(OpticalFlowSessionCreateFlagsNV, Flags);
impl OpticalFlowSessionCreateFlagsNV {
    pub const ENABLE_HINT: Self = Self(0b1);
    pub const ENABLE_COST: Self = Self(0b10);
    pub const ENABLE_GLOBAL_FLOW: Self = Self(0b100);
    pub const ALLOW_REGIONS: Self = Self(0b1000);
    pub const BOTH_DIRECTIONS: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkOpticalFlowExecuteFlagBitsNV.html>"]
pub struct OpticalFlowExecuteFlagsNV(pub(crate) Flags);
vk_bitflags_wrapped!(OpticalFlowExecuteFlagsNV, Flags);
impl OpticalFlowExecuteFlagsNV {
    pub const DISABLE_TEMPORAL_HINTS: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkBuildMicromapFlagBitsEXT.html>"]
pub struct BuildMicromapFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(BuildMicromapFlagsEXT, Flags);
impl BuildMicromapFlagsEXT {
    pub const PREFER_FAST_TRACE: Self = Self(0b1);
    pub const PREFER_FAST_BUILD: Self = Self(0b10);
    pub const ALLOW_COMPACTION: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkMicromapCreateFlagBitsEXT.html>"]
pub struct MicromapCreateFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(MicromapCreateFlagsEXT, Flags);
impl MicromapCreateFlagsEXT {
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceFaultFlagBitsKHR.html>"]
pub struct DeviceFaultFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(DeviceFaultFlagsKHR, Flags);
impl DeviceFaultFlagsKHR {
    pub const FLAG_DEVICE_LOST: Self = Self(0b1);
    pub const FLAG_MEMORY_ADDRESS: Self = Self(0b10);
    pub const FLAG_INSTRUCTION_ADDRESS: Self = Self(0b100);
    pub const FLAG_VENDOR: Self = Self(0b1000);
    pub const FLAG_WATCHDOG_TIMEOUT: Self = Self(0b1_0000);
    pub const FLAG_OVERFLOW: Self = Self(0b10_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkIndirectCommandsLayoutUsageFlagBitsEXT.html>"]
pub struct IndirectCommandsLayoutUsageFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(IndirectCommandsLayoutUsageFlagsEXT, Flags);
impl IndirectCommandsLayoutUsageFlagsEXT {
    pub const EXPLICIT_PREPROCESS: Self = Self(0b1);
    pub const UNORDERED_SEQUENCES: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkIndirectCommandsInputModeFlagBitsEXT.html>"]
pub struct IndirectCommandsInputModeFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(IndirectCommandsInputModeFlagsEXT, Flags);
impl IndirectCommandsInputModeFlagsEXT {
    pub const VULKAN_INDEX_BUFFER: Self = Self(0b1);
    pub const DXGI_INDEX_BUFFER: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderCreateFlagBitsEXT.html>"]
pub struct ShaderCreateFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(ShaderCreateFlagsEXT, Flags);
impl ShaderCreateFlagsEXT {
    pub const LINK_STAGE: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryUnmapFlagBits.html>"]
pub struct MemoryUnmapFlags(pub(crate) Flags);
vk_bitflags_wrapped!(MemoryUnmapFlags, Flags);
impl MemoryUnmapFlags {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkWaylandSurfaceCreateFlagBitsKHR.html>"]
pub struct WaylandSurfaceCreateFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(WaylandSurfaceCreateFlagsKHR, Flags);
impl WaylandSurfaceCreateFlagsKHR {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkAccessFlagBits3KHR.html>"]
pub struct AccessFlags3KHR(pub(crate) Flags64);
vk_bitflags_wrapped!(AccessFlags3KHR, Flags64);
impl AccessFlags3KHR {
    pub const NONE: Self = Self(0);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkTileShadingRenderPassFlagBitsQCOM.html>"]
pub struct TileShadingRenderPassFlagsQCOM(pub(crate) Flags);
vk_bitflags_wrapped!(TileShadingRenderPassFlagsQCOM, Flags);
impl TileShadingRenderPassFlagsQCOM {
    pub const ENABLE: Self = Self(0b1);
    pub const PER_TILE_EXECUTION: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkAddressCopyFlagBitsKHR.html>"]
pub struct AddressCopyFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(AddressCopyFlagsKHR, Flags);
impl AddressCopyFlagsKHR {
    pub const DEVICE_LOCAL: Self = Self(0b1);
    pub const SPARSE: Self = Self(0b10);
    pub const PROTECTED: Self = Self(0b100);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkTensorCreateFlagBitsARM.html>"]
pub struct TensorCreateFlagsARM(pub(crate) Flags64);
vk_bitflags_wrapped!(TensorCreateFlagsARM, Flags64);
impl TensorCreateFlagsARM {
    pub const MUTABLE_FORMAT: Self = Self(0b1);
    pub const PROTECTED: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkTensorUsageFlagBitsARM.html>"]
pub struct TensorUsageFlagsARM(pub(crate) Flags64);
vk_bitflags_wrapped!(TensorUsageFlagsARM, Flags64);
impl TensorUsageFlagsARM {
    #[doc = "Tensor written/read through shader descriptor"]
    pub const SHADER: Self = Self(0b10);
    #[doc = "Tensor can be src of a transfer operation"]
    pub const TRANSFER_SRC: Self = Self(0b100);
    #[doc = "Tensor can be dst of a transfer operation"]
    pub const TRANSFER_DST: Self = Self(0b1000);
    #[doc = "Tensor can be aliased with an image"]
    pub const IMAGE_ALIASING: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkTensorViewCreateFlagBitsARM.html>"]
pub struct TensorViewCreateFlagsARM(pub(crate) Flags64);
vk_bitflags_wrapped!(TensorViewCreateFlagsARM, Flags64);
impl TensorViewCreateFlagsARM {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphPipelineSessionCreateFlagBitsARM.html>"]
pub struct DataGraphPipelineSessionCreateFlagsARM(pub(crate) Flags64);
vk_bitflags_wrapped!(DataGraphPipelineSessionCreateFlagsARM, Flags64);
impl DataGraphPipelineSessionCreateFlagsARM {
    pub const PROTECTED: Self = Self(0b1);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphPipelineDispatchFlagBitsARM.html>"]
pub struct DataGraphPipelineDispatchFlagsARM(pub(crate) Flags64);
vk_bitflags_wrapped!(DataGraphPipelineDispatchFlagsARM, Flags64);
impl DataGraphPipelineDispatchFlagsARM {}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeRgbModelConversionFlagBitsVALVE.html>"]
pub struct VideoEncodeRgbModelConversionFlagsVALVE(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeRgbModelConversionFlagsVALVE, Flags);
impl VideoEncodeRgbModelConversionFlagsVALVE {
    pub const RGB_IDENTITY: Self = Self(0b1);
    pub const YCBCR_IDENTITY: Self = Self(0b10);
    pub const YCBCR_709: Self = Self(0b100);
    pub const YCBCR_601: Self = Self(0b1000);
    pub const YCBCR_2020: Self = Self(0b1_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeRgbRangeCompressionFlagBitsVALVE.html>"]
pub struct VideoEncodeRgbRangeCompressionFlagsVALVE(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeRgbRangeCompressionFlagsVALVE, Flags);
impl VideoEncodeRgbRangeCompressionFlagsVALVE {
    pub const FULL_RANGE: Self = Self(0b1);
    pub const NARROW_RANGE: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkVideoEncodeRgbChromaOffsetFlagBitsVALVE.html>"]
pub struct VideoEncodeRgbChromaOffsetFlagsVALVE(pub(crate) Flags);
vk_bitflags_wrapped!(VideoEncodeRgbChromaOffsetFlagsVALVE, Flags);
impl VideoEncodeRgbChromaOffsetFlagsVALVE {
    pub const COSITED_EVEN: Self = Self(0b1);
    pub const MIDPOINT: Self = Self(0b10);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkSpirvResourceTypeFlagBitsEXT.html>"]
pub struct SpirvResourceTypeFlagsEXT(pub(crate) Flags);
vk_bitflags_wrapped!(SpirvResourceTypeFlagsEXT, Flags);
impl SpirvResourceTypeFlagsEXT {
    pub const ALL: Self = Self(0x7fffffff);
    pub const SAMPLER: Self = Self(0b1);
    pub const SAMPLED_IMAGE: Self = Self(0b10);
    pub const READ_ONLY_IMAGE: Self = Self(0b100);
    pub const READ_WRITE_IMAGE: Self = Self(0b1000);
    pub const COMBINED_SAMPLED_IMAGE: Self = Self(0b1_0000);
    pub const UNIFORM_BUFFER: Self = Self(0b10_0000);
    pub const READ_ONLY_STORAGE_BUFFER: Self = Self(0b100_0000);
    pub const READ_WRITE_STORAGE_BUFFER: Self = Self(0b1000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkGpaSqShaderStageFlagBitsAMD.html>"]
pub struct GpaSqShaderStageFlagsAMD(pub(crate) Flags);
vk_bitflags_wrapped!(GpaSqShaderStageFlagsAMD, Flags);
impl GpaSqShaderStageFlagsAMD {
    pub const PS: Self = Self(0b1);
    pub const VS: Self = Self(0b10);
    pub const GS: Self = Self(0b100);
    pub const ES: Self = Self(0b1000);
    pub const HS: Self = Self(0b1_0000);
    pub const LS: Self = Self(0b10_0000);
    pub const CS: Self = Self(0b100_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkAddressCommandFlagBitsKHR.html>"]
pub struct AddressCommandFlagsKHR(pub(crate) Flags);
vk_bitflags_wrapped!(AddressCommandFlagsKHR, Flags);
impl AddressCommandFlagsKHR {
    pub const PROTECTED: Self = Self(0b1);
    pub const FULLY_BOUND: Self = Self(0b10);
    pub const STORAGE_BUFFER_USAGE: Self = Self(0b100);
    pub const UNKNOWN_STORAGE_BUFFER_USAGE: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphTOSAQualityFlagBitsARM.html>"]
pub struct DataGraphTOSAQualityFlagsARM(pub(crate) Flags);
vk_bitflags_wrapped!(DataGraphTOSAQualityFlagsARM, Flags);
impl DataGraphTOSAQualityFlagsARM {
    pub const ACCELERATED: Self = Self(0b1);
    pub const CONFORMANT: Self = Self(0b10);
    pub const EXPERIMENTAL: Self = Self(0b100);
    pub const DEPRECATED: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphOpticalFlowGridSizeFlagBitsARM.html>"]
pub struct DataGraphOpticalFlowGridSizeFlagsARM(pub(crate) Flags);
vk_bitflags_wrapped!(DataGraphOpticalFlowGridSizeFlagsARM, Flags);
impl DataGraphOpticalFlowGridSizeFlagsARM {
    pub const UNKNOWN: Self = Self(0);
    pub const TYPE_1X1: Self = Self(0b1);
    pub const TYPE_2X2: Self = Self(0b10);
    pub const TYPE_4X4: Self = Self(0b100);
    pub const TYPE_8X8: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphOpticalFlowImageUsageFlagBitsARM.html>"]
pub struct DataGraphOpticalFlowImageUsageFlagsARM(pub(crate) Flags);
vk_bitflags_wrapped!(DataGraphOpticalFlowImageUsageFlagsARM, Flags);
impl DataGraphOpticalFlowImageUsageFlagsARM {
    pub const UNKNOWN: Self = Self(0);
    pub const INPUT: Self = Self(0b1);
    pub const OUTPUT: Self = Self(0b10);
    pub const HINT: Self = Self(0b100);
    pub const COST: Self = Self(0b1000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphOpticalFlowCreateFlagBitsARM.html>"]
pub struct DataGraphOpticalFlowCreateFlagsARM(pub(crate) Flags);
vk_bitflags_wrapped!(DataGraphOpticalFlowCreateFlagsARM, Flags);
impl DataGraphOpticalFlowCreateFlagsARM {
    pub const ENABLE_HINT: Self = Self(0b1);
    pub const ENABLE_COST: Self = Self(0b10);
    pub const RESERVED_30: Self = Self(0b100_0000_0000_0000_0000_0000_0000_0000);
}
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc = "<https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphOpticalFlowExecuteFlagBitsARM.html>"]
pub struct DataGraphOpticalFlowExecuteFlagsARM(pub(crate) Flags);
vk_bitflags_wrapped!(DataGraphOpticalFlowExecuteFlagsARM, Flags);
impl DataGraphOpticalFlowExecuteFlagsARM {
    pub const DISABLE_TEMPORAL_HINTS: Self = Self(0b1);
    pub const INPUT_UNCHANGED: Self = Self(0b10);
    pub const REFERENCE_UNCHANGED: Self = Self(0b100);
    pub const INPUT_IS_PREVIOUS_REFERENCE: Self = Self(0b1000);
    pub const REFERENCE_IS_PREVIOUS_INPUT: Self = Self(0b1_0000);
}
