#![allow(unused_qualifications)]
use super::aliases::*;
use super::bitflags::*;
use super::definitions::*;
use super::enums::*;
use super::platform_types::*;
use core::ffi::*;
#[doc = "Generated from 'VK_KHR_surface'"]
impl ObjectType {
    pub const SURFACE_KHR: Self = Self(1_000_000_000);
}
#[doc = "Generated from 'VK_KHR_surface'"]
impl Result {
    pub const ERROR_SURFACE_LOST_KHR: Self = Self(-1_000_000_000);
    pub const ERROR_NATIVE_WINDOW_IN_USE_KHR: Self = Self(-1_000_000_001);
}
#[doc = "Generated from 'VK_KHR_swapchain'"]
impl ImageLayout {
    pub const PRESENT_SRC_KHR: Self = Self(1_000_001_002);
}
#[doc = "Generated from 'VK_KHR_swapchain'"]
impl ObjectType {
    pub const SWAPCHAIN_KHR: Self = Self(1_000_001_000);
}
#[doc = "Generated from 'VK_KHR_swapchain'"]
impl Result {
    pub const SUBOPTIMAL_KHR: Self = Self(1_000_001_003);
    pub const ERROR_OUT_OF_DATE_KHR: Self = Self(-1_000_001_004);
}
#[doc = "Generated from 'VK_KHR_swapchain'"]
impl StructureType {
    pub const SWAPCHAIN_CREATE_INFO_KHR: Self = Self(1_000_001_000);
    pub const PRESENT_INFO_KHR: Self = Self(1_000_001_001);
    pub const DEVICE_GROUP_PRESENT_CAPABILITIES_KHR: Self = Self(1_000_060_007);
    pub const IMAGE_SWAPCHAIN_CREATE_INFO_KHR: Self = Self(1_000_060_008);
    pub const BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR: Self = Self(1_000_060_009);
    pub const ACQUIRE_NEXT_IMAGE_INFO_KHR: Self = Self(1_000_060_010);
    pub const DEVICE_GROUP_PRESENT_INFO_KHR: Self = Self(1_000_060_011);
    pub const DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR: Self = Self(1_000_060_012);
}
#[doc = "Generated from 'VK_KHR_swapchain'"]
impl SwapchainCreateFlagsKHR {
    #[doc = "Allow images with VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT"]
    pub const SPLIT_INSTANCE_BIND_REGIONS: Self = Self(0b1);
    #[doc = "Swapchain is protected"]
    pub const PROTECTED: Self = Self(0b10);
}
#[doc = "Generated from 'VK_KHR_display'"]
impl ObjectType {
    pub const DISPLAY_KHR: Self = Self(1_000_002_000);
    pub const DISPLAY_MODE_KHR: Self = Self(1_000_002_001);
}
#[doc = "Generated from 'VK_KHR_display'"]
impl StructureType {
    pub const DISPLAY_MODE_CREATE_INFO_KHR: Self = Self(1_000_002_000);
    pub const DISPLAY_SURFACE_CREATE_INFO_KHR: Self = Self(1_000_002_001);
}
#[doc = "Generated from 'VK_KHR_display_swapchain'"]
impl Result {
    pub const ERROR_INCOMPATIBLE_DISPLAY_KHR: Self = Self(-1_000_003_001);
}
#[doc = "Generated from 'VK_KHR_display_swapchain'"]
impl StructureType {
    pub const DISPLAY_PRESENT_INFO_KHR: Self = Self(1_000_003_000);
}
#[doc = "Generated from 'VK_KHR_xlib_surface'"]
impl StructureType {
    pub const XLIB_SURFACE_CREATE_INFO_KHR: Self = Self(1_000_004_000);
}
#[doc = "Generated from 'VK_KHR_xcb_surface'"]
impl StructureType {
    pub const XCB_SURFACE_CREATE_INFO_KHR: Self = Self(1_000_005_000);
}
#[doc = "Generated from 'VK_KHR_wayland_surface'"]
impl StructureType {
    pub const WAYLAND_SURFACE_CREATE_INFO_KHR: Self = Self(1_000_006_000);
}
#[doc = "Generated from 'VK_KHR_android_surface'"]
impl StructureType {
    pub const ANDROID_SURFACE_CREATE_INFO_KHR: Self = Self(1_000_008_000);
}
#[doc = "Generated from 'VK_KHR_win32_surface'"]
impl StructureType {
    pub const WIN32_SURFACE_CREATE_INFO_KHR: Self = Self(1_000_009_000);
}
#[doc = "Generated from 'VK_ANDROID_native_buffer'"]
impl StructureType {
    pub const NATIVE_BUFFER_ANDROID: Self = Self(1_000_010_000);
    pub const SWAPCHAIN_IMAGE_CREATE_INFO_ANDROID: Self = Self(1_000_010_001);
    pub const PHYSICAL_DEVICE_PRESENTATION_PROPERTIES_ANDROID: Self = Self(1_000_010_002);
}
#[doc = "Generated from 'VK_EXT_debug_report'"]
impl DebugReportObjectTypeEXT {
    pub const SAMPLER_YCBCR_CONVERSION: Self = Self(1_000_156_000);
    pub const DESCRIPTOR_UPDATE_TEMPLATE: Self = Self(1_000_085_000);
}
#[doc = "Generated from 'VK_EXT_debug_report'"]
impl ObjectType {
    pub const DEBUG_REPORT_CALLBACK_EXT: Self = Self(1_000_011_000);
}
#[doc = "Generated from 'VK_EXT_debug_report'"]
impl Result {
    pub const ERROR_VALIDATION_FAILED_EXT: Self = Self::ERROR_VALIDATION_FAILED;
}
#[doc = "Generated from 'VK_EXT_debug_report'"]
impl StructureType {
    pub const DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT: Self = Self(1_000_011_000);
}
#[doc = "Generated from 'VK_NV_glsl_shader'"]
impl Result {
    pub const ERROR_INVALID_SHADER_NV: Self = Self(-1_000_012_000);
}
#[doc = "Generated from 'VK_KHR_sampler_mirror_clamp_to_edge'"]
impl SamplerAddressMode {
    #[doc = "Note that this defines what was previously a core enum, and so uses the 'value' attribute rather than 'offset', and does not have a suffix. This is a special case, and should not be repeated"]
    pub const MIRROR_CLAMP_TO_EDGE: Self = Self(4);
}
#[doc = "Generated from 'VK_IMG_filter_cubic'"]
impl Filter {
    pub const CUBIC_IMG: Self = Self::CUBIC_EXT;
}
#[doc = "Generated from 'VK_IMG_filter_cubic'"]
impl FormatFeatureFlags {
    #[doc = "Format can be filtered with VK_FILTER_CUBIC_IMG when being sampled"]
    pub const SAMPLED_IMAGE_FILTER_CUBIC_IMG: Self = Self::SAMPLED_IMAGE_FILTER_CUBIC_EXT;
}
#[doc = "Generated from 'VK_AMD_rasterization_order'"]
impl StructureType {
    pub const PIPELINE_RASTERIZATION_STATE_RASTERIZATION_ORDER_AMD: Self = Self(1_000_018_000);
}
#[doc = "Generated from 'VK_EXT_debug_marker'"]
impl StructureType {
    pub const DEBUG_MARKER_OBJECT_NAME_INFO_EXT: Self = Self(1_000_022_000);
    pub const DEBUG_MARKER_OBJECT_TAG_INFO_EXT: Self = Self(1_000_022_001);
    pub const DEBUG_MARKER_MARKER_INFO_EXT: Self = Self(1_000_022_002);
}
#[doc = "Generated from 'VK_KHR_video_queue'"]
impl ObjectType {
    #[doc = "VkVideoSessionKHR"]
    pub const VIDEO_SESSION_KHR: Self = Self(1_000_023_000);
    #[doc = "VkVideoSessionParametersKHR"]
    pub const VIDEO_SESSION_PARAMETERS_KHR: Self = Self(1_000_023_001);
}
#[doc = "Generated from 'VK_KHR_video_queue'"]
impl QueryResultFlags {
    pub const WITH_STATUS_KHR: Self = Self(0b1_0000);
}
#[doc = "Generated from 'VK_KHR_video_queue'"]
impl QueryType {
    pub const RESULT_STATUS_ONLY_KHR: Self = Self(1_000_023_000);
}
#[doc = "Generated from 'VK_KHR_video_queue'"]
impl Result {
    pub const ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR: Self = Self(-1_000_023_000);
    pub const ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR: Self = Self(-1_000_023_001);
    pub const ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR: Self = Self(-1_000_023_002);
    pub const ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR: Self = Self(-1_000_023_003);
    pub const ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR: Self = Self(-1_000_023_004);
    pub const ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR: Self = Self(-1_000_023_005);
}
#[doc = "Generated from 'VK_KHR_video_queue'"]
impl StructureType {
    pub const VIDEO_PROFILE_INFO_KHR: Self = Self(1_000_023_000);
    pub const VIDEO_CAPABILITIES_KHR: Self = Self(1_000_023_001);
    pub const VIDEO_PICTURE_RESOURCE_INFO_KHR: Self = Self(1_000_023_002);
    pub const VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR: Self = Self(1_000_023_003);
    pub const BIND_VIDEO_SESSION_MEMORY_INFO_KHR: Self = Self(1_000_023_004);
    pub const VIDEO_SESSION_CREATE_INFO_KHR: Self = Self(1_000_023_005);
    pub const VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR: Self = Self(1_000_023_006);
    pub const VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR: Self = Self(1_000_023_007);
    pub const VIDEO_BEGIN_CODING_INFO_KHR: Self = Self(1_000_023_008);
    pub const VIDEO_END_CODING_INFO_KHR: Self = Self(1_000_023_009);
    pub const VIDEO_CODING_CONTROL_INFO_KHR: Self = Self(1_000_023_010);
    pub const VIDEO_REFERENCE_SLOT_INFO_KHR: Self = Self(1_000_023_011);
    pub const QUEUE_FAMILY_VIDEO_PROPERTIES_KHR: Self = Self(1_000_023_012);
    pub const VIDEO_PROFILE_LIST_INFO_KHR: Self = Self(1_000_023_013);
    pub const PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR: Self = Self(1_000_023_014);
    pub const VIDEO_FORMAT_PROPERTIES_KHR: Self = Self(1_000_023_015);
    pub const QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES_KHR: Self = Self(1_000_023_016);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl AccessFlags2 {
    pub const VIDEO_DECODE_READ_KHR: Self = Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const VIDEO_DECODE_WRITE_KHR: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl BufferUsageFlags {
    pub const VIDEO_DECODE_SRC_KHR: Self = Self(0b10_0000_0000_0000);
    pub const VIDEO_DECODE_DST_KHR: Self = Self(0b100_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl FormatFeatureFlags {
    pub const VIDEO_DECODE_OUTPUT_KHR: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
    pub const VIDEO_DECODE_DPB_KHR: Self = Self(0b100_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl FormatFeatureFlags2 {
    pub const VIDEO_DECODE_OUTPUT_KHR: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
    pub const VIDEO_DECODE_DPB_KHR: Self = Self(0b100_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl ImageLayout {
    pub const VIDEO_DECODE_DST_KHR: Self = Self(1_000_024_000);
    pub const VIDEO_DECODE_SRC_KHR: Self = Self(1_000_024_001);
    pub const VIDEO_DECODE_DPB_KHR: Self = Self(1_000_024_002);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl ImageUsageFlags {
    pub const VIDEO_DECODE_DST_KHR: Self = Self(0b100_0000_0000);
    pub const VIDEO_DECODE_SRC_KHR: Self = Self(0b1000_0000_0000);
    pub const VIDEO_DECODE_DPB_KHR: Self = Self(0b1_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl PipelineStageFlags2 {
    pub const VIDEO_DECODE_KHR: Self = Self(0b100_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl QueueFlags {
    pub const VIDEO_DECODE_KHR: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_queue'"]
impl StructureType {
    pub const VIDEO_DECODE_INFO_KHR: Self = Self(1_000_024_000);
    pub const VIDEO_DECODE_CAPABILITIES_KHR: Self = Self(1_000_024_001);
    pub const VIDEO_DECODE_USAGE_INFO_KHR: Self = Self(1_000_024_002);
}
#[doc = "Generated from 'VK_NV_dedicated_allocation'"]
impl StructureType {
    pub const DEDICATED_ALLOCATION_IMAGE_CREATE_INFO_NV: Self = Self(1_000_026_000);
    pub const DEDICATED_ALLOCATION_BUFFER_CREATE_INFO_NV: Self = Self(1_000_026_001);
    pub const DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO_NV: Self = Self(1_000_026_002);
}
#[doc = "Generated from 'VK_EXT_transform_feedback'"]
impl AccessFlags {
    pub const TRANSFORM_FEEDBACK_WRITE_EXT: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
    pub const TRANSFORM_FEEDBACK_COUNTER_READ_EXT: Self = Self(0b100_0000_0000_0000_0000_0000_0000);
    pub const TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_transform_feedback'"]
impl BufferUsageFlags {
    pub const TRANSFORM_FEEDBACK_BUFFER_EXT: Self = Self(0b1000_0000_0000);
    pub const TRANSFORM_FEEDBACK_COUNTER_BUFFER_EXT: Self = Self(0b1_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_transform_feedback'"]
impl PipelineStageFlags {
    pub const TRANSFORM_FEEDBACK_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_transform_feedback'"]
impl QueryType {
    pub const TRANSFORM_FEEDBACK_STREAM_EXT: Self = Self(1_000_028_004);
}
#[doc = "Generated from 'VK_EXT_transform_feedback'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT: Self = Self(1_000_028_000);
    pub const PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT: Self = Self(1_000_028_001);
    pub const PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO_EXT: Self = Self(1_000_028_002);
}
#[doc = "Generated from 'VK_NVX_binary_import'"]
impl DebugReportObjectTypeEXT {
    pub const CU_MODULE_NVX: Self = Self(1_000_029_000);
    pub const CU_FUNCTION_NVX: Self = Self(1_000_029_001);
}
#[doc = "Generated from 'VK_NVX_binary_import'"]
impl ObjectType {
    pub const CU_MODULE_NVX: Self = Self(1_000_029_000);
    pub const CU_FUNCTION_NVX: Self = Self(1_000_029_001);
}
#[doc = "Generated from 'VK_NVX_binary_import'"]
impl StructureType {
    pub const CU_MODULE_CREATE_INFO_NVX: Self = Self(1_000_029_000);
    pub const CU_FUNCTION_CREATE_INFO_NVX: Self = Self(1_000_029_001);
    pub const CU_LAUNCH_INFO_NVX: Self = Self(1_000_029_002);
    pub const CU_MODULE_TEXTURING_MODE_CREATE_INFO_NVX: Self = Self(1_000_029_004);
}
#[doc = "Generated from 'VK_NVX_image_view_handle'"]
impl StructureType {
    pub const IMAGE_VIEW_HANDLE_INFO_NVX: Self = Self(1_000_030_000);
    pub const IMAGE_VIEW_ADDRESS_PROPERTIES_NVX: Self = Self(1_000_030_001);
}
#[doc = "Generated from 'VK_KHR_video_encode_h264'"]
impl StructureType {
    pub const VIDEO_ENCODE_H264_CAPABILITIES_KHR: Self = Self(1_000_038_000);
    pub const VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR: Self = Self(1_000_038_001);
    pub const VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR: Self = Self(1_000_038_002);
    pub const VIDEO_ENCODE_H264_PICTURE_INFO_KHR: Self = Self(1_000_038_003);
    pub const VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR: Self = Self(1_000_038_004);
    pub const VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR: Self = Self(1_000_038_005);
    pub const VIDEO_ENCODE_H264_GOP_REMAINING_FRAME_INFO_KHR: Self = Self(1_000_038_006);
    pub const VIDEO_ENCODE_H264_PROFILE_INFO_KHR: Self = Self(1_000_038_007);
    pub const VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR: Self = Self(1_000_038_008);
    pub const VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR: Self = Self(1_000_038_009);
    pub const VIDEO_ENCODE_H264_SESSION_CREATE_INFO_KHR: Self = Self(1_000_038_010);
    pub const VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES_KHR: Self = Self(1_000_038_011);
    pub const VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR: Self = Self(1_000_038_012);
    pub const VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO_KHR: Self = Self(1_000_038_013);
}
#[doc = "Generated from 'VK_KHR_video_encode_h264'"]
impl VideoCodecOperationFlagsKHR {
    pub const ENCODE_H264: Self = Self(0b1_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_h265'"]
impl StructureType {
    pub const VIDEO_ENCODE_H265_CAPABILITIES_KHR: Self = Self(1_000_039_000);
    pub const VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR: Self = Self(1_000_039_001);
    pub const VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR: Self = Self(1_000_039_002);
    pub const VIDEO_ENCODE_H265_PICTURE_INFO_KHR: Self = Self(1_000_039_003);
    pub const VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR: Self = Self(1_000_039_004);
    pub const VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO_KHR: Self = Self(1_000_039_005);
    pub const VIDEO_ENCODE_H265_GOP_REMAINING_FRAME_INFO_KHR: Self = Self(1_000_039_006);
    pub const VIDEO_ENCODE_H265_PROFILE_INFO_KHR: Self = Self(1_000_039_007);
    pub const VIDEO_ENCODE_H265_RATE_CONTROL_INFO_KHR: Self = Self(1_000_039_009);
    pub const VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_KHR: Self = Self(1_000_039_010);
    pub const VIDEO_ENCODE_H265_SESSION_CREATE_INFO_KHR: Self = Self(1_000_039_011);
    pub const VIDEO_ENCODE_H265_QUALITY_LEVEL_PROPERTIES_KHR: Self = Self(1_000_039_012);
    pub const VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO_KHR: Self = Self(1_000_039_013);
    pub const VIDEO_ENCODE_H265_SESSION_PARAMETERS_FEEDBACK_INFO_KHR: Self = Self(1_000_039_014);
}
#[doc = "Generated from 'VK_KHR_video_encode_h265'"]
impl VideoCodecOperationFlagsKHR {
    pub const ENCODE_H265: Self = Self(0b10_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_h264'"]
impl StructureType {
    pub const VIDEO_DECODE_H264_CAPABILITIES_KHR: Self = Self(1_000_040_000);
    pub const VIDEO_DECODE_H264_PICTURE_INFO_KHR: Self = Self(1_000_040_001);
    pub const VIDEO_DECODE_H264_PROFILE_INFO_KHR: Self = Self(1_000_040_003);
    pub const VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR: Self = Self(1_000_040_004);
    pub const VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR: Self = Self(1_000_040_005);
    pub const VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR: Self = Self(1_000_040_006);
}
#[doc = "Generated from 'VK_KHR_video_decode_h264'"]
impl VideoCodecOperationFlagsKHR {
    pub const DECODE_H264: Self = Self(0b1);
}
#[doc = "Generated from 'VK_AMD_texture_gather_bias_lod'"]
impl StructureType {
    pub const TEXTURE_LOD_GATHER_FORMAT_PROPERTIES_AMD: Self = Self(1_000_041_000);
}
#[doc = "Generated from 'VK_KHR_dynamic_rendering'"]
impl AttachmentStoreOp {
    pub const NONE_KHR: Self = Self::NONE;
}
#[doc = "Generated from 'VK_KHR_dynamic_rendering'"]
impl RenderingFlags {
    pub const CONTENTS_SECONDARY_COMMAND_BUFFERS_KHR: Self =
        Self::CONTENTS_SECONDARY_COMMAND_BUFFERS;
    pub const SUSPENDING_KHR: Self = Self::SUSPENDING;
    pub const RESUMING_KHR: Self = Self::RESUMING;
}
#[doc = "Generated from 'VK_KHR_dynamic_rendering'"]
impl StructureType {
    pub const RENDERING_INFO_KHR: Self = Self::RENDERING_INFO;
    pub const RENDERING_ATTACHMENT_INFO_KHR: Self = Self::RENDERING_ATTACHMENT_INFO;
    pub const PIPELINE_RENDERING_CREATE_INFO_KHR: Self = Self::PIPELINE_RENDERING_CREATE_INFO;
    pub const PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    pub const COMMAND_BUFFER_INHERITANCE_RENDERING_INFO_KHR: Self =
        Self::COMMAND_BUFFER_INHERITANCE_RENDERING_INFO;
}
#[doc = "Generated from 'VK_GGP_stream_descriptor_surface'"]
impl StructureType {
    pub const STREAM_DESCRIPTOR_SURFACE_CREATE_INFO_GGP: Self = Self(1_000_049_000);
}
#[doc = "Generated from 'VK_NV_corner_sampled_image'"]
impl ImageCreateFlags {
    pub const CORNER_SAMPLED_NV: Self = Self(0b10_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_corner_sampled_image'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_CORNER_SAMPLED_IMAGE_FEATURES_NV: Self = Self(1_000_050_000);
}
#[doc = "Generated from 'VK_KHR_multiview'"]
impl DependencyFlags {
    pub const VIEW_LOCAL_KHR: Self = Self::VIEW_LOCAL;
}
#[doc = "Generated from 'VK_KHR_multiview'"]
impl StructureType {
    pub const RENDER_PASS_MULTIVIEW_CREATE_INFO_KHR: Self = Self::RENDER_PASS_MULTIVIEW_CREATE_INFO;
    pub const PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
    pub const PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES;
}
#[doc = "Generated from 'VK_IMG_format_pvrtc'"]
impl Format {
    pub const PVRTC1_2BPP_UNORM_BLOCK_IMG: Self = Self(1_000_054_000);
    pub const PVRTC1_4BPP_UNORM_BLOCK_IMG: Self = Self(1_000_054_001);
    pub const PVRTC2_2BPP_UNORM_BLOCK_IMG: Self = Self(1_000_054_002);
    pub const PVRTC2_4BPP_UNORM_BLOCK_IMG: Self = Self(1_000_054_003);
    pub const PVRTC1_2BPP_SRGB_BLOCK_IMG: Self = Self(1_000_054_004);
    pub const PVRTC1_4BPP_SRGB_BLOCK_IMG: Self = Self(1_000_054_005);
    pub const PVRTC2_2BPP_SRGB_BLOCK_IMG: Self = Self(1_000_054_006);
    pub const PVRTC2_4BPP_SRGB_BLOCK_IMG: Self = Self(1_000_054_007);
}
#[doc = "Generated from 'VK_NV_external_memory'"]
impl StructureType {
    pub const EXTERNAL_MEMORY_IMAGE_CREATE_INFO_NV: Self = Self(1_000_056_000);
    pub const EXPORT_MEMORY_ALLOCATE_INFO_NV: Self = Self(1_000_056_001);
}
#[doc = "Generated from 'VK_NV_external_memory_win32'"]
impl StructureType {
    pub const IMPORT_MEMORY_WIN32_HANDLE_INFO_NV: Self = Self(1_000_057_000);
    pub const EXPORT_MEMORY_WIN32_HANDLE_INFO_NV: Self = Self(1_000_057_001);
}
#[doc = "Generated from 'VK_NV_win32_keyed_mutex'"]
impl StructureType {
    pub const WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_NV: Self = Self(1_000_058_000);
}
#[doc = "Generated from 'VK_KHR_get_physical_device_properties2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FEATURES_2_KHR: Self = Self::PHYSICAL_DEVICE_FEATURES_2;
    pub const PHYSICAL_DEVICE_PROPERTIES_2_KHR: Self = Self::PHYSICAL_DEVICE_PROPERTIES_2;
    pub const FORMAT_PROPERTIES_2_KHR: Self = Self::FORMAT_PROPERTIES_2;
    pub const IMAGE_FORMAT_PROPERTIES_2_KHR: Self = Self::IMAGE_FORMAT_PROPERTIES_2;
    pub const PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR: Self =
        Self::PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    pub const QUEUE_FAMILY_PROPERTIES_2_KHR: Self = Self::QUEUE_FAMILY_PROPERTIES_2;
    pub const PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR: Self =
        Self::PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    pub const SPARSE_IMAGE_FORMAT_PROPERTIES_2_KHR: Self = Self::SPARSE_IMAGE_FORMAT_PROPERTIES_2;
    pub const PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2_KHR: Self =
        Self::PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2;
}
#[doc = "Generated from 'VK_KHR_device_group'"]
impl DependencyFlags {
    pub const DEVICE_GROUP_KHR: Self = Self::DEVICE_GROUP;
}
#[doc = "Generated from 'VK_KHR_device_group'"]
impl ImageCreateFlags {
    pub const SPLIT_INSTANCE_BIND_REGIONS_KHR: Self = Self::SPLIT_INSTANCE_BIND_REGIONS;
}
#[doc = "Generated from 'VK_KHR_device_group'"]
impl MemoryAllocateFlags {
    pub const DEVICE_MASK_KHR: Self = Self::DEVICE_MASK;
}
#[doc = "Generated from 'VK_KHR_device_group'"]
impl PeerMemoryFeatureFlags {
    pub const COPY_SRC_KHR: Self = Self::COPY_SRC;
    pub const COPY_DST_KHR: Self = Self::COPY_DST;
    pub const GENERIC_SRC_KHR: Self = Self::GENERIC_SRC;
    pub const GENERIC_DST_KHR: Self = Self::GENERIC_DST;
}
#[doc = "Generated from 'VK_KHR_device_group'"]
impl PipelineCreateFlags {
    pub const VIEW_INDEX_FROM_DEVICE_INDEX_KHR: Self = Self::VIEW_INDEX_FROM_DEVICE_INDEX;
}
#[doc = "Generated from 'VK_KHR_device_group'"]
impl StructureType {
    pub const MEMORY_ALLOCATE_FLAGS_INFO_KHR: Self = Self::MEMORY_ALLOCATE_FLAGS_INFO;
    pub const DEVICE_GROUP_RENDER_PASS_BEGIN_INFO_KHR: Self =
        Self::DEVICE_GROUP_RENDER_PASS_BEGIN_INFO;
    pub const DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO_KHR: Self =
        Self::DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO;
    pub const DEVICE_GROUP_SUBMIT_INFO_KHR: Self = Self::DEVICE_GROUP_SUBMIT_INFO;
    pub const DEVICE_GROUP_BIND_SPARSE_INFO_KHR: Self = Self::DEVICE_GROUP_BIND_SPARSE_INFO;
    pub const BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO_KHR: Self =
        Self::BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO;
    pub const BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO_KHR: Self =
        Self::BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO;
}
#[doc = "Generated from 'VK_EXT_validation_flags'"]
impl StructureType {
    pub const VALIDATION_FLAGS_EXT: Self = Self(1_000_061_000);
}
#[doc = "Generated from 'VK_NN_vi_surface'"]
impl StructureType {
    pub const VI_SURFACE_CREATE_INFO_NN: Self = Self(1_000_062_000);
}
#[doc = "Generated from 'VK_EXT_texture_compression_astc_hdr'"]
impl Format {
    pub const ASTC_4X4_SFLOAT_BLOCK_EXT: Self = Self::ASTC_4X4_SFLOAT_BLOCK;
    pub const ASTC_5X4_SFLOAT_BLOCK_EXT: Self = Self::ASTC_5X4_SFLOAT_BLOCK;
    pub const ASTC_5X5_SFLOAT_BLOCK_EXT: Self = Self::ASTC_5X5_SFLOAT_BLOCK;
    pub const ASTC_6X5_SFLOAT_BLOCK_EXT: Self = Self::ASTC_6X5_SFLOAT_BLOCK;
    pub const ASTC_6X6_SFLOAT_BLOCK_EXT: Self = Self::ASTC_6X6_SFLOAT_BLOCK;
    pub const ASTC_8X5_SFLOAT_BLOCK_EXT: Self = Self::ASTC_8X5_SFLOAT_BLOCK;
    pub const ASTC_8X6_SFLOAT_BLOCK_EXT: Self = Self::ASTC_8X6_SFLOAT_BLOCK;
    pub const ASTC_8X8_SFLOAT_BLOCK_EXT: Self = Self::ASTC_8X8_SFLOAT_BLOCK;
    pub const ASTC_10X5_SFLOAT_BLOCK_EXT: Self = Self::ASTC_10X5_SFLOAT_BLOCK;
    pub const ASTC_10X6_SFLOAT_BLOCK_EXT: Self = Self::ASTC_10X6_SFLOAT_BLOCK;
    pub const ASTC_10X8_SFLOAT_BLOCK_EXT: Self = Self::ASTC_10X8_SFLOAT_BLOCK;
    pub const ASTC_10X10_SFLOAT_BLOCK_EXT: Self = Self::ASTC_10X10_SFLOAT_BLOCK;
    pub const ASTC_12X10_SFLOAT_BLOCK_EXT: Self = Self::ASTC_12X10_SFLOAT_BLOCK;
    pub const ASTC_12X12_SFLOAT_BLOCK_EXT: Self = Self::ASTC_12X12_SFLOAT_BLOCK;
}
#[doc = "Generated from 'VK_EXT_texture_compression_astc_hdr'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES;
}
#[doc = "Generated from 'VK_EXT_astc_decode_mode'"]
impl StructureType {
    pub const IMAGE_VIEW_ASTC_DECODE_MODE_EXT: Self = Self(1_000_067_000);
    pub const PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT: Self = Self(1_000_067_001);
}
#[doc = "Generated from 'VK_EXT_pipeline_robustness'"]
impl PipelineRobustnessBufferBehavior {
    pub const DEVICE_DEFAULT_EXT: Self = Self::DEVICE_DEFAULT;
    pub const DISABLED_EXT: Self = Self::DISABLED;
    pub const ROBUST_BUFFER_ACCESS_EXT: Self = Self::ROBUST_BUFFER_ACCESS;
    pub const ROBUST_BUFFER_ACCESS_2_EXT: Self = Self::ROBUST_BUFFER_ACCESS_2;
}
#[doc = "Generated from 'VK_EXT_pipeline_robustness'"]
impl PipelineRobustnessImageBehavior {
    pub const DEVICE_DEFAULT_EXT: Self = Self::DEVICE_DEFAULT;
    pub const DISABLED_EXT: Self = Self::DISABLED;
    pub const ROBUST_IMAGE_ACCESS_EXT: Self = Self::ROBUST_IMAGE_ACCESS;
    pub const ROBUST_IMAGE_ACCESS_2_EXT: Self = Self::ROBUST_IMAGE_ACCESS_2;
}
#[doc = "Generated from 'VK_EXT_pipeline_robustness'"]
impl StructureType {
    pub const PIPELINE_ROBUSTNESS_CREATE_INFO_EXT: Self = Self::PIPELINE_ROBUSTNESS_CREATE_INFO;
    pub const PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES;
    pub const PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES;
}
#[doc = "Generated from 'VK_KHR_maintenance1'"]
impl FormatFeatureFlags {
    pub const TRANSFER_SRC_KHR: Self = Self::TRANSFER_SRC;
    pub const TRANSFER_DST_KHR: Self = Self::TRANSFER_DST;
}
#[doc = "Generated from 'VK_KHR_maintenance1'"]
impl ImageCreateFlags {
    pub const TYPE_2D_ARRAY_COMPATIBLE_KHR: Self = Self::TYPE_2D_ARRAY_COMPATIBLE;
}
#[doc = "Generated from 'VK_KHR_maintenance1'"]
impl Result {
    pub const ERROR_OUT_OF_POOL_MEMORY_KHR: Self = Self::ERROR_OUT_OF_POOL_MEMORY;
}
#[doc = "Generated from 'VK_KHR_device_group_creation'"]
impl MemoryHeapFlags {
    pub const MULTI_INSTANCE_KHR: Self = Self::MULTI_INSTANCE;
}
#[doc = "Generated from 'VK_KHR_device_group_creation'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_GROUP_PROPERTIES_KHR: Self = Self::PHYSICAL_DEVICE_GROUP_PROPERTIES;
    pub const DEVICE_GROUP_DEVICE_CREATE_INFO_KHR: Self = Self::DEVICE_GROUP_DEVICE_CREATE_INFO;
}
#[doc = "Generated from 'VK_KHR_external_memory_capabilities'"]
impl ExternalMemoryFeatureFlags {
    pub const DEDICATED_ONLY_KHR: Self = Self::DEDICATED_ONLY;
    pub const EXPORTABLE_KHR: Self = Self::EXPORTABLE;
    pub const IMPORTABLE_KHR: Self = Self::IMPORTABLE;
}
#[doc = "Generated from 'VK_KHR_external_memory_capabilities'"]
impl ExternalMemoryHandleTypeFlags {
    pub const OPAQUE_FD_KHR: Self = Self::OPAQUE_FD;
    pub const OPAQUE_WIN32_KHR: Self = Self::OPAQUE_WIN32;
    pub const OPAQUE_WIN32_KMT_KHR: Self = Self::OPAQUE_WIN32_KMT;
    pub const D3D11_TEXTURE_KHR: Self = Self::D3D11_TEXTURE;
    pub const D3D11_TEXTURE_KMT_KHR: Self = Self::D3D11_TEXTURE_KMT;
    pub const D3D12_HEAP_KHR: Self = Self::D3D12_HEAP;
    pub const D3D12_RESOURCE_KHR: Self = Self::D3D12_RESOURCE;
}
#[doc = "Generated from 'VK_KHR_external_memory_capabilities'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR: Self =
        Self::PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
    pub const EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR: Self = Self::EXTERNAL_IMAGE_FORMAT_PROPERTIES;
    pub const PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR: Self =
        Self::PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    pub const EXTERNAL_BUFFER_PROPERTIES_KHR: Self = Self::EXTERNAL_BUFFER_PROPERTIES;
    pub const PHYSICAL_DEVICE_ID_PROPERTIES_KHR: Self = Self::PHYSICAL_DEVICE_ID_PROPERTIES;
}
#[doc = "Generated from 'VK_KHR_external_memory'"]
impl Result {
    pub const ERROR_INVALID_EXTERNAL_HANDLE_KHR: Self = Self::ERROR_INVALID_EXTERNAL_HANDLE;
}
#[doc = "Generated from 'VK_KHR_external_memory'"]
impl StructureType {
    pub const EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR: Self =
        Self::EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    pub const EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR: Self = Self::EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    pub const EXPORT_MEMORY_ALLOCATE_INFO_KHR: Self = Self::EXPORT_MEMORY_ALLOCATE_INFO;
}
#[doc = "Generated from 'VK_KHR_external_memory_win32'"]
impl StructureType {
    pub const IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_073_000);
    pub const EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_073_001);
    pub const MEMORY_WIN32_HANDLE_PROPERTIES_KHR: Self = Self(1_000_073_002);
    pub const MEMORY_GET_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_073_003);
}
#[doc = "Generated from 'VK_KHR_external_memory_fd'"]
impl StructureType {
    pub const IMPORT_MEMORY_FD_INFO_KHR: Self = Self(1_000_074_000);
    pub const MEMORY_FD_PROPERTIES_KHR: Self = Self(1_000_074_001);
    pub const MEMORY_GET_FD_INFO_KHR: Self = Self(1_000_074_002);
}
#[doc = "Generated from 'VK_KHR_win32_keyed_mutex'"]
impl StructureType {
    pub const WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_KHR: Self = Self(1_000_075_000);
}
#[doc = "Generated from 'VK_KHR_external_semaphore_capabilities'"]
impl ExternalSemaphoreFeatureFlags {
    pub const EXPORTABLE_KHR: Self = Self::EXPORTABLE;
    pub const IMPORTABLE_KHR: Self = Self::IMPORTABLE;
}
#[doc = "Generated from 'VK_KHR_external_semaphore_capabilities'"]
impl ExternalSemaphoreHandleTypeFlags {
    pub const OPAQUE_FD_KHR: Self = Self::OPAQUE_FD;
    pub const OPAQUE_WIN32_KHR: Self = Self::OPAQUE_WIN32;
    pub const OPAQUE_WIN32_KMT_KHR: Self = Self::OPAQUE_WIN32_KMT;
    pub const D3D12_FENCE_KHR: Self = Self::D3D12_FENCE;
    pub const SYNC_FD_KHR: Self = Self::SYNC_FD;
}
#[doc = "Generated from 'VK_KHR_external_semaphore_capabilities'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR: Self =
        Self::PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    pub const EXTERNAL_SEMAPHORE_PROPERTIES_KHR: Self = Self::EXTERNAL_SEMAPHORE_PROPERTIES;
}
#[doc = "Generated from 'VK_KHR_external_semaphore'"]
impl SemaphoreImportFlags {
    pub const TEMPORARY_KHR: Self = Self::TEMPORARY;
}
#[doc = "Generated from 'VK_KHR_external_semaphore'"]
impl StructureType {
    pub const EXPORT_SEMAPHORE_CREATE_INFO_KHR: Self = Self::EXPORT_SEMAPHORE_CREATE_INFO;
}
#[doc = "Generated from 'VK_KHR_external_semaphore_win32'"]
impl StructureType {
    pub const IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_078_000);
    pub const EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_078_001);
    pub const D3D12_FENCE_SUBMIT_INFO_KHR: Self = Self(1_000_078_002);
    pub const SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_078_003);
}
#[doc = "Generated from 'VK_KHR_external_semaphore_fd'"]
impl StructureType {
    pub const IMPORT_SEMAPHORE_FD_INFO_KHR: Self = Self(1_000_079_000);
    pub const SEMAPHORE_GET_FD_INFO_KHR: Self = Self(1_000_079_001);
}
#[doc = "Generated from 'VK_KHR_push_descriptor'"]
impl DescriptorSetLayoutCreateFlags {
    #[doc = "Descriptors are pushed via flink:vkCmdPushDescriptorSet"]
    pub const PUSH_DESCRIPTOR_KHR: Self = Self::PUSH_DESCRIPTOR;
}
#[doc = "Generated from 'VK_KHR_push_descriptor'"]
impl DescriptorUpdateTemplateType {
    #[doc = "Create descriptor update template for pushed descriptor updates"]
    pub const PUSH_DESCRIPTORS_KHR: Self = Self::PUSH_DESCRIPTORS;
}
#[doc = "Generated from 'VK_KHR_push_descriptor'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES;
}
#[doc = "Generated from 'VK_EXT_conditional_rendering'"]
impl AccessFlags {
    #[doc = "read access flag for reading conditional rendering predicate"]
    pub const CONDITIONAL_RENDERING_READ_EXT: Self = Self(0b1_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_conditional_rendering'"]
impl BufferUsageFlags {
    #[doc = "Specifies the buffer can be used as predicate in conditional rendering"]
    pub const CONDITIONAL_RENDERING_EXT: Self = Self(0b10_0000_0000);
}
#[doc = "Generated from 'VK_EXT_conditional_rendering'"]
impl PipelineStageFlags {
    #[doc = "A pipeline stage for conditional rendering predicate fetch"]
    pub const CONDITIONAL_RENDERING_EXT: Self = Self(0b100_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_conditional_rendering'"]
impl StructureType {
    pub const COMMAND_BUFFER_INHERITANCE_CONDITIONAL_RENDERING_INFO_EXT: Self = Self(1_000_081_000);
    pub const PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT: Self = Self(1_000_081_001);
    pub const CONDITIONAL_RENDERING_BEGIN_INFO_EXT: Self = Self(1_000_081_002);
}
#[doc = "Generated from 'VK_KHR_shader_float16_int8'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    pub const PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
}
#[doc = "Generated from 'VK_KHR_16bit_storage'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
}
#[doc = "Generated from 'VK_KHR_incremental_present'"]
impl StructureType {
    pub const PRESENT_REGIONS_KHR: Self = Self(1_000_084_000);
}
#[doc = "Generated from 'VK_KHR_descriptor_update_template'"]
impl DebugReportObjectTypeEXT {
    pub const DESCRIPTOR_UPDATE_TEMPLATE_KHR: Self = Self::DESCRIPTOR_UPDATE_TEMPLATE;
}
#[doc = "Generated from 'VK_KHR_descriptor_update_template'"]
impl DescriptorUpdateTemplateType {
    pub const DESCRIPTOR_SET_KHR: Self = Self::DESCRIPTOR_SET;
}
#[doc = "Generated from 'VK_KHR_descriptor_update_template'"]
impl ObjectType {
    pub const DESCRIPTOR_UPDATE_TEMPLATE_KHR: Self = Self::DESCRIPTOR_UPDATE_TEMPLATE;
}
#[doc = "Generated from 'VK_KHR_descriptor_update_template'"]
impl StructureType {
    pub const DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR: Self =
        Self::DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
}
#[doc = "Generated from 'VK_NV_clip_space_w_scaling'"]
impl DynamicState {
    pub const VIEWPORT_W_SCALING_NV: Self = Self(1_000_087_000);
}
#[doc = "Generated from 'VK_NV_clip_space_w_scaling'"]
impl StructureType {
    pub const PIPELINE_VIEWPORT_W_SCALING_STATE_CREATE_INFO_NV: Self = Self(1_000_087_000);
}
#[doc = "Generated from 'VK_EXT_display_surface_counter'"]
impl StructureType {
    pub const SURFACE_CAPABILITIES_2_EXT: Self = Self(1_000_090_000);
}
#[doc = "Generated from 'VK_EXT_display_control'"]
impl StructureType {
    pub const DISPLAY_POWER_INFO_EXT: Self = Self(1_000_091_000);
    pub const DEVICE_EVENT_INFO_EXT: Self = Self(1_000_091_001);
    pub const DISPLAY_EVENT_INFO_EXT: Self = Self(1_000_091_002);
    pub const SWAPCHAIN_COUNTER_CREATE_INFO_EXT: Self = Self(1_000_091_003);
}
#[doc = "Generated from 'VK_GOOGLE_display_timing'"]
impl StructureType {
    pub const PRESENT_TIMES_INFO_GOOGLE: Self = Self(1_000_092_000);
}
#[doc = "Generated from 'VK_NVX_multiview_per_view_attributes'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_ATTRIBUTES_PROPERTIES_NVX: Self =
        Self(1_000_097_000);
    pub const MULTIVIEW_PER_VIEW_ATTRIBUTES_INFO_NVX: Self = Self(1_000_044_009);
}
#[doc = "Generated from 'VK_NVX_multiview_per_view_attributes'"]
impl SubpassDescriptionFlags {
    pub const PER_VIEW_ATTRIBUTES_NVX: Self = Self(0b1);
    pub const PER_VIEW_POSITION_X_ONLY_NVX: Self = Self(0b10);
}
#[doc = "Generated from 'VK_NV_viewport_swizzle'"]
impl StructureType {
    pub const PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV: Self = Self(1_000_098_000);
}
#[doc = "Generated from 'VK_EXT_discard_rectangles'"]
impl DynamicState {
    pub const DISCARD_RECTANGLE_EXT: Self = Self(1_000_099_000);
    pub const DISCARD_RECTANGLE_ENABLE_EXT: Self = Self(1_000_099_001);
    pub const DISCARD_RECTANGLE_MODE_EXT: Self = Self(1_000_099_002);
}
#[doc = "Generated from 'VK_EXT_discard_rectangles'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DISCARD_RECTANGLE_PROPERTIES_EXT: Self = Self(1_000_099_000);
    pub const PIPELINE_DISCARD_RECTANGLE_STATE_CREATE_INFO_EXT: Self = Self(1_000_099_001);
}
#[doc = "Generated from 'VK_EXT_conservative_rasterization'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT: Self = Self(1_000_101_000);
    pub const PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT: Self = Self(1_000_101_001);
}
#[doc = "Generated from 'VK_EXT_depth_clip_enable'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT: Self = Self(1_000_102_000);
    pub const PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT: Self = Self(1_000_102_001);
}
#[doc = "Generated from 'VK_EXT_swapchain_colorspace'"]
impl ColorSpaceKHR {
    pub const DISPLAY_P3_NONLINEAR_EXT: Self = Self(1_000_104_001);
    pub const EXTENDED_SRGB_LINEAR_EXT: Self = Self(1_000_104_002);
    pub const DISPLAY_P3_LINEAR_EXT: Self = Self(1_000_104_003);
    pub const DCI_P3_NONLINEAR_EXT: Self = Self(1_000_104_004);
    pub const BT709_LINEAR_EXT: Self = Self(1_000_104_005);
    pub const BT709_NONLINEAR_EXT: Self = Self(1_000_104_006);
    pub const BT2020_LINEAR_EXT: Self = Self(1_000_104_007);
    pub const HDR10_ST2084_EXT: Self = Self(1_000_104_008);
    #[deprecated]
    pub const DOLBYVISION_EXT: Self = Self(1_000_104_009);
    pub const HDR10_HLG_EXT: Self = Self(1_000_104_010);
    pub const ADOBERGB_LINEAR_EXT: Self = Self(1_000_104_011);
    pub const ADOBERGB_NONLINEAR_EXT: Self = Self(1_000_104_012);
    pub const PASS_THROUGH_EXT: Self = Self(1_000_104_013);
    pub const EXTENDED_SRGB_NONLINEAR_EXT: Self = Self(1_000_104_014);
}
#[doc = "Generated from 'VK_EXT_hdr_metadata'"]
impl StructureType {
    pub const HDR_METADATA_EXT: Self = Self(1_000_105_000);
}
#[doc = "Generated from 'VK_KHR_imageless_framebuffer'"]
impl FramebufferCreateFlags {
    pub const IMAGELESS_KHR: Self = Self::IMAGELESS;
}
#[doc = "Generated from 'VK_KHR_imageless_framebuffer'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES;
    pub const FRAMEBUFFER_ATTACHMENTS_CREATE_INFO_KHR: Self =
        Self::FRAMEBUFFER_ATTACHMENTS_CREATE_INFO;
    pub const FRAMEBUFFER_ATTACHMENT_IMAGE_INFO_KHR: Self = Self::FRAMEBUFFER_ATTACHMENT_IMAGE_INFO;
    pub const RENDER_PASS_ATTACHMENT_BEGIN_INFO_KHR: Self = Self::RENDER_PASS_ATTACHMENT_BEGIN_INFO;
}
#[doc = "Generated from 'VK_KHR_create_renderpass2'"]
impl StructureType {
    pub const ATTACHMENT_DESCRIPTION_2_KHR: Self = Self::ATTACHMENT_DESCRIPTION_2;
    pub const ATTACHMENT_REFERENCE_2_KHR: Self = Self::ATTACHMENT_REFERENCE_2;
    pub const SUBPASS_DESCRIPTION_2_KHR: Self = Self::SUBPASS_DESCRIPTION_2;
    pub const SUBPASS_DEPENDENCY_2_KHR: Self = Self::SUBPASS_DEPENDENCY_2;
    pub const RENDER_PASS_CREATE_INFO_2_KHR: Self = Self::RENDER_PASS_CREATE_INFO_2;
    pub const SUBPASS_BEGIN_INFO_KHR: Self = Self::SUBPASS_BEGIN_INFO;
    pub const SUBPASS_END_INFO_KHR: Self = Self::SUBPASS_END_INFO;
}
#[doc = "Generated from 'VK_IMG_relaxed_line_rasterization'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RELAXED_LINE_RASTERIZATION_FEATURES_IMG: Self = Self(1_000_110_000);
}
#[doc = "Generated from 'VK_KHR_shared_presentable_image'"]
impl ImageLayout {
    pub const SHARED_PRESENT_KHR: Self = Self(1_000_111_000);
}
#[doc = "Generated from 'VK_KHR_shared_presentable_image'"]
impl PresentModeKHR {
    pub const SHARED_DEMAND_REFRESH: Self = Self(1_000_111_000);
    pub const SHARED_CONTINUOUS_REFRESH: Self = Self(1_000_111_001);
}
#[doc = "Generated from 'VK_KHR_shared_presentable_image'"]
impl StructureType {
    pub const SHARED_PRESENT_SURFACE_CAPABILITIES_KHR: Self = Self(1_000_111_000);
}
#[doc = "Generated from 'VK_KHR_external_fence_capabilities'"]
impl ExternalFenceFeatureFlags {
    pub const EXPORTABLE_KHR: Self = Self::EXPORTABLE;
    pub const IMPORTABLE_KHR: Self = Self::IMPORTABLE;
}
#[doc = "Generated from 'VK_KHR_external_fence_capabilities'"]
impl ExternalFenceHandleTypeFlags {
    pub const OPAQUE_FD_KHR: Self = Self::OPAQUE_FD;
    pub const OPAQUE_WIN32_KHR: Self = Self::OPAQUE_WIN32;
    pub const OPAQUE_WIN32_KMT_KHR: Self = Self::OPAQUE_WIN32_KMT;
    pub const SYNC_FD_KHR: Self = Self::SYNC_FD;
}
#[doc = "Generated from 'VK_KHR_external_fence_capabilities'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO_KHR: Self =
        Self::PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO;
    pub const EXTERNAL_FENCE_PROPERTIES_KHR: Self = Self::EXTERNAL_FENCE_PROPERTIES;
}
#[doc = "Generated from 'VK_KHR_external_fence'"]
impl FenceImportFlags {
    pub const TEMPORARY_KHR: Self = Self::TEMPORARY;
}
#[doc = "Generated from 'VK_KHR_external_fence'"]
impl StructureType {
    pub const EXPORT_FENCE_CREATE_INFO_KHR: Self = Self::EXPORT_FENCE_CREATE_INFO;
}
#[doc = "Generated from 'VK_KHR_external_fence_win32'"]
impl StructureType {
    pub const IMPORT_FENCE_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_114_000);
    pub const EXPORT_FENCE_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_114_001);
    pub const FENCE_GET_WIN32_HANDLE_INFO_KHR: Self = Self(1_000_114_002);
}
#[doc = "Generated from 'VK_KHR_external_fence_fd'"]
impl StructureType {
    pub const IMPORT_FENCE_FD_INFO_KHR: Self = Self(1_000_115_000);
    pub const FENCE_GET_FD_INFO_KHR: Self = Self(1_000_115_001);
}
#[doc = "Generated from 'VK_KHR_performance_query'"]
impl QueryType {
    pub const PERFORMANCE_QUERY_KHR: Self = Self(1_000_116_000);
}
#[doc = "Generated from 'VK_KHR_performance_query'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR: Self = Self(1_000_116_000);
    pub const PHYSICAL_DEVICE_PERFORMANCE_QUERY_PROPERTIES_KHR: Self = Self(1_000_116_001);
    pub const QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR: Self = Self(1_000_116_002);
    pub const PERFORMANCE_QUERY_SUBMIT_INFO_KHR: Self = Self(1_000_116_003);
    pub const ACQUIRE_PROFILING_LOCK_INFO_KHR: Self = Self(1_000_116_004);
    pub const PERFORMANCE_COUNTER_KHR: Self = Self(1_000_116_005);
    pub const PERFORMANCE_COUNTER_DESCRIPTION_KHR: Self = Self(1_000_116_006);
}
#[doc = "Generated from 'VK_KHR_maintenance2'"]
impl ImageCreateFlags {
    pub const BLOCK_TEXEL_VIEW_COMPATIBLE_KHR: Self = Self::BLOCK_TEXEL_VIEW_COMPATIBLE;
    pub const EXTENDED_USAGE_KHR: Self = Self::EXTENDED_USAGE;
}
#[doc = "Generated from 'VK_KHR_maintenance2'"]
impl ImageLayout {
    pub const DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR: Self =
        Self::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL;
    pub const DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR: Self =
        Self::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
}
#[doc = "Generated from 'VK_KHR_maintenance2'"]
impl PointClippingBehavior {
    pub const ALL_CLIP_PLANES_KHR: Self = Self::ALL_CLIP_PLANES;
    pub const USER_CLIP_PLANES_ONLY_KHR: Self = Self::USER_CLIP_PLANES_ONLY;
}
#[doc = "Generated from 'VK_KHR_maintenance2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES;
    pub const RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO_KHR: Self =
        Self::RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO;
    pub const IMAGE_VIEW_USAGE_CREATE_INFO_KHR: Self = Self::IMAGE_VIEW_USAGE_CREATE_INFO;
    pub const PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO_KHR: Self =
        Self::PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO;
}
#[doc = "Generated from 'VK_KHR_maintenance2'"]
impl TessellationDomainOrigin {
    pub const UPPER_LEFT_KHR: Self = Self::UPPER_LEFT;
    pub const LOWER_LEFT_KHR: Self = Self::LOWER_LEFT;
}
#[doc = "Generated from 'VK_KHR_get_surface_capabilities2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SURFACE_INFO_2_KHR: Self = Self(1_000_119_000);
    pub const SURFACE_CAPABILITIES_2_KHR: Self = Self(1_000_119_001);
    pub const SURFACE_FORMAT_2_KHR: Self = Self(1_000_119_002);
}
#[doc = "Generated from 'VK_KHR_variable_pointers'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
    pub const PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR;
}
#[doc = "Generated from 'VK_KHR_get_display_properties2'"]
impl StructureType {
    pub const DISPLAY_PROPERTIES_2_KHR: Self = Self(1_000_121_000);
    pub const DISPLAY_PLANE_PROPERTIES_2_KHR: Self = Self(1_000_121_001);
    pub const DISPLAY_MODE_PROPERTIES_2_KHR: Self = Self(1_000_121_002);
    pub const DISPLAY_PLANE_INFO_2_KHR: Self = Self(1_000_121_003);
    pub const DISPLAY_PLANE_CAPABILITIES_2_KHR: Self = Self(1_000_121_004);
}
#[doc = "Generated from 'VK_MVK_ios_surface'"]
impl StructureType {
    pub const IOS_SURFACE_CREATE_INFO_MVK: Self = Self(1_000_122_000);
}
#[doc = "Generated from 'VK_MVK_macos_surface'"]
impl StructureType {
    pub const MACOS_SURFACE_CREATE_INFO_MVK: Self = Self(1_000_123_000);
}
#[doc = "Generated from 'VK_EXT_external_memory_dma_buf'"]
impl ExternalMemoryHandleTypeFlags {
    pub const DMA_BUF_EXT: Self = Self(0b10_0000_0000);
}
#[doc = "Generated from 'VK_KHR_dedicated_allocation'"]
impl StructureType {
    pub const MEMORY_DEDICATED_REQUIREMENTS_KHR: Self = Self::MEMORY_DEDICATED_REQUIREMENTS;
    pub const MEMORY_DEDICATED_ALLOCATE_INFO_KHR: Self = Self::MEMORY_DEDICATED_ALLOCATE_INFO;
}
#[doc = "Generated from 'VK_EXT_debug_utils'"]
impl ObjectType {
    pub const DEBUG_UTILS_MESSENGER_EXT: Self = Self(1_000_128_000);
}
#[doc = "Generated from 'VK_EXT_debug_utils'"]
impl StructureType {
    pub const DEBUG_UTILS_OBJECT_NAME_INFO_EXT: Self = Self(1_000_128_000);
    pub const DEBUG_UTILS_OBJECT_TAG_INFO_EXT: Self = Self(1_000_128_001);
    pub const DEBUG_UTILS_LABEL_EXT: Self = Self(1_000_128_002);
    pub const DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT: Self = Self(1_000_128_003);
    pub const DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT: Self = Self(1_000_128_004);
}
#[doc = "Generated from 'VK_ANDROID_external_memory_android_hardware_buffer'"]
impl ExternalMemoryHandleTypeFlags {
    pub const ANDROID_HARDWARE_BUFFER_ANDROID: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_ANDROID_external_memory_android_hardware_buffer'"]
impl StructureType {
    pub const ANDROID_HARDWARE_BUFFER_USAGE_ANDROID: Self = Self(1_000_129_000);
    pub const ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID: Self = Self(1_000_129_001);
    pub const ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID: Self = Self(1_000_129_002);
    pub const IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID: Self = Self(1_000_129_003);
    pub const MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID: Self = Self(1_000_129_004);
    pub const EXTERNAL_FORMAT_ANDROID: Self = Self(1_000_129_005);
    pub const ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_2_ANDROID: Self = Self(1_000_129_006);
}
#[doc = "Generated from 'VK_EXT_sampler_filter_minmax'"]
impl FormatFeatureFlags {
    pub const SAMPLED_IMAGE_FILTER_MINMAX_EXT: Self = Self::SAMPLED_IMAGE_FILTER_MINMAX;
}
#[doc = "Generated from 'VK_EXT_sampler_filter_minmax'"]
impl SamplerReductionMode {
    pub const WEIGHTED_AVERAGE_EXT: Self = Self::WEIGHTED_AVERAGE;
    pub const MIN_EXT: Self = Self::MIN;
    pub const MAX_EXT: Self = Self::MAX;
}
#[doc = "Generated from 'VK_EXT_sampler_filter_minmax'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES;
    pub const SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT: Self =
        Self::SAMPLER_REDUCTION_MODE_CREATE_INFO;
}
#[doc = "Generated from 'VK_AMD_gpa_interface'"]
impl ObjectType {
    pub const GPA_SESSION_AMD: Self = Self(1_000_133_000);
}
#[doc = "Generated from 'VK_AMD_gpa_interface'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_GPA_FEATURES_AMD: Self = Self(1_000_133_000);
    pub const PHYSICAL_DEVICE_GPA_PROPERTIES_AMD: Self = Self(1_000_133_001);
    pub const GPA_SAMPLE_BEGIN_INFO_AMD: Self = Self(1_000_133_002);
    pub const GPA_SESSION_CREATE_INFO_AMD: Self = Self(1_000_133_003);
    pub const GPA_DEVICE_CLOCK_MODE_INFO_AMD: Self = Self(1_000_133_004);
    pub const PHYSICAL_DEVICE_GPA_PROPERTIES_2_AMD: Self = Self(1_000_133_005);
    pub const GPA_DEVICE_GET_CLOCK_INFO_AMD: Self = Self(1_000_133_006);
}
#[doc = "Generated from 'VK_AMDX_shader_enqueue'"]
#[cfg(feature = "provisional")]
impl BufferUsageFlags {
    pub const EXECUTION_GRAPH_SCRATCH_AMDX: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_AMDX_shader_enqueue'"]
#[cfg(feature = "provisional")]
impl BufferUsageFlags2 {
    pub const EXECUTION_GRAPH_SCRATCH_AMDX: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_AMDX_shader_enqueue'"]
#[cfg(feature = "provisional")]
impl PipelineBindPoint {
    pub const EXECUTION_GRAPH_AMDX: Self = Self(1_000_134_000);
}
#[doc = "Generated from 'VK_AMDX_shader_enqueue'"]
#[cfg(feature = "provisional")]
impl PipelineCreateFlags2 {
    pub const EXECUTION_GRAPH_AMDX: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_AMDX_shader_enqueue'"]
#[cfg(feature = "provisional")]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_ENQUEUE_FEATURES_AMDX: Self = Self(1_000_134_000);
    pub const PHYSICAL_DEVICE_SHADER_ENQUEUE_PROPERTIES_AMDX: Self = Self(1_000_134_001);
    pub const EXECUTION_GRAPH_PIPELINE_SCRATCH_SIZE_AMDX: Self = Self(1_000_134_002);
    pub const EXECUTION_GRAPH_PIPELINE_CREATE_INFO_AMDX: Self = Self(1_000_134_003);
    pub const PIPELINE_SHADER_STAGE_NODE_CREATE_INFO_AMDX: Self = Self(1_000_134_004);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl AccessFlags2 {
    pub const SAMPLER_HEAP_READ_EXT: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const RESOURCE_HEAP_READ_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl BufferUsageFlags {
    pub const DESCRIPTOR_HEAP_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl BufferUsageFlags2 {
    pub const DESCRIPTOR_HEAP_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl DescriptorMappingSourceEXT {
    pub const HEAP_WITH_SHADER_RECORD_INDEX: Self = Self(8);
    pub const SHADER_RECORD_DATA: Self = Self(9);
    pub const SHADER_RECORD_ADDRESS: Self = Self(10);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl ImageCreateFlags {
    pub const DESCRIPTOR_HEAP_CAPTURE_REPLAY_EXT: Self = Self(0b1_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl IndirectCommandsTokenTypeEXT {
    pub const PUSH_DATA: Self = Self(1_000_135_000);
    pub const PUSH_DATA_SEQUENCE_INDEX: Self = Self(1_000_135_001);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl IndirectCommandsTokenTypeNV {
    pub const PUSH_DATA: Self = Self(1_000_135_000);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl PipelineCreateFlags2 {
    pub const DESCRIPTOR_HEAP_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl ShaderCreateFlagsEXT {
    pub const DESCRIPTOR_HEAP: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl SpirvResourceTypeFlagsEXT {
    pub const ACCELERATION_STRUCTURE: Self = Self(0b1_0000_0000);
    pub const TENSOR_ARM: Self = Self(0b10_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl StructureType {
    pub const TEXEL_BUFFER_DESCRIPTOR_INFO_EXT: Self = Self(1_000_135_000);
    pub const IMAGE_DESCRIPTOR_INFO_EXT: Self = Self(1_000_135_001);
    pub const RESOURCE_DESCRIPTOR_INFO_EXT: Self = Self(1_000_135_002);
    pub const BIND_HEAP_INFO_EXT: Self = Self(1_000_135_003);
    pub const PUSH_DATA_INFO_EXT: Self = Self(1_000_135_004);
    pub const DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT: Self = Self(1_000_135_005);
    pub const SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT: Self = Self(1_000_135_006);
    pub const OPAQUE_CAPTURE_DATA_CREATE_INFO_EXT: Self = Self(1_000_135_007);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT: Self = Self(1_000_135_008);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT: Self = Self(1_000_135_009);
    pub const COMMAND_BUFFER_INHERITANCE_DESCRIPTOR_HEAP_INFO_EXT: Self = Self(1_000_135_010);
    pub const SAMPLER_CUSTOM_BORDER_COLOR_INDEX_CREATE_INFO_EXT: Self = Self(1_000_135_011);
    pub const INDIRECT_COMMANDS_LAYOUT_PUSH_DATA_TOKEN_NV: Self = Self(1_000_135_012);
    pub const SUBSAMPLED_IMAGE_FORMAT_PROPERTIES_EXT: Self = Self(1_000_135_013);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_HEAP_TENSOR_PROPERTIES_ARM: Self = Self(1_000_135_014);
}
#[doc = "Generated from 'VK_EXT_descriptor_heap'"]
impl TensorCreateFlagsARM {
    pub const DESCRIPTOR_HEAP_CAPTURE_REPLAY: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_AMD_mixed_attachment_samples'"]
impl StructureType {
    pub const ATTACHMENT_SAMPLE_COUNT_INFO_AMD: Self = Self(1_000_044_008);
}
#[doc = "Generated from 'VK_EXT_inline_uniform_block'"]
impl DescriptorType {
    pub const INLINE_UNIFORM_BLOCK_EXT: Self = Self::INLINE_UNIFORM_BLOCK;
}
#[doc = "Generated from 'VK_EXT_inline_uniform_block'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES;
    pub const PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES;
    pub const WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT: Self =
        Self::WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK;
    pub const DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO_EXT: Self =
        Self::DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO;
}
#[doc = "Generated from 'VK_KHR_shader_bfloat16'"]
impl ComponentTypeKHR {
    pub const BFLOAT16: Self = Self(1_000_141_000);
}
#[doc = "Generated from 'VK_KHR_shader_bfloat16'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR: Self = Self(1_000_141_000);
}
#[doc = "Generated from 'VK_EXT_sample_locations'"]
impl DynamicState {
    pub const SAMPLE_LOCATIONS_EXT: Self = Self(1_000_143_000);
}
#[doc = "Generated from 'VK_EXT_sample_locations'"]
impl ImageCreateFlags {
    pub const SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT: Self = Self(0b1_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_sample_locations'"]
impl StructureType {
    pub const SAMPLE_LOCATIONS_INFO_EXT: Self = Self(1_000_143_000);
    pub const RENDER_PASS_SAMPLE_LOCATIONS_BEGIN_INFO_EXT: Self = Self(1_000_143_001);
    pub const PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO_EXT: Self = Self(1_000_143_002);
    pub const PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES_EXT: Self = Self(1_000_143_003);
    pub const MULTISAMPLE_PROPERTIES_EXT: Self = Self(1_000_143_004);
}
#[doc = "Generated from 'VK_KHR_get_memory_requirements2'"]
impl StructureType {
    pub const BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR: Self = Self::BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    pub const IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR: Self = Self::IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    pub const IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2_KHR: Self =
        Self::IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2;
    pub const MEMORY_REQUIREMENTS_2_KHR: Self = Self::MEMORY_REQUIREMENTS_2;
    pub const SPARSE_IMAGE_MEMORY_REQUIREMENTS_2_KHR: Self =
        Self::SPARSE_IMAGE_MEMORY_REQUIREMENTS_2;
}
#[doc = "Generated from 'VK_KHR_image_format_list'"]
impl StructureType {
    pub const IMAGE_FORMAT_LIST_CREATE_INFO_KHR: Self = Self::IMAGE_FORMAT_LIST_CREATE_INFO;
}
#[doc = "Generated from 'VK_EXT_blend_operation_advanced'"]
impl AccessFlags {
    pub const COLOR_ATTACHMENT_READ_NONCOHERENT_EXT: Self = Self(0b1000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_blend_operation_advanced'"]
impl BlendOp {
    pub const ZERO_EXT: Self = Self(1_000_148_000);
    pub const SRC_EXT: Self = Self(1_000_148_001);
    pub const DST_EXT: Self = Self(1_000_148_002);
    pub const SRC_OVER_EXT: Self = Self(1_000_148_003);
    pub const DST_OVER_EXT: Self = Self(1_000_148_004);
    pub const SRC_IN_EXT: Self = Self(1_000_148_005);
    pub const DST_IN_EXT: Self = Self(1_000_148_006);
    pub const SRC_OUT_EXT: Self = Self(1_000_148_007);
    pub const DST_OUT_EXT: Self = Self(1_000_148_008);
    pub const SRC_ATOP_EXT: Self = Self(1_000_148_009);
    pub const DST_ATOP_EXT: Self = Self(1_000_148_010);
    pub const XOR_EXT: Self = Self(1_000_148_011);
    pub const MULTIPLY_EXT: Self = Self(1_000_148_012);
    pub const SCREEN_EXT: Self = Self(1_000_148_013);
    pub const OVERLAY_EXT: Self = Self(1_000_148_014);
    pub const DARKEN_EXT: Self = Self(1_000_148_015);
    pub const LIGHTEN_EXT: Self = Self(1_000_148_016);
    pub const COLORDODGE_EXT: Self = Self(1_000_148_017);
    pub const COLORBURN_EXT: Self = Self(1_000_148_018);
    pub const HARDLIGHT_EXT: Self = Self(1_000_148_019);
    pub const SOFTLIGHT_EXT: Self = Self(1_000_148_020);
    pub const DIFFERENCE_EXT: Self = Self(1_000_148_021);
    pub const EXCLUSION_EXT: Self = Self(1_000_148_022);
    pub const INVERT_EXT: Self = Self(1_000_148_023);
    pub const INVERT_RGB_EXT: Self = Self(1_000_148_024);
    pub const LINEARDODGE_EXT: Self = Self(1_000_148_025);
    pub const LINEARBURN_EXT: Self = Self(1_000_148_026);
    pub const VIVIDLIGHT_EXT: Self = Self(1_000_148_027);
    pub const LINEARLIGHT_EXT: Self = Self(1_000_148_028);
    pub const PINLIGHT_EXT: Self = Self(1_000_148_029);
    pub const HARDMIX_EXT: Self = Self(1_000_148_030);
    pub const HSL_HUE_EXT: Self = Self(1_000_148_031);
    pub const HSL_SATURATION_EXT: Self = Self(1_000_148_032);
    pub const HSL_COLOR_EXT: Self = Self(1_000_148_033);
    pub const HSL_LUMINOSITY_EXT: Self = Self(1_000_148_034);
    pub const PLUS_EXT: Self = Self(1_000_148_035);
    pub const PLUS_CLAMPED_EXT: Self = Self(1_000_148_036);
    pub const PLUS_CLAMPED_ALPHA_EXT: Self = Self(1_000_148_037);
    pub const PLUS_DARKER_EXT: Self = Self(1_000_148_038);
    pub const MINUS_EXT: Self = Self(1_000_148_039);
    pub const MINUS_CLAMPED_EXT: Self = Self(1_000_148_040);
    pub const CONTRAST_EXT: Self = Self(1_000_148_041);
    pub const INVERT_OVG_EXT: Self = Self(1_000_148_042);
    pub const RED_EXT: Self = Self(1_000_148_043);
    pub const GREEN_EXT: Self = Self(1_000_148_044);
    pub const BLUE_EXT: Self = Self(1_000_148_045);
}
#[doc = "Generated from 'VK_EXT_blend_operation_advanced'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT: Self = Self(1_000_148_000);
    pub const PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES_EXT: Self = Self(1_000_148_001);
    pub const PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO_EXT: Self = Self(1_000_148_002);
}
#[doc = "Generated from 'VK_NV_fragment_coverage_to_color'"]
impl StructureType {
    pub const PIPELINE_COVERAGE_TO_COLOR_STATE_CREATE_INFO_NV: Self = Self(1_000_149_000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl AccessFlags {
    pub const ACCELERATION_STRUCTURE_READ_KHR: Self = Self(0b10_0000_0000_0000_0000_0000);
    pub const ACCELERATION_STRUCTURE_WRITE_KHR: Self = Self(0b100_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl BufferUsageFlags {
    pub const ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR: Self =
        Self(0b1000_0000_0000_0000_0000);
    pub const ACCELERATION_STRUCTURE_STORAGE_KHR: Self = Self(0b1_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl CopyAccelerationStructureModeKHR {
    pub const SERIALIZE: Self = Self(2);
    pub const DESERIALIZE: Self = Self(3);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl DebugReportObjectTypeEXT {
    pub const ACCELERATION_STRUCTURE_KHR: Self = Self(1_000_150_000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl DescriptorType {
    pub const ACCELERATION_STRUCTURE_KHR: Self = Self(1_000_150_000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl FormatFeatureFlags {
    pub const ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl FormatFeatureFlags2 {
    pub const ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl IndexType {
    pub const NONE_KHR: Self = Self(1_000_165_000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl ObjectType {
    pub const ACCELERATION_STRUCTURE_KHR: Self = Self(1_000_150_000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl PipelineStageFlags {
    pub const ACCELERATION_STRUCTURE_BUILD_KHR: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl QueryType {
    pub const ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR: Self = Self(1_000_150_000);
    pub const ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR: Self = Self(1_000_150_001);
}
#[doc = "Generated from 'VK_KHR_acceleration_structure'"]
impl StructureType {
    pub const WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR: Self = Self(1_000_150_007);
    pub const ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR: Self = Self(1_000_150_000);
    pub const ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR: Self = Self(1_000_150_002);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR: Self = Self(1_000_150_003);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR: Self = Self(1_000_150_004);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR: Self = Self(1_000_150_005);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_KHR: Self = Self(1_000_150_006);
    pub const ACCELERATION_STRUCTURE_VERSION_INFO_KHR: Self = Self(1_000_150_009);
    pub const COPY_ACCELERATION_STRUCTURE_INFO_KHR: Self = Self(1_000_150_010);
    pub const COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR: Self = Self(1_000_150_011);
    pub const COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR: Self = Self(1_000_150_012);
    pub const PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR: Self = Self(1_000_150_013);
    pub const PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR: Self = Self(1_000_150_014);
    pub const ACCELERATION_STRUCTURE_CREATE_INFO_KHR: Self = Self(1_000_150_017);
    pub const ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR: Self = Self(1_000_150_020);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_pipeline'"]
impl BufferUsageFlags {
    pub const SHADER_BINDING_TABLE_KHR: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_pipeline'"]
impl DynamicState {
    pub const RAY_TRACING_PIPELINE_STACK_SIZE_KHR: Self = Self(1_000_347_000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_pipeline'"]
impl PipelineBindPoint {
    pub const RAY_TRACING_KHR: Self = Self(1_000_165_000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_pipeline'"]
impl PipelineCreateFlags {
    pub const RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_KHR: Self = Self(0b100_0000_0000_0000);
    pub const RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_KHR: Self = Self(0b1000_0000_0000_0000);
    pub const RAY_TRACING_NO_NULL_MISS_SHADERS_KHR: Self = Self(0b1_0000_0000_0000_0000);
    pub const RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_KHR: Self = Self(0b10_0000_0000_0000_0000);
    pub const RAY_TRACING_SKIP_TRIANGLES_KHR: Self = Self(0b1_0000_0000_0000);
    pub const RAY_TRACING_SKIP_AABBS_KHR: Self = Self(0b10_0000_0000_0000);
    pub const RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_KHR: Self =
        Self(0b1000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_pipeline'"]
impl PipelineCreateFlags2 {
    pub const RAY_TRACING_SKIP_BUILT_IN_PRIMITIVES_KHR: Self = Self::RAY_TRACING_SKIP_TRIANGLES_KHR;
}
#[doc = "Generated from 'VK_KHR_ray_tracing_pipeline'"]
impl PipelineStageFlags {
    pub const RAY_TRACING_SHADER_KHR: Self = Self(0b10_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_pipeline'"]
impl ShaderStageFlags {
    pub const RAYGEN_KHR: Self = Self(0b1_0000_0000);
    pub const ANY_HIT_KHR: Self = Self(0b10_0000_0000);
    pub const CLOSEST_HIT_KHR: Self = Self(0b100_0000_0000);
    pub const MISS_KHR: Self = Self(0b1000_0000_0000);
    pub const INTERSECTION_KHR: Self = Self(0b1_0000_0000_0000);
    pub const CALLABLE_KHR: Self = Self(0b10_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_pipeline'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR: Self = Self(1_000_347_000);
    pub const PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR: Self = Self(1_000_347_001);
    pub const RAY_TRACING_PIPELINE_CREATE_INFO_KHR: Self = Self(1_000_150_015);
    pub const RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR: Self = Self(1_000_150_016);
    pub const RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO_KHR: Self = Self(1_000_150_018);
}
#[doc = "Generated from 'VK_KHR_ray_query'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR: Self = Self(1_000_348_013);
}
#[doc = "Generated from 'VK_NV_framebuffer_mixed_samples'"]
impl StructureType {
    pub const PIPELINE_COVERAGE_MODULATION_STATE_CREATE_INFO_NV: Self = Self(1_000_152_000);
    pub const ATTACHMENT_SAMPLE_COUNT_INFO_NV: Self = Self::ATTACHMENT_SAMPLE_COUNT_INFO_AMD;
}
#[doc = "Generated from 'VK_NV_fill_rectangle'"]
impl PolygonMode {
    pub const FILL_RECTANGLE_NV: Self = Self(1_000_153_000);
}
#[doc = "Generated from 'VK_NV_shader_sm_builtins'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV: Self = Self(1_000_154_000);
    pub const PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV: Self = Self(1_000_154_001);
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl ChromaLocation {
    pub const COSITED_EVEN_KHR: Self = Self::COSITED_EVEN;
    pub const MIDPOINT_KHR: Self = Self::MIDPOINT;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl DebugReportObjectTypeEXT {
    pub const SAMPLER_YCBCR_CONVERSION_KHR: Self = Self::SAMPLER_YCBCR_CONVERSION;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl Format {
    pub const G8B8G8R8_422_UNORM_KHR: Self = Self::G8B8G8R8_422_UNORM;
    pub const B8G8R8G8_422_UNORM_KHR: Self = Self::B8G8R8G8_422_UNORM;
    pub const G8_B8_R8_3PLANE_420_UNORM_KHR: Self = Self::G8_B8_R8_3PLANE_420_UNORM;
    pub const G8_B8R8_2PLANE_420_UNORM_KHR: Self = Self::G8_B8R8_2PLANE_420_UNORM;
    pub const G8_B8_R8_3PLANE_422_UNORM_KHR: Self = Self::G8_B8_R8_3PLANE_422_UNORM;
    pub const G8_B8R8_2PLANE_422_UNORM_KHR: Self = Self::G8_B8R8_2PLANE_422_UNORM;
    pub const G8_B8_R8_3PLANE_444_UNORM_KHR: Self = Self::G8_B8_R8_3PLANE_444_UNORM;
    pub const R10X6_UNORM_PACK16_KHR: Self = Self::R10X6_UNORM_PACK16;
    pub const R10X6G10X6_UNORM_2PACK16_KHR: Self = Self::R10X6G10X6_UNORM_2PACK16;
    pub const R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR: Self =
        Self::R10X6G10X6B10X6A10X6_UNORM_4PACK16;
    pub const G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR: Self =
        Self::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16;
    pub const B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR: Self =
        Self::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16;
    pub const G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR: Self =
        Self::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16;
    pub const G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR: Self =
        Self::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
    pub const G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR: Self =
        Self::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16;
    pub const G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR: Self =
        Self::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16;
    pub const G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR: Self =
        Self::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16;
    pub const R12X4_UNORM_PACK16_KHR: Self = Self::R12X4_UNORM_PACK16;
    pub const R12X4G12X4_UNORM_2PACK16_KHR: Self = Self::R12X4G12X4_UNORM_2PACK16;
    pub const R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR: Self =
        Self::R12X4G12X4B12X4A12X4_UNORM_4PACK16;
    pub const G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR: Self =
        Self::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16;
    pub const B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR: Self =
        Self::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16;
    pub const G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR: Self =
        Self::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16;
    pub const G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR: Self =
        Self::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16;
    pub const G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR: Self =
        Self::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16;
    pub const G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR: Self =
        Self::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16;
    pub const G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR: Self =
        Self::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16;
    pub const G16B16G16R16_422_UNORM_KHR: Self = Self::G16B16G16R16_422_UNORM;
    pub const B16G16R16G16_422_UNORM_KHR: Self = Self::B16G16R16G16_422_UNORM;
    pub const G16_B16_R16_3PLANE_420_UNORM_KHR: Self = Self::G16_B16_R16_3PLANE_420_UNORM;
    pub const G16_B16R16_2PLANE_420_UNORM_KHR: Self = Self::G16_B16R16_2PLANE_420_UNORM;
    pub const G16_B16_R16_3PLANE_422_UNORM_KHR: Self = Self::G16_B16_R16_3PLANE_422_UNORM;
    pub const G16_B16R16_2PLANE_422_UNORM_KHR: Self = Self::G16_B16R16_2PLANE_422_UNORM;
    pub const G16_B16_R16_3PLANE_444_UNORM_KHR: Self = Self::G16_B16_R16_3PLANE_444_UNORM;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl FormatFeatureFlags {
    pub const MIDPOINT_CHROMA_SAMPLES_KHR: Self = Self::MIDPOINT_CHROMA_SAMPLES;
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_KHR: Self =
        Self::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER;
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_KHR: Self =
        Self::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER;
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_KHR: Self =
        Self::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT;
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_KHR: Self =
        Self::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE;
    pub const DISJOINT_KHR: Self = Self::DISJOINT;
    pub const COSITED_CHROMA_SAMPLES_KHR: Self = Self::COSITED_CHROMA_SAMPLES;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl ImageAspectFlags {
    pub const PLANE_0_KHR: Self = Self::PLANE_0;
    pub const PLANE_1_KHR: Self = Self::PLANE_1;
    pub const PLANE_2_KHR: Self = Self::PLANE_2;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl ImageCreateFlags {
    pub const DISJOINT_KHR: Self = Self::DISJOINT;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl ObjectType {
    pub const SAMPLER_YCBCR_CONVERSION_KHR: Self = Self::SAMPLER_YCBCR_CONVERSION;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl SamplerYcbcrModelConversion {
    pub const RGB_IDENTITY_KHR: Self = Self::RGB_IDENTITY;
    pub const YCBCR_IDENTITY_KHR: Self = Self::YCBCR_IDENTITY;
    pub const YCBCR_709_KHR: Self = Self::YCBCR_709;
    pub const YCBCR_601_KHR: Self = Self::YCBCR_601;
    pub const YCBCR_2020_KHR: Self = Self::YCBCR_2020;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl SamplerYcbcrRange {
    pub const ITU_FULL_KHR: Self = Self::ITU_FULL;
    pub const ITU_NARROW_KHR: Self = Self::ITU_NARROW;
}
#[doc = "Generated from 'VK_KHR_sampler_ycbcr_conversion'"]
impl StructureType {
    pub const SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR: Self =
        Self::SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
    pub const SAMPLER_YCBCR_CONVERSION_INFO_KHR: Self = Self::SAMPLER_YCBCR_CONVERSION_INFO;
    pub const BIND_IMAGE_PLANE_MEMORY_INFO_KHR: Self = Self::BIND_IMAGE_PLANE_MEMORY_INFO;
    pub const IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO_KHR: Self =
        Self::IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO;
    pub const PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
    pub const SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES_KHR: Self =
        Self::SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES;
}
#[doc = "Generated from 'VK_KHR_bind_memory2'"]
impl ImageCreateFlags {
    pub const ALIAS_KHR: Self = Self::ALIAS;
}
#[doc = "Generated from 'VK_KHR_bind_memory2'"]
impl StructureType {
    pub const BIND_BUFFER_MEMORY_INFO_KHR: Self = Self::BIND_BUFFER_MEMORY_INFO;
    pub const BIND_IMAGE_MEMORY_INFO_KHR: Self = Self::BIND_IMAGE_MEMORY_INFO;
}
#[doc = "Generated from 'VK_EXT_image_drm_format_modifier'"]
impl ImageAspectFlags {
    pub const MEMORY_PLANE_0_EXT: Self = Self(0b1000_0000);
    pub const MEMORY_PLANE_1_EXT: Self = Self(0b1_0000_0000);
    pub const MEMORY_PLANE_2_EXT: Self = Self(0b10_0000_0000);
    pub const MEMORY_PLANE_3_EXT: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_EXT_image_drm_format_modifier'"]
impl ImageTiling {
    pub const DRM_FORMAT_MODIFIER_EXT: Self = Self(1_000_158_000);
}
#[doc = "Generated from 'VK_EXT_image_drm_format_modifier'"]
impl Result {
    pub const ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: Self = Self(-1_000_158_000);
}
#[doc = "Generated from 'VK_EXT_image_drm_format_modifier'"]
impl StructureType {
    pub const DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT: Self = Self(1_000_158_000);
    pub const PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT: Self = Self(1_000_158_002);
    pub const IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT: Self = Self(1_000_158_003);
    pub const IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT: Self = Self(1_000_158_004);
    pub const IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT: Self = Self(1_000_158_005);
    pub const DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT: Self = Self(1_000_158_006);
}
#[doc = "Generated from 'VK_EXT_validation_cache'"]
impl ObjectType {
    pub const VALIDATION_CACHE_EXT: Self = Self(1_000_160_000);
}
#[doc = "Generated from 'VK_EXT_validation_cache'"]
impl StructureType {
    pub const VALIDATION_CACHE_CREATE_INFO_EXT: Self = Self(1_000_160_000);
    pub const SHADER_MODULE_VALIDATION_CACHE_CREATE_INFO_EXT: Self = Self(1_000_160_001);
}
#[doc = "Generated from 'VK_EXT_descriptor_indexing'"]
impl DescriptorBindingFlags {
    pub const UPDATE_AFTER_BIND_EXT: Self = Self::UPDATE_AFTER_BIND;
    pub const UPDATE_UNUSED_WHILE_PENDING_EXT: Self = Self::UPDATE_UNUSED_WHILE_PENDING;
    pub const PARTIALLY_BOUND_EXT: Self = Self::PARTIALLY_BOUND;
    pub const VARIABLE_DESCRIPTOR_COUNT_EXT: Self = Self::VARIABLE_DESCRIPTOR_COUNT;
}
#[doc = "Generated from 'VK_EXT_descriptor_indexing'"]
impl DescriptorPoolCreateFlags {
    pub const UPDATE_AFTER_BIND_EXT: Self = Self::UPDATE_AFTER_BIND;
}
#[doc = "Generated from 'VK_EXT_descriptor_indexing'"]
impl DescriptorSetLayoutCreateFlags {
    pub const UPDATE_AFTER_BIND_POOL_EXT: Self = Self::UPDATE_AFTER_BIND_POOL;
}
#[doc = "Generated from 'VK_EXT_descriptor_indexing'"]
impl Result {
    pub const ERROR_FRAGMENTATION_EXT: Self = Self::ERROR_FRAGMENTATION;
}
#[doc = "Generated from 'VK_EXT_descriptor_indexing'"]
impl StructureType {
    pub const DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT: Self =
        Self::DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    pub const PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    pub const PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
    pub const DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT: Self =
        Self::DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    pub const DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT_EXT: Self =
        Self::DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT;
}
#[doc = "Generated from 'VK_KHR_portability_subset'"]
#[cfg(feature = "provisional")]
impl StructureType {
    pub const PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR: Self = Self(1_000_163_000);
    pub const PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES_KHR: Self = Self(1_000_163_001);
}
#[doc = "Generated from 'VK_NV_shading_rate_image'"]
impl AccessFlags {
    pub const SHADING_RATE_IMAGE_READ_NV: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR;
}
#[doc = "Generated from 'VK_NV_shading_rate_image'"]
impl DynamicState {
    pub const VIEWPORT_SHADING_RATE_PALETTE_NV: Self = Self(1_000_164_004);
    pub const VIEWPORT_COARSE_SAMPLE_ORDER_NV: Self = Self(1_000_164_006);
}
#[doc = "Generated from 'VK_NV_shading_rate_image'"]
impl ImageLayout {
    pub const SHADING_RATE_OPTIMAL_NV: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
}
#[doc = "Generated from 'VK_NV_shading_rate_image'"]
impl ImageUsageFlags {
    pub const SHADING_RATE_IMAGE_NV: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR;
}
#[doc = "Generated from 'VK_NV_shading_rate_image'"]
impl PipelineStageFlags {
    pub const SHADING_RATE_IMAGE_NV: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR;
}
#[doc = "Generated from 'VK_NV_shading_rate_image'"]
impl StructureType {
    pub const PIPELINE_VIEWPORT_SHADING_RATE_IMAGE_STATE_CREATE_INFO_NV: Self = Self(1_000_164_000);
    pub const PHYSICAL_DEVICE_SHADING_RATE_IMAGE_FEATURES_NV: Self = Self(1_000_164_001);
    pub const PHYSICAL_DEVICE_SHADING_RATE_IMAGE_PROPERTIES_NV: Self = Self(1_000_164_002);
    pub const PIPELINE_VIEWPORT_COARSE_SAMPLE_ORDER_STATE_CREATE_INFO_NV: Self =
        Self(1_000_164_005);
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl AccelerationStructureTypeKHR {
    pub const TOP_LEVEL_NV: Self = Self::TOP_LEVEL;
    pub const BOTTOM_LEVEL_NV: Self = Self::BOTTOM_LEVEL;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl AccessFlags {
    pub const ACCELERATION_STRUCTURE_READ_NV: Self = Self::ACCELERATION_STRUCTURE_READ_KHR;
    pub const ACCELERATION_STRUCTURE_WRITE_NV: Self = Self::ACCELERATION_STRUCTURE_WRITE_KHR;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl BufferUsageFlags {
    pub const RAY_TRACING_NV: Self = Self::SHADER_BINDING_TABLE_KHR;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl BuildAccelerationStructureFlagsKHR {
    pub const ALLOW_UPDATE_NV: Self = Self::ALLOW_UPDATE;
    pub const ALLOW_COMPACTION_NV: Self = Self::ALLOW_COMPACTION;
    pub const PREFER_FAST_TRACE_NV: Self = Self::PREFER_FAST_TRACE;
    pub const PREFER_FAST_BUILD_NV: Self = Self::PREFER_FAST_BUILD;
    pub const LOW_MEMORY_NV: Self = Self::LOW_MEMORY;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl CopyAccelerationStructureModeKHR {
    pub const CLONE_NV: Self = Self::CLONE;
    pub const COMPACT_NV: Self = Self::COMPACT;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl DebugReportObjectTypeEXT {
    pub const ACCELERATION_STRUCTURE_NV: Self = Self(1_000_165_000);
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl DescriptorType {
    pub const ACCELERATION_STRUCTURE_NV: Self = Self(1_000_165_000);
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl GeometryFlagsKHR {
    pub const OPAQUE_NV: Self = Self::OPAQUE;
    pub const NO_DUPLICATE_ANY_HIT_INVOCATION_NV: Self = Self::NO_DUPLICATE_ANY_HIT_INVOCATION;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl GeometryInstanceFlagsKHR {
    pub const TRIANGLE_CULL_DISABLE_NV: Self = Self::TRIANGLE_FACING_CULL_DISABLE;
    pub const TRIANGLE_FRONT_COUNTERCLOCKWISE_NV: Self = Self::TRIANGLE_FRONT_COUNTERCLOCKWISE;
    pub const FORCE_OPAQUE_NV: Self = Self::FORCE_OPAQUE;
    pub const FORCE_NO_OPAQUE_NV: Self = Self::FORCE_NO_OPAQUE;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl GeometryTypeKHR {
    pub const TRIANGLES_NV: Self = Self::TRIANGLES;
    pub const AABBS_NV: Self = Self::AABBS;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl IndexType {
    pub const NONE_NV: Self = Self::NONE_KHR;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl ObjectType {
    pub const ACCELERATION_STRUCTURE_NV: Self = Self(1_000_165_000);
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl PipelineBindPoint {
    pub const RAY_TRACING_NV: Self = Self::RAY_TRACING_KHR;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl PipelineCreateFlags {
    pub const DEFER_COMPILE_NV: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl PipelineStageFlags {
    pub const RAY_TRACING_SHADER_NV: Self = Self::RAY_TRACING_SHADER_KHR;
    pub const ACCELERATION_STRUCTURE_BUILD_NV: Self = Self::ACCELERATION_STRUCTURE_BUILD_KHR;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl QueryType {
    pub const ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV: Self = Self(1_000_165_000);
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl RayTracingShaderGroupTypeKHR {
    pub const GENERAL_NV: Self = Self::GENERAL;
    pub const TRIANGLES_HIT_GROUP_NV: Self = Self::TRIANGLES_HIT_GROUP;
    pub const PROCEDURAL_HIT_GROUP_NV: Self = Self::PROCEDURAL_HIT_GROUP;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl ShaderStageFlags {
    pub const RAYGEN_NV: Self = Self::RAYGEN_KHR;
    pub const ANY_HIT_NV: Self = Self::ANY_HIT_KHR;
    pub const CLOSEST_HIT_NV: Self = Self::CLOSEST_HIT_KHR;
    pub const MISS_NV: Self = Self::MISS_KHR;
    pub const INTERSECTION_NV: Self = Self::INTERSECTION_KHR;
    pub const CALLABLE_NV: Self = Self::CALLABLE_KHR;
}
#[doc = "Generated from 'VK_NV_ray_tracing'"]
impl StructureType {
    pub const RAY_TRACING_PIPELINE_CREATE_INFO_NV: Self = Self(1_000_165_000);
    pub const ACCELERATION_STRUCTURE_CREATE_INFO_NV: Self = Self(1_000_165_001);
    pub const GEOMETRY_NV: Self = Self(1_000_165_003);
    pub const GEOMETRY_TRIANGLES_NV: Self = Self(1_000_165_004);
    pub const GEOMETRY_AABB_NV: Self = Self(1_000_165_005);
    pub const BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV: Self = Self(1_000_165_006);
    pub const WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV: Self = Self(1_000_165_007);
    pub const ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV: Self = Self(1_000_165_008);
    pub const PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV: Self = Self(1_000_165_009);
    pub const RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV: Self = Self(1_000_165_011);
    pub const ACCELERATION_STRUCTURE_INFO_NV: Self = Self(1_000_165_012);
}
#[doc = "Generated from 'VK_NV_representative_fragment_test'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV: Self = Self(1_000_166_000);
    pub const PIPELINE_REPRESENTATIVE_FRAGMENT_TEST_STATE_CREATE_INFO_NV: Self =
        Self(1_000_166_001);
}
#[doc = "Generated from 'VK_KHR_maintenance3'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    pub const DESCRIPTOR_SET_LAYOUT_SUPPORT_KHR: Self = Self::DESCRIPTOR_SET_LAYOUT_SUPPORT;
}
#[doc = "Generated from 'VK_EXT_filter_cubic'"]
impl Filter {
    pub const CUBIC_EXT: Self = Self(1_000_015_000);
}
#[doc = "Generated from 'VK_EXT_filter_cubic'"]
impl FormatFeatureFlags {
    pub const SAMPLED_IMAGE_FILTER_CUBIC_EXT: Self = Self(0b10_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_filter_cubic'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_VIEW_IMAGE_FORMAT_INFO_EXT: Self = Self(1_000_170_000);
    pub const FILTER_CUBIC_IMAGE_VIEW_IMAGE_FORMAT_PROPERTIES_EXT: Self = Self(1_000_170_001);
}
#[doc = "Generated from 'VK_QCOM_render_pass_shader_resolve'"]
impl SubpassDescriptionFlags {
    pub const FRAGMENT_REGION_QCOM: Self = Self::FRAGMENT_REGION_EXT;
    pub const SHADER_RESOLVE_QCOM: Self = Self::CUSTOM_RESOLVE_EXT;
}
#[doc = "Generated from 'VK_QCOM_cooperative_matrix_conversion'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_CONVERSION_FEATURES_QCOM: Self =
        Self(1_000_172_000);
}
#[doc = "Generated from 'VK_QCOM_elapsed_timer_query'"]
impl QueryType {
    pub const TIME_ELAPSED_QCOM: Self = Self(1_000_173_000);
}
#[doc = "Generated from 'VK_QCOM_elapsed_timer_query'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ELAPSED_TIMER_QUERY_FEATURES_QCOM: Self = Self(1_000_173_000);
}
#[doc = "Generated from 'VK_EXT_global_priority'"]
impl QueueGlobalPriority {
    pub const LOW_EXT: Self = Self::LOW;
    pub const MEDIUM_EXT: Self = Self::MEDIUM;
    pub const HIGH_EXT: Self = Self::HIGH;
    pub const REALTIME_EXT: Self = Self::REALTIME;
}
#[doc = "Generated from 'VK_EXT_global_priority'"]
impl Result {
    pub const ERROR_NOT_PERMITTED_EXT: Self = Self::ERROR_NOT_PERMITTED;
}
#[doc = "Generated from 'VK_EXT_global_priority'"]
impl StructureType {
    pub const DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT: Self =
        Self::DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO;
}
#[doc = "Generated from 'VK_KHR_shader_subgroup_extended_types'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES;
}
#[doc = "Generated from 'VK_KHR_8bit_storage'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
}
#[doc = "Generated from 'VK_EXT_external_memory_host'"]
impl ExternalMemoryHandleTypeFlags {
    pub const HOST_ALLOCATION_EXT: Self = Self(0b1000_0000);
    pub const HOST_MAPPED_FOREIGN_MEMORY_EXT: Self = Self(0b1_0000_0000);
}
#[doc = "Generated from 'VK_EXT_external_memory_host'"]
impl StructureType {
    pub const IMPORT_MEMORY_HOST_POINTER_INFO_EXT: Self = Self(1_000_178_000);
    pub const MEMORY_HOST_POINTER_PROPERTIES_EXT: Self = Self(1_000_178_001);
    pub const PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT: Self = Self(1_000_178_002);
}
#[doc = "Generated from 'VK_KHR_shader_atomic_int64'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
}
#[doc = "Generated from 'VK_KHR_shader_clock'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR: Self = Self(1_000_181_000);
}
#[doc = "Generated from 'VK_AMD_pipeline_compiler_control'"]
impl StructureType {
    pub const PIPELINE_COMPILER_CONTROL_CREATE_INFO_AMD: Self = Self(1_000_183_000);
}
#[doc = "Generated from 'VK_EXT_calibrated_timestamps'"]
impl StructureType {
    pub const CALIBRATED_TIMESTAMP_INFO_EXT: Self = Self::CALIBRATED_TIMESTAMP_INFO_KHR;
}
#[doc = "Generated from 'VK_EXT_calibrated_timestamps'"]
impl TimeDomainKHR {
    pub const DEVICE_EXT: Self = Self::DEVICE;
    pub const CLOCK_MONOTONIC_EXT: Self = Self::CLOCK_MONOTONIC;
    pub const CLOCK_MONOTONIC_RAW_EXT: Self = Self::CLOCK_MONOTONIC_RAW;
    pub const QUERY_PERFORMANCE_COUNTER_EXT: Self = Self::QUERY_PERFORMANCE_COUNTER;
}
#[doc = "Generated from 'VK_AMD_shader_core_properties'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD: Self = Self(1_000_185_000);
}
#[doc = "Generated from 'VK_KHR_video_decode_h265'"]
impl StructureType {
    pub const VIDEO_DECODE_H265_CAPABILITIES_KHR: Self = Self(1_000_187_000);
    pub const VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR: Self = Self(1_000_187_001);
    pub const VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR: Self = Self(1_000_187_002);
    pub const VIDEO_DECODE_H265_PROFILE_INFO_KHR: Self = Self(1_000_187_003);
    pub const VIDEO_DECODE_H265_PICTURE_INFO_KHR: Self = Self(1_000_187_004);
    pub const VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR: Self = Self(1_000_187_005);
}
#[doc = "Generated from 'VK_KHR_video_decode_h265'"]
impl VideoCodecOperationFlagsKHR {
    pub const DECODE_H265: Self = Self(0b10);
}
#[doc = "Generated from 'VK_KHR_global_priority'"]
impl QueueGlobalPriority {
    pub const LOW_KHR: Self = Self::LOW;
    pub const MEDIUM_KHR: Self = Self::MEDIUM;
    pub const HIGH_KHR: Self = Self::HIGH;
    pub const REALTIME_KHR: Self = Self::REALTIME;
}
#[doc = "Generated from 'VK_KHR_global_priority'"]
impl Result {
    pub const ERROR_NOT_PERMITTED_KHR: Self = Self::ERROR_NOT_PERMITTED;
}
#[doc = "Generated from 'VK_KHR_global_priority'"]
impl StructureType {
    pub const DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR: Self =
        Self::DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO;
    pub const PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES;
    pub const QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR: Self =
        Self::QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES;
}
#[doc = "Generated from 'VK_AMD_memory_overallocation_behavior'"]
impl StructureType {
    pub const DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD: Self = Self(1_000_189_000);
}
#[doc = "Generated from 'VK_EXT_vertex_attribute_divisor'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT: Self = Self(1_000_190_000);
    pub const PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT: Self =
        Self::PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO;
    pub const PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES;
}
#[doc = "Generated from 'VK_GGP_frame_token'"]
impl StructureType {
    pub const PRESENT_FRAME_TOKEN_GGP: Self = Self(1_000_191_000);
}
#[doc = "Generated from 'VK_EXT_pipeline_creation_feedback'"]
impl PipelineCreationFeedbackFlags {
    pub const VALID_EXT: Self = Self::VALID;
    pub const APPLICATION_PIPELINE_CACHE_HIT_EXT: Self = Self::APPLICATION_PIPELINE_CACHE_HIT;
    pub const BASE_PIPELINE_ACCELERATION_EXT: Self = Self::BASE_PIPELINE_ACCELERATION;
}
#[doc = "Generated from 'VK_EXT_pipeline_creation_feedback'"]
impl StructureType {
    pub const PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT: Self =
        Self::PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
}
#[doc = "Generated from 'VK_KHR_driver_properties'"]
impl DriverId {
    pub const AMD_PROPRIETARY_KHR: Self = Self::AMD_PROPRIETARY;
    pub const AMD_OPEN_SOURCE_KHR: Self = Self::AMD_OPEN_SOURCE;
    pub const MESA_RADV_KHR: Self = Self::MESA_RADV;
    pub const NVIDIA_PROPRIETARY_KHR: Self = Self::NVIDIA_PROPRIETARY;
    pub const INTEL_PROPRIETARY_WINDOWS_KHR: Self = Self::INTEL_PROPRIETARY_WINDOWS;
    pub const INTEL_OPEN_SOURCE_MESA_KHR: Self = Self::INTEL_OPEN_SOURCE_MESA;
    pub const IMAGINATION_PROPRIETARY_KHR: Self = Self::IMAGINATION_PROPRIETARY;
    pub const QUALCOMM_PROPRIETARY_KHR: Self = Self::QUALCOMM_PROPRIETARY;
    pub const ARM_PROPRIETARY_KHR: Self = Self::ARM_PROPRIETARY;
    pub const GOOGLE_SWIFTSHADER_KHR: Self = Self::GOOGLE_SWIFTSHADER;
    pub const GGP_PROPRIETARY_KHR: Self = Self::GGP_PROPRIETARY;
    pub const BROADCOM_PROPRIETARY_KHR: Self = Self::BROADCOM_PROPRIETARY;
}
#[doc = "Generated from 'VK_KHR_driver_properties'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR: Self = Self::PHYSICAL_DEVICE_DRIVER_PROPERTIES;
}
#[doc = "Generated from 'VK_KHR_shader_float_controls'"]
impl ShaderFloatControlsIndependence {
    pub const TYPE_32_ONLY_KHR: Self = Self::TYPE_32_ONLY;
    pub const ALL_KHR: Self = Self::ALL;
    pub const NONE_KHR: Self = Self::NONE;
}
#[doc = "Generated from 'VK_KHR_shader_float_controls'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
}
#[doc = "Generated from 'VK_NV_shader_subgroup_partitioned'"]
impl SubgroupFeatureFlags {
    pub const PARTITIONED_NV: Self = Self::PARTITIONED_EXT;
}
#[doc = "Generated from 'VK_KHR_depth_stencil_resolve'"]
impl ResolveModeFlags {
    pub const NONE_KHR: Self = Self::NONE;
    pub const SAMPLE_ZERO_KHR: Self = Self::SAMPLE_ZERO;
    pub const AVERAGE_KHR: Self = Self::AVERAGE;
    pub const MIN_KHR: Self = Self::MIN;
    pub const MAX_KHR: Self = Self::MAX;
}
#[doc = "Generated from 'VK_KHR_depth_stencil_resolve'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES;
    pub const SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE_KHR: Self =
        Self::SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE;
}
#[doc = "Generated from 'VK_KHR_swapchain_mutable_format'"]
impl SwapchainCreateFlagsKHR {
    pub const MUTABLE_FORMAT: Self = Self(0b100);
}
#[doc = "Generated from 'VK_NV_compute_shader_derivatives'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV: Self =
        Self::PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR;
}
#[doc = "Generated from 'VK_NV_mesh_shader'"]
impl IndirectCommandsTokenTypeEXT {
    pub const DRAW_MESH_TASKS_NV: Self = Self(1_000_202_002);
    pub const DRAW_MESH_TASKS_COUNT_NV: Self = Self(1_000_202_003);
}
#[doc = "Generated from 'VK_NV_mesh_shader'"]
impl PipelineStageFlags {
    pub const TASK_SHADER_NV: Self = Self::TASK_SHADER_EXT;
    pub const MESH_SHADER_NV: Self = Self::MESH_SHADER_EXT;
}
#[doc = "Generated from 'VK_NV_mesh_shader'"]
impl ShaderStageFlags {
    pub const TASK_NV: Self = Self::TASK_EXT;
    pub const MESH_NV: Self = Self::MESH_EXT;
}
#[doc = "Generated from 'VK_NV_mesh_shader'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV: Self = Self(1_000_202_000);
    pub const PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_NV: Self = Self(1_000_202_001);
}
#[doc = "Generated from 'VK_NV_fragment_shader_barycentric'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV: Self =
        Self::PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR;
}
#[doc = "Generated from 'VK_NV_shader_image_footprint'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV: Self = Self(1_000_204_000);
}
#[doc = "Generated from 'VK_NV_scissor_exclusive'"]
impl DynamicState {
    pub const EXCLUSIVE_SCISSOR_ENABLE_NV: Self = Self(1_000_205_000);
    pub const EXCLUSIVE_SCISSOR_NV: Self = Self(1_000_205_001);
}
#[doc = "Generated from 'VK_NV_scissor_exclusive'"]
impl StructureType {
    pub const PIPELINE_VIEWPORT_EXCLUSIVE_SCISSOR_STATE_CREATE_INFO_NV: Self = Self(1_000_205_000);
    pub const PHYSICAL_DEVICE_EXCLUSIVE_SCISSOR_FEATURES_NV: Self = Self(1_000_205_002);
}
#[doc = "Generated from 'VK_NV_device_diagnostic_checkpoints'"]
impl StructureType {
    pub const CHECKPOINT_DATA_NV: Self = Self(1_000_206_000);
    pub const QUEUE_FAMILY_CHECKPOINT_PROPERTIES_NV: Self = Self(1_000_206_001);
    pub const QUEUE_FAMILY_CHECKPOINT_PROPERTIES_2_NV: Self = Self(1_000_314_008);
    pub const CHECKPOINT_DATA_2_NV: Self = Self(1_000_314_009);
}
#[doc = "Generated from 'VK_KHR_timeline_semaphore'"]
impl SemaphoreType {
    pub const BINARY_KHR: Self = Self::BINARY;
    pub const TIMELINE_KHR: Self = Self::TIMELINE;
}
#[doc = "Generated from 'VK_KHR_timeline_semaphore'"]
impl SemaphoreWaitFlags {
    pub const ANY_KHR: Self = Self::ANY;
}
#[doc = "Generated from 'VK_KHR_timeline_semaphore'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    pub const PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES;
    pub const SEMAPHORE_TYPE_CREATE_INFO_KHR: Self = Self::SEMAPHORE_TYPE_CREATE_INFO;
    pub const TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR: Self = Self::TIMELINE_SEMAPHORE_SUBMIT_INFO;
    pub const SEMAPHORE_WAIT_INFO_KHR: Self = Self::SEMAPHORE_WAIT_INFO;
    pub const SEMAPHORE_SIGNAL_INFO_KHR: Self = Self::SEMAPHORE_SIGNAL_INFO;
}
#[doc = "Generated from 'VK_EXT_present_timing'"]
impl Result {
    pub const ERROR_PRESENT_TIMING_QUEUE_FULL_EXT: Self = Self(-1_000_208_000);
}
#[doc = "Generated from 'VK_EXT_present_timing'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT: Self = Self(1_000_208_000);
    pub const SWAPCHAIN_TIMING_PROPERTIES_EXT: Self = Self(1_000_208_001);
    pub const SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT: Self = Self(1_000_208_002);
    pub const PRESENT_TIMINGS_INFO_EXT: Self = Self(1_000_208_003);
    pub const PRESENT_TIMING_INFO_EXT: Self = Self(1_000_208_004);
    pub const PAST_PRESENTATION_TIMING_INFO_EXT: Self = Self(1_000_208_005);
    pub const PAST_PRESENTATION_TIMING_PROPERTIES_EXT: Self = Self(1_000_208_006);
    pub const PAST_PRESENTATION_TIMING_EXT: Self = Self(1_000_208_007);
    pub const PRESENT_TIMING_SURFACE_CAPABILITIES_EXT: Self = Self(1_000_208_008);
    pub const SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT: Self = Self(1_000_208_009);
}
#[doc = "Generated from 'VK_EXT_present_timing'"]
impl SwapchainCreateFlagsKHR {
    pub const PRESENT_TIMING_EXT: Self = Self(0b10_0000_0000);
}
#[doc = "Generated from 'VK_EXT_present_timing'"]
impl TimeDomainKHR {
    pub const PRESENT_STAGE_LOCAL_EXT: Self = Self(1_000_208_000);
    pub const SWAPCHAIN_LOCAL_EXT: Self = Self(1_000_208_001);
}
#[doc = "Generated from 'VK_INTEL_shader_integer_functions2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL: Self = Self(1_000_209_000);
}
#[doc = "Generated from 'VK_INTEL_performance_query'"]
impl ObjectType {
    pub const PERFORMANCE_CONFIGURATION_INTEL: Self = Self(1_000_210_000);
}
#[doc = "Generated from 'VK_INTEL_performance_query'"]
impl QueryType {
    pub const PERFORMANCE_QUERY_INTEL: Self = Self(1_000_210_000);
}
#[doc = "Generated from 'VK_INTEL_performance_query'"]
impl StructureType {
    pub const QUERY_POOL_PERFORMANCE_QUERY_CREATE_INFO_INTEL: Self = Self(1_000_210_000);
    pub const INITIALIZE_PERFORMANCE_API_INFO_INTEL: Self = Self(1_000_210_001);
    pub const PERFORMANCE_MARKER_INFO_INTEL: Self = Self(1_000_210_002);
    pub const PERFORMANCE_STREAM_MARKER_INFO_INTEL: Self = Self(1_000_210_003);
    pub const PERFORMANCE_OVERRIDE_INFO_INTEL: Self = Self(1_000_210_004);
    pub const PERFORMANCE_CONFIGURATION_ACQUIRE_INFO_INTEL: Self = Self(1_000_210_005);
}
#[doc = "Generated from 'VK_KHR_vulkan_memory_model'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
}
#[doc = "Generated from 'VK_EXT_pci_bus_info'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT: Self = Self(1_000_212_000);
}
#[doc = "Generated from 'VK_AMD_display_native_hdr'"]
impl ColorSpaceKHR {
    pub const DISPLAY_NATIVE_AMD: Self = Self(1_000_213_000);
}
#[doc = "Generated from 'VK_AMD_display_native_hdr'"]
impl StructureType {
    pub const DISPLAY_NATIVE_HDR_SURFACE_CAPABILITIES_AMD: Self = Self(1_000_213_000);
    pub const SWAPCHAIN_DISPLAY_NATIVE_HDR_CREATE_INFO_AMD: Self = Self(1_000_213_001);
}
#[doc = "Generated from 'VK_FUCHSIA_imagepipe_surface'"]
impl StructureType {
    pub const IMAGEPIPE_SURFACE_CREATE_INFO_FUCHSIA: Self = Self(1_000_214_000);
}
#[doc = "Generated from 'VK_KHR_shader_terminate_invocation'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES;
}
#[doc = "Generated from 'VK_EXT_metal_surface'"]
impl StructureType {
    pub const METAL_SURFACE_CREATE_INFO_EXT: Self = Self(1_000_217_000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl AccessFlags {
    pub const FRAGMENT_DENSITY_MAP_READ_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl FormatFeatureFlags {
    pub const FRAGMENT_DENSITY_MAP_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl FormatFeatureFlags2 {
    pub const FRAGMENT_DENSITY_MAP_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl ImageCreateFlags {
    pub const SUBSAMPLED_EXT: Self = Self(0b100_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl ImageLayout {
    pub const FRAGMENT_DENSITY_MAP_OPTIMAL_EXT: Self = Self(1_000_218_000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl ImageUsageFlags {
    pub const FRAGMENT_DENSITY_MAP_EXT: Self = Self(0b10_0000_0000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl ImageViewCreateFlags {
    pub const FRAGMENT_DENSITY_MAP_DYNAMIC_EXT: Self = Self(0b1);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl PipelineCreateFlags {
    pub const RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl PipelineStageFlags {
    pub const FRAGMENT_DENSITY_PROCESS_EXT: Self = Self(0b1000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl SamplerCreateFlags {
    pub const SUBSAMPLED_EXT: Self = Self(0b1);
    pub const SUBSAMPLED_COARSE_RECONSTRUCTION_EXT: Self = Self(0b10);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT: Self = Self(1_000_218_000);
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES_EXT: Self = Self(1_000_218_001);
    pub const RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT: Self = Self(1_000_218_002);
    pub const RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO_EXT: Self = Self(1_000_044_007);
}
#[doc = "Generated from 'VK_EXT_scalar_block_layout'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES;
}
#[doc = "Generated from 'VK_EXT_subgroup_size_control'"]
impl PipelineShaderStageCreateFlags {
    pub const ALLOW_VARYING_SUBGROUP_SIZE_EXT: Self = Self::ALLOW_VARYING_SUBGROUP_SIZE;
    pub const REQUIRE_FULL_SUBGROUPS_EXT: Self = Self::REQUIRE_FULL_SUBGROUPS;
}
#[doc = "Generated from 'VK_EXT_subgroup_size_control'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    pub const PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT: Self =
        Self::PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    pub const PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl AccessFlags {
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR: Self =
        Self(0b1000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl DynamicState {
    pub const FRAGMENT_SHADING_RATE_KHR: Self = Self(1_000_226_000);
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl FormatFeatureFlags {
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_KHR: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl FormatFeatureFlags2 {
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_KHR: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl ImageLayout {
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR: Self = Self(1_000_164_003);
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl ImageUsageFlags {
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_KHR: Self = Self(0b1_0000_0000);
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl PipelineCreateFlags {
    pub const RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_KHR: Self =
        Self(0b10_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl PipelineStageFlags {
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_KHR: Self = Self(0b100_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_fragment_shading_rate'"]
impl StructureType {
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR: Self = Self(1_000_226_000);
    pub const PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR: Self = Self(1_000_226_001);
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR: Self = Self(1_000_226_002);
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR: Self = Self(1_000_226_003);
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR: Self = Self(1_000_226_004);
    pub const RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR: Self = Self(1_000_044_006);
}
#[doc = "Generated from 'VK_AMD_shader_core_properties2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD: Self = Self(1_000_227_000);
}
#[doc = "Generated from 'VK_AMD_device_coherent_memory'"]
impl MemoryPropertyFlags {
    pub const DEVICE_COHERENT_AMD: Self = Self(0b100_0000);
    pub const DEVICE_UNCACHED_AMD: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_AMD_device_coherent_memory'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD: Self = Self(1_000_229_000);
}
#[doc = "Generated from 'VK_KHR_shader_constant_data'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_CONSTANT_DATA_FEATURES_KHR: Self = Self(1_000_231_000);
}
#[doc = "Generated from 'VK_KHR_dynamic_rendering_local_read'"]
impl ImageLayout {
    pub const RENDERING_LOCAL_READ_KHR: Self = Self::RENDERING_LOCAL_READ;
}
#[doc = "Generated from 'VK_KHR_dynamic_rendering_local_read'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES;
    pub const RENDERING_ATTACHMENT_LOCATION_INFO_KHR: Self =
        Self::RENDERING_ATTACHMENT_LOCATION_INFO;
    pub const RENDERING_INPUT_ATTACHMENT_INDEX_INFO_KHR: Self =
        Self::RENDERING_INPUT_ATTACHMENT_INDEX_INFO;
}
#[doc = "Generated from 'VK_KHR_shader_abort'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_ABORT_FEATURES_KHR: Self = Self(1_000_233_000);
    pub const DEVICE_FAULT_SHADER_ABORT_MESSAGE_INFO_KHR: Self = Self(1_000_233_001);
    pub const PHYSICAL_DEVICE_SHADER_ABORT_PROPERTIES_KHR: Self = Self(1_000_233_002);
}
#[doc = "Generated from 'VK_EXT_shader_image_atomic_int64'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT: Self = Self(1_000_234_000);
}
#[doc = "Generated from 'VK_KHR_shader_quad_control'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR: Self = Self(1_000_235_000);
}
#[doc = "Generated from 'VK_EXT_memory_budget'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT: Self = Self(1_000_237_000);
}
#[doc = "Generated from 'VK_EXT_memory_priority'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT: Self = Self(1_000_238_000);
    pub const MEMORY_PRIORITY_ALLOCATE_INFO_EXT: Self = Self(1_000_238_001);
}
#[doc = "Generated from 'VK_KHR_surface_protected_capabilities'"]
impl StructureType {
    pub const SURFACE_PROTECTED_CAPABILITIES_KHR: Self = Self(1_000_239_000);
}
#[doc = "Generated from 'VK_NV_dedicated_allocation_image_aliasing'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEDICATED_ALLOCATION_IMAGE_ALIASING_FEATURES_NV: Self =
        Self(1_000_240_000);
}
#[doc = "Generated from 'VK_KHR_separate_depth_stencil_layouts'"]
impl ImageLayout {
    pub const DEPTH_ATTACHMENT_OPTIMAL_KHR: Self = Self::DEPTH_ATTACHMENT_OPTIMAL;
    pub const DEPTH_READ_ONLY_OPTIMAL_KHR: Self = Self::DEPTH_READ_ONLY_OPTIMAL;
    pub const STENCIL_ATTACHMENT_OPTIMAL_KHR: Self = Self::STENCIL_ATTACHMENT_OPTIMAL;
    pub const STENCIL_READ_ONLY_OPTIMAL_KHR: Self = Self::STENCIL_READ_ONLY_OPTIMAL;
}
#[doc = "Generated from 'VK_KHR_separate_depth_stencil_layouts'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES;
    pub const ATTACHMENT_REFERENCE_STENCIL_LAYOUT_KHR: Self =
        Self::ATTACHMENT_REFERENCE_STENCIL_LAYOUT;
    pub const ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT_KHR: Self =
        Self::ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT;
}
#[doc = "Generated from 'VK_EXT_buffer_device_address'"]
impl BufferCreateFlags {
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY_EXT: Self = Self::DEVICE_ADDRESS_CAPTURE_REPLAY;
}
#[doc = "Generated from 'VK_EXT_buffer_device_address'"]
impl BufferUsageFlags {
    pub const SHADER_DEVICE_ADDRESS_EXT: Self = Self::SHADER_DEVICE_ADDRESS;
}
#[doc = "Generated from 'VK_EXT_buffer_device_address'"]
impl Result {
    pub const ERROR_INVALID_DEVICE_ADDRESS_EXT: Self = Self::ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
}
#[doc = "Generated from 'VK_EXT_buffer_device_address'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT: Self = Self(1_000_244_000);
    pub const PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT;
    pub const BUFFER_DEVICE_ADDRESS_INFO_EXT: Self = Self::BUFFER_DEVICE_ADDRESS_INFO;
    pub const BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT: Self = Self(1_000_244_002);
}
#[doc = "Generated from 'VK_EXT_tooling_info'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT: Self = Self::PHYSICAL_DEVICE_TOOL_PROPERTIES;
}
#[doc = "Generated from 'VK_EXT_tooling_info'"]
impl ToolPurposeFlags {
    pub const VALIDATION_EXT: Self = Self::VALIDATION;
    pub const PROFILING_EXT: Self = Self::PROFILING;
    pub const TRACING_EXT: Self = Self::TRACING;
    pub const ADDITIONAL_FEATURES_EXT: Self = Self::ADDITIONAL_FEATURES;
    pub const MODIFYING_FEATURES_EXT: Self = Self::MODIFYING_FEATURES;
    pub const DEBUG_REPORTING_EXT: Self = Self(0b10_0000);
    pub const DEBUG_MARKERS_EXT: Self = Self(0b100_0000);
}
#[doc = "Generated from 'VK_EXT_separate_stencil_usage'"]
impl StructureType {
    pub const IMAGE_STENCIL_USAGE_CREATE_INFO_EXT: Self = Self::IMAGE_STENCIL_USAGE_CREATE_INFO;
}
#[doc = "Generated from 'VK_EXT_validation_features'"]
impl StructureType {
    pub const VALIDATION_FEATURES_EXT: Self = Self(1_000_247_000);
}
#[doc = "Generated from 'VK_KHR_present_wait'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR: Self = Self(1_000_248_000);
}
#[doc = "Generated from 'VK_NV_cooperative_matrix'"]
impl ComponentTypeKHR {
    pub const FLOAT16_NV: Self = Self::FLOAT16;
    pub const FLOAT32_NV: Self = Self::FLOAT32;
    pub const FLOAT64_NV: Self = Self::FLOAT64;
    pub const SINT8_NV: Self = Self::SINT8;
    pub const SINT16_NV: Self = Self::SINT16;
    pub const SINT32_NV: Self = Self::SINT32;
    pub const SINT64_NV: Self = Self::SINT64;
    pub const UINT8_NV: Self = Self::UINT8;
    pub const UINT16_NV: Self = Self::UINT16;
    pub const UINT32_NV: Self = Self::UINT32;
    pub const UINT64_NV: Self = Self::UINT64;
}
#[doc = "Generated from 'VK_NV_cooperative_matrix'"]
impl ScopeKHR {
    pub const DEVICE_NV: Self = Self::DEVICE;
    pub const WORKGROUP_NV: Self = Self::WORKGROUP;
    pub const SUBGROUP_NV: Self = Self::SUBGROUP;
    pub const QUEUE_FAMILY_NV: Self = Self::QUEUE_FAMILY;
}
#[doc = "Generated from 'VK_NV_cooperative_matrix'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV: Self = Self(1_000_249_000);
    pub const COOPERATIVE_MATRIX_PROPERTIES_NV: Self = Self(1_000_249_001);
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_NV: Self = Self(1_000_249_002);
}
#[doc = "Generated from 'VK_NV_coverage_reduction_mode'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COVERAGE_REDUCTION_MODE_FEATURES_NV: Self = Self(1_000_250_000);
    pub const PIPELINE_COVERAGE_REDUCTION_STATE_CREATE_INFO_NV: Self = Self(1_000_250_001);
    pub const FRAMEBUFFER_MIXED_SAMPLES_COMBINATION_NV: Self = Self(1_000_250_002);
}
#[doc = "Generated from 'VK_EXT_fragment_shader_interlock'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT: Self = Self(1_000_251_000);
}
#[doc = "Generated from 'VK_EXT_ycbcr_image_arrays'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_YCBCR_IMAGE_ARRAYS_FEATURES_EXT: Self = Self(1_000_252_000);
}
#[doc = "Generated from 'VK_KHR_uniform_buffer_standard_layout'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES;
}
#[doc = "Generated from 'VK_EXT_provoking_vertex'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT: Self = Self(1_000_254_000);
    pub const PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT: Self =
        Self(1_000_254_001);
    pub const PHYSICAL_DEVICE_PROVOKING_VERTEX_PROPERTIES_EXT: Self = Self(1_000_254_002);
}
#[doc = "Generated from 'VK_EXT_full_screen_exclusive'"]
impl Result {
    pub const ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: Self = Self(-1_000_255_000);
}
#[doc = "Generated from 'VK_EXT_full_screen_exclusive'"]
impl StructureType {
    pub const SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT: Self = Self(1_000_255_000);
    pub const SURFACE_CAPABILITIES_FULL_SCREEN_EXCLUSIVE_EXT: Self = Self(1_000_255_002);
    pub const SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT: Self = Self(1_000_255_001);
}
#[doc = "Generated from 'VK_EXT_headless_surface'"]
impl StructureType {
    pub const HEADLESS_SURFACE_CREATE_INFO_EXT: Self = Self(1_000_256_000);
}
#[doc = "Generated from 'VK_KHR_buffer_device_address'"]
impl BufferCreateFlags {
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY_KHR: Self = Self::DEVICE_ADDRESS_CAPTURE_REPLAY;
}
#[doc = "Generated from 'VK_KHR_buffer_device_address'"]
impl BufferUsageFlags {
    pub const SHADER_DEVICE_ADDRESS_KHR: Self = Self::SHADER_DEVICE_ADDRESS;
}
#[doc = "Generated from 'VK_KHR_buffer_device_address'"]
impl MemoryAllocateFlags {
    pub const DEVICE_ADDRESS_KHR: Self = Self::DEVICE_ADDRESS;
    pub const DEVICE_ADDRESS_CAPTURE_REPLAY_KHR: Self = Self::DEVICE_ADDRESS_CAPTURE_REPLAY;
}
#[doc = "Generated from 'VK_KHR_buffer_device_address'"]
impl Result {
    pub const ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR: Self =
        Self::ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
}
#[doc = "Generated from 'VK_KHR_buffer_device_address'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    pub const BUFFER_DEVICE_ADDRESS_INFO_KHR: Self = Self::BUFFER_DEVICE_ADDRESS_INFO;
    pub const BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO_KHR: Self =
        Self::BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO;
    pub const MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO_KHR: Self =
        Self::MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO;
    pub const DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO_KHR: Self =
        Self::DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO;
}
#[doc = "Generated from 'VK_EXT_line_rasterization'"]
impl DynamicState {
    pub const LINE_STIPPLE_EXT: Self = Self::LINE_STIPPLE;
}
#[doc = "Generated from 'VK_EXT_line_rasterization'"]
impl LineRasterizationMode {
    pub const DEFAULT_EXT: Self = Self::DEFAULT;
    pub const RECTANGULAR_EXT: Self = Self::RECTANGULAR;
    pub const BRESENHAM_EXT: Self = Self::BRESENHAM;
    pub const RECTANGULAR_SMOOTH_EXT: Self = Self::RECTANGULAR_SMOOTH;
}
#[doc = "Generated from 'VK_EXT_line_rasterization'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES;
    pub const PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT: Self =
        Self::PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO;
    pub const PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES;
}
#[doc = "Generated from 'VK_EXT_shader_atomic_float'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT: Self = Self(1_000_260_000);
}
#[doc = "Generated from 'VK_EXT_host_query_reset'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES;
}
#[doc = "Generated from 'VK_EXT_index_type_uint8'"]
impl IndexType {
    pub const UINT8_EXT: Self = Self::UINT8;
}
#[doc = "Generated from 'VK_EXT_index_type_uint8'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES;
}
#[doc = "Generated from 'VK_EXT_extended_dynamic_state'"]
impl DynamicState {
    pub const CULL_MODE_EXT: Self = Self::CULL_MODE;
    pub const FRONT_FACE_EXT: Self = Self::FRONT_FACE;
    pub const PRIMITIVE_TOPOLOGY_EXT: Self = Self::PRIMITIVE_TOPOLOGY;
    pub const VIEWPORT_WITH_COUNT_EXT: Self = Self::VIEWPORT_WITH_COUNT;
    pub const SCISSOR_WITH_COUNT_EXT: Self = Self::SCISSOR_WITH_COUNT;
    pub const VERTEX_INPUT_BINDING_STRIDE_EXT: Self = Self::VERTEX_INPUT_BINDING_STRIDE;
    pub const DEPTH_TEST_ENABLE_EXT: Self = Self::DEPTH_TEST_ENABLE;
    pub const DEPTH_WRITE_ENABLE_EXT: Self = Self::DEPTH_WRITE_ENABLE;
    pub const DEPTH_COMPARE_OP_EXT: Self = Self::DEPTH_COMPARE_OP;
    pub const DEPTH_BOUNDS_TEST_ENABLE_EXT: Self = Self::DEPTH_BOUNDS_TEST_ENABLE;
    pub const STENCIL_TEST_ENABLE_EXT: Self = Self::STENCIL_TEST_ENABLE;
    pub const STENCIL_OP_EXT: Self = Self::STENCIL_OP;
}
#[doc = "Generated from 'VK_EXT_extended_dynamic_state'"]
impl StructureType {
    #[doc = "Not promoted to 1.3"]
    pub const PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT: Self = Self(1_000_267_000);
}
#[doc = "Generated from 'VK_KHR_deferred_host_operations'"]
impl ObjectType {
    pub const DEFERRED_OPERATION_KHR: Self = Self(1_000_268_000);
}
#[doc = "Generated from 'VK_KHR_deferred_host_operations'"]
impl Result {
    pub const THREAD_IDLE_KHR: Self = Self(1_000_268_000);
    pub const THREAD_DONE_KHR: Self = Self(1_000_268_001);
    pub const OPERATION_DEFERRED_KHR: Self = Self(1_000_268_002);
    pub const OPERATION_NOT_DEFERRED_KHR: Self = Self(1_000_268_003);
}
#[doc = "Generated from 'VK_KHR_pipeline_executable_properties'"]
impl PipelineCreateFlags {
    pub const CAPTURE_STATISTICS_KHR: Self = Self(0b100_0000);
    pub const CAPTURE_INTERNAL_REPRESENTATIONS_KHR: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_KHR_pipeline_executable_properties'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR: Self =
        Self(1_000_269_000);
    pub const PIPELINE_INFO_KHR: Self = Self(1_000_269_001);
    pub const PIPELINE_EXECUTABLE_PROPERTIES_KHR: Self = Self(1_000_269_002);
    pub const PIPELINE_EXECUTABLE_INFO_KHR: Self = Self(1_000_269_003);
    pub const PIPELINE_EXECUTABLE_STATISTIC_KHR: Self = Self(1_000_269_004);
    pub const PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR: Self = Self(1_000_269_005);
}
#[doc = "Generated from 'VK_EXT_host_image_copy'"]
impl FormatFeatureFlags2 {
    #[doc = "Host image copies are supported"]
    pub const HOST_IMAGE_TRANSFER_EXT: Self = Self::HOST_IMAGE_TRANSFER;
}
#[doc = "Generated from 'VK_EXT_host_image_copy'"]
impl HostImageCopyFlags {
    pub const MEMCPY_EXT: Self = Self::MEMCPY;
}
#[doc = "Generated from 'VK_EXT_host_image_copy'"]
impl ImageUsageFlags {
    #[doc = "Can be used with host image copies"]
    pub const HOST_TRANSFER_EXT: Self = Self::HOST_TRANSFER;
}
#[doc = "Generated from 'VK_EXT_host_image_copy'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES;
    pub const PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES;
    pub const MEMORY_TO_IMAGE_COPY_EXT: Self = Self::MEMORY_TO_IMAGE_COPY;
    pub const IMAGE_TO_MEMORY_COPY_EXT: Self = Self::IMAGE_TO_MEMORY_COPY;
    pub const COPY_IMAGE_TO_MEMORY_INFO_EXT: Self = Self::COPY_IMAGE_TO_MEMORY_INFO;
    pub const COPY_MEMORY_TO_IMAGE_INFO_EXT: Self = Self::COPY_MEMORY_TO_IMAGE_INFO;
    pub const HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT: Self = Self::HOST_IMAGE_LAYOUT_TRANSITION_INFO;
    pub const COPY_IMAGE_TO_IMAGE_INFO_EXT: Self = Self::COPY_IMAGE_TO_IMAGE_INFO;
    pub const SUBRESOURCE_HOST_MEMCPY_SIZE_EXT: Self = Self::SUBRESOURCE_HOST_MEMCPY_SIZE;
    pub const HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT: Self =
        Self::HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY;
}
#[doc = "Generated from 'VK_KHR_map_memory2'"]
impl StructureType {
    pub const MEMORY_MAP_INFO_KHR: Self = Self::MEMORY_MAP_INFO;
    pub const MEMORY_UNMAP_INFO_KHR: Self = Self::MEMORY_UNMAP_INFO;
}
#[doc = "Generated from 'VK_EXT_map_memory_placed'"]
impl MemoryMapFlags {
    pub const PLACED_EXT: Self = Self(0b1);
}
#[doc = "Generated from 'VK_EXT_map_memory_placed'"]
impl MemoryUnmapFlags {
    pub const RESERVE_EXT: Self = Self(0b1);
}
#[doc = "Generated from 'VK_EXT_map_memory_placed'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT: Self = Self(1_000_272_000);
    pub const PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT: Self = Self(1_000_272_001);
    pub const MEMORY_MAP_PLACED_INFO_EXT: Self = Self(1_000_272_002);
}
#[doc = "Generated from 'VK_EXT_shader_atomic_float2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT: Self = Self(1_000_273_000);
}
#[doc = "Generated from 'VK_EXT_surface_maintenance1'"]
impl StructureType {
    pub const SURFACE_PRESENT_MODE_EXT: Self = Self::SURFACE_PRESENT_MODE_KHR;
    pub const SURFACE_PRESENT_SCALING_CAPABILITIES_EXT: Self =
        Self::SURFACE_PRESENT_SCALING_CAPABILITIES_KHR;
    pub const SURFACE_PRESENT_MODE_COMPATIBILITY_EXT: Self =
        Self::SURFACE_PRESENT_MODE_COMPATIBILITY_KHR;
}
#[doc = "Generated from 'VK_EXT_swapchain_maintenance1'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR;
    pub const SWAPCHAIN_PRESENT_FENCE_INFO_EXT: Self = Self::SWAPCHAIN_PRESENT_FENCE_INFO_KHR;
    pub const SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT: Self =
        Self::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR;
    pub const SWAPCHAIN_PRESENT_MODE_INFO_EXT: Self = Self::SWAPCHAIN_PRESENT_MODE_INFO_KHR;
    pub const SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT: Self =
        Self::SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_KHR;
    pub const RELEASE_SWAPCHAIN_IMAGES_INFO_EXT: Self = Self::RELEASE_SWAPCHAIN_IMAGES_INFO_KHR;
}
#[doc = "Generated from 'VK_EXT_swapchain_maintenance1'"]
impl SwapchainCreateFlagsKHR {
    pub const DEFERRED_MEMORY_ALLOCATION_EXT: Self = Self::DEFERRED_MEMORY_ALLOCATION;
}
#[doc = "Generated from 'VK_EXT_shader_demote_to_helper_invocation'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES;
}
#[doc = "Generated from 'VK_NV_device_generated_commands'"]
impl AccessFlags {
    pub const COMMAND_PREPROCESS_READ_NV: Self = Self::COMMAND_PREPROCESS_READ_EXT;
    pub const COMMAND_PREPROCESS_WRITE_NV: Self = Self::COMMAND_PREPROCESS_WRITE_EXT;
}
#[doc = "Generated from 'VK_NV_device_generated_commands'"]
impl ObjectType {
    pub const INDIRECT_COMMANDS_LAYOUT_NV: Self = Self(1_000_277_000);
}
#[doc = "Generated from 'VK_NV_device_generated_commands'"]
impl PipelineCreateFlags {
    pub const INDIRECT_BINDABLE_NV: Self = Self(0b100_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_device_generated_commands'"]
impl PipelineStageFlags {
    pub const COMMAND_PREPROCESS_NV: Self = Self::COMMAND_PREPROCESS_EXT;
}
#[doc = "Generated from 'VK_NV_device_generated_commands'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_NV: Self = Self(1_000_277_000);
    pub const GRAPHICS_SHADER_GROUP_CREATE_INFO_NV: Self = Self(1_000_277_001);
    pub const GRAPHICS_PIPELINE_SHADER_GROUPS_CREATE_INFO_NV: Self = Self(1_000_277_002);
    pub const INDIRECT_COMMANDS_LAYOUT_TOKEN_NV: Self = Self(1_000_277_003);
    pub const INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_NV: Self = Self(1_000_277_004);
    pub const GENERATED_COMMANDS_INFO_NV: Self = Self(1_000_277_005);
    pub const GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_NV: Self = Self(1_000_277_006);
    pub const PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV: Self = Self(1_000_277_007);
}
#[doc = "Generated from 'VK_NV_inherited_viewport_scissor'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES_NV: Self = Self(1_000_278_000);
    pub const COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO_NV: Self = Self(1_000_278_001);
}
#[doc = "Generated from 'VK_KHR_shader_integer_dot_product'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    pub const PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
}
#[doc = "Generated from 'VK_EXT_texel_buffer_alignment'"]
impl StructureType {
    #[doc = "Not promoted to 1.3"]
    pub const PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT: Self = Self(1_000_281_000);
    pub const PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES;
}
#[doc = "Generated from 'VK_QCOM_render_pass_transform'"]
impl RenderPassCreateFlags {
    pub const TRANSFORM_QCOM: Self = Self(0b10);
}
#[doc = "Generated from 'VK_QCOM_render_pass_transform'"]
impl StructureType {
    pub const COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM: Self =
        Self(1_000_282_000);
    pub const RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM: Self = Self(1_000_282_001);
}
#[doc = "Generated from 'VK_EXT_depth_bias_control'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES_EXT: Self = Self(1_000_283_000);
    pub const DEPTH_BIAS_INFO_EXT: Self = Self(1_000_283_001);
    pub const DEPTH_BIAS_REPRESENTATION_INFO_EXT: Self = Self(1_000_283_002);
}
#[doc = "Generated from 'VK_EXT_device_memory_report'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT: Self = Self(1_000_284_000);
    pub const DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT: Self = Self(1_000_284_001);
    pub const DEVICE_MEMORY_REPORT_CALLBACK_DATA_EXT: Self = Self(1_000_284_002);
}
#[doc = "Generated from 'VK_EXT_robustness2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR;
    pub const PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT: Self =
        Self::PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_KHR;
}
#[doc = "Generated from 'VK_EXT_custom_border_color'"]
impl BorderColor {
    pub const FLOAT_CUSTOM_EXT: Self = Self(1_000_287_003);
    pub const INT_CUSTOM_EXT: Self = Self(1_000_287_004);
}
#[doc = "Generated from 'VK_EXT_custom_border_color'"]
impl StructureType {
    pub const SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT: Self = Self(1_000_287_000);
    pub const PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT: Self = Self(1_000_287_001);
    pub const PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT: Self = Self(1_000_287_002);
}
#[doc = "Generated from 'VK_EXT_texture_compression_astc_3d'"]
impl Format {
    pub const ASTC_3X3X3_UNORM_BLOCK_EXT: Self = Self(1_000_288_000);
    pub const ASTC_3X3X3_SRGB_BLOCK_EXT: Self = Self(1_000_288_001);
    pub const ASTC_3X3X3_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_002);
    pub const ASTC_4X3X3_UNORM_BLOCK_EXT: Self = Self(1_000_288_003);
    pub const ASTC_4X3X3_SRGB_BLOCK_EXT: Self = Self(1_000_288_004);
    pub const ASTC_4X3X3_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_005);
    pub const ASTC_4X4X3_UNORM_BLOCK_EXT: Self = Self(1_000_288_006);
    pub const ASTC_4X4X3_SRGB_BLOCK_EXT: Self = Self(1_000_288_007);
    pub const ASTC_4X4X3_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_008);
    pub const ASTC_4X4X4_UNORM_BLOCK_EXT: Self = Self(1_000_288_009);
    pub const ASTC_4X4X4_SRGB_BLOCK_EXT: Self = Self(1_000_288_010);
    pub const ASTC_4X4X4_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_011);
    pub const ASTC_5X4X4_UNORM_BLOCK_EXT: Self = Self(1_000_288_012);
    pub const ASTC_5X4X4_SRGB_BLOCK_EXT: Self = Self(1_000_288_013);
    pub const ASTC_5X4X4_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_014);
    pub const ASTC_5X5X4_UNORM_BLOCK_EXT: Self = Self(1_000_288_015);
    pub const ASTC_5X5X4_SRGB_BLOCK_EXT: Self = Self(1_000_288_016);
    pub const ASTC_5X5X4_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_017);
    pub const ASTC_5X5X5_UNORM_BLOCK_EXT: Self = Self(1_000_288_018);
    pub const ASTC_5X5X5_SRGB_BLOCK_EXT: Self = Self(1_000_288_019);
    pub const ASTC_5X5X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_020);
    pub const ASTC_6X5X5_UNORM_BLOCK_EXT: Self = Self(1_000_288_021);
    pub const ASTC_6X5X5_SRGB_BLOCK_EXT: Self = Self(1_000_288_022);
    pub const ASTC_6X5X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_023);
    pub const ASTC_6X6X5_UNORM_BLOCK_EXT: Self = Self(1_000_288_024);
    pub const ASTC_6X6X5_SRGB_BLOCK_EXT: Self = Self(1_000_288_025);
    pub const ASTC_6X6X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_026);
    pub const ASTC_6X6X6_UNORM_BLOCK_EXT: Self = Self(1_000_288_027);
    pub const ASTC_6X6X6_SRGB_BLOCK_EXT: Self = Self(1_000_288_028);
    pub const ASTC_6X6X6_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_029);
}
#[doc = "Generated from 'VK_EXT_texture_compression_astc_3d'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_3D_FEATURES_EXT: Self = Self(1_000_288_000);
}
#[doc = "Generated from 'VK_KHR_pipeline_library'"]
impl PipelineCreateFlags {
    pub const LIBRARY_KHR: Self = Self(0b1000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_pipeline_library'"]
impl StructureType {
    pub const PIPELINE_LIBRARY_CREATE_INFO_KHR: Self = Self(1_000_290_000);
}
#[doc = "Generated from 'VK_NV_present_barrier'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRESENT_BARRIER_FEATURES_NV: Self = Self(1_000_292_000);
    pub const SURFACE_CAPABILITIES_PRESENT_BARRIER_NV: Self = Self(1_000_292_001);
    pub const SWAPCHAIN_PRESENT_BARRIER_CREATE_INFO_NV: Self = Self(1_000_292_002);
}
#[doc = "Generated from 'VK_KHR_present_id'"]
impl StructureType {
    pub const PRESENT_ID_KHR: Self = Self(1_000_294_000);
    pub const PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR: Self = Self(1_000_294_001);
}
#[doc = "Generated from 'VK_EXT_private_data'"]
impl ObjectType {
    pub const PRIVATE_DATA_SLOT_EXT: Self = Self::PRIVATE_DATA_SLOT;
}
#[doc = "Generated from 'VK_EXT_private_data'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES;
    pub const DEVICE_PRIVATE_DATA_CREATE_INFO_EXT: Self = Self::DEVICE_PRIVATE_DATA_CREATE_INFO;
    pub const PRIVATE_DATA_SLOT_CREATE_INFO_EXT: Self = Self::PRIVATE_DATA_SLOT_CREATE_INFO;
}
#[doc = "Generated from 'VK_EXT_pipeline_creation_cache_control'"]
impl PipelineCacheCreateFlags {
    pub const EXTERNALLY_SYNCHRONIZED_EXT: Self = Self::EXTERNALLY_SYNCHRONIZED;
}
#[doc = "Generated from 'VK_EXT_pipeline_creation_cache_control'"]
impl PipelineCreateFlags {
    pub const FAIL_ON_PIPELINE_COMPILE_REQUIRED_EXT: Self = Self::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
    pub const EARLY_RETURN_ON_FAILURE_EXT: Self = Self::EARLY_RETURN_ON_FAILURE;
}
#[doc = "Generated from 'VK_EXT_pipeline_creation_cache_control'"]
impl Result {
    pub const PIPELINE_COMPILE_REQUIRED_EXT: Self = Self::PIPELINE_COMPILE_REQUIRED;
    pub const ERROR_PIPELINE_COMPILE_REQUIRED_EXT: Self = Self::PIPELINE_COMPILE_REQUIRED;
}
#[doc = "Generated from 'VK_EXT_pipeline_creation_cache_control'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES;
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl AccessFlags2 {
    pub const VIDEO_ENCODE_READ_KHR: Self = Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const VIDEO_ENCODE_WRITE_KHR: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl BufferUsageFlags {
    pub const VIDEO_ENCODE_DST_KHR: Self = Self(0b1000_0000_0000_0000);
    pub const VIDEO_ENCODE_SRC_KHR: Self = Self(0b1_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl FormatFeatureFlags {
    pub const VIDEO_ENCODE_INPUT_KHR: Self = Self(0b1000_0000_0000_0000_0000_0000_0000);
    pub const VIDEO_ENCODE_DPB_KHR: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl FormatFeatureFlags2 {
    pub const VIDEO_ENCODE_INPUT_KHR: Self = Self(0b1000_0000_0000_0000_0000_0000_0000);
    pub const VIDEO_ENCODE_DPB_KHR: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl ImageLayout {
    pub const VIDEO_ENCODE_DST_KHR: Self = Self(1_000_299_000);
    pub const VIDEO_ENCODE_SRC_KHR: Self = Self(1_000_299_001);
    pub const VIDEO_ENCODE_DPB_KHR: Self = Self(1_000_299_002);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl ImageUsageFlags {
    pub const VIDEO_ENCODE_DST_KHR: Self = Self(0b10_0000_0000_0000);
    pub const VIDEO_ENCODE_SRC_KHR: Self = Self(0b100_0000_0000_0000);
    pub const VIDEO_ENCODE_DPB_KHR: Self = Self(0b1000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl PipelineStageFlags2 {
    pub const VIDEO_ENCODE_KHR: Self = Self(0b1000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl QueryResultStatusKHR {
    pub const INSUFFICIENTSTREAM_BUFFER_RANGE: Self = Self(-1_000_299_000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl QueryType {
    pub const VIDEO_ENCODE_FEEDBACK_KHR: Self = Self(1_000_299_000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl QueueFlags {
    pub const VIDEO_ENCODE_KHR: Self = Self(0b100_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl Result {
    pub const ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR: Self = Self(-1_000_299_000);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl StructureType {
    pub const VIDEO_ENCODE_INFO_KHR: Self = Self(1_000_299_000);
    pub const VIDEO_ENCODE_RATE_CONTROL_INFO_KHR: Self = Self(1_000_299_001);
    pub const VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR: Self = Self(1_000_299_002);
    pub const VIDEO_ENCODE_CAPABILITIES_KHR: Self = Self(1_000_299_003);
    pub const VIDEO_ENCODE_USAGE_INFO_KHR: Self = Self(1_000_299_004);
    pub const QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR: Self = Self(1_000_299_005);
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR: Self = Self(1_000_299_006);
    pub const VIDEO_ENCODE_QUALITY_LEVEL_PROPERTIES_KHR: Self = Self(1_000_299_007);
    pub const VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR: Self = Self(1_000_299_008);
    pub const VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR: Self = Self(1_000_299_009);
    pub const VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR: Self = Self(1_000_299_010);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl VideoCodingControlFlagsKHR {
    pub const ENCODE_RATE_CONTROL: Self = Self(0b10);
    pub const ENCODE_QUALITY_LEVEL: Self = Self(0b100);
}
#[doc = "Generated from 'VK_KHR_video_encode_queue'"]
impl VideoSessionCreateFlagsKHR {
    pub const ALLOW_ENCODE_PARAMETER_OPTIMIZATIONS: Self = Self(0b10);
}
#[doc = "Generated from 'VK_NV_device_diagnostics_config'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV: Self = Self(1_000_300_000);
    pub const DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV: Self = Self(1_000_300_001);
}
#[doc = "Generated from 'VK_QCOM_render_pass_store_ops'"]
impl AttachmentStoreOp {
    pub const NONE_QCOM: Self = Self::NONE;
}
#[doc = "Generated from 'VK_QCOM_queue_perf_hint'"]
impl StructureType {
    pub const PERF_HINT_INFO_QCOM: Self = Self(1_000_302_000);
    pub const PHYSICAL_DEVICE_QUEUE_PERF_HINT_FEATURES_QCOM: Self = Self(1_000_302_001);
    pub const PHYSICAL_DEVICE_QUEUE_PERF_HINT_PROPERTIES_QCOM: Self = Self(1_000_302_002);
}
#[doc = "Generated from 'VK_QCOM_image_processing3'"]
impl FormatFeatureFlags2 {
    pub const BLOCK_MATCHING_SXD_QCOM: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_QCOM_image_processing3'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_3_FEATURES_QCOM: Self = Self(1_000_303_000);
}
#[doc = "Generated from 'VK_QCOM_shader_multiple_wait_queues'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_MULTIPLE_WAIT_QUEUES_FEATURES_QCOM: Self = Self(1_000_304_000);
    pub const PHYSICAL_DEVICE_SHADER_MULTIPLE_WAIT_QUEUES_PROPERTIES_QCOM: Self =
        Self(1_000_304_001);
}
#[doc = "Generated from 'VK_EXT_shader_split_barrier'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_SPLIT_BARRIER_FEATURES_EXT: Self = Self(1_000_305_000);
    pub const PHYSICAL_DEVICE_SHADER_SPLIT_BARRIER_PROPERTIES_EXT: Self = Self(1_000_305_001);
}
#[doc = "Generated from 'VK_NV_cuda_kernel_launch'"]
#[cfg(feature = "provisional")]
impl DebugReportObjectTypeEXT {
    pub const CUDA_MODULE_NV: Self = Self(1_000_307_000);
    pub const CUDA_FUNCTION_NV: Self = Self(1_000_307_001);
}
#[doc = "Generated from 'VK_NV_cuda_kernel_launch'"]
#[cfg(feature = "provisional")]
impl ObjectType {
    pub const CUDA_MODULE_NV: Self = Self(1_000_307_000);
    pub const CUDA_FUNCTION_NV: Self = Self(1_000_307_001);
}
#[doc = "Generated from 'VK_NV_cuda_kernel_launch'"]
#[cfg(feature = "provisional")]
impl StructureType {
    pub const CUDA_MODULE_CREATE_INFO_NV: Self = Self(1_000_307_000);
    pub const CUDA_FUNCTION_CREATE_INFO_NV: Self = Self(1_000_307_001);
    pub const CUDA_LAUNCH_INFO_NV: Self = Self(1_000_307_002);
    pub const PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV: Self = Self(1_000_307_003);
    pub const PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES_NV: Self = Self(1_000_307_004);
}
#[doc = "Generated from 'VK_QCOM_tile_shading'"]
impl AccessFlags2 {
    pub const SHADER_TILE_ATTACHMENT_READ_QCOM: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const SHADER_TILE_ATTACHMENT_WRITE_QCOM: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_QCOM_tile_shading'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_TILE_SHADING_FEATURES_QCOM: Self = Self(1_000_309_000);
    pub const PHYSICAL_DEVICE_TILE_SHADING_PROPERTIES_QCOM: Self = Self(1_000_309_001);
    pub const RENDER_PASS_TILE_SHADING_CREATE_INFO_QCOM: Self = Self(1_000_309_002);
    pub const PER_TILE_BEGIN_INFO_QCOM: Self = Self(1_000_309_003);
    pub const PER_TILE_END_INFO_QCOM: Self = Self(1_000_309_004);
    pub const DISPATCH_TILE_INFO_QCOM: Self = Self(1_000_309_005);
}
#[doc = "Generated from 'VK_QCOM_tile_shading'"]
impl SubpassDescriptionFlags {
    pub const TILE_SHADING_APRON_QCOM: Self = Self(0b1_0000_0000);
}
#[doc = "Generated from 'VK_NV_low_latency'"]
impl StructureType {
    pub const QUERY_LOW_LATENCY_SUPPORT_NV: Self = Self(1_000_310_000);
}
#[doc = "Generated from 'VK_EXT_metal_objects'"]
impl StructureType {
    pub const EXPORT_METAL_OBJECT_CREATE_INFO_EXT: Self = Self(1_000_311_000);
    pub const EXPORT_METAL_OBJECTS_INFO_EXT: Self = Self(1_000_311_001);
    pub const EXPORT_METAL_DEVICE_INFO_EXT: Self = Self(1_000_311_002);
    pub const EXPORT_METAL_COMMAND_QUEUE_INFO_EXT: Self = Self(1_000_311_003);
    pub const EXPORT_METAL_BUFFER_INFO_EXT: Self = Self(1_000_311_004);
    pub const IMPORT_METAL_BUFFER_INFO_EXT: Self = Self(1_000_311_005);
    pub const EXPORT_METAL_TEXTURE_INFO_EXT: Self = Self(1_000_311_006);
    pub const IMPORT_METAL_TEXTURE_INFO_EXT: Self = Self(1_000_311_007);
    pub const EXPORT_METAL_IO_SURFACE_INFO_EXT: Self = Self(1_000_311_008);
    pub const IMPORT_METAL_IO_SURFACE_INFO_EXT: Self = Self(1_000_311_009);
    pub const EXPORT_METAL_SHARED_EVENT_INFO_EXT: Self = Self(1_000_311_010);
    pub const IMPORT_METAL_SHARED_EVENT_INFO_EXT: Self = Self(1_000_311_011);
}
#[doc = "Generated from 'VK_KHR_synchronization2'"]
impl AccessFlags {
    pub const NONE_KHR: Self = Self::NONE;
}
#[doc = "Generated from 'VK_KHR_synchronization2'"]
impl AccessFlags2 {
    pub const NONE_KHR: Self = Self::NONE;
    pub const INDIRECT_COMMAND_READ_KHR: Self = Self::INDIRECT_COMMAND_READ;
    pub const INDEX_READ_KHR: Self = Self::INDEX_READ;
    pub const VERTEX_ATTRIBUTE_READ_KHR: Self = Self::VERTEX_ATTRIBUTE_READ;
    pub const UNIFORM_READ_KHR: Self = Self::UNIFORM_READ;
    pub const INPUT_ATTACHMENT_READ_KHR: Self = Self::INPUT_ATTACHMENT_READ;
    pub const SHADER_READ_KHR: Self = Self::SHADER_READ;
    pub const SHADER_WRITE_KHR: Self = Self::SHADER_WRITE;
    pub const COLOR_ATTACHMENT_READ_KHR: Self = Self::COLOR_ATTACHMENT_READ;
    pub const COLOR_ATTACHMENT_WRITE_KHR: Self = Self::COLOR_ATTACHMENT_WRITE;
    pub const DEPTH_STENCIL_ATTACHMENT_READ_KHR: Self = Self::DEPTH_STENCIL_ATTACHMENT_READ;
    pub const DEPTH_STENCIL_ATTACHMENT_WRITE_KHR: Self = Self::DEPTH_STENCIL_ATTACHMENT_WRITE;
    pub const TRANSFER_READ_KHR: Self = Self::TRANSFER_READ;
    pub const TRANSFER_WRITE_KHR: Self = Self::TRANSFER_WRITE;
    pub const HOST_READ_KHR: Self = Self::HOST_READ;
    pub const HOST_WRITE_KHR: Self = Self::HOST_WRITE;
    pub const MEMORY_READ_KHR: Self = Self::MEMORY_READ;
    pub const MEMORY_WRITE_KHR: Self = Self::MEMORY_WRITE;
    pub const SHADER_SAMPLED_READ_KHR: Self = Self::SHADER_SAMPLED_READ;
    pub const SHADER_STORAGE_READ_KHR: Self = Self::SHADER_STORAGE_READ;
    pub const SHADER_STORAGE_WRITE_KHR: Self = Self::SHADER_STORAGE_WRITE;
    pub const TRANSFORM_FEEDBACK_WRITE_EXT: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
    pub const TRANSFORM_FEEDBACK_COUNTER_READ_EXT: Self = Self(0b100_0000_0000_0000_0000_0000_0000);
    pub const TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000);
    #[doc = "read access flag for reading conditional rendering predicate"]
    pub const CONDITIONAL_RENDERING_READ_EXT: Self = Self(0b1_0000_0000_0000_0000_0000);
    pub const COMMAND_PREPROCESS_READ_NV: Self = Self::COMMAND_PREPROCESS_READ_EXT;
    pub const COMMAND_PREPROCESS_WRITE_NV: Self = Self::COMMAND_PREPROCESS_WRITE_EXT;
    pub const COMMAND_PREPROCESS_READ_EXT: Self = Self(0b10_0000_0000_0000_0000);
    pub const COMMAND_PREPROCESS_WRITE_EXT: Self = Self(0b100_0000_0000_0000_0000);
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR: Self =
        Self(0b1000_0000_0000_0000_0000_0000);
    pub const SHADING_RATE_IMAGE_READ_NV: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR;
    pub const ACCELERATION_STRUCTURE_READ_KHR: Self = Self(0b10_0000_0000_0000_0000_0000);
    pub const ACCELERATION_STRUCTURE_WRITE_KHR: Self = Self(0b100_0000_0000_0000_0000_0000);
    pub const ACCELERATION_STRUCTURE_READ_NV: Self = Self::ACCELERATION_STRUCTURE_READ_KHR;
    pub const ACCELERATION_STRUCTURE_WRITE_NV: Self = Self::ACCELERATION_STRUCTURE_WRITE_KHR;
    pub const FRAGMENT_DENSITY_MAP_READ_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
    pub const COLOR_ATTACHMENT_READ_NONCOHERENT_EXT: Self = Self(0b1000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_synchronization2'"]
impl EventCreateFlags {
    pub const DEVICE_ONLY_KHR: Self = Self::DEVICE_ONLY;
}
#[doc = "Generated from 'VK_KHR_synchronization2'"]
impl ImageLayout {
    pub const READ_ONLY_OPTIMAL_KHR: Self = Self::READ_ONLY_OPTIMAL;
    pub const ATTACHMENT_OPTIMAL_KHR: Self = Self::ATTACHMENT_OPTIMAL;
}
#[doc = "Generated from 'VK_KHR_synchronization2'"]
impl PipelineStageFlags {
    pub const NONE_KHR: Self = Self::NONE;
}
#[doc = "Generated from 'VK_KHR_synchronization2'"]
impl PipelineStageFlags2 {
    pub const NONE_KHR: Self = Self::NONE;
    pub const TOP_OF_PIPE_KHR: Self = Self::TOP_OF_PIPE;
    pub const DRAW_INDIRECT_KHR: Self = Self::DRAW_INDIRECT;
    pub const VERTEX_INPUT_KHR: Self = Self::VERTEX_INPUT;
    pub const VERTEX_SHADER_KHR: Self = Self::VERTEX_SHADER;
    pub const TESSELLATION_CONTROL_SHADER_KHR: Self = Self::TESSELLATION_CONTROL_SHADER;
    pub const TESSELLATION_EVALUATION_SHADER_KHR: Self = Self::TESSELLATION_EVALUATION_SHADER;
    pub const GEOMETRY_SHADER_KHR: Self = Self::GEOMETRY_SHADER;
    pub const FRAGMENT_SHADER_KHR: Self = Self::FRAGMENT_SHADER;
    pub const EARLY_FRAGMENT_TESTS_KHR: Self = Self::EARLY_FRAGMENT_TESTS;
    pub const LATE_FRAGMENT_TESTS_KHR: Self = Self::LATE_FRAGMENT_TESTS;
    pub const COLOR_ATTACHMENT_OUTPUT_KHR: Self = Self::COLOR_ATTACHMENT_OUTPUT;
    pub const COMPUTE_SHADER_KHR: Self = Self::COMPUTE_SHADER;
    pub const ALL_TRANSFER_KHR: Self = Self::ALL_TRANSFER;
    pub const TRANSFER_KHR: Self = Self::ALL_TRANSFER;
    pub const BOTTOM_OF_PIPE_KHR: Self = Self::BOTTOM_OF_PIPE;
    pub const HOST_KHR: Self = Self::HOST;
    pub const ALL_GRAPHICS_KHR: Self = Self::ALL_GRAPHICS;
    pub const ALL_COMMANDS_KHR: Self = Self::ALL_COMMANDS;
    pub const COPY_KHR: Self = Self::COPY;
    pub const RESOLVE_KHR: Self = Self::RESOLVE;
    pub const BLIT_KHR: Self = Self::BLIT;
    pub const CLEAR_KHR: Self = Self::CLEAR;
    pub const INDEX_INPUT_KHR: Self = Self::INDEX_INPUT;
    pub const VERTEX_ATTRIBUTE_INPUT_KHR: Self = Self::VERTEX_ATTRIBUTE_INPUT;
    pub const PRE_RASTERIZATION_SHADERS_KHR: Self = Self::PRE_RASTERIZATION_SHADERS;
    pub const TRANSFORM_FEEDBACK_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
    #[doc = "A pipeline stage for conditional rendering predicate fetch"]
    pub const CONDITIONAL_RENDERING_EXT: Self = Self(0b100_0000_0000_0000_0000);
    pub const COMMAND_PREPROCESS_NV: Self = Self::COMMAND_PREPROCESS_EXT;
    pub const COMMAND_PREPROCESS_EXT: Self = Self(0b10_0000_0000_0000_0000);
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT_KHR: Self = Self(0b100_0000_0000_0000_0000_0000);
    pub const SHADING_RATE_IMAGE_NV: Self = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR;
    pub const ACCELERATION_STRUCTURE_BUILD_KHR: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
    pub const RAY_TRACING_SHADER_KHR: Self = Self(0b10_0000_0000_0000_0000_0000);
    pub const RAY_TRACING_SHADER_NV: Self = Self::RAY_TRACING_SHADER_KHR;
    pub const ACCELERATION_STRUCTURE_BUILD_NV: Self = Self::ACCELERATION_STRUCTURE_BUILD_KHR;
    pub const FRAGMENT_DENSITY_PROCESS_EXT: Self = Self(0b1000_0000_0000_0000_0000_0000);
    pub const TASK_SHADER_NV: Self = Self::TASK_SHADER_EXT;
    pub const MESH_SHADER_NV: Self = Self::MESH_SHADER_EXT;
    pub const TASK_SHADER_EXT: Self = Self(0b1000_0000_0000_0000_0000);
    pub const MESH_SHADER_EXT: Self = Self(0b1_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_synchronization2'"]
impl StructureType {
    pub const MEMORY_BARRIER_2_KHR: Self = Self::MEMORY_BARRIER_2;
    pub const BUFFER_MEMORY_BARRIER_2_KHR: Self = Self::BUFFER_MEMORY_BARRIER_2;
    pub const IMAGE_MEMORY_BARRIER_2_KHR: Self = Self::IMAGE_MEMORY_BARRIER_2;
    pub const DEPENDENCY_INFO_KHR: Self = Self::DEPENDENCY_INFO;
    pub const SUBMIT_INFO_2_KHR: Self = Self::SUBMIT_INFO_2;
    pub const SEMAPHORE_SUBMIT_INFO_KHR: Self = Self::SEMAPHORE_SUBMIT_INFO;
    pub const COMMAND_BUFFER_SUBMIT_INFO_KHR: Self = Self::COMMAND_BUFFER_SUBMIT_INFO;
    pub const PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
}
#[doc = "Generated from 'VK_KHR_synchronization2'"]
impl SubmitFlags {
    pub const PROTECTED_KHR: Self = Self::PROTECTED;
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl AccelerationStructureCreateFlagsKHR {
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY_EXT: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl AccessFlags2 {
    pub const DESCRIPTOR_BUFFER_READ_EXT: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl BufferCreateFlags {
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY_EXT: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl BufferUsageFlags {
    pub const SAMPLER_DESCRIPTOR_BUFFER_EXT: Self = Self(0b10_0000_0000_0000_0000_0000);
    pub const RESOURCE_DESCRIPTOR_BUFFER_EXT: Self = Self(0b100_0000_0000_0000_0000_0000);
    pub const PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl DescriptorSetLayoutCreateFlags {
    pub const DESCRIPTOR_BUFFER_EXT: Self = Self(0b1_0000);
    pub const EMBEDDED_IMMUTABLE_SAMPLERS_EXT: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl ImageCreateFlags {
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY_EXT: Self = Self::DESCRIPTOR_HEAP_CAPTURE_REPLAY_EXT;
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl ImageViewCreateFlags {
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY_EXT: Self = Self(0b100);
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl PipelineCreateFlags {
    pub const DESCRIPTOR_BUFFER_EXT: Self = Self(0b10_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl SamplerCreateFlags {
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY_EXT: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_EXT_descriptor_buffer'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT: Self = Self(1_000_316_000);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_DENSITY_MAP_PROPERTIES_EXT: Self =
        Self(1_000_316_001);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT: Self = Self(1_000_316_002);
    pub const DESCRIPTOR_ADDRESS_INFO_EXT: Self = Self(1_000_316_003);
    pub const DESCRIPTOR_GET_INFO_EXT: Self = Self(1_000_316_004);
    pub const BUFFER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT: Self = Self(1_000_316_005);
    pub const IMAGE_CAPTURE_DESCRIPTOR_DATA_INFO_EXT: Self = Self(1_000_316_006);
    pub const IMAGE_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO_EXT: Self = Self(1_000_316_007);
    pub const SAMPLER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT: Self = Self(1_000_316_008);
    pub const OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO_EXT: Self = Self(1_000_316_010);
    pub const DESCRIPTOR_BUFFER_BINDING_INFO_EXT: Self = Self(1_000_316_011);
    pub const DESCRIPTOR_BUFFER_BINDING_PUSH_DESCRIPTOR_BUFFER_HANDLE_EXT: Self =
        Self(1_000_316_012);
    pub const ACCELERATION_STRUCTURE_CAPTURE_DESCRIPTOR_DATA_INFO_EXT: Self = Self(1_000_316_009);
}
#[doc = "Generated from 'VK_KHR_device_address_commands'"]
impl AddressCommandFlagsKHR {
    pub const TRANSFORM_FEEDBACK_BUFFER_USAGE: Self = Self(0b1_0000);
    pub const UNKNOWN_TRANSFORM_FEEDBACK_BUFFER_USAGE: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_KHR_device_address_commands'"]
impl StructureType {
    pub const DEVICE_MEMORY_COPY_KHR: Self = Self(1_000_318_000);
    pub const COPY_DEVICE_MEMORY_INFO_KHR: Self = Self(1_000_318_001);
    pub const DEVICE_MEMORY_IMAGE_COPY_KHR: Self = Self(1_000_318_002);
    pub const COPY_DEVICE_MEMORY_IMAGE_INFO_KHR: Self = Self(1_000_318_003);
    pub const MEMORY_RANGE_BARRIERS_INFO_KHR: Self = Self(1_000_318_004);
    pub const MEMORY_RANGE_BARRIER_KHR: Self = Self(1_000_318_005);
    pub const PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR: Self = Self(1_000_318_006);
    pub const BIND_INDEX_BUFFER_3_INFO_KHR: Self = Self(1_000_318_007);
    pub const BIND_VERTEX_BUFFER_3_INFO_KHR: Self = Self(1_000_318_008);
    pub const DRAW_INDIRECT_2_INFO_KHR: Self = Self(1_000_318_009);
    pub const DRAW_INDIRECT_COUNT_2_INFO_KHR: Self = Self(1_000_318_010);
    pub const DISPATCH_INDIRECT_2_INFO_KHR: Self = Self(1_000_318_011);
    pub const CONDITIONAL_RENDERING_BEGIN_INFO_2_EXT: Self = Self(1_000_318_012);
    pub const BIND_TRANSFORM_FEEDBACK_BUFFER_2_INFO_EXT: Self = Self(1_000_318_013);
    pub const MEMORY_MARKER_INFO_AMD: Self = Self(1_000_318_014);
    pub const ACCELERATION_STRUCTURE_CREATE_INFO_2_KHR: Self = Self(1_000_318_015);
}
#[doc = "Generated from 'VK_EXT_graphics_pipeline_library'"]
impl PipelineCreateFlags {
    pub const RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT: Self = Self(0b1000_0000_0000_0000_0000_0000);
    pub const LINK_TIME_OPTIMIZATION_EXT: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_EXT_graphics_pipeline_library'"]
impl PipelineLayoutCreateFlags {
    pub const INDEPENDENT_SETS_EXT: Self = Self(0b10);
}
#[doc = "Generated from 'VK_EXT_graphics_pipeline_library'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT: Self = Self(1_000_320_000);
    pub const PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT: Self = Self(1_000_320_001);
    pub const GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT: Self = Self(1_000_320_002);
}
#[doc = "Generated from 'VK_AMD_shader_early_and_late_fragment_tests'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_FEATURES_AMD: Self =
        Self(1_000_321_000);
}
#[doc = "Generated from 'VK_KHR_fragment_shader_barycentric'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR: Self = Self(1_000_203_000);
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_PROPERTIES_KHR: Self =
        Self(1_000_322_000);
}
#[doc = "Generated from 'VK_KHR_shader_subgroup_uniform_control_flow'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR: Self =
        Self(1_000_323_000);
}
#[doc = "Generated from 'VK_KHR_zero_initialize_workgroup_memory'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES;
}
#[doc = "Generated from 'VK_NV_fragment_shading_rate_enums'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_PROPERTIES_NV: Self = Self(1_000_326_000);
    pub const PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_FEATURES_NV: Self = Self(1_000_326_001);
    pub const PIPELINE_FRAGMENT_SHADING_RATE_ENUM_STATE_CREATE_INFO_NV: Self = Self(1_000_326_002);
}
#[doc = "Generated from 'VK_NV_ray_tracing_motion_blur'"]
impl AccelerationStructureCreateFlagsKHR {
    pub const MOTION_NV: Self = Self(0b100);
}
#[doc = "Generated from 'VK_NV_ray_tracing_motion_blur'"]
impl BuildAccelerationStructureFlagsKHR {
    pub const MOTION_NV: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_NV_ray_tracing_motion_blur'"]
impl PipelineCreateFlags {
    pub const RAY_TRACING_ALLOW_MOTION_NV: Self = Self(0b1_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_ray_tracing_motion_blur'"]
impl StructureType {
    pub const ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV: Self = Self(1_000_327_000);
    pub const PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV: Self = Self(1_000_327_001);
    pub const ACCELERATION_STRUCTURE_MOTION_INFO_NV: Self = Self(1_000_327_002);
}
#[doc = "Generated from 'VK_EXT_mesh_shader'"]
impl IndirectCommandsTokenTypeEXT {
    pub const DRAW_MESH_TASKS: Self = Self(1_000_328_000);
    pub const DRAW_MESH_TASKS_COUNT: Self = Self(1_000_328_001);
}
#[doc = "Generated from 'VK_EXT_mesh_shader'"]
impl IndirectCommandsTokenTypeNV {
    pub const DRAW_MESH_TASKS: Self = Self(1_000_328_000);
}
#[doc = "Generated from 'VK_EXT_mesh_shader'"]
impl PipelineStageFlags {
    pub const TASK_SHADER_EXT: Self = Self(0b1000_0000_0000_0000_0000);
    pub const MESH_SHADER_EXT: Self = Self(0b1_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_mesh_shader'"]
impl QueryPipelineStatisticFlags {
    pub const TASK_SHADER_INVOCATIONS_EXT: Self = Self(0b1000_0000_0000);
    pub const MESH_SHADER_INVOCATIONS_EXT: Self = Self(0b1_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_mesh_shader'"]
impl QueryType {
    pub const MESH_PRIMITIVES_GENERATED_EXT: Self = Self(1_000_328_000);
}
#[doc = "Generated from 'VK_EXT_mesh_shader'"]
impl ShaderStageFlags {
    pub const TASK_EXT: Self = Self(0b100_0000);
    pub const MESH_EXT: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_EXT_mesh_shader'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT: Self = Self(1_000_328_000);
    pub const PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT: Self = Self(1_000_328_001);
}
#[doc = "Generated from 'VK_EXT_ycbcr_2plane_444_formats'"]
impl Format {
    pub const G8_B8R8_2PLANE_444_UNORM_EXT: Self = Self::G8_B8R8_2PLANE_444_UNORM;
    pub const G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT: Self =
        Self::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16;
    pub const G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT: Self =
        Self::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16;
    pub const G16_B16R16_2PLANE_444_UNORM_EXT: Self = Self::G16_B16R16_2PLANE_444_UNORM;
}
#[doc = "Generated from 'VK_EXT_ycbcr_2plane_444_formats'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_YCBCR_2_PLANE_444_FORMATS_FEATURES_EXT: Self = Self(1_000_330_000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map2'"]
impl ImageViewCreateFlags {
    pub const FRAGMENT_DENSITY_MAP_DEFERRED_EXT: Self = Self(0b10);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT: Self = Self(1_000_332_000);
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_PROPERTIES_EXT: Self = Self(1_000_332_001);
}
#[doc = "Generated from 'VK_QCOM_rotated_copy_commands'"]
impl StructureType {
    pub const COPY_COMMAND_TRANSFORM_INFO_QCOM: Self = Self(1_000_333_000);
}
#[doc = "Generated from 'VK_EXT_image_robustness'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES;
}
#[doc = "Generated from 'VK_KHR_workgroup_memory_explicit_layout'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR: Self =
        Self(1_000_336_000);
}
#[doc = "Generated from 'VK_KHR_copy_commands2'"]
impl StructureType {
    pub const COPY_BUFFER_INFO_2_KHR: Self = Self::COPY_BUFFER_INFO_2;
    pub const COPY_IMAGE_INFO_2_KHR: Self = Self::COPY_IMAGE_INFO_2;
    pub const COPY_BUFFER_TO_IMAGE_INFO_2_KHR: Self = Self::COPY_BUFFER_TO_IMAGE_INFO_2;
    pub const COPY_IMAGE_TO_BUFFER_INFO_2_KHR: Self = Self::COPY_IMAGE_TO_BUFFER_INFO_2;
    pub const BLIT_IMAGE_INFO_2_KHR: Self = Self::BLIT_IMAGE_INFO_2;
    pub const RESOLVE_IMAGE_INFO_2_KHR: Self = Self::RESOLVE_IMAGE_INFO_2;
    pub const BUFFER_COPY_2_KHR: Self = Self::BUFFER_COPY_2;
    pub const IMAGE_COPY_2_KHR: Self = Self::IMAGE_COPY_2;
    pub const IMAGE_BLIT_2_KHR: Self = Self::IMAGE_BLIT_2;
    pub const BUFFER_IMAGE_COPY_2_KHR: Self = Self::BUFFER_IMAGE_COPY_2;
    pub const IMAGE_RESOLVE_2_KHR: Self = Self::IMAGE_RESOLVE_2;
}
#[doc = "Generated from 'VK_EXT_image_compression_control'"]
impl Result {
    pub const ERROR_COMPRESSION_EXHAUSTED_EXT: Self = Self(-1_000_338_000);
}
#[doc = "Generated from 'VK_EXT_image_compression_control'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT: Self = Self(1_000_338_000);
    pub const IMAGE_COMPRESSION_CONTROL_EXT: Self = Self(1_000_338_001);
    pub const SUBRESOURCE_LAYOUT_2_EXT: Self = Self::SUBRESOURCE_LAYOUT_2;
    pub const IMAGE_SUBRESOURCE_2_EXT: Self = Self::IMAGE_SUBRESOURCE_2;
    pub const IMAGE_COMPRESSION_PROPERTIES_EXT: Self = Self(1_000_338_004);
}
#[doc = "Generated from 'VK_EXT_attachment_feedback_loop_layout'"]
impl DependencyFlags {
    #[doc = "Dependency may be a feedback loop"]
    pub const FEEDBACK_LOOP_EXT: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_EXT_attachment_feedback_loop_layout'"]
impl ImageLayout {
    pub const ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT: Self = Self(1_000_339_000);
}
#[doc = "Generated from 'VK_EXT_attachment_feedback_loop_layout'"]
impl ImageUsageFlags {
    pub const ATTACHMENT_FEEDBACK_LOOP_EXT: Self = Self(0b1000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_attachment_feedback_loop_layout'"]
impl PipelineCreateFlags {
    pub const COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
    pub const DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_attachment_feedback_loop_layout'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT: Self =
        Self(1_000_339_000);
}
#[doc = "Generated from 'VK_EXT_4444_formats'"]
impl Format {
    pub const A4R4G4B4_UNORM_PACK16_EXT: Self = Self::A4R4G4B4_UNORM_PACK16;
    pub const A4B4G4R4_UNORM_PACK16_EXT: Self = Self::A4B4G4R4_UNORM_PACK16;
}
#[doc = "Generated from 'VK_EXT_4444_formats'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT: Self = Self(1_000_340_000);
}
#[doc = "Generated from 'VK_EXT_device_fault'"]
impl DeviceFaultAddressTypeKHR {
    pub const NONE_EXT: Self = Self::NONE;
    pub const READ_INVALID_EXT: Self = Self::READ_INVALID;
    pub const WRITE_INVALID_EXT: Self = Self::WRITE_INVALID;
    pub const EXECUTE_INVALID_EXT: Self = Self::EXECUTE_INVALID;
    pub const INSTRUCTION_POINTER_UNKNOWN_EXT: Self = Self::INSTRUCTION_POINTER_UNKNOWN;
    pub const INSTRUCTION_POINTER_INVALID_EXT: Self = Self::INSTRUCTION_POINTER_INVALID;
    pub const INSTRUCTION_POINTER_FAULT_EXT: Self = Self::INSTRUCTION_POINTER_FAULT;
}
#[doc = "Generated from 'VK_EXT_device_fault'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FAULT_FEATURES_EXT: Self = Self(1_000_341_000);
    pub const DEVICE_FAULT_COUNTS_EXT: Self = Self(1_000_341_001);
    pub const DEVICE_FAULT_INFO_EXT: Self = Self(1_000_341_002);
}
#[doc = "Generated from 'VK_ARM_rasterization_order_attachment_access'"]
impl PipelineColorBlendStateCreateFlags {
    pub const RASTERIZATION_ORDER_ATTACHMENT_ACCESS_ARM: Self =
        Self::RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXT;
}
#[doc = "Generated from 'VK_ARM_rasterization_order_attachment_access'"]
impl PipelineDepthStencilStateCreateFlags {
    pub const RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_ARM: Self =
        Self::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_EXT;
    pub const RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_ARM: Self =
        Self::RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_EXT;
}
#[doc = "Generated from 'VK_ARM_rasterization_order_attachment_access'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM: Self =
        Self::PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_EXT;
}
#[doc = "Generated from 'VK_ARM_rasterization_order_attachment_access'"]
impl SubpassDescriptionFlags {
    pub const RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_ARM: Self =
        Self::RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_EXT;
    pub const RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_ARM: Self =
        Self::RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_EXT;
    pub const RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_ARM: Self =
        Self::RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_EXT;
}
#[doc = "Generated from 'VK_EXT_rgba10x6_formats'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RGBA10X6_FORMATS_FEATURES_EXT: Self = Self(1_000_344_000);
}
#[doc = "Generated from 'VK_EXT_directfb_surface'"]
impl StructureType {
    pub const DIRECTFB_SURFACE_CREATE_INFO_EXT: Self = Self(1_000_346_000);
}
#[doc = "Generated from 'VK_VALVE_mutable_descriptor_type'"]
impl DescriptorPoolCreateFlags {
    pub const HOST_ONLY_VALVE: Self = Self::HOST_ONLY_EXT;
}
#[doc = "Generated from 'VK_VALVE_mutable_descriptor_type'"]
impl DescriptorSetLayoutCreateFlags {
    pub const HOST_ONLY_POOL_VALVE: Self = Self::HOST_ONLY_POOL_EXT;
}
#[doc = "Generated from 'VK_VALVE_mutable_descriptor_type'"]
impl DescriptorType {
    pub const MUTABLE_VALVE: Self = Self::MUTABLE_EXT;
}
#[doc = "Generated from 'VK_VALVE_mutable_descriptor_type'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_VALVE: Self =
        Self::PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
    pub const MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_VALVE: Self =
        Self::MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT;
}
#[doc = "Generated from 'VK_EXT_vertex_input_dynamic_state'"]
impl DynamicState {
    pub const VERTEX_INPUT_EXT: Self = Self(1_000_352_000);
}
#[doc = "Generated from 'VK_EXT_vertex_input_dynamic_state'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT: Self = Self(1_000_352_000);
    pub const VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT: Self = Self(1_000_352_001);
    pub const VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT: Self = Self(1_000_352_002);
}
#[doc = "Generated from 'VK_EXT_physical_device_drm'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DRM_PROPERTIES_EXT: Self = Self(1_000_353_000);
}
#[doc = "Generated from 'VK_EXT_device_address_binding_report'"]
impl DebugUtilsMessageTypeFlagsEXT {
    pub const DEVICE_ADDRESS_BINDING: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_EXT_device_address_binding_report'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT: Self = Self(1_000_354_000);
    pub const DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT: Self = Self(1_000_354_001);
}
#[doc = "Generated from 'VK_EXT_depth_clip_control'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT: Self = Self(1_000_355_000);
    pub const PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT: Self = Self(1_000_355_001);
}
#[doc = "Generated from 'VK_EXT_primitive_topology_list_restart'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT: Self =
        Self(1_000_356_000);
}
#[doc = "Generated from 'VK_KHR_format_feature_flags2'"]
impl FormatFeatureFlags2 {
    pub const SAMPLED_IMAGE_KHR: Self = Self::SAMPLED_IMAGE;
    pub const STORAGE_IMAGE_KHR: Self = Self::STORAGE_IMAGE;
    pub const STORAGE_IMAGE_ATOMIC_KHR: Self = Self::STORAGE_IMAGE_ATOMIC;
    pub const UNIFORM_TEXEL_BUFFER_KHR: Self = Self::UNIFORM_TEXEL_BUFFER;
    pub const STORAGE_TEXEL_BUFFER_KHR: Self = Self::STORAGE_TEXEL_BUFFER;
    pub const STORAGE_TEXEL_BUFFER_ATOMIC_KHR: Self = Self::STORAGE_TEXEL_BUFFER_ATOMIC;
    pub const VERTEX_BUFFER_KHR: Self = Self::VERTEX_BUFFER;
    pub const COLOR_ATTACHMENT_KHR: Self = Self::COLOR_ATTACHMENT;
    pub const COLOR_ATTACHMENT_BLEND_KHR: Self = Self::COLOR_ATTACHMENT_BLEND;
    pub const DEPTH_STENCIL_ATTACHMENT_KHR: Self = Self::DEPTH_STENCIL_ATTACHMENT;
    pub const BLIT_SRC_KHR: Self = Self::BLIT_SRC;
    pub const BLIT_DST_KHR: Self = Self::BLIT_DST;
    pub const SAMPLED_IMAGE_FILTER_LINEAR_KHR: Self = Self::SAMPLED_IMAGE_FILTER_LINEAR;
    pub const TRANSFER_SRC_KHR: Self = Self::TRANSFER_SRC;
    pub const TRANSFER_DST_KHR: Self = Self::TRANSFER_DST;
    pub const MIDPOINT_CHROMA_SAMPLES_KHR: Self = Self::MIDPOINT_CHROMA_SAMPLES;
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_KHR: Self =
        Self::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER;
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_KHR: Self =
        Self::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER;
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_KHR: Self =
        Self::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT;
    pub const SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_KHR: Self =
        Self::SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE;
    pub const DISJOINT_KHR: Self = Self::DISJOINT;
    pub const COSITED_CHROMA_SAMPLES_KHR: Self = Self::COSITED_CHROMA_SAMPLES;
    pub const STORAGE_READ_WITHOUT_FORMAT_KHR: Self = Self::STORAGE_READ_WITHOUT_FORMAT;
    pub const STORAGE_WRITE_WITHOUT_FORMAT_KHR: Self = Self::STORAGE_WRITE_WITHOUT_FORMAT;
    pub const SAMPLED_IMAGE_DEPTH_COMPARISON_KHR: Self = Self::SAMPLED_IMAGE_DEPTH_COMPARISON;
    pub const SAMPLED_IMAGE_FILTER_MINMAX_KHR: Self = Self::SAMPLED_IMAGE_FILTER_MINMAX;
    pub const SAMPLED_IMAGE_FILTER_CUBIC_EXT: Self = Self::SAMPLED_IMAGE_FILTER_CUBIC;
}
#[doc = "Generated from 'VK_KHR_format_feature_flags2'"]
impl StructureType {
    pub const FORMAT_PROPERTIES_3_KHR: Self = Self::FORMAT_PROPERTIES_3;
}
#[doc = "Generated from 'VK_EXT_present_mode_fifo_latest_ready'"]
impl PresentModeKHR {
    pub const FIFO_LATEST_READY_EXT: Self = Self::FIFO_LATEST_READY;
}
#[doc = "Generated from 'VK_EXT_present_mode_fifo_latest_ready'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR;
}
#[doc = "Generated from 'VK_FUCHSIA_external_memory'"]
impl ExternalMemoryHandleTypeFlags {
    pub const ZIRCON_VMO_FUCHSIA: Self = Self(0b1000_0000_0000);
}
#[doc = "Generated from 'VK_FUCHSIA_external_memory'"]
impl StructureType {
    pub const IMPORT_MEMORY_ZIRCON_HANDLE_INFO_FUCHSIA: Self = Self(1_000_364_000);
    pub const MEMORY_ZIRCON_HANDLE_PROPERTIES_FUCHSIA: Self = Self(1_000_364_001);
    pub const MEMORY_GET_ZIRCON_HANDLE_INFO_FUCHSIA: Self = Self(1_000_364_002);
}
#[doc = "Generated from 'VK_FUCHSIA_external_semaphore'"]
impl ExternalSemaphoreHandleTypeFlags {
    pub const ZIRCON_EVENT_FUCHSIA: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_FUCHSIA_external_semaphore'"]
impl StructureType {
    pub const IMPORT_SEMAPHORE_ZIRCON_HANDLE_INFO_FUCHSIA: Self = Self(1_000_365_000);
    pub const SEMAPHORE_GET_ZIRCON_HANDLE_INFO_FUCHSIA: Self = Self(1_000_365_001);
}
#[doc = "Generated from 'VK_FUCHSIA_buffer_collection'"]
impl DebugReportObjectTypeEXT {
    pub const BUFFER_COLLECTION_FUCHSIA: Self = Self(1_000_366_000);
}
#[doc = "Generated from 'VK_FUCHSIA_buffer_collection'"]
impl ObjectType {
    #[doc = "VkBufferCollectionFUCHSIA"]
    pub const BUFFER_COLLECTION_FUCHSIA: Self = Self(1_000_366_000);
}
#[doc = "Generated from 'VK_FUCHSIA_buffer_collection'"]
impl StructureType {
    pub const BUFFER_COLLECTION_CREATE_INFO_FUCHSIA: Self = Self(1_000_366_000);
    pub const IMPORT_MEMORY_BUFFER_COLLECTION_FUCHSIA: Self = Self(1_000_366_001);
    pub const BUFFER_COLLECTION_IMAGE_CREATE_INFO_FUCHSIA: Self = Self(1_000_366_002);
    pub const BUFFER_COLLECTION_PROPERTIES_FUCHSIA: Self = Self(1_000_366_003);
    pub const BUFFER_CONSTRAINTS_INFO_FUCHSIA: Self = Self(1_000_366_004);
    pub const BUFFER_COLLECTION_BUFFER_CREATE_INFO_FUCHSIA: Self = Self(1_000_366_005);
    pub const IMAGE_CONSTRAINTS_INFO_FUCHSIA: Self = Self(1_000_366_006);
    pub const IMAGE_FORMAT_CONSTRAINTS_INFO_FUCHSIA: Self = Self(1_000_366_007);
    pub const SYSMEM_COLOR_SPACE_FUCHSIA: Self = Self(1_000_366_008);
    pub const BUFFER_COLLECTION_CONSTRAINTS_INFO_FUCHSIA: Self = Self(1_000_366_009);
}
#[doc = "Generated from 'VK_HUAWEI_subpass_shading'"]
impl PipelineBindPoint {
    pub const SUBPASS_SHADING_HUAWEI: Self = Self(1_000_369_003);
}
#[doc = "Generated from 'VK_HUAWEI_subpass_shading'"]
impl PipelineStageFlags2 {
    pub const SUBPASS_SHADER_HUAWEI: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_HUAWEI_subpass_shading'"]
impl ShaderStageFlags {
    pub const SUBPASS_SHADING_HUAWEI: Self = Self(0b100_0000_0000_0000);
}
#[doc = "Generated from 'VK_HUAWEI_subpass_shading'"]
impl StructureType {
    pub const SUBPASS_SHADING_PIPELINE_CREATE_INFO_HUAWEI: Self = Self(1_000_369_000);
    pub const PHYSICAL_DEVICE_SUBPASS_SHADING_FEATURES_HUAWEI: Self = Self(1_000_369_001);
    pub const PHYSICAL_DEVICE_SUBPASS_SHADING_PROPERTIES_HUAWEI: Self = Self(1_000_369_002);
}
#[doc = "Generated from 'VK_HUAWEI_invocation_mask'"]
impl AccessFlags2 {
    pub const INVOCATION_MASK_READ_HUAWEI: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_HUAWEI_invocation_mask'"]
impl ImageUsageFlags {
    pub const INVOCATION_MASK_HUAWEI: Self = Self(0b100_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_HUAWEI_invocation_mask'"]
impl PipelineStageFlags2 {
    pub const INVOCATION_MASK_HUAWEI: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_HUAWEI_invocation_mask'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_INVOCATION_MASK_FEATURES_HUAWEI: Self = Self(1_000_370_000);
}
#[doc = "Generated from 'VK_NV_external_memory_rdma'"]
impl ExternalMemoryHandleTypeFlags {
    pub const RDMA_ADDRESS_NV: Self = Self(0b1_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_external_memory_rdma'"]
impl MemoryPropertyFlags {
    pub const RDMA_CAPABLE_NV: Self = Self(0b1_0000_0000);
}
#[doc = "Generated from 'VK_NV_external_memory_rdma'"]
impl StructureType {
    pub const MEMORY_GET_REMOTE_ADDRESS_INFO_NV: Self = Self(1_000_371_000);
    pub const PHYSICAL_DEVICE_EXTERNAL_MEMORY_RDMA_FEATURES_NV: Self = Self(1_000_371_001);
}
#[doc = "Generated from 'VK_EXT_pipeline_properties'"]
impl StructureType {
    pub const PIPELINE_PROPERTIES_IDENTIFIER_EXT: Self = Self(1_000_372_000);
    pub const PHYSICAL_DEVICE_PIPELINE_PROPERTIES_FEATURES_EXT: Self = Self(1_000_372_001);
    pub const PIPELINE_INFO_EXT: Self = Self::PIPELINE_INFO_KHR;
}
#[doc = "Generated from 'VK_EXT_frame_boundary'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAME_BOUNDARY_FEATURES_EXT: Self = Self(1_000_375_000);
    pub const FRAME_BOUNDARY_EXT: Self = Self(1_000_375_001);
}
#[doc = "Generated from 'VK_EXT_multisampled_render_to_single_sampled'"]
impl ImageCreateFlags {
    pub const MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXT: Self = Self(0b100_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_multisampled_render_to_single_sampled'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES_EXT: Self =
        Self(1_000_376_000);
    pub const SUBPASS_RESOLVE_PERFORMANCE_QUERY_EXT: Self = Self(1_000_376_001);
    pub const MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO_EXT: Self = Self(1_000_376_002);
}
#[doc = "Generated from 'VK_EXT_extended_dynamic_state2'"]
impl DynamicState {
    #[doc = "Not promoted to 1.3"]
    pub const PATCH_CONTROL_POINTS_EXT: Self = Self(1_000_377_000);
    pub const RASTERIZER_DISCARD_ENABLE_EXT: Self = Self::RASTERIZER_DISCARD_ENABLE;
    pub const DEPTH_BIAS_ENABLE_EXT: Self = Self::DEPTH_BIAS_ENABLE;
    #[doc = "Not promoted to 1.3"]
    pub const LOGIC_OP_EXT: Self = Self(1_000_377_003);
    pub const PRIMITIVE_RESTART_ENABLE_EXT: Self = Self::PRIMITIVE_RESTART_ENABLE;
}
#[doc = "Generated from 'VK_EXT_extended_dynamic_state2'"]
impl StructureType {
    #[doc = "Not promoted to 1.3"]
    pub const PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT: Self = Self(1_000_377_000);
}
#[doc = "Generated from 'VK_QNX_screen_surface'"]
impl StructureType {
    pub const SCREEN_SURFACE_CREATE_INFO_QNX: Self = Self(1_000_378_000);
}
#[doc = "Generated from 'VK_EXT_color_write_enable'"]
impl DynamicState {
    pub const COLOR_WRITE_ENABLE_EXT: Self = Self(1_000_381_000);
}
#[doc = "Generated from 'VK_EXT_color_write_enable'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT: Self = Self(1_000_381_000);
    pub const PIPELINE_COLOR_WRITE_CREATE_INFO_EXT: Self = Self(1_000_381_001);
}
#[doc = "Generated from 'VK_EXT_primitives_generated_query'"]
impl QueryType {
    pub const PRIMITIVES_GENERATED_EXT: Self = Self(1_000_382_000);
}
#[doc = "Generated from 'VK_EXT_primitives_generated_query'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT: Self = Self(1_000_382_000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_maintenance1'"]
impl AccessFlags2 {
    pub const SHADER_BINDING_TABLE_READ_KHR: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_maintenance1'"]
impl IndirectCommandsTokenTypeEXT {
    pub const TRACE_RAYS2: Self = Self(1_000_386_004);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_maintenance1'"]
impl PipelineStageFlags2 {
    pub const ACCELERATION_STRUCTURE_COPY_KHR: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_maintenance1'"]
impl QueryType {
    pub const ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR: Self =
        Self(1_000_386_000);
    pub const ACCELERATION_STRUCTURE_SIZE_KHR: Self = Self(1_000_386_001);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_maintenance1'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR: Self = Self(1_000_386_000);
}
#[doc = "Generated from 'VK_KHR_shader_untyped_pointers'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR: Self = Self(1_000_387_000);
}
#[doc = "Generated from 'VK_EXT_global_priority_query'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES;
    pub const QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_EXT: Self =
        Self::QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES;
}
#[doc = "Generated from 'VK_VALVE_video_encode_rgb_conversion'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_RGB_CONVERSION_FEATURES_VALVE: Self =
        Self(1_000_390_000);
    pub const VIDEO_ENCODE_RGB_CONVERSION_CAPABILITIES_VALVE: Self = Self(1_000_390_001);
    pub const VIDEO_ENCODE_PROFILE_RGB_CONVERSION_INFO_VALVE: Self = Self(1_000_390_002);
    pub const VIDEO_ENCODE_SESSION_RGB_CONVERSION_CREATE_INFO_VALVE: Self = Self(1_000_390_003);
}
#[doc = "Generated from 'VK_EXT_image_view_min_lod'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT: Self = Self(1_000_391_000);
    pub const IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT: Self = Self(1_000_391_001);
}
#[doc = "Generated from 'VK_EXT_multi_draw'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT: Self = Self(1_000_392_000);
    pub const PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES_EXT: Self = Self(1_000_392_001);
}
#[doc = "Generated from 'VK_EXT_image_2d_view_of_3d'"]
impl ImageCreateFlags {
    #[doc = "Image is created with a layout where individual slices are capable of being used as 2D images"]
    pub const TYPE_2D_VIEW_COMPATIBLE_EXT: Self = Self(0b10_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_image_2d_view_of_3d'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT: Self = Self(1_000_393_000);
}
#[doc = "Generated from 'VK_KHR_portability_enumeration'"]
impl InstanceCreateFlags {
    pub const ENUMERATE_PORTABILITY_KHR: Self = Self(0b1);
}
#[doc = "Generated from 'VK_EXT_shader_tile_image'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_TILE_IMAGE_FEATURES_EXT: Self = Self(1_000_395_000);
    pub const PHYSICAL_DEVICE_SHADER_TILE_IMAGE_PROPERTIES_EXT: Self = Self(1_000_395_001);
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl AccessFlags2 {
    pub const MICROMAP_READ_EXT: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const MICROMAP_WRITE_EXT: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl BufferUsageFlags {
    pub const MICROMAP_BUILD_INPUT_READ_ONLY_EXT: Self = Self(0b1000_0000_0000_0000_0000_0000);
    pub const MICROMAP_STORAGE_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl BufferUsageFlags2 {
    pub const MICROMAP_BUILD_INPUT_READ_ONLY_EXT: Self = Self(0b1000_0000_0000_0000_0000_0000);
    pub const MICROMAP_STORAGE_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl BuildAccelerationStructureFlagsKHR {
    pub const ALLOW_OPACITY_MICROMAP_UPDATE_EXT: Self = Self::ALLOW_OPACITY_MICROMAP_UPDATE;
    pub const ALLOW_DISABLE_OPACITY_MICROMAPS_EXT: Self = Self::ALLOW_DISABLE_OPACITY_MICROMAPS;
    pub const ALLOW_OPACITY_MICROMAP_DATA_UPDATE_EXT: Self = Self(0b1_0000_0000);
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl GeometryInstanceFlagsKHR {
    pub const FORCE_OPACITY_MICROMAP_2_STATE_EXT: Self = Self::FORCE_OPACITY_MICROMAP_2_STATE;
    pub const DISABLE_OPACITY_MICROMAPS_EXT: Self = Self::DISABLE_OPACITY_MICROMAPS;
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl ObjectType {
    pub const MICROMAP_EXT: Self = Self(1_000_396_000);
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl OpacityMicromapFormatKHR {
    pub const TYPE_2_STATE_EXT: Self = Self::TYPE_2_STATE;
    pub const TYPE_4_STATE_EXT: Self = Self::TYPE_4_STATE;
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl OpacityMicromapSpecialIndexKHR {
    pub const FULLY_TRANSPARENT_EXT: Self = Self::FULLY_TRANSPARENT;
    pub const FULLY_OPAQUE_EXT: Self = Self::FULLY_OPAQUE;
    pub const FULLY_UNKNOWN_TRANSPARENT_EXT: Self = Self::FULLY_UNKNOWN_TRANSPARENT;
    pub const FULLY_UNKNOWN_OPAQUE_EXT: Self = Self::FULLY_UNKNOWN_OPAQUE;
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl PipelineCreateFlags {
    pub const RAY_TRACING_OPACITY_MICROMAP_EXT: Self = Self::RAY_TRACING_OPACITY_MICROMAP_KHR;
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl PipelineCreateFlags2 {
    pub const RAY_TRACING_OPACITY_MICROMAP_EXT: Self = Self::RAY_TRACING_OPACITY_MICROMAP_KHR;
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl PipelineStageFlags2 {
    pub const MICROMAP_BUILD_EXT: Self = Self(0b100_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl QueryType {
    pub const MICROMAP_SERIALIZATION_SIZE_EXT: Self = Self(1_000_396_000);
    pub const MICROMAP_COMPACTED_SIZE_EXT: Self = Self(1_000_396_001);
}
#[doc = "Generated from 'VK_EXT_opacity_micromap'"]
impl StructureType {
    pub const MICROMAP_BUILD_INFO_EXT: Self = Self(1_000_396_000);
    pub const MICROMAP_VERSION_INFO_EXT: Self = Self(1_000_396_001);
    pub const COPY_MICROMAP_INFO_EXT: Self = Self(1_000_396_002);
    pub const COPY_MICROMAP_TO_MEMORY_INFO_EXT: Self = Self(1_000_396_003);
    pub const COPY_MEMORY_TO_MICROMAP_INFO_EXT: Self = Self(1_000_396_004);
    pub const PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT: Self = Self(1_000_396_005);
    pub const PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT: Self = Self(1_000_396_006);
    pub const MICROMAP_CREATE_INFO_EXT: Self = Self(1_000_396_007);
    pub const MICROMAP_BUILD_SIZES_INFO_EXT: Self = Self(1_000_396_008);
    pub const ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT: Self = Self(1_000_396_009);
}
#[doc = "Generated from 'VK_NV_displacement_micromap'"]
#[cfg(feature = "provisional")]
impl BuildAccelerationStructureFlagsKHR {
    pub const ALLOW_DISPLACEMENT_MICROMAP_UPDATE_NV: Self = Self(0b10_0000_0000);
}
#[doc = "Generated from 'VK_NV_displacement_micromap'"]
#[cfg(feature = "provisional")]
impl MicromapTypeEXT {
    pub const DISPLACEMENT_MICROMAP_NV: Self = Self(1_000_397_000);
}
#[doc = "Generated from 'VK_NV_displacement_micromap'"]
#[cfg(feature = "provisional")]
impl PipelineCreateFlags {
    pub const RAY_TRACING_DISPLACEMENT_MICROMAP_NV: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_displacement_micromap'"]
#[cfg(feature = "provisional")]
impl StructureType {
    pub const PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV: Self = Self(1_000_397_000);
    pub const PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_PROPERTIES_NV: Self = Self(1_000_397_001);
    pub const ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP_NV: Self = Self(1_000_397_002);
}
#[doc = "Generated from 'VK_EXT_load_store_op_none'"]
impl AttachmentLoadOp {
    pub const NONE_EXT: Self = Self::NONE;
}
#[doc = "Generated from 'VK_EXT_load_store_op_none'"]
impl AttachmentStoreOp {
    pub const NONE_EXT: Self = Self::NONE;
}
#[doc = "Generated from 'VK_HUAWEI_cluster_culling_shader'"]
impl PipelineStageFlags2 {
    pub const CLUSTER_CULLING_SHADER_HUAWEI: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_HUAWEI_cluster_culling_shader'"]
impl QueryPipelineStatisticFlags {
    pub const CLUSTER_CULLING_SHADER_INVOCATIONS_HUAWEI: Self = Self(0b10_0000_0000_0000);
}
#[doc = "Generated from 'VK_HUAWEI_cluster_culling_shader'"]
impl ShaderStageFlags {
    pub const CLUSTER_CULLING_HUAWEI: Self = Self(0b1000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_HUAWEI_cluster_culling_shader'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_FEATURES_HUAWEI: Self = Self(1_000_404_000);
    pub const PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_PROPERTIES_HUAWEI: Self = Self(1_000_404_001);
    pub const PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_VRS_FEATURES_HUAWEI: Self =
        Self(1_000_404_002);
}
#[doc = "Generated from 'VK_EXT_border_color_swizzle'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES_EXT: Self = Self(1_000_411_000);
    pub const SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT: Self = Self(1_000_411_001);
}
#[doc = "Generated from 'VK_EXT_pageable_device_local_memory'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT: Self = Self(1_000_412_000);
}
#[doc = "Generated from 'VK_KHR_maintenance4'"]
impl ImageAspectFlags {
    pub const NONE_KHR: Self = Self::NONE;
}
#[doc = "Generated from 'VK_KHR_maintenance4'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    pub const PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
    pub const DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR: Self = Self::DEVICE_BUFFER_MEMORY_REQUIREMENTS;
    pub const DEVICE_IMAGE_MEMORY_REQUIREMENTS_KHR: Self = Self::DEVICE_IMAGE_MEMORY_REQUIREMENTS;
}
#[doc = "Generated from 'VK_ARM_shader_core_properties'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_ARM: Self = Self(1_000_415_000);
}
#[doc = "Generated from 'VK_KHR_shader_subgroup_rotate'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES;
}
#[doc = "Generated from 'VK_KHR_shader_subgroup_rotate'"]
impl SubgroupFeatureFlags {
    pub const ROTATE_KHR: Self = Self::ROTATE;
    pub const ROTATE_CLUSTERED_KHR: Self = Self::ROTATE_CLUSTERED;
}
#[doc = "Generated from 'VK_ARM_scheduling_controls'"]
impl StructureType {
    pub const DEVICE_QUEUE_SHADER_CORE_CONTROL_CREATE_INFO_ARM: Self = Self(1_000_417_000);
    pub const PHYSICAL_DEVICE_SCHEDULING_CONTROLS_FEATURES_ARM: Self = Self(1_000_417_001);
    pub const PHYSICAL_DEVICE_SCHEDULING_CONTROLS_PROPERTIES_ARM: Self = Self(1_000_417_002);
    pub const DISPATCH_PARAMETERS_ARM: Self = Self(1_000_417_003);
    pub const PHYSICAL_DEVICE_SCHEDULING_CONTROLS_DISPATCH_PARAMETERS_PROPERTIES_ARM: Self =
        Self(1_000_417_004);
}
#[doc = "Generated from 'VK_EXT_image_sliced_view_of_3d'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_SLICED_VIEW_OF_3D_FEATURES_EXT: Self = Self(1_000_418_000);
    pub const IMAGE_VIEW_SLICED_CREATE_INFO_EXT: Self = Self(1_000_418_001);
}
#[doc = "Generated from 'VK_VALVE_descriptor_set_host_mapping'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DESCRIPTOR_SET_HOST_MAPPING_FEATURES_VALVE: Self =
        Self(1_000_420_000);
    pub const DESCRIPTOR_SET_BINDING_REFERENCE_VALVE: Self = Self(1_000_420_001);
    pub const DESCRIPTOR_SET_LAYOUT_HOST_MAPPING_INFO_VALVE: Self = Self(1_000_420_002);
}
#[doc = "Generated from 'VK_EXT_depth_clamp_zero_one'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_KHR;
}
#[doc = "Generated from 'VK_EXT_non_seamless_cube_map'"]
impl SamplerCreateFlags {
    pub const NON_SEAMLESS_CUBE_MAP_EXT: Self = Self(0b100);
}
#[doc = "Generated from 'VK_EXT_non_seamless_cube_map'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT: Self = Self(1_000_422_000);
}
#[doc = "Generated from 'VK_ARM_render_pass_striped'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RENDER_PASS_STRIPED_FEATURES_ARM: Self = Self(1_000_424_000);
    pub const PHYSICAL_DEVICE_RENDER_PASS_STRIPED_PROPERTIES_ARM: Self = Self(1_000_424_001);
    pub const RENDER_PASS_STRIPE_BEGIN_INFO_ARM: Self = Self(1_000_424_002);
    pub const RENDER_PASS_STRIPE_INFO_ARM: Self = Self(1_000_424_003);
    pub const RENDER_PASS_STRIPE_SUBMIT_INFO_ARM: Self = Self(1_000_424_004);
}
#[doc = "Generated from 'VK_QCOM_fragment_density_map_offset'"]
impl ImageCreateFlags {
    pub const FRAGMENT_DENSITY_MAP_OFFSET_QCOM: Self = Self::FRAGMENT_DENSITY_MAP_OFFSET_EXT;
}
#[doc = "Generated from 'VK_QCOM_fragment_density_map_offset'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_QCOM: Self =
        Self::PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_EXT;
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_PROPERTIES_QCOM: Self =
        Self::PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_PROPERTIES_EXT;
    pub const SUBPASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_QCOM: Self =
        Self::RENDER_PASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_EXT;
}
#[doc = "Generated from 'VK_NV_copy_memory_indirect'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_NV: Self = Self(1_000_426_000);
    pub const PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_PROPERTIES_NV: Self =
        Self::PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_PROPERTIES_KHR;
}
#[doc = "Generated from 'VK_NV_memory_decompression'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_NV: Self =
        Self::PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_EXT;
    pub const PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_NV: Self =
        Self::PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_EXT;
}
#[doc = "Generated from 'VK_NV_device_generated_commands_compute'"]
impl DescriptorSetLayoutCreateFlags {
    pub const INDIRECT_BINDABLE_NV: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_NV_device_generated_commands_compute'"]
impl IndirectCommandsTokenTypeNV {
    pub const PIPELINE: Self = Self(1_000_428_003);
    pub const DISPATCH: Self = Self(1_000_428_004);
}
#[doc = "Generated from 'VK_NV_device_generated_commands_compute'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_COMPUTE_FEATURES_NV: Self =
        Self(1_000_428_000);
    pub const COMPUTE_PIPELINE_INDIRECT_BUFFER_INFO_NV: Self = Self(1_000_428_001);
    pub const PIPELINE_INDIRECT_DEVICE_ADDRESS_INFO_NV: Self = Self(1_000_428_002);
}
#[doc = "Generated from 'VK_NV_ray_tracing_linear_swept_spheres'"]
impl FormatFeatureFlags2 {
    pub const ACCELERATION_STRUCTURE_RADIUS_BUFFER_NV: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_ray_tracing_linear_swept_spheres'"]
impl GeometryTypeKHR {
    pub const SPHERES_NV: Self = Self(1_000_429_004);
    pub const LINEAR_SWEPT_SPHERES_NV: Self = Self(1_000_429_005);
}
#[doc = "Generated from 'VK_NV_ray_tracing_linear_swept_spheres'"]
impl PipelineCreateFlags2 {
    pub const RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES_NV: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_ray_tracing_linear_swept_spheres'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV: Self =
        Self(1_000_429_008);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_LINEAR_SWEPT_SPHERES_DATA_NV: Self =
        Self(1_000_429_009);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_SPHERES_DATA_NV: Self = Self(1_000_429_010);
}
#[doc = "Generated from 'VK_NV_linear_color_attachment'"]
impl FormatFeatureFlags2 {
    #[doc = "Format support linear image as render target, it cannot be mixed with non linear attachment"]
    pub const LINEAR_COLOR_ATTACHMENT_NV: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_linear_color_attachment'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_LINEAR_COLOR_ATTACHMENT_FEATURES_NV: Self = Self(1_000_430_000);
}
#[doc = "Generated from 'VK_KHR_shader_maximal_reconvergence'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR: Self = Self(1_000_434_000);
}
#[doc = "Generated from 'VK_EXT_image_compression_control_swapchain'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_FEATURES_EXT: Self =
        Self(1_000_437_000);
}
#[doc = "Generated from 'VK_QCOM_image_processing'"]
impl DescriptorType {
    pub const SAMPLE_WEIGHT_IMAGE_QCOM: Self = Self(1_000_440_000);
    pub const BLOCK_MATCH_IMAGE_QCOM: Self = Self(1_000_440_001);
}
#[doc = "Generated from 'VK_QCOM_image_processing'"]
impl FormatFeatureFlags2 {
    pub const WEIGHT_IMAGE_QCOM: Self = Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const WEIGHT_SAMPLED_IMAGE_QCOM: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const BLOCK_MATCHING_QCOM: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const BOX_FILTER_SAMPLED_QCOM: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_QCOM_image_processing'"]
impl ImageUsageFlags {
    pub const SAMPLE_WEIGHT_QCOM: Self = Self(0b1_0000_0000_0000_0000_0000);
    pub const SAMPLE_BLOCK_MATCH_QCOM: Self = Self(0b10_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_QCOM_image_processing'"]
impl SamplerCreateFlags {
    pub const IMAGE_PROCESSING_QCOM: Self = Self(0b1_0000);
}
#[doc = "Generated from 'VK_QCOM_image_processing'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM: Self = Self(1_000_440_000);
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES_QCOM: Self = Self(1_000_440_001);
    pub const IMAGE_VIEW_SAMPLE_WEIGHT_CREATE_INFO_QCOM: Self = Self(1_000_440_002);
}
#[doc = "Generated from 'VK_EXT_nested_command_buffer'"]
impl RenderingFlags {
    pub const CONTENTS_INLINE_EXT: Self = Self::CONTENTS_INLINE_KHR;
}
#[doc = "Generated from 'VK_EXT_nested_command_buffer'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT: Self = Self(1_000_451_000);
    pub const PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_PROPERTIES_EXT: Self = Self(1_000_451_001);
}
#[doc = "Generated from 'VK_EXT_nested_command_buffer'"]
impl SubpassContents {
    pub const INLINE_AND_SECONDARY_COMMAND_BUFFERS_EXT: Self =
        Self::INLINE_AND_SECONDARY_COMMAND_BUFFERS_KHR;
}
#[doc = "Generated from 'VK_OHOS_external_memory'"]
impl ExternalMemoryHandleTypeFlags {
    pub const OH_NATIVE_BUFFER_OHOS: Self = Self(0b1000_0000_0000_0000);
}
#[doc = "Generated from 'VK_OHOS_external_memory'"]
impl StructureType {
    pub const NATIVE_BUFFER_USAGE_OHOS: Self = Self(1_000_452_000);
    pub const NATIVE_BUFFER_PROPERTIES_OHOS: Self = Self(1_000_452_001);
    pub const NATIVE_BUFFER_FORMAT_PROPERTIES_OHOS: Self = Self(1_000_452_002);
    pub const IMPORT_NATIVE_BUFFER_INFO_OHOS: Self = Self(1_000_452_003);
    pub const MEMORY_GET_NATIVE_BUFFER_INFO_OHOS: Self = Self(1_000_452_004);
    pub const EXTERNAL_FORMAT_OHOS: Self = Self(1_000_452_005);
}
#[doc = "Generated from 'VK_EXT_external_memory_acquire_unmodified'"]
impl StructureType {
    pub const EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED_EXT: Self = Self(1_000_453_000);
}
#[doc = "Generated from 'VK_EXT_extended_dynamic_state3'"]
impl DynamicState {
    pub const DEPTH_CLAMP_ENABLE_EXT: Self = Self(1_000_455_003);
    pub const POLYGON_MODE_EXT: Self = Self(1_000_455_004);
    pub const RASTERIZATION_SAMPLES_EXT: Self = Self(1_000_455_005);
    pub const SAMPLE_MASK_EXT: Self = Self(1_000_455_006);
    pub const ALPHA_TO_COVERAGE_ENABLE_EXT: Self = Self(1_000_455_007);
    pub const ALPHA_TO_ONE_ENABLE_EXT: Self = Self(1_000_455_008);
    pub const LOGIC_OP_ENABLE_EXT: Self = Self(1_000_455_009);
    pub const COLOR_BLEND_ENABLE_EXT: Self = Self(1_000_455_010);
    pub const COLOR_BLEND_EQUATION_EXT: Self = Self(1_000_455_011);
    pub const COLOR_WRITE_MASK_EXT: Self = Self(1_000_455_012);
    pub const TESSELLATION_DOMAIN_ORIGIN_EXT: Self = Self(1_000_455_002);
    pub const RASTERIZATION_STREAM_EXT: Self = Self(1_000_455_013);
    pub const CONSERVATIVE_RASTERIZATION_MODE_EXT: Self = Self(1_000_455_014);
    pub const EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_EXT: Self = Self(1_000_455_015);
    pub const DEPTH_CLIP_ENABLE_EXT: Self = Self(1_000_455_016);
    pub const SAMPLE_LOCATIONS_ENABLE_EXT: Self = Self(1_000_455_017);
    pub const COLOR_BLEND_ADVANCED_EXT: Self = Self(1_000_455_018);
    pub const PROVOKING_VERTEX_MODE_EXT: Self = Self(1_000_455_019);
    pub const LINE_RASTERIZATION_MODE_EXT: Self = Self(1_000_455_020);
    pub const LINE_STIPPLE_ENABLE_EXT: Self = Self(1_000_455_021);
    pub const DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT: Self = Self(1_000_455_022);
    pub const VIEWPORT_W_SCALING_ENABLE_NV: Self = Self(1_000_455_023);
    pub const VIEWPORT_SWIZZLE_NV: Self = Self(1_000_455_024);
    pub const COVERAGE_TO_COLOR_ENABLE_NV: Self = Self(1_000_455_025);
    pub const COVERAGE_TO_COLOR_LOCATION_NV: Self = Self(1_000_455_026);
    pub const COVERAGE_MODULATION_MODE_NV: Self = Self(1_000_455_027);
    pub const COVERAGE_MODULATION_TABLE_ENABLE_NV: Self = Self(1_000_455_028);
    pub const COVERAGE_MODULATION_TABLE_NV: Self = Self(1_000_455_029);
    pub const SHADING_RATE_IMAGE_ENABLE_NV: Self = Self(1_000_455_030);
    pub const REPRESENTATIVE_FRAGMENT_TEST_ENABLE_NV: Self = Self(1_000_455_031);
    pub const COVERAGE_REDUCTION_MODE_NV: Self = Self(1_000_455_032);
}
#[doc = "Generated from 'VK_EXT_extended_dynamic_state3'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT: Self = Self(1_000_455_000);
    pub const PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT: Self = Self(1_000_455_001);
}
#[doc = "Generated from 'VK_EXT_subpass_merge_feedback'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SUBPASS_MERGE_FEEDBACK_FEATURES_EXT: Self = Self(1_000_458_000);
    pub const RENDER_PASS_CREATION_CONTROL_EXT: Self = Self(1_000_458_001);
    pub const RENDER_PASS_CREATION_FEEDBACK_CREATE_INFO_EXT: Self = Self(1_000_458_002);
    pub const RENDER_PASS_SUBPASS_FEEDBACK_CREATE_INFO_EXT: Self = Self(1_000_458_003);
}
#[doc = "Generated from 'VK_LUNARG_direct_driver_loading'"]
impl StructureType {
    pub const DIRECT_DRIVER_LOADING_INFO_LUNARG: Self = Self(1_000_459_000);
    pub const DIRECT_DRIVER_LOADING_LIST_LUNARG: Self = Self(1_000_459_001);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl DescriptorType {
    pub const TENSOR_ARM: Self = Self(1_000_460_000);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl Format {
    pub const R8_BOOL_ARM: Self = Self(1_000_460_000);
    pub const R16_SFLOAT_FPENCODING_BFLOAT16_ARM: Self = Self(1_000_460_001);
    pub const R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM: Self = Self(1_000_460_002);
    pub const R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM: Self = Self(1_000_460_003);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl FormatFeatureFlags2 {
    pub const TENSOR_SHADER_ARM: Self = Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const TENSOR_IMAGE_ALIASING_ARM: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl ImageLayout {
    pub const TENSOR_ALIASING_ARM: Self = Self(1_000_460_000);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl ImageUsageFlags {
    pub const TENSOR_ALIASING_ARM: Self = Self(0b1000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl ObjectType {
    pub const TENSOR_ARM: Self = Self(1_000_460_000);
    pub const TENSOR_VIEW_ARM: Self = Self(1_000_460_001);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl StructureType {
    pub const TENSOR_CREATE_INFO_ARM: Self = Self(1_000_460_000);
    pub const TENSOR_VIEW_CREATE_INFO_ARM: Self = Self(1_000_460_001);
    pub const BIND_TENSOR_MEMORY_INFO_ARM: Self = Self(1_000_460_002);
    pub const WRITE_DESCRIPTOR_SET_TENSOR_ARM: Self = Self(1_000_460_003);
    pub const PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM: Self = Self(1_000_460_004);
    pub const TENSOR_FORMAT_PROPERTIES_ARM: Self = Self(1_000_460_005);
    pub const TENSOR_DESCRIPTION_ARM: Self = Self(1_000_460_006);
    pub const TENSOR_MEMORY_REQUIREMENTS_INFO_ARM: Self = Self(1_000_460_007);
    pub const TENSOR_MEMORY_BARRIER_ARM: Self = Self(1_000_460_008);
    pub const PHYSICAL_DEVICE_TENSOR_FEATURES_ARM: Self = Self(1_000_460_009);
    pub const DEVICE_TENSOR_MEMORY_REQUIREMENTS_ARM: Self = Self(1_000_460_010);
    pub const COPY_TENSOR_INFO_ARM: Self = Self(1_000_460_011);
    pub const TENSOR_COPY_ARM: Self = Self(1_000_460_012);
    pub const TENSOR_DEPENDENCY_INFO_ARM: Self = Self(1_000_460_013);
    pub const MEMORY_DEDICATED_ALLOCATE_INFO_TENSOR_ARM: Self = Self(1_000_460_014);
    pub const PHYSICAL_DEVICE_EXTERNAL_TENSOR_INFO_ARM: Self = Self(1_000_460_015);
    pub const EXTERNAL_TENSOR_PROPERTIES_ARM: Self = Self(1_000_460_016);
    pub const EXTERNAL_MEMORY_TENSOR_CREATE_INFO_ARM: Self = Self(1_000_460_017);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_TENSOR_FEATURES_ARM: Self = Self(1_000_460_018);
    pub const PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_TENSOR_PROPERTIES_ARM: Self = Self(1_000_460_019);
    pub const DESCRIPTOR_GET_TENSOR_INFO_ARM: Self = Self(1_000_460_020);
    pub const TENSOR_CAPTURE_DESCRIPTOR_DATA_INFO_ARM: Self = Self(1_000_460_021);
    pub const TENSOR_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO_ARM: Self = Self(1_000_460_022);
    pub const FRAME_BOUNDARY_TENSORS_ARM: Self = Self(1_000_460_023);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl TensorCreateFlagsARM {
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY: Self = Self(0b100);
}
#[doc = "Generated from 'VK_ARM_tensors'"]
impl TensorViewCreateFlagsARM {
    pub const DESCRIPTOR_BUFFER_CAPTURE_REPLAY: Self = Self(0b1);
}
#[doc = "Generated from 'VK_EXT_shader_module_identifier'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_FEATURES_EXT: Self = Self(1_000_462_000);
    pub const PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_PROPERTIES_EXT: Self = Self(1_000_462_001);
    pub const PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT: Self = Self(1_000_462_002);
    pub const SHADER_MODULE_IDENTIFIER_EXT: Self = Self(1_000_462_003);
}
#[doc = "Generated from 'VK_EXT_rasterization_order_attachment_access'"]
impl PipelineColorBlendStateCreateFlags {
    pub const RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXT: Self = Self(0b1);
}
#[doc = "Generated from 'VK_EXT_rasterization_order_attachment_access'"]
impl PipelineDepthStencilStateCreateFlags {
    pub const RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_EXT: Self = Self(0b1);
    pub const RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_EXT: Self = Self(0b10);
}
#[doc = "Generated from 'VK_EXT_rasterization_order_attachment_access'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_EXT: Self =
        Self(1_000_342_000);
}
#[doc = "Generated from 'VK_EXT_rasterization_order_attachment_access'"]
impl SubpassDescriptionFlags {
    pub const RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_EXT: Self = Self(0b1_0000);
    pub const RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_EXT: Self = Self(0b10_0000);
    pub const RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_EXT: Self = Self(0b100_0000);
}
#[doc = "Generated from 'VK_NV_optical_flow'"]
impl AccessFlags2 {
    pub const OPTICAL_FLOW_READ_NV: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const OPTICAL_FLOW_WRITE_NV: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_optical_flow'"]
impl Format {
    pub const R16G16_SFIXED5_NV: Self = Self(1_000_464_000);
}
#[doc = "Generated from 'VK_NV_optical_flow'"]
impl FormatFeatureFlags2 {
    pub const OPTICAL_FLOW_IMAGE_NV: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const OPTICAL_FLOW_VECTOR_NV: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const OPTICAL_FLOW_COST_NV: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_optical_flow'"]
impl ObjectType {
    pub const OPTICAL_FLOW_SESSION_NV: Self = Self(1_000_464_000);
}
#[doc = "Generated from 'VK_NV_optical_flow'"]
impl PipelineStageFlags2 {
    pub const OPTICAL_FLOW_NV: Self = Self(0b10_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_optical_flow'"]
impl QueueFlags {
    pub const OPTICAL_FLOW_NV: Self = Self(0b1_0000_0000);
}
#[doc = "Generated from 'VK_NV_optical_flow'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV: Self = Self(1_000_464_000);
    pub const PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV: Self = Self(1_000_464_001);
    pub const OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV: Self = Self(1_000_464_002);
    pub const OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_NV: Self = Self(1_000_464_003);
    pub const OPTICAL_FLOW_SESSION_CREATE_INFO_NV: Self = Self(1_000_464_004);
    pub const OPTICAL_FLOW_EXECUTE_INFO_NV: Self = Self(1_000_464_005);
    pub const OPTICAL_FLOW_SESSION_CREATE_PRIVATE_DATA_INFO_NV: Self = Self(1_000_464_010);
}
#[doc = "Generated from 'VK_EXT_legacy_dithering'"]
impl PipelineCreateFlags2 {
    pub const ENABLE_LEGACY_DITHERING_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_legacy_dithering'"]
impl RenderingFlags {
    pub const ENABLE_LEGACY_DITHERING_EXT: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_EXT_legacy_dithering'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_LEGACY_DITHERING_FEATURES_EXT: Self = Self(1_000_465_000);
}
#[doc = "Generated from 'VK_EXT_legacy_dithering'"]
impl SubpassDescriptionFlags {
    pub const ENABLE_LEGACY_DITHERING_EXT: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_EXT_pipeline_protected_access'"]
impl PipelineCreateFlags {
    pub const NO_PROTECTED_ACCESS_EXT: Self = Self::NO_PROTECTED_ACCESS;
    pub const PROTECTED_ACCESS_ONLY_EXT: Self = Self::PROTECTED_ACCESS_ONLY;
}
#[doc = "Generated from 'VK_EXT_pipeline_protected_access'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES_EXT: Self =
        Self::PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES;
}
#[doc = "Generated from 'VK_ANDROID_external_format_resolve'"]
impl ResolveModeFlags {
    pub const EXTERNAL_FORMAT_DOWNSAMPLE_ANDROID: Self = Self(0b1_0000);
}
#[doc = "Generated from 'VK_ANDROID_external_format_resolve'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_FEATURES_ANDROID: Self = Self(1_000_468_000);
    pub const PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_PROPERTIES_ANDROID: Self =
        Self(1_000_468_001);
    pub const ANDROID_HARDWARE_BUFFER_FORMAT_RESOLVE_PROPERTIES_ANDROID: Self = Self(1_000_468_002);
}
#[doc = "Generated from 'VK_KHR_maintenance5'"]
impl BufferUsageFlags2 {
    pub const TRANSFER_SRC_KHR: Self = Self::TRANSFER_SRC;
    pub const TRANSFER_DST_KHR: Self = Self::TRANSFER_DST;
    pub const UNIFORM_TEXEL_BUFFER_KHR: Self = Self::UNIFORM_TEXEL_BUFFER;
    pub const STORAGE_TEXEL_BUFFER_KHR: Self = Self::STORAGE_TEXEL_BUFFER;
    pub const UNIFORM_BUFFER_KHR: Self = Self::UNIFORM_BUFFER;
    pub const STORAGE_BUFFER_KHR: Self = Self::STORAGE_BUFFER;
    pub const INDEX_BUFFER_KHR: Self = Self::INDEX_BUFFER;
    pub const VERTEX_BUFFER_KHR: Self = Self::VERTEX_BUFFER;
    pub const INDIRECT_BUFFER_KHR: Self = Self::INDIRECT_BUFFER;
    pub const CONDITIONAL_RENDERING_EXT: Self = Self(0b10_0000_0000);
    pub const SHADER_BINDING_TABLE_KHR: Self = Self(0b100_0000_0000);
    pub const RAY_TRACING_NV: Self = Self::SHADER_BINDING_TABLE_KHR;
    pub const TRANSFORM_FEEDBACK_BUFFER_EXT: Self = Self(0b1000_0000_0000);
    pub const TRANSFORM_FEEDBACK_COUNTER_BUFFER_EXT: Self = Self(0b1_0000_0000_0000);
    pub const VIDEO_DECODE_SRC_KHR: Self = Self(0b10_0000_0000_0000);
    pub const VIDEO_DECODE_DST_KHR: Self = Self(0b100_0000_0000_0000);
    pub const VIDEO_ENCODE_DST_KHR: Self = Self(0b1000_0000_0000_0000);
    pub const VIDEO_ENCODE_SRC_KHR: Self = Self(0b1_0000_0000_0000_0000);
    pub const SHADER_DEVICE_ADDRESS_KHR: Self = Self::SHADER_DEVICE_ADDRESS;
    pub const ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR: Self =
        Self(0b1000_0000_0000_0000_0000);
    pub const ACCELERATION_STRUCTURE_STORAGE_KHR: Self = Self(0b1_0000_0000_0000_0000_0000);
    pub const SAMPLER_DESCRIPTOR_BUFFER_EXT: Self = Self(0b10_0000_0000_0000_0000_0000);
    pub const RESOURCE_DESCRIPTOR_BUFFER_EXT: Self = Self(0b100_0000_0000_0000_0000_0000);
    pub const PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance5'"]
impl Format {
    pub const A1B5G5R5_UNORM_PACK16_KHR: Self = Self::A1B5G5R5_UNORM_PACK16;
    pub const A8_UNORM_KHR: Self = Self::A8_UNORM;
}
#[doc = "Generated from 'VK_KHR_maintenance5'"]
impl PipelineCreateFlags2 {
    pub const DISABLE_OPTIMIZATION_KHR: Self = Self::DISABLE_OPTIMIZATION;
    pub const ALLOW_DERIVATIVES_KHR: Self = Self::ALLOW_DERIVATIVES;
    pub const DERIVATIVE_KHR: Self = Self::DERIVATIVE;
    pub const VIEW_INDEX_FROM_DEVICE_INDEX_KHR: Self = Self::VIEW_INDEX_FROM_DEVICE_INDEX;
    pub const DEFER_COMPILE_NV: Self = Self(0b10_0000);
    pub const CAPTURE_STATISTICS_KHR: Self = Self(0b100_0000);
    pub const CAPTURE_INTERNAL_REPRESENTATIONS_KHR: Self = Self(0b1000_0000);
    pub const FAIL_ON_PIPELINE_COMPILE_REQUIRED_KHR: Self = Self::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
    pub const EARLY_RETURN_ON_FAILURE_KHR: Self = Self::EARLY_RETURN_ON_FAILURE;
    pub const LINK_TIME_OPTIMIZATION_EXT: Self = Self(0b100_0000_0000);
    pub const RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT: Self = Self(0b1000_0000_0000_0000_0000_0000);
    pub const LIBRARY_KHR: Self = Self(0b1000_0000_0000);
    pub const RAY_TRACING_SKIP_TRIANGLES_KHR: Self = Self(0b1_0000_0000_0000);
    pub const RAY_TRACING_SKIP_AABBS_KHR: Self = Self(0b10_0000_0000_0000);
    pub const RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_KHR: Self = Self(0b100_0000_0000_0000);
    pub const RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_KHR: Self = Self(0b1000_0000_0000_0000);
    pub const RAY_TRACING_NO_NULL_MISS_SHADERS_KHR: Self = Self(0b1_0000_0000_0000_0000);
    pub const RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_KHR: Self = Self(0b10_0000_0000_0000_0000);
    pub const RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_KHR: Self =
        Self(0b1000_0000_0000_0000_0000);
    pub const INDIRECT_BINDABLE_NV: Self = Self(0b100_0000_0000_0000_0000);
    pub const RAY_TRACING_ALLOW_MOTION_NV: Self = Self(0b1_0000_0000_0000_0000_0000);
    pub const RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_KHR: Self =
        Self(0b10_0000_0000_0000_0000_0000);
    pub const RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000);
    pub const COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT: Self = Self(0b10_0000_0000_0000_0000_0000_0000);
    pub const DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000);
    pub const NO_PROTECTED_ACCESS_EXT: Self = Self::NO_PROTECTED_ACCESS;
    pub const PROTECTED_ACCESS_ONLY_EXT: Self = Self::PROTECTED_ACCESS_ONLY;
    pub const RAY_TRACING_DISPLACEMENT_MICROMAP_NV: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000);
    pub const DESCRIPTOR_BUFFER_EXT: Self = Self(0b10_0000_0000_0000_0000_0000_0000_0000);
    pub const DISALLOW_OPACITY_MICROMAP_ARM: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const INSTRUMENT_SHADERS_ARM: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance5'"]
impl ShaderCreateFlagsEXT {
    pub const INSTRUMENT_SHADER_ARM: Self = Self(0b1000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance5'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;
    pub const PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES;
    pub const RENDERING_AREA_INFO_KHR: Self = Self::RENDERING_AREA_INFO;
    pub const DEVICE_IMAGE_SUBRESOURCE_INFO_KHR: Self = Self::DEVICE_IMAGE_SUBRESOURCE_INFO;
    pub const SUBRESOURCE_LAYOUT_2_KHR: Self = Self::SUBRESOURCE_LAYOUT_2;
    pub const IMAGE_SUBRESOURCE_2_KHR: Self = Self::IMAGE_SUBRESOURCE_2;
    pub const PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR: Self =
        Self::PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
    pub const BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR: Self = Self::BUFFER_USAGE_FLAGS_2_CREATE_INFO;
}
#[doc = "Generated from 'VK_AMD_anti_lag'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ANTI_LAG_FEATURES_AMD: Self = Self(1_000_476_000);
    pub const ANTI_LAG_DATA_AMD: Self = Self(1_000_476_001);
    pub const ANTI_LAG_PRESENTATION_INFO_AMD: Self = Self(1_000_476_002);
}
#[doc = "Generated from 'VK_AMDX_dense_geometry_format'"]
#[cfg(feature = "provisional")]
impl BufferUsageFlags2 {
    pub const COMPRESSED_DATA_DGF1_AMDX: Self = Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_AMDX_dense_geometry_format'"]
#[cfg(feature = "provisional")]
impl GeometryTypeKHR {
    pub const DENSE_GEOMETRY_FORMAT_TRIANGLES_AMDX: Self = Self(1_000_478_000);
}
#[doc = "Generated from 'VK_AMDX_dense_geometry_format'"]
#[cfg(feature = "provisional")]
impl StructureType {
    pub const PHYSICAL_DEVICE_DENSE_GEOMETRY_FORMAT_FEATURES_AMDX: Self = Self(1_000_478_000);
    pub const ACCELERATION_STRUCTURE_DENSE_GEOMETRY_FORMAT_TRIANGLES_DATA_AMDX: Self =
        Self(1_000_478_001);
}
#[doc = "Generated from 'VK_KHR_present_id2'"]
impl StructureType {
    pub const SURFACE_CAPABILITIES_PRESENT_ID_2_KHR: Self = Self(1_000_479_000);
    pub const PRESENT_ID_2_KHR: Self = Self(1_000_479_001);
    pub const PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR: Self = Self(1_000_479_002);
}
#[doc = "Generated from 'VK_KHR_present_id2'"]
impl SwapchainCreateFlagsKHR {
    #[doc = "Allow use of VK_KHR_present_id2 with this swapchain"]
    pub const PRESENT_ID_2: Self = Self(0b100_0000);
}
#[doc = "Generated from 'VK_KHR_present_wait2'"]
impl StructureType {
    pub const SURFACE_CAPABILITIES_PRESENT_WAIT_2_KHR: Self = Self(1_000_480_000);
    pub const PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR: Self = Self(1_000_480_001);
    pub const PRESENT_WAIT_2_INFO_KHR: Self = Self(1_000_480_002);
}
#[doc = "Generated from 'VK_KHR_present_wait2'"]
impl SwapchainCreateFlagsKHR {
    #[doc = "Allow use of VK_KHR_present_wait2 with this swapchain"]
    pub const PRESENT_WAIT_2: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_position_fetch'"]
impl BuildAccelerationStructureFlagsKHR {
    pub const ALLOW_DATA_ACCESS: Self = Self(0b1000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_ray_tracing_position_fetch'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR: Self = Self(1_000_481_000);
}
#[doc = "Generated from 'VK_EXT_shader_object'"]
impl ObjectType {
    pub const SHADER_EXT: Self = Self(1_000_482_000);
}
#[doc = "Generated from 'VK_EXT_shader_object'"]
impl Result {
    pub const INCOMPATIBLE_SHADER_BINARY_EXT: Self = Self(1_000_482_000);
}
#[doc = "Generated from 'VK_EXT_shader_object'"]
impl ShaderCreateFlagsEXT {
    pub const ALLOW_VARYING_SUBGROUP_SIZE: Self = Self(0b10);
    pub const REQUIRE_FULL_SUBGROUPS: Self = Self(0b100);
    pub const NO_TASK_SHADER: Self = Self(0b1000);
    pub const DISPATCH_BASE: Self = Self(0b1_0000);
    pub const FRAGMENT_SHADING_RATE_ATTACHMENT: Self = Self(0b10_0000);
    pub const FRAGMENT_DENSITY_MAP_ATTACHMENT: Self = Self(0b100_0000);
}
#[doc = "Generated from 'VK_EXT_shader_object'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT: Self = Self(1_000_482_000);
    pub const PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT: Self = Self(1_000_482_001);
    pub const SHADER_CREATE_INFO_EXT: Self = Self(1_000_482_002);
    pub const SHADER_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT: Self =
        Self::PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
}
#[doc = "Generated from 'VK_KHR_pipeline_binary'"]
impl ObjectType {
    pub const PIPELINE_BINARY_KHR: Self = Self(1_000_483_000);
}
#[doc = "Generated from 'VK_KHR_pipeline_binary'"]
impl PipelineCreateFlags2 {
    pub const CAPTURE_DATA_KHR: Self = Self(0b1000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_pipeline_binary'"]
impl Result {
    pub const PIPELINE_BINARY_MISSING_KHR: Self = Self(1_000_483_000);
    pub const ERROR_NOT_ENOUGH_SPACE_KHR: Self = Self(-1_000_483_000);
}
#[doc = "Generated from 'VK_KHR_pipeline_binary'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR: Self = Self(1_000_483_000);
    pub const PIPELINE_BINARY_CREATE_INFO_KHR: Self = Self(1_000_483_001);
    pub const PIPELINE_BINARY_INFO_KHR: Self = Self(1_000_483_002);
    pub const PIPELINE_BINARY_KEY_KHR: Self = Self(1_000_483_003);
    pub const PHYSICAL_DEVICE_PIPELINE_BINARY_PROPERTIES_KHR: Self = Self(1_000_483_004);
    pub const RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR: Self = Self(1_000_483_005);
    pub const PIPELINE_BINARY_DATA_INFO_KHR: Self = Self(1_000_483_006);
    pub const PIPELINE_CREATE_INFO_KHR: Self = Self(1_000_483_007);
    pub const DEVICE_PIPELINE_BINARY_INTERNAL_CACHE_CONTROL_KHR: Self = Self(1_000_483_008);
    pub const PIPELINE_BINARY_HANDLES_INFO_KHR: Self = Self(1_000_483_009);
}
#[doc = "Generated from 'VK_QCOM_tile_properties'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_TILE_PROPERTIES_FEATURES_QCOM: Self = Self(1_000_484_000);
    pub const TILE_PROPERTIES_QCOM: Self = Self(1_000_484_001);
}
#[doc = "Generated from 'VK_SEC_amigo_profiling'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_AMIGO_PROFILING_FEATURES_SEC: Self = Self(1_000_485_000);
    pub const AMIGO_PROFILING_SUBMIT_INFO_SEC: Self = Self(1_000_485_001);
}
#[doc = "Generated from 'VK_KHR_surface_maintenance1'"]
impl StructureType {
    pub const SURFACE_PRESENT_MODE_KHR: Self = Self(1_000_274_000);
    pub const SURFACE_PRESENT_SCALING_CAPABILITIES_KHR: Self = Self(1_000_274_001);
    pub const SURFACE_PRESENT_MODE_COMPATIBILITY_KHR: Self = Self(1_000_274_002);
}
#[doc = "Generated from 'VK_KHR_swapchain_maintenance1'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR: Self = Self(1_000_275_000);
    pub const SWAPCHAIN_PRESENT_FENCE_INFO_KHR: Self = Self(1_000_275_001);
    pub const SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR: Self = Self(1_000_275_002);
    pub const SWAPCHAIN_PRESENT_MODE_INFO_KHR: Self = Self(1_000_275_003);
    pub const SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_KHR: Self = Self(1_000_275_004);
    pub const RELEASE_SWAPCHAIN_IMAGES_INFO_KHR: Self = Self(1_000_275_005);
}
#[doc = "Generated from 'VK_KHR_swapchain_maintenance1'"]
impl SwapchainCreateFlagsKHR {
    pub const DEFERRED_MEMORY_ALLOCATION: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_QCOM_multiview_per_view_viewports'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_VIEWPORTS_FEATURES_QCOM: Self =
        Self(1_000_488_000);
}
#[doc = "Generated from 'VK_NV_ray_tracing_invocation_reorder'"]
impl RayTracingInvocationReorderModeEXT {
    pub const NONE_NV: Self = Self::NONE;
    pub const REORDER_NV: Self = Self::REORDER;
}
#[doc = "Generated from 'VK_NV_ray_tracing_invocation_reorder'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV: Self =
        Self(1_000_490_000);
    pub const PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV: Self =
        Self(1_000_490_001);
}
#[doc = "Generated from 'VK_NV_cooperative_vector'"]
impl ComponentTypeKHR {
    pub const SINT8_PACKED_NV: Self = Self(1_000_491_000);
    pub const UINT8_PACKED_NV: Self = Self(1_000_491_001);
    pub const FLOAT_E4M3_NV: Self = Self::FLOAT8_E4M3_EXT;
    pub const FLOAT_E5M2_NV: Self = Self::FLOAT8_E5M2_EXT;
}
#[doc = "Generated from 'VK_NV_cooperative_vector'"]
impl PipelineStageFlags2 {
    pub const CONVERT_COOPERATIVE_VECTOR_MATRIX_NV: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_cooperative_vector'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV: Self = Self(1_000_491_000);
    pub const PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV: Self = Self(1_000_491_001);
    pub const COOPERATIVE_VECTOR_PROPERTIES_NV: Self = Self(1_000_491_002);
    pub const CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV: Self = Self(1_000_491_004);
}
#[doc = "Generated from 'VK_NV_extended_sparse_address_space'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_FEATURES_NV: Self = Self(1_000_492_000);
    pub const PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_PROPERTIES_NV: Self =
        Self(1_000_492_001);
}
#[doc = "Generated from 'VK_EXT_mutable_descriptor_type'"]
impl DescriptorPoolCreateFlags {
    pub const HOST_ONLY_EXT: Self = Self(0b100);
}
#[doc = "Generated from 'VK_EXT_mutable_descriptor_type'"]
impl DescriptorSetLayoutCreateFlags {
    pub const HOST_ONLY_POOL_EXT: Self = Self(0b100);
}
#[doc = "Generated from 'VK_EXT_mutable_descriptor_type'"]
impl DescriptorType {
    pub const MUTABLE_EXT: Self = Self(1_000_351_000);
}
#[doc = "Generated from 'VK_EXT_mutable_descriptor_type'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT: Self = Self(1_000_351_000);
    pub const MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT: Self = Self(1_000_351_002);
}
#[doc = "Generated from 'VK_EXT_legacy_vertex_attributes'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_LEGACY_VERTEX_ATTRIBUTES_FEATURES_EXT: Self = Self(1_000_495_000);
    pub const PHYSICAL_DEVICE_LEGACY_VERTEX_ATTRIBUTES_PROPERTIES_EXT: Self = Self(1_000_495_001);
}
#[doc = "Generated from 'VK_EXT_layer_settings'"]
impl StructureType {
    pub const LAYER_SETTINGS_CREATE_INFO_EXT: Self = Self(1_000_496_000);
}
#[doc = "Generated from 'VK_ARM_shader_core_builtins'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_FEATURES_ARM: Self = Self(1_000_497_000);
    pub const PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM: Self = Self(1_000_497_001);
}
#[doc = "Generated from 'VK_EXT_pipeline_library_group_handles'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_EXT: Self =
        Self(1_000_498_000);
}
#[doc = "Generated from 'VK_EXT_dynamic_rendering_unused_attachments'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT: Self =
        Self(1_000_499_000);
}
#[doc = "Generated from 'VK_KHR_internally_synchronized_queues'"]
impl DeviceQueueCreateFlags {
    pub const INTERNALLY_SYNCHRONIZED_KHR: Self = Self(0b100);
}
#[doc = "Generated from 'VK_KHR_internally_synchronized_queues'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR: Self =
        Self(1_000_504_000);
}
#[doc = "Generated from 'VK_NV_low_latency2'"]
impl StructureType {
    pub const LATENCY_SLEEP_MODE_INFO_NV: Self = Self(1_000_505_000);
    pub const LATENCY_SLEEP_INFO_NV: Self = Self(1_000_505_001);
    pub const SET_LATENCY_MARKER_INFO_NV: Self = Self(1_000_505_002);
    pub const GET_LATENCY_MARKER_INFO_NV: Self = Self(1_000_505_003);
    pub const LATENCY_TIMINGS_FRAME_REPORT_NV: Self = Self(1_000_505_004);
    pub const LATENCY_SUBMISSION_PRESENT_ID_NV: Self = Self(1_000_505_005);
    pub const OUT_OF_BAND_QUEUE_TYPE_INFO_NV: Self = Self(1_000_505_006);
    pub const SWAPCHAIN_LATENCY_CREATE_INFO_NV: Self = Self(1_000_505_007);
    pub const LATENCY_SURFACE_CAPABILITIES_NV: Self = Self(1_000_505_008);
}
#[doc = "Generated from 'VK_KHR_cooperative_matrix'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR: Self = Self(1_000_506_000);
    pub const COOPERATIVE_MATRIX_PROPERTIES_KHR: Self = Self(1_000_506_001);
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR: Self = Self(1_000_506_002);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl AccessFlags2 {
    pub const DATA_GRAPH_READ_ARM: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const DATA_GRAPH_WRITE_ARM: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl BufferUsageFlags2 {
    pub const DATA_GRAPH_FOREIGN_DESCRIPTOR_ARM: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl FormatFeatureFlags2 {
    pub const TENSOR_DATA_GRAPH_ARM: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl ObjectType {
    pub const DATA_GRAPH_PIPELINE_SESSION_ARM: Self = Self(1_000_507_000);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl PipelineBindPoint {
    pub const DATA_GRAPH_ARM: Self = Self(1_000_507_000);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl PipelineStageFlags2 {
    pub const DATA_GRAPH_ARM: Self = Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl QueueFlags {
    pub const DATA_GRAPH_ARM: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl StructureType {
    pub const DATA_GRAPH_PIPELINE_CREATE_INFO_ARM: Self = Self(1_000_507_000);
    pub const DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM: Self = Self(1_000_507_001);
    pub const DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM: Self = Self(1_000_507_002);
    pub const DATA_GRAPH_PIPELINE_CONSTANT_ARM: Self = Self(1_000_507_003);
    pub const DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM: Self = Self(1_000_507_004);
    pub const BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM: Self = Self(1_000_507_005);
    pub const PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM: Self = Self(1_000_507_006);
    pub const DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM: Self = Self(1_000_507_007);
    pub const DATA_GRAPH_PIPELINE_PROPERTY_QUERY_RESULT_ARM: Self = Self(1_000_507_008);
    pub const DATA_GRAPH_PIPELINE_INFO_ARM: Self = Self(1_000_507_009);
    pub const DATA_GRAPH_PIPELINE_COMPILER_CONTROL_CREATE_INFO_ARM: Self = Self(1_000_507_010);
    pub const DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM: Self =
        Self(1_000_507_011);
    pub const DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM: Self = Self(1_000_507_012);
    pub const DATA_GRAPH_PIPELINE_IDENTIFIER_CREATE_INFO_ARM: Self = Self(1_000_507_013);
    pub const DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM: Self = Self(1_000_507_014);
    pub const DATA_GRAPH_PROCESSING_ENGINE_CREATE_INFO_ARM: Self = Self(1_000_507_016);
    pub const QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_PROPERTIES_ARM: Self = Self(1_000_507_017);
    pub const QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM: Self = Self(1_000_507_018);
    pub const PHYSICAL_DEVICE_QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_INFO_ARM: Self =
        Self(1_000_507_019);
    pub const DATA_GRAPH_PIPELINE_CONSTANT_TENSOR_SEMI_STRUCTURED_SPARSITY_INFO_ARM: Self =
        Self(1_000_507_015);
}
#[doc = "Generated from 'VK_ARM_data_graph'"]
impl TensorUsageFlagsARM {
    pub const DATA_GRAPH: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_ARM_data_graph_instruction_set_tosa'"]
impl StructureType {
    pub const QUEUE_FAMILY_DATA_GRAPH_TOSA_PROPERTIES_ARM: Self = Self(1_000_508_000);
}
#[doc = "Generated from 'VK_QCOM_multiview_per_view_render_areas'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_RENDER_AREAS_FEATURES_QCOM: Self =
        Self(1_000_510_000);
    pub const MULTIVIEW_PER_VIEW_RENDER_AREAS_RENDER_PASS_BEGIN_INFO_QCOM: Self =
        Self(1_000_510_001);
}
#[doc = "Generated from 'VK_KHR_compute_shader_derivatives'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR: Self = Self(1_000_201_000);
    pub const PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_PROPERTIES_KHR: Self = Self(1_000_511_000);
}
#[doc = "Generated from 'VK_KHR_video_decode_av1'"]
impl StructureType {
    pub const VIDEO_DECODE_AV1_CAPABILITIES_KHR: Self = Self(1_000_512_000);
    pub const VIDEO_DECODE_AV1_PICTURE_INFO_KHR: Self = Self(1_000_512_001);
    pub const VIDEO_DECODE_AV1_PROFILE_INFO_KHR: Self = Self(1_000_512_003);
    pub const VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR: Self = Self(1_000_512_004);
    pub const VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR: Self = Self(1_000_512_005);
}
#[doc = "Generated from 'VK_KHR_video_decode_av1'"]
impl VideoCodecOperationFlagsKHR {
    pub const DECODE_AV1: Self = Self(0b100);
}
#[doc = "Generated from 'VK_KHR_video_encode_av1'"]
impl StructureType {
    pub const VIDEO_ENCODE_AV1_CAPABILITIES_KHR: Self = Self(1_000_513_000);
    pub const VIDEO_ENCODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR: Self = Self(1_000_513_001);
    pub const VIDEO_ENCODE_AV1_PICTURE_INFO_KHR: Self = Self(1_000_513_002);
    pub const VIDEO_ENCODE_AV1_DPB_SLOT_INFO_KHR: Self = Self(1_000_513_003);
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_AV1_FEATURES_KHR: Self = Self(1_000_513_004);
    pub const VIDEO_ENCODE_AV1_PROFILE_INFO_KHR: Self = Self(1_000_513_005);
    pub const VIDEO_ENCODE_AV1_RATE_CONTROL_INFO_KHR: Self = Self(1_000_513_006);
    pub const VIDEO_ENCODE_AV1_RATE_CONTROL_LAYER_INFO_KHR: Self = Self(1_000_513_007);
    pub const VIDEO_ENCODE_AV1_QUALITY_LEVEL_PROPERTIES_KHR: Self = Self(1_000_513_008);
    pub const VIDEO_ENCODE_AV1_SESSION_CREATE_INFO_KHR: Self = Self(1_000_513_009);
    pub const VIDEO_ENCODE_AV1_GOP_REMAINING_FRAME_INFO_KHR: Self = Self(1_000_513_010);
}
#[doc = "Generated from 'VK_KHR_video_encode_av1'"]
impl VideoCodecOperationFlagsKHR {
    pub const ENCODE_AV1: Self = Self(0b100_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_decode_vp9'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR: Self = Self(1_000_514_000);
    pub const VIDEO_DECODE_VP9_CAPABILITIES_KHR: Self = Self(1_000_514_001);
    pub const VIDEO_DECODE_VP9_PICTURE_INFO_KHR: Self = Self(1_000_514_002);
    pub const VIDEO_DECODE_VP9_PROFILE_INFO_KHR: Self = Self(1_000_514_003);
}
#[doc = "Generated from 'VK_KHR_video_decode_vp9'"]
impl VideoCodecOperationFlagsKHR {
    pub const DECODE_VP9: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_KHR_video_maintenance1'"]
impl BufferCreateFlags {
    pub const VIDEO_PROFILE_INDEPENDENT_KHR: Self = Self(0b100_0000);
}
#[doc = "Generated from 'VK_KHR_video_maintenance1'"]
impl ImageCreateFlags {
    pub const VIDEO_PROFILE_INDEPENDENT_KHR: Self = Self(0b1_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_maintenance1'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR: Self = Self(1_000_515_000);
    pub const VIDEO_INLINE_QUERY_INFO_KHR: Self = Self(1_000_515_001);
}
#[doc = "Generated from 'VK_KHR_video_maintenance1'"]
impl VideoSessionCreateFlagsKHR {
    pub const INLINE_QUERIES: Self = Self(0b100);
}
#[doc = "Generated from 'VK_NV_per_stage_descriptor_set'"]
impl DescriptorSetLayoutCreateFlags {
    pub const PER_STAGE_NV: Self = Self(0b100_0000);
}
#[doc = "Generated from 'VK_NV_per_stage_descriptor_set'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PER_STAGE_DESCRIPTOR_SET_FEATURES_NV: Self = Self(1_000_516_000);
}
#[doc = "Generated from 'VK_QCOM_image_processing2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_2_FEATURES_QCOM: Self = Self(1_000_518_000);
    pub const PHYSICAL_DEVICE_IMAGE_PROCESSING_2_PROPERTIES_QCOM: Self = Self(1_000_518_001);
    pub const SAMPLER_BLOCK_MATCH_WINDOW_CREATE_INFO_QCOM: Self = Self(1_000_518_002);
}
#[doc = "Generated from 'VK_QCOM_filter_cubic_weights'"]
impl StructureType {
    pub const SAMPLER_CUBIC_WEIGHTS_CREATE_INFO_QCOM: Self = Self(1_000_519_000);
    pub const PHYSICAL_DEVICE_CUBIC_WEIGHTS_FEATURES_QCOM: Self = Self(1_000_519_001);
    pub const BLIT_IMAGE_CUBIC_WEIGHTS_INFO_QCOM: Self = Self(1_000_519_002);
}
#[doc = "Generated from 'VK_QCOM_ycbcr_degamma'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_YCBCR_DEGAMMA_FEATURES_QCOM: Self = Self(1_000_520_000);
    pub const SAMPLER_YCBCR_CONVERSION_YCBCR_DEGAMMA_CREATE_INFO_QCOM: Self = Self(1_000_520_001);
}
#[doc = "Generated from 'VK_QCOM_filter_cubic_clamp'"]
impl SamplerReductionMode {
    pub const WEIGHTED_AVERAGE_RANGECLAMP_QCOM: Self = Self(1_000_521_000);
}
#[doc = "Generated from 'VK_QCOM_filter_cubic_clamp'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_CUBIC_CLAMP_FEATURES_QCOM: Self = Self(1_000_521_000);
}
#[doc = "Generated from 'VK_EXT_attachment_feedback_loop_dynamic_state'"]
impl DynamicState {
    pub const ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT: Self = Self(1_000_524_000);
}
#[doc = "Generated from 'VK_EXT_attachment_feedback_loop_dynamic_state'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_FEATURES_EXT: Self =
        Self(1_000_524_000);
}
#[doc = "Generated from 'VK_KHR_vertex_attribute_divisor'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES;
    pub const PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR: Self =
        Self::PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO;
    pub const PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES;
}
#[doc = "Generated from 'VK_KHR_load_store_op_none'"]
impl AttachmentLoadOp {
    pub const NONE_KHR: Self = Self::NONE;
}
#[doc = "Generated from 'VK_KHR_unified_image_layouts'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR: Self = Self(1_000_527_000);
    pub const ATTACHMENT_FEEDBACK_LOOP_INFO_EXT: Self = Self(1_000_527_001);
}
#[doc = "Generated from 'VK_KHR_shader_float_controls2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES;
}
#[doc = "Generated from 'VK_QNX_external_memory_screen_buffer'"]
impl ExternalMemoryHandleTypeFlags {
    pub const SCREEN_BUFFER_QNX: Self = Self(0b100_0000_0000_0000);
}
#[doc = "Generated from 'VK_QNX_external_memory_screen_buffer'"]
impl StructureType {
    pub const SCREEN_BUFFER_PROPERTIES_QNX: Self = Self(1_000_529_000);
    pub const SCREEN_BUFFER_FORMAT_PROPERTIES_QNX: Self = Self(1_000_529_001);
    pub const IMPORT_SCREEN_BUFFER_INFO_QNX: Self = Self(1_000_529_002);
    pub const EXTERNAL_FORMAT_QNX: Self = Self(1_000_529_003);
    pub const PHYSICAL_DEVICE_EXTERNAL_MEMORY_SCREEN_BUFFER_FEATURES_QNX: Self =
        Self(1_000_529_004);
}
#[doc = "Generated from 'VK_MSFT_layered_driver'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_LAYERED_DRIVER_PROPERTIES_MSFT: Self = Self(1_000_530_000);
}
#[doc = "Generated from 'VK_KHR_index_type_uint8'"]
impl IndexType {
    pub const UINT8_KHR: Self = Self::UINT8;
}
#[doc = "Generated from 'VK_KHR_index_type_uint8'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES;
}
#[doc = "Generated from 'VK_KHR_line_rasterization'"]
impl DynamicState {
    pub const LINE_STIPPLE_KHR: Self = Self::LINE_STIPPLE;
}
#[doc = "Generated from 'VK_KHR_line_rasterization'"]
impl LineRasterizationMode {
    pub const DEFAULT_KHR: Self = Self::DEFAULT;
    pub const RECTANGULAR_KHR: Self = Self::RECTANGULAR;
    pub const BRESENHAM_KHR: Self = Self::BRESENHAM;
    pub const RECTANGULAR_SMOOTH_KHR: Self = Self::RECTANGULAR_SMOOTH;
}
#[doc = "Generated from 'VK_KHR_line_rasterization'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES;
    pub const PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_KHR: Self =
        Self::PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO;
    pub const PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES;
}
#[doc = "Generated from 'VK_KHR_calibrated_timestamps'"]
impl StructureType {
    pub const CALIBRATED_TIMESTAMP_INFO_KHR: Self = Self(1_000_184_000);
}
#[doc = "Generated from 'VK_KHR_shader_expect_assume'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES;
}
#[doc = "Generated from 'VK_KHR_maintenance6'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES_KHR: Self =
        Self::PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES;
    pub const PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES_KHR: Self =
        Self::PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES;
    pub const BIND_MEMORY_STATUS_KHR: Self = Self::BIND_MEMORY_STATUS;
    pub const BIND_DESCRIPTOR_SETS_INFO_KHR: Self = Self::BIND_DESCRIPTOR_SETS_INFO;
    pub const PUSH_CONSTANTS_INFO_KHR: Self = Self::PUSH_CONSTANTS_INFO;
    pub const PUSH_DESCRIPTOR_SET_INFO_KHR: Self = Self::PUSH_DESCRIPTOR_SET_INFO;
    pub const PUSH_DESCRIPTOR_SET_WITH_TEMPLATE_INFO_KHR: Self =
        Self::PUSH_DESCRIPTOR_SET_WITH_TEMPLATE_INFO;
    pub const SET_DESCRIPTOR_BUFFER_OFFSETS_INFO_EXT: Self = Self(1_000_545_007);
    pub const BIND_DESCRIPTOR_BUFFER_EMBEDDED_SAMPLERS_INFO_EXT: Self = Self(1_000_545_008);
}
#[doc = "Generated from 'VK_NV_descriptor_pool_overallocation'"]
impl DescriptorPoolCreateFlags {
    pub const ALLOW_OVERALLOCATION_SETS_NV: Self = Self(0b1000);
    pub const ALLOW_OVERALLOCATION_POOLS_NV: Self = Self(0b1_0000);
}
#[doc = "Generated from 'VK_NV_descriptor_pool_overallocation'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DESCRIPTOR_POOL_OVERALLOCATION_FEATURES_NV: Self =
        Self(1_000_546_000);
}
#[doc = "Generated from 'VK_QCOM_tile_memory_heap'"]
impl BufferUsageFlags {
    pub const TILE_MEMORY_QCOM: Self = Self(0b1000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_QCOM_tile_memory_heap'"]
impl BufferUsageFlags2 {
    pub const TILE_MEMORY_QCOM: Self = Self(0b1000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_QCOM_tile_memory_heap'"]
impl ImageUsageFlags {
    pub const TILE_MEMORY_QCOM: Self = Self(0b1000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_QCOM_tile_memory_heap'"]
impl MemoryHeapFlags {
    pub const TILE_MEMORY_QCOM: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_QCOM_tile_memory_heap'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_TILE_MEMORY_HEAP_FEATURES_QCOM: Self = Self(1_000_547_000);
    pub const PHYSICAL_DEVICE_TILE_MEMORY_HEAP_PROPERTIES_QCOM: Self = Self(1_000_547_001);
    pub const TILE_MEMORY_REQUIREMENTS_QCOM: Self = Self(1_000_547_002);
    pub const TILE_MEMORY_BIND_INFO_QCOM: Self = Self(1_000_547_003);
    pub const TILE_MEMORY_SIZE_INFO_QCOM: Self = Self(1_000_547_004);
}
#[doc = "Generated from 'VK_KHR_copy_memory_indirect'"]
impl FormatFeatureFlags2 {
    pub const COPY_IMAGE_INDIRECT_DST_KHR: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_copy_memory_indirect'"]
impl PipelineStageFlags2 {
    pub const COPY_INDIRECT_KHR: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_copy_memory_indirect'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_KHR: Self = Self(1_000_549_000);
    pub const PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_PROPERTIES_KHR: Self = Self(1_000_426_001);
    pub const COPY_MEMORY_INDIRECT_INFO_KHR: Self = Self(1_000_549_002);
    pub const COPY_MEMORY_TO_IMAGE_INDIRECT_INFO_KHR: Self = Self(1_000_549_003);
}
#[doc = "Generated from 'VK_EXT_memory_decompression'"]
impl AccessFlags2 {
    pub const MEMORY_DECOMPRESSION_READ_EXT: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const MEMORY_DECOMPRESSION_WRITE_EXT: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_memory_decompression'"]
impl BufferUsageFlags2 {
    pub const MEMORY_DECOMPRESSION_EXT: Self = Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_memory_decompression'"]
impl PipelineStageFlags2 {
    pub const MEMORY_DECOMPRESSION_EXT: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_memory_decompression'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_EXT: Self = Self(1_000_427_000);
    pub const PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_EXT: Self = Self(1_000_427_001);
    pub const DECOMPRESS_MEMORY_INFO_EXT: Self = Self(1_000_550_002);
}
#[doc = "Generated from 'VK_NV_display_stereo'"]
impl StructureType {
    pub const DISPLAY_SURFACE_STEREO_CREATE_INFO_NV: Self = Self(1_000_551_000);
    pub const DISPLAY_MODE_STEREO_PROPERTIES_NV: Self = Self(1_000_551_001);
}
#[doc = "Generated from 'VK_KHR_video_encode_intra_refresh'"]
impl StructureType {
    pub const VIDEO_ENCODE_INTRA_REFRESH_CAPABILITIES_KHR: Self = Self(1_000_552_000);
    pub const VIDEO_ENCODE_SESSION_INTRA_REFRESH_CREATE_INFO_KHR: Self = Self(1_000_552_001);
    pub const VIDEO_ENCODE_INTRA_REFRESH_INFO_KHR: Self = Self(1_000_552_002);
    pub const VIDEO_REFERENCE_INTRA_REFRESH_INFO_KHR: Self = Self(1_000_552_003);
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_INTRA_REFRESH_FEATURES_KHR: Self = Self(1_000_552_004);
}
#[doc = "Generated from 'VK_KHR_video_encode_intra_refresh'"]
impl VideoEncodeAV1CapabilityFlagsKHR {
    pub const COMPOUND_PREDICTION_INTRA_REFRESH: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_intra_refresh'"]
impl VideoEncodeFlagsKHR {
    pub const INTRA_REFRESH: Self = Self(0b100);
}
#[doc = "Generated from 'VK_KHR_video_encode_intra_refresh'"]
impl VideoEncodeH264CapabilityFlagsKHR {
    pub const B_PICTURE_INTRA_REFRESH: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_intra_refresh'"]
impl VideoEncodeH265CapabilityFlagsKHR {
    pub const B_PICTURE_INTRA_REFRESH: Self = Self(0b1000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl FormatFeatureFlags2 {
    pub const VIDEO_ENCODE_QUANTIZATION_DELTA_MAP_KHR: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const VIDEO_ENCODE_EMPHASIS_MAP_KHR: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl ImageLayout {
    pub const VIDEO_ENCODE_QUANTIZATION_MAP_KHR: Self = Self(1_000_553_000);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl ImageUsageFlags {
    pub const VIDEO_ENCODE_QUANTIZATION_DELTA_MAP_KHR: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000);
    pub const VIDEO_ENCODE_EMPHASIS_MAP_KHR: Self = Self(0b100_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl StructureType {
    pub const VIDEO_ENCODE_QUANTIZATION_MAP_CAPABILITIES_KHR: Self = Self(1_000_553_000);
    pub const VIDEO_FORMAT_QUANTIZATION_MAP_PROPERTIES_KHR: Self = Self(1_000_553_001);
    pub const VIDEO_ENCODE_QUANTIZATION_MAP_INFO_KHR: Self = Self(1_000_553_002);
    pub const VIDEO_ENCODE_QUANTIZATION_MAP_SESSION_PARAMETERS_CREATE_INFO_KHR: Self =
        Self(1_000_553_005);
    pub const PHYSICAL_DEVICE_VIDEO_ENCODE_QUANTIZATION_MAP_FEATURES_KHR: Self =
        Self(1_000_553_009);
    pub const VIDEO_ENCODE_H264_QUANTIZATION_MAP_CAPABILITIES_KHR: Self = Self(1_000_553_003);
    pub const VIDEO_ENCODE_H265_QUANTIZATION_MAP_CAPABILITIES_KHR: Self = Self(1_000_553_004);
    pub const VIDEO_FORMAT_H265_QUANTIZATION_MAP_PROPERTIES_KHR: Self = Self(1_000_553_006);
    pub const VIDEO_ENCODE_AV1_QUANTIZATION_MAP_CAPABILITIES_KHR: Self = Self(1_000_553_007);
    pub const VIDEO_FORMAT_AV1_QUANTIZATION_MAP_PROPERTIES_KHR: Self = Self(1_000_553_008);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl VideoEncodeCapabilityFlagsKHR {
    pub const QUANTIZATION_DELTA_MAP: Self = Self(0b100);
    pub const EMPHASIS_MAP: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl VideoEncodeFlagsKHR {
    pub const WITH_QUANTIZATION_DELTA_MAP: Self = Self(0b1);
    pub const WITH_EMPHASIS_MAP: Self = Self(0b10);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl VideoEncodeH264CapabilityFlagsKHR {
    pub const MB_QP_DIFF_WRAPAROUND: Self = Self(0b10_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl VideoEncodeH265CapabilityFlagsKHR {
    pub const CU_QP_DIFF_WRAPAROUND: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl VideoSessionCreateFlagsKHR {
    pub const ALLOW_ENCODE_QUANTIZATION_DELTA_MAP: Self = Self(0b1000);
    pub const ALLOW_ENCODE_EMPHASIS_MAP: Self = Self(0b1_0000);
}
#[doc = "Generated from 'VK_KHR_video_encode_quantization_map'"]
impl VideoSessionParametersCreateFlagsKHR {
    pub const QUANTIZATION_MAP_COMPATIBLE: Self = Self(0b1);
}
#[doc = "Generated from 'VK_NV_raw_access_chains'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAW_ACCESS_CHAINS_FEATURES_NV: Self = Self(1_000_555_000);
}
#[doc = "Generated from 'VK_NV_external_compute_queue'"]
impl ObjectType {
    pub const EXTERNAL_COMPUTE_QUEUE_NV: Self = Self(1_000_556_000);
}
#[doc = "Generated from 'VK_NV_external_compute_queue'"]
impl StructureType {
    pub const EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV: Self = Self(1_000_556_000);
    pub const EXTERNAL_COMPUTE_QUEUE_CREATE_INFO_NV: Self = Self(1_000_556_001);
    pub const EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS_NV: Self = Self(1_000_556_002);
    pub const PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES_NV: Self = Self(1_000_556_003);
}
#[doc = "Generated from 'VK_KHR_shader_relaxed_extended_instruction'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_RELAXED_EXTENDED_INSTRUCTION_FEATURES_KHR: Self =
        Self(1_000_558_000);
}
#[doc = "Generated from 'VK_NV_command_buffer_inheritance'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COMMAND_BUFFER_INHERITANCE_FEATURES_NV: Self = Self(1_000_559_000);
}
#[doc = "Generated from 'VK_KHR_maintenance7'"]
impl RenderingFlags {
    #[doc = "Promoted from extension 452"]
    pub const CONTENTS_INLINE_KHR: Self = Self(0b1_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance7'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_7_FEATURES_KHR: Self = Self(1_000_562_000);
    pub const PHYSICAL_DEVICE_MAINTENANCE_7_PROPERTIES_KHR: Self = Self(1_000_562_001);
    pub const PHYSICAL_DEVICE_LAYERED_API_PROPERTIES_LIST_KHR: Self = Self(1_000_562_002);
    pub const PHYSICAL_DEVICE_LAYERED_API_PROPERTIES_KHR: Self = Self(1_000_562_003);
    pub const PHYSICAL_DEVICE_LAYERED_API_VULKAN_PROPERTIES_KHR: Self = Self(1_000_562_004);
}
#[doc = "Generated from 'VK_KHR_maintenance7'"]
impl SubpassContents {
    pub const INLINE_AND_SECONDARY_COMMAND_BUFFERS_KHR: Self = Self(1_000_451_000);
}
#[doc = "Generated from 'VK_NV_shader_atomic_float16_vector'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT16_VECTOR_FEATURES_NV: Self = Self(1_000_563_000);
}
#[doc = "Generated from 'VK_EXT_shader_replicated_composites'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT: Self = Self(1_000_564_000);
}
#[doc = "Generated from 'VK_EXT_shader_float8'"]
impl ComponentTypeKHR {
    pub const FLOAT8_E4M3_EXT: Self = Self(1_000_491_002);
    pub const FLOAT8_E5M2_EXT: Self = Self(1_000_491_003);
}
#[doc = "Generated from 'VK_EXT_shader_float8'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT: Self = Self(1_000_567_000);
}
#[doc = "Generated from 'VK_NV_ray_tracing_validation'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV: Self = Self(1_000_568_000);
}
#[doc = "Generated from 'VK_NV_cluster_acceleration_structure'"]
impl BuildAccelerationStructureFlagsKHR {
    pub const ALLOW_CLUSTER_OPACITY_MICROMAPS_NV: Self = Self(0b1_0000_0000_0000);
}
#[doc = "Generated from 'VK_NV_cluster_acceleration_structure'"]
impl OpacityMicromapSpecialIndexKHR {
    pub const CLUSTER_GEOMETRY_DISABLE_OPACITY_MICROMAP_NV: Self = Self(-5);
}
#[doc = "Generated from 'VK_NV_cluster_acceleration_structure'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV: Self =
        Self(1_000_569_000);
    pub const PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV: Self =
        Self(1_000_569_001);
    pub const CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV: Self =
        Self(1_000_569_002);
    pub const CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV: Self = Self(1_000_569_003);
    pub const CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV: Self = Self(1_000_569_004);
    pub const CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV: Self = Self(1_000_569_005);
    pub const CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV: Self = Self(1_000_569_006);
    pub const RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV: Self =
        Self(1_000_569_007);
}
#[doc = "Generated from 'VK_NV_partitioned_acceleration_structure'"]
impl DescriptorType {
    pub const PARTITIONED_ACCELERATION_STRUCTURE_NV: Self = Self(1_000_570_000);
}
#[doc = "Generated from 'VK_NV_partitioned_acceleration_structure'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV: Self =
        Self(1_000_570_000);
    pub const PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES_NV: Self =
        Self(1_000_570_001);
    pub const WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE_NV: Self =
        Self(1_000_570_002);
    pub const PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV: Self = Self(1_000_570_003);
    pub const BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV: Self = Self(1_000_570_004);
    pub const PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV: Self = Self(1_000_570_005);
}
#[doc = "Generated from 'VK_EXT_device_generated_commands'"]
impl AccessFlags {
    pub const COMMAND_PREPROCESS_READ_EXT: Self = Self(0b10_0000_0000_0000_0000);
    pub const COMMAND_PREPROCESS_WRITE_EXT: Self = Self(0b100_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_device_generated_commands'"]
impl BufferUsageFlags2 {
    pub const PREPROCESS_BUFFER_EXT: Self = Self(0b1000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_device_generated_commands'"]
impl ObjectType {
    pub const INDIRECT_COMMANDS_LAYOUT_EXT: Self = Self(1_000_572_000);
    pub const INDIRECT_EXECUTION_SET_EXT: Self = Self(1_000_572_001);
}
#[doc = "Generated from 'VK_EXT_device_generated_commands'"]
impl PipelineCreateFlags2 {
    pub const INDIRECT_BINDABLE_EXT: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_device_generated_commands'"]
impl PipelineStageFlags {
    pub const COMMAND_PREPROCESS_EXT: Self = Self(0b10_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_device_generated_commands'"]
impl ShaderCreateFlagsEXT {
    pub const INDIRECT_BINDABLE: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_EXT_device_generated_commands'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT: Self = Self(1_000_572_000);
    pub const PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT: Self = Self(1_000_572_001);
    pub const GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT: Self = Self(1_000_572_002);
    pub const INDIRECT_EXECUTION_SET_CREATE_INFO_EXT: Self = Self(1_000_572_003);
    pub const GENERATED_COMMANDS_INFO_EXT: Self = Self(1_000_572_004);
    pub const INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT: Self = Self(1_000_572_006);
    pub const INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT: Self = Self(1_000_572_007);
    pub const WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT: Self = Self(1_000_572_008);
    pub const WRITE_INDIRECT_EXECUTION_SET_SHADER_EXT: Self = Self(1_000_572_009);
    pub const INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT: Self = Self(1_000_572_010);
    pub const INDIRECT_EXECUTION_SET_SHADER_INFO_EXT: Self = Self(1_000_572_011);
    pub const INDIRECT_EXECUTION_SET_SHADER_LAYOUT_INFO_EXT: Self = Self(1_000_572_012);
    pub const GENERATED_COMMANDS_PIPELINE_INFO_EXT: Self = Self(1_000_572_013);
    pub const GENERATED_COMMANDS_SHADER_INFO_EXT: Self = Self(1_000_572_014);
}
#[doc = "Generated from 'VK_KHR_device_fault'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FAULT_FEATURES_KHR: Self = Self(1_000_573_000);
    pub const PHYSICAL_DEVICE_FAULT_PROPERTIES_KHR: Self = Self(1_000_573_001);
    pub const DEVICE_FAULT_INFO_KHR: Self = Self(1_000_573_002);
    pub const DEVICE_FAULT_DEBUG_INFO_KHR: Self = Self(1_000_573_003);
}
#[doc = "Generated from 'VK_KHR_maintenance8'"]
impl DependencyFlags {
    pub const QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_KHR: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance8'"]
impl PipelineCacheCreateFlags {
    pub const INTERNALLY_SYNCHRONIZED_MERGE_KHR: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_KHR_maintenance8'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR: Self = Self(1_000_574_000);
    pub const MEMORY_BARRIER_ACCESS_FLAGS_3_KHR: Self = Self(1_000_574_002);
}
#[doc = "Generated from 'VK_MESA_image_alignment_control'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_IMAGE_ALIGNMENT_CONTROL_FEATURES_MESA: Self = Self(1_000_575_000);
    pub const PHYSICAL_DEVICE_IMAGE_ALIGNMENT_CONTROL_PROPERTIES_MESA: Self = Self(1_000_575_001);
    pub const IMAGE_ALIGNMENT_CONTROL_CREATE_INFO_MESA: Self = Self(1_000_575_002);
}
#[doc = "Generated from 'VK_KHR_shader_fma'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR: Self = Self(1_000_579_000);
}
#[doc = "Generated from 'VK_NV_push_constant_bank'"]
impl StructureType {
    pub const PUSH_CONSTANT_BANK_INFO_NV: Self = Self(1_000_580_000);
    pub const PHYSICAL_DEVICE_PUSH_CONSTANT_BANK_FEATURES_NV: Self = Self(1_000_580_001);
    pub const PHYSICAL_DEVICE_PUSH_CONSTANT_BANK_PROPERTIES_NV: Self = Self(1_000_580_002);
}
#[doc = "Generated from 'VK_EXT_ray_tracing_invocation_reorder'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT: Self =
        Self(1_000_581_000);
    pub const PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT: Self =
        Self(1_000_581_001);
}
#[doc = "Generated from 'VK_EXT_depth_clamp_control'"]
impl DynamicState {
    pub const DEPTH_CLAMP_RANGE_EXT: Self = Self(1_000_582_000);
}
#[doc = "Generated from 'VK_EXT_depth_clamp_control'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEPTH_CLAMP_CONTROL_FEATURES_EXT: Self = Self(1_000_582_000);
    pub const PIPELINE_VIEWPORT_DEPTH_CLAMP_CONTROL_CREATE_INFO_EXT: Self = Self(1_000_582_001);
}
#[doc = "Generated from 'VK_KHR_maintenance9'"]
impl DependencyFlags {
    pub const ASYMMETRIC_EVENT_KHR: Self = Self(0b100_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance9'"]
impl QueryPoolCreateFlags {
    pub const RESET_KHR: Self = Self(0b1);
}
#[doc = "Generated from 'VK_KHR_maintenance9'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR: Self = Self(1_000_584_000);
    pub const PHYSICAL_DEVICE_MAINTENANCE_9_PROPERTIES_KHR: Self = Self(1_000_584_001);
    pub const QUEUE_FAMILY_OWNERSHIP_TRANSFER_PROPERTIES_KHR: Self = Self(1_000_584_002);
}
#[doc = "Generated from 'VK_KHR_video_maintenance2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES_KHR: Self = Self(1_000_586_000);
    pub const VIDEO_DECODE_H264_INLINE_SESSION_PARAMETERS_INFO_KHR: Self = Self(1_000_586_001);
    pub const VIDEO_DECODE_H265_INLINE_SESSION_PARAMETERS_INFO_KHR: Self = Self(1_000_586_002);
    pub const VIDEO_DECODE_AV1_INLINE_SESSION_PARAMETERS_INFO_KHR: Self = Self(1_000_586_003);
}
#[doc = "Generated from 'VK_KHR_video_maintenance2'"]
impl VideoSessionCreateFlagsKHR {
    pub const INLINE_SESSION_PARAMETERS: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_OHOS_surface'"]
impl StructureType {
    pub const SURFACE_CREATE_INFO_OHOS: Self = Self(1_000_685_000);
}
#[doc = "Generated from 'VK_HUAWEI_hdr_vivid'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_HDR_VIVID_FEATURES_HUAWEI: Self = Self(1_000_590_000);
    pub const HDR_VIVID_DYNAMIC_METADATA_HUAWEI: Self = Self(1_000_590_001);
}
#[doc = "Generated from 'VK_NV_cooperative_matrix2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV: Self = Self(1_000_593_000);
    pub const COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV: Self = Self(1_000_593_001);
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV: Self = Self(1_000_593_002);
}
#[doc = "Generated from 'VK_ARM_pipeline_opacity_micromap'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PIPELINE_OPACITY_MICROMAP_FEATURES_ARM: Self = Self(1_000_596_000);
}
#[doc = "Generated from 'VK_EXT_external_memory_metal'"]
impl ExternalMemoryHandleTypeFlags {
    pub const MTLBUFFER_EXT: Self = Self(0b1_0000_0000_0000_0000);
    pub const MTLTEXTURE_EXT: Self = Self(0b10_0000_0000_0000_0000);
    pub const MTLHEAP_EXT: Self = Self(0b100_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_external_memory_metal'"]
impl StructureType {
    pub const IMPORT_MEMORY_METAL_HANDLE_INFO_EXT: Self = Self(1_000_602_000);
    pub const MEMORY_METAL_HANDLE_PROPERTIES_EXT: Self = Self(1_000_602_001);
    pub const MEMORY_GET_METAL_HANDLE_INFO_EXT: Self = Self(1_000_602_002);
}
#[doc = "Generated from 'VK_KHR_depth_clamp_zero_one'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_KHR: Self = Self(1_000_421_000);
}
#[doc = "Generated from 'VK_ARM_performance_counters_by_region'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PERFORMANCE_COUNTERS_BY_REGION_FEATURES_ARM: Self =
        Self(1_000_605_000);
    pub const PHYSICAL_DEVICE_PERFORMANCE_COUNTERS_BY_REGION_PROPERTIES_ARM: Self =
        Self(1_000_605_001);
    pub const PERFORMANCE_COUNTER_ARM: Self = Self(1_000_605_002);
    pub const PERFORMANCE_COUNTER_DESCRIPTION_ARM: Self = Self(1_000_605_003);
    pub const RENDER_PASS_PERFORMANCE_COUNTERS_BY_REGION_BEGIN_INFO_ARM: Self = Self(1_000_605_004);
}
#[doc = "Generated from 'VK_ARM_shader_instrumentation'"]
impl ObjectType {
    pub const SHADER_INSTRUMENTATION_ARM: Self = Self(1_000_607_000);
}
#[doc = "Generated from 'VK_ARM_shader_instrumentation'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_INSTRUMENTATION_FEATURES_ARM: Self = Self(1_000_607_000);
    pub const PHYSICAL_DEVICE_SHADER_INSTRUMENTATION_PROPERTIES_ARM: Self = Self(1_000_607_001);
    pub const SHADER_INSTRUMENTATION_CREATE_INFO_ARM: Self = Self(1_000_607_002);
    pub const SHADER_INSTRUMENTATION_METRIC_DESCRIPTION_ARM: Self = Self(1_000_607_003);
}
#[doc = "Generated from 'VK_EXT_vertex_attribute_robustness'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_ROBUSTNESS_FEATURES_EXT: Self = Self(1_000_608_000);
}
#[doc = "Generated from 'VK_ARM_format_pack'"]
impl Format {
    pub const R10X6_UINT_PACK16_ARM: Self = Self(1_000_609_000);
    pub const R10X6G10X6_UINT_2PACK16_ARM: Self = Self(1_000_609_001);
    pub const R10X6G10X6B10X6A10X6_UINT_4PACK16_ARM: Self = Self(1_000_609_002);
    pub const R12X4_UINT_PACK16_ARM: Self = Self(1_000_609_003);
    pub const R12X4G12X4_UINT_2PACK16_ARM: Self = Self(1_000_609_004);
    pub const R12X4G12X4B12X4A12X4_UINT_4PACK16_ARM: Self = Self(1_000_609_005);
    pub const R14X2_UINT_PACK16_ARM: Self = Self(1_000_609_006);
    pub const R14X2G14X2_UINT_2PACK16_ARM: Self = Self(1_000_609_007);
    pub const R14X2G14X2B14X2A14X2_UINT_4PACK16_ARM: Self = Self(1_000_609_008);
    pub const R14X2_UNORM_PACK16_ARM: Self = Self(1_000_609_009);
    pub const R14X2G14X2_UNORM_2PACK16_ARM: Self = Self(1_000_609_010);
    pub const R14X2G14X2B14X2A14X2_UNORM_4PACK16_ARM: Self = Self(1_000_609_011);
    pub const G14X2_B14X2R14X2_2PLANE_420_UNORM_3PACK16_ARM: Self = Self(1_000_609_012);
    pub const G14X2_B14X2R14X2_2PLANE_422_UNORM_3PACK16_ARM: Self = Self(1_000_609_013);
}
#[doc = "Generated from 'VK_ARM_format_pack'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FORMAT_PACK_FEATURES_ARM: Self = Self(1_000_609_000);
}
#[doc = "Generated from 'VK_VALVE_fragment_density_map_layered'"]
impl PipelineCreateFlags2 {
    pub const PER_LAYER_FRAGMENT_DENSITY_VALVE: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_VALVE_fragment_density_map_layered'"]
impl RenderPassCreateFlags {
    pub const PER_LAYER_FRAGMENT_DENSITY_VALVE: Self = Self(0b100);
}
#[doc = "Generated from 'VK_VALVE_fragment_density_map_layered'"]
impl RenderingFlags {
    pub const PER_LAYER_FRAGMENT_DENSITY_VALVE: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_VALVE_fragment_density_map_layered'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_LAYERED_FEATURES_VALVE: Self =
        Self(1_000_611_000);
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_LAYERED_PROPERTIES_VALVE: Self =
        Self(1_000_611_001);
    pub const PIPELINE_FRAGMENT_DENSITY_MAP_LAYERED_CREATE_INFO_VALVE: Self = Self(1_000_611_002);
}
#[doc = "Generated from 'VK_KHR_robustness2'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR: Self = Self(1_000_286_000);
    pub const PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_KHR: Self = Self(1_000_286_001);
}
#[doc = "Generated from 'VK_NV_present_metering'"]
impl StructureType {
    pub const SET_PRESENT_CONFIG_NV: Self = Self(1_000_613_000);
    pub const PHYSICAL_DEVICE_PRESENT_METERING_FEATURES_NV: Self = Self(1_000_613_001);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map_offset'"]
impl ImageCreateFlags {
    pub const FRAGMENT_DENSITY_MAP_OFFSET_EXT: Self = Self(0b1000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_fragment_density_map_offset'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_EXT: Self = Self(1_000_425_000);
    pub const PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_PROPERTIES_EXT: Self =
        Self(1_000_425_001);
    pub const RENDER_PASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_EXT: Self = Self(1_000_425_002);
    pub const RENDERING_END_INFO_EXT: Self = Self::RENDERING_END_INFO_KHR;
}
#[doc = "Generated from 'VK_EXT_zero_initialize_device_memory'"]
impl ImageLayout {
    pub const ZERO_INITIALIZED_EXT: Self = Self(1_000_620_000);
}
#[doc = "Generated from 'VK_EXT_zero_initialize_device_memory'"]
impl MemoryAllocateFlags {
    pub const ZERO_INITIALIZE_EXT: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_EXT_zero_initialize_device_memory'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_ZERO_INITIALIZE_DEVICE_MEMORY_FEATURES_EXT: Self =
        Self(1_000_620_000);
}
#[doc = "Generated from 'VK_KHR_present_mode_fifo_latest_ready'"]
impl PresentModeKHR {
    pub const FIFO_LATEST_READY: Self = Self(1_000_361_000);
}
#[doc = "Generated from 'VK_KHR_present_mode_fifo_latest_ready'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR: Self =
        Self(1_000_361_000);
}
#[doc = "Generated from 'VK_KHR_opacity_micromap'"]
impl AccelerationStructureTypeKHR {
    pub const OPACITY_MICROMAP: Self = Self(1_000_623_000);
}
#[doc = "Generated from 'VK_KHR_opacity_micromap'"]
impl BuildAccelerationStructureFlagsKHR {
    pub const ALLOW_OPACITY_MICROMAP_UPDATE: Self = Self(0b100_0000);
    pub const ALLOW_DISABLE_OPACITY_MICROMAPS: Self = Self(0b1000_0000);
    pub const MICROMAP_LOSSY: Self = Self(0b100_0000_0000);
}
#[doc = "Generated from 'VK_KHR_opacity_micromap'"]
impl GeometryInstanceFlagsKHR {
    pub const FORCE_OPACITY_MICROMAP_2_STATE: Self = Self(0b1_0000);
    pub const DISABLE_OPACITY_MICROMAPS: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_KHR_opacity_micromap'"]
impl GeometryTypeKHR {
    pub const MICROMAP: Self = Self(1_000_623_000);
}
#[doc = "Generated from 'VK_KHR_opacity_micromap'"]
impl PipelineCreateFlags {
    pub const RAY_TRACING_OPACITY_MICROMAP_KHR: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_opacity_micromap'"]
impl PipelineCreateFlags2 {
    pub const RAY_TRACING_OPACITY_MICROMAP_KHR: Self = Self(0b1_0000_0000_0000_0000_0000_0000);
    pub const OPACITY_MICROMAP_DISALLOW_MIXED_SPECIAL_INDEX_KHR: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_opacity_micromap'"]
impl ShaderCreateFlagsEXT {
    pub const OPACITY_MICROMAP_DISALLOW_MIXED_SPECIAL_INDEX: Self = Self(0b1_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_opacity_micromap'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_KHR: Self = Self(1_000_623_000);
    pub const PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR: Self = Self(1_000_623_001);
    pub const ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR: Self = Self(1_000_623_002);
    pub const ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_KHR: Self = Self(1_000_623_003);
}
#[doc = "Generated from 'VK_EXT_shader_64bit_indexing'"]
impl PipelineCreateFlags2 {
    pub const TYPE_64_INDEXING_EXT: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_shader_64bit_indexing'"]
impl ShaderCreateFlagsEXT {
    pub const TYPE_64_INDEXING: Self = Self(0b1000_0000_0000_0000);
}
#[doc = "Generated from 'VK_EXT_shader_64bit_indexing'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_64_INDEXING_FEATURES_EXT: Self = Self(1_000_627_000);
}
#[doc = "Generated from 'VK_EXT_custom_resolve'"]
impl RenderingFlags {
    pub const FRAGMENT_REGION_EXT: Self = Self(0b100_0000);
    pub const CUSTOM_RESOLVE_EXT: Self = Self(0b1000_0000);
}
#[doc = "Generated from 'VK_EXT_custom_resolve'"]
impl ResolveModeFlags {
    pub const CUSTOM_EXT: Self = Self(0b10_0000);
}
#[doc = "Generated from 'VK_EXT_custom_resolve'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_CUSTOM_RESOLVE_FEATURES_EXT: Self = Self(1_000_628_000);
    pub const BEGIN_CUSTOM_RESOLVE_INFO_EXT: Self = Self(1_000_628_001);
    pub const CUSTOM_RESOLVE_CREATE_INFO_EXT: Self = Self(1_000_628_002);
}
#[doc = "Generated from 'VK_EXT_custom_resolve'"]
impl SubpassDescriptionFlags {
    pub const FRAGMENT_REGION_EXT: Self = Self(0b100);
    pub const CUSTOM_RESOLVE_EXT: Self = Self(0b1000);
}
#[doc = "Generated from 'VK_QCOM_data_graph_model'"]
impl PhysicalDeviceDataGraphOperationTypeARM {
    pub const NEURAL_MODEL_QCOM: Self = Self(1_000_629_000);
    pub const BUILTIN_MODEL_QCOM: Self = Self(1_000_629_001);
}
#[doc = "Generated from 'VK_QCOM_data_graph_model'"]
impl PhysicalDeviceDataGraphProcessingEngineTypeARM {
    pub const NEURAL_QCOM: Self = Self(1_000_629_000);
    pub const COMPUTE_QCOM: Self = Self(1_000_629_001);
}
#[doc = "Generated from 'VK_QCOM_data_graph_model'"]
impl PipelineCacheHeaderVersion {
    pub const DATA_GRAPH_QCOM: Self = Self(1_000_629_000);
}
#[doc = "Generated from 'VK_QCOM_data_graph_model'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DATA_GRAPH_MODEL_FEATURES_QCOM: Self = Self(1_000_629_000);
    pub const DATA_GRAPH_PIPELINE_BUILTIN_MODEL_CREATE_INFO_QCOM: Self = Self(1_000_629_001);
}
#[doc = "Generated from 'VK_KHR_maintenance10'"]
impl AttachmentDescriptionFlags {
    pub const RESOLVE_SKIP_TRANSFER_FUNCTION_KHR: Self = Self(0b10);
    pub const RESOLVE_ENABLE_TRANSFER_FUNCTION_KHR: Self = Self(0b100);
}
#[doc = "Generated from 'VK_KHR_maintenance10'"]
impl FormatFeatureFlags2 {
    pub const DEPTH_COPY_ON_COMPUTE_QUEUE_KHR: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const DEPTH_COPY_ON_TRANSFER_QUEUE_KHR: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const STENCIL_COPY_ON_COMPUTE_QUEUE_KHR: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const STENCIL_COPY_ON_TRANSFER_QUEUE_KHR: Self =
        Self(0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance10'"]
impl RenderingAttachmentFlagsKHR {
    pub const INPUT_ATTACHMENT_FEEDBACK: Self = Self(0b1);
    pub const RESOLVE_SKIP_TRANSFER_FUNCTION: Self = Self(0b10);
    pub const RESOLVE_ENABLE_TRANSFER_FUNCTION: Self = Self(0b100);
}
#[doc = "Generated from 'VK_KHR_maintenance10'"]
impl RenderingFlags {
    pub const LOCAL_READ_CONCURRENT_ACCESS_CONTROL_KHR: Self = Self(0b1_0000_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance10'"]
impl ResolveImageFlagsKHR {
    pub const SKIP_TRANSFER_FUNCTION: Self = Self(0b1);
    pub const ENABLE_TRANSFER_FUNCTION: Self = Self(0b10);
}
#[doc = "Generated from 'VK_KHR_maintenance10'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_10_FEATURES_KHR: Self = Self(1_000_630_000);
    pub const PHYSICAL_DEVICE_MAINTENANCE_10_PROPERTIES_KHR: Self = Self(1_000_630_001);
    pub const RENDERING_ATTACHMENT_FLAGS_INFO_KHR: Self = Self(1_000_630_002);
    pub const RENDERING_END_INFO_KHR: Self = Self(1_000_619_003);
    pub const RESOLVE_IMAGE_MODE_INFO_KHR: Self = Self(1_000_630_004);
}
#[doc = "Generated from 'VK_ARM_data_graph_optical_flow'"]
impl DataGraphPipelineNodeConnectionTypeARM {
    pub const OPTICAL_FLOW_INPUT: Self = Self(1_000_631_000);
    pub const OPTICAL_FLOW_REFERENCE: Self = Self(1_000_631_001);
    pub const OPTICAL_FLOW_HINT: Self = Self(1_000_631_002);
    pub const OPTICAL_FLOW_FLOW_VECTOR: Self = Self(1_000_631_003);
    pub const OPTICAL_FLOW_COST: Self = Self(1_000_631_004);
}
#[doc = "Generated from 'VK_ARM_data_graph_optical_flow'"]
impl DataGraphPipelineNodeTypeARM {
    pub const OPTICAL_FLOW: Self = Self(1_000_631_000);
}
#[doc = "Generated from 'VK_ARM_data_graph_optical_flow'"]
impl DataGraphPipelineSessionBindPointARM {
    pub const OPTICAL_FLOW_CACHE: Self = Self(1_000_631_001);
}
#[doc = "Generated from 'VK_ARM_data_graph_optical_flow'"]
impl DataGraphPipelineSessionCreateFlagsARM {
    pub const OPTICAL_FLOW_CACHE: Self = Self(0b10);
}
#[doc = "Generated from 'VK_ARM_data_graph_optical_flow'"]
impl FormatFeatureFlags2 {
    pub const DATA_GRAPH_OPTICAL_FLOW_IMAGE_ARM: Self =
        Self(0b1_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const DATA_GRAPH_OPTICAL_FLOW_VECTOR_ARM: Self =
        Self(0b10_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
    pub const DATA_GRAPH_OPTICAL_FLOW_COST_ARM: Self =
        Self(0b100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_ARM_data_graph_optical_flow'"]
impl PhysicalDeviceDataGraphOperationTypeARM {
    pub const OPTICAL_FLOW: Self = Self(1_000_631_000);
}
#[doc = "Generated from 'VK_ARM_data_graph_optical_flow'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM: Self = Self(1_000_631_000);
    pub const QUEUE_FAMILY_DATA_GRAPH_OPTICAL_FLOW_PROPERTIES_ARM: Self = Self(1_000_631_001);
    pub const DATA_GRAPH_OPTICAL_FLOW_IMAGE_FORMAT_INFO_ARM: Self = Self(1_000_631_003);
    pub const DATA_GRAPH_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_ARM: Self = Self(1_000_631_004);
    pub const DATA_GRAPH_PIPELINE_OPTICAL_FLOW_DISPATCH_INFO_ARM: Self = Self(1_000_631_005);
    pub const DATA_GRAPH_PIPELINE_OPTICAL_FLOW_CREATE_INFO_ARM: Self = Self(1_000_631_002);
    pub const DATA_GRAPH_PIPELINE_RESOURCE_INFO_IMAGE_LAYOUT_ARM: Self = Self(1_000_631_006);
    pub const DATA_GRAPH_PIPELINE_SINGLE_NODE_CREATE_INFO_ARM: Self = Self(1_000_631_007);
    pub const DATA_GRAPH_PIPELINE_SINGLE_NODE_CONNECTION_ARM: Self = Self(1_000_631_008);
}
#[doc = "Generated from 'VK_EXT_shader_long_vector'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_LONG_VECTOR_FEATURES_EXT: Self = Self(1_000_635_000);
    pub const PHYSICAL_DEVICE_SHADER_LONG_VECTOR_PROPERTIES_EXT: Self = Self(1_000_635_001);
}
#[doc = "Generated from 'VK_SEC_pipeline_cache_incremental_mode'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PIPELINE_CACHE_INCREMENTAL_MODE_FEATURES_SEC: Self =
        Self(1_000_637_000);
}
#[doc = "Generated from 'VK_EXT_shader_uniform_buffer_unsized_array'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_FEATURES_EXT: Self =
        Self(1_000_642_000);
}
#[doc = "Generated from 'VK_NV_compute_occupancy_priority'"]
impl StructureType {
    pub const COMPUTE_OCCUPANCY_PRIORITY_PARAMETERS_NV: Self = Self(1_000_645_000);
    pub const PHYSICAL_DEVICE_COMPUTE_OCCUPANCY_PRIORITY_FEATURES_NV: Self = Self(1_000_645_001);
}
#[doc = "Generated from 'VK_KHR_maintenance11'"]
impl ImageCreateFlags {
    pub const ALIAS_SINGLE_LAYER_DESCRIPTOR_KHR: Self = Self(0b100_0000_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance11'"]
impl PipelineLayoutCreateFlags {
    pub const NO_TASK_SHADER_KHR: Self = Self(0b100);
}
#[doc = "Generated from 'VK_KHR_maintenance11'"]
impl ShaderCreateFlagsEXT {
    pub const INDEPENDENT_SETS_KHR: Self = Self(0b100_0000_0000_0000_0000);
}
#[doc = "Generated from 'VK_KHR_maintenance11'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_MAINTENANCE_11_FEATURES_KHR: Self = Self(1_000_657_000);
    pub const QUEUE_FAMILY_OPTIMAL_IMAGE_TRANSFER_GRANULARITY_PROPERTIES_KHR: Self =
        Self(1_000_657_001);
}
#[doc = "Generated from 'VK_EXT_shader_subgroup_partitioned'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_SUBGROUP_PARTITIONED_FEATURES_EXT: Self = Self(1_000_662_000);
}
#[doc = "Generated from 'VK_EXT_shader_subgroup_partitioned'"]
impl SubgroupFeatureFlags {
    pub const PARTITIONED_EXT: Self = Self(0b1_0000_0000);
}
#[doc = "Generated from 'VK_SEC_ubm_surface'"]
impl StructureType {
    pub const UBM_SURFACE_CREATE_INFO_SEC: Self = Self(1_000_664_000);
}
#[doc = "Generated from 'VK_VALVE_shader_mixed_float_dot_product'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE: Self =
        Self(1_000_673_000);
}
#[doc = "Generated from 'VK_SEC_throttle_hint'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_THROTTLE_HINT_FEATURES_SEC: Self = Self(1_000_674_000);
    pub const THROTTLE_HINT_SUBMIT_INFO_SEC: Self = Self(1_000_674_001);
}
#[doc = "Generated from 'VK_ARM_data_graph_neural_accelerator_statistics'"]
impl DataGraphPipelinePropertyARM {
    pub const NEURAL_ACCELERATOR_DEBUG_DATABASE: Self = Self(1_000_676_000);
    pub const NEURAL_ACCELERATOR_STATISTICS_INFO: Self = Self(1_000_676_001);
}
#[doc = "Generated from 'VK_ARM_data_graph_neural_accelerator_statistics'"]
impl DataGraphPipelineSessionBindPointARM {
    pub const NEURAL_ACCELERATOR_STATISTICS: Self = Self(1_000_676_000);
}
#[doc = "Generated from 'VK_ARM_data_graph_neural_accelerator_statistics'"]
impl StructureType {
    pub const DATA_GRAPH_PIPELINE_NEURAL_STATISTICS_CREATE_INFO_ARM: Self = Self(1_000_676_000);
    pub const DATA_GRAPH_PIPELINE_SESSION_NEURAL_STATISTICS_CREATE_INFO_ARM: Self =
        Self(1_000_676_001);
    pub const PHYSICAL_DEVICE_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_FEATURES_ARM: Self =
        Self(1_000_676_002);
}
#[doc = "Generated from 'VK_EXT_primitive_restart_index'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_PRIMITIVE_RESTART_INDEX_FEATURES_EXT: Self = Self(1_000_678_000);
}
#[doc = "Generated from 'VK_NV_cooperative_matrix_decode_vector'"]
impl StructureType {
    pub const PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV: Self =
        Self(1_000_689_000);
}
pub const KHR_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_surface\0") };
pub const KHR_SURFACE_SPEC_VERSION: u32 = 25u32;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroySurfaceKHR = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    surface: SurfaceKHR,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSurfaceSupportKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    queue_family_index: u32,
    surface: SurfaceKHR,
    p_supported: *mut Bool32,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    surface: SurfaceKHR,
    p_surface_capabilities: *mut SurfaceCapabilitiesKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSurfaceFormatsKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    surface: SurfaceKHR,
    p_surface_format_count: *mut u32,
    p_surface_formats: *mut SurfaceFormatKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSurfacePresentModesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    surface: SurfaceKHR,
    p_present_mode_count: *mut u32,
    p_present_modes: *mut PresentModeKHR,
) -> Result;
pub const KHR_SWAPCHAIN_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain\0") };
pub const KHR_SWAPCHAIN_SPEC_VERSION: u32 = 70u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDevicePresentRectanglesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    surface: SurfaceKHR,
    p_rect_count: *mut u32,
    p_rects: *mut Rect2D,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateSwapchainKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const SwapchainCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_swapchain: *mut SwapchainKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroySwapchainKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetSwapchainImagesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_swapchain_image_count: *mut u32,
    p_swapchain_images: *mut Image,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkAcquireNextImageKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    timeout: u64,
    semaphore: Semaphore,
    fence: Fence,
    p_image_index: *mut u32,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkQueuePresentKHR =
    unsafe extern "system" fn(queue: Queue, p_present_info: *const PresentInfoKHR<'_>) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceGroupPresentCapabilitiesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_device_group_present_capabilities: *mut DeviceGroupPresentCapabilitiesKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceGroupSurfacePresentModesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    surface: SurfaceKHR,
    p_modes: *mut DeviceGroupPresentModeFlagsKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkAcquireNextImage2KHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_acquire_info: *const AcquireNextImageInfoKHR<'_>,
    p_image_index: *mut u32,
) -> Result;
pub const KHR_DISPLAY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_display\0") };
pub const KHR_DISPLAY_SPEC_VERSION: u32 = 23u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceDisplayPropertiesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_property_count: *mut u32,
    p_properties: *mut DisplayPropertiesKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_property_count: *mut u32,
    p_properties: *mut DisplayPlanePropertiesKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDisplayPlaneSupportedDisplaysKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    plane_index: u32,
    p_display_count: *mut u32,
    p_displays: *mut DisplayKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDisplayModePropertiesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    display: DisplayKHR,
    p_property_count: *mut u32,
    p_properties: *mut DisplayModePropertiesKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDisplayModeKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    display: DisplayKHR,
    p_create_info: *const DisplayModeCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_mode: *mut DisplayModeKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDisplayPlaneCapabilitiesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    mode: DisplayModeKHR,
    plane_index: u32,
    p_capabilities: *mut DisplayPlaneCapabilitiesKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDisplayPlaneSurfaceKHR = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const DisplaySurfaceCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const KHR_DISPLAY_SWAPCHAIN_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_display_swapchain\0") };
pub const KHR_DISPLAY_SWAPCHAIN_SPEC_VERSION: u32 = 10u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateSharedSwapchainsKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain_count: u32,
    p_create_infos: *const SwapchainCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_swapchains: *mut SwapchainKHR,
) -> Result;
pub const KHR_XLIB_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_xlib_surface\0") };
pub const KHR_XLIB_SURFACE_SPEC_VERSION: u32 = 6u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateXlibSurfaceKHR = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const XlibSurfaceCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    queue_family_index: u32,
    dpy: *mut Display,
    visual_id: VisualID,
) -> Bool32;
pub const KHR_XCB_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_xcb_surface\0") };
pub const KHR_XCB_SURFACE_SPEC_VERSION: u32 = 6u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateXcbSurfaceKHR = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const XcbSurfaceCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    queue_family_index: u32,
    connection: *mut xcb_connection_t,
    visual_id: xcb_visualid_t,
) -> Bool32;
pub const KHR_WAYLAND_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_wayland_surface\0") };
pub const KHR_WAYLAND_SURFACE_SPEC_VERSION: u32 = 6u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateWaylandSurfaceKHR = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const WaylandSurfaceCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    queue_family_index: u32,
    display: *mut wl_display,
)
    -> Bool32;
pub const KHR_ANDROID_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_android_surface\0") };
pub const KHR_ANDROID_SURFACE_SPEC_VERSION: u32 = 6u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateAndroidSurfaceKHR = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const AndroidSurfaceCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const KHR_WIN32_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_win32_surface\0") };
pub const KHR_WIN32_SURFACE_SPEC_VERSION: u32 = 6u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateWin32SurfaceKHR = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const Win32SurfaceCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR =
    unsafe extern "system" fn(physical_device: PhysicalDevice, queue_family_index: u32) -> Bool32;
pub const ANDROID_NATIVE_BUFFER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ANDROID_native_buffer\0") };
pub const ANDROID_NATIVE_BUFFER_SPEC_VERSION: u32 = 8u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSwapchainGrallocUsageANDROID = unsafe extern "system" fn(
    device: crate::vk::Device,
    format: Format,
    image_usage: ImageUsageFlags,
    gralloc_usage: *mut c_int,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkAcquireImageANDROID = unsafe extern "system" fn(
    device: crate::vk::Device,
    image: Image,
    native_fence_fd: c_int,
    semaphore: Semaphore,
    fence: Fence,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkQueueSignalReleaseImageANDROID = unsafe extern "system" fn(
    queue: Queue,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const Semaphore,
    image: Image,
    p_native_fence_fd: *mut c_int,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSwapchainGrallocUsage2ANDROID = unsafe extern "system" fn(
    device: crate::vk::Device,
    format: Format,
    image_usage: ImageUsageFlags,
    swapchain_image_usage: SwapchainImageUsageFlagsANDROID,
    gralloc_consumer_usage: *mut u64,
    gralloc_producer_usage: *mut u64,
) -> Result;
pub const EXT_DEBUG_REPORT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_debug_report\0") };
pub const EXT_DEBUG_REPORT_SPEC_VERSION: u32 = 10u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDebugReportCallbackEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const DebugReportCallbackCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_callback: *mut DebugReportCallbackEXT,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyDebugReportCallbackEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    callback: DebugReportCallbackEXT,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkDebugReportMessageEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    flags: DebugReportFlagsEXT,
    object_type: DebugReportObjectTypeEXT,
    object: u64,
    location: usize,
    message_code: i32,
    p_layer_prefix: *const c_char,
    p_message: *const c_char,
);
pub const NV_GLSL_SHADER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_glsl_shader\0") };
pub const NV_GLSL_SHADER_SPEC_VERSION: u32 = 1u32;
pub const EXT_DEPTH_RANGE_UNRESTRICTED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_depth_range_unrestricted\0") };
pub const EXT_DEPTH_RANGE_UNRESTRICTED_SPEC_VERSION: u32 = 1u32;
pub const KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_sampler_mirror_clamp_to_edge\0") };
pub const KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_SPEC_VERSION: u32 = 3u32;
pub const IMG_FILTER_CUBIC_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_IMG_filter_cubic\0") };
pub const IMG_FILTER_CUBIC_SPEC_VERSION: u32 = 1u32;
pub const AMD_RASTERIZATION_ORDER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_rasterization_order\0") };
pub const AMD_RASTERIZATION_ORDER_SPEC_VERSION: u32 = 1u32;
pub const AMD_SHADER_TRINARY_MINMAX_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_trinary_minmax\0") };
pub const AMD_SHADER_TRINARY_MINMAX_SPEC_VERSION: u32 = 1u32;
pub const AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_explicit_vertex_parameter\0") };
pub const AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_SPEC_VERSION: u32 = 1u32;
pub const EXT_DEBUG_MARKER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_debug_marker\0") };
pub const EXT_DEBUG_MARKER_SPEC_VERSION: u32 = 4u32;
#[allow(non_camel_case_types)]
pub type PFN_vkDebugMarkerSetObjectTagEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_tag_info: *const DebugMarkerObjectTagInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDebugMarkerSetObjectNameEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_name_info: *const DebugMarkerObjectNameInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDebugMarkerBeginEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_marker_info: *const DebugMarkerMarkerInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDebugMarkerEndEXT = unsafe extern "system" fn(command_buffer: CommandBuffer);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDebugMarkerInsertEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_marker_info: *const DebugMarkerMarkerInfoEXT<'_>,
);
pub const KHR_VIDEO_QUEUE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_queue\0") };
pub const KHR_VIDEO_QUEUE_SPEC_VERSION: u32 = 8u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_video_profile: *const VideoProfileInfoKHR<'_>,
    p_capabilities: *mut VideoCapabilitiesKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_video_format_info: *const PhysicalDeviceVideoFormatInfoKHR<'_>,
    p_video_format_property_count: *mut u32,
    p_video_format_properties: *mut VideoFormatPropertiesKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateVideoSessionKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const VideoSessionCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_video_session: *mut VideoSessionKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyVideoSessionKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    video_session: VideoSessionKHR,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetVideoSessionMemoryRequirementsKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    video_session: VideoSessionKHR,
    p_memory_requirements_count: *mut u32,
    p_memory_requirements: *mut VideoSessionMemoryRequirementsKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkBindVideoSessionMemoryKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    video_session: VideoSessionKHR,
    bind_session_memory_info_count: u32,
    p_bind_session_memory_infos: *const BindVideoSessionMemoryInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateVideoSessionParametersKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const VideoSessionParametersCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_video_session_parameters: *mut VideoSessionParametersKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkUpdateVideoSessionParametersKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    video_session_parameters: VideoSessionParametersKHR,
    p_update_info: *const VideoSessionParametersUpdateInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyVideoSessionParametersKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    video_session_parameters: VideoSessionParametersKHR,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginVideoCodingKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_begin_info: *const VideoBeginCodingInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndVideoCodingKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_end_coding_info: *const VideoEndCodingInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdControlVideoCodingKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_coding_control_info: *const VideoCodingControlInfoKHR<'_>,
);
pub const KHR_VIDEO_DECODE_QUEUE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_queue\0") };
pub const KHR_VIDEO_DECODE_QUEUE_SPEC_VERSION: u32 = 8u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDecodeVideoKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_decode_info: *const VideoDecodeInfoKHR<'_>,
);
pub const AMD_GCN_SHADER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_gcn_shader\0") };
pub const AMD_GCN_SHADER_SPEC_VERSION: u32 = 1u32;
pub const NV_DEDICATED_ALLOCATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_dedicated_allocation\0") };
pub const NV_DEDICATED_ALLOCATION_SPEC_VERSION: u32 = 1u32;
pub const EXT_TRANSFORM_FEEDBACK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_transform_feedback\0") };
pub const EXT_TRANSFORM_FEEDBACK_SPEC_VERSION: u32 = 1u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindTransformFeedbackBuffersEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_buffers: *const Buffer,
    p_offsets: *const DeviceSize,
    p_sizes: *const DeviceSize,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginTransformFeedbackEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_counter_buffer: u32,
    counter_buffer_count: u32,
    p_counter_buffers: *const Buffer,
    p_counter_buffer_offsets: *const DeviceSize,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndTransformFeedbackEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_counter_buffer: u32,
    counter_buffer_count: u32,
    p_counter_buffers: *const Buffer,
    p_counter_buffer_offsets: *const DeviceSize,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginQueryIndexedEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    query_pool: QueryPool,
    query: u32,
    flags: QueryControlFlags,
    index: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndQueryIndexedEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    query_pool: QueryPool,
    query: u32,
    index: u32,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawIndirectByteCountEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    instance_count: u32,
    first_instance: u32,
    counter_buffer: Buffer,
    counter_buffer_offset: DeviceSize,
    counter_offset: u32,
    vertex_stride: u32,
);
pub const NVX_BINARY_IMPORT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NVX_binary_import\0") };
pub const NVX_BINARY_IMPORT_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateCuModuleNVX = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const CuModuleCreateInfoNVX<'_>,
    p_allocator: *const AllocationCallbacks,
    p_module: *mut CuModuleNVX,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateCuFunctionNVX = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const CuFunctionCreateInfoNVX<'_>,
    p_allocator: *const AllocationCallbacks,
    p_function: *mut CuFunctionNVX,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyCuModuleNVX = unsafe extern "system" fn(
    device: crate::vk::Device,
    module: CuModuleNVX,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyCuFunctionNVX = unsafe extern "system" fn(
    device: crate::vk::Device,
    function: CuFunctionNVX,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCuLaunchKernelNVX = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_launch_info: *const CuLaunchInfoNVX<'_>,
);
pub const NVX_IMAGE_VIEW_HANDLE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NVX_image_view_handle\0") };
pub const NVX_IMAGE_VIEW_HANDLE_SPEC_VERSION: u32 = 4u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageViewHandleNVX = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ImageViewHandleInfoNVX<'_>,
) -> u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageViewHandle64NVX = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ImageViewHandleInfoNVX<'_>,
) -> u64;
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageViewAddressNVX = unsafe extern "system" fn(
    device: crate::vk::Device,
    image_view: ImageView,
    p_properties: *mut ImageViewAddressPropertiesNVX<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceCombinedImageSamplerIndexNVX = unsafe extern "system" fn(
    device: crate::vk::Device,
    image_view_index: u64,
    sampler_index: u64,
) -> u64;
pub const AMD_DRAW_INDIRECT_COUNT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_draw_indirect_count\0") };
pub const AMD_DRAW_INDIRECT_COUNT_SPEC_VERSION: u32 = 2u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawIndirectCount = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    count_buffer: Buffer,
    count_buffer_offset: DeviceSize,
    max_draw_count: u32,
    stride: u32,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawIndexedIndirectCount = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    count_buffer: Buffer,
    count_buffer_offset: DeviceSize,
    max_draw_count: u32,
    stride: u32,
);
pub const AMD_NEGATIVE_VIEWPORT_HEIGHT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_negative_viewport_height\0") };
pub const AMD_NEGATIVE_VIEWPORT_HEIGHT_SPEC_VERSION: u32 = 1u32;
pub const AMD_GPU_SHADER_HALF_FLOAT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_gpu_shader_half_float\0") };
pub const AMD_GPU_SHADER_HALF_FLOAT_SPEC_VERSION: u32 = 2u32;
pub const AMD_SHADER_BALLOT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_ballot\0") };
pub const AMD_SHADER_BALLOT_SPEC_VERSION: u32 = 1u32;
pub const KHR_VIDEO_ENCODE_H264_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_h264\0") };
pub const KHR_VIDEO_ENCODE_H264_SPEC_VERSION: u32 = 14u32;
pub const KHR_VIDEO_ENCODE_H265_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_h265\0") };
pub const KHR_VIDEO_ENCODE_H265_SPEC_VERSION: u32 = 14u32;
pub const KHR_VIDEO_DECODE_H264_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_h264\0") };
pub const KHR_VIDEO_DECODE_H264_SPEC_VERSION: u32 = 9u32;
pub const AMD_TEXTURE_GATHER_BIAS_LOD_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_texture_gather_bias_lod\0") };
pub const AMD_TEXTURE_GATHER_BIAS_LOD_SPEC_VERSION: u32 = 1u32;
pub const AMD_SHADER_INFO_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_info\0") };
pub const AMD_SHADER_INFO_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetShaderInfoAMD = unsafe extern "system" fn(
    device: crate::vk::Device,
    pipeline: Pipeline,
    shader_stage: ShaderStageFlags,
    info_type: ShaderInfoTypeAMD,
    p_info_size: *mut usize,
    p_info: *mut c_void,
) -> Result;
pub const KHR_DYNAMIC_RENDERING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_dynamic_rendering\0") };
pub const KHR_DYNAMIC_RENDERING_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginRendering = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_rendering_info: *const RenderingInfo<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndRendering = unsafe extern "system" fn(command_buffer: CommandBuffer);
pub const AMD_SHADER_IMAGE_LOAD_STORE_LOD_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_image_load_store_lod\0") };
pub const AMD_SHADER_IMAGE_LOAD_STORE_LOD_SPEC_VERSION: u32 = 1u32;
pub const GGP_STREAM_DESCRIPTOR_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_GGP_stream_descriptor_surface\0") };
pub const GGP_STREAM_DESCRIPTOR_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateStreamDescriptorSurfaceGGP = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const StreamDescriptorSurfaceCreateInfoGGP<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const NV_CORNER_SAMPLED_IMAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_corner_sampled_image\0") };
pub const NV_CORNER_SAMPLED_IMAGE_SPEC_VERSION: u32 = 2u32;
pub const KHR_MULTIVIEW_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_multiview\0") };
pub const KHR_MULTIVIEW_SPEC_VERSION: u32 = 1u32;
pub const IMG_FORMAT_PVRTC_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_IMG_format_pvrtc\0") };
pub const IMG_FORMAT_PVRTC_SPEC_VERSION: u32 = 1u32;
pub const NV_EXTERNAL_MEMORY_CAPABILITIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_external_memory_capabilities\0") };
pub const NV_EXTERNAL_MEMORY_CAPABILITIES_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        format: Format,
        ty: ImageType,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        flags: ImageCreateFlags,
        external_handle_type: ExternalMemoryHandleTypeFlagsNV,
        p_external_image_format_properties: *mut ExternalImageFormatPropertiesNV,
    ) -> Result;
pub const NV_EXTERNAL_MEMORY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_external_memory\0") };
pub const NV_EXTERNAL_MEMORY_SPEC_VERSION: u32 = 1u32;
pub const NV_EXTERNAL_MEMORY_WIN32_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_external_memory_win32\0") };
pub const NV_EXTERNAL_MEMORY_WIN32_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryWin32HandleNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    memory: DeviceMemory,
    handle_type: ExternalMemoryHandleTypeFlagsNV,
    p_handle: *mut HANDLE,
) -> Result;
pub const NV_WIN32_KEYED_MUTEX_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_win32_keyed_mutex\0") };
pub const NV_WIN32_KEYED_MUTEX_SPEC_VERSION: u32 = 2u32;
pub const KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_get_physical_device_properties2\0") };
pub const KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceFeatures2 = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_features: *mut PhysicalDeviceFeatures2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceProperties2 = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_properties: *mut PhysicalDeviceProperties2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceFormatProperties2 = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    format: Format,
    p_format_properties: *mut FormatProperties2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceImageFormatProperties2 = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_image_format_info: *const PhysicalDeviceImageFormatInfo2<'_>,
    p_image_format_properties: *mut ImageFormatProperties2<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceQueueFamilyProperties2 = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut QueueFamilyProperties2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceMemoryProperties2 = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_memory_properties: *mut PhysicalDeviceMemoryProperties2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSparseImageFormatProperties2 = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_format_info: *const PhysicalDeviceSparseImageFormatInfo2<'_>,
    p_property_count: *mut u32,
    p_properties: *mut SparseImageFormatProperties2<'_>,
);
pub const KHR_DEVICE_GROUP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_device_group\0") };
pub const KHR_DEVICE_GROUP_SPEC_VERSION: u32 = 4u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceGroupPeerMemoryFeatures = unsafe extern "system" fn(
    device: crate::vk::Device,
    heap_index: u32,
    local_device_index: u32,
    remote_device_index: u32,
    p_peer_memory_features: *mut PeerMemoryFeatureFlags,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDeviceMask =
    unsafe extern "system" fn(command_buffer: CommandBuffer, device_mask: u32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDispatchBase = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    base_group_x: u32,
    base_group_y: u32,
    base_group_z: u32,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
);
pub const EXT_VALIDATION_FLAGS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_validation_flags\0") };
pub const EXT_VALIDATION_FLAGS_SPEC_VERSION: u32 = 3u32;
pub const NN_VI_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NN_vi_surface\0") };
pub const NN_VI_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateViSurfaceNN = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const ViSurfaceCreateInfoNN<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const KHR_SHADER_DRAW_PARAMETERS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_draw_parameters\0") };
pub const KHR_SHADER_DRAW_PARAMETERS_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_SUBGROUP_BALLOT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_subgroup_ballot\0") };
pub const EXT_SHADER_SUBGROUP_BALLOT_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_SUBGROUP_VOTE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_subgroup_vote\0") };
pub const EXT_SHADER_SUBGROUP_VOTE_SPEC_VERSION: u32 = 1u32;
pub const EXT_TEXTURE_COMPRESSION_ASTC_HDR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_texture_compression_astc_hdr\0") };
pub const EXT_TEXTURE_COMPRESSION_ASTC_HDR_SPEC_VERSION: u32 = 1u32;
pub const EXT_ASTC_DECODE_MODE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_astc_decode_mode\0") };
pub const EXT_ASTC_DECODE_MODE_SPEC_VERSION: u32 = 1u32;
pub const EXT_PIPELINE_ROBUSTNESS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_pipeline_robustness\0") };
pub const EXT_PIPELINE_ROBUSTNESS_SPEC_VERSION: u32 = 1u32;
pub const KHR_MAINTENANCE1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance1\0") };
pub const KHR_MAINTENANCE1_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkTrimCommandPool = unsafe extern "system" fn(
    device: crate::vk::Device,
    command_pool: CommandPool,
    flags: CommandPoolTrimFlags,
);
pub const KHR_DEVICE_GROUP_CREATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_device_group_creation\0") };
pub const KHR_DEVICE_GROUP_CREATION_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkEnumeratePhysicalDeviceGroups = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_physical_device_group_count: *mut u32,
    p_physical_device_group_properties: *mut PhysicalDeviceGroupProperties<'_>,
) -> Result;
pub const KHR_EXTERNAL_MEMORY_CAPABILITIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_memory_capabilities\0") };
pub const KHR_EXTERNAL_MEMORY_CAPABILITIES_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceExternalBufferProperties = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_external_buffer_info: *const PhysicalDeviceExternalBufferInfo<'_>,
    p_external_buffer_properties: *mut ExternalBufferProperties<'_>,
);
pub const KHR_EXTERNAL_MEMORY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_memory\0") };
pub const KHR_EXTERNAL_MEMORY_SPEC_VERSION: u32 = 1u32;
pub const KHR_EXTERNAL_MEMORY_WIN32_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_memory_win32\0") };
pub const KHR_EXTERNAL_MEMORY_WIN32_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryWin32HandleKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_win32_handle_info: *const MemoryGetWin32HandleInfoKHR<'_>,
    p_handle: *mut HANDLE,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryWin32HandlePropertiesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    handle_type: ExternalMemoryHandleTypeFlags,
    handle: HANDLE,
    p_memory_win32_handle_properties: *mut MemoryWin32HandlePropertiesKHR<'_>,
) -> Result;
pub const KHR_EXTERNAL_MEMORY_FD_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_memory_fd\0") };
pub const KHR_EXTERNAL_MEMORY_FD_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryFdKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_fd_info: *const MemoryGetFdInfoKHR<'_>,
    p_fd: *mut c_int,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryFdPropertiesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    handle_type: ExternalMemoryHandleTypeFlags,
    fd: c_int,
    p_memory_fd_properties: *mut MemoryFdPropertiesKHR<'_>,
) -> Result;
pub const KHR_WIN32_KEYED_MUTEX_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_win32_keyed_mutex\0") };
pub const KHR_WIN32_KEYED_MUTEX_SPEC_VERSION: u32 = 1u32;
pub const KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_semaphore_capabilities\0") };
pub const KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceExternalSemaphoreProperties = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_external_semaphore_info: *const PhysicalDeviceExternalSemaphoreInfo<'_>,
    p_external_semaphore_properties: *mut ExternalSemaphoreProperties<'_>,
);
pub const KHR_EXTERNAL_SEMAPHORE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_semaphore\0") };
pub const KHR_EXTERNAL_SEMAPHORE_SPEC_VERSION: u32 = 1u32;
pub const KHR_EXTERNAL_SEMAPHORE_WIN32_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_semaphore_win32\0") };
pub const KHR_EXTERNAL_SEMAPHORE_WIN32_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkImportSemaphoreWin32HandleKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_import_semaphore_win32_handle_info: *const ImportSemaphoreWin32HandleInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSemaphoreWin32HandleKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_win32_handle_info: *const SemaphoreGetWin32HandleInfoKHR<'_>,
    p_handle: *mut HANDLE,
) -> Result;
pub const KHR_EXTERNAL_SEMAPHORE_FD_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_semaphore_fd\0") };
pub const KHR_EXTERNAL_SEMAPHORE_FD_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkImportSemaphoreFdKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_import_semaphore_fd_info: *const ImportSemaphoreFdInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSemaphoreFdKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_fd_info: *const SemaphoreGetFdInfoKHR<'_>,
    p_fd: *mut c_int,
) -> Result;
pub const KHR_PUSH_DESCRIPTOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_push_descriptor\0") };
pub const KHR_PUSH_DESCRIPTOR_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPushDescriptorSet = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    pipeline_bind_point: PipelineBindPoint,
    layout: PipelineLayout,
    set: u32,
    descriptor_write_count: u32,
    p_descriptor_writes: *const WriteDescriptorSet<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPushDescriptorSetWithTemplate = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    descriptor_update_template: DescriptorUpdateTemplate,
    layout: PipelineLayout,
    set: u32,
    p_data: *const c_void,
);
pub const EXT_CONDITIONAL_RENDERING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_conditional_rendering\0") };
pub const EXT_CONDITIONAL_RENDERING_SPEC_VERSION: u32 = 2u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginConditionalRenderingEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_conditional_rendering_begin: *const ConditionalRenderingBeginInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndConditionalRenderingEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer);
pub const KHR_SHADER_FLOAT16_INT8_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_float16_int8\0") };
pub const KHR_SHADER_FLOAT16_INT8_SPEC_VERSION: u32 = 1u32;
pub const KHR_16BIT_STORAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_16bit_storage\0") };
pub const KHR_16BIT_STORAGE_SPEC_VERSION: u32 = 1u32;
pub const KHR_INCREMENTAL_PRESENT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_incremental_present\0") };
pub const KHR_INCREMENTAL_PRESENT_SPEC_VERSION: u32 = 2u32;
pub const KHR_DESCRIPTOR_UPDATE_TEMPLATE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_descriptor_update_template\0") };
pub const KHR_DESCRIPTOR_UPDATE_TEMPLATE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDescriptorUpdateTemplate = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const DescriptorUpdateTemplateCreateInfo<'_>,
    p_allocator: *const AllocationCallbacks,
    p_descriptor_update_template: *mut DescriptorUpdateTemplate,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyDescriptorUpdateTemplate = unsafe extern "system" fn(
    device: crate::vk::Device,
    descriptor_update_template: DescriptorUpdateTemplate,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkUpdateDescriptorSetWithTemplate = unsafe extern "system" fn(
    device: crate::vk::Device,
    descriptor_set: DescriptorSet,
    descriptor_update_template: DescriptorUpdateTemplate,
    p_data: *const c_void,
);
pub const NV_CLIP_SPACE_W_SCALING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_clip_space_w_scaling\0") };
pub const NV_CLIP_SPACE_W_SCALING_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetViewportWScalingNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_viewport: u32,
    viewport_count: u32,
    p_viewport_w_scalings: *const ViewportWScalingNV,
);
pub const EXT_DIRECT_MODE_DISPLAY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_direct_mode_display\0") };
pub const EXT_DIRECT_MODE_DISPLAY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkReleaseDisplayEXT =
    unsafe extern "system" fn(physical_device: PhysicalDevice, display: DisplayKHR) -> Result;
pub const EXT_ACQUIRE_XLIB_DISPLAY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_acquire_xlib_display\0") };
pub const EXT_ACQUIRE_XLIB_DISPLAY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkAcquireXlibDisplayEXT = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    dpy: *mut Display,
    display: DisplayKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetRandROutputDisplayEXT = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    dpy: *mut Display,
    rr_output: RROutput,
    p_display: *mut DisplayKHR,
) -> Result;
pub const EXT_DISPLAY_SURFACE_COUNTER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_display_surface_counter\0") };
pub const EXT_DISPLAY_SURFACE_COUNTER_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    surface: SurfaceKHR,
    p_surface_capabilities: *mut SurfaceCapabilities2EXT<'_>,
) -> Result;
pub const EXT_DISPLAY_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_display_control\0") };
pub const EXT_DISPLAY_CONTROL_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkDisplayPowerControlEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    display: DisplayKHR,
    p_display_power_info: *const DisplayPowerInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkRegisterDeviceEventEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_device_event_info: *const DeviceEventInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_fence: *mut Fence,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkRegisterDisplayEventEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    display: DisplayKHR,
    p_display_event_info: *const DisplayEventInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_fence: *mut Fence,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSwapchainCounterEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    counter: SurfaceCounterFlagsEXT,
    p_counter_value: *mut u64,
) -> Result;
pub const GOOGLE_DISPLAY_TIMING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_display_timing\0") };
pub const GOOGLE_DISPLAY_TIMING_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetRefreshCycleDurationGOOGLE = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_display_timing_properties: *mut RefreshCycleDurationGOOGLE,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPastPresentationTimingGOOGLE = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_presentation_timing_count: *mut u32,
    p_presentation_timings: *mut PastPresentationTimingGOOGLE,
) -> Result;
pub const NV_SAMPLE_MASK_OVERRIDE_COVERAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_sample_mask_override_coverage\0") };
pub const NV_SAMPLE_MASK_OVERRIDE_COVERAGE_SPEC_VERSION: u32 = 1u32;
pub const NV_GEOMETRY_SHADER_PASSTHROUGH_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_geometry_shader_passthrough\0") };
pub const NV_GEOMETRY_SHADER_PASSTHROUGH_SPEC_VERSION: u32 = 1u32;
pub const NV_VIEWPORT_ARRAY2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_viewport_array2\0") };
pub const NV_VIEWPORT_ARRAY2_SPEC_VERSION: u32 = 1u32;
pub const NVX_MULTIVIEW_PER_VIEW_ATTRIBUTES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NVX_multiview_per_view_attributes\0") };
pub const NVX_MULTIVIEW_PER_VIEW_ATTRIBUTES_SPEC_VERSION: u32 = 1u32;
pub const NV_VIEWPORT_SWIZZLE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_viewport_swizzle\0") };
pub const NV_VIEWPORT_SWIZZLE_SPEC_VERSION: u32 = 1u32;
pub const EXT_DISCARD_RECTANGLES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_discard_rectangles\0") };
pub const EXT_DISCARD_RECTANGLES_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDiscardRectangleEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_discard_rectangle: u32,
    discard_rectangle_count: u32,
    p_discard_rectangles: *const Rect2D,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDiscardRectangleEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, discard_rectangle_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDiscardRectangleModeEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    discard_rectangle_mode: DiscardRectangleModeEXT,
);
pub const EXT_CONSERVATIVE_RASTERIZATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_conservative_rasterization\0") };
pub const EXT_CONSERVATIVE_RASTERIZATION_SPEC_VERSION: u32 = 1u32;
pub const EXT_DEPTH_CLIP_ENABLE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_depth_clip_enable\0") };
pub const EXT_DEPTH_CLIP_ENABLE_SPEC_VERSION: u32 = 1u32;
pub const EXT_SWAPCHAIN_COLORSPACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_swapchain_colorspace\0") };
pub const EXT_SWAPCHAIN_COLORSPACE_SPEC_VERSION: u32 = 5u32;
pub const EXT_HDR_METADATA_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_hdr_metadata\0") };
pub const EXT_HDR_METADATA_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkSetHdrMetadataEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain_count: u32,
    p_swapchains: *const SwapchainKHR,
    p_metadata: *const HdrMetadataEXT<'_>,
);
pub const KHR_IMAGELESS_FRAMEBUFFER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_imageless_framebuffer\0") };
pub const KHR_IMAGELESS_FRAMEBUFFER_SPEC_VERSION: u32 = 1u32;
pub const KHR_CREATE_RENDERPASS2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_create_renderpass2\0") };
pub const KHR_CREATE_RENDERPASS2_SPEC_VERSION: u32 = 1u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-dynamicrendering>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCreateRenderPass2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const RenderPassCreateInfo2<'_>,
    p_allocator: *const AllocationCallbacks,
    p_render_pass: *mut RenderPass,
) -> Result;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-dynamicrendering>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginRenderPass2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_render_pass_begin: *const RenderPassBeginInfo<'_>,
    p_subpass_begin_info: *const SubpassBeginInfo<'_>,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-dynamicrendering>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdNextSubpass2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_subpass_begin_info: *const SubpassBeginInfo<'_>,
    p_subpass_end_info: *const SubpassEndInfo<'_>,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-dynamicrendering>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndRenderPass2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_subpass_end_info: *const SubpassEndInfo<'_>,
);
pub const IMG_RELAXED_LINE_RASTERIZATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_IMG_relaxed_line_rasterization\0") };
pub const IMG_RELAXED_LINE_RASTERIZATION_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHARED_PRESENTABLE_IMAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shared_presentable_image\0") };
pub const KHR_SHARED_PRESENTABLE_IMAGE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSwapchainStatusKHR =
    unsafe extern "system" fn(device: crate::vk::Device, swapchain: SwapchainKHR) -> Result;
pub const KHR_EXTERNAL_FENCE_CAPABILITIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_fence_capabilities\0") };
pub const KHR_EXTERNAL_FENCE_CAPABILITIES_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceExternalFenceProperties = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_external_fence_info: *const PhysicalDeviceExternalFenceInfo<'_>,
    p_external_fence_properties: *mut ExternalFenceProperties<'_>,
);
pub const KHR_EXTERNAL_FENCE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_fence\0") };
pub const KHR_EXTERNAL_FENCE_SPEC_VERSION: u32 = 1u32;
pub const KHR_EXTERNAL_FENCE_WIN32_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_fence_win32\0") };
pub const KHR_EXTERNAL_FENCE_WIN32_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkImportFenceWin32HandleKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_import_fence_win32_handle_info: *const ImportFenceWin32HandleInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetFenceWin32HandleKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_win32_handle_info: *const FenceGetWin32HandleInfoKHR<'_>,
    p_handle: *mut HANDLE,
) -> Result;
pub const KHR_EXTERNAL_FENCE_FD_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_fence_fd\0") };
pub const KHR_EXTERNAL_FENCE_FD_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkImportFenceFdKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_import_fence_fd_info: *const ImportFenceFdInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetFenceFdKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_fd_info: *const FenceGetFdInfoKHR<'_>,
    p_fd: *mut c_int,
) -> Result;
pub const KHR_PERFORMANCE_QUERY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_performance_query\0") };
pub const KHR_PERFORMANCE_QUERY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_counter_count: *mut u32,
        p_counters: *mut PerformanceCounterKHR<'_>,
        p_counter_descriptions: *mut PerformanceCounterDescriptionKHR<'_>,
    ) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_performance_query_create_info: *const QueryPoolPerformanceCreateInfoKHR<'_>,
        p_num_passes: *mut u32,
    );
#[allow(non_camel_case_types)]
pub type PFN_vkAcquireProfilingLockKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const AcquireProfilingLockInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkReleaseProfilingLockKHR = unsafe extern "system" fn(device: crate::vk::Device);
pub const KHR_MAINTENANCE2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance2\0") };
pub const KHR_MAINTENANCE2_SPEC_VERSION: u32 = 1u32;
pub const KHR_GET_SURFACE_CAPABILITIES2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_get_surface_capabilities2\0") };
pub const KHR_GET_SURFACE_CAPABILITIES2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_surface_info: *const PhysicalDeviceSurfaceInfo2KHR<'_>,
    p_surface_capabilities: *mut SurfaceCapabilities2KHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSurfaceFormats2KHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_surface_info: *const PhysicalDeviceSurfaceInfo2KHR<'_>,
    p_surface_format_count: *mut u32,
    p_surface_formats: *mut SurfaceFormat2KHR<'_>,
) -> Result;
pub const KHR_VARIABLE_POINTERS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_variable_pointers\0") };
pub const KHR_VARIABLE_POINTERS_SPEC_VERSION: u32 = 1u32;
pub const KHR_GET_DISPLAY_PROPERTIES2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_get_display_properties2\0") };
pub const KHR_GET_DISPLAY_PROPERTIES2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceDisplayProperties2KHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_property_count: *mut u32,
    p_properties: *mut DisplayProperties2KHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_property_count: *mut u32,
    p_properties: *mut DisplayPlaneProperties2KHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDisplayModeProperties2KHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    display: DisplayKHR,
    p_property_count: *mut u32,
    p_properties: *mut DisplayModeProperties2KHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDisplayPlaneCapabilities2KHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_display_plane_info: *const DisplayPlaneInfo2KHR<'_>,
    p_capabilities: *mut DisplayPlaneCapabilities2KHR<'_>,
) -> Result;
pub const MVK_IOS_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_MVK_ios_surface\0") };
pub const MVK_IOS_SURFACE_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateIOSSurfaceMVK = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const IOSSurfaceCreateInfoMVK<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const MVK_MACOS_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_MVK_macos_surface\0") };
pub const MVK_MACOS_SURFACE_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateMacOSSurfaceMVK = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const MacOSSurfaceCreateInfoMVK<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const EXT_EXTERNAL_MEMORY_DMA_BUF_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_external_memory_dma_buf\0") };
pub const EXT_EXTERNAL_MEMORY_DMA_BUF_SPEC_VERSION: u32 = 1u32;
pub const EXT_QUEUE_FAMILY_FOREIGN_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_queue_family_foreign\0") };
pub const EXT_QUEUE_FAMILY_FOREIGN_SPEC_VERSION: u32 = 1u32;
pub const KHR_DEDICATED_ALLOCATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_dedicated_allocation\0") };
pub const KHR_DEDICATED_ALLOCATION_SPEC_VERSION: u32 = 3u32;
pub const EXT_DEBUG_UTILS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_debug_utils\0") };
pub const EXT_DEBUG_UTILS_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDebugUtilsMessengerEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const DebugUtilsMessengerCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_messenger: *mut DebugUtilsMessengerEXT,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyDebugUtilsMessengerEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    messenger: DebugUtilsMessengerEXT,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkSubmitDebugUtilsMessageEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_types: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkSetDebugUtilsObjectNameEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_name_info: *const DebugUtilsObjectNameInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkSetDebugUtilsObjectTagEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_tag_info: *const DebugUtilsObjectTagInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkQueueBeginDebugUtilsLabelEXT =
    unsafe extern "system" fn(queue: Queue, p_label_info: *const DebugUtilsLabelEXT<'_>);
#[allow(non_camel_case_types)]
pub type PFN_vkQueueEndDebugUtilsLabelEXT = unsafe extern "system" fn(queue: Queue);
#[allow(non_camel_case_types)]
pub type PFN_vkQueueInsertDebugUtilsLabelEXT =
    unsafe extern "system" fn(queue: Queue, p_label_info: *const DebugUtilsLabelEXT<'_>);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginDebugUtilsLabelEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_label_info: *const DebugUtilsLabelEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndDebugUtilsLabelEXT = unsafe extern "system" fn(command_buffer: CommandBuffer);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdInsertDebugUtilsLabelEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_label_info: *const DebugUtilsLabelEXT<'_>,
);
pub const ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_ANDROID_external_memory_android_hardware_buffer\0")
};
pub const ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_SPEC_VERSION: u32 = 5u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetAndroidHardwareBufferPropertiesANDROID = unsafe extern "system" fn(
    device: crate::vk::Device,
    buffer: *const AHardwareBuffer,
    p_properties: *mut AndroidHardwareBufferPropertiesANDROID<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryAndroidHardwareBufferANDROID = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const MemoryGetAndroidHardwareBufferInfoANDROID<'_>,
    p_buffer: *mut *mut AHardwareBuffer,
) -> Result;
pub const EXT_SAMPLER_FILTER_MINMAX_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_sampler_filter_minmax\0") };
pub const EXT_SAMPLER_FILTER_MINMAX_SPEC_VERSION: u32 = 2u32;
pub const KHR_STORAGE_BUFFER_STORAGE_CLASS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_storage_buffer_storage_class\0") };
pub const KHR_STORAGE_BUFFER_STORAGE_CLASS_SPEC_VERSION: u32 = 1u32;
pub const AMD_GPU_SHADER_INT16_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_gpu_shader_int16\0") };
pub const AMD_GPU_SHADER_INT16_SPEC_VERSION: u32 = 2u32;
pub const AMD_GPA_INTERFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_gpa_interface\0") };
pub const AMD_GPA_INTERFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateGpaSessionAMD = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const GpaSessionCreateInfoAMD<'_>,
    p_allocator: *const AllocationCallbacks,
    p_gpa_session: *mut GpaSessionAMD,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyGpaSessionAMD = unsafe extern "system" fn(
    device: crate::vk::Device,
    gpa_session: GpaSessionAMD,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkSetGpaDeviceClockModeAMD = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *mut GpaDeviceClockModeInfoAMD<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetGpaDeviceClockInfoAMD = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *mut GpaDeviceGetClockInfoAMD<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginGpaSessionAMD =
    unsafe extern "system" fn(command_buffer: CommandBuffer, gpa_session: GpaSessionAMD) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndGpaSessionAMD =
    unsafe extern "system" fn(command_buffer: CommandBuffer, gpa_session: GpaSessionAMD) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginGpaSampleAMD = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    gpa_session: GpaSessionAMD,
    p_gpa_sample_begin_info: *const GpaSampleBeginInfoAMD<'_>,
    p_sample_id: *mut u32,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndGpaSampleAMD = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    gpa_session: GpaSessionAMD,
    sample_id: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetGpaSessionStatusAMD =
    unsafe extern "system" fn(device: crate::vk::Device, gpa_session: GpaSessionAMD) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetGpaSessionResultsAMD = unsafe extern "system" fn(
    device: crate::vk::Device,
    gpa_session: GpaSessionAMD,
    sample_id: u32,
    p_size_in_bytes: *mut usize,
    p_data: *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkResetGpaSessionAMD =
    unsafe extern "system" fn(device: crate::vk::Device, gpa_session: GpaSessionAMD) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyGpaSessionResultsAMD =
    unsafe extern "system" fn(command_buffer: CommandBuffer, gpa_session: GpaSessionAMD);
#[cfg(feature = "provisional")]
pub const AMDX_SHADER_ENQUEUE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMDX_shader_enqueue\0") };
#[cfg(feature = "provisional")]
pub const AMDX_SHADER_ENQUEUE_SPEC_VERSION: u32 = 2u32;
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkCreateExecutionGraphPipelinesAMDX = unsafe extern "system" fn(
    device: crate::vk::Device,
    pipeline_cache: PipelineCache,
    create_info_count: u32,
    p_create_infos: *const ExecutionGraphPipelineCreateInfoAMDX<'_>,
    p_allocator: *const AllocationCallbacks,
    p_pipelines: *mut Pipeline,
) -> Result;
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkGetExecutionGraphPipelineScratchSizeAMDX = unsafe extern "system" fn(
    device: crate::vk::Device,
    execution_graph: Pipeline,
    p_size_info: *mut ExecutionGraphPipelineScratchSizeAMDX<'_>,
) -> Result;
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkGetExecutionGraphPipelineNodeIndexAMDX = unsafe extern "system" fn(
    device: crate::vk::Device,
    execution_graph: Pipeline,
    p_node_info: *const PipelineShaderStageNodeCreateInfoAMDX<'_>,
    p_node_index: *mut u32,
) -> Result;
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdInitializeGraphScratchMemoryAMDX = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    execution_graph: Pipeline,
    scratch: DeviceAddress,
    scratch_size: DeviceSize,
);
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDispatchGraphAMDX = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    scratch: DeviceAddress,
    scratch_size: DeviceSize,
    p_count_info: *const DispatchGraphCountInfoAMDX,
);
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDispatchGraphIndirectAMDX = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    scratch: DeviceAddress,
    scratch_size: DeviceSize,
    p_count_info: *const DispatchGraphCountInfoAMDX,
);
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDispatchGraphIndirectCountAMDX = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    scratch: DeviceAddress,
    scratch_size: DeviceSize,
    count_info: DeviceAddress,
);
pub const EXT_DESCRIPTOR_HEAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_descriptor_heap\0") };
pub const EXT_DESCRIPTOR_HEAP_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceDescriptorSizeEXT = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    descriptor_type: DescriptorType,
) -> DeviceSize;
#[allow(non_camel_case_types)]
pub type PFN_vkWriteSamplerDescriptorsEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    sampler_count: u32,
    p_samplers: *const SamplerCreateInfo<'_>,
    p_descriptors: *const HostAddressRangeEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkWriteResourceDescriptorsEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    resource_count: u32,
    p_resources: *const ResourceDescriptorInfoEXT<'_>,
    p_descriptors: *const HostAddressRangeEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindSamplerHeapEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_bind_info: *const BindHeapInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindResourceHeapEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_bind_info: *const BindHeapInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPushDataEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_push_data_info: *const PushDataInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageOpaqueCaptureDataEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    image_count: u32,
    p_images: *const Image,
    p_datas: *mut HostAddressRangeEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkRegisterCustomBorderColorEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_border_color: *const SamplerCustomBorderColorCreateInfoEXT<'_>,
    request_index: Bool32,
    p_index: *mut u32,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkUnregisterCustomBorderColorEXT =
    unsafe extern "system" fn(device: crate::vk::Device, index: u32);
#[allow(non_camel_case_types)]
pub type PFN_vkGetTensorOpaqueCaptureDataARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    tensor_count: u32,
    p_tensors: *const TensorARM,
    p_datas: *mut HostAddressRangeEXT<'_>,
) -> Result;
pub const AMD_MIXED_ATTACHMENT_SAMPLES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_mixed_attachment_samples\0") };
pub const AMD_MIXED_ATTACHMENT_SAMPLES_SPEC_VERSION: u32 = 1u32;
pub const AMD_SHADER_FRAGMENT_MASK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_fragment_mask\0") };
pub const AMD_SHADER_FRAGMENT_MASK_SPEC_VERSION: u32 = 1u32;
pub const EXT_INLINE_UNIFORM_BLOCK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_inline_uniform_block\0") };
pub const EXT_INLINE_UNIFORM_BLOCK_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_STENCIL_EXPORT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_stencil_export\0") };
pub const EXT_SHADER_STENCIL_EXPORT_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_BFLOAT16_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_bfloat16\0") };
pub const KHR_SHADER_BFLOAT16_SPEC_VERSION: u32 = 1u32;
pub const EXT_SAMPLE_LOCATIONS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_sample_locations\0") };
pub const EXT_SAMPLE_LOCATIONS_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    samples: SampleCountFlags,
    p_multisample_properties: *mut MultisamplePropertiesEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetSampleLocationsEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_sample_locations_info: *const SampleLocationsInfoEXT<'_>,
);
pub const KHR_RELAXED_BLOCK_LAYOUT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_relaxed_block_layout\0") };
pub const KHR_RELAXED_BLOCK_LAYOUT_SPEC_VERSION: u32 = 1u32;
pub const KHR_GET_MEMORY_REQUIREMENTS2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_get_memory_requirements2\0") };
pub const KHR_GET_MEMORY_REQUIREMENTS2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageMemoryRequirements2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ImageMemoryRequirementsInfo2<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetBufferMemoryRequirements2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const BufferMemoryRequirementsInfo2<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageSparseMemoryRequirements2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ImageSparseMemoryRequirementsInfo2<'_>,
    p_sparse_memory_requirement_count: *mut u32,
    p_sparse_memory_requirements: *mut SparseImageMemoryRequirements2<'_>,
);
pub const KHR_IMAGE_FORMAT_LIST_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_image_format_list\0") };
pub const KHR_IMAGE_FORMAT_LIST_SPEC_VERSION: u32 = 1u32;
pub const EXT_BLEND_OPERATION_ADVANCED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_blend_operation_advanced\0") };
pub const EXT_BLEND_OPERATION_ADVANCED_SPEC_VERSION: u32 = 2u32;
pub const NV_FRAGMENT_COVERAGE_TO_COLOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_fragment_coverage_to_color\0") };
pub const NV_FRAGMENT_COVERAGE_TO_COLOR_SPEC_VERSION: u32 = 1u32;
pub const KHR_ACCELERATION_STRUCTURE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_acceleration_structure\0") };
pub const KHR_ACCELERATION_STRUCTURE_SPEC_VERSION: u32 = 13u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCreateAccelerationStructureKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const AccelerationStructureCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_acceleration_structure: *mut AccelerationStructureKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyAccelerationStructureKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    acceleration_structure: AccelerationStructureKHR,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBuildAccelerationStructuresKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    info_count: u32,
    p_infos: *const AccelerationStructureBuildGeometryInfoKHR<'_>,
    pp_build_range_infos: *const *const AccelerationStructureBuildRangeInfoKHR,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBuildAccelerationStructuresIndirectKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    info_count: u32,
    p_infos: *const AccelerationStructureBuildGeometryInfoKHR<'_>,
    p_indirect_device_addresses: *const DeviceAddress,
    p_indirect_strides: *const u32,
    pp_max_primitive_counts: *const *const u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkBuildAccelerationStructuresKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    info_count: u32,
    p_infos: *const AccelerationStructureBuildGeometryInfoKHR<'_>,
    pp_build_range_infos: *const *const AccelerationStructureBuildRangeInfoKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyAccelerationStructureKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    p_info: *const CopyAccelerationStructureInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyAccelerationStructureToMemoryKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    p_info: *const CopyAccelerationStructureToMemoryInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyMemoryToAccelerationStructureKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    p_info: *const CopyMemoryToAccelerationStructureInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkWriteAccelerationStructuresPropertiesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    acceleration_structure_count: u32,
    p_acceleration_structures: *const AccelerationStructureKHR,
    query_type: QueryType,
    data_size: usize,
    p_data: *mut c_void,
    stride: usize,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyAccelerationStructureKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const CopyAccelerationStructureInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyAccelerationStructureToMemoryKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const CopyAccelerationStructureToMemoryInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMemoryToAccelerationStructureKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const CopyMemoryToAccelerationStructureInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetAccelerationStructureDeviceAddressKHR =
    unsafe extern "system" fn(
        device: crate::vk::Device,
        p_info: *const AccelerationStructureDeviceAddressInfoKHR<'_>,
    ) -> DeviceAddress;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdWriteAccelerationStructuresPropertiesKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    acceleration_structure_count: u32,
    p_acceleration_structures: *const AccelerationStructureKHR,
    query_type: QueryType,
    query_pool: QueryPool,
    first_query: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceAccelerationStructureCompatibilityKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_version_info: *const AccelerationStructureVersionInfoKHR<'_>,
    p_compatibility: *mut AccelerationStructureCompatibilityKHR,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetAccelerationStructureBuildSizesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    build_type: AccelerationStructureBuildTypeKHR,
    p_build_info: *const AccelerationStructureBuildGeometryInfoKHR<'_>,
    p_max_primitive_counts: *const u32,
    p_size_info: *mut AccelerationStructureBuildSizesInfoKHR<'_>,
);
pub const KHR_RAY_TRACING_PIPELINE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_ray_tracing_pipeline\0") };
pub const KHR_RAY_TRACING_PIPELINE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdTraceRaysKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_raygen_shader_binding_table: *const StridedDeviceAddressRegionKHR,
    p_miss_shader_binding_table: *const StridedDeviceAddressRegionKHR,
    p_hit_shader_binding_table: *const StridedDeviceAddressRegionKHR,
    p_callable_shader_binding_table: *const StridedDeviceAddressRegionKHR,
    width: u32,
    height: u32,
    depth: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateRayTracingPipelinesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    pipeline_cache: PipelineCache,
    create_info_count: u32,
    p_create_infos: *const RayTracingPipelineCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_pipelines: *mut Pipeline,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetRayTracingShaderGroupHandlesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    pipeline: Pipeline,
    first_group: u32,
    group_count: u32,
    data_size: usize,
    p_data: *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR =
    unsafe extern "system" fn(
        device: crate::vk::Device,
        pipeline: Pipeline,
        first_group: u32,
        group_count: u32,
        data_size: usize,
        p_data: *mut c_void,
    ) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdTraceRaysIndirectKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_raygen_shader_binding_table: *const StridedDeviceAddressRegionKHR,
    p_miss_shader_binding_table: *const StridedDeviceAddressRegionKHR,
    p_hit_shader_binding_table: *const StridedDeviceAddressRegionKHR,
    p_callable_shader_binding_table: *const StridedDeviceAddressRegionKHR,
    indirect_device_address: DeviceAddress,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetRayTracingShaderGroupStackSizeKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    pipeline: Pipeline,
    group: u32,
    group_shader: ShaderGroupShaderKHR,
) -> DeviceSize;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetRayTracingPipelineStackSizeKHR =
    unsafe extern "system" fn(command_buffer: CommandBuffer, pipeline_stack_size: u32);
pub const KHR_RAY_QUERY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_ray_query\0") };
pub const KHR_RAY_QUERY_SPEC_VERSION: u32 = 1u32;
pub const NV_FRAMEBUFFER_MIXED_SAMPLES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_framebuffer_mixed_samples\0") };
pub const NV_FRAMEBUFFER_MIXED_SAMPLES_SPEC_VERSION: u32 = 1u32;
pub const NV_FILL_RECTANGLE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_fill_rectangle\0") };
pub const NV_FILL_RECTANGLE_SPEC_VERSION: u32 = 1u32;
pub const NV_SHADER_SM_BUILTINS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_shader_sm_builtins\0") };
pub const NV_SHADER_SM_BUILTINS_SPEC_VERSION: u32 = 1u32;
pub const EXT_POST_DEPTH_COVERAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_post_depth_coverage\0") };
pub const EXT_POST_DEPTH_COVERAGE_SPEC_VERSION: u32 = 1u32;
pub const KHR_SAMPLER_YCBCR_CONVERSION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_sampler_ycbcr_conversion\0") };
pub const KHR_SAMPLER_YCBCR_CONVERSION_SPEC_VERSION: u32 = 14u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateSamplerYcbcrConversion = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const SamplerYcbcrConversionCreateInfo<'_>,
    p_allocator: *const AllocationCallbacks,
    p_ycbcr_conversion: *mut SamplerYcbcrConversion,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroySamplerYcbcrConversion = unsafe extern "system" fn(
    device: crate::vk::Device,
    ycbcr_conversion: SamplerYcbcrConversion,
    p_allocator: *const AllocationCallbacks,
);
pub const KHR_BIND_MEMORY2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_bind_memory2\0") };
pub const KHR_BIND_MEMORY2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkBindBufferMemory2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const BindBufferMemoryInfo<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkBindImageMemory2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const BindImageMemoryInfo<'_>,
) -> Result;
pub const EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_image_drm_format_modifier\0") };
pub const EXT_IMAGE_DRM_FORMAT_MODIFIER_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageDrmFormatModifierPropertiesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    image: Image,
    p_properties: *mut ImageDrmFormatModifierPropertiesEXT<'_>,
) -> Result;
pub const EXT_VALIDATION_CACHE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_validation_cache\0") };
pub const EXT_VALIDATION_CACHE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateValidationCacheEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const ValidationCacheCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_validation_cache: *mut ValidationCacheEXT,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyValidationCacheEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    validation_cache: ValidationCacheEXT,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkMergeValidationCachesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    dst_cache: ValidationCacheEXT,
    src_cache_count: u32,
    p_src_caches: *const ValidationCacheEXT,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetValidationCacheDataEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    validation_cache: ValidationCacheEXT,
    p_data_size: *mut usize,
    p_data: *mut c_void,
) -> Result;
pub const EXT_DESCRIPTOR_INDEXING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_descriptor_indexing\0") };
pub const EXT_DESCRIPTOR_INDEXING_SPEC_VERSION: u32 = 2u32;
pub const EXT_SHADER_VIEWPORT_INDEX_LAYER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_viewport_index_layer\0") };
pub const EXT_SHADER_VIEWPORT_INDEX_LAYER_SPEC_VERSION: u32 = 1u32;
#[cfg(feature = "provisional")]
pub const KHR_PORTABILITY_SUBSET_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_portability_subset\0") };
#[cfg(feature = "provisional")]
pub const KHR_PORTABILITY_SUBSET_SPEC_VERSION: u32 = 1u32;
pub const NV_SHADING_RATE_IMAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_shading_rate_image\0") };
pub const NV_SHADING_RATE_IMAGE_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindShadingRateImageNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    image_view: ImageView,
    image_layout: ImageLayout,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetViewportShadingRatePaletteNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_viewport: u32,
    viewport_count: u32,
    p_shading_rate_palettes: *const ShadingRatePaletteNV<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCoarseSampleOrderNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    sample_order_type: CoarseSampleOrderTypeNV,
    custom_sample_order_count: u32,
    p_custom_sample_orders: *const CoarseSampleOrderCustomNV<'_>,
);
pub const NV_RAY_TRACING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_ray_tracing\0") };
pub const NV_RAY_TRACING_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateAccelerationStructureNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const AccelerationStructureCreateInfoNV<'_>,
    p_allocator: *const AllocationCallbacks,
    p_acceleration_structure: *mut AccelerationStructureNV,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyAccelerationStructureNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    acceleration_structure: AccelerationStructureNV,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetAccelerationStructureMemoryRequirementsNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const AccelerationStructureMemoryRequirementsInfoNV<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkBindAccelerationStructureMemoryNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const BindAccelerationStructureMemoryInfoNV<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBuildAccelerationStructureNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const AccelerationStructureInfoNV<'_>,
    instance_data: Buffer,
    instance_offset: DeviceSize,
    update: Bool32,
    dst: AccelerationStructureNV,
    src: AccelerationStructureNV,
    scratch: Buffer,
    scratch_offset: DeviceSize,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyAccelerationStructureNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    dst: AccelerationStructureNV,
    src: AccelerationStructureNV,
    mode: CopyAccelerationStructureModeKHR,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdTraceRaysNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    raygen_shader_binding_table_buffer: Buffer,
    raygen_shader_binding_offset: DeviceSize,
    miss_shader_binding_table_buffer: Buffer,
    miss_shader_binding_offset: DeviceSize,
    miss_shader_binding_stride: DeviceSize,
    hit_shader_binding_table_buffer: Buffer,
    hit_shader_binding_offset: DeviceSize,
    hit_shader_binding_stride: DeviceSize,
    callable_shader_binding_table_buffer: Buffer,
    callable_shader_binding_offset: DeviceSize,
    callable_shader_binding_stride: DeviceSize,
    width: u32,
    height: u32,
    depth: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateRayTracingPipelinesNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    pipeline_cache: PipelineCache,
    create_info_count: u32,
    p_create_infos: *const RayTracingPipelineCreateInfoNV<'_>,
    p_allocator: *const AllocationCallbacks,
    p_pipelines: *mut Pipeline,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetAccelerationStructureHandleNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    acceleration_structure: AccelerationStructureNV,
    data_size: usize,
    p_data: *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdWriteAccelerationStructuresPropertiesNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    acceleration_structure_count: u32,
    p_acceleration_structures: *const AccelerationStructureNV,
    query_type: QueryType,
    query_pool: QueryPool,
    first_query: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCompileDeferredNV =
    unsafe extern "system" fn(device: crate::vk::Device, pipeline: Pipeline, shader: u32) -> Result;
pub const NV_REPRESENTATIVE_FRAGMENT_TEST_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_representative_fragment_test\0") };
pub const NV_REPRESENTATIVE_FRAGMENT_TEST_SPEC_VERSION: u32 = 2u32;
pub const KHR_MAINTENANCE3_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance3\0") };
pub const KHR_MAINTENANCE3_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDescriptorSetLayoutSupport = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const DescriptorSetLayoutCreateInfo<'_>,
    p_support: *mut DescriptorSetLayoutSupport<'_>,
);
pub const KHR_DRAW_INDIRECT_COUNT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_draw_indirect_count\0") };
pub const KHR_DRAW_INDIRECT_COUNT_SPEC_VERSION: u32 = 1u32;
pub const EXT_FILTER_CUBIC_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_filter_cubic\0") };
pub const EXT_FILTER_CUBIC_SPEC_VERSION: u32 = 3u32;
pub const QCOM_RENDER_PASS_SHADER_RESOLVE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_render_pass_shader_resolve\0") };
pub const QCOM_RENDER_PASS_SHADER_RESOLVE_SPEC_VERSION: u32 = 4u32;
pub const QCOM_COOPERATIVE_MATRIX_CONVERSION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_cooperative_matrix_conversion\0") };
pub const QCOM_COOPERATIVE_MATRIX_CONVERSION_SPEC_VERSION: u32 = 1u32;
pub const QCOM_ELAPSED_TIMER_QUERY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_elapsed_timer_query\0") };
pub const QCOM_ELAPSED_TIMER_QUERY_SPEC_VERSION: u32 = 1u32;
pub const EXT_GLOBAL_PRIORITY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_global_priority\0") };
pub const EXT_GLOBAL_PRIORITY_SPEC_VERSION: u32 = 2u32;
pub const KHR_SHADER_SUBGROUP_EXTENDED_TYPES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_subgroup_extended_types\0") };
pub const KHR_SHADER_SUBGROUP_EXTENDED_TYPES_SPEC_VERSION: u32 = 1u32;
pub const KHR_8BIT_STORAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_8bit_storage\0") };
pub const KHR_8BIT_STORAGE_SPEC_VERSION: u32 = 1u32;
pub const EXT_EXTERNAL_MEMORY_HOST_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_external_memory_host\0") };
pub const EXT_EXTERNAL_MEMORY_HOST_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryHostPointerPropertiesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    handle_type: ExternalMemoryHandleTypeFlags,
    p_host_pointer: *const c_void,
    p_memory_host_pointer_properties: *mut MemoryHostPointerPropertiesEXT<'_>,
) -> Result;
pub const AMD_BUFFER_MARKER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_buffer_marker\0") };
pub const AMD_BUFFER_MARKER_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdWriteBufferMarkerAMD = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    pipeline_stage: PipelineStageFlags,
    dst_buffer: Buffer,
    dst_offset: DeviceSize,
    marker: u32,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdWriteBufferMarker2AMD = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    stage: PipelineStageFlags2,
    dst_buffer: Buffer,
    dst_offset: DeviceSize,
    marker: u32,
);
pub const KHR_SHADER_ATOMIC_INT64_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_atomic_int64\0") };
pub const KHR_SHADER_ATOMIC_INT64_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_CLOCK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_clock\0") };
pub const KHR_SHADER_CLOCK_SPEC_VERSION: u32 = 1u32;
pub const AMD_PIPELINE_COMPILER_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_pipeline_compiler_control\0") };
pub const AMD_PIPELINE_COMPILER_CONTROL_SPEC_VERSION: u32 = 1u32;
pub const EXT_CALIBRATED_TIMESTAMPS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_calibrated_timestamps\0") };
pub const EXT_CALIBRATED_TIMESTAMPS_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_time_domain_count: *mut u32,
    p_time_domains: *mut TimeDomainKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetCalibratedTimestampsKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    timestamp_count: u32,
    p_timestamp_infos: *const CalibratedTimestampInfoKHR<'_>,
    p_timestamps: *mut u64,
    p_max_deviation: *mut u64,
) -> Result;
pub const AMD_SHADER_CORE_PROPERTIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_core_properties\0") };
pub const AMD_SHADER_CORE_PROPERTIES_SPEC_VERSION: u32 = 2u32;
pub const KHR_VIDEO_DECODE_H265_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_h265\0") };
pub const KHR_VIDEO_DECODE_H265_SPEC_VERSION: u32 = 8u32;
pub const KHR_GLOBAL_PRIORITY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_global_priority\0") };
pub const KHR_GLOBAL_PRIORITY_SPEC_VERSION: u32 = 1u32;
pub const AMD_MEMORY_OVERALLOCATION_BEHAVIOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_memory_overallocation_behavior\0") };
pub const AMD_MEMORY_OVERALLOCATION_BEHAVIOR_SPEC_VERSION: u32 = 1u32;
pub const EXT_VERTEX_ATTRIBUTE_DIVISOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_vertex_attribute_divisor\0") };
pub const EXT_VERTEX_ATTRIBUTE_DIVISOR_SPEC_VERSION: u32 = 3u32;
pub const GGP_FRAME_TOKEN_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_GGP_frame_token\0") };
pub const GGP_FRAME_TOKEN_SPEC_VERSION: u32 = 1u32;
pub const EXT_PIPELINE_CREATION_FEEDBACK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_pipeline_creation_feedback\0") };
pub const EXT_PIPELINE_CREATION_FEEDBACK_SPEC_VERSION: u32 = 1u32;
pub const KHR_DRIVER_PROPERTIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_driver_properties\0") };
pub const KHR_DRIVER_PROPERTIES_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_FLOAT_CONTROLS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_float_controls\0") };
pub const KHR_SHADER_FLOAT_CONTROLS_SPEC_VERSION: u32 = 4u32;
pub const NV_SHADER_SUBGROUP_PARTITIONED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_shader_subgroup_partitioned\0") };
pub const NV_SHADER_SUBGROUP_PARTITIONED_SPEC_VERSION: u32 = 1u32;
pub const KHR_DEPTH_STENCIL_RESOLVE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_depth_stencil_resolve\0") };
pub const KHR_DEPTH_STENCIL_RESOLVE_SPEC_VERSION: u32 = 1u32;
pub const KHR_SWAPCHAIN_MUTABLE_FORMAT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain_mutable_format\0") };
pub const KHR_SWAPCHAIN_MUTABLE_FORMAT_SPEC_VERSION: u32 = 1u32;
pub const NV_COMPUTE_SHADER_DERIVATIVES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_compute_shader_derivatives\0") };
pub const NV_COMPUTE_SHADER_DERIVATIVES_SPEC_VERSION: u32 = 1u32;
pub const NV_MESH_SHADER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_mesh_shader\0") };
pub const NV_MESH_SHADER_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMeshTasksNV =
    unsafe extern "system" fn(command_buffer: CommandBuffer, task_count: u32, first_task: u32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMeshTasksIndirectNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    draw_count: u32,
    stride: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMeshTasksIndirectCountNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    count_buffer: Buffer,
    count_buffer_offset: DeviceSize,
    max_draw_count: u32,
    stride: u32,
);
pub const NV_FRAGMENT_SHADER_BARYCENTRIC_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_fragment_shader_barycentric\0") };
pub const NV_FRAGMENT_SHADER_BARYCENTRIC_SPEC_VERSION: u32 = 1u32;
pub const NV_SHADER_IMAGE_FOOTPRINT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_shader_image_footprint\0") };
pub const NV_SHADER_IMAGE_FOOTPRINT_SPEC_VERSION: u32 = 2u32;
pub const NV_SCISSOR_EXCLUSIVE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_scissor_exclusive\0") };
pub const NV_SCISSOR_EXCLUSIVE_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetExclusiveScissorEnableNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_exclusive_scissor: u32,
    exclusive_scissor_count: u32,
    p_exclusive_scissor_enables: *const Bool32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetExclusiveScissorNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_exclusive_scissor: u32,
    exclusive_scissor_count: u32,
    p_exclusive_scissors: *const Rect2D,
);
pub const NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_device_diagnostic_checkpoints\0") };
pub const NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCheckpointNV =
    unsafe extern "system" fn(command_buffer: CommandBuffer, p_checkpoint_marker: *const c_void);
#[allow(non_camel_case_types)]
pub type PFN_vkGetQueueCheckpointDataNV = unsafe extern "system" fn(
    queue: Queue,
    p_checkpoint_data_count: *mut u32,
    p_checkpoint_data: *mut CheckpointDataNV<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetQueueCheckpointData2NV = unsafe extern "system" fn(
    queue: Queue,
    p_checkpoint_data_count: *mut u32,
    p_checkpoint_data: *mut CheckpointData2NV<'_>,
);
pub const KHR_TIMELINE_SEMAPHORE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_timeline_semaphore\0") };
pub const KHR_TIMELINE_SEMAPHORE_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSemaphoreCounterValue = unsafe extern "system" fn(
    device: crate::vk::Device,
    semaphore: Semaphore,
    p_value: *mut u64,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkWaitSemaphores = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_wait_info: *const SemaphoreWaitInfo<'_>,
    timeout: u64,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkSignalSemaphore = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_signal_info: *const SemaphoreSignalInfo<'_>,
) -> Result;
pub const EXT_PRESENT_TIMING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_present_timing\0") };
pub const EXT_PRESENT_TIMING_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkSetSwapchainPresentTimingQueueSizeEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    size: u32,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSwapchainTimingPropertiesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_swapchain_timing_properties: *mut SwapchainTimingPropertiesEXT<'_>,
    p_swapchain_timing_properties_counter: *mut u64,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSwapchainTimeDomainPropertiesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_swapchain_time_domain_properties: *mut SwapchainTimeDomainPropertiesEXT<'_>,
    p_time_domains_counter: *mut u64,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPastPresentationTimingEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_past_presentation_timing_info: *const PastPresentationTimingInfoEXT<'_>,
    p_past_presentation_timing_properties: *mut PastPresentationTimingPropertiesEXT<'_>,
) -> Result;
pub const INTEL_SHADER_INTEGER_FUNCTIONS2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_INTEL_shader_integer_functions2\0") };
pub const INTEL_SHADER_INTEGER_FUNCTIONS2_SPEC_VERSION: u32 = 1u32;
pub const INTEL_PERFORMANCE_QUERY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_INTEL_performance_query\0") };
pub const INTEL_PERFORMANCE_QUERY_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkInitializePerformanceApiINTEL = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_initialize_info: *const InitializePerformanceApiInfoINTEL<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkUninitializePerformanceApiINTEL =
    unsafe extern "system" fn(device: crate::vk::Device);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetPerformanceMarkerINTEL = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_marker_info: *const PerformanceMarkerInfoINTEL<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetPerformanceStreamMarkerINTEL = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_marker_info: *const PerformanceStreamMarkerInfoINTEL<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetPerformanceOverrideINTEL = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_override_info: *const PerformanceOverrideInfoINTEL<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkAcquirePerformanceConfigurationINTEL = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_acquire_info: *const PerformanceConfigurationAcquireInfoINTEL<'_>,
    p_configuration: *mut PerformanceConfigurationINTEL,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkReleasePerformanceConfigurationINTEL = unsafe extern "system" fn(
    device: crate::vk::Device,
    configuration: PerformanceConfigurationINTEL,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkQueueSetPerformanceConfigurationINTEL =
    unsafe extern "system" fn(queue: Queue, configuration: PerformanceConfigurationINTEL) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPerformanceParameterINTEL = unsafe extern "system" fn(
    device: crate::vk::Device,
    parameter: PerformanceParameterTypeINTEL,
    p_value: *mut PerformanceValueINTEL,
) -> Result;
pub const KHR_VULKAN_MEMORY_MODEL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_vulkan_memory_model\0") };
pub const KHR_VULKAN_MEMORY_MODEL_SPEC_VERSION: u32 = 3u32;
pub const EXT_PCI_BUS_INFO_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_pci_bus_info\0") };
pub const EXT_PCI_BUS_INFO_SPEC_VERSION: u32 = 2u32;
pub const AMD_DISPLAY_NATIVE_HDR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_display_native_hdr\0") };
pub const AMD_DISPLAY_NATIVE_HDR_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkSetLocalDimmingAMD = unsafe extern "system" fn(
    device: crate::vk::Device,
    swap_chain: SwapchainKHR,
    local_dimming_enable: Bool32,
);
pub const FUCHSIA_IMAGEPIPE_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_FUCHSIA_imagepipe_surface\0") };
pub const FUCHSIA_IMAGEPIPE_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateImagePipeSurfaceFUCHSIA = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const ImagePipeSurfaceCreateInfoFUCHSIA<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const KHR_SHADER_TERMINATE_INVOCATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_terminate_invocation\0") };
pub const KHR_SHADER_TERMINATE_INVOCATION_SPEC_VERSION: u32 = 1u32;
pub const EXT_METAL_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_metal_surface\0") };
pub const EXT_METAL_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateMetalSurfaceEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const MetalSurfaceCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const EXT_FRAGMENT_DENSITY_MAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_fragment_density_map\0") };
pub const EXT_FRAGMENT_DENSITY_MAP_SPEC_VERSION: u32 = 3u32;
pub const EXT_SCALAR_BLOCK_LAYOUT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_scalar_block_layout\0") };
pub const EXT_SCALAR_BLOCK_LAYOUT_SPEC_VERSION: u32 = 1u32;
pub const GOOGLE_HLSL_FUNCTIONALITY1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_hlsl_functionality1\0") };
pub const GOOGLE_HLSL_FUNCTIONALITY1_SPEC_VERSION: u32 = 1u32;
pub const GOOGLE_DECORATE_STRING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_decorate_string\0") };
pub const GOOGLE_DECORATE_STRING_SPEC_VERSION: u32 = 1u32;
pub const EXT_SUBGROUP_SIZE_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_subgroup_size_control\0") };
pub const EXT_SUBGROUP_SIZE_CONTROL_SPEC_VERSION: u32 = 2u32;
pub const KHR_FRAGMENT_SHADING_RATE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_fragment_shading_rate\0") };
pub const KHR_FRAGMENT_SHADING_RATE_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_fragment_shading_rate_count: *mut u32,
    p_fragment_shading_rates: *mut PhysicalDeviceFragmentShadingRateKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetFragmentShadingRateKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_fragment_size: *const Extent2D,
    combiner_ops: *const [FragmentShadingRateCombinerOpKHR; 2usize],
);
pub const AMD_SHADER_CORE_PROPERTIES2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_core_properties2\0") };
pub const AMD_SHADER_CORE_PROPERTIES2_SPEC_VERSION: u32 = 1u32;
pub const AMD_DEVICE_COHERENT_MEMORY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_device_coherent_memory\0") };
pub const AMD_DEVICE_COHERENT_MEMORY_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_CONSTANT_DATA_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_constant_data\0") };
pub const KHR_SHADER_CONSTANT_DATA_SPEC_VERSION: u32 = 1u32;
pub const KHR_DYNAMIC_RENDERING_LOCAL_READ_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_dynamic_rendering_local_read\0") };
pub const KHR_DYNAMIC_RENDERING_LOCAL_READ_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetRenderingAttachmentLocations = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_location_info: *const RenderingAttachmentLocationInfo<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetRenderingInputAttachmentIndices = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_input_attachment_index_info: *const RenderingInputAttachmentIndexInfo<'_>,
);
pub const KHR_SHADER_ABORT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_abort\0") };
pub const KHR_SHADER_ABORT_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_IMAGE_ATOMIC_INT64_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_image_atomic_int64\0") };
pub const EXT_SHADER_IMAGE_ATOMIC_INT64_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_QUAD_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_quad_control\0") };
pub const KHR_SHADER_QUAD_CONTROL_SPEC_VERSION: u32 = 1u32;
pub const KHR_SPIRV_1_4_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_spirv_1_4\0") };
pub const KHR_SPIRV_1_4_SPEC_VERSION: u32 = 1u32;
pub const EXT_MEMORY_BUDGET_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_memory_budget\0") };
pub const EXT_MEMORY_BUDGET_SPEC_VERSION: u32 = 1u32;
pub const EXT_MEMORY_PRIORITY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_memory_priority\0") };
pub const EXT_MEMORY_PRIORITY_SPEC_VERSION: u32 = 1u32;
pub const KHR_SURFACE_PROTECTED_CAPABILITIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_surface_protected_capabilities\0") };
pub const KHR_SURFACE_PROTECTED_CAPABILITIES_SPEC_VERSION: u32 = 1u32;
pub const NV_DEDICATED_ALLOCATION_IMAGE_ALIASING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_dedicated_allocation_image_aliasing\0") };
pub const NV_DEDICATED_ALLOCATION_IMAGE_ALIASING_SPEC_VERSION: u32 = 1u32;
pub const KHR_SEPARATE_DEPTH_STENCIL_LAYOUTS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_separate_depth_stencil_layouts\0") };
pub const KHR_SEPARATE_DEPTH_STENCIL_LAYOUTS_SPEC_VERSION: u32 = 1u32;
pub const EXT_BUFFER_DEVICE_ADDRESS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_buffer_device_address\0") };
pub const EXT_BUFFER_DEVICE_ADDRESS_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetBufferDeviceAddress = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const BufferDeviceAddressInfo<'_>,
) -> DeviceAddress;
pub const EXT_TOOLING_INFO_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_tooling_info\0") };
pub const EXT_TOOLING_INFO_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceToolProperties = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_tool_count: *mut u32,
    p_tool_properties: *mut PhysicalDeviceToolProperties<'_>,
) -> Result;
pub const EXT_SEPARATE_STENCIL_USAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_separate_stencil_usage\0") };
pub const EXT_SEPARATE_STENCIL_USAGE_SPEC_VERSION: u32 = 1u32;
pub const EXT_VALIDATION_FEATURES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_validation_features\0") };
pub const EXT_VALIDATION_FEATURES_SPEC_VERSION: u32 = 6u32;
pub const KHR_PRESENT_WAIT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_present_wait\0") };
pub const KHR_PRESENT_WAIT_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkWaitForPresentKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    present_id: u64,
    timeout: u64,
) -> Result;
pub const NV_COOPERATIVE_MATRIX_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_cooperative_matrix\0") };
pub const NV_COOPERATIVE_MATRIX_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_property_count: *mut u32,
    p_properties: *mut CooperativeMatrixPropertiesNV<'_>,
)
    -> Result;
pub const NV_COVERAGE_REDUCTION_MODE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_coverage_reduction_mode\0") };
pub const NV_COVERAGE_REDUCTION_MODE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_combination_count: *mut u32,
        p_combinations: *mut FramebufferMixedSamplesCombinationNV<'_>,
    ) -> Result;
pub const EXT_FRAGMENT_SHADER_INTERLOCK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_fragment_shader_interlock\0") };
pub const EXT_FRAGMENT_SHADER_INTERLOCK_SPEC_VERSION: u32 = 1u32;
pub const EXT_YCBCR_IMAGE_ARRAYS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_ycbcr_image_arrays\0") };
pub const EXT_YCBCR_IMAGE_ARRAYS_SPEC_VERSION: u32 = 1u32;
pub const KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_uniform_buffer_standard_layout\0") };
pub const KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_SPEC_VERSION: u32 = 1u32;
pub const EXT_PROVOKING_VERTEX_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_provoking_vertex\0") };
pub const EXT_PROVOKING_VERTEX_SPEC_VERSION: u32 = 1u32;
pub const EXT_FULL_SCREEN_EXCLUSIVE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_full_screen_exclusive\0") };
pub const EXT_FULL_SCREEN_EXCLUSIVE_SPEC_VERSION: u32 = 4u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_surface_info: *const PhysicalDeviceSurfaceInfo2KHR<'_>,
    p_present_mode_count: *mut u32,
    p_present_modes: *mut PresentModeKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkAcquireFullScreenExclusiveModeEXT =
    unsafe extern "system" fn(device: crate::vk::Device, swapchain: SwapchainKHR) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkReleaseFullScreenExclusiveModeEXT =
    unsafe extern "system" fn(device: crate::vk::Device, swapchain: SwapchainKHR) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceGroupSurfacePresentModes2EXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_surface_info: *const PhysicalDeviceSurfaceInfo2KHR<'_>,
    p_modes: *mut DeviceGroupPresentModeFlagsKHR,
) -> Result;
pub const EXT_HEADLESS_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_headless_surface\0") };
pub const EXT_HEADLESS_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateHeadlessSurfaceEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const HeadlessSurfaceCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const KHR_BUFFER_DEVICE_ADDRESS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_buffer_device_address\0") };
pub const KHR_BUFFER_DEVICE_ADDRESS_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetBufferOpaqueCaptureAddress = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const BufferDeviceAddressInfo<'_>,
) -> u64;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceMemoryOpaqueCaptureAddress = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const DeviceMemoryOpaqueCaptureAddressInfo<'_>,
) -> u64;
pub const EXT_LINE_RASTERIZATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_line_rasterization\0") };
pub const EXT_LINE_RASTERIZATION_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetLineStipple = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    line_stipple_factor: u32,
    line_stipple_pattern: u16,
);
pub const EXT_SHADER_ATOMIC_FLOAT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_atomic_float\0") };
pub const EXT_SHADER_ATOMIC_FLOAT_SPEC_VERSION: u32 = 1u32;
pub const EXT_HOST_QUERY_RESET_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_host_query_reset\0") };
pub const EXT_HOST_QUERY_RESET_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkResetQueryPool = unsafe extern "system" fn(
    device: crate::vk::Device,
    query_pool: QueryPool,
    first_query: u32,
    query_count: u32,
);
pub const EXT_INDEX_TYPE_UINT8_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_index_type_uint8\0") };
pub const EXT_INDEX_TYPE_UINT8_SPEC_VERSION: u32 = 1u32;
pub const EXT_EXTENDED_DYNAMIC_STATE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_extended_dynamic_state\0") };
pub const EXT_EXTENDED_DYNAMIC_STATE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCullMode =
    unsafe extern "system" fn(command_buffer: CommandBuffer, cull_mode: CullModeFlags);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetFrontFace =
    unsafe extern "system" fn(command_buffer: CommandBuffer, front_face: FrontFace);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetPrimitiveTopology =
    unsafe extern "system" fn(command_buffer: CommandBuffer, primitive_topology: PrimitiveTopology);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetViewportWithCount = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    viewport_count: u32,
    p_viewports: *const Viewport,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetScissorWithCount = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    scissor_count: u32,
    p_scissors: *const Rect2D,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindVertexBuffers2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_buffers: *const Buffer,
    p_offsets: *const DeviceSize,
    p_sizes: *const DeviceSize,
    p_strides: *const DeviceSize,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthTestEnable =
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_test_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthWriteEnable =
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_write_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthCompareOp =
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_compare_op: CompareOp);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthBoundsTestEnable =
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_bounds_test_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetStencilTestEnable =
    unsafe extern "system" fn(command_buffer: CommandBuffer, stencil_test_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetStencilOp = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    face_mask: StencilFaceFlags,
    fail_op: StencilOp,
    pass_op: StencilOp,
    depth_fail_op: StencilOp,
    compare_op: CompareOp,
);
pub const KHR_DEFERRED_HOST_OPERATIONS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_deferred_host_operations\0") };
pub const KHR_DEFERRED_HOST_OPERATIONS_SPEC_VERSION: u32 = 4u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDeferredOperationKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_allocator: *const AllocationCallbacks,
    p_deferred_operation: *mut DeferredOperationKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyDeferredOperationKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    operation: DeferredOperationKHR,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeferredOperationMaxConcurrencyKHR =
    unsafe extern "system" fn(device: crate::vk::Device, operation: DeferredOperationKHR) -> u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeferredOperationResultKHR =
    unsafe extern "system" fn(device: crate::vk::Device, operation: DeferredOperationKHR) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDeferredOperationJoinKHR =
    unsafe extern "system" fn(device: crate::vk::Device, operation: DeferredOperationKHR) -> Result;
pub const KHR_PIPELINE_EXECUTABLE_PROPERTIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_pipeline_executable_properties\0") };
pub const KHR_PIPELINE_EXECUTABLE_PROPERTIES_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPipelineExecutablePropertiesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_pipeline_info: *const PipelineInfoKHR<'_>,
    p_executable_count: *mut u32,
    p_properties: *mut PipelineExecutablePropertiesKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPipelineExecutableStatisticsKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_executable_info: *const PipelineExecutableInfoKHR<'_>,
    p_statistic_count: *mut u32,
    p_statistics: *mut PipelineExecutableStatisticKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPipelineExecutableInternalRepresentationsKHR =
    unsafe extern "system" fn(
        device: crate::vk::Device,
        p_executable_info: *const PipelineExecutableInfoKHR<'_>,
        p_internal_representation_count: *mut u32,
        p_internal_representations: *mut PipelineExecutableInternalRepresentationKHR<'_>,
    ) -> Result;
pub const EXT_HOST_IMAGE_COPY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_host_image_copy\0") };
pub const EXT_HOST_IMAGE_COPY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyMemoryToImage = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_copy_memory_to_image_info: *const CopyMemoryToImageInfo<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyImageToMemory = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_copy_image_to_memory_info: *const CopyImageToMemoryInfo<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyImageToImage = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_copy_image_to_image_info: *const CopyImageToImageInfo<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkTransitionImageLayout = unsafe extern "system" fn(
    device: crate::vk::Device,
    transition_count: u32,
    p_transitions: *const HostImageLayoutTransitionInfo<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageSubresourceLayout2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    image: Image,
    p_subresource: *const ImageSubresource2<'_>,
    p_layout: *mut SubresourceLayout2<'_>,
);
pub const KHR_MAP_MEMORY2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_map_memory2\0") };
pub const KHR_MAP_MEMORY2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkMapMemory2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_memory_map_info: *const MemoryMapInfo<'_>,
    pp_data: *mut *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkUnmapMemory2 = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_memory_unmap_info: *const MemoryUnmapInfo<'_>,
) -> Result;
pub const EXT_MAP_MEMORY_PLACED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_map_memory_placed\0") };
pub const EXT_MAP_MEMORY_PLACED_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_ATOMIC_FLOAT2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_atomic_float2\0") };
pub const EXT_SHADER_ATOMIC_FLOAT2_SPEC_VERSION: u32 = 1u32;
pub const EXT_SURFACE_MAINTENANCE1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_surface_maintenance1\0") };
pub const EXT_SURFACE_MAINTENANCE1_SPEC_VERSION: u32 = 1u32;
pub const EXT_SWAPCHAIN_MAINTENANCE1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_swapchain_maintenance1\0") };
pub const EXT_SWAPCHAIN_MAINTENANCE1_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkReleaseSwapchainImagesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_release_info: *const ReleaseSwapchainImagesInfoKHR<'_>,
) -> Result;
pub const EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_demote_to_helper_invocation\0") };
pub const EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_SPEC_VERSION: u32 = 1u32;
pub const NV_DEVICE_GENERATED_COMMANDS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_device_generated_commands\0") };
pub const NV_DEVICE_GENERATED_COMMANDS_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetGeneratedCommandsMemoryRequirementsNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const GeneratedCommandsMemoryRequirementsInfoNV<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPreprocessGeneratedCommandsNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_generated_commands_info: *const GeneratedCommandsInfoNV<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdExecuteGeneratedCommandsNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    is_preprocessed: Bool32,
    p_generated_commands_info: *const GeneratedCommandsInfoNV<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindPipelineShaderGroupNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    pipeline_bind_point: PipelineBindPoint,
    pipeline: Pipeline,
    group_index: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateIndirectCommandsLayoutNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const IndirectCommandsLayoutCreateInfoNV<'_>,
    p_allocator: *const AllocationCallbacks,
    p_indirect_commands_layout: *mut IndirectCommandsLayoutNV,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyIndirectCommandsLayoutNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    indirect_commands_layout: IndirectCommandsLayoutNV,
    p_allocator: *const AllocationCallbacks,
);
pub const NV_INHERITED_VIEWPORT_SCISSOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_inherited_viewport_scissor\0") };
pub const NV_INHERITED_VIEWPORT_SCISSOR_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_INTEGER_DOT_PRODUCT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_integer_dot_product\0") };
pub const KHR_SHADER_INTEGER_DOT_PRODUCT_SPEC_VERSION: u32 = 1u32;
pub const EXT_TEXEL_BUFFER_ALIGNMENT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_texel_buffer_alignment\0") };
pub const EXT_TEXEL_BUFFER_ALIGNMENT_SPEC_VERSION: u32 = 1u32;
pub const QCOM_RENDER_PASS_TRANSFORM_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_render_pass_transform\0") };
pub const QCOM_RENDER_PASS_TRANSFORM_SPEC_VERSION: u32 = 5u32;
pub const EXT_DEPTH_BIAS_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_depth_bias_control\0") };
pub const EXT_DEPTH_BIAS_CONTROL_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthBias2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_depth_bias_info: *const DepthBiasInfoEXT<'_>,
);
pub const EXT_DEVICE_MEMORY_REPORT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_device_memory_report\0") };
pub const EXT_DEVICE_MEMORY_REPORT_SPEC_VERSION: u32 = 2u32;
pub const EXT_ACQUIRE_DRM_DISPLAY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_acquire_drm_display\0") };
pub const EXT_ACQUIRE_DRM_DISPLAY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkAcquireDrmDisplayEXT = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    drm_fd: i32,
    display: DisplayKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDrmDisplayEXT = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    drm_fd: i32,
    connector_id: u32,
    display: *mut DisplayKHR,
) -> Result;
pub const EXT_ROBUSTNESS2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_robustness2\0") };
pub const EXT_ROBUSTNESS2_SPEC_VERSION: u32 = 1u32;
pub const EXT_CUSTOM_BORDER_COLOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_custom_border_color\0") };
pub const EXT_CUSTOM_BORDER_COLOR_SPEC_VERSION: u32 = 12u32;
pub const EXT_TEXTURE_COMPRESSION_ASTC_3D_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_texture_compression_astc_3d\0") };
pub const EXT_TEXTURE_COMPRESSION_ASTC_3D_SPEC_VERSION: u32 = 1u32;
pub const GOOGLE_USER_TYPE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_user_type\0") };
pub const GOOGLE_USER_TYPE_SPEC_VERSION: u32 = 1u32;
pub const KHR_PIPELINE_LIBRARY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_pipeline_library\0") };
pub const KHR_PIPELINE_LIBRARY_SPEC_VERSION: u32 = 1u32;
pub const NV_PRESENT_BARRIER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_present_barrier\0") };
pub const NV_PRESENT_BARRIER_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_NON_SEMANTIC_INFO_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_non_semantic_info\0") };
pub const KHR_SHADER_NON_SEMANTIC_INFO_SPEC_VERSION: u32 = 1u32;
pub const KHR_PRESENT_ID_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_present_id\0") };
pub const KHR_PRESENT_ID_SPEC_VERSION: u32 = 1u32;
pub const EXT_PRIVATE_DATA_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_private_data\0") };
pub const EXT_PRIVATE_DATA_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreatePrivateDataSlot = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const PrivateDataSlotCreateInfo<'_>,
    p_allocator: *const AllocationCallbacks,
    p_private_data_slot: *mut PrivateDataSlot,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyPrivateDataSlot = unsafe extern "system" fn(
    device: crate::vk::Device,
    private_data_slot: PrivateDataSlot,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkSetPrivateData = unsafe extern "system" fn(
    device: crate::vk::Device,
    object_type: ObjectType,
    object_handle: u64,
    private_data_slot: PrivateDataSlot,
    data: u64,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPrivateData = unsafe extern "system" fn(
    device: crate::vk::Device,
    object_type: ObjectType,
    object_handle: u64,
    private_data_slot: PrivateDataSlot,
    p_data: *mut u64,
);
pub const EXT_PIPELINE_CREATION_CACHE_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_pipeline_creation_cache_control\0") };
pub const EXT_PIPELINE_CREATION_CACHE_CONTROL_SPEC_VERSION: u32 = 3u32;
pub const KHR_VIDEO_ENCODE_QUEUE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_queue\0") };
pub const KHR_VIDEO_ENCODE_QUEUE_SPEC_VERSION: u32 = 12u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_quality_level_info: *const PhysicalDeviceVideoEncodeQualityLevelInfoKHR<'_>,
        p_quality_level_properties: *mut VideoEncodeQualityLevelPropertiesKHR<'_>,
    ) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetEncodedVideoSessionParametersKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_video_session_parameters_info: *const VideoEncodeSessionParametersGetInfoKHR<'_>,
    p_feedback_info: *mut VideoEncodeSessionParametersFeedbackInfoKHR<'_>,
    p_data_size: *mut usize,
    p_data: *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEncodeVideoKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_encode_info: *const VideoEncodeInfoKHR<'_>,
);
pub const NV_DEVICE_DIAGNOSTICS_CONFIG_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_device_diagnostics_config\0") };
pub const NV_DEVICE_DIAGNOSTICS_CONFIG_SPEC_VERSION: u32 = 2u32;
pub const QCOM_RENDER_PASS_STORE_OPS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_render_pass_store_ops\0") };
pub const QCOM_RENDER_PASS_STORE_OPS_SPEC_VERSION: u32 = 2u32;
pub const QCOM_QUEUE_PERF_HINT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_queue_perf_hint\0") };
pub const QCOM_QUEUE_PERF_HINT_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkQueueSetPerfHintQCOM = unsafe extern "system" fn(
    queue: Queue,
    p_perf_hint_info: *const PerfHintInfoQCOM<'_>,
) -> Result;
pub const QCOM_IMAGE_PROCESSING3_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_image_processing3\0") };
pub const QCOM_IMAGE_PROCESSING3_SPEC_VERSION: u32 = 1u32;
pub const QCOM_SHADER_MULTIPLE_WAIT_QUEUES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_shader_multiple_wait_queues\0") };
pub const QCOM_SHADER_MULTIPLE_WAIT_QUEUES_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_SPLIT_BARRIER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_split_barrier\0") };
pub const EXT_SHADER_SPLIT_BARRIER_SPEC_VERSION: u32 = 1u32;
#[cfg(feature = "provisional")]
pub const NV_CUDA_KERNEL_LAUNCH_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_cuda_kernel_launch\0") };
#[cfg(feature = "provisional")]
pub const NV_CUDA_KERNEL_LAUNCH_SPEC_VERSION: u32 = 2u32;
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkCreateCudaModuleNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const CudaModuleCreateInfoNV<'_>,
    p_allocator: *const AllocationCallbacks,
    p_module: *mut CudaModuleNV,
) -> Result;
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkGetCudaModuleCacheNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    module: CudaModuleNV,
    p_cache_size: *mut usize,
    p_cache_data: *mut c_void,
) -> Result;
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkCreateCudaFunctionNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const CudaFunctionCreateInfoNV<'_>,
    p_allocator: *const AllocationCallbacks,
    p_function: *mut CudaFunctionNV,
) -> Result;
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyCudaModuleNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    module: CudaModuleNV,
    p_allocator: *const AllocationCallbacks,
);
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyCudaFunctionNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    function: CudaFunctionNV,
    p_allocator: *const AllocationCallbacks,
);
#[cfg(feature = "provisional")]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCudaLaunchKernelNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_launch_info: *const CudaLaunchInfoNV<'_>,
);
pub const QCOM_TILE_SHADING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_tile_shading\0") };
pub const QCOM_TILE_SHADING_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDispatchTileQCOM = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_dispatch_tile_info: *const DispatchTileInfoQCOM<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginPerTileExecutionQCOM = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_per_tile_begin_info: *const PerTileBeginInfoQCOM<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndPerTileExecutionQCOM = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_per_tile_end_info: *const PerTileEndInfoQCOM<'_>,
);
pub const NV_LOW_LATENCY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_low_latency\0") };
pub const NV_LOW_LATENCY_SPEC_VERSION: u32 = 1u32;
pub const EXT_METAL_OBJECTS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_metal_objects\0") };
pub const EXT_METAL_OBJECTS_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkExportMetalObjectsEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_metal_objects_info: *mut ExportMetalObjectsInfoEXT<'_>,
);
pub const KHR_SYNCHRONIZATION2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_synchronization2\0") };
pub const KHR_SYNCHRONIZATION2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetEvent2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    event: Event,
    p_dependency_info: *const DependencyInfo<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdResetEvent2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    event: Event,
    stage_mask: PipelineStageFlags2,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdWaitEvents2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    event_count: u32,
    p_events: *const Event,
    p_dependency_infos: *const DependencyInfo<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPipelineBarrier2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_dependency_info: *const DependencyInfo<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdWriteTimestamp2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    stage: PipelineStageFlags2,
    query_pool: QueryPool,
    query: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkQueueSubmit2 = unsafe extern "system" fn(
    queue: Queue,
    submit_count: u32,
    p_submits: *const SubmitInfo2<'_>,
    fence: Fence,
) -> Result;
pub const EXT_DESCRIPTOR_BUFFER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_descriptor_buffer\0") };
pub const EXT_DESCRIPTOR_BUFFER_SPEC_VERSION: u32 = 1u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkGetDescriptorSetLayoutSizeEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    layout: DescriptorSetLayout,
    p_layout_size_in_bytes: *mut DeviceSize,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkGetDescriptorSetLayoutBindingOffsetEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    layout: DescriptorSetLayout,
    binding: u32,
    p_offset: *mut DeviceSize,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkGetDescriptorEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_descriptor_info: *const DescriptorGetInfoEXT<'_>,
    data_size: usize,
    p_descriptor: *mut c_void,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindDescriptorBuffersEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    buffer_count: u32,
    p_binding_infos: *const DescriptorBufferBindingInfoEXT<'_>,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDescriptorBufferOffsetsEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    pipeline_bind_point: PipelineBindPoint,
    layout: PipelineLayout,
    first_set: u32,
    set_count: u32,
    p_buffer_indices: *const u32,
    p_offsets: *const DeviceSize,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    pipeline_bind_point: PipelineBindPoint,
    layout: PipelineLayout,
    set: u32,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const BufferCaptureDescriptorDataInfoEXT<'_>,
    p_data: *mut c_void,
) -> Result;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageOpaqueCaptureDescriptorDataEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ImageCaptureDescriptorDataInfoEXT<'_>,
    p_data: *mut c_void,
) -> Result;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ImageViewCaptureDescriptorDataInfoEXT<'_>,
    p_data: *mut c_void,
) -> Result;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const SamplerCaptureDescriptorDataInfoEXT<'_>,
    p_data: *mut c_void,
) -> Result;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT =
    unsafe extern "system" fn(
        device: crate::vk::Device,
        p_info: *const AccelerationStructureCaptureDescriptorDataInfoEXT<'_>,
        p_data: *mut c_void,
    ) -> Result;
pub const KHR_DEVICE_ADDRESS_COMMANDS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_device_address_commands\0") };
pub const KHR_DEVICE_ADDRESS_COMMANDS_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindIndexBuffer3KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const BindIndexBuffer3InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindVertexBuffers3KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_binding_infos: *const BindVertexBuffer3InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawIndirect2KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const DrawIndirect2InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawIndexedIndirect2KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const DrawIndirect2InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDispatchIndirect2KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const DispatchIndirect2InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMemoryKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_memory_info: *const CopyDeviceMemoryInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMemoryToImageKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_memory_info: *const CopyDeviceMemoryImageInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyImageToMemoryKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_memory_info: *const CopyDeviceMemoryImageInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdUpdateMemoryKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_dst_range: *const DeviceAddressRangeKHR,
    dst_flags: AddressCommandFlagsKHR,
    data_size: DeviceSize,
    p_data: *const c_void,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdFillMemoryKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_dst_range: *const DeviceAddressRangeKHR,
    dst_flags: AddressCommandFlagsKHR,
    data: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyQueryPoolResultsToMemoryKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    query_pool: QueryPool,
    first_query: u32,
    query_count: u32,
    p_dst_range: *const StridedDeviceAddressRangeKHR,
    dst_flags: AddressCommandFlagsKHR,
    query_result_flags: QueryResultFlags,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawIndirectCount2KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const DrawIndirectCount2InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawIndexedIndirectCount2KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const DrawIndirectCount2InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginConditionalRendering2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_conditional_rendering_begin: *const ConditionalRenderingBeginInfo2EXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindTransformFeedbackBuffers2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_binding_infos: *const BindTransformFeedbackBuffer2InfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginTransformFeedback2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_counter_range: u32,
    counter_range_count: u32,
    p_counter_infos: *const BindTransformFeedbackBuffer2InfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndTransformFeedback2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_counter_range: u32,
    counter_range_count: u32,
    p_counter_infos: *const BindTransformFeedbackBuffer2InfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawIndirectByteCount2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    instance_count: u32,
    first_instance: u32,
    p_counter_info: *const BindTransformFeedbackBuffer2InfoEXT<'_>,
    counter_offset: u32,
    vertex_stride: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMeshTasksIndirect2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const DrawIndirect2InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMeshTasksIndirectCount2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const DrawIndirectCount2InfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdWriteMarkerToMemoryAMD = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const MemoryMarkerInfoAMD<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateAccelerationStructure2KHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const AccelerationStructureCreateInfo2KHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_acceleration_structure: *mut AccelerationStructureKHR,
) -> Result;
pub const EXT_GRAPHICS_PIPELINE_LIBRARY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_graphics_pipeline_library\0") };
pub const EXT_GRAPHICS_PIPELINE_LIBRARY_SPEC_VERSION: u32 = 1u32;
pub const AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_AMD_shader_early_and_late_fragment_tests\0")
};
pub const AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_SPEC_VERSION: u32 = 1u32;
pub const KHR_FRAGMENT_SHADER_BARYCENTRIC_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_fragment_shader_barycentric\0") };
pub const KHR_FRAGMENT_SHADER_BARYCENTRIC_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_subgroup_uniform_control_flow\0")
};
pub const KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_SPEC_VERSION: u32 = 1u32;
pub const KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_zero_initialize_workgroup_memory\0") };
pub const KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_SPEC_VERSION: u32 = 1u32;
pub const NV_FRAGMENT_SHADING_RATE_ENUMS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_fragment_shading_rate_enums\0") };
pub const NV_FRAGMENT_SHADING_RATE_ENUMS_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetFragmentShadingRateEnumNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    shading_rate: FragmentShadingRateNV,
    combiner_ops: *const [FragmentShadingRateCombinerOpKHR; 2usize],
);
pub const NV_RAY_TRACING_MOTION_BLUR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_ray_tracing_motion_blur\0") };
pub const NV_RAY_TRACING_MOTION_BLUR_SPEC_VERSION: u32 = 1u32;
pub const EXT_MESH_SHADER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_mesh_shader\0") };
pub const EXT_MESH_SHADER_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMeshTasksEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMeshTasksIndirectEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    draw_count: u32,
    stride: u32,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMeshTasksIndirectCountEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    count_buffer: Buffer,
    count_buffer_offset: DeviceSize,
    max_draw_count: u32,
    stride: u32,
);
pub const EXT_YCBCR_2PLANE_444_FORMATS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_ycbcr_2plane_444_formats\0") };
pub const EXT_YCBCR_2PLANE_444_FORMATS_SPEC_VERSION: u32 = 1u32;
pub const EXT_FRAGMENT_DENSITY_MAP2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_fragment_density_map2\0") };
pub const EXT_FRAGMENT_DENSITY_MAP2_SPEC_VERSION: u32 = 1u32;
pub const QCOM_ROTATED_COPY_COMMANDS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_rotated_copy_commands\0") };
pub const QCOM_ROTATED_COPY_COMMANDS_SPEC_VERSION: u32 = 2u32;
pub const EXT_IMAGE_ROBUSTNESS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_image_robustness\0") };
pub const EXT_IMAGE_ROBUSTNESS_SPEC_VERSION: u32 = 1u32;
pub const KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_workgroup_memory_explicit_layout\0") };
pub const KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_SPEC_VERSION: u32 = 1u32;
pub const KHR_COPY_COMMANDS2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_copy_commands2\0") };
pub const KHR_COPY_COMMANDS2_SPEC_VERSION: u32 = 1u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyBuffer2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_buffer_info: *const CopyBufferInfo2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyImage2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_image_info: *const CopyImageInfo2<'_>,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyBufferToImage2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_buffer_to_image_info: *const CopyBufferToImageInfo2<'_>,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyImageToBuffer2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_image_to_buffer_info: *const CopyImageToBufferInfo2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBlitImage2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_blit_image_info: *const BlitImageInfo2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdResolveImage2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_resolve_image_info: *const ResolveImageInfo2<'_>,
);
pub const EXT_IMAGE_COMPRESSION_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_image_compression_control\0") };
pub const EXT_IMAGE_COMPRESSION_CONTROL_SPEC_VERSION: u32 = 1u32;
pub const EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_attachment_feedback_loop_layout\0") };
pub const EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_SPEC_VERSION: u32 = 2u32;
pub const EXT_4444_FORMATS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_4444_formats\0") };
pub const EXT_4444_FORMATS_SPEC_VERSION: u32 = 1u32;
pub const EXT_DEVICE_FAULT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_device_fault\0") };
pub const EXT_DEVICE_FAULT_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceFaultInfoEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_fault_counts: *mut DeviceFaultCountsEXT<'_>,
    p_fault_info: *mut DeviceFaultInfoEXT<'_>,
) -> Result;
pub const ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_ARM_rasterization_order_attachment_access\0")
};
pub const ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_SPEC_VERSION: u32 = 1u32;
pub const EXT_RGBA10X6_FORMATS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_rgba10x6_formats\0") };
pub const EXT_RGBA10X6_FORMATS_SPEC_VERSION: u32 = 1u32;
pub const NV_ACQUIRE_WINRT_DISPLAY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_acquire_winrt_display\0") };
pub const NV_ACQUIRE_WINRT_DISPLAY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkAcquireWinrtDisplayNV =
    unsafe extern "system" fn(physical_device: PhysicalDevice, display: DisplayKHR) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetWinrtDisplayNV = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    device_relative_id: u32,
    p_display: *mut DisplayKHR,
) -> Result;
pub const EXT_DIRECTFB_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_directfb_surface\0") };
pub const EXT_DIRECTFB_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDirectFBSurfaceEXT = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const DirectFBSurfaceCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        dfb: *mut IDirectFB,
    ) -> Bool32;
pub const VALVE_MUTABLE_DESCRIPTOR_TYPE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_VALVE_mutable_descriptor_type\0") };
pub const VALVE_MUTABLE_DESCRIPTOR_TYPE_SPEC_VERSION: u32 = 1u32;
pub const EXT_VERTEX_INPUT_DYNAMIC_STATE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_vertex_input_dynamic_state\0") };
pub const EXT_VERTEX_INPUT_DYNAMIC_STATE_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetVertexInputEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    vertex_binding_description_count: u32,
    p_vertex_binding_descriptions: *const VertexInputBindingDescription2EXT<'_>,
    vertex_attribute_description_count: u32,
    p_vertex_attribute_descriptions: *const VertexInputAttributeDescription2EXT<'_>,
);
pub const EXT_PHYSICAL_DEVICE_DRM_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_physical_device_drm\0") };
pub const EXT_PHYSICAL_DEVICE_DRM_SPEC_VERSION: u32 = 1u32;
pub const EXT_DEVICE_ADDRESS_BINDING_REPORT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_device_address_binding_report\0") };
pub const EXT_DEVICE_ADDRESS_BINDING_REPORT_SPEC_VERSION: u32 = 1u32;
pub const EXT_DEPTH_CLIP_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_depth_clip_control\0") };
pub const EXT_DEPTH_CLIP_CONTROL_SPEC_VERSION: u32 = 1u32;
pub const EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_primitive_topology_list_restart\0") };
pub const EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_SPEC_VERSION: u32 = 1u32;
pub const KHR_FORMAT_FEATURE_FLAGS2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_format_feature_flags2\0") };
pub const KHR_FORMAT_FEATURE_FLAGS2_SPEC_VERSION: u32 = 2u32;
pub const EXT_PRESENT_MODE_FIFO_LATEST_READY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_present_mode_fifo_latest_ready\0") };
pub const EXT_PRESENT_MODE_FIFO_LATEST_READY_SPEC_VERSION: u32 = 1u32;
pub const FUCHSIA_EXTERNAL_MEMORY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_FUCHSIA_external_memory\0") };
pub const FUCHSIA_EXTERNAL_MEMORY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryZirconHandleFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_zircon_handle_info: *const MemoryGetZirconHandleInfoFUCHSIA<'_>,
    p_zircon_handle: *mut zx_handle_t,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    handle_type: ExternalMemoryHandleTypeFlags,
    zircon_handle: zx_handle_t,
    p_memory_zircon_handle_properties: *mut MemoryZirconHandlePropertiesFUCHSIA<'_>,
) -> Result;
pub const FUCHSIA_EXTERNAL_SEMAPHORE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_FUCHSIA_external_semaphore\0") };
pub const FUCHSIA_EXTERNAL_SEMAPHORE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkImportSemaphoreZirconHandleFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_import_semaphore_zircon_handle_info: *const ImportSemaphoreZirconHandleInfoFUCHSIA<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetSemaphoreZirconHandleFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_zircon_handle_info: *const SemaphoreGetZirconHandleInfoFUCHSIA<'_>,
    p_zircon_handle: *mut zx_handle_t,
) -> Result;
pub const FUCHSIA_BUFFER_COLLECTION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_FUCHSIA_buffer_collection\0") };
pub const FUCHSIA_BUFFER_COLLECTION_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateBufferCollectionFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const BufferCollectionCreateInfoFUCHSIA<'_>,
    p_allocator: *const AllocationCallbacks,
    p_collection: *mut BufferCollectionFUCHSIA,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkSetBufferCollectionImageConstraintsFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    collection: BufferCollectionFUCHSIA,
    p_image_constraints_info: *const ImageConstraintsInfoFUCHSIA<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    collection: BufferCollectionFUCHSIA,
    p_buffer_constraints_info: *const BufferConstraintsInfoFUCHSIA<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyBufferCollectionFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    collection: BufferCollectionFUCHSIA,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetBufferCollectionPropertiesFUCHSIA = unsafe extern "system" fn(
    device: crate::vk::Device,
    collection: BufferCollectionFUCHSIA,
    p_properties: *mut BufferCollectionPropertiesFUCHSIA<'_>,
) -> Result;
pub const HUAWEI_SUBPASS_SHADING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_HUAWEI_subpass_shading\0") };
pub const HUAWEI_SUBPASS_SHADING_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI = unsafe extern "system" fn(
    device: crate::vk::Device,
    renderpass: RenderPass,
    p_max_workgroup_size: *mut Extent2D,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSubpassShadingHUAWEI = unsafe extern "system" fn(command_buffer: CommandBuffer);
pub const HUAWEI_INVOCATION_MASK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_HUAWEI_invocation_mask\0") };
pub const HUAWEI_INVOCATION_MASK_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindInvocationMaskHUAWEI = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    image_view: ImageView,
    image_layout: ImageLayout,
);
pub const NV_EXTERNAL_MEMORY_RDMA_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_external_memory_rdma\0") };
pub const NV_EXTERNAL_MEMORY_RDMA_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryRemoteAddressNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_memory_get_remote_address_info: *const MemoryGetRemoteAddressInfoNV<'_>,
    p_address: *mut RemoteAddressNV,
) -> Result;
pub const EXT_PIPELINE_PROPERTIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_pipeline_properties\0") };
pub const EXT_PIPELINE_PROPERTIES_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPipelinePropertiesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_pipeline_info: *const PipelineInfoKHR<'_>,
    p_pipeline_properties: *mut BaseOutStructure<'_>,
) -> Result;
pub const EXT_FRAME_BOUNDARY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_frame_boundary\0") };
pub const EXT_FRAME_BOUNDARY_SPEC_VERSION: u32 = 1u32;
pub const EXT_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_EXT_multisampled_render_to_single_sampled\0")
};
pub const EXT_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_SPEC_VERSION: u32 = 1u32;
pub const EXT_EXTENDED_DYNAMIC_STATE2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_extended_dynamic_state2\0") };
pub const EXT_EXTENDED_DYNAMIC_STATE2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetPatchControlPointsEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, patch_control_points: u32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetRasterizerDiscardEnable =
    unsafe extern "system" fn(command_buffer: CommandBuffer, rasterizer_discard_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthBiasEnable =
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_bias_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetLogicOpEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, logic_op: LogicOp);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetPrimitiveRestartEnable =
    unsafe extern "system" fn(command_buffer: CommandBuffer, primitive_restart_enable: Bool32);
pub const QNX_SCREEN_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QNX_screen_surface\0") };
pub const QNX_SCREEN_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateScreenSurfaceQNX = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const ScreenSurfaceCreateInfoQNX<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    queue_family_index: u32,
    window: *mut _screen_window,
) -> Bool32;
pub const EXT_COLOR_WRITE_ENABLE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_color_write_enable\0") };
pub const EXT_COLOR_WRITE_ENABLE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetColorWriteEnableEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    attachment_count: u32,
    p_color_write_enables: *const Bool32,
);
pub const EXT_PRIMITIVES_GENERATED_QUERY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_primitives_generated_query\0") };
pub const EXT_PRIMITIVES_GENERATED_QUERY_SPEC_VERSION: u32 = 1u32;
pub const KHR_RAY_TRACING_MAINTENANCE1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_ray_tracing_maintenance1\0") };
pub const KHR_RAY_TRACING_MAINTENANCE1_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdTraceRaysIndirect2KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    indirect_device_address: DeviceAddress,
);
pub const KHR_SHADER_UNTYPED_POINTERS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_untyped_pointers\0") };
pub const KHR_SHADER_UNTYPED_POINTERS_SPEC_VERSION: u32 = 1u32;
pub const EXT_GLOBAL_PRIORITY_QUERY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_global_priority_query\0") };
pub const EXT_GLOBAL_PRIORITY_QUERY_SPEC_VERSION: u32 = 1u32;
pub const VALVE_VIDEO_ENCODE_RGB_CONVERSION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_VALVE_video_encode_rgb_conversion\0") };
pub const VALVE_VIDEO_ENCODE_RGB_CONVERSION_SPEC_VERSION: u32 = 1u32;
pub const EXT_IMAGE_VIEW_MIN_LOD_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_image_view_min_lod\0") };
pub const EXT_IMAGE_VIEW_MIN_LOD_SPEC_VERSION: u32 = 1u32;
pub const EXT_MULTI_DRAW_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_multi_draw\0") };
pub const EXT_MULTI_DRAW_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMultiEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    draw_count: u32,
    p_vertex_info: *const MultiDrawInfoEXT,
    instance_count: u32,
    first_instance: u32,
    stride: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawMultiIndexedEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    draw_count: u32,
    p_index_info: *const MultiDrawIndexedInfoEXT,
    instance_count: u32,
    first_instance: u32,
    stride: u32,
    p_vertex_offset: *const i32,
);
pub const EXT_IMAGE_2D_VIEW_OF_3D_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_image_2d_view_of_3d\0") };
pub const EXT_IMAGE_2D_VIEW_OF_3D_SPEC_VERSION: u32 = 1u32;
pub const KHR_PORTABILITY_ENUMERATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_portability_enumeration\0") };
pub const KHR_PORTABILITY_ENUMERATION_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_TILE_IMAGE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_tile_image\0") };
pub const EXT_SHADER_TILE_IMAGE_SPEC_VERSION: u32 = 1u32;
pub const EXT_OPACITY_MICROMAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_opacity_micromap\0") };
pub const EXT_OPACITY_MICROMAP_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateMicromapEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const MicromapCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_micromap: *mut MicromapEXT,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyMicromapEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    micromap: MicromapEXT,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBuildMicromapsEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    info_count: u32,
    p_infos: *const MicromapBuildInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkBuildMicromapsEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    info_count: u32,
    p_infos: *const MicromapBuildInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyMicromapEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    p_info: *const CopyMicromapInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyMicromapToMemoryEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    p_info: *const CopyMicromapToMemoryInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCopyMemoryToMicromapEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    p_info: *const CopyMemoryToMicromapInfoEXT<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkWriteMicromapsPropertiesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    micromap_count: u32,
    p_micromaps: *const MicromapEXT,
    query_type: QueryType,
    data_size: usize,
    p_data: *mut c_void,
    stride: usize,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMicromapEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const CopyMicromapInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMicromapToMemoryEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const CopyMicromapToMemoryInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMemoryToMicromapEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_info: *const CopyMemoryToMicromapInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdWriteMicromapsPropertiesEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    micromap_count: u32,
    p_micromaps: *const MicromapEXT,
    query_type: QueryType,
    query_pool: QueryPool,
    first_query: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceMicromapCompatibilityEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_version_info: *const MicromapVersionInfoEXT<'_>,
    p_compatibility: *mut AccelerationStructureCompatibilityKHR,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetMicromapBuildSizesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    build_type: AccelerationStructureBuildTypeKHR,
    p_build_info: *const MicromapBuildInfoEXT<'_>,
    p_size_info: *mut MicromapBuildSizesInfoEXT<'_>,
);
#[cfg(feature = "provisional")]
pub const NV_DISPLACEMENT_MICROMAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_displacement_micromap\0") };
#[cfg(feature = "provisional")]
pub const NV_DISPLACEMENT_MICROMAP_SPEC_VERSION: u32 = 2u32;
pub const EXT_LOAD_STORE_OP_NONE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_load_store_op_none\0") };
pub const EXT_LOAD_STORE_OP_NONE_SPEC_VERSION: u32 = 1u32;
pub const HUAWEI_CLUSTER_CULLING_SHADER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_HUAWEI_cluster_culling_shader\0") };
pub const HUAWEI_CLUSTER_CULLING_SHADER_SPEC_VERSION: u32 = 3u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawClusterHUAWEI = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDrawClusterIndirectHUAWEI =
    unsafe extern "system" fn(command_buffer: CommandBuffer, buffer: Buffer, offset: DeviceSize);
pub const EXT_BORDER_COLOR_SWIZZLE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_border_color_swizzle\0") };
pub const EXT_BORDER_COLOR_SWIZZLE_SPEC_VERSION: u32 = 1u32;
pub const EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_pageable_device_local_memory\0") };
pub const EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkSetDeviceMemoryPriorityEXT =
    unsafe extern "system" fn(device: crate::vk::Device, memory: DeviceMemory, priority: f32);
pub const KHR_MAINTENANCE4_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance4\0") };
pub const KHR_MAINTENANCE4_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceBufferMemoryRequirements = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const DeviceBufferMemoryRequirements<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceImageMemoryRequirements = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const DeviceImageMemoryRequirements<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceImageSparseMemoryRequirements = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const DeviceImageMemoryRequirements<'_>,
    p_sparse_memory_requirement_count: *mut u32,
    p_sparse_memory_requirements: *mut SparseImageMemoryRequirements2<'_>,
);
pub const ARM_SHADER_CORE_PROPERTIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_shader_core_properties\0") };
pub const ARM_SHADER_CORE_PROPERTIES_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_SUBGROUP_ROTATE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_subgroup_rotate\0") };
pub const KHR_SHADER_SUBGROUP_ROTATE_SPEC_VERSION: u32 = 2u32;
pub const ARM_SCHEDULING_CONTROLS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_scheduling_controls\0") };
pub const ARM_SCHEDULING_CONTROLS_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDispatchParametersARM = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_dispatch_parameters: *const DispatchParametersARM<'_>,
);
pub const EXT_IMAGE_SLICED_VIEW_OF_3D_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_image_sliced_view_of_3d\0") };
pub const EXT_IMAGE_SLICED_VIEW_OF_3D_SPEC_VERSION: u32 = 1u32;
pub const VALVE_DESCRIPTOR_SET_HOST_MAPPING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_VALVE_descriptor_set_host_mapping\0") };
pub const VALVE_DESCRIPTOR_SET_HOST_MAPPING_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_binding_reference: *const DescriptorSetBindingReferenceVALVE<'_>,
    p_host_mapping: *mut DescriptorSetLayoutHostMappingInfoVALVE<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetDescriptorSetHostMappingVALVE = unsafe extern "system" fn(
    device: crate::vk::Device,
    descriptor_set: DescriptorSet,
    pp_data: *mut *mut c_void,
);
pub const EXT_DEPTH_CLAMP_ZERO_ONE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_depth_clamp_zero_one\0") };
pub const EXT_DEPTH_CLAMP_ZERO_ONE_SPEC_VERSION: u32 = 1u32;
pub const EXT_NON_SEAMLESS_CUBE_MAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_non_seamless_cube_map\0") };
pub const EXT_NON_SEAMLESS_CUBE_MAP_SPEC_VERSION: u32 = 1u32;
pub const ARM_RENDER_PASS_STRIPED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_render_pass_striped\0") };
pub const ARM_RENDER_PASS_STRIPED_SPEC_VERSION: u32 = 1u32;
pub const QCOM_FRAGMENT_DENSITY_MAP_OFFSET_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_fragment_density_map_offset\0") };
pub const QCOM_FRAGMENT_DENSITY_MAP_OFFSET_SPEC_VERSION: u32 = 3u32;
pub const NV_COPY_MEMORY_INDIRECT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_copy_memory_indirect\0") };
pub const NV_COPY_MEMORY_INDIRECT_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMemoryIndirectNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    copy_buffer_address: DeviceAddress,
    copy_count: u32,
    stride: u32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMemoryToImageIndirectNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    copy_buffer_address: DeviceAddress,
    copy_count: u32,
    stride: u32,
    dst_image: Image,
    dst_image_layout: ImageLayout,
    p_image_subresources: *const ImageSubresourceLayers,
);
pub const NV_MEMORY_DECOMPRESSION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_memory_decompression\0") };
pub const NV_MEMORY_DECOMPRESSION_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDecompressMemoryNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    decompress_region_count: u32,
    p_decompress_memory_regions: *const DecompressMemoryRegionNV,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDecompressMemoryIndirectCountNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    indirect_commands_address: DeviceAddress,
    indirect_commands_count_address: DeviceAddress,
    stride: u32,
);
pub const NV_DEVICE_GENERATED_COMMANDS_COMPUTE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_device_generated_commands_compute\0") };
pub const NV_DEVICE_GENERATED_COMMANDS_COMPUTE_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPipelineIndirectMemoryRequirementsNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const ComputePipelineCreateInfo<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdUpdatePipelineIndirectBufferNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    pipeline_bind_point: PipelineBindPoint,
    pipeline: Pipeline,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetPipelineIndirectDeviceAddressNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const PipelineIndirectDeviceAddressInfoNV<'_>,
) -> DeviceAddress;
pub const NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_ray_tracing_linear_swept_spheres\0") };
pub const NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_SPEC_VERSION: u32 = 1u32;
pub const NV_LINEAR_COLOR_ATTACHMENT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_linear_color_attachment\0") };
pub const NV_LINEAR_COLOR_ATTACHMENT_SPEC_VERSION: u32 = 1u32;
pub const GOOGLE_SURFACELESS_QUERY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_surfaceless_query\0") };
pub const GOOGLE_SURFACELESS_QUERY_SPEC_VERSION: u32 = 2u32;
pub const KHR_SHADER_MAXIMAL_RECONVERGENCE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_maximal_reconvergence\0") };
pub const KHR_SHADER_MAXIMAL_RECONVERGENCE_SPEC_VERSION: u32 = 1u32;
pub const EXT_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_image_compression_control_swapchain\0") };
pub const EXT_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_SPEC_VERSION: u32 = 1u32;
pub const QCOM_IMAGE_PROCESSING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_image_processing\0") };
pub const QCOM_IMAGE_PROCESSING_SPEC_VERSION: u32 = 1u32;
pub const EXT_NESTED_COMMAND_BUFFER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_nested_command_buffer\0") };
pub const EXT_NESTED_COMMAND_BUFFER_SPEC_VERSION: u32 = 1u32;
pub const OHOS_EXTERNAL_MEMORY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_OHOS_external_memory\0") };
pub const OHOS_EXTERNAL_MEMORY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetNativeBufferPropertiesOHOS = unsafe extern "system" fn(
    device: crate::vk::Device,
    buffer: *const OH_NativeBuffer,
    p_properties: *mut NativeBufferPropertiesOHOS<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryNativeBufferOHOS = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const MemoryGetNativeBufferInfoOHOS<'_>,
    p_buffer: *mut *mut OH_NativeBuffer,
) -> Result;
pub const EXT_EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_external_memory_acquire_unmodified\0") };
pub const EXT_EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED_SPEC_VERSION: u32 = 1u32;
pub const EXT_EXTENDED_DYNAMIC_STATE3_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_extended_dynamic_state3\0") };
pub const EXT_EXTENDED_DYNAMIC_STATE3_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthClampEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_clamp_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetPolygonModeEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, polygon_mode: PolygonMode);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetRasterizationSamplesEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    rasterization_samples: SampleCountFlags,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetSampleMaskEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    samples: SampleCountFlags,
    p_sample_mask: *const SampleMask,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetAlphaToCoverageEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, alpha_to_coverage_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetAlphaToOneEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, alpha_to_one_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetLogicOpEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, logic_op_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetColorBlendEnableEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_attachment: u32,
    attachment_count: u32,
    p_color_blend_enables: *const Bool32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetColorBlendEquationEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_attachment: u32,
    attachment_count: u32,
    p_color_blend_equations: *const ColorBlendEquationEXT,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetColorWriteMaskEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_attachment: u32,
    attachment_count: u32,
    p_color_write_masks: *const ColorComponentFlags,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetTessellationDomainOriginEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    domain_origin: TessellationDomainOrigin,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetRasterizationStreamEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, rasterization_stream: u32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetConservativeRasterizationModeEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    conservative_rasterization_mode: ConservativeRasterizationModeEXT,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    extra_primitive_overestimation_size: f32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthClipEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, depth_clip_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetSampleLocationsEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, sample_locations_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetColorBlendAdvancedEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_attachment: u32,
    attachment_count: u32,
    p_color_blend_advanced: *const ColorBlendAdvancedEXT,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetProvokingVertexModeEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    provoking_vertex_mode: ProvokingVertexModeEXT,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetLineRasterizationModeEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    line_rasterization_mode: LineRasterizationModeEXT,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetLineStippleEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, stippled_line_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthClipNegativeOneToOneEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, negative_one_to_one: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetViewportWScalingEnableNV =
    unsafe extern "system" fn(command_buffer: CommandBuffer, viewport_w_scaling_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetViewportSwizzleNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    first_viewport: u32,
    viewport_count: u32,
    p_viewport_swizzles: *const ViewportSwizzleNV,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCoverageToColorEnableNV =
    unsafe extern "system" fn(command_buffer: CommandBuffer, coverage_to_color_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCoverageToColorLocationNV =
    unsafe extern "system" fn(command_buffer: CommandBuffer, coverage_to_color_location: u32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCoverageModulationModeNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    coverage_modulation_mode: CoverageModulationModeNV,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCoverageModulationTableEnableNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    coverage_modulation_table_enable: Bool32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCoverageModulationTableNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    coverage_modulation_table_count: u32,
    p_coverage_modulation_table: *const f32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetShadingRateImageEnableNV =
    unsafe extern "system" fn(command_buffer: CommandBuffer, shading_rate_image_enable: Bool32);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetRepresentativeFragmentTestEnableNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    representative_fragment_test_enable: Bool32,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetCoverageReductionModeNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    coverage_reduction_mode: CoverageReductionModeNV,
);
pub const EXT_SUBPASS_MERGE_FEEDBACK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_subpass_merge_feedback\0") };
pub const EXT_SUBPASS_MERGE_FEEDBACK_SPEC_VERSION: u32 = 2u32;
pub const LUNARG_DIRECT_DRIVER_LOADING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LUNARG_direct_driver_loading\0") };
pub const LUNARG_DIRECT_DRIVER_LOADING_SPEC_VERSION: u32 = 1u32;
pub const ARM_TENSORS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_tensors\0") };
pub const ARM_TENSORS_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceExternalTensorPropertiesARM = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_external_tensor_info: *const PhysicalDeviceExternalTensorInfoARM<'_>,
    p_external_tensor_properties: *mut ExternalTensorPropertiesARM<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateTensorARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const TensorCreateInfoARM<'_>,
    p_allocator: *const AllocationCallbacks,
    p_tensor: *mut TensorARM,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyTensorARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    tensor: TensorARM,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateTensorViewARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const TensorViewCreateInfoARM<'_>,
    p_allocator: *const AllocationCallbacks,
    p_view: *mut TensorViewARM,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyTensorViewARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    tensor_view: TensorViewARM,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetTensorMemoryRequirementsARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const TensorMemoryRequirementsInfoARM<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkBindTensorMemoryARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const BindTensorMemoryInfoARM<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceTensorMemoryRequirementsARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const DeviceTensorMemoryRequirementsARM<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyTensorARM = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_tensor_info: *const CopyTensorInfoARM<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetTensorOpaqueCaptureDescriptorDataARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const TensorCaptureDescriptorDataInfoARM<'_>,
    p_data: *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetTensorViewOpaqueCaptureDescriptorDataARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const TensorViewCaptureDescriptorDataInfoARM<'_>,
    p_data: *mut c_void,
) -> Result;
pub const EXT_SHADER_MODULE_IDENTIFIER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_module_identifier\0") };
pub const EXT_SHADER_MODULE_IDENTIFIER_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetShaderModuleIdentifierEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    shader_module: ShaderModule,
    p_identifier: *mut ShaderModuleIdentifierEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetShaderModuleCreateInfoIdentifierEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const ShaderModuleCreateInfo<'_>,
    p_identifier: *mut ShaderModuleIdentifierEXT<'_>,
);
pub const EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_EXT_rasterization_order_attachment_access\0")
};
pub const EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_SPEC_VERSION: u32 = 1u32;
pub const NV_OPTICAL_FLOW_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_optical_flow\0") };
pub const NV_OPTICAL_FLOW_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_optical_flow_image_format_info: *const OpticalFlowImageFormatInfoNV<'_>,
    p_format_count: *mut u32,
    p_image_format_properties: *mut OpticalFlowImageFormatPropertiesNV<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateOpticalFlowSessionNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const OpticalFlowSessionCreateInfoNV<'_>,
    p_allocator: *const AllocationCallbacks,
    p_session: *mut OpticalFlowSessionNV,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyOpticalFlowSessionNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    session: OpticalFlowSessionNV,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkBindOpticalFlowSessionImageNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    session: OpticalFlowSessionNV,
    binding_point: OpticalFlowSessionBindingPointNV,
    view: ImageView,
    layout: ImageLayout,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdOpticalFlowExecuteNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    session: OpticalFlowSessionNV,
    p_execute_info: *const OpticalFlowExecuteInfoNV<'_>,
);
pub const EXT_LEGACY_DITHERING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_legacy_dithering\0") };
pub const EXT_LEGACY_DITHERING_SPEC_VERSION: u32 = 2u32;
pub const EXT_PIPELINE_PROTECTED_ACCESS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_pipeline_protected_access\0") };
pub const EXT_PIPELINE_PROTECTED_ACCESS_SPEC_VERSION: u32 = 1u32;
pub const ANDROID_EXTERNAL_FORMAT_RESOLVE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ANDROID_external_format_resolve\0") };
pub const ANDROID_EXTERNAL_FORMAT_RESOLVE_SPEC_VERSION: u32 = 1u32;
pub const KHR_MAINTENANCE5_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance5\0") };
pub const KHR_MAINTENANCE5_SPEC_VERSION: u32 = 1u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindIndexBuffer2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    size: DeviceSize,
    index_type: IndexType,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetRenderingAreaGranularity = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_rendering_area_info: *const RenderingAreaInfo<'_>,
    p_granularity: *mut Extent2D,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceImageSubresourceLayout = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const DeviceImageSubresourceInfo<'_>,
    p_layout: *mut SubresourceLayout2<'_>,
);
pub const AMD_ANTI_LAG_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMD_anti_lag\0") };
pub const AMD_ANTI_LAG_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkAntiLagUpdateAMD =
    unsafe extern "system" fn(device: crate::vk::Device, p_data: *const AntiLagDataAMD<'_>);
#[cfg(feature = "provisional")]
pub const AMDX_DENSE_GEOMETRY_FORMAT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_AMDX_dense_geometry_format\0") };
#[cfg(feature = "provisional")]
pub const AMDX_DENSE_GEOMETRY_FORMAT_SPEC_VERSION: u32 = 1u32;
pub const KHR_PRESENT_ID2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_present_id2\0") };
pub const KHR_PRESENT_ID2_SPEC_VERSION: u32 = 1u32;
pub const KHR_PRESENT_WAIT2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_present_wait2\0") };
pub const KHR_PRESENT_WAIT2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkWaitForPresent2KHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_present_wait2_info: *const PresentWait2InfoKHR<'_>,
) -> Result;
pub const KHR_RAY_TRACING_POSITION_FETCH_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_ray_tracing_position_fetch\0") };
pub const KHR_RAY_TRACING_POSITION_FETCH_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_OBJECT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_object\0") };
pub const EXT_SHADER_OBJECT_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateShadersEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    create_info_count: u32,
    p_create_infos: *const ShaderCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_shaders: *mut ShaderEXT,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyShaderEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    shader: ShaderEXT,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetShaderBinaryDataEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    shader: ShaderEXT,
    p_data_size: *mut usize,
    p_data: *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindShadersEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    stage_count: u32,
    p_stages: *const ShaderStageFlags,
    p_shaders: *const ShaderEXT,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDepthClampRangeEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    depth_clamp_mode: DepthClampModeEXT,
    p_depth_clamp_range: *const DepthClampRangeEXT,
);
pub const KHR_PIPELINE_BINARY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_pipeline_binary\0") };
pub const KHR_PIPELINE_BINARY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreatePipelineBinariesKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const PipelineBinaryCreateInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
    p_binaries: *mut PipelineBinaryHandlesInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyPipelineBinaryKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    pipeline_binary: PipelineBinaryKHR,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetPipelineKeyKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_pipeline_create_info: *const PipelineCreateInfoKHR<'_>,
    p_pipeline_key: *mut PipelineBinaryKeyKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPipelineBinaryDataKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const PipelineBinaryDataInfoKHR<'_>,
    p_pipeline_binary_key: *mut PipelineBinaryKeyKHR<'_>,
    p_pipeline_binary_data_size: *mut usize,
    p_pipeline_binary_data: *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkReleaseCapturedPipelineDataKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ReleaseCapturedPipelineDataInfoKHR<'_>,
    p_allocator: *const AllocationCallbacks,
) -> Result;
pub const QCOM_TILE_PROPERTIES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_tile_properties\0") };
pub const QCOM_TILE_PROPERTIES_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetFramebufferTilePropertiesQCOM = unsafe extern "system" fn(
    device: crate::vk::Device,
    framebuffer: Framebuffer,
    p_properties_count: *mut u32,
    p_properties: *mut TilePropertiesQCOM<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDynamicRenderingTilePropertiesQCOM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_rendering_info: *const RenderingInfo<'_>,
    p_properties: *mut TilePropertiesQCOM<'_>,
) -> Result;
pub const SEC_AMIGO_PROFILING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_SEC_amigo_profiling\0") };
pub const SEC_AMIGO_PROFILING_SPEC_VERSION: u32 = 1u32;
pub const KHR_SURFACE_MAINTENANCE1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_surface_maintenance1\0") };
pub const KHR_SURFACE_MAINTENANCE1_SPEC_VERSION: u32 = 1u32;
pub const KHR_SWAPCHAIN_MAINTENANCE1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain_maintenance1\0") };
pub const KHR_SWAPCHAIN_MAINTENANCE1_SPEC_VERSION: u32 = 1u32;
pub const QCOM_MULTIVIEW_PER_VIEW_VIEWPORTS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_multiview_per_view_viewports\0") };
pub const QCOM_MULTIVIEW_PER_VIEW_VIEWPORTS_SPEC_VERSION: u32 = 1u32;
pub const NV_RAY_TRACING_INVOCATION_REORDER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_ray_tracing_invocation_reorder\0") };
pub const NV_RAY_TRACING_INVOCATION_REORDER_SPEC_VERSION: u32 = 1u32;
pub const NV_COOPERATIVE_VECTOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_cooperative_vector\0") };
pub const NV_COOPERATIVE_VECTOR_SPEC_VERSION: u32 = 4u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    p_property_count: *mut u32,
    p_properties: *mut CooperativeVectorPropertiesNV<'_>,
)
    -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkConvertCooperativeVectorMatrixNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ConvertCooperativeVectorMatrixInfoNV<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdConvertCooperativeVectorMatrixNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    info_count: u32,
    p_infos: *const ConvertCooperativeVectorMatrixInfoNV<'_>,
);
pub const NV_EXTENDED_SPARSE_ADDRESS_SPACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_extended_sparse_address_space\0") };
pub const NV_EXTENDED_SPARSE_ADDRESS_SPACE_SPEC_VERSION: u32 = 1u32;
pub const EXT_MUTABLE_DESCRIPTOR_TYPE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_mutable_descriptor_type\0") };
pub const EXT_MUTABLE_DESCRIPTOR_TYPE_SPEC_VERSION: u32 = 1u32;
pub const EXT_LEGACY_VERTEX_ATTRIBUTES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_legacy_vertex_attributes\0") };
pub const EXT_LEGACY_VERTEX_ATTRIBUTES_SPEC_VERSION: u32 = 1u32;
pub const EXT_LAYER_SETTINGS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_layer_settings\0") };
pub const EXT_LAYER_SETTINGS_SPEC_VERSION: u32 = 2u32;
pub const ARM_SHADER_CORE_BUILTINS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_shader_core_builtins\0") };
pub const ARM_SHADER_CORE_BUILTINS_SPEC_VERSION: u32 = 2u32;
pub const EXT_PIPELINE_LIBRARY_GROUP_HANDLES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_pipeline_library_group_handles\0") };
pub const EXT_PIPELINE_LIBRARY_GROUP_HANDLES_SPEC_VERSION: u32 = 1u32;
pub const EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_EXT_dynamic_rendering_unused_attachments\0")
};
pub const EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_SPEC_VERSION: u32 = 1u32;
pub const KHR_INTERNALLY_SYNCHRONIZED_QUEUES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_internally_synchronized_queues\0") };
pub const KHR_INTERNALLY_SYNCHRONIZED_QUEUES_SPEC_VERSION: u32 = 1u32;
pub const NV_LOW_LATENCY2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_low_latency2\0") };
pub const NV_LOW_LATENCY2_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkSetLatencySleepModeNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_sleep_mode_info: *const LatencySleepModeInfoNV<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkLatencySleepNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_sleep_info: *const LatencySleepInfoNV<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkSetLatencyMarkerNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_latency_marker_info: *const SetLatencyMarkerInfoNV<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetLatencyTimingsNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    swapchain: SwapchainKHR,
    p_latency_marker_info: *mut GetLatencyMarkerInfoNV<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkQueueNotifyOutOfBandNV =
    unsafe extern "system" fn(queue: Queue, p_queue_type_info: *const OutOfBandQueueTypeInfoNV<'_>);
pub const KHR_COOPERATIVE_MATRIX_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_cooperative_matrix\0") };
pub const KHR_COOPERATIVE_MATRIX_SPEC_VERSION: u32 = 2u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut CooperativeMatrixPropertiesKHR<'_>,
    ) -> Result;
pub const ARM_DATA_GRAPH_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_data_graph\0") };
pub const ARM_DATA_GRAPH_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_queue_family_data_graph_property_count: *mut u32,
        p_queue_family_data_graph_properties: *mut QueueFamilyDataGraphPropertiesARM<'_>,
    ) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM = unsafe extern "system" fn (physical_device : PhysicalDevice , p_queue_family_data_graph_processing_engine_info : * const PhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM < '_ > , p_queue_family_data_graph_processing_engine_properties : * mut QueueFamilyDataGraphProcessingEnginePropertiesARM < '_ > ,) ;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDataGraphPipelinesARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    deferred_operation: DeferredOperationKHR,
    pipeline_cache: PipelineCache,
    create_info_count: u32,
    p_create_infos: *const DataGraphPipelineCreateInfoARM<'_>,
    p_allocator: *const AllocationCallbacks,
    p_pipelines: *mut Pipeline,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateDataGraphPipelineSessionARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const DataGraphPipelineSessionCreateInfoARM<'_>,
    p_allocator: *const AllocationCallbacks,
    p_session: *mut DataGraphPipelineSessionARM,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDataGraphPipelineSessionBindPointRequirementsARM =
    unsafe extern "system" fn(
        device: crate::vk::Device,
        p_info: *const DataGraphPipelineSessionBindPointRequirementsInfoARM<'_>,
        p_bind_point_requirement_count: *mut u32,
        p_bind_point_requirements: *mut DataGraphPipelineSessionBindPointRequirementARM<'_>,
    ) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const DataGraphPipelineSessionMemoryRequirementsInfoARM<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkBindDataGraphPipelineSessionMemoryARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const BindDataGraphPipelineSessionMemoryInfoARM<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyDataGraphPipelineSessionARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    session: DataGraphPipelineSessionARM,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDispatchDataGraphARM = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    session: DataGraphPipelineSessionARM,
    p_info: *const DataGraphPipelineDispatchInfoARM<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkGetDataGraphPipelineAvailablePropertiesARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_pipeline_info: *const DataGraphPipelineInfoARM<'_>,
    p_properties_count: *mut u32,
    p_properties: *mut DataGraphPipelinePropertyARM,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDataGraphPipelinePropertiesARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_pipeline_info: *const DataGraphPipelineInfoARM<'_>,
    properties_count: u32,
    p_properties: *mut DataGraphPipelinePropertyQueryResultARM<'_>,
) -> Result;
pub const ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_data_graph_instruction_set_tosa\0") };
pub const ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_queue_family_data_graph_properties: *const QueueFamilyDataGraphPropertiesARM<'_>,
        p_properties: *mut BaseOutStructure<'_>,
    ) -> Result;
pub const QCOM_MULTIVIEW_PER_VIEW_RENDER_AREAS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_multiview_per_view_render_areas\0") };
pub const QCOM_MULTIVIEW_PER_VIEW_RENDER_AREAS_SPEC_VERSION: u32 = 1u32;
pub const KHR_COMPUTE_SHADER_DERIVATIVES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_compute_shader_derivatives\0") };
pub const KHR_COMPUTE_SHADER_DERIVATIVES_SPEC_VERSION: u32 = 1u32;
pub const KHR_VIDEO_DECODE_AV1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_av1\0") };
pub const KHR_VIDEO_DECODE_AV1_SPEC_VERSION: u32 = 1u32;
pub const KHR_VIDEO_ENCODE_AV1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_av1\0") };
pub const KHR_VIDEO_ENCODE_AV1_SPEC_VERSION: u32 = 1u32;
pub const KHR_VIDEO_DECODE_VP9_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_vp9\0") };
pub const KHR_VIDEO_DECODE_VP9_SPEC_VERSION: u32 = 1u32;
pub const KHR_VIDEO_MAINTENANCE1_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_maintenance1\0") };
pub const KHR_VIDEO_MAINTENANCE1_SPEC_VERSION: u32 = 1u32;
pub const NV_PER_STAGE_DESCRIPTOR_SET_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_per_stage_descriptor_set\0") };
pub const NV_PER_STAGE_DESCRIPTOR_SET_SPEC_VERSION: u32 = 1u32;
pub const QCOM_IMAGE_PROCESSING2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_image_processing2\0") };
pub const QCOM_IMAGE_PROCESSING2_SPEC_VERSION: u32 = 1u32;
pub const QCOM_FILTER_CUBIC_WEIGHTS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_filter_cubic_weights\0") };
pub const QCOM_FILTER_CUBIC_WEIGHTS_SPEC_VERSION: u32 = 1u32;
pub const QCOM_YCBCR_DEGAMMA_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_ycbcr_degamma\0") };
pub const QCOM_YCBCR_DEGAMMA_SPEC_VERSION: u32 = 1u32;
pub const QCOM_FILTER_CUBIC_CLAMP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_filter_cubic_clamp\0") };
pub const QCOM_FILTER_CUBIC_CLAMP_SPEC_VERSION: u32 = 1u32;
pub const EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_EXT_attachment_feedback_loop_dynamic_state\0")
};
pub const EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, aspect_mask: ImageAspectFlags);
pub const KHR_VERTEX_ATTRIBUTE_DIVISOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_vertex_attribute_divisor\0") };
pub const KHR_VERTEX_ATTRIBUTE_DIVISOR_SPEC_VERSION: u32 = 1u32;
pub const KHR_LOAD_STORE_OP_NONE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_load_store_op_none\0") };
pub const KHR_LOAD_STORE_OP_NONE_SPEC_VERSION: u32 = 1u32;
pub const KHR_UNIFIED_IMAGE_LAYOUTS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_unified_image_layouts\0") };
pub const KHR_UNIFIED_IMAGE_LAYOUTS_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_FLOAT_CONTROLS2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_float_controls2\0") };
pub const KHR_SHADER_FLOAT_CONTROLS2_SPEC_VERSION: u32 = 1u32;
pub const QNX_EXTERNAL_MEMORY_SCREEN_BUFFER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QNX_external_memory_screen_buffer\0") };
pub const QNX_EXTERNAL_MEMORY_SCREEN_BUFFER_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetScreenBufferPropertiesQNX = unsafe extern "system" fn(
    device: crate::vk::Device,
    buffer: *const _screen_buffer,
    p_properties: *mut ScreenBufferPropertiesQNX<'_>,
) -> Result;
pub const MSFT_LAYERED_DRIVER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_MSFT_layered_driver\0") };
pub const MSFT_LAYERED_DRIVER_SPEC_VERSION: u32 = 1u32;
pub const KHR_INDEX_TYPE_UINT8_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_index_type_uint8\0") };
pub const KHR_INDEX_TYPE_UINT8_SPEC_VERSION: u32 = 1u32;
pub const KHR_LINE_RASTERIZATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_line_rasterization\0") };
pub const KHR_LINE_RASTERIZATION_SPEC_VERSION: u32 = 1u32;
pub const KHR_CALIBRATED_TIMESTAMPS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_calibrated_timestamps\0") };
pub const KHR_CALIBRATED_TIMESTAMPS_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_EXPECT_ASSUME_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_expect_assume\0") };
pub const KHR_SHADER_EXPECT_ASSUME_SPEC_VERSION: u32 = 1u32;
pub const KHR_MAINTENANCE6_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance6\0") };
pub const KHR_MAINTENANCE6_SPEC_VERSION: u32 = 1u32;
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindDescriptorSets2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_bind_descriptor_sets_info: *const BindDescriptorSetsInfo<'_>,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPushConstants2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_push_constants_info: *const PushConstantsInfo<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPushDescriptorSet2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_push_descriptor_set_info: *const PushDescriptorSetInfo<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPushDescriptorSetWithTemplate2 = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_push_descriptor_set_with_template_info: *const PushDescriptorSetWithTemplateInfo<'_>,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetDescriptorBufferOffsets2EXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_set_descriptor_buffer_offsets_info: *const SetDescriptorBufferOffsetsInfoEXT<'_>,
);
#[doc = "Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>"]
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT = unsafe extern "system" fn (command_buffer : CommandBuffer , p_bind_descriptor_buffer_embedded_samplers_info : * const BindDescriptorBufferEmbeddedSamplersInfoEXT < '_ > ,) ;
pub const NV_DESCRIPTOR_POOL_OVERALLOCATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_descriptor_pool_overallocation\0") };
pub const NV_DESCRIPTOR_POOL_OVERALLOCATION_SPEC_VERSION: u32 = 1u32;
pub const QCOM_TILE_MEMORY_HEAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_tile_memory_heap\0") };
pub const QCOM_TILE_MEMORY_HEAP_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBindTileMemoryQCOM = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_tile_memory_bind_info: *const TileMemoryBindInfoQCOM<'_>,
);
pub const KHR_COPY_MEMORY_INDIRECT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_copy_memory_indirect\0") };
pub const KHR_COPY_MEMORY_INDIRECT_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMemoryIndirectKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_memory_indirect_info: *const CopyMemoryIndirectInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdCopyMemoryToImageIndirectKHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_copy_memory_to_image_indirect_info: *const CopyMemoryToImageIndirectInfoKHR<'_>,
);
pub const EXT_MEMORY_DECOMPRESSION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_memory_decompression\0") };
pub const EXT_MEMORY_DECOMPRESSION_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDecompressMemoryEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_decompress_memory_info_ext: *const DecompressMemoryInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdDecompressMemoryIndirectCountEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    decompression_method: MemoryDecompressionMethodFlagsEXT,
    indirect_commands_address: DeviceAddress,
    indirect_commands_count_address: DeviceAddress,
    max_decompression_count: u32,
    stride: u32,
);
pub const NV_DISPLAY_STEREO_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_display_stereo\0") };
pub const NV_DISPLAY_STEREO_SPEC_VERSION: u32 = 1u32;
pub const KHR_VIDEO_ENCODE_INTRA_REFRESH_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_intra_refresh\0") };
pub const KHR_VIDEO_ENCODE_INTRA_REFRESH_SPEC_VERSION: u32 = 1u32;
pub const KHR_VIDEO_ENCODE_QUANTIZATION_MAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_quantization_map\0") };
pub const KHR_VIDEO_ENCODE_QUANTIZATION_MAP_SPEC_VERSION: u32 = 2u32;
pub const NV_RAW_ACCESS_CHAINS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_raw_access_chains\0") };
pub const NV_RAW_ACCESS_CHAINS_SPEC_VERSION: u32 = 1u32;
pub const NV_EXTERNAL_COMPUTE_QUEUE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_external_compute_queue\0") };
pub const NV_EXTERNAL_COMPUTE_QUEUE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetExternalComputeQueueDataNV = unsafe extern "system" fn(
    external_queue: ExternalComputeQueueNV,
    params: *mut ExternalComputeQueueDataParamsNV<'_>,
    p_data: *mut c_void,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateExternalComputeQueueNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const ExternalComputeQueueCreateInfoNV<'_>,
    p_allocator: *const AllocationCallbacks,
    p_external_queue: *mut ExternalComputeQueueNV,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyExternalComputeQueueNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    external_queue: ExternalComputeQueueNV,
    p_allocator: *const AllocationCallbacks,
);
pub const KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_relaxed_extended_instruction\0") };
pub const KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_SPEC_VERSION: u32 = 1u32;
pub const NV_COMMAND_BUFFER_INHERITANCE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_command_buffer_inheritance\0") };
pub const NV_COMMAND_BUFFER_INHERITANCE_SPEC_VERSION: u32 = 1u32;
pub const KHR_MAINTENANCE7_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance7\0") };
pub const KHR_MAINTENANCE7_SPEC_VERSION: u32 = 1u32;
pub const NV_SHADER_ATOMIC_FLOAT16_VECTOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_shader_atomic_float16_vector\0") };
pub const NV_SHADER_ATOMIC_FLOAT16_VECTOR_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_REPLICATED_COMPOSITES_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_replicated_composites\0") };
pub const EXT_SHADER_REPLICATED_COMPOSITES_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_FLOAT8_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_float8\0") };
pub const EXT_SHADER_FLOAT8_SPEC_VERSION: u32 = 1u32;
pub const NV_RAY_TRACING_VALIDATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_ray_tracing_validation\0") };
pub const NV_RAY_TRACING_VALIDATION_SPEC_VERSION: u32 = 1u32;
pub const NV_CLUSTER_ACCELERATION_STRUCTURE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_cluster_acceleration_structure\0") };
pub const NV_CLUSTER_ACCELERATION_STRUCTURE_SPEC_VERSION: u32 = 4u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetClusterAccelerationStructureBuildSizesNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const ClusterAccelerationStructureInputInfoNV<'_>,
    p_size_info: *mut AccelerationStructureBuildSizesInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBuildClusterAccelerationStructureIndirectNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_command_infos: *const ClusterAccelerationStructureCommandsInfoNV<'_>,
);
pub const NV_PARTITIONED_ACCELERATION_STRUCTURE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_partitioned_acceleration_structure\0") };
pub const NV_PARTITIONED_ACCELERATION_STRUCTURE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPartitionedAccelerationStructuresBuildSizesNV = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const PartitionedAccelerationStructureInstancesInputNV<'_>,
    p_size_info: *mut AccelerationStructureBuildSizesInfoKHR<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBuildPartitionedAccelerationStructuresNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_build_info: *const BuildPartitionedAccelerationStructureInfoNV<'_>,
);
pub const EXT_DEVICE_GENERATED_COMMANDS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_device_generated_commands\0") };
pub const EXT_DEVICE_GENERATED_COMMANDS_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetGeneratedCommandsMemoryRequirementsEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_info: *const GeneratedCommandsMemoryRequirementsInfoEXT<'_>,
    p_memory_requirements: *mut MemoryRequirements2<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdPreprocessGeneratedCommandsEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_generated_commands_info: *const GeneratedCommandsInfoEXT<'_>,
    state_command_buffer: CommandBuffer,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdExecuteGeneratedCommandsEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    is_preprocessed: Bool32,
    p_generated_commands_info: *const GeneratedCommandsInfoEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateIndirectCommandsLayoutEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const IndirectCommandsLayoutCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_indirect_commands_layout: *mut IndirectCommandsLayoutEXT,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyIndirectCommandsLayoutEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    indirect_commands_layout: IndirectCommandsLayoutEXT,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCreateIndirectExecutionSetEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const IndirectExecutionSetCreateInfoEXT<'_>,
    p_allocator: *const AllocationCallbacks,
    p_indirect_execution_set: *mut IndirectExecutionSetEXT,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyIndirectExecutionSetEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    indirect_execution_set: IndirectExecutionSetEXT,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkUpdateIndirectExecutionSetPipelineEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    indirect_execution_set: IndirectExecutionSetEXT,
    execution_set_write_count: u32,
    p_execution_set_writes: *const WriteIndirectExecutionSetPipelineEXT<'_>,
);
#[allow(non_camel_case_types)]
pub type PFN_vkUpdateIndirectExecutionSetShaderEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    indirect_execution_set: IndirectExecutionSetEXT,
    execution_set_write_count: u32,
    p_execution_set_writes: *const WriteIndirectExecutionSetShaderEXT<'_>,
);
pub const KHR_DEVICE_FAULT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_device_fault\0") };
pub const KHR_DEVICE_FAULT_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceFaultReportsKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    timeout: u64,
    p_fault_counts: *mut u32,
    p_fault_info: *mut DeviceFaultInfoKHR<'_>,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetDeviceFaultDebugInfoKHR = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_debug_info: *mut DeviceFaultDebugInfoKHR<'_>,
) -> Result;
pub const KHR_MAINTENANCE8_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance8\0") };
pub const KHR_MAINTENANCE8_SPEC_VERSION: u32 = 1u32;
pub const MESA_IMAGE_ALIGNMENT_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_MESA_image_alignment_control\0") };
pub const MESA_IMAGE_ALIGNMENT_CONTROL_SPEC_VERSION: u32 = 1u32;
pub const KHR_SHADER_FMA_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_fma\0") };
pub const KHR_SHADER_FMA_SPEC_VERSION: u32 = 1u32;
pub const NV_PUSH_CONSTANT_BANK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_push_constant_bank\0") };
pub const NV_PUSH_CONSTANT_BANK_SPEC_VERSION: u32 = 1u32;
pub const EXT_RAY_TRACING_INVOCATION_REORDER_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_ray_tracing_invocation_reorder\0") };
pub const EXT_RAY_TRACING_INVOCATION_REORDER_SPEC_VERSION: u32 = 2u32;
pub const EXT_DEPTH_CLAMP_CONTROL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_depth_clamp_control\0") };
pub const EXT_DEPTH_CLAMP_CONTROL_SPEC_VERSION: u32 = 1u32;
pub const KHR_MAINTENANCE9_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance9\0") };
pub const KHR_MAINTENANCE9_SPEC_VERSION: u32 = 1u32;
pub const KHR_VIDEO_MAINTENANCE2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_maintenance2\0") };
pub const KHR_VIDEO_MAINTENANCE2_SPEC_VERSION: u32 = 1u32;
pub const OHOS_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_OHOS_surface\0") };
pub const OHOS_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateSurfaceOHOS = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const SurfaceCreateInfoOHOS<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
pub const HUAWEI_HDR_VIVID_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_HUAWEI_hdr_vivid\0") };
pub const HUAWEI_HDR_VIVID_SPEC_VERSION: u32 = 1u32;
pub const NV_COOPERATIVE_MATRIX2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_cooperative_matrix2\0") };
pub const NV_COOPERATIVE_MATRIX2_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_property_count: *mut u32,
        p_properties: *mut CooperativeMatrixFlexibleDimensionsPropertiesNV<'_>,
    ) -> Result;
pub const ARM_PIPELINE_OPACITY_MICROMAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_pipeline_opacity_micromap\0") };
pub const ARM_PIPELINE_OPACITY_MICROMAP_SPEC_VERSION: u32 = 1u32;
pub const EXT_EXTERNAL_MEMORY_METAL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_external_memory_metal\0") };
pub const EXT_EXTERNAL_MEMORY_METAL_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryMetalHandleEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_get_metal_handle_info: *const MemoryGetMetalHandleInfoEXT<'_>,
    p_handle: *mut *mut c_void,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetMemoryMetalHandlePropertiesEXT = unsafe extern "system" fn(
    device: crate::vk::Device,
    handle_type: ExternalMemoryHandleTypeFlags,
    p_handle: *const c_void,
    p_memory_metal_handle_properties: *mut MemoryMetalHandlePropertiesEXT<'_>,
) -> Result;
pub const KHR_DEPTH_CLAMP_ZERO_ONE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_depth_clamp_zero_one\0") };
pub const KHR_DEPTH_CLAMP_ZERO_ONE_SPEC_VERSION: u32 = 1u32;
pub const ARM_PERFORMANCE_COUNTERS_BY_REGION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_performance_counters_by_region\0") };
pub const ARM_PERFORMANCE_COUNTERS_BY_REGION_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceCountersByRegionARM =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_counter_count: *mut u32,
        p_counters: *mut PerformanceCounterARM<'_>,
        p_counter_descriptions: *mut PerformanceCounterDescriptionARM<'_>,
    ) -> Result;
pub const ARM_SHADER_INSTRUMENTATION_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_shader_instrumentation\0") };
pub const ARM_SHADER_INSTRUMENTATION_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkEnumeratePhysicalDeviceShaderInstrumentationMetricsARM =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        p_description_count: *mut u32,
        p_descriptions: *mut ShaderInstrumentationMetricDescriptionARM<'_>,
    ) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateShaderInstrumentationARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    p_create_info: *const ShaderInstrumentationCreateInfoARM<'_>,
    p_allocator: *const AllocationCallbacks,
    p_instrumentation: *mut ShaderInstrumentationARM,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkDestroyShaderInstrumentationARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    instrumentation: ShaderInstrumentationARM,
    p_allocator: *const AllocationCallbacks,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginShaderInstrumentationARM = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    instrumentation: ShaderInstrumentationARM,
);
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndShaderInstrumentationARM =
    unsafe extern "system" fn(command_buffer: CommandBuffer);
#[allow(non_camel_case_types)]
pub type PFN_vkGetShaderInstrumentationValuesARM = unsafe extern "system" fn(
    device: crate::vk::Device,
    instrumentation: ShaderInstrumentationARM,
    p_metric_block_count: *mut u32,
    p_metric_values: *mut c_void,
    flags: ShaderInstrumentationValuesFlagsARM,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkClearShaderInstrumentationMetricsARM =
    unsafe extern "system" fn(device: crate::vk::Device, instrumentation: ShaderInstrumentationARM);
pub const EXT_VERTEX_ATTRIBUTE_ROBUSTNESS_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_vertex_attribute_robustness\0") };
pub const EXT_VERTEX_ATTRIBUTE_ROBUSTNESS_SPEC_VERSION: u32 = 1u32;
pub const ARM_FORMAT_PACK_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_format_pack\0") };
pub const ARM_FORMAT_PACK_SPEC_VERSION: u32 = 1u32;
pub const VALVE_FRAGMENT_DENSITY_MAP_LAYERED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_VALVE_fragment_density_map_layered\0") };
pub const VALVE_FRAGMENT_DENSITY_MAP_LAYERED_SPEC_VERSION: u32 = 1u32;
pub const KHR_ROBUSTNESS2_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_robustness2\0") };
pub const KHR_ROBUSTNESS2_SPEC_VERSION: u32 = 1u32;
pub const NV_PRESENT_METERING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_present_metering\0") };
pub const NV_PRESENT_METERING_SPEC_VERSION: u32 = 1u32;
pub const EXT_FRAGMENT_DENSITY_MAP_OFFSET_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_fragment_density_map_offset\0") };
pub const EXT_FRAGMENT_DENSITY_MAP_OFFSET_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdEndRendering2KHR = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_rendering_end_info: *const RenderingEndInfoKHR<'_>,
);
pub const EXT_ZERO_INITIALIZE_DEVICE_MEMORY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_zero_initialize_device_memory\0") };
pub const EXT_ZERO_INITIALIZE_DEVICE_MEMORY_SPEC_VERSION: u32 = 1u32;
pub const KHR_PRESENT_MODE_FIFO_LATEST_READY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_present_mode_fifo_latest_ready\0") };
pub const KHR_PRESENT_MODE_FIFO_LATEST_READY_SPEC_VERSION: u32 = 1u32;
pub const KHR_OPACITY_MICROMAP_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_opacity_micromap\0") };
pub const KHR_OPACITY_MICROMAP_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_64BIT_INDEXING_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_64bit_indexing\0") };
pub const EXT_SHADER_64BIT_INDEXING_SPEC_VERSION: u32 = 1u32;
pub const EXT_CUSTOM_RESOLVE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_custom_resolve\0") };
pub const EXT_CUSTOM_RESOLVE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdBeginCustomResolveEXT = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_begin_custom_resolve_info: *const BeginCustomResolveInfoEXT<'_>,
);
pub const QCOM_DATA_GRAPH_MODEL_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_QCOM_data_graph_model\0") };
pub const QCOM_DATA_GRAPH_MODEL_SPEC_VERSION: u32 = 1u32;
pub const KHR_MAINTENANCE10_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance10\0") };
pub const KHR_MAINTENANCE10_SPEC_VERSION: u32 = 1u32;
pub const ARM_DATA_GRAPH_OPTICAL_FLOW_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_ARM_data_graph_optical_flow\0") };
pub const ARM_DATA_GRAPH_OPTICAL_FLOW_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceQueueFamilyDataGraphOpticalFlowImageFormatsARM =
    unsafe extern "system" fn(
        physical_device: PhysicalDevice,
        queue_family_index: u32,
        p_queue_family_data_graph_properties: *const QueueFamilyDataGraphPropertiesARM<'_>,
        p_optical_flow_image_format_info: *const DataGraphOpticalFlowImageFormatInfoARM<'_>,
        p_format_count: *mut u32,
        p_image_format_properties: *mut DataGraphOpticalFlowImageFormatPropertiesARM<'_>,
    ) -> Result;
pub const EXT_SHADER_LONG_VECTOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_long_vector\0") };
pub const EXT_SHADER_LONG_VECTOR_SPEC_VERSION: u32 = 1u32;
pub const SEC_PIPELINE_CACHE_INCREMENTAL_MODE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_SEC_pipeline_cache_incremental_mode\0") };
pub const SEC_PIPELINE_CACHE_INCREMENTAL_MODE_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_uniform_buffer_unsized_array\0") };
pub const EXT_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_SPEC_VERSION: u32 = 1u32;
pub const NV_COMPUTE_OCCUPANCY_PRIORITY_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_compute_occupancy_priority\0") };
pub const NV_COMPUTE_OCCUPANCY_PRIORITY_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetComputeOccupancyPriorityNV = unsafe extern "system" fn(
    command_buffer: CommandBuffer,
    p_parameters: *const ComputeOccupancyPriorityParametersNV<'_>,
);
pub const KHR_MAINTENANCE11_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_maintenance11\0") };
pub const KHR_MAINTENANCE11_SPEC_VERSION: u32 = 1u32;
pub const EXT_SHADER_SUBGROUP_PARTITIONED_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_shader_subgroup_partitioned\0") };
pub const EXT_SHADER_SUBGROUP_PARTITIONED_SPEC_VERSION: u32 = 1u32;
pub const SEC_UBM_SURFACE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_SEC_ubm_surface\0") };
pub const SEC_UBM_SURFACE_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCreateUbmSurfaceSEC = unsafe extern "system" fn(
    instance: crate::vk::Instance,
    p_create_info: *const UbmSurfaceCreateInfoSEC<'_>,
    p_allocator: *const AllocationCallbacks,
    p_surface: *mut SurfaceKHR,
) -> Result;
#[allow(non_camel_case_types)]
pub type PFN_vkGetPhysicalDeviceUbmPresentationSupportSEC = unsafe extern "system" fn(
    physical_device: PhysicalDevice,
    queue_family_index: u32,
    device: *mut ubm_device,
) -> Bool32;
pub const VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_VALVE_shader_mixed_float_dot_product\0") };
pub const VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_SPEC_VERSION: u32 = 1u32;
pub const SEC_THROTTLE_HINT_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_SEC_throttle_hint\0") };
pub const SEC_THROTTLE_HINT_SPEC_VERSION: u32 = 1u32;
pub const ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_NAME: &CStr = unsafe {
    CStr::from_bytes_with_nul_unchecked(b"VK_ARM_data_graph_neural_accelerator_statistics\0")
};
pub const ARM_DATA_GRAPH_NEURAL_ACCELERATOR_STATISTICS_SPEC_VERSION: u32 = 1u32;
pub const EXT_PRIMITIVE_RESTART_INDEX_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_EXT_primitive_restart_index\0") };
pub const EXT_PRIMITIVE_RESTART_INDEX_SPEC_VERSION: u32 = 1u32;
#[allow(non_camel_case_types)]
pub type PFN_vkCmdSetPrimitiveRestartIndexEXT =
    unsafe extern "system" fn(command_buffer: CommandBuffer, primitive_restart_index: u32);
pub const NV_COOPERATIVE_MATRIX_DECODE_VECTOR_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_NV_cooperative_matrix_decode_vector\0") };
pub const NV_COOPERATIVE_MATRIX_DECODE_VECTOR_SPEC_VERSION: u32 = 1u32;
