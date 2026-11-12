use super::physical::{
    ConformanceVersion, DriverId, MemoryDecompressionMethods, OpticalFlowGridSizes,
    PhysicalDeviceType, PipelineRobustnessBufferBehavior, PipelineRobustnessImageBehavior,
    PointClippingBehavior, RayTracingInvocationReorderMode, ShaderCoreProperties,
    ShaderFloatControlsIndependence, SubgroupFeatures,
};
use crate::{
    buffer::BufferUsage,
    device::{
        physical::{LayeredDriverUnderlyingApi, PhysicalDeviceSchedulingControlsFlags},
        DeviceExtensions, QueueFlags,
    },
    image::{sampler::ycbcr::ChromaLocation, ImageLayout, ImageUsage, SampleCount, SampleCounts},
    instance::InstanceExtensions,
    memory::DeviceAlignment,
    render_pass::ResolveModes,
    shader::ShaderStages,
    DeviceSize, Version,
};
use ash::vk;
use ash::vk::TaggedStructure;
use std::ffi::c_char;

include!(crate::autogen_output!("properties.rs"));

// A bit of a hack...
// TODO: integrate into autogen?
pub(crate) trait FromVulkan<F>
where
    Self: Sized,
{
    fn from_vulkan(val: F) -> Option<Self>;
}

impl FromVulkan<u8> for u8 {
    #[inline]
    fn from_vulkan(val: u8) -> Option<Self> {
        Some(val)
    }
}

impl<const N: usize> FromVulkan<[u8; N]> for [u8; N] {
    #[inline]
    fn from_vulkan(val: [u8; N]) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<u32> for u32 {
    #[inline]
    fn from_vulkan(val: u32) -> Option<Self> {
        Some(val)
    }
}

impl<const N: usize> FromVulkan<[u32; N]> for [u32; N] {
    #[inline]
    fn from_vulkan(val: [u32; N]) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<u64> for u64 {
    #[inline]
    fn from_vulkan(val: u64) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<u64> for DeviceAlignment {
    #[inline]
    fn from_vulkan(val: u64) -> Option<Self> {
        DeviceAlignment::new(val)
    }
}

impl FromVulkan<usize> for usize {
    #[inline]
    fn from_vulkan(val: usize) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<u64> for usize {
    #[inline]
    fn from_vulkan(val: u64) -> Option<Self> {
        Some(val as usize)
    }
}

impl FromVulkan<i32> for i32 {
    #[inline]
    fn from_vulkan(val: i32) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<i64> for i64 {
    #[inline]
    fn from_vulkan(val: i64) -> Option<Self> {
        Some(val)
    }
}

impl FromVulkan<f32> for f32 {
    #[inline]
    fn from_vulkan(val: f32) -> Option<Self> {
        Some(val)
    }
}

impl<const N: usize> FromVulkan<[f32; N]> for [f32; N] {
    #[inline]
    fn from_vulkan(val: [f32; N]) -> Option<Self> {
        Some(val)
    }
}

impl<const N: usize> FromVulkan<[c_char; N]> for String {
    /// <https://github.com/ash-rs/ash/blob/b724b78dac8d83879ed7a1aad2b91bb9f2beb5cf/ash/src/vk/prelude.rs#L66-L73>
    #[inline]
    fn from_vulkan(val: [c_char; N]) -> Option<Self> {
        // SAFETY: The cast from c_char to u8 is ok because a c_char is always one byte.
        let bytes = unsafe { core::slice::from_raw_parts(val.as_ptr().cast(), val.len()) };
        Some(
            core::ffi::CStr::from_bytes_until_nul(bytes)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
        )
    }
}

impl FromVulkan<u32> for Version {
    #[inline]
    fn from_vulkan(val: u32) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::Bool32> for bool {
    #[inline]
    fn from_vulkan(val: vk::Bool32) -> Option<Self> {
        Some(val != 0)
    }
}

impl FromVulkan<vk::ConformanceVersion> for ConformanceVersion {
    #[inline]
    fn from_vulkan(val: vk::ConformanceVersion) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::DriverId> for DriverId {
    #[inline]
    fn from_vulkan(val: vk::DriverId) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::Extent2D> for [u32; 2] {
    #[inline]
    fn from_vulkan(val: vk::Extent2D) -> Option<Self> {
        Some([val.width, val.height])
    }
}

impl FromVulkan<vk::MemoryDecompressionMethodFlagsNV> for MemoryDecompressionMethods {
    #[inline]
    fn from_vulkan(val: vk::MemoryDecompressionMethodFlagsNV) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::OpticalFlowGridSizeFlagsNV> for OpticalFlowGridSizes {
    #[inline]
    fn from_vulkan(val: vk::OpticalFlowGridSizeFlagsNV) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::PhysicalDeviceType> for PhysicalDeviceType {
    #[inline]
    fn from_vulkan(val: vk::PhysicalDeviceType) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::PipelineRobustnessBufferBehaviorEXT> for PipelineRobustnessBufferBehavior {
    #[inline]
    fn from_vulkan(val: vk::PipelineRobustnessBufferBehaviorEXT) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::PipelineRobustnessImageBehaviorEXT> for PipelineRobustnessImageBehavior {
    #[inline]
    fn from_vulkan(val: vk::PipelineRobustnessImageBehaviorEXT) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::PointClippingBehavior> for PointClippingBehavior {
    #[inline]
    fn from_vulkan(val: vk::PointClippingBehavior) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::QueueFlags> for QueueFlags {
    #[inline]
    fn from_vulkan(val: vk::QueueFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::RayTracingInvocationReorderModeNV> for RayTracingInvocationReorderMode {
    #[inline]
    fn from_vulkan(val: vk::RayTracingInvocationReorderModeNV) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::ResolveModeFlags> for ResolveModes {
    #[inline]
    fn from_vulkan(val: vk::ResolveModeFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::SampleCountFlags> for SampleCounts {
    #[inline]
    fn from_vulkan(val: vk::SampleCountFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::SampleCountFlags> for SampleCount {
    #[inline]
    fn from_vulkan(val: vk::SampleCountFlags) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::ShaderCorePropertiesFlagsAMD> for ShaderCoreProperties {
    #[inline]
    fn from_vulkan(val: vk::ShaderCorePropertiesFlagsAMD) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::ShaderFloatControlsIndependence> for ShaderFloatControlsIndependence {
    #[inline]
    fn from_vulkan(val: vk::ShaderFloatControlsIndependence) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::ShaderStageFlags> for ShaderStages {
    #[inline]
    fn from_vulkan(val: vk::ShaderStageFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::SubgroupFeatureFlags> for SubgroupFeatures {
    #[inline]
    fn from_vulkan(val: vk::SubgroupFeatureFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::BufferUsageFlags> for BufferUsage {
    #[inline]
    fn from_vulkan(val: vk::BufferUsageFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::ImageUsageFlags> for ImageUsage {
    #[inline]
    fn from_vulkan(val: vk::ImageUsageFlags) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::ChromaLocation> for ChromaLocation {
    #[inline]
    fn from_vulkan(val: vk::ChromaLocation) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::PhysicalDeviceSchedulingControlsFlagsARM>
    for PhysicalDeviceSchedulingControlsFlags
{
    #[inline]
    fn from_vulkan(val: vk::PhysicalDeviceSchedulingControlsFlagsARM) -> Option<Self> {
        Some(val.into())
    }
}

impl FromVulkan<vk::LayeredDriverUnderlyingApiMSFT> for LayeredDriverUnderlyingApi {
    #[inline]
    fn from_vulkan(val: vk::LayeredDriverUnderlyingApiMSFT) -> Option<Self> {
        val.try_into().ok()
    }
}

impl FromVulkan<vk::ImageLayout> for ImageLayout {
    #[inline]
    fn from_vulkan(val: vk::ImageLayout) -> Option<Self> {
        val.try_into().ok()
    }
}

impl<U: for<'a> FromVulkan<&'a T>, T> FromVulkan<&[T]> for Vec<U> {
    #[inline]
    fn from_vulkan(val: &[T]) -> Option<Vec<U>> {
        val.iter()
            .map(|it| U::from_vulkan(it))
            .collect::<Option<Vec<_>>>()
    }
}

impl<U: FromVulkan<T>, T: Copy> FromVulkan<&T> for U {
    #[inline]
    fn from_vulkan(val: &T) -> Option<Self> {
        U::from_vulkan(*val)
    }
}
