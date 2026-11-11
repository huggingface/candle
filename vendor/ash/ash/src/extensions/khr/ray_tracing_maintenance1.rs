//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_ray_tracing_maintenance1.html>

use crate::vk;

impl crate::khr::ray_tracing_maintenance1::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdTraceRaysIndirect2KHR.html>
    ///
    /// `indirect_device_address` is a buffer device address which is a pointer to a [`vk::TraceRaysIndirectCommand2KHR`] structure containing the trace ray parameters.
    #[inline]
    pub unsafe fn cmd_trace_rays_indirect2(
        &self,
        command_buffer: vk::CommandBuffer,
        indirect_device_address: vk::DeviceAddress,
    ) {
        (self.fp.cmd_trace_rays_indirect2_khr)(command_buffer, indirect_device_address)
    }
}
