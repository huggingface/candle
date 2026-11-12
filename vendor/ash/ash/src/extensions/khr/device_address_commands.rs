//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_device_address_commands.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::khr::device_address_commands::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindIndexBuffer3KHR.html>
    #[inline]
    #[doc(alias = "vkCmdBindIndexBuffer3KHR")]
    pub unsafe fn cmd_bind_index_buffer3(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::BindIndexBuffer3InfoKHR<'_>,
    ) {
        (self.fp.cmd_bind_index_buffer3_khr)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindVertexBuffers3KHR.html>
    #[inline]
    #[doc(alias = "vkCmdBindVertexBuffers3KHR")]
    pub unsafe fn cmd_bind_vertex_buffers3(
        &self,
        command_buffer: vk::CommandBuffer,
        first_binding: u32,
        binding_infos: &[vk::BindVertexBuffer3InfoKHR<'_>],
    ) {
        (self.fp.cmd_bind_vertex_buffers3_khr)(
            command_buffer,
            first_binding,
            binding_infos.len() as u32,
            binding_infos.as_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndirect2KHR.html>
    #[inline]
    #[doc(alias = "vkCmdDrawIndirect2KHR")]
    pub unsafe fn cmd_draw_indirect2(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::DrawIndirect2InfoKHR<'_>,
    ) {
        (self.fp.cmd_draw_indirect2_khr)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndexedIndirect2KHR.html>
    #[inline]
    #[doc(alias = "vkCmdDrawIndexedIndirect2KHR")]
    pub unsafe fn cmd_draw_indexed_indirect2(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::DrawIndirect2InfoKHR<'_>,
    ) {
        (self.fp.cmd_draw_indexed_indirect2_khr)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDispatchIndirect2KHR.html>
    #[inline]
    #[doc(alias = "vkCmdDispatchIndirect2KHR")]
    pub unsafe fn cmd_dispatch_indirect2(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::DispatchIndirect2InfoKHR<'_>,
    ) {
        (self.fp.cmd_dispatch_indirect2_khr)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyMemoryKHR.html>
    #[inline]
    #[doc(alias = "vkCmdCopyMemoryKHR")]
    pub unsafe fn cmd_copy_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        copy_memory_info: &vk::CopyDeviceMemoryInfoKHR<'_>,
    ) {
        (self.fp.cmd_copy_memory_khr)(command_buffer, copy_memory_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyMemoryToImageKHR.html>
    #[inline]
    #[doc(alias = "vkCmdCopyMemoryToImageKHR")]
    pub unsafe fn cmd_copy_memory_to_image(
        &self,
        command_buffer: vk::CommandBuffer,
        copy_memory_info: &vk::CopyDeviceMemoryImageInfoKHR<'_>,
    ) {
        (self.fp.cmd_copy_memory_to_image_khr)(command_buffer, copy_memory_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImageToMemoryKHR.html>
    #[inline]
    #[doc(alias = "vkCmdCopyImageToMemoryKHR")]
    pub unsafe fn cmd_copy_image_to_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        copy_memory_info: &vk::CopyDeviceMemoryImageInfoKHR<'_>,
    ) {
        (self.fp.cmd_copy_image_to_memory_khr)(command_buffer, copy_memory_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdUpdateMemoryKHR.html>
    #[inline]
    #[doc(alias = "vkCmdUpdateMemoryKHR")]
    pub unsafe fn cmd_update_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        dst_range: &vk::DeviceAddressRangeKHR,
        dst_flags: vk::AddressCommandFlagsKHR,
        data: &[u8],
    ) {
        (self.fp.cmd_update_memory_khr)(
            command_buffer,
            dst_range,
            dst_flags,
            data.len() as vk::DeviceSize,
            data.as_ptr().cast(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdFillMemoryKHR.html>
    #[inline]
    #[doc(alias = "vkCmdFillMemoryKHR")]
    pub unsafe fn cmd_fill_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        dst_range: &vk::DeviceAddressRangeKHR,
        dst_flags: vk::AddressCommandFlagsKHR,
        data: u32,
    ) {
        (self.fp.cmd_fill_memory_khr)(command_buffer, dst_range, dst_flags, data)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyQueryPoolResultsToMemoryKHR.html>
    #[inline]
    #[doc(alias = "vkCmdCopyQueryPoolResultsToMemoryKHR")]
    pub unsafe fn cmd_copy_query_pool_results_to_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        query_pool: vk::QueryPool,
        first_query: u32,
        query_count: u32,
        dst_range: &vk::StridedDeviceAddressRangeKHR,
        dst_flags: vk::AddressCommandFlagsKHR,
        flags: vk::QueryResultFlags,
    ) {
        (self.fp.cmd_copy_query_pool_results_to_memory_khr)(
            command_buffer,
            query_pool,
            first_query,
            query_count,
            dst_range,
            dst_flags,
            flags,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndirectCount2KHR.html>
    #[inline]
    #[doc(alias = "vkCmdDrawIndirectCount2KHR")]
    pub unsafe fn cmd_draw_indirect_count2(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::DrawIndirectCount2InfoKHR<'_>,
    ) {
        (self.fp.cmd_draw_indirect_count2_khr)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndexedIndirectCount2KHR.html>
    #[inline]
    #[doc(alias = "vkCmdDrawIndexedIndirectCount2KHR")]
    pub unsafe fn cmd_draw_indexed_indirect_count2(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::DrawIndirectCount2InfoKHR<'_>,
    ) {
        (self.fp.cmd_draw_indexed_indirect_count2_khr)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBeginConditionalRendering2EXT.html>
    #[inline]
    #[doc(alias = "vkCmdBeginConditionalRendering2EXT")]
    pub unsafe fn cmd_begin_conditional_rendering2(
        &self,
        command_buffer: vk::CommandBuffer,
        conditional_rendering_begin: &vk::ConditionalRenderingBeginInfo2EXT<'_>,
    ) {
        (self.fp.cmd_begin_conditional_rendering2_ext)(command_buffer, conditional_rendering_begin)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindTransformFeedbackBuffers2EXT.html>
    #[inline]
    #[doc(alias = "vkCmdBindTransformFeedbackBuffers2EXT")]
    pub unsafe fn cmd_bind_transform_feedback_buffers2(
        &self,
        command_buffer: vk::CommandBuffer,
        first_binding: u32,
        binding_infos: &[vk::BindTransformFeedbackBuffer2InfoEXT<'_>],
    ) {
        (self.fp.cmd_bind_transform_feedback_buffers2_ext)(
            command_buffer,
            first_binding,
            binding_infos.len() as u32,
            binding_infos.as_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBeginTransformFeedback2EXT.html>
    #[inline]
    #[doc(alias = "vkCmdBeginTransformFeedback2EXT")]
    pub unsafe fn cmd_begin_transform_feedback2(
        &self,
        command_buffer: vk::CommandBuffer,
        first_counter_range: u32,
        counter_infos: &[vk::BindTransformFeedbackBuffer2InfoEXT<'_>],
    ) {
        (self.fp.cmd_begin_transform_feedback2_ext)(
            command_buffer,
            first_counter_range,
            counter_infos.len() as u32,
            counter_infos.as_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdEndTransformFeedback2EXT.html>
    #[inline]
    #[doc(alias = "vkCmdEndTransformFeedback2EXT")]
    pub unsafe fn cmd_end_transform_feedback2(
        &self,
        command_buffer: vk::CommandBuffer,
        first_counter_range: u32,
        counter_infos: &[vk::BindTransformFeedbackBuffer2InfoEXT<'_>],
    ) {
        (self.fp.cmd_end_transform_feedback2_ext)(
            command_buffer,
            first_counter_range,
            counter_infos.len() as u32,
            counter_infos.as_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawIndirectByteCount2EXT.html>
    #[inline]
    #[doc(alias = "vkCmdDrawIndirectByteCount2EXT")]
    pub unsafe fn cmd_draw_indirect_byte_count2(
        &self,
        command_buffer: vk::CommandBuffer,
        instance_count: u32,
        first_instance: u32,
        counter_info: &vk::BindTransformFeedbackBuffer2InfoEXT<'_>,
        counter_offset: u32,
        vertex_stride: u32,
    ) {
        (self.fp.cmd_draw_indirect_byte_count2_ext)(
            command_buffer,
            instance_count,
            first_instance,
            counter_info,
            counter_offset,
            vertex_stride,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawMeshTasksIndirect2EXT.html>
    #[inline]
    #[doc(alias = "vkCmdDrawMeshTasksIndirect2EXT")]
    pub unsafe fn cmd_draw_mesh_tasks_indirect2(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::DrawIndirect2InfoKHR<'_>,
    ) {
        (self.fp.cmd_draw_mesh_tasks_indirect2_ext)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawMeshTasksIndirectCount2EXT.html>
    #[inline]
    #[doc(alias = "vkCmdDrawMeshTasksIndirectCount2EXT")]
    pub unsafe fn cmd_draw_mesh_tasks_indirect_count2(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::DrawIndirectCount2InfoKHR<'_>,
    ) {
        (self.fp.cmd_draw_mesh_tasks_indirect_count2_ext)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteMarkerToMemoryAMD.html>
    #[inline]
    #[doc(alias = "vkCmdWriteMarkerToMemoryAMD")]
    pub unsafe fn cmd_write_marker_to_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        info: &vk::MemoryMarkerInfoAMD<'_>,
    ) {
        (self.fp.cmd_write_marker_to_memory_amd)(command_buffer, info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateAccelerationStructure2KHR.html>
    #[inline]
    #[doc(alias = "vkCreateAccelerationStructure2KHR")]
    pub unsafe fn create_acceleration_structure2(
        &self,
        create_info: &vk::AccelerationStructureCreateInfo2KHR<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::AccelerationStructureKHR> {
        let mut accel_struct = mem::MaybeUninit::uninit();
        (self.fp.create_acceleration_structure2_khr)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            accel_struct.as_mut_ptr(),
        )
        .assume_init_on_success(accel_struct)
    }
}
