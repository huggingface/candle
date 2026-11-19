//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_mesh_shader.html>

use crate::vk;

impl crate::nv::mesh_shader::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawMeshTasksNV.html>
    #[inline]
    pub unsafe fn cmd_draw_mesh_tasks(
        &self,
        command_buffer: vk::CommandBuffer,
        task_count: u32,
        first_task: u32,
    ) {
        (self.fp.cmd_draw_mesh_tasks_nv)(command_buffer, task_count, first_task)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawMeshTasksIndirectNV.html>
    #[inline]
    pub unsafe fn cmd_draw_mesh_tasks_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        (self.fp.cmd_draw_mesh_tasks_indirect_nv)(
            command_buffer,
            buffer,
            offset,
            draw_count,
            stride,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDrawMeshTasksIndirectCountNV.html>
    #[inline]
    pub unsafe fn cmd_draw_mesh_tasks_indirect_count(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_buffer_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        (self.fp.cmd_draw_mesh_tasks_indirect_count_nv)(
            command_buffer,
            buffer,
            offset,
            count_buffer,
            count_buffer_offset,
            max_draw_count,
            stride,
        )
    }
}
