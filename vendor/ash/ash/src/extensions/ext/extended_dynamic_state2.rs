//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_extended_dynamic_state2.html>

use crate::vk;

impl crate::ext::extended_dynamic_state2::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetPatchControlPointsEXT.html>
    #[inline]
    pub unsafe fn cmd_set_patch_control_points(
        &self,
        command_buffer: vk::CommandBuffer,
        patch_control_points: u32,
    ) {
        (self.fp.cmd_set_patch_control_points_ext)(command_buffer, patch_control_points)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetRasterizerDiscardEnableEXT.html>
    #[inline]
    pub unsafe fn cmd_set_rasterizer_discard_enable(
        &self,
        command_buffer: vk::CommandBuffer,
        rasterizer_discard_enable: bool,
    ) {
        (self.fp.cmd_set_rasterizer_discard_enable_ext)(
            command_buffer,
            rasterizer_discard_enable.into(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetDepthBiasEnableEXT.html>
    #[inline]
    pub unsafe fn cmd_set_depth_bias_enable(
        &self,
        command_buffer: vk::CommandBuffer,
        depth_bias_enable: bool,
    ) {
        (self.fp.cmd_set_depth_bias_enable_ext)(command_buffer, depth_bias_enable.into())
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetLogicOpEXT.html>
    #[inline]
    pub unsafe fn cmd_set_logic_op(
        &self,
        command_buffer: vk::CommandBuffer,
        logic_op: vk::LogicOp,
    ) {
        (self.fp.cmd_set_logic_op_ext)(command_buffer, logic_op)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetPrimitiveRestartEnableEXT.html>
    #[inline]
    pub unsafe fn cmd_set_primitive_restart_enable(
        &self,
        command_buffer: vk::CommandBuffer,
        primitive_restart_enable: bool,
    ) {
        (self.fp.cmd_set_primitive_restart_enable_ext)(
            command_buffer,
            primitive_restart_enable.into(),
        )
    }
}
