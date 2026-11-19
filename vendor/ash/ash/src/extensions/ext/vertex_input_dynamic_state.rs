//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_vertex_input_dynamic_state.html>

use crate::vk;

impl crate::ext::vertex_input_dynamic_state::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetVertexInputEXT.html>
    #[inline]
    pub unsafe fn cmd_set_vertex_input(
        &self,
        command_buffer: vk::CommandBuffer,
        vertex_binding_descriptions: &[vk::VertexInputBindingDescription2EXT<'_>],
        vertex_attribute_descriptions: &[vk::VertexInputAttributeDescription2EXT<'_>],
    ) {
        (self.fp.cmd_set_vertex_input_ext)(
            command_buffer,
            vertex_binding_descriptions.len() as u32,
            vertex_binding_descriptions.as_ptr(),
            vertex_attribute_descriptions.len() as u32,
            vertex_attribute_descriptions.as_ptr(),
        )
    }
}
