//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_maintenance6.html>

use crate::vk;

impl crate::khr::maintenance6::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindDescriptorSets2KHR.html>
    ///
    /// Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>
    #[inline]
    #[doc(alias = "vkCmdBindDescriptorSets2KHR")]
    pub unsafe fn cmd_bind_descriptor_sets2(
        &self,
        command_buffer: vk::CommandBuffer,
        bind_descriptor_sets_info: &vk::BindDescriptorSetsInfoKHR<'_>,
    ) {
        (self.fp.cmd_bind_descriptor_sets2_khr)(command_buffer, bind_descriptor_sets_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPushConstants2KHR.html>
    ///
    /// Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>
    #[inline]
    #[doc(alias = "vkCmdPushConstants2KHR")]
    pub unsafe fn cmd_push_constants2(
        &self,
        command_buffer: vk::CommandBuffer,
        push_constants_info: &vk::PushConstantsInfoKHR<'_>,
    ) {
        (self.fp.cmd_push_constants2_khr)(command_buffer, push_constants_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPushDescriptorSet2KHR.html>
    #[inline]
    #[doc(alias = "vkCmdPushDescriptorSet2KHR")]
    pub unsafe fn cmd_push_descriptor_set2(
        &self,
        command_buffer: vk::CommandBuffer,
        push_descriptor_set_info: &vk::PushDescriptorSetInfoKHR<'_>,
    ) {
        (self.fp.cmd_push_descriptor_set2_khr)(command_buffer, push_descriptor_set_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPushDescriptorSetWithTemplate2KHR.html>
    #[inline]
    #[doc(alias = "vkCmdPushDescriptorSetWithTemplate2KHR")]
    pub unsafe fn cmd_push_descriptor_set_with_template2(
        &self,
        command_buffer: vk::CommandBuffer,
        push_descriptor_set_with_template_info: &vk::PushDescriptorSetWithTemplateInfoKHR<'_>,
    ) {
        (self.fp.cmd_push_descriptor_set_with_template2_khr)(
            command_buffer,
            push_descriptor_set_with_template_info,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetDescriptorBufferOffsets2EXT.html>
    ///
    /// Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>
    #[inline]
    #[doc(alias = "vkCmdSetDescriptorBufferOffsets2EXT")]
    pub unsafe fn cmd_set_descriptor_buffer_offsets2(
        &self,
        command_buffer: vk::CommandBuffer,
        set_descriptor_buffer_offsets_info: &vk::SetDescriptorBufferOffsetsInfoEXT<'_>,
    ) {
        (self.fp.cmd_set_descriptor_buffer_offsets2_ext)(
            command_buffer,
            set_descriptor_buffer_offsets_info,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindDescriptorBufferEmbeddedSamplers2EXT.html>
    ///
    /// Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-descriptor-sets>
    #[inline]
    #[doc(alias = "vkCmdBindDescriptorBufferEmbeddedSamplers2EXT")]
    pub unsafe fn cmd_bind_descriptor_buffer_embedded_samplers2(
        &self,
        command_buffer: vk::CommandBuffer,
        bind_descriptor_buffer_embedded_samplers_info: &vk::BindDescriptorBufferEmbeddedSamplersInfoEXT<'_>,
    ) {
        (self.fp.cmd_bind_descriptor_buffer_embedded_samplers2_ext)(
            command_buffer,
            bind_descriptor_buffer_embedded_samplers_info,
        )
    }
}
