//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_device_generated_commands.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::ext::device_generated_commands::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetGeneratedCommandsMemoryRequirementsEXT.html>
    #[inline]
    #[doc(alias = "vkGetGeneratedCommandsMemoryRequirementsEXT")]
    pub unsafe fn get_generated_commands_memory_requirements(
        &self,
        info: &vk::GeneratedCommandsMemoryRequirementsInfoEXT<'_>,
        memory_requirements: &mut vk::MemoryRequirements2<'_>,
    ) {
        (self.fp.get_generated_commands_memory_requirements_ext)(
            self.handle,
            info,
            memory_requirements,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPreprocessGeneratedCommandsEXT.html>
    #[inline]
    #[doc(alias = "vkCmdPreprocessGeneratedCommandsEXT")]
    pub unsafe fn cmd_preprocess_generated_commands(
        &self,
        command_buffer: vk::CommandBuffer,
        generated_commands_info: &vk::GeneratedCommandsInfoEXT<'_>,
        state_command_buffer: vk::CommandBuffer,
    ) {
        (self.fp.cmd_preprocess_generated_commands_ext)(
            command_buffer,
            generated_commands_info,
            state_command_buffer,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdExecuteGeneratedCommandsEXT.html>
    #[inline]
    #[doc(alias = "vkCmdExecuteGeneratedCommandsEXT")]
    pub unsafe fn cmd_execute_generated_commands(
        &self,
        command_buffer: vk::CommandBuffer,
        is_preprocessed: bool,
        generated_commands_info: &vk::GeneratedCommandsInfoEXT<'_>,
    ) {
        (self.fp.cmd_execute_generated_commands_ext)(
            command_buffer,
            is_preprocessed.into(),
            generated_commands_info,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateIndirectCommandsLayoutEXT.html>
    #[inline]
    #[doc(alias = "vkCreateIndirectCommandsLayoutEXT")]
    pub unsafe fn create_indirect_commands_layout(
        &self,
        create_info: &vk::IndirectCommandsLayoutCreateInfoEXT<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::IndirectCommandsLayoutEXT> {
        let mut indirect_commands_layout = mem::MaybeUninit::uninit();
        (self.fp.create_indirect_commands_layout_ext)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            indirect_commands_layout.as_mut_ptr(),
        )
        .assume_init_on_success(indirect_commands_layout)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyIndirectCommandsLayoutEXT.html>
    #[inline]
    #[doc(alias = "vkDestroyIndirectCommandsLayoutEXT")]
    pub unsafe fn destroy_indirect_commands_layout(
        &self,
        indirect_commands_layout: vk::IndirectCommandsLayoutEXT,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_indirect_commands_layout_ext)(
            self.handle,
            indirect_commands_layout,
            allocation_callbacks.to_raw_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateIndirectExecutionSetEXT.html>
    #[inline]
    #[doc(alias = "vkCreateIndirectExecutionSetEXT")]
    pub unsafe fn create_indirect_execution_set(
        &self,
        create_info: &vk::IndirectExecutionSetCreateInfoEXT<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::IndirectExecutionSetEXT> {
        let mut indirect_execution_set = mem::MaybeUninit::uninit();
        (self.fp.create_indirect_execution_set_ext)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            indirect_execution_set.as_mut_ptr(),
        )
        .assume_init_on_success(indirect_execution_set)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyIndirectExecutionSetEXT.html>
    #[inline]
    #[doc(alias = "vkDestroyIndirectExecutionSetEXT")]
    pub unsafe fn destroy_indirect_execution_set(
        &self,
        indirect_execution_set: vk::IndirectExecutionSetEXT,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_indirect_execution_set_ext)(
            self.handle,
            indirect_execution_set,
            allocation_callbacks.to_raw_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateIndirectExecutionSetPipelineEXT.html>
    #[inline]
    #[doc(alias = "vkUpdateIndirectExecutionSetPipelineEXT")]
    pub unsafe fn update_indirect_execution_set_pipeline(
        &self,
        indirect_execution_set: vk::IndirectExecutionSetEXT,
        execution_set_writes: &[vk::WriteIndirectExecutionSetPipelineEXT<'_>],
    ) {
        (self.fp.update_indirect_execution_set_pipeline_ext)(
            self.handle,
            indirect_execution_set,
            execution_set_writes.len() as u32,
            execution_set_writes.as_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateIndirectExecutionSetShaderEXT.html>
    #[inline]
    #[doc(alias = "vkUpdateIndirectExecutionSetShaderEXT")]
    pub unsafe fn update_indirect_execution_set_shader(
        &self,
        indirect_execution_set: vk::IndirectExecutionSetEXT,
        execution_set_writes: &[vk::WriteIndirectExecutionSetShaderEXT<'_>],
    ) {
        (self.fp.update_indirect_execution_set_shader_ext)(
            self.handle,
            indirect_execution_set,
            execution_set_writes.len() as u32,
            execution_set_writes.as_ptr(),
        )
    }
}
