//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_debug_utils.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::ext::debug_utils::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkSetDebugUtilsObjectNameEXT.html>
    #[inline]
    pub unsafe fn set_debug_utils_object_name(
        &self,
        name_info: &vk::DebugUtilsObjectNameInfoEXT<'_>,
    ) -> VkResult<()> {
        (self.fp.set_debug_utils_object_name_ext)(self.handle, name_info).result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkSetDebugUtilsObjectTagEXT.html>
    #[inline]
    pub unsafe fn set_debug_utils_object_tag(
        &self,
        tag_info: &vk::DebugUtilsObjectTagInfoEXT<'_>,
    ) -> VkResult<()> {
        (self.fp.set_debug_utils_object_tag_ext)(self.handle, tag_info).result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBeginDebugUtilsLabelEXT.html>
    #[inline]
    pub unsafe fn cmd_begin_debug_utils_label(
        &self,
        command_buffer: vk::CommandBuffer,
        label: &vk::DebugUtilsLabelEXT<'_>,
    ) {
        (self.fp.cmd_begin_debug_utils_label_ext)(command_buffer, label)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdEndDebugUtilsLabelEXT.html>
    #[inline]
    pub unsafe fn cmd_end_debug_utils_label(&self, command_buffer: vk::CommandBuffer) {
        (self.fp.cmd_end_debug_utils_label_ext)(command_buffer)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdInsertDebugUtilsLabelEXT.html>
    #[inline]
    pub unsafe fn cmd_insert_debug_utils_label(
        &self,
        command_buffer: vk::CommandBuffer,
        label: &vk::DebugUtilsLabelEXT<'_>,
    ) {
        (self.fp.cmd_insert_debug_utils_label_ext)(command_buffer, label)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueBeginDebugUtilsLabelEXT.html>
    #[inline]
    pub unsafe fn queue_begin_debug_utils_label(
        &self,
        queue: vk::Queue,
        label: &vk::DebugUtilsLabelEXT<'_>,
    ) {
        (self.fp.queue_begin_debug_utils_label_ext)(queue, label)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueEndDebugUtilsLabelEXT.html>
    #[inline]
    pub unsafe fn queue_end_debug_utils_label(&self, queue: vk::Queue) {
        (self.fp.queue_end_debug_utils_label_ext)(queue)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueInsertDebugUtilsLabelEXT.html>
    #[inline]
    pub unsafe fn queue_insert_debug_utils_label(
        &self,
        queue: vk::Queue,
        label: &vk::DebugUtilsLabelEXT<'_>,
    ) {
        (self.fp.queue_insert_debug_utils_label_ext)(queue, label)
    }
}

impl crate::ext::debug_utils::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateDebugUtilsMessengerEXT.html>
    #[inline]
    pub unsafe fn create_debug_utils_messenger(
        &self,
        create_info: &vk::DebugUtilsMessengerCreateInfoEXT<'_>,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::DebugUtilsMessengerEXT> {
        let mut messenger = mem::MaybeUninit::uninit();
        (self.fp.create_debug_utils_messenger_ext)(
            self.handle,
            create_info,
            allocator.to_raw_ptr(),
            messenger.as_mut_ptr(),
        )
        .assume_init_on_success(messenger)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyDebugUtilsMessengerEXT.html>
    #[inline]
    pub unsafe fn destroy_debug_utils_messenger(
        &self,
        messenger: vk::DebugUtilsMessengerEXT,
        allocator: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_debug_utils_messenger_ext)(self.handle, messenger, allocator.to_raw_ptr())
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkSubmitDebugUtilsMessageEXT.html>
    #[inline]
    pub unsafe fn submit_debug_utils_message(
        &self,
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_types: vk::DebugUtilsMessageTypeFlagsEXT,
        callback_data: &vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    ) {
        (self.fp.submit_debug_utils_message_ext)(
            self.handle,
            message_severity,
            message_types,
            callback_data,
        )
    }
}
