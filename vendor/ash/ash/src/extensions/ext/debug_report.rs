//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_debug_report.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::ext::debug_report::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyDebugReportCallbackEXT.html>
    #[inline]
    pub unsafe fn destroy_debug_report_callback(
        &self,
        debug: vk::DebugReportCallbackEXT,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_debug_report_callback_ext)(
            self.handle,
            debug,
            allocation_callbacks.to_raw_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateDebugReportCallbackEXT.html>
    #[inline]
    pub unsafe fn create_debug_report_callback(
        &self,
        create_info: &vk::DebugReportCallbackCreateInfoEXT<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::DebugReportCallbackEXT> {
        let mut debug_cb = mem::MaybeUninit::uninit();
        (self.fp.create_debug_report_callback_ext)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            debug_cb.as_mut_ptr(),
        )
        .assume_init_on_success(debug_cb)
    }
}
