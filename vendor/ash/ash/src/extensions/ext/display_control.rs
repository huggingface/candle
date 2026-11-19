//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_display_control.html>

use crate::vk;
use crate::RawPtr as _;
use crate::VkResult;
use core::mem;

impl crate::ext::display_control::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDisplayPowerControlEXT.html>
    #[inline]
    #[doc(alias = "vkDisplayPowerControlEXT")]
    pub unsafe fn display_power_control(
        &self,
        display: vk::DisplayKHR,
        display_power_info: &vk::DisplayPowerInfoEXT<'_>,
    ) -> VkResult<()> {
        (self.fp.display_power_control_ext)(self.handle, display, display_power_info).result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkRegisterDeviceEventEXT.html>
    #[inline]
    #[doc(alias = "vkRegisterDeviceEventEXT")]
    pub unsafe fn register_device_event(
        &self,
        device_event_info: &vk::DeviceEventInfoEXT<'_>,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::Fence> {
        let mut fence = mem::MaybeUninit::uninit();
        (self.fp.register_device_event_ext)(
            self.handle,
            device_event_info,
            allocator.to_raw_ptr(),
            fence.as_mut_ptr(),
        )
        .assume_init_on_success(fence)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkRegisterDisplayEventEXT.html>
    #[inline]
    #[doc(alias = "vkRegisterDisplayEventEXT")]
    pub unsafe fn register_display_event(
        &self,
        display: vk::DisplayKHR,
        display_event_info: &vk::DisplayEventInfoEXT<'_>,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::Fence> {
        let mut fence = mem::MaybeUninit::uninit();
        (self.fp.register_display_event_ext)(
            self.handle,
            display,
            display_event_info,
            allocator.to_raw_ptr(),
            fence.as_mut_ptr(),
        )
        .assume_init_on_success(fence)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetSwapchainCounterEXT.html>
    #[inline]
    #[doc(alias = "vkGetSwapchainCounterEXT")]
    pub unsafe fn get_swapchain_counter(
        &self,
        swapchain: vk::SwapchainKHR,
        counter: vk::SurfaceCounterFlagsEXT,
    ) -> VkResult<u64> {
        let mut counter_value = mem::MaybeUninit::uninit();
        (self.fp.get_swapchain_counter_ext)(
            self.handle,
            swapchain,
            counter,
            counter_value.as_mut_ptr(),
        )
        .assume_init_on_success(counter_value)
    }
}
