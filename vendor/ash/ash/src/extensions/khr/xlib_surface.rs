//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_xlib_surface.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;

impl crate::khr::xlib_surface::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateXlibSurfaceKHR.html>
    #[inline]
    pub unsafe fn create_xlib_surface(
        &self,
        create_info: &vk::XlibSurfaceCreateInfoKHR<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::SurfaceKHR> {
        let mut surface = mem::MaybeUninit::uninit();
        (self.fp.create_xlib_surface_khr)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            surface.as_mut_ptr(),
        )
        .assume_init_on_success(surface)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceXlibPresentationSupportKHR.html>
    #[inline]
    pub unsafe fn get_physical_device_xlib_presentation_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        display: *mut vk::Display,
        visual_id: vk::VisualID,
    ) -> bool {
        let b = (self.fp.get_physical_device_xlib_presentation_support_khr)(
            physical_device,
            queue_family_index,
            display,
            visual_id,
        );

        b > 0
    }
}
