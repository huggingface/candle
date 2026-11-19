//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_surface.html>

use crate::read_into_uninitialized_vector;
use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use alloc::vec::Vec;
use core::mem;

impl crate::khr::surface::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceSurfaceSupportKHR.html>
    #[inline]
    pub unsafe fn get_physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        surface: vk::SurfaceKHR,
    ) -> VkResult<bool> {
        let mut b = mem::MaybeUninit::uninit();
        (self.fp.get_physical_device_surface_support_khr)(
            physical_device,
            queue_family_index,
            surface,
            b.as_mut_ptr(),
        )
        .result()?;
        Ok(b.assume_init() > 0)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceSurfacePresentModesKHR.html>
    #[inline]
    pub unsafe fn get_physical_device_surface_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Vec<vk::PresentModeKHR>> {
        read_into_uninitialized_vector(|count, data| {
            (self.fp.get_physical_device_surface_present_modes_khr)(
                physical_device,
                surface,
                count,
                data,
            )
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceSurfaceCapabilitiesKHR.html>
    #[inline]
    pub unsafe fn get_physical_device_surface_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        let mut surface_capabilities = mem::MaybeUninit::uninit();
        (self.fp.get_physical_device_surface_capabilities_khr)(
            physical_device,
            surface,
            surface_capabilities.as_mut_ptr(),
        )
        .assume_init_on_success(surface_capabilities)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceSurfaceFormatsKHR.html>
    #[inline]
    pub unsafe fn get_physical_device_surface_formats(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        read_into_uninitialized_vector(|count, data| {
            (self.fp.get_physical_device_surface_formats_khr)(physical_device, surface, count, data)
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroySurfaceKHR.html>
    #[inline]
    pub unsafe fn destroy_surface(
        &self,
        surface: vk::SurfaceKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_surface_khr)(self.handle, surface, allocation_callbacks.to_raw_ptr())
    }
}
