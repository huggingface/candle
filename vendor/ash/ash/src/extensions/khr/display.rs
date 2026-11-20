//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_display.html>

use crate::read_into_uninitialized_vector;
use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use alloc::vec::Vec;
use core::mem;

impl crate::khr::display::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceDisplayPropertiesKHR.html>
    #[inline]
    pub unsafe fn get_physical_device_display_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VkResult<Vec<vk::DisplayPropertiesKHR<'_>>> {
        read_into_uninitialized_vector(|count, data| {
            (self.fp.get_physical_device_display_properties_khr)(physical_device, count, data)
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceDisplayPlanePropertiesKHR.html>
    #[inline]
    pub unsafe fn get_physical_device_display_plane_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VkResult<Vec<vk::DisplayPlanePropertiesKHR>> {
        read_into_uninitialized_vector(|count, data| {
            (self.fp.get_physical_device_display_plane_properties_khr)(physical_device, count, data)
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDisplayPlaneSupportedDisplaysKHR.html>
    #[inline]
    pub unsafe fn get_display_plane_supported_displays(
        &self,
        physical_device: vk::PhysicalDevice,
        plane_index: u32,
    ) -> VkResult<Vec<vk::DisplayKHR>> {
        read_into_uninitialized_vector(|count, data| {
            (self.fp.get_display_plane_supported_displays_khr)(
                physical_device,
                plane_index,
                count,
                data,
            )
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDisplayModePropertiesKHR.html>
    #[inline]
    pub unsafe fn get_display_mode_properties(
        &self,
        physical_device: vk::PhysicalDevice,
        display: vk::DisplayKHR,
    ) -> VkResult<Vec<vk::DisplayModePropertiesKHR>> {
        read_into_uninitialized_vector(|count, data| {
            (self.fp.get_display_mode_properties_khr)(physical_device, display, count, data)
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateDisplayModeKHR.html>
    #[inline]
    pub unsafe fn create_display_mode(
        &self,
        physical_device: vk::PhysicalDevice,
        display: vk::DisplayKHR,
        create_info: &vk::DisplayModeCreateInfoKHR<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::DisplayModeKHR> {
        let mut display_mode = mem::MaybeUninit::uninit();
        (self.fp.create_display_mode_khr)(
            physical_device,
            display,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            display_mode.as_mut_ptr(),
        )
        .assume_init_on_success(display_mode)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDisplayPlaneCapabilitiesKHR.html>
    #[inline]
    pub unsafe fn get_display_plane_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
        mode: vk::DisplayModeKHR,
        plane_index: u32,
    ) -> VkResult<vk::DisplayPlaneCapabilitiesKHR> {
        let mut display_plane_capabilities = mem::MaybeUninit::uninit();
        (self.fp.get_display_plane_capabilities_khr)(
            physical_device,
            mode,
            plane_index,
            display_plane_capabilities.as_mut_ptr(),
        )
        .assume_init_on_success(display_plane_capabilities)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateDisplayPlaneSurfaceKHR.html>
    #[inline]
    pub unsafe fn create_display_plane_surface(
        &self,
        create_info: &vk::DisplaySurfaceCreateInfoKHR<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::SurfaceKHR> {
        let mut surface = mem::MaybeUninit::uninit();
        (self.fp.create_display_plane_surface_khr)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            surface.as_mut_ptr(),
        )
        .assume_init_on_success(surface)
    }
}
