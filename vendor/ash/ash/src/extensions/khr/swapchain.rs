//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_swapchain.html>

#[cfg(doc)]
use crate::khr;
use crate::read_into_uninitialized_vector;
use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use alloc::vec::Vec;
use core::mem;

impl crate::khr::swapchain::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateSwapchainKHR.html>
    #[inline]
    pub unsafe fn create_swapchain(
        &self,
        create_info: &vk::SwapchainCreateInfoKHR<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::SwapchainKHR> {
        let mut swapchain = mem::MaybeUninit::uninit();
        (self.fp.create_swapchain_khr)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            swapchain.as_mut_ptr(),
        )
        .assume_init_on_success(swapchain)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroySwapchainKHR.html>
    #[inline]
    pub unsafe fn destroy_swapchain(
        &self,
        swapchain: vk::SwapchainKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_swapchain_khr)(self.handle, swapchain, allocation_callbacks.to_raw_ptr())
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetSwapchainImagesKHR.html>
    #[inline]
    pub unsafe fn get_swapchain_images(
        &self,
        swapchain: vk::SwapchainKHR,
    ) -> VkResult<Vec<vk::Image>> {
        read_into_uninitialized_vector(|count, data| {
            (self.fp.get_swapchain_images_khr)(self.handle, swapchain, count, data)
        })
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkAcquireNextImageKHR.html>
    ///
    /// # Returns
    /// Returns the next image's index, and [`false`] if the swapchain is optimal for the surface
    /// ([`vk::Result::SUCCESS`]), [`true`] if the swapchain is _suboptimal_ for the surface
    /// ([`vk::Result::SUBOPTIMAL_KHR`]), or [`Err`] on failure.
    #[inline]
    pub unsafe fn acquire_next_image(
        &self,
        swapchain: vk::SwapchainKHR,
        timeout: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> VkResult<(u32, bool)> {
        let mut index = mem::MaybeUninit::uninit();
        let err_code = (self.fp.acquire_next_image_khr)(
            self.handle,
            swapchain,
            timeout,
            semaphore,
            fence,
            index.as_mut_ptr(),
        );
        match err_code {
            vk::Result::SUCCESS => Ok((index.assume_init(), false)),
            vk::Result::SUBOPTIMAL_KHR => Ok((index.assume_init(), true)),
            _ => Err(err_code),
        }
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkQueuePresentKHR.html>
    ///
    /// # Returns
    /// Returns [`false`] if the swapchain is optimal for the surface ([`vk::Result::SUCCESS`]),
    /// [`true`] if the swapchain is _suboptimal_ for the surface ([`vk::Result::SUBOPTIMAL_KHR`]),
    /// or [`Err`] on failure.
    #[inline]
    pub unsafe fn queue_present(
        &self,
        queue: vk::Queue,
        present_info: &vk::PresentInfoKHR<'_>,
    ) -> VkResult<bool> {
        let err_code = (self.fp.queue_present_khr)(queue, present_info);
        match err_code {
            vk::Result::SUCCESS => Ok(false),
            vk::Result::SUBOPTIMAL_KHR => Ok(true),
            _ => Err(err_code),
        }
    }

    /// Only available since [Vulkan 1.1].
    ///
    /// Also available as [`khr::device_group::Device::get_device_group_present_capabilities()`]
    /// when [`VK_KHR_surface`] is enabled.
    ///
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceGroupPresentCapabilitiesKHR.html>
    ///
    /// [Vulkan 1.1]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_VERSION_1_1.html
    /// [`VK_KHR_surface`]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_surface.html
    #[inline]
    pub unsafe fn get_device_group_present_capabilities(
        &self,
        device_group_present_capabilities: &mut vk::DeviceGroupPresentCapabilitiesKHR<'_>,
    ) -> VkResult<()> {
        (self.fp.get_device_group_present_capabilities_khr)(
            self.handle,
            device_group_present_capabilities,
        )
        .result()
    }

    /// Only available since [Vulkan 1.1].
    ///
    /// Also available as [`khr::device_group::Device::get_device_group_surface_present_modes()`]
    /// when [`VK_KHR_surface`] is enabled.
    ///
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceGroupSurfacePresentModesKHR.html>
    ///
    /// [Vulkan 1.1]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_VERSION_1_1.html
    /// [`VK_KHR_surface`]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_surface.html
    #[inline]
    pub unsafe fn get_device_group_surface_present_modes(
        &self,
        surface: vk::SurfaceKHR,
    ) -> VkResult<vk::DeviceGroupPresentModeFlagsKHR> {
        let mut modes = mem::MaybeUninit::uninit();
        (self.fp.get_device_group_surface_present_modes_khr)(
            self.handle,
            surface,
            modes.as_mut_ptr(),
        )
        .assume_init_on_success(modes)
    }

    /// Only available since [Vulkan 1.1].
    ///
    /// Also available as [`khr::device_group::Device::acquire_next_image2()`]
    /// when [`VK_KHR_swapchain`] is enabled.
    ///
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkAcquireNextImage2KHR.html>
    ///
    /// [Vulkan 1.1]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_VERSION_1_1.html
    /// [`VK_KHR_swapchain`]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_swapchain.html
    ///
    /// # Returns
    /// Returns the next image's index, and [`false`] if the swapchain is optimal for the surface
    /// ([`vk::Result::SUCCESS`]), [`true`] if the swapchain is _suboptimal_ for the surface
    /// ([`vk::Result::SUBOPTIMAL_KHR`]), or [`Err`] on failure.
    #[inline]
    pub unsafe fn acquire_next_image2(
        &self,
        acquire_info: &vk::AcquireNextImageInfoKHR<'_>,
    ) -> VkResult<(u32, bool)> {
        let mut index = mem::MaybeUninit::uninit();
        let err_code =
            (self.fp.acquire_next_image2_khr)(self.handle, acquire_info, index.as_mut_ptr());
        match err_code {
            vk::Result::SUCCESS => Ok((index.assume_init(), false)),
            vk::Result::SUBOPTIMAL_KHR => Ok((index.assume_init(), true)),
            _ => Err(err_code),
        }
    }
}

impl crate::khr::swapchain::Instance {
    /// Only available since [Vulkan 1.1].
    ///
    /// Also available as [`khr::device_group::Instance::get_physical_device_present_rectangles()`]
    /// when [`VK_KHR_surface`] is enabled.
    ///
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDevicePresentRectanglesKHR.html>
    ///
    /// [Vulkan 1.1]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_VERSION_1_1.html
    /// [`VK_KHR_surface`]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_surface.html
    #[inline]
    pub unsafe fn get_physical_device_present_rectangles(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Vec<vk::Rect2D>> {
        read_into_uninitialized_vector(|count, data| {
            (self.fp.get_physical_device_present_rectangles_khr)(
                physical_device,
                surface,
                count,
                data,
            )
        })
    }
}
