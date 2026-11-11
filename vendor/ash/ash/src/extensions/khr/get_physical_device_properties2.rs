//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_get_physical_device_properties2.html>

use crate::vk;
use crate::VkResult;
use core::mem;
use core::ptr;

impl crate::khr::get_physical_device_properties2::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceFeatures2KHR.html>
    #[inline]
    pub unsafe fn get_physical_device_features2(
        &self,
        physical_device: vk::PhysicalDevice,
        features: &mut vk::PhysicalDeviceFeatures2KHR<'_>,
    ) {
        (self.fp.get_physical_device_features2_khr)(physical_device, features)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceFormatProperties2KHR.html>
    #[inline]
    pub unsafe fn get_physical_device_format_properties2(
        &self,
        physical_device: vk::PhysicalDevice,
        format: vk::Format,
        format_properties: &mut vk::FormatProperties2KHR<'_>,
    ) {
        (self.fp.get_physical_device_format_properties2_khr)(
            physical_device,
            format,
            format_properties,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceImageFormatProperties2KHR.html>
    #[inline]
    pub unsafe fn get_physical_device_image_format_properties2(
        &self,
        physical_device: vk::PhysicalDevice,
        image_format_info: &vk::PhysicalDeviceImageFormatInfo2KHR<'_>,
        image_format_properties: &mut vk::ImageFormatProperties2KHR<'_>,
    ) -> VkResult<()> {
        (self.fp.get_physical_device_image_format_properties2_khr)(
            physical_device,
            image_format_info,
            image_format_properties,
        )
        .result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceMemoryProperties2KHR.html>
    #[inline]
    pub unsafe fn get_physical_device_memory_properties2(
        &self,
        physical_device: vk::PhysicalDevice,
        memory_properties: &mut vk::PhysicalDeviceMemoryProperties2KHR<'_>,
    ) {
        (self.fp.get_physical_device_memory_properties2_khr)(physical_device, memory_properties)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceProperties2KHR.html>
    #[inline]
    pub unsafe fn get_physical_device_properties2(
        &self,
        physical_device: vk::PhysicalDevice,
        properties: &mut vk::PhysicalDeviceProperties2KHR<'_>,
    ) {
        (self.fp.get_physical_device_properties2_khr)(physical_device, properties)
    }

    /// Retrieve the number of elements to pass to [`get_physical_device_queue_family_properties2()`][Self::get_physical_device_queue_family_properties2()]
    #[inline]
    pub unsafe fn get_physical_device_queue_family_properties2_len(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> usize {
        let mut count = mem::MaybeUninit::uninit();
        (self.fp.get_physical_device_queue_family_properties2_khr)(
            physical_device,
            count.as_mut_ptr(),
            ptr::null_mut(),
        );
        count.assume_init() as usize
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceQueueFamilyProperties2KHR.html>
    ///
    /// Call [`get_physical_device_queue_family_properties2_len()`][Self::get_physical_device_queue_family_properties2_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    pub unsafe fn get_physical_device_queue_family_properties2(
        &self,
        physical_device: vk::PhysicalDevice,
        out: &mut [vk::QueueFamilyProperties2KHR<'_>],
    ) {
        let mut count = out.len() as u32;
        (self.fp.get_physical_device_queue_family_properties2_khr)(
            physical_device,
            &mut count,
            out.as_mut_ptr(),
        );
        assert_eq!(count as usize, out.len());
    }

    /// Retrieve the number of elements to pass to [`get_physical_device_sparse_image_format_properties2()`][Self::get_physical_device_sparse_image_format_properties2()]
    #[inline]
    pub unsafe fn get_physical_device_sparse_image_format_properties2_len(
        &self,
        physical_device: vk::PhysicalDevice,
        format_info: &vk::PhysicalDeviceSparseImageFormatInfo2KHR<'_>,
    ) -> usize {
        let mut count = mem::MaybeUninit::uninit();
        (self
            .fp
            .get_physical_device_sparse_image_format_properties2_khr)(
            physical_device,
            format_info,
            count.as_mut_ptr(),
            ptr::null_mut(),
        );
        count.assume_init() as usize
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceSparseImageFormatProperties2KHR.html>
    ///
    /// Call [`get_physical_device_sparse_image_format_properties2_len()`][Self::get_physical_device_sparse_image_format_properties2_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    pub unsafe fn get_physical_device_sparse_image_format_properties2(
        &self,
        physical_device: vk::PhysicalDevice,
        format_info: &vk::PhysicalDeviceSparseImageFormatInfo2KHR<'_>,
        out: &mut [vk::SparseImageFormatProperties2KHR<'_>],
    ) {
        let mut count = out.len() as u32;
        (self
            .fp
            .get_physical_device_sparse_image_format_properties2_khr)(
            physical_device,
            format_info,
            &mut count,
            out.as_mut_ptr(),
        );
        assert_eq!(count as usize, out.len());
    }
}
