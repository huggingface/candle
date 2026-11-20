//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_maintenance5.html>

#[cfg(doc)]
use crate::ext;
use crate::vk;
use core::mem;

impl crate::khr::maintenance5::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindIndexBuffer2KHR.html>
    ///
    /// Deprecated: <https://docs.vulkan.org/spec/latest/appendices/legacy.html#legacy-buffer-commands>
    #[inline]
    #[doc(alias = "vkCmdBindIndexBuffer2KHR")]
    pub unsafe fn cmd_bind_index_buffer2(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        (self.fp.cmd_bind_index_buffer2_khr)(command_buffer, buffer, offset, size, index_type)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetRenderingAreaGranularityKHR.html>
    #[inline]
    #[doc(alias = "vkGetRenderingAreaGranularityKHR")]
    pub unsafe fn get_rendering_area_granularity(
        &self,
        rendering_area_info: &vk::RenderingAreaInfoKHR<'_>,
    ) -> vk::Extent2D {
        let mut granularity = mem::MaybeUninit::uninit();
        (self.fp.get_rendering_area_granularity_khr)(
            self.handle,
            rendering_area_info,
            granularity.as_mut_ptr(),
        );
        granularity.assume_init()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceImageSubresourceLayoutKHR.html>
    #[inline]
    #[doc(alias = "vkGetDeviceImageSubresourceLayoutKHR")]
    pub unsafe fn get_device_image_subresource_layout(
        &self,
        info: &vk::DeviceImageSubresourceInfoKHR<'_>,
        layout: &mut vk::SubresourceLayout2KHR<'_>,
    ) {
        (self.fp.get_device_image_subresource_layout_khr)(self.handle, info, layout)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetImageSubresourceLayout2KHR.html>
    ///
    /// Also available as [`ext::host_image_copy::Device::get_image_subresource_layout2()`]
    /// when [`VK_EXT_host_image_copy`] is enabled.
    ///
    /// Also available as [`ext::image_compression_control::Device::get_image_subresource_layout2()`]
    /// when [`VK_EXT_image_compression_control`] is enabled.
    ///
    /// [`VK_EXT_host_image_copy`]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_host_image_copy.html
    /// [`VK_EXT_image_compression_control`]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_image_compression_control.html
    #[inline]
    #[doc(alias = "vkGetImageSubresourceLayout2KHR")]
    pub unsafe fn get_image_subresource_layout2(
        &self,
        image: vk::Image,
        subresource: &vk::ImageSubresource2KHR<'_>,
        layout: &mut vk::SubresourceLayout2KHR<'_>,
    ) {
        (self.fp.get_image_subresource_layout2_khr)(self.handle, image, subresource, layout)
    }
}
