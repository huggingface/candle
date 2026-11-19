//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_buffer_device_address.html>

use crate::vk;

impl crate::ext::buffer_device_address::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetBufferDeviceAddressEXT.html>
    #[inline]
    pub unsafe fn get_buffer_device_address(
        &self,
        info: &vk::BufferDeviceAddressInfoEXT<'_>,
    ) -> vk::DeviceAddress {
        (self.fp.get_buffer_device_address_ext)(self.handle, info)
    }
}
