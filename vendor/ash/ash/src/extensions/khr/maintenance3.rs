//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_maintenance3.html>

use crate::vk;

impl crate::khr::maintenance3::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDescriptorSetLayoutSupportKHR.html>
    #[inline]
    pub unsafe fn get_descriptor_set_layout_support(
        &self,
        create_info: &vk::DescriptorSetLayoutCreateInfo<'_>,
        out: &mut vk::DescriptorSetLayoutSupportKHR<'_>,
    ) {
        (self.fp.get_descriptor_set_layout_support_khr)(self.handle, create_info, out)
    }
}
