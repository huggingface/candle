//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_image_drm_format_modifier.html>

use crate::vk;
use crate::VkResult;

impl crate::ext::image_drm_format_modifier::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetImageDrmFormatModifierPropertiesEXT.html>
    #[inline]
    pub unsafe fn get_image_drm_format_modifier_properties(
        &self,
        image: vk::Image,
        properties: &mut vk::ImageDrmFormatModifierPropertiesEXT<'_>,
    ) -> VkResult<()> {
        (self.fp.get_image_drm_format_modifier_properties_ext)(self.handle, image, properties)
            .result()
    }
}
