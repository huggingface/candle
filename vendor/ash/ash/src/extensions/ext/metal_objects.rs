//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_metal_objects.html>

use crate::vk;

impl crate::ext::metal_objects::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkExportMetalObjectsEXT.html>
    #[inline]
    #[doc(alias = "vkExportMetalObjectsEXT")]
    pub unsafe fn export_metal_objects(
        &self,
        metal_objects_info: &mut vk::ExportMetalObjectsInfoEXT<'_>,
    ) {
        (self.fp.export_metal_objects_ext)(self.handle, metal_objects_info)
    }
}
