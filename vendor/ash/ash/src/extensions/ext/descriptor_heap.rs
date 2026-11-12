//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_descriptor_heap.html>

use crate::vk;
use crate::VkResult;

impl crate::ext::descriptor_heap::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkWriteSamplerDescriptorsEXT.html>
    #[inline]
    pub unsafe fn write_sampler_descriptors(
        &self,
        samplers: &[vk::SamplerCreateInfo<'_>],
        descriptors: &[vk::HostAddressRangeEXT<'_>],
    ) -> VkResult<()> {
        assert_eq!(samplers.len(), descriptors.len());
        (self.fp.write_sampler_descriptors_ext)(
            self.handle,
            samplers.len() as u32,
            samplers.as_ptr(),
            descriptors.as_ptr(),
        )
        .result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkWriteResourceDescriptorsEXT.html>
    #[inline]
    pub unsafe fn write_resource_descriptors(
        &self,
        resources: &[vk::ResourceDescriptorInfoEXT<'_>],
        descriptors: &[vk::HostAddressRangeEXT<'_>],
    ) -> VkResult<()> {
        assert_eq!(resources.len(), descriptors.len());
        (self.fp.write_resource_descriptors_ext)(
            self.handle,
            resources.len() as u32,
            resources.as_ptr(),
            descriptors.as_ptr(),
        )
        .result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindSamplerHeapEXT.html>
    #[inline]
    pub unsafe fn cmd_bind_sampler_heap(
        &self,
        command_buffer: vk::CommandBuffer,
        bind_info: &vk::BindHeapInfoEXT<'_>,
    ) {
        (self.fp.cmd_bind_sampler_heap_ext)(command_buffer, bind_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindResourceHeapEXT.html>
    #[inline]
    pub unsafe fn cmd_bind_resource_heap(
        &self,
        command_buffer: vk::CommandBuffer,
        bind_info: &vk::BindHeapInfoEXT<'_>,
    ) {
        (self.fp.cmd_bind_resource_heap_ext)(command_buffer, bind_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPushDataEXT.html>
    #[inline]
    pub unsafe fn cmd_push_data(
        &self,
        command_buffer: vk::CommandBuffer,
        push_data_info: &vk::PushDataInfoEXT<'_>,
    ) {
        (self.fp.cmd_push_data_ext)(command_buffer, push_data_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetImageOpaqueCaptureDataEXT.html>
    #[inline]
    pub unsafe fn get_image_opaque_capture_data(
        &self,
        images: &[vk::Image],
        datas: &mut [vk::HostAddressRangeEXT<'_>],
    ) -> VkResult<()> {
        assert_eq!(images.len(), datas.len());
        (self.fp.get_image_opaque_capture_data_ext)(
            self.handle,
            images.len() as u32,
            images.as_ptr(),
            datas.as_mut_ptr(),
        )
        .result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkRegisterCustomBorderColorEXT.html>
    #[inline]
    pub unsafe fn register_custom_border_color(
        &self,
        border_color: &vk::SamplerCustomBorderColorCreateInfoEXT<'_>,
        index: Option<u32>,
    ) -> VkResult<u32> {
        let request_index = index.is_some() as vk::Bool32;
        let mut index = index.unwrap_or_default();
        (self.fp.register_custom_border_color_ext)(
            self.handle,
            border_color,
            request_index,
            &mut index,
        )
        .result()?;
        Ok(index)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkUnregisterCustomBorderColorEXT.html>
    #[inline]
    pub unsafe fn unregister_custom_border_color(&self, index: u32) {
        (self.fp.unregister_custom_border_color_ext)(self.handle, index)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetTensorOpaqueCaptureDataARM.html>
    #[inline]
    pub unsafe fn get_tensor_opaque_capture_data_arm(
        &self,
        tensors: &[vk::TensorARM],
        datas: &mut [vk::HostAddressRangeEXT<'_>],
    ) -> VkResult<()> {
        assert_eq!(tensors.len(), datas.len());
        (self.fp.get_tensor_opaque_capture_data_arm)(
            self.handle,
            tensors.len() as u32,
            tensors.as_ptr(),
            datas.as_mut_ptr(),
        )
        .result()
    }
}

impl crate::ext::descriptor_heap::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceDescriptorSizeEXT.html>
    #[inline]
    pub unsafe fn get_physical_device_descriptor_size(
        &self,
        physical_device: vk::PhysicalDevice,
        descriptor_type: vk::DescriptorType,
    ) -> vk::DeviceSize {
        (self.fp.get_physical_device_descriptor_size_ext)(physical_device, descriptor_type)
    }
}
