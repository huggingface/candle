//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_video_queue.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use core::mem;
use core::ptr;

impl crate::khr::video_queue::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkBindVideoSessionMemoryKHR.html>
    #[inline]
    #[doc(alias = "vkBindVideoSessionMemoryKHR")]
    pub unsafe fn bind_video_session_memory(
        &self,
        video_session: vk::VideoSessionKHR,
        bind_session_memory_infos: &[vk::BindVideoSessionMemoryInfoKHR<'_>],
    ) -> VkResult<()> {
        (self.fp.bind_video_session_memory_khr)(
            self.handle,
            video_session,
            bind_session_memory_infos.len() as u32,
            bind_session_memory_infos.as_ptr(),
        )
        .result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBeginVideoCodingKHR.html>
    #[inline]
    #[doc(alias = "vkCmdBeginVideoCodingKHR")]
    pub unsafe fn cmd_begin_video_coding(
        &self,
        command_buffer: vk::CommandBuffer,
        begin_info: &vk::VideoBeginCodingInfoKHR<'_>,
    ) {
        (self.fp.cmd_begin_video_coding_khr)(command_buffer, begin_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdControlVideoCodingKHR.html>
    #[inline]
    #[doc(alias = "vkCmdControlVideoCodingKHR")]
    pub unsafe fn cmd_control_video_coding(
        &self,
        command_buffer: vk::CommandBuffer,
        coding_control_info: &vk::VideoCodingControlInfoKHR<'_>,
    ) {
        (self.fp.cmd_control_video_coding_khr)(command_buffer, coding_control_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdEndVideoCodingKHR.html>
    #[inline]
    #[doc(alias = "vkCmdEndVideoCodingKHR")]
    pub unsafe fn cmd_end_video_coding(
        &self,
        command_buffer: vk::CommandBuffer,
        end_coding_info: &vk::VideoEndCodingInfoKHR<'_>,
    ) {
        (self.fp.cmd_end_video_coding_khr)(command_buffer, end_coding_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateVideoSessionKHR.html>
    #[inline]
    #[doc(alias = "vkCreateVideoSessionKHR")]
    pub unsafe fn create_video_session(
        &self,
        create_info: &vk::VideoSessionCreateInfoKHR<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::VideoSessionKHR> {
        let mut video_session = mem::MaybeUninit::uninit();
        (self.fp.create_video_session_khr)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            video_session.as_mut_ptr(),
        )
        .assume_init_on_success(video_session)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateVideoSessionParametersKHR.html>
    #[inline]
    #[doc(alias = "vkCreateVideoSessionParametersKHR")]
    pub unsafe fn create_video_session_parameters(
        &self,
        create_info: &vk::VideoSessionParametersCreateInfoKHR<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::VideoSessionParametersKHR> {
        let mut video_session_parameters = mem::MaybeUninit::uninit();
        (self.fp.create_video_session_parameters_khr)(
            self.handle,
            create_info,
            allocation_callbacks.to_raw_ptr(),
            video_session_parameters.as_mut_ptr(),
        )
        .assume_init_on_success(video_session_parameters)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyVideoSessionKHR.html>
    #[inline]
    #[doc(alias = "vkDestroyVideoSessionKHR")]
    pub unsafe fn destroy_video_session(
        &self,
        video_session: vk::VideoSessionKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_video_session_khr)(
            self.handle,
            video_session,
            allocation_callbacks.to_raw_ptr(),
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyVideoSessionParametersKHR.html>
    #[inline]
    #[doc(alias = "vkDestroyVideoSessionParametersKHR")]
    pub unsafe fn destroy_video_session_parameters(
        &self,
        video_session_parameters: vk::VideoSessionParametersKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_video_session_parameters_khr)(
            self.handle,
            video_session_parameters,
            allocation_callbacks.to_raw_ptr(),
        )
    }

    /// Retrieve the number of elements to pass to [`get_video_session_memory_requirements()`][Self::get_video_session_memory_requirements()].
    #[inline]
    pub unsafe fn get_video_session_memory_requirements_len(
        &self,
        video_session: vk::VideoSessionKHR,
    ) -> VkResult<usize> {
        let mut count = mem::MaybeUninit::uninit();
        (self.fp.get_video_session_memory_requirements_khr)(
            self.handle,
            video_session,
            count.as_mut_ptr(),
            ptr::null_mut(),
        )
        .assume_init_on_success(count)
        .map(|c| c as usize)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetVideoSessionMemoryRequirementsKHR.html>
    ///
    /// Call [`get_video_session_memory_requirements_len()`][Self::get_video_session_memory_requirements_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    #[doc(alias = "vkGetVideoSessionMemoryRequirementsKHR")]
    pub unsafe fn get_video_session_memory_requirements(
        &self,
        video_session: vk::VideoSessionKHR,
        out: &mut [vk::VideoSessionMemoryRequirementsKHR<'_>],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self.fp.get_video_session_memory_requirements_khr)(
            self.handle,
            video_session,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateVideoSessionParametersKHR.html>
    #[inline]
    #[doc(alias = "vkUpdateVideoSessionParametersKHR")]
    pub unsafe fn update_video_session_parameters(
        &self,
        video_session_parameters: vk::VideoSessionParametersKHR,
        update_info: &vk::VideoSessionParametersUpdateInfoKHR<'_>,
    ) -> VkResult<()> {
        (self.fp.update_video_session_parameters_khr)(
            self.handle,
            video_session_parameters,
            update_info,
        )
        .result()
    }
}

impl crate::khr::video_queue::Instance {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceVideoCapabilitiesKHR.html>
    #[inline]
    #[doc(alias = "vkGetPhysicalDeviceVideoCapabilitiesKHR")]
    pub unsafe fn get_physical_device_video_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
        video_profile: &vk::VideoProfileInfoKHR<'_>,
        capabilities: &mut vk::VideoCapabilitiesKHR<'_>,
    ) -> VkResult<()> {
        (self.fp.get_physical_device_video_capabilities_khr)(
            physical_device,
            video_profile,
            capabilities,
        )
        .result()
    }

    /// Retrieve the number of elements to pass to [`get_physical_device_video_format_properties()`][Self::get_physical_device_video_format_properties()].
    #[inline]
    pub unsafe fn get_physical_device_video_format_properties_len(
        &self,
        physical_device: vk::PhysicalDevice,
        video_format_info: &vk::PhysicalDeviceVideoFormatInfoKHR<'_>,
    ) -> VkResult<usize> {
        let mut count = mem::MaybeUninit::uninit();
        (self.fp.get_physical_device_video_format_properties_khr)(
            physical_device,
            video_format_info,
            count.as_mut_ptr(),
            ptr::null_mut(),
        )
        .assume_init_on_success(count)
        .map(|c| c as usize)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceVideoFormatPropertiesKHR.html>
    ///
    /// Call [`get_physical_device_video_format_properties_len()`][Self::get_physical_device_video_format_properties_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    #[doc(alias = "vkGetPhysicalDeviceVideoFormatPropertiesKHR")]
    pub unsafe fn get_physical_device_video_format_properties(
        &self,
        physical_device: vk::PhysicalDevice,
        video_format_info: &vk::PhysicalDeviceVideoFormatInfoKHR<'_>,
        out: &mut [vk::VideoFormatPropertiesKHR<'_>],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self.fp.get_physical_device_video_format_properties_khr)(
            physical_device,
            video_format_info,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }
}
