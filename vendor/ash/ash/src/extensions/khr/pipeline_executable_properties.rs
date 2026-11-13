//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_executable_properties.html>

use crate::vk;
use crate::VkResult;
use core::mem;
use core::ptr;

impl crate::khr::pipeline_executable_properties::Device {
    /// Retrieve the number of elements to pass to [`get_pipeline_executable_internal_representations()`][Self::get_pipeline_executable_internal_representations()]
    #[inline]
    pub unsafe fn get_pipeline_executable_internal_representations_len(
        &self,
        executable_info: &vk::PipelineExecutableInfoKHR<'_>,
    ) -> VkResult<usize> {
        let mut count = mem::MaybeUninit::uninit();
        (self.fp.get_pipeline_executable_internal_representations_khr)(
            self.handle,
            executable_info,
            count.as_mut_ptr(),
            ptr::null_mut(),
        )
        .assume_init_on_success(count)
        .map(|c| c as usize)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPipelineExecutableInternalRepresentationsKHR.html>
    ///
    /// Call [`get_pipeline_executable_internal_representations_len()`][Self::get_pipeline_executable_internal_representations_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    #[doc(alias = "vkGetPipelineExecutableInternalRepresentationsKHR")]
    pub unsafe fn get_pipeline_executable_internal_representations(
        &self,
        executable_info: &vk::PipelineExecutableInfoKHR<'_>,
        out: &mut [vk::PipelineExecutableInternalRepresentationKHR<'_>],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self.fp.get_pipeline_executable_internal_representations_khr)(
            self.handle,
            executable_info,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }

    /// Retrieve the number of elements to pass to [`get_pipeline_executable_properties()`][Self::get_pipeline_executable_properties()]
    #[inline]
    pub unsafe fn get_pipeline_executable_properties_len(
        &self,
        pipeline_info: &vk::PipelineInfoKHR<'_>,
    ) -> VkResult<usize> {
        let mut count = mem::MaybeUninit::uninit();
        (self.fp.get_pipeline_executable_properties_khr)(
            self.handle,
            pipeline_info,
            count.as_mut_ptr(),
            ptr::null_mut(),
        )
        .assume_init_on_success(count)
        .map(|c| c as usize)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPipelineExecutablePropertiesKHR.html>
    ///
    /// Call [`get_pipeline_executable_properties_len()`][Self::get_pipeline_executable_properties_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    #[doc(alias = "vkGetPipelineExecutablePropertiesKHR")]
    pub unsafe fn get_pipeline_executable_properties(
        &self,
        pipeline_info: &vk::PipelineInfoKHR<'_>,
        out: &mut [vk::PipelineExecutablePropertiesKHR<'_>],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self.fp.get_pipeline_executable_properties_khr)(
            self.handle,
            pipeline_info,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }

    /// Retrieve the number of elements to pass to [`get_pipeline_executable_statistics()`][Self::get_pipeline_executable_statistics()]
    #[inline]
    pub unsafe fn get_pipeline_executable_statistics_len(
        &self,
        executable_info: &vk::PipelineExecutableInfoKHR<'_>,
    ) -> VkResult<usize> {
        let mut count = mem::MaybeUninit::uninit();
        (self.fp.get_pipeline_executable_statistics_khr)(
            self.handle,
            executable_info,
            count.as_mut_ptr(),
            ptr::null_mut(),
        )
        .assume_init_on_success(count)
        .map(|c| c as usize)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPipelineExecutableStatisticsKHR.html>
    ///
    /// Call [`get_pipeline_executable_statistics_len()`][Self::get_pipeline_executable_statistics_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    #[inline]
    #[doc(alias = "vkGetPipelineExecutableStatisticsKHR")]
    pub unsafe fn get_pipeline_executable_statistics(
        &self,
        executable_info: &vk::PipelineExecutableInfoKHR<'_>,
        out: &mut [vk::PipelineExecutableStatisticKHR<'_>],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self.fp.get_pipeline_executable_statistics_khr)(
            self.handle,
            executable_info,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }
}
