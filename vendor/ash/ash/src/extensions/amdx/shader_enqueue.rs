#![cfg(feature = "provisional")]
//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_AMDX_shader_enqueue.html>

use crate::vk;
use crate::RawPtr;
use crate::VkResult;
use alloc::vec::Vec;
use core::mem;

impl crate::amdx::shader_enqueue::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateExecutionGraphPipelinesAMDX.html>
    ///
    /// Pipelines are created and returned as described for [Multiple Pipeline Creation].
    ///
    /// [Multiple Pipeline Creation]: https://docs.vulkan.org/spec/latest/chapters/pipelines.html#pipelines-multiple
    #[inline]
    pub unsafe fn create_execution_graph_pipelines(
        &self,
        pipeline_cache: vk::PipelineCache,
        create_infos: &[vk::ExecutionGraphPipelineCreateInfoAMDX<'_>],
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)> {
        let mut pipelines = Vec::with_capacity(create_infos.len());
        let err_code = (self.fp.create_execution_graph_pipelines_amdx)(
            self.handle,
            pipeline_cache,
            create_infos.len() as u32,
            create_infos.as_ptr(),
            allocation_callbacks.to_raw_ptr(),
            pipelines.as_mut_ptr(),
        );
        pipelines.set_len(create_infos.len());
        match err_code {
            vk::Result::SUCCESS => Ok(pipelines),
            _ => Err((pipelines, err_code)),
        }
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetExecutionGraphPipelineScratchSizeAMDX.html>
    #[inline]
    pub unsafe fn get_execution_graph_pipeline_scratch_size(
        &self,
        execution_graph: vk::Pipeline,
        size_info: &mut vk::ExecutionGraphPipelineScratchSizeAMDX<'_>,
    ) -> VkResult<()> {
        (self.fp.get_execution_graph_pipeline_scratch_size_amdx)(
            self.handle,
            execution_graph,
            size_info,
        )
        .result()
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetExecutionGraphPipelineNodeIndexAMDX.html>
    #[inline]
    pub unsafe fn get_execution_graph_pipeline_node_index(
        &self,
        execution_graph: vk::Pipeline,
        node_info: &vk::PipelineShaderStageNodeCreateInfoAMDX<'_>,
    ) -> VkResult<u32> {
        let mut node_index = mem::MaybeUninit::uninit();
        (self.fp.get_execution_graph_pipeline_node_index_amdx)(
            self.handle,
            execution_graph,
            node_info,
            node_index.as_mut_ptr(),
        )
        .assume_init_on_success(node_index)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdInitializeGraphScratchMemoryAMDX.html>
    #[inline]
    pub unsafe fn cmd_initialize_graph_scratch_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        execution_graph: vk::Pipeline,
        scratch: vk::DeviceAddress,
        scratch_size: vk::DeviceSize,
    ) {
        (self.fp.cmd_initialize_graph_scratch_memory_amdx)(
            command_buffer,
            execution_graph,
            scratch,
            scratch_size,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDispatchGraphAMDX.html>
    #[inline]
    pub unsafe fn cmd_dispatch_graph(
        &self,
        command_buffer: vk::CommandBuffer,
        scratch: vk::DeviceAddress,
        scratch_size: vk::DeviceSize,
        count_info: &vk::DispatchGraphCountInfoAMDX,
    ) {
        (self.fp.cmd_dispatch_graph_amdx)(command_buffer, scratch, scratch_size, count_info)
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDispatchGraphIndirectAMDX.html>
    #[inline]
    pub unsafe fn cmd_dispatch_graph_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        scratch: vk::DeviceAddress,
        scratch_size: vk::DeviceSize,
        count_info: &vk::DispatchGraphCountInfoAMDX,
    ) {
        (self.fp.cmd_dispatch_graph_indirect_amdx)(
            command_buffer,
            scratch,
            scratch_size,
            count_info,
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDispatchGraphIndirectCountAMDX.html>
    #[inline]
    pub unsafe fn cmd_dispatch_graph_indirect_count(
        &self,
        command_buffer: vk::CommandBuffer,
        scratch: vk::DeviceAddress,
        scratch_size: vk::DeviceSize,
        count_info: vk::DeviceAddress,
    ) {
        (self.fp.cmd_dispatch_graph_indirect_count_amdx)(
            command_buffer,
            scratch,
            scratch_size,
            count_info,
        )
    }
}
