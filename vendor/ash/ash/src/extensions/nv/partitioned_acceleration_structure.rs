//! <https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_partitioned_acceleration_structure.html>

use crate::vk;

impl crate::nv::partitioned_acceleration_structure::Device {
    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPartitionedAccelerationStructuresBuildSizesNV.html>
    #[inline]
    pub unsafe fn get_partitioned_acceleration_structures_build_sizes(
        &self,
        info: &vk::PartitionedAccelerationStructureInstancesInputNV<'_>,
        size_info: &mut vk::AccelerationStructureBuildSizesInfoKHR<'_>,
    ) {
        (self
            .fp
            .get_partitioned_acceleration_structures_build_sizes_nv)(
            self.handle, info, size_info
        )
    }

    /// <https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBuildPartitionedAccelerationStructuresNV.html>
    #[inline]
    pub unsafe fn cmd_build_partitioned_acceleration_structures(
        &self,
        command_buffer: vk::CommandBuffer,
        build_info: &vk::BuildPartitionedAccelerationStructureInfoNV<'_>,
    ) {
        (self.fp.cmd_build_partitioned_acceleration_structures_nv)(command_buffer, build_info)
    }
}
