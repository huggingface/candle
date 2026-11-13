use crate::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureGeometry, AccelerationStructureGeometryAabbsData,
        AccelerationStructureGeometryData, AccelerationStructureGeometryInstancesData,
        AccelerationStructureGeometryTrianglesData, BuildAccelerationStructureMode,
        CopyAccelerationStructureInfo, CopyAccelerationStructureToMemoryInfo,
        CopyMemoryToAccelerationStructureInfo,
    },
    command_buffer::sys::RecordingCommandBuffer,
    device::{DeviceOwned, QueueFlags},
    query::{QueryPool, QueryType},
    DeviceAddress, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use smallvec::SmallVec;
use std::sync::Arc;

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn build_acceleration_structure(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo<'_>,
        build_range_infos: &[AccelerationStructureBuildRangeInfo],
    ) -> &mut Self {
        unsafe { self.try_build_acceleration_structure(info, build_range_infos) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_build_acceleration_structure(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo<'_>,
        build_range_infos: &[AccelerationStructureBuildRangeInfo],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_build_acceleration_structure(info, build_range_infos)?;

        Ok(unsafe { self.build_acceleration_structure_unchecked(info, build_range_infos) })
    }

    pub(crate) fn validate_build_acceleration_structure(
        &self,
        info: &AccelerationStructureBuildGeometryInfo<'_>,
        build_range_infos: &[AccelerationStructureBuildRangeInfo],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        let &AccelerationStructureBuildGeometryInfo {
            ty: _,
            flags: _,
            mode,
            src_acceleration_structure,
            dst_acceleration_structure,
            geometries,
            scratch_data,
            _ne,
        } = info;

        let Some(dst_acceleration_structure) = dst_acceleration_structure else {
            return Err(Box::new(ValidationError {
                context: "info.dst_acceleration_structure".into(),
                problem: "is `None`".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03800"],
                ..Default::default()
            }));
        };

        // VUID-vkCmdBuildAccelerationStructuresKHR-mode-04628
        // Ensured as long as `BuildAccelerationStructureMode` is exhaustive.

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03403
        // VUID-vkCmdBuildAccelerationStructuresKHR-None-03407
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03698
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03702
        // VUID-vkCmdBuildAccelerationStructuresKHR-scratchData-03704
        // Ensured as long as only one element is provided in `info`.

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03668
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03701
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03703
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03706
        // VUID-vkCmdBuildAccelerationStructuresKHR-scratchData-03705
        // Ensured by unsafe on `AccelerationStructure::new`.

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12260
        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12261
        // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
        // unsafe

        if mode == BuildAccelerationStructureMode::Update && src_acceleration_structure.is_none() {
            return Err(Box::new(ValidationError {
                problem: "`mode` is `BuildAccelerationStructureMode::Update`, but \
                    `dst_acceleration_structure` is `None`"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-04630"],
                ..Default::default()
            }));
        }

        let min_acceleration_structure_scratch_offset_alignment = self
            .device()
            .physical_device()
            .properties()
            .min_acceleration_structure_scratch_offset_alignment
            .unwrap();

        if !scratch_data.is_multiple_of(min_acceleration_structure_scratch_offset_alignment as u64)
        {
            return Err(Box::new(ValidationError {
                context: "info.scratch_data".into(),
                problem: "is not a multiple of the \
                    `min_acceleration_structure_scratch_offset_alignment` device property"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03710"],
                ..Default::default()
            }));
        }

        if geometries.len() != build_range_infos.len() {
            return Err(Box::new(ValidationError {
                problem: "`info.geometries` and `build_range_infos` do not have the same length"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-ppBuildRangeInfos-11543"],
                ..Default::default()
            }));
        }

        let mut total_primitive_count = 0;
        let max_primitive_count = self
            .device()
            .physical_device()
            .properties()
            .max_primitive_count
            .unwrap();
        let max_instance_count = self
            .device()
            .physical_device()
            .properties()
            .max_instance_count
            .unwrap();

        for (geometry_index, (geometry, build_range_info)) in
            geometries.iter().zip(build_range_infos).enumerate()
        {
            let AccelerationStructureGeometry {
                geometry,
                flags: _,
                _ne: _,
            } = geometry;

            let &AccelerationStructureBuildRangeInfo {
                primitive_count,
                primitive_offset,
                first_vertex: _,
                transform_offset,
            } = build_range_info;

            total_primitive_count += primitive_count as u64;

            match geometry {
                AccelerationStructureGeometryData::Triangles(triangles_data) => {
                    if total_primitive_count > max_primitive_count {
                        return Err(Box::new(ValidationError {
                            context: "info.geometries".into(),
                            problem: "the total number of triangles exceeds the \
                                `max_primitive_count` limit"
                                .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03795"],
                            ..Default::default()
                        }));
                    }

                    let &AccelerationStructureGeometryTrianglesData {
                        vertex_format,
                        vertex_data,
                        vertex_stride: _,
                        max_vertex: _,
                        index_type,
                        index_data,
                        transform_data,
                        _ne,
                    } = triangles_data;

                    let smallest_component_bits = vertex_format
                        .components()
                        .into_iter()
                        .filter(|&c| c != 0)
                        .min()
                        .unwrap() as u32;
                    let smallest_component_bytes = smallest_component_bits.div_ceil(8);

                    if !vertex_data.is_multiple_of(smallest_component_bytes as u64) {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].vertex_data", geometry_index)
                                .into(),
                            problem: "is not a multiple of the byte size of the smallest component
                                of `vertex_format`"
                                .into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03711"],
                            ..Default::default()
                        }));
                    }

                    if let Some(index_type) = index_type {
                        if !index_data.is_multiple_of(index_type.size()) {
                            return Err(Box::new(ValidationError {
                                context: format!("info.geometries[{}].index_data", geometry_index)
                                    .into(),
                                problem: "is not a multiple of the size of the index type".into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03712"],
                                ..Default::default()
                            }));
                        }

                        if !(primitive_offset as u64).is_multiple_of(index_type.size()) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`info.geometries[{0}].geometry` is \
                                    `AccelerationStructureGeometryData::Triangles`, and \
                                    `build_range_infos[{0}].primitive_offset` is not a multiple \
                                    of the size of `info.geometries[{0}].index_type.size()`",
                                    geometry_index,
                                )
                                .into(),
                                vuids: &["VUID-VkAccelerationStructureBuildRangeInfoKHR-\
                                    primitiveOffset-03656"],
                                ..Default::default()
                            }));
                        }
                    } else {
                        if !primitive_offset.is_multiple_of(smallest_component_bytes) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`info.geometries[{0}].geometry` is \
                                    `AccelerationStructureGeometryData::Triangles`, and
                                    `build_range_infos[{0}].primitive_offset` is not a multiple \
                                    of the byte size of the smallest component of \
                                    `info.geometries[{0}].vertex_format`",
                                    geometry_index,
                                )
                                .into(),
                                vuids: &["VUID-VkAccelerationStructureBuildRangeInfoKHR-\
                                    primitiveOffset-03657"],
                                ..Default::default()
                            }));
                        }
                    }

                    if transform_data != 0 {
                        if !transform_data.is_multiple_of(16) {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "info.geometries[{}].transform_data",
                                    geometry_index,
                                )
                                .into(),
                                problem: "is not a multiple of 16".into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03810"],
                                ..Default::default()
                            }));
                        }

                        if !transform_offset.is_multiple_of(16) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`info.geometries[{0}].geometry` is \
                                    `AccelerationStructureGeometryData::Triangles`, and \
                                    `build_range_infos[{0}].transform_offset` is not a multiple \
                                    of 16",
                                    geometry_index,
                                )
                                .into(),
                                vuids: &["VUID-VkAccelerationStructureBuildRangeInfoKHR-\
                                    transformOffset-03658"],
                                ..Default::default()
                            }));
                        }
                    }
                }
                AccelerationStructureGeometryData::Aabbs(aabbs_data) => {
                    if total_primitive_count > max_primitive_count {
                        return Err(Box::new(ValidationError {
                            problem: "the total number of AABBs exceeds the `max_primitive_count` \
                                limit"
                                .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03794"],
                            ..Default::default()
                        }));
                    }

                    let &AccelerationStructureGeometryAabbsData {
                        data,
                        stride: _,
                        _ne,
                    } = aabbs_data;

                    if !data.is_multiple_of(8) {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].data", geometry_index).into(),
                            problem: "is not a multiple of 8".into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03714"],
                            ..Default::default()
                        }));
                    }

                    if !primitive_offset.is_multiple_of(8) {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`info.geometries[{0}].geometry` is \
                                `AccelerationStructureGeometryData::Aabbs`, and \
                                `build_range_infos[{0}].primitive_offset` is not a multiple of 8",
                                geometry_index,
                            )
                            .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildRangeInfoKHR-\
                                primitiveOffset-03659"],
                            ..Default::default()
                        }));
                    }
                }
                AccelerationStructureGeometryData::Instances(instances_data) => {
                    if total_primitive_count > max_instance_count {
                        return Err(Box::new(ValidationError {
                            context: "build_range_infos[0].primitive_count".into(),
                            problem: "exceeds the the `max_instance_count` limit".into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03801"],
                            ..Default::default()
                        }));
                    }

                    let &AccelerationStructureGeometryInstancesData {
                        array_of_pointers,
                        data,
                        _ne,
                    } = instances_data;

                    if !primitive_offset.is_multiple_of(16) {
                        return Err(Box::new(ValidationError {
                            problem: "`info.geometries[0].geometry` is \
                                `AccelerationStructureGeometryData::Instances`, and \
                                `build_range_infos[0].primitive_offset` is not a multiple of 16"
                                .into(),
                            vuids: &[
                                "VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-\
                                03660",
                            ],
                            ..Default::default()
                        }));
                    }

                    if array_of_pointers {
                        if !data.is_multiple_of(8) {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries[0]".into(),
                                problem: "`geometry` is \
                                    `AccelerationStructureGeometryData::Instances`, \
                                    `geometry.array_of_pointers` is `true`, and `geometry.data` \
                                    is not a multiple of 8"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03716"],
                                ..Default::default()
                            }));
                        }

                        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03717
                        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03813
                        // unsafe
                    } else {
                        if !data.is_multiple_of(16) {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries[0]".into(),
                                problem: "`geometry` is \
                                    `AccelerationStructureGeometryData::Instances`, and \
                                    `geometry.array_of_pointers` is `false`, and `geometry.data` \
                                    is not a multiple of 16"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03715"],
                                ..Default::default()
                            }));
                        }
                    }

                    // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-06707
                    // unsafe
                }
            }
        }

        let build_size_info = unsafe {
            self.device().acceleration_structure_build_sizes_unchecked(
                AccelerationStructureBuildType::Device,
                info,
                &build_range_infos
                    .iter()
                    .map(|info| info.primitive_count)
                    .collect::<SmallVec<[_; 8]>>(),
            )
        };

        if dst_acceleration_structure.size() < build_size_info.acceleration_structure_size {
            return Err(Box::new(ValidationError {
                context: "info.dst_acceleration_structure".into(),
                problem: "size is too small to hold the resulting acceleration structure data"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03675"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12258
        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-12259
        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03667
        // unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn build_acceleration_structure_unchecked(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo<'_>,
        build_range_infos: &[AccelerationStructureBuildRangeInfo],
    ) -> &mut Self {
        let info_fields1_vk = info.to_vk_fields1();
        let info_vk = info.to_vk(&info_fields1_vk);

        let build_range_info_elements_vk: SmallVec<[_; 8]> = build_range_infos
            .iter()
            .map(AccelerationStructureBuildRangeInfo::to_vk)
            .collect();
        let build_range_info_pointers_vk: SmallVec<[_; 8]> = build_range_info_elements_vk
            .iter()
            .map(|p| -> *const _ { p })
            .collect();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_build_acceleration_structures_khr)(
                self.handle(),
                1,
                &info_vk,
                build_range_info_pointers_vk.as_ptr(),
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn build_acceleration_structure_indirect(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo<'_>,
        indirect_device_address: DeviceAddress,
        indirect_stride: u32,
        max_primitive_counts: &[u32],
    ) -> &mut Self {
        unsafe {
            self.try_build_acceleration_structure_indirect(
                info,
                indirect_device_address,
                indirect_stride,
                max_primitive_counts,
            )
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_build_acceleration_structure_indirect(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo<'_>,
        indirect_device_address: DeviceAddress,
        indirect_stride: u32,
        max_primitive_counts: &[u32],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_build_acceleration_structure_indirect(
            info,
            indirect_device_address,
            indirect_stride,
            max_primitive_counts,
        )?;

        Ok(unsafe {
            self.build_acceleration_structure_indirect_unchecked(
                info,
                indirect_device_address,
                indirect_stride,
                max_primitive_counts,
            )
        })
    }

    pub(crate) fn validate_build_acceleration_structure_indirect(
        &self,
        info: &AccelerationStructureBuildGeometryInfo<'_>,
        indirect_device_address: DeviceAddress,
        indirect_stride: u32,
        max_primitive_counts: &[u32],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .enabled_features()
            .acceleration_structure_indirect_build
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "acceleration_structure_indirect_build",
                )])]),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-\
                    accelerationStructureIndirectBuild-03650"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        let &AccelerationStructureBuildGeometryInfo {
            ty: _,
            flags: _,
            mode,
            src_acceleration_structure,
            dst_acceleration_structure,
            geometries,
            scratch_data,
            _ne,
        } = info;

        let Some(dst_acceleration_structure) = dst_acceleration_structure else {
            return Err(Box::new(ValidationError {
                context: "info.dst_acceleration_structure".into(),
                problem: "is `None`".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-\
                    dstAccelerationStructure-03800"],
                ..Default::default()
            }));
        };

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-mode-04628
        // Ensured as long as `BuildAccelerationStructureMode` is exhaustive.

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03403
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-None-03407
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03698
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03702
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-scratchData-03704
        // Ensured as long as only one element is provided in `info`.

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03668
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03701
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03703
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03706
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-scratchData-03705
        // Ensured by unsafe on `AccelerationStructure::new`.

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-12260
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-12261
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-geometry-03673
        // unsafe

        if mode == BuildAccelerationStructureMode::Update && src_acceleration_structure.is_none() {
            return Err(Box::new(ValidationError {
                problem: "`mode` is `BuildAccelerationStructureMode::Update`, but \
                    `dst_acceleration_structure` is `None`"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-04630"],
                ..Default::default()
            }));
        }

        let min_acceleration_structure_scratch_offset_alignment = self
            .device()
            .physical_device()
            .properties()
            .min_acceleration_structure_scratch_offset_alignment
            .unwrap();

        if !scratch_data.is_multiple_of(min_acceleration_structure_scratch_offset_alignment as u64)
        {
            return Err(Box::new(ValidationError {
                context: "info.scratch_data".into(),
                problem: "the device address of the buffer is not a multiple of the \
                    `min_acceleration_structure_scratch_offset_alignment` device property"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03710"],
                ..Default::default()
            }));
        }

        if geometries.len() != max_primitive_counts.len() {
            return Err(Box::new(ValidationError {
                problem: "`info.geometries` and `max_primitive_counts` do not have the same length"
                    .into(),
                vuids: &[
                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-ppMaxPrimitiveCounts-\
                    parameter",
                ],
                ..Default::default()
            }));
        }

        for (geometry_index, geometry) in geometries.iter().enumerate() {
            let AccelerationStructureGeometry {
                geometry,
                flags: _,
                _ne: _,
            } = geometry;

            match geometry {
                AccelerationStructureGeometryData::Triangles(triangles_data) => {
                    // VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03795
                    // unsafe

                    let &AccelerationStructureGeometryTrianglesData {
                        vertex_format,
                        vertex_data,
                        vertex_stride: _,
                        max_vertex: _,
                        index_type,
                        index_data,
                        transform_data,
                        _ne,
                    } = triangles_data;

                    let smallest_component_bits = vertex_format
                        .components()
                        .into_iter()
                        .filter(|&c| c != 0)
                        .min()
                        .unwrap() as u32;
                    let smallest_component_bytes = smallest_component_bits.div_ceil(8);

                    if !vertex_data.is_multiple_of(smallest_component_bytes as u64) {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].vertex_data", geometry_index)
                                .into(),
                            problem: "is not a multiple of the byte size of the smallest component
                                of `vertex_format`"
                                .into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-\
                                03711"],
                            ..Default::default()
                        }));
                    }

                    if let Some(index_type) = index_type {
                        if !index_data.is_multiple_of(index_type.size()) {
                            return Err(Box::new(ValidationError {
                                context: format!("info.geometries[{}].index_data", geometry_index)
                                    .into(),
                                problem: "is not a multiple of the size of the index type".into(),
                                vuids: &[
                                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-\
                                    03712",
                                ],
                                ..Default::default()
                            }));
                        }

                        // VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03656
                        // unsafe
                    } else {
                        // VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03657
                        // unsafe
                    }

                    if transform_data != 0 {
                        if !transform_data.is_multiple_of(16) {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "info.geometries[{}].transform_data",
                                    geometry_index,
                                )
                                .into(),
                                problem: "is not a multiple of 16".into(),
                                vuids: &[
                                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-\
                                    03810",
                                ],
                                ..Default::default()
                            }));
                        }

                        // VUID-VkAccelerationStructureBuildRangeInfoKHR-transformOffset-03658
                        // unsafe
                    }
                }
                AccelerationStructureGeometryData::Aabbs(aabbs_data) => {
                    // VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03794
                    // unsafe

                    let &AccelerationStructureGeometryAabbsData {
                        data,
                        stride: _,
                        _ne,
                    } = aabbs_data;

                    if !data.is_multiple_of(8) {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].data", geometry_index).into(),
                            problem: "is not a multiple of 8".into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-\
                                03714"],
                            ..Default::default()
                        }));
                    }

                    // VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03659
                    // unsafe
                }
                AccelerationStructureGeometryData::Instances(instances_data) => {
                    // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03801
                    // unsafe

                    let &AccelerationStructureGeometryInstancesData {
                        array_of_pointers,
                        data,
                        _ne,
                    } = instances_data;

                    // VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03660
                    // unsafe

                    if array_of_pointers {
                        if !data.is_multiple_of(8) {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries[0]".into(),
                                problem: "`geometry` is \
                                    `AccelerationStructureGeometryData::Instances`, \
                                    `geometry.array_of_pointers` is `true`, and `geometry.data` \
                                    is not a multiple of 8"
                                    .into(),
                                vuids: &[
                                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-\
                                    03716",
                                ],
                                ..Default::default()
                            }));
                        }

                        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03717
                        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03813
                        // unsafe
                    } else {
                        if !data.is_multiple_of(16) {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries[0]".into(),
                                problem: "`geometry` is \
                                    `AccelerationStructureGeometryData::Instances`, and \
                                    `geometry.array_of_pointers` is `false`, and `geometry.data` \
                                    is not a multiple of 16"
                                    .into(),
                                vuids: &[
                                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-\
                                    03715",
                                ],
                                ..Default::default()
                            }));
                        }
                    }

                    // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-06707
                    // unsafe
                }
            }
        }

        let build_size_info = unsafe {
            self.device().acceleration_structure_build_sizes_unchecked(
                AccelerationStructureBuildType::Device,
                info,
                max_primitive_counts,
            )
        };

        if dst_acceleration_structure.size() < build_size_info.acceleration_structure_size {
            return Err(Box::new(ValidationError {
                context: "info.dst_acceleration_structure".into(),
                problem: "size is too small to hold the resulting acceleration structure data"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03652"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-12258
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-12259
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03667
        // unsafe

        if !indirect_device_address.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "indirect_device_address".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-\
                    pIndirectDeviceAddresses-03648"],
                ..Default::default()
            }));
        }

        if !indirect_stride.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "indirect_stride".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectStrides-03787"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03651
        // unsafe

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-ppMaxPrimitiveCounts-03653
        // unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn build_acceleration_structure_indirect_unchecked(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo<'_>,
        indirect_device_address: DeviceAddress,
        indirect_stride: u32,
        max_primitive_counts: &[u32],
    ) -> &mut Self {
        let info_fields1_vk = info.to_vk_fields1();
        let info_vk = info.to_vk(&info_fields1_vk);

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_build_acceleration_structures_indirect_khr)(
                self.handle(),
                1,
                &info_vk,
                &indirect_device_address,
                &indirect_stride,
                &max_primitive_counts.as_ptr(),
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn copy_acceleration_structure(
        &mut self,
        info: &CopyAccelerationStructureInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_copy_acceleration_structure(info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_copy_acceleration_structure(
        &mut self,
        info: &CopyAccelerationStructureInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_acceleration_structure(info)?;

        Ok(unsafe { self.copy_acceleration_structure_unchecked(info) })
    }

    pub(crate) fn validate_copy_acceleration_structure(
        &self,
        info: &CopyAccelerationStructureInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdCopyAccelerationStructureKHR-pInfo-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_acceleration_structure_unchecked(
        &mut self,
        info: &CopyAccelerationStructureInfo<'_>,
    ) -> &mut Self {
        let info_vk = info.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_copy_acceleration_structure_khr)(self.handle(), &info_vk)
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn copy_acceleration_structure_to_memory(
        &mut self,
        info: &CopyAccelerationStructureToMemoryInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_copy_acceleration_structure_to_memory(info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_copy_acceleration_structure_to_memory(
        &mut self,
        info: &CopyAccelerationStructureToMemoryInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_acceleration_structure_to_memory(info)?;

        Ok(unsafe { self.copy_acceleration_structure_to_memory_unchecked(info) })
    }

    pub(crate) fn validate_copy_acceleration_structure_to_memory(
        &self,
        info: &CopyAccelerationStructureToMemoryInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureToMemoryKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !info.dst.is_multiple_of(256) {
            return Err(Box::new(ValidationError {
                context: "info.dst".into(),
                problem: "is not a multiple of 256".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureToMemoryKHR-pInfo-03740"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdCopyAccelerationStructureToMemoryKHR-pInfo-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_acceleration_structure_to_memory_unchecked(
        &mut self,
        info: &CopyAccelerationStructureToMemoryInfo<'_>,
    ) -> &mut Self {
        let info_vk = info.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_copy_acceleration_structure_to_memory_khr)(self.handle(), &info_vk)
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn copy_memory_to_acceleration_structure(
        &mut self,
        info: &CopyMemoryToAccelerationStructureInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_copy_memory_to_acceleration_structure(info) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_copy_memory_to_acceleration_structure(
        &mut self,
        info: &CopyMemoryToAccelerationStructureInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_memory_to_acceleration_structure(info)?;

        Ok(unsafe { self.copy_memory_to_acceleration_structure_unchecked(info) })
    }

    pub(crate) fn validate_copy_memory_to_acceleration_structure(
        &self,
        info: &CopyMemoryToAccelerationStructureInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdCopyMemoryToAccelerationStructureKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if !info.src.is_multiple_of(256) {
            return Err(Box::new(ValidationError {
                context: "info.src".into(),
                problem: "the device address of the buffer is not a multiple of 256".into(),
                vuids: &["VUID-vkCmdCopyMemoryToAccelerationStructureKHR-pInfo-03743"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdCopyMemoryToAccelerationStructureKHR-pInfo-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_memory_to_acceleration_structure_unchecked(
        &mut self,
        info: &CopyMemoryToAccelerationStructureInfo<'_>,
    ) -> &mut Self {
        let info_vk = info.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_copy_memory_to_acceleration_structure_khr)(self.handle(), &info_vk)
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn write_acceleration_structures_properties(
        &mut self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> &mut Self {
        unsafe {
            self.try_write_acceleration_structures_properties(
                acceleration_structures,
                query_pool,
                first_query,
            )
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_write_acceleration_structures_properties(
        &mut self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_write_acceleration_structures_properties(
            acceleration_structures,
            query_pool,
            first_query,
        )?;

        Ok(unsafe {
            self.write_acceleration_structures_properties_unchecked(
                acceleration_structures,
                query_pool,
                first_query,
            )
        })
    }

    pub(crate) fn validate_write_acceleration_structures_properties(
        &self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &[
                    "VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-commandBuffer-cmdpool",
                ],
                ..Default::default()
            }));
        }

        for acs in acceleration_structures {
            // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-commonparent
            assert_eq!(self.device(), acs.device());

            // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-pAccelerationStructures-04964
            // unsafe

            // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-accelerationStructures-03431
            // unsafe
        }

        // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-commonparent
        assert_eq!(self.device(), query_pool.device());

        // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-queryPool-02494
        // unsafe

        if first_query as usize + acceleration_structures.len() > query_pool.query_count() as usize
        {
            return Err(Box::new(ValidationError {
                problem: "`first_query` + `acceleration_structures.len()` is greater than \
                    `query_pool.query_count`"
                    .into(),
                vuids: &["VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-query-04880"],
                ..Default::default()
            }));
        }

        if !matches!(
            query_pool.query_type(),
            QueryType::AccelerationStructureSize
                | QueryType::AccelerationStructureSerializationBottomLevelPointers
                | QueryType::AccelerationStructureCompactedSize
                | QueryType::AccelerationStructureSerializationSize,
        ) {
            return Err(Box::new(ValidationError {
                context: "query_pool.query_type()".into(),
                problem: "is not an acceleration structure query".into(),
                vuids: &["VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-queryType-06742"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_acceleration_structures_properties_unchecked(
        &mut self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> &mut Self {
        if acceleration_structures.is_empty() {
            return self;
        }

        let acceleration_structures_vk: SmallVec<[_; 4]> = acceleration_structures
            .iter()
            .map(VulkanObject::handle)
            .collect();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_write_acceleration_structures_properties_khr)(
                self.handle(),
                acceleration_structures_vk.len() as u32,
                acceleration_structures_vk.as_ptr(),
                query_pool.query_type().into(),
                query_pool.handle(),
                first_query,
            )
        };

        self
    }
}
