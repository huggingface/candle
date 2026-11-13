use crate::{
    buffer::{Buffer, BufferContents, BufferUsage, IndexType},
    command_buffer::sys::RecordingCommandBuffer,
    descriptor_set::{
        layout::{DescriptorBindingFlags, DescriptorSetLayoutCreateFlags, DescriptorType},
        sys::RawDescriptorSet,
        WriteDescriptorSet,
    },
    device::{DeviceOwned, QueueFlags},
    memory::is_aligned,
    pipeline::{Pipeline, PipelineBindPoint, PipelineLayout},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use ash::vk;
use smallvec::SmallVec;
use std::{cmp::min, ffi::c_void, ptr};

impl RecordingCommandBuffer {
    #[inline]
    #[track_caller]
    pub unsafe fn bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&RawDescriptorSet],
        dynamic_offsets: &[u32],
    ) -> &mut Self {
        unsafe {
            self.try_bind_descriptor_sets(
                pipeline_bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&RawDescriptorSet],
        dynamic_offsets: &[u32],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_descriptor_sets(
            pipeline_bind_point,
            layout,
            first_set,
            descriptor_sets,
            dynamic_offsets,
        )?;

        Ok(unsafe {
            self.bind_descriptor_sets_unchecked(
                pipeline_bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        })
    }

    pub(crate) fn validate_bind_descriptor_sets(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&RawDescriptorSet],
        dynamic_offsets: &[u32],
    ) -> Result<(), Box<ValidationError>> {
        self.validate_bind_descriptor_sets_inner(
            pipeline_bind_point,
            layout,
            first_set,
            descriptor_sets.len(),
        )?;

        let properties = self.device().physical_device().properties();
        let mut dynamic_offsets_remaining = dynamic_offsets;
        let mut required_dynamic_offset_count = 0;

        for (descriptor_sets_index, set) in descriptor_sets.iter().enumerate() {
            let set_num = first_set as usize + descriptor_sets_index;

            // VUID-vkCmdBindDescriptorSets-commonparent
            assert_eq!(self.device(), set.device());

            let set_layout = set.layout();
            let pipeline_set_layout = &layout.set_layouts()[set_num];

            if !pipeline_set_layout.is_compatible_with(set_layout) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`descriptor_sets[{0}]` (for set number {1}) is not compatible with \
                        `pipeline_layout.set_layouts()[{1}]`",
                        descriptor_sets_index, set_num
                    )
                    .into(),
                    vuids: &["VUID-vkCmdBindDescriptorSets-pDescriptorSets-00358"],
                    ..Default::default()
                }));
            }

            for binding in set_layout.bindings() {
                let binding_num = binding.binding;

                let required_alignment = match binding.descriptor_type {
                    DescriptorType::UniformBufferDynamic => {
                        properties.min_uniform_buffer_offset_alignment
                    }
                    DescriptorType::StorageBufferDynamic => {
                        properties.min_storage_buffer_offset_alignment
                    }
                    _ => continue,
                };

                let count = if binding
                    .binding_flags
                    .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                {
                    set.variable_descriptor_count()
                } else {
                    binding.descriptor_count
                } as usize;

                required_dynamic_offset_count += count;

                if !dynamic_offsets_remaining.is_empty() {
                    let split_index = min(count, dynamic_offsets_remaining.len());
                    let dynamic_offsets = &dynamic_offsets_remaining[..split_index];
                    dynamic_offsets_remaining = &dynamic_offsets_remaining[split_index..];

                    for (index, &offset) in dynamic_offsets.iter().enumerate() {
                        if !is_aligned(offset as DeviceSize, required_alignment) {
                            match binding.descriptor_type {
                                DescriptorType::UniformBufferDynamic => {
                                    return Err(Box::new(ValidationError {
                                        problem: format!(
                                            "the descriptor type of `descriptor_sets[{}]` \
                                            (for set number {}) is \
                                            `DescriptorType::UniformBufferDynamic`, but the \
                                            dynamic offset provided for binding {} index {} is \
                                            not aligned to the \
                                            `min_uniform_buffer_offset_alignment` device property",
                                            descriptor_sets_index, set_num, binding_num, index,
                                        )
                                        .into(),
                                        vuids: &[
                                            "VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01971",
                                        ],
                                        ..Default::default()
                                    }));
                                }
                                DescriptorType::StorageBufferDynamic => {
                                    return Err(Box::new(ValidationError {
                                        problem: format!(
                                            "the descriptor type of `descriptor_sets[{}]` \
                                            (for set number {}) is \
                                            `DescriptorType::StorageBufferDynamic`, but the \
                                            dynamic offset provided for binding {} index {} is \
                                            not aligned to the \
                                            `min_storage_buffer_offset_alignment` device property",
                                            descriptor_sets_index, set_num, binding_num, index,
                                        )
                                        .into(),
                                        vuids: &[
                                            "VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01972",
                                        ],
                                        ..Default::default()
                                    }));
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
        }

        if dynamic_offsets.len() != required_dynamic_offset_count {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "the number of dynamic offsets provided does not equal the number required \
                    ({})",
                    required_dynamic_offset_count,
                )
                .into(),
                vuids: &["VUID-vkCmdBindDescriptorSets-dynamicOffsetCount-00359"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn validate_bind_descriptor_sets_inner(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: usize,
    ) -> Result<(), Box<ValidationError>> {
        pipeline_bind_point
            .validate_device(self.device())
            .map_err(|err| {
                err.add_context("pipeline_bind_point")
                    .set_vuids(&["VUID-vkCmdBindDescriptorSets-pipelineBindPoint-parameter"])
            })?;

        let queue_family_properties = self.queue_family_properties();

        match pipeline_bind_point {
            PipelineBindPoint::Compute => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "pipeline_bind_point".into(),
                        problem: "is `PipelineBindPoint::Compute`, but \
                            the queue family of the command buffer does not support \
                            compute operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdBindDescriptorSets-pipelineBindPoint-00361",
                            "VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
            PipelineBindPoint::Graphics => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        context: "pipeline_bind_point".into(),
                        problem: "is `PipelineBindPoint::Graphics`, but \
                            the queue family of the command buffer does not support \
                            graphics operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdBindDescriptorSets-pipelineBindPoint-00361",
                            "VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
            PipelineBindPoint::RayTracing => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "pipeline_bind_point".into(),
                        problem: "is `PipelineBindPoint::RayTracing`, but \
                            the queue family of the command buffer does not support \
                            compute operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdBindDescriptorSets-pipelineBindPoint-02391",
                            "VUID-vkCmdBindDescriptorSets-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if (first_set as usize)
            .checked_add(descriptor_sets)
            .is_none_or(|end| end > layout.set_layouts().len())
        {
            return Err(Box::new(ValidationError {
                problem: "`first_set + descriptor_sets.len()` is greater than \
                    `pipeline_layout.set_layouts().len()`"
                    .into(),
                vuids: &["VUID-vkCmdBindDescriptorSets-firstSet-00360"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_descriptor_sets_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&RawDescriptorSet],
        dynamic_offsets: &[u32],
    ) -> &mut Self {
        if descriptor_sets.is_empty() {
            return self;
        }

        let descriptor_sets_vk: SmallVec<[_; 12]> =
            descriptor_sets.iter().map(|x| x.handle()).collect();

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_bind_descriptor_sets)(
                self.handle(),
                pipeline_bind_point.into(),
                layout.handle(),
                first_set,
                descriptor_sets_vk.len() as u32,
                descriptor_sets_vk.as_ptr(),
                dynamic_offsets.len() as u32,
                dynamic_offsets.as_ptr(),
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn bind_index_buffer(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        size: Option<DeviceSize>,
        index_type: IndexType,
    ) -> &mut Self {
        unsafe { self.try_bind_index_buffer(buffer, offset, size, index_type) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_bind_index_buffer(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        size: Option<DeviceSize>,
        index_type: IndexType,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_index_buffer(buffer, offset, size, index_type)?;

        Ok(unsafe { self.bind_index_buffer_unchecked(buffer, offset, size, index_type) })
    }

    pub(crate) fn validate_bind_index_buffer(
        &self,
        buffer: &Buffer,
        offset: DeviceSize,
        size: Option<DeviceSize>,
        index_type: IndexType,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdBindIndexBuffer2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBindIndexBuffer2-commonparent
        assert_eq!(self.device(), buffer.device());

        if offset >= buffer.size() {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not less than `buffer.size()`".into(),
                vuids: &["VUID-vkCmdBindIndexBuffer2-offset-08782"],
                ..Default::default()
            }));
        }

        if !buffer.usage().intersects(BufferUsage::INDEX_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDEX_BUFFER`".into(),
                vuids: &["VUID-vkCmdBindIndexBuffer2-buffer-08784"],
                ..Default::default()
            }));
        }

        if index_type == IndexType::U8 && !self.device().enabled_features().index_type_uint8 {
            return Err(Box::new(ValidationError {
                context: "index_type".into(),
                problem: "is `IndexType::U8`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "index_type_uint8",
                )])]),
                vuids: &["VUID-vkCmdBindIndexBuffer2-indexType-08787"],
            }));
        }

        if let Some(size) = size {
            if size % index_type.size() != 0 {
                return Err(Box::new(ValidationError {
                    context: "size".into(),
                    problem: "is not a multiple of `index_type.size()`".into(),
                    vuids: &["VUID-vkCmdBindIndexBuffer2-size-08767"],
                    ..Default::default()
                }));
            }

            if size > buffer.size() - offset {
                return Err(Box::new(ValidationError {
                    problem: "`offset + size` is greater than `buffer.size()`".into(),
                    vuids: &["VUID-vkCmdBindIndexBuffer2-size-08768"],
                    ..Default::default()
                }));
            }
        }

        // TODO:
        // VUID-vkCmdBindIndexBuffer2-offset-08783

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_index_buffer_unchecked(
        &mut self,
        buffer: &Buffer,
        offset: DeviceSize,
        size: Option<DeviceSize>,
        index_type: IndexType,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_extensions().khr_maintenance5 {
            unsafe {
                (fns.khr_maintenance5.cmd_bind_index_buffer2_khr)(
                    self.handle(),
                    buffer.handle(),
                    offset,
                    size.unwrap_or(vk::WHOLE_SIZE),
                    index_type.into(),
                )
            };
        } else {
            unsafe {
                (fns.v1_0.cmd_bind_index_buffer)(
                    self.handle(),
                    buffer.handle(),
                    offset,
                    index_type.into(),
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn bind_pipeline<'a>(&mut self, pipeline: impl Into<Pipeline<'a>>) -> &mut Self {
        unsafe { self.bind_pipeline_inner(pipeline.into()) }
    }

    #[inline]
    #[track_caller]
    unsafe fn bind_pipeline_inner(&mut self, pipeline: Pipeline<'_>) -> &mut Self {
        unsafe { self.try_bind_pipeline_inner(pipeline) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_bind_pipeline<'a>(
        &mut self,
        pipeline: impl Into<Pipeline<'a>>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        unsafe { self.try_bind_pipeline_inner(pipeline.into()) }
    }

    #[inline]
    unsafe fn try_bind_pipeline_inner<'a>(
        &mut self,
        pipeline: Pipeline<'a>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline(pipeline)?;

        Ok(unsafe { self.bind_pipeline_unchecked_inner(pipeline) })
    }

    pub(crate) fn validate_bind_pipeline(
        &self,
        pipeline: Pipeline<'_>,
    ) -> Result<(), Box<ValidationError>> {
        match pipeline {
            Pipeline::Compute(_) => {
                if !self
                    .queue_family_properties()
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        problem: "the queue family of the command buffer does not support \
                            compute operations"
                            .into(),
                        vuids: &["VUID-vkCmdBindPipeline-pipelineBindPoint-00777"],
                        ..Default::default()
                    }));
                }
            }
            Pipeline::Graphics(_) => {
                if !self
                    .queue_family_properties()
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        problem: "the queue family of the command buffer does not support \
                            graphics operations"
                            .into(),
                        vuids: &["VUID-vkCmdBindPipeline-pipelineBindPoint-00778"],
                        ..Default::default()
                    }));
                }
            }
            Pipeline::RayTracing(_) => {
                if !self
                    .queue_family_properties()
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        problem: "the queue family of the command buffer does not support \
                            compute operations"
                            .into(),
                        vuids: &["VUID-vkCmdBindPipeline-pipelineBindPoint-02391"],
                        ..Default::default()
                    }));
                }

                if self.is_protected() {
                    return Err(Box::new(ValidationError {
                        context: "pipeline".into(),
                        problem: "is a ray tracing pipeline, but the command buffer is a \
                            protected command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdBindPipeline-pipelineBindPoint-06721"],
                        ..Default::default()
                    }));
                }
            }
        }

        // VUID-vkCmdBindPipeline-commonparent
        assert_eq!(self.device(), pipeline.device());

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_unchecked<'a>(
        &mut self,
        pipeline: impl Into<Pipeline<'a>>,
    ) -> &mut Self {
        unsafe { self.bind_pipeline_unchecked_inner(pipeline.into()) }
    }

    unsafe fn bind_pipeline_unchecked_inner(&mut self, pipeline: Pipeline<'_>) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_bind_pipeline)(
                self.handle(),
                pipeline.bind_point().into(),
                pipeline.handle(),
            )
        };

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[&Buffer],
        offsets: &[DeviceSize],
        sizes: &[DeviceSize],
        strides: &[DeviceSize],
    ) -> &mut Self {
        unsafe { self.try_bind_vertex_buffers(first_binding, buffers, offsets, sizes, strides) }
            .unwrap()
    }

    #[inline]
    pub unsafe fn try_bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[&Buffer],
        offsets: &[DeviceSize],
        sizes: &[DeviceSize],
        strides: &[DeviceSize],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_vertex_buffers(first_binding, buffers, offsets, sizes, strides)?;

        Ok(unsafe {
            self.bind_vertex_buffers_unchecked(first_binding, buffers, offsets, sizes, strides)
        })
    }

    pub(crate) fn validate_bind_vertex_buffers(
        &self,
        first_binding: u32,
        buffers: &[&Buffer],
        offsets: &[DeviceSize],
        sizes: &[DeviceSize],
        strides: &[DeviceSize],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics operations"
                    .into(),
                vuids: &["VUID-vkCmdBindVertexBuffers-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if first_binding >= properties.max_vertex_input_bindings {
            return Err(Box::new(ValidationError {
                problem: "`first_binding` is not less than the `max_vertex_input_bindings` limit"
                    .into(),
                vuids: &["VUID-vkCmdBindVertexBuffers2-firstBinding-03355"],
                ..Default::default()
            }));
        }

        if buffers.len() > (properties.max_vertex_input_bindings - first_binding) as usize {
            return Err(Box::new(ValidationError {
                problem: "`first_binding + buffers.len()` is greater than the \
                    `max_vertex_input_bindings` limit"
                    .into(),
                vuids: &["VUID-vkCmdBindVertexBuffers2-firstBinding-03356"],
                ..Default::default()
            }));
        }

        assert_eq!(offsets.len(), buffers.len());

        for (buffers_index, (&buffer, &offset)) in buffers.iter().zip(offsets).enumerate() {
            // VUID-vkCmdBindVertexBuffers2-commonparent
            assert_eq!(self.device(), buffer.device());

            if offset >= buffer.size() {
                return Err(Box::new(ValidationError {
                    context: format!("offsets[{}]", buffers_index).into(),
                    problem: format!("is not less than `buffers[{}].size()`", buffers_index).into(),
                    vuids: &["VUID-vkCmdBindVertexBuffers2-pOffsets-03357"],
                    ..Default::default()
                }));
            }

            if !buffer.usage().intersects(BufferUsage::VERTEX_BUFFER) {
                return Err(Box::new(ValidationError {
                    context: format!("buffers[{}].usage()", buffers_index).into(),
                    problem: "does not contain `BufferUsage::VERTEX_BUFFER`".into(),
                    vuids: &["VUID-vkCmdBindVertexBuffers2-pBuffers-03359"],
                    ..Default::default()
                }));
            }
        }

        if !sizes.is_empty() {
            assert_eq!(sizes.len(), buffers.len());

            for (buffers_index, ((&buffer, &offset), &size)) in
                buffers.iter().zip(offsets).zip(sizes).enumerate()
            {
                if size > buffer.size() - offset {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`offsets[{0}] + sizes[{0}]` is greater than `buffers[{0}].size()`",
                            buffers_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdBindVertexBuffers2-pBuffers-03359"],
                        ..Default::default()
                    }));
                }
            }
        }

        if !strides.is_empty() {
            assert_eq!(strides.len(), buffers.len());

            for (buffers_index, &stride) in strides.iter().enumerate() {
                if stride > properties.max_vertex_input_binding_stride as DeviceSize {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`strides[{}]` is greater than the `max_vertex_input_binding_stride` \
                            limit",
                            buffers_index,
                        )
                        .into(),
                        vuids: &["VUID-vkCmdBindVertexBuffers2-pStrides-03362"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_vertex_buffers_unchecked(
        &mut self,
        first_binding: u32,
        buffers: &[&Buffer],
        offsets: &[DeviceSize],
        sizes: &[DeviceSize],
        strides: &[DeviceSize],
    ) -> &mut Self {
        if buffers.is_empty() {
            return self;
        }

        let device = self.device();

        if device.api_version() >= Version::V1_3
            || device.enabled_extensions().ext_extended_dynamic_state
            || device.enabled_extensions().ext_shader_object
        {
            let buffers_vk = buffers
                .iter()
                .map(VulkanObject::handle)
                .collect::<SmallVec<[_; 2]>>();

            let fns = self.device().fns();
            let cmd_bind_vertex_buffers2 = if device.api_version() >= Version::V1_3 {
                fns.v1_3.cmd_bind_vertex_buffers2
            } else if device.enabled_extensions().ext_extended_dynamic_state {
                fns.ext_extended_dynamic_state.cmd_bind_vertex_buffers2_ext
            } else {
                fns.ext_shader_object.cmd_bind_vertex_buffers2_ext
            };

            unsafe {
                cmd_bind_vertex_buffers2(
                    self.handle(),
                    first_binding,
                    buffers_vk.len() as u32,
                    buffers_vk.as_ptr(),
                    offsets.as_ptr(),
                    if sizes.is_empty() {
                        ptr::null()
                    } else {
                        sizes.as_ptr()
                    },
                    if strides.is_empty() {
                        ptr::null()
                    } else {
                        strides.as_ptr()
                    },
                )
            }
        } else {
            let buffers_vk = buffers
                .iter()
                .map(VulkanObject::handle)
                .collect::<SmallVec<[_; 2]>>();

            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.cmd_bind_vertex_buffers)(
                    self.handle(),
                    first_binding,
                    buffers_vk.len() as u32,
                    buffers_vk.as_ptr(),
                    offsets.as_ptr(),
                )
            };
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn push_constants(
        &mut self,
        layout: &PipelineLayout,
        offset: u32,
        values: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        unsafe { self.try_push_constants(layout, offset, values) }.unwrap()
    }

    #[inline]
    pub unsafe fn try_push_constants(
        &mut self,
        layout: &PipelineLayout,
        offset: u32,
        values: &(impl BufferContents + ?Sized),
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_push_constants(layout, offset, size_of_val(values).try_into().unwrap())?;

        Ok(unsafe { self.push_constants_unchecked(layout, offset, values) })
    }

    pub(crate) fn validate_push_constants(
        &self,
        layout: &PipelineLayout,
        offset: u32,
        size: u32,
    ) -> Result<(), Box<ValidationError>> {
        let mut remaining_size = size as usize;

        if !offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdPushConstants-offset-00368"],
                ..Default::default()
            }));
        }

        if !remaining_size.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "values".into(),
                problem: "the size is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdPushConstants-size-00369"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();

        if offset >= properties.max_push_constants_size {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not less than the `max_push_constants_size` limit".into(),
                vuids: &["VUID-vkCmdPushConstants-offset-00370"],
                ..Default::default()
            }));
        }

        if (offset as usize)
            .checked_add(remaining_size)
            .is_none_or(|end| end > properties.max_push_constants_size as usize)
        {
            return Err(Box::new(ValidationError {
                problem: "`offset + size_of_val(values)` is greater than the \
                    `max_push_constants_size` limit"
                    .into(),
                vuids: &["VUID-vkCmdPushConstants-size-00371"],
                ..Default::default()
            }));
        }

        let mut current_offset = offset as usize;

        for range in layout
            .push_constant_ranges_disjoint()
            .iter()
            .skip_while(|range| range.offset + range.size <= offset)
        {
            // There is a gap between ranges, but the passed `values` contain some bytes in this
            // gap. Exit the loop and report error.
            if range.offset as usize > current_offset {
                break;
            }

            // Push the minimum of the whole remaining data and the part until the end of this
            // range.
            let push_size =
                remaining_size.min(range.offset as usize + range.size as usize - current_offset);
            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        if remaining_size != 0 {
            return Err(Box::new(ValidationError {
                problem: "one or more bytes of `values` are not within any push constant range of \
                    `layout`"
                    .into(),
                vuids: &["VUID-vkCmdPushConstants-offset-01795"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_constants_unchecked(
        &mut self,
        layout: &PipelineLayout,
        offset: u32,
        values: &(impl BufferContents + ?Sized),
    ) -> &mut Self {
        unsafe {
            self.push_constants_unchecked_inner(
                layout,
                offset,
                <*const _>::cast(values),
                size_of_val(values) as u32,
            )
        }
    }

    unsafe fn push_constants_unchecked_inner(
        &mut self,
        layout: &PipelineLayout,
        offset: u32,
        values: *const c_void,
        size: u32,
    ) -> &mut Self {
        if size == 0 {
            return self;
        }

        let fns = self.device().fns();
        let mut current_offset = offset;
        let mut remaining_size = size;

        for range in layout
            .push_constant_ranges_disjoint()
            .iter()
            .skip_while(|range| range.offset + range.size <= offset)
        {
            // There is a gap between ranges, but the passed `values` contain some bytes in this
            // gap.
            if range.offset > current_offset {
                std::process::abort();
            }

            // Push the minimum of the whole remaining data and the part until the end of this
            // range.
            let push_size = remaining_size.min(range.offset + range.size - current_offset);
            let push_offset = (current_offset - offset) as usize;
            debug_assert!(push_offset < size as usize);
            let push_values = unsafe { values.add(push_offset) };

            unsafe {
                (fns.v1_0.cmd_push_constants)(
                    self.handle(),
                    layout.handle(),
                    range.stages.into(),
                    current_offset,
                    push_size,
                    push_values,
                )
            };

            current_offset += push_size;
            remaining_size -= push_size;

            if remaining_size == 0 {
                break;
            }
        }

        self
    }

    #[inline]
    #[track_caller]
    pub unsafe fn push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: &PipelineLayout,
        set: u32,
        descriptor_writes: &[WriteDescriptorSet<'_>],
    ) -> &mut Self {
        unsafe {
            self.try_push_descriptor_set(pipeline_bind_point, pipeline, set, descriptor_writes)
        }
        .unwrap()
    }

    #[inline]
    pub unsafe fn try_push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: &PipelineLayout,
        set: u32,
        descriptor_writes: &[WriteDescriptorSet<'_>],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_push_descriptor_set(pipeline_bind_point, pipeline, set, descriptor_writes)?;

        Ok(unsafe {
            self.push_descriptor_set_unchecked(
                pipeline_bind_point,
                pipeline,
                set,
                descriptor_writes,
            )
        })
    }

    pub(crate) fn validate_push_descriptor_set(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        set: u32,
        descriptor_writes: &[WriteDescriptorSet<'_>],
    ) -> Result<(), Box<ValidationError>> {
        if !self.device().enabled_extensions().khr_push_descriptor {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_push_descriptor",
                )])]),
                ..Default::default()
            }));
        }

        pipeline_bind_point
            .validate_device(self.device())
            .map_err(|err| {
                err.add_context("pipeline_bind_point")
                    .set_vuids(&["VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-parameter"])
            })?;

        let queue_family_properties = self.queue_family_properties();

        match pipeline_bind_point {
            PipelineBindPoint::Compute => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "self".into(),
                        problem: "`pipeline_bind_point` is `PipelineBindPoint::Compute`, and the \
                            queue family does not support compute operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-00363",
                            "VUID-vkCmdPushDescriptorSetKHR-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
            PipelineBindPoint::Graphics => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
                {
                    return Err(Box::new(ValidationError {
                        context: "self".into(),
                        problem: "`pipeline_bind_point` is `PipelineBindPoint::Graphics`, and the \
                            queue family does not support graphics operations"
                            .into(),
                        vuids: &[
                            "VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-00363",
                            "VUID-vkCmdPushDescriptorSetKHR-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
            PipelineBindPoint::RayTracing => {
                if !queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                {
                    return Err(Box::new(ValidationError {
                        context: "self".into(),
                        problem:
                            "`pipeline_bind_point` is `PipelineBindPoint::RayTracing`, and the \
                            queue family does not support compute operations"
                                .into(),
                        vuids: &[
                            "VUID-vkCmdPushDescriptorSetKHR-pipelineBindPoint-02391",
                            "VUID-vkCmdPushDescriptorSetKHR-commandBuffer-cmdpool",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        // VUID-vkCmdPushDescriptorSetKHR-commonparent
        assert_eq!(self.device(), layout.device());

        if set as usize > layout.set_layouts().len() {
            return Err(Box::new(ValidationError {
                problem: "`set_num` is greater than the number of descriptor set layouts in \
                    `pipeline_layout`"
                    .into(),
                vuids: &["VUID-vkCmdPushDescriptorSetKHR-set-00364"],
                ..Default::default()
            }));
        }

        let descriptor_set_layout = &layout.set_layouts()[set as usize];

        if !descriptor_set_layout
            .flags()
            .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
        {
            return Err(Box::new(ValidationError {
                problem: "the descriptor set layout with the number `set_num` in \
                    `pipeline_layout` was not created with the \
                    `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR` flag"
                    .into(),
                vuids: &["VUID-vkCmdPushDescriptorSetKHR-set-00365"],
                ..Default::default()
            }));
        }

        for (index, write) in descriptor_writes.iter().enumerate() {
            write
                .validate(descriptor_set_layout, 0)
                .map_err(|err| err.add_context(format!("descriptor_writes[{}]", index)))?;
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_descriptor_set_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        set: u32,
        descriptor_writes: &[WriteDescriptorSet<'_>],
    ) -> &mut Self {
        if descriptor_writes.is_empty() {
            return self;
        }

        let set_layout = &layout.set_layouts()[set as usize];
        let writes_fields1_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .map(|write| {
                let default_image_layout = set_layout
                    .binding(write.dst_binding())
                    .unwrap()
                    .descriptor_type
                    .default_image_layout();
                write.to_vk_fields1(default_image_layout)
            })
            .collect();
        let mut writes_extensions_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .zip(&writes_fields1_vk)
            .map(|(write, fields1_vk)| write.to_vk_extensions(fields1_vk))
            .collect();
        let writes_vk: SmallVec<[_; 8]> = descriptor_writes
            .iter()
            .zip(&writes_fields1_vk)
            .zip(&mut writes_extensions_vk)
            .map(|((write, write_info_vk), write_extension_vk)| {
                write.to_vk(
                    vk::DescriptorSet::null(),
                    set_layout
                        .binding(write.dst_binding())
                        .unwrap()
                        .descriptor_type,
                    write_info_vk,
                    write_extension_vk,
                )
            })
            .collect();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_push_descriptor.cmd_push_descriptor_set_khr)(
                self.handle(),
                pipeline_bind_point.into(),
                layout.handle(),
                set,
                writes_vk.len() as u32,
                writes_vk.as_ptr(),
            )
        };

        self
    }
}
