use crate::{
    acceleration_structure::AccelerationStructure,
    buffer::{view::BufferView, BufferContents, IndexBuffer, Subbuffer},
    command_buffer::{auto::SetOrPush, sys::RecordingCommandBuffer, AutoCommandBufferBuilder},
    descriptor_set::{
        layout::{DescriptorBindingFlags, DescriptorSetLayoutCreateFlags, DescriptorType},
        DescriptorBindingResources, DescriptorBufferInfo, DescriptorImageInfo,
        DescriptorSetResources, DescriptorSetWithOffsets, DescriptorSetsCollection,
        OwnedDescriptorBufferInfo, OwnedDescriptorImageInfo, OwnedWriteDescriptorSetElements,
        WriteDescriptorSet, WriteDescriptorSetElements,
    },
    device::DeviceOwned,
    memory::is_aligned,
    pipeline::{
        graphics::vertex_input::VertexBuffersCollection, ray_tracing::RayTracingPipeline,
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    },
    DeviceSize, ValidationError,
};
use smallvec::SmallVec;
use std::{cmp::min, sync::Arc};

/// # Commands to bind or push state for pipeline execution commands.
///
/// These commands require a queue with a pipeline type that uses the given state.
impl<L> AutoCommandBufferBuilder<L> {
    /// Binds descriptor sets for future dispatch or draw calls.
    pub fn bind_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        descriptor_sets: impl DescriptorSetsCollection,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let descriptor_sets = descriptor_sets.into_vec();
        self.validate_bind_descriptor_sets(
            pipeline_bind_point,
            &pipeline_layout,
            first_set,
            &descriptor_sets,
        )?;

        Ok(unsafe {
            self.bind_descriptor_sets_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                first_set,
                descriptor_sets,
            )
        })
    }

    // TODO: The validation here is somewhat duplicated because of how different the parameters are
    // here compared to the raw command buffer.
    fn validate_bind_descriptor_sets(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[DescriptorSetWithOffsets],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_bind_descriptor_sets_inner(
            pipeline_bind_point,
            pipeline_layout,
            first_set,
            descriptor_sets.len(),
        )?;

        let properties = self.device().physical_device().properties();

        for (descriptor_sets_index, set) in descriptor_sets.iter().enumerate() {
            let set_num = first_set + descriptor_sets_index as u32;
            let (set, dynamic_offsets) = set.as_ref();

            // VUID-vkCmdBindDescriptorSets-commonparent
            assert_eq!(self.device(), set.device());

            let set_layout = set.layout();
            let pipeline_set_layout = &pipeline_layout.set_layouts()[set_num as usize];

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

            let mut dynamic_offsets_remaining = dynamic_offsets;
            let mut required_dynamic_offset_count = 0;

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

                    let resources = set.resources();
                    let elements = match resources.binding(binding_num) {
                        Some(DescriptorBindingResources::Buffer(elements)) => elements.as_slice(),
                        _ => unreachable!(),
                    };

                    for (index, (&dynamic_offset, element)) in
                        dynamic_offsets.iter().zip(elements).enumerate()
                    {
                        if !is_aligned(dynamic_offset as DeviceSize, required_alignment) {
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

                        if let Some(buffer_info) = element {
                            let &OwnedDescriptorBufferInfo {
                                ref buffer,
                                offset,
                                range,
                            } = buffer_info;

                            if let (Some(buffer), Some(range)) = (buffer, range) {
                                if (dynamic_offset as DeviceSize)
                                    .checked_add(offset)
                                    .and_then(|x| x.checked_add(range))
                                    .is_none_or(|end| end > buffer.size())
                                {
                                    return Err(Box::new(ValidationError {
                                        problem: format!(
                                            "the dynamic offset of `descriptor_sets[{}]` \
                                            (for set number {}) for binding {} index {}, when \
                                            added to `offset + range` of the descriptor write, is \
                                            greater than the size of the bound buffer",
                                            descriptor_sets_index, set_num, binding_num, index,
                                        )
                                        .into(),
                                        vuids: &[
                                            "VUID-vkCmdBindDescriptorSets-pDescriptorSets-01979",
                                        ],
                                        ..Default::default()
                                    }));
                                }
                            }
                        }
                    }
                }
            }

            if dynamic_offsets.len() != required_dynamic_offset_count {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the number of dynamic offsets provided for `descriptor_sets[{}]` \
                        (for set number {}) does not equal the number required ({})",
                        descriptor_sets_index, set_num, required_dynamic_offset_count,
                    )
                    .into(),
                    vuids: &["VUID-vkCmdBindDescriptorSets-dynamicOffsetCount-00359"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_descriptor_sets_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        first_set: u32,
        descriptor_sets: impl DescriptorSetsCollection,
    ) -> &mut Self {
        let descriptor_sets = descriptor_sets.into_vec();
        if descriptor_sets.is_empty() {
            return self;
        }

        let state = self.builder_state.invalidate_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout.clone(),
            first_set,
            descriptor_sets.len() as u32,
        );

        for (set_num, set) in descriptor_sets.iter().enumerate() {
            state
                .descriptor_sets
                .insert(first_set + set_num as u32, SetOrPush::Set(set.clone()));
        }

        self.add_command(
            "bind_descriptor_sets",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                let dynamic_offsets: SmallVec<[_; 32]> = descriptor_sets
                    .iter()
                    .flat_map(|x| x.as_ref().1.iter().copied())
                    .collect();
                let descriptor_sets: SmallVec<[_; 12]> = descriptor_sets
                    .iter()
                    .map(|x| x.as_ref().0.as_raw())
                    .collect();

                unsafe {
                    out.bind_descriptor_sets_unchecked(
                        pipeline_bind_point,
                        &pipeline_layout,
                        first_set,
                        &descriptor_sets,
                        &dynamic_offsets,
                    )
                };
            },
        );

        self
    }

    /// Binds an index buffer for future indexed draw calls.
    pub fn bind_index_buffer(
        &mut self,
        index_buffer: impl Into<IndexBuffer>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let index_buffer = index_buffer.into();
        self.validate_bind_index_buffer(&index_buffer)?;

        Ok(unsafe { self.bind_index_buffer_unchecked(index_buffer) })
    }

    fn validate_bind_index_buffer(
        &self,
        index_buffer: &IndexBuffer,
    ) -> Result<(), Box<ValidationError>> {
        let index_buffer_bytes = index_buffer.as_bytes();
        self.inner.validate_bind_index_buffer(
            index_buffer_bytes.buffer(),
            index_buffer_bytes.offset(),
            Some(index_buffer_bytes.size()),
            index_buffer.index_type(),
        )?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_index_buffer_unchecked(
        &mut self,
        index_buffer: impl Into<IndexBuffer>,
    ) -> &mut Self {
        let index_buffer = index_buffer.into();
        self.builder_state.index_buffer = Some(index_buffer.clone());
        self.add_command(
            "bind_index_buffer",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                let index_buffer_bytes = index_buffer.as_bytes();
                unsafe {
                    out.bind_index_buffer_unchecked(
                        index_buffer_bytes.buffer(),
                        index_buffer_bytes.offset(),
                        Some(index_buffer_bytes.size()),
                        index_buffer.index_type(),
                    )
                };
            },
        );

        self
    }

    /// Binds a compute pipeline for future dispatch calls.
    pub fn bind_pipeline_compute(
        &mut self,
        pipeline: Arc<ComputePipeline>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_compute(&pipeline)?;

        Ok(unsafe { self.bind_pipeline_compute_unchecked(pipeline) })
    }

    fn validate_bind_pipeline_compute(
        &self,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_bind_pipeline(Pipeline::Compute(pipeline))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_compute_unchecked(
        &mut self,
        pipeline: Arc<ComputePipeline>,
    ) -> &mut Self {
        self.builder_state.pipeline_compute = Some(pipeline.clone());
        self.add_command(
            "bind_pipeline_compute",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.bind_pipeline_unchecked(Pipeline::Compute(&pipeline)) };
            },
        );

        self
    }

    /// Binds a graphics pipeline for future draw calls.
    pub fn bind_pipeline_graphics(
        &mut self,
        pipeline: Arc<GraphicsPipeline>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_bind_pipeline_graphics(&pipeline)?;

        Ok(unsafe { self.bind_pipeline_graphics_unchecked(pipeline) })
    }

    fn validate_bind_pipeline_graphics(
        &self,
        pipeline: &Arc<GraphicsPipeline>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_bind_pipeline(Pipeline::Graphics(pipeline))?;

        // VUID-vkCmdBindPipeline-pipeline-00781
        // TODO:

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_graphics_unchecked(
        &mut self,
        pipeline: Arc<GraphicsPipeline>,
    ) -> &mut Self {
        // Reset any states that are fixed in the new pipeline. The pipeline bind command will
        // overwrite these states.
        self.builder_state
            .reset_dynamic_states(pipeline.fixed_state().iter().copied());
        self.builder_state.pipeline_graphics = Some(pipeline.clone());
        self.add_command(
            "bind_pipeline_graphics",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.bind_pipeline_unchecked(Pipeline::Graphics(&pipeline)) };
            },
        );

        self
    }

    /// Binds a ray tracing pipeline for future ray tracing calls.
    pub fn bind_pipeline_ray_tracing(
        &mut self,
        pipeline: Arc<RayTracingPipeline>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.inner
            .validate_bind_pipeline(Pipeline::RayTracing(&pipeline))?;
        Ok(unsafe { self.bind_pipeline_ray_tracing_unchecked(pipeline) })
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_pipeline_ray_tracing_unchecked(
        &mut self,
        pipeline: Arc<RayTracingPipeline>,
    ) -> &mut Self {
        self.builder_state.pipeline_ray_tracing = Some(pipeline.clone());
        self.add_command(
            "bind_pipeline_ray_tracing",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.bind_pipeline_unchecked(Pipeline::RayTracing(&pipeline)) };
            },
        );

        self
    }

    /// Binds vertex buffers for future draw calls.
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        vertex_buffers: impl VertexBuffersCollection,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let vertex_buffers = vertex_buffers.into_vec();
        self.validate_bind_vertex_buffers(first_binding, &vertex_buffers)?;

        Ok(unsafe { self.bind_vertex_buffers_unchecked(first_binding, vertex_buffers) })
    }

    fn validate_bind_vertex_buffers(
        &self,
        first_binding: u32,
        vertex_buffers: &[Subbuffer<[u8]>],
    ) -> Result<(), Box<ValidationError>> {
        let buffers_raw = vertex_buffers
            .iter()
            .map(Subbuffer::buffer)
            .map(Arc::as_ref)
            .collect::<SmallVec<[_; 2]>>();
        let offsets_raw = vertex_buffers
            .iter()
            .map(Subbuffer::offset)
            .collect::<SmallVec<[_; 2]>>();
        let sizes_raw = vertex_buffers
            .iter()
            .map(Subbuffer::size)
            .collect::<SmallVec<[_; 2]>>();
        self.inner.validate_bind_vertex_buffers(
            first_binding,
            &buffers_raw,
            &offsets_raw,
            &sizes_raw,
            &[],
        )?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_vertex_buffers_unchecked(
        &mut self,
        first_binding: u32,
        vertex_buffers: impl VertexBuffersCollection,
    ) -> &mut Self {
        let vertex_buffers = vertex_buffers.into_vec();

        for (i, buffer) in vertex_buffers.iter().enumerate() {
            self.builder_state
                .vertex_buffers
                .insert(first_binding + i as u32, buffer.clone());
        }

        self.add_command(
            "bind_vertex_buffers",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                let buffers_raw = vertex_buffers
                    .iter()
                    .map(Subbuffer::buffer)
                    .map(Arc::as_ref)
                    .collect::<SmallVec<[_; 2]>>();
                let offsets_raw = vertex_buffers
                    .iter()
                    .map(Subbuffer::offset)
                    .collect::<SmallVec<[_; 2]>>();
                let sizes_raw = vertex_buffers
                    .iter()
                    .map(Subbuffer::size)
                    .collect::<SmallVec<[_; 2]>>();
                unsafe {
                    out.bind_vertex_buffers_unchecked(
                        first_binding,
                        &buffers_raw,
                        &offsets_raw,
                        &sizes_raw,
                        &[],
                    )
                };
            },
        );

        self
    }

    /// Sets push constants for future dispatch or draw calls.
    pub fn push_constants<Pc>(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        push_constants: Pc,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        Pc: BufferContents,
    {
        if size_of::<Pc>() == 0 {
            return Ok(self);
        }

        self.validate_push_constants(&pipeline_layout, offset, &push_constants)?;

        Ok(unsafe { self.push_constants_unchecked(pipeline_layout, offset, push_constants) })
    }

    fn validate_push_constants<Pc: BufferContents>(
        &self,
        pipeline_layout: &PipelineLayout,
        offset: u32,
        push_constants: &Pc,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_push_constants(
            pipeline_layout,
            offset,
            size_of_val(push_constants).try_into().unwrap(),
        )?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_constants_unchecked<Pc>(
        &mut self,
        pipeline_layout: Arc<PipelineLayout>,
        offset: u32,
        push_constants: Pc,
    ) -> &mut Self
    where
        Pc: BufferContents,
    {
        // TODO: Push constant invalidations.
        // The Vulkan spec currently is unclear about this, so Vulkano currently just marks
        // push constants as set, and never unsets them. See:
        // https://github.com/KhronosGroup/Vulkan-Docs/issues/1485
        // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2711
        self.builder_state
            .push_constants
            .insert(offset..offset + size_of::<Pc>() as u32);
        self.builder_state.push_constants_pipeline_layout = Some(pipeline_layout.clone());

        self.add_command(
            "push_constants",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.push_constants_unchecked(&pipeline_layout, offset, &push_constants) };
            },
        );

        self
    }

    /// Pushes descriptor data directly into the command buffer for future dispatch or draw calls.
    pub fn push_descriptor_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        descriptor_writes: &[WriteDescriptorSet<'_>],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_push_descriptor_set(
            pipeline_bind_point,
            &pipeline_layout,
            set_num,
            descriptor_writes,
        )?;

        Ok(unsafe {
            self.push_descriptor_set_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                set_num,
                descriptor_writes,
            )
        })
    }

    fn validate_push_descriptor_set(
        &self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &PipelineLayout,
        set_num: u32,
        descriptor_writes: &[WriteDescriptorSet<'_>],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_push_descriptor_set(
            pipeline_bind_point,
            pipeline_layout,
            set_num,
            descriptor_writes,
        )?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn push_descriptor_set_unchecked(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: Arc<PipelineLayout>,
        set_num: u32,
        descriptor_writes: &[WriteDescriptorSet<'_>],
    ) -> &mut Self {
        let state = self.builder_state.invalidate_descriptor_sets(
            pipeline_bind_point,
            pipeline_layout.clone(),
            set_num,
            1,
        );
        let layout = state.pipeline_layout.set_layouts()[set_num as usize].as_ref();
        debug_assert!(layout
            .flags()
            .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR));

        let set_resources = match state
            .descriptor_sets
            .entry(set_num)
            .or_insert_with(|| SetOrPush::Push(DescriptorSetResources::new(layout, 0)))
        {
            SetOrPush::Push(set_resources) => set_resources,
            _ => unreachable!(),
        };

        for write in descriptor_writes {
            set_resources.write(write, layout);
        }

        let descriptor_writes = descriptor_writes
            .iter()
            .map(WriteDescriptorSet::to_owned)
            .collect::<Vec<_>>();

        self.add_command(
            "push_descriptor_set",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                enum Elements<'a> {
                    Image(SmallVec<[DescriptorImageInfo<'a>; 1]>),
                    Buffer(SmallVec<[DescriptorBufferInfo<'a>; 1]>),
                    BufferView(SmallVec<[Option<&'a Arc<BufferView>>; 1]>),
                    InlineUniformBlock(&'a [u8]),
                    AccelerationStructure(SmallVec<[Option<&'a Arc<AccelerationStructure>>; 1]>),
                }

                let elements = descriptor_writes
                    .iter()
                    .map(|descriptor_write| match &descriptor_write.elements {
                        OwnedWriteDescriptorSetElements::Image(elements) => Elements::Image(
                            elements
                                .iter()
                                .map(OwnedDescriptorImageInfo::as_ref)
                                .collect(),
                        ),
                        OwnedWriteDescriptorSetElements::Buffer(elements) => Elements::Buffer(
                            elements
                                .iter()
                                .map(OwnedDescriptorBufferInfo::as_ref)
                                .collect(),
                        ),
                        OwnedWriteDescriptorSetElements::BufferView(elements) => {
                            Elements::BufferView(elements.iter().map(Option::as_ref).collect())
                        }
                        OwnedWriteDescriptorSetElements::InlineUniformBlock(data) => {
                            Elements::InlineUniformBlock(data)
                        }
                        OwnedWriteDescriptorSetElements::AccelerationStructure(elements) => {
                            Elements::AccelerationStructure(
                                elements.iter().map(Option::as_ref).collect(),
                            )
                        }
                    })
                    .collect::<Vec<_>>();
                let descriptor_writes = descriptor_writes
                    .iter()
                    .zip(&elements)
                    .map(|(descriptor_write, elements)| WriteDescriptorSet {
                        dst_binding: descriptor_write.dst_binding,
                        dst_array_element: descriptor_write.dst_array_element,
                        elements: match elements {
                            Elements::Image(elements) => {
                                WriteDescriptorSetElements::Image(elements)
                            }
                            Elements::Buffer(elements) => {
                                WriteDescriptorSetElements::Buffer(elements)
                            }
                            Elements::BufferView(elements) => {
                                WriteDescriptorSetElements::BufferView(elements)
                            }
                            Elements::InlineUniformBlock(data) => {
                                WriteDescriptorSetElements::InlineUniformBlock(data)
                            }
                            Elements::AccelerationStructure(elements) => {
                                WriteDescriptorSetElements::AccelerationStructure(elements)
                            }
                        },
                    })
                    .collect::<Vec<_>>();

                unsafe {
                    out.push_descriptor_set_unchecked(
                        pipeline_bind_point,
                        &pipeline_layout,
                        set_num,
                        &descriptor_writes,
                    )
                };
            },
        );

        self
    }
}
