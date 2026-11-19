#[cfg(doc)]
use crate::device::{DeviceFeatures, DeviceProperties};
use crate::{
    acceleration_structure::AccelerationStructure,
    buffer::{view::BufferView, BufferUsage, Subbuffer},
    command_buffer::{
        auto::{RenderPassState, RenderPassStateType, Resource, ResourceUseRef2},
        sys::RecordingCommandBuffer,
        AutoCommandBufferBuilder, DispatchIndirectCommand, DrawIndexedIndirectCommand,
        DrawIndirectCommand, DrawMeshTasksIndirectCommand, ResourceInCommand, SubpassContents,
        TraceRaysIndirectCommand,
    },
    descriptor_set::{
        layout::{DescriptorBindingFlags, DescriptorType},
        DescriptorBindingResources, OwnedDescriptorBufferInfo, OwnedDescriptorImageInfo,
    },
    device::DeviceOwned,
    format::{FormatFeatures, NumericType},
    image::{sampler::Sampler, view::ImageView, ImageAspects, ImageLayout, SampleCount},
    pipeline::{
        graphics::{
            input_assembly::PrimitiveTopology,
            subpass::PipelineSubpassType,
            vertex_input::{RequiredVertexInputsVUIDs, VertexInputRate},
        },
        ray_tracing::ShaderBindingTableAddresses,
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout,
    },
    query::QueryType,
    shader::{DescriptorBindingRequirements, DescriptorIdentifier, ShaderStages},
    sync::{PipelineStageAccess, PipelineStageAccessFlags},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use std::sync::Arc;

macro_rules! vuids {
    ($vuid_type:ident, $($id:literal),+ $(,)?) => {
        match $vuid_type {
            VUIDType::Dispatch => &[$(concat!("VUID-vkCmdDispatch-", $id)),+],
            VUIDType::DispatchIndirect => &[$(concat!("VUID-vkCmdDispatchIndirect-", $id)),+],
            VUIDType::Draw => &[$(concat!("VUID-vkCmdDraw-", $id)),+],
            VUIDType::DrawIndirect => &[$(concat!("VUID-vkCmdDrawIndirect-", $id)),+],
            VUIDType::DrawIndirectCount => &[$(concat!("VUID-vkCmdDrawIndirectCount-", $id)),+],
            VUIDType::DrawIndexed => &[$(concat!("VUID-vkCmdDrawIndexed-", $id)),+],
            VUIDType::DrawIndexedIndirect => &[$(concat!("VUID-vkCmdDrawIndexedIndirect-", $id)),+],
            VUIDType::DrawIndexedIndirectCount => &[$(concat!("VUID-vkCmdDrawIndexedIndirectCount-", $id)),+],
            VUIDType::DrawMeshTasks => &[$(concat!("VUID-vkCmdDrawMeshTasksEXT-", $id)),+],
            VUIDType::DrawMeshTasksIndirect => &[$(concat!("VUID-vkCmdDrawMeshTasksIndirectEXT-", $id)),+],
            VUIDType::DrawMeshTasksIndirectCount => &[$(concat!("VUID-vkCmdDrawMeshTasksIndirectCountEXT-", $id)),+],
        }
    };
}

/// # Commands to execute a bound pipeline.
///
/// Dispatch commands require a compute queue, draw commands require a graphics queue.
impl<L> AutoCommandBufferBuilder<L> {
    /// Perform a single compute operation using a compute pipeline.
    ///
    /// A compute pipeline must have been bound using
    /// [`bind_pipeline_compute`](Self::bind_pipeline_compute). Any resources used by the compute
    /// pipeline, such as descriptor sets, must have been set beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    pub unsafe fn dispatch(
        &mut self,
        group_counts: [u32; 3],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_dispatch(group_counts)?;

        Ok(unsafe { self.dispatch_unchecked(group_counts) })
    }

    fn validate_dispatch(&self, group_counts: [u32; 3]) -> Result<(), Box<ValidationError>> {
        self.inner.validate_dispatch(group_counts)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdDispatch-renderpass"],
                ..Default::default()
            }));
        }

        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no compute pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDispatch-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::Dispatch;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        let pipeline = self.builder_state.pipeline_compute.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());

        self.add_command(
            "dispatch",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.dispatch_unchecked(group_counts) };
            },
        );

        self
    }

    /// Perform multiple compute operations using a compute pipeline. One dispatch is performed for
    /// each [`DispatchIndirectCommand`] struct in `indirect_buffer`.
    ///
    /// A compute pipeline must have been bound using
    /// [`bind_pipeline_compute`](Self::bind_pipeline_compute). Any resources used by the compute
    /// pipeline, such as descriptor sets, must have been set beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    /// - The [safety requirements for `DispatchIndirectCommand`](DispatchIndirectCommand#safety)
    ///   apply.
    pub unsafe fn dispatch_indirect(
        &mut self,
        indirect_buffer: Subbuffer<[DispatchIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_dispatch_indirect(indirect_buffer.as_bytes())?;

        Ok(unsafe { self.dispatch_indirect_unchecked(indirect_buffer) })
    }

    fn validate_dispatch_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_dispatch_indirect(indirect_buffer.buffer(), indirect_buffer.offset())?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdDispatchIndirect-renderpass"],
                ..Default::default()
            }));
        }

        let pipeline = self
            .builder_state
            .pipeline_compute
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no compute pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDispatchIndirect-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DispatchIndirect;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn dispatch_indirect_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DispatchIndirectCommand]>,
    ) -> &mut Self {
        let pipeline = self.builder_state.pipeline_compute.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());

        self.add_command(
            "dispatch",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.dispatch_indirect_unchecked(
                        indirect_buffer.buffer(),
                        indirect_buffer.offset(),
                    )
                };
            },
        );

        self
    }

    /// Perform a single draw operation using a primitive shading graphics pipeline.
    ///
    /// The parameters specify the first vertex and the number of vertices to draw, and the first
    /// instance and number of instances. For non-instanced drawing, specify `instance_count` as 1
    /// and `first_instance` as 0.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets, vertex buffers and dynamic state, must have
    /// been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// provided vertex and instance ranges must be in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw(vertex_count, instance_count, first_vertex, first_instance)?;

        Ok(unsafe {
            self.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance)
        })
    }

    fn validate_draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_draw(vertex_count, instance_count, first_vertex, first_instance)?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDraw-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDraw-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::Draw;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_primitive_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        let view_mask = match pipeline.subpass() {
            PipelineSubpassType::BeginRenderPass(subpass) => subpass.render_pass().views_used(),
            PipelineSubpassType::BeginRendering(rendering_info) => rendering_info.view_mask,
        };

        if view_mask != 0 {
            let properties = self.device().physical_device().properties();

            if (first_instance + instance_count).saturating_sub(1)
                > properties.max_multiview_instance_index.unwrap_or(0)
            {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance has a nonzero view mask, but \
                        `first_instance + instance_count - 1` is greater than the \
                        `max_multiview_instance_index` limit"
                        .into(),
                    vuids: &["VUID-vkCmdDraw-maxMultiviewInstanceIndex-02688"],
                    ..Default::default()
                }));
            }
        }

        let vertex_input_state = if pipeline
            .dynamic_state()
            .contains(&DynamicState::VertexInput)
        {
            self.builder_state.vertex_input.as_ref().unwrap()
        } else {
            pipeline.vertex_input_state().unwrap()
        };

        for (&binding_num, binding_desc) in &vertex_input_state.bindings {
            let vertex_buffer = &self.builder_state.vertex_buffers[&binding_num];

            // Per spec:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap22.html#fxvertex-input-address-calculation
            match binding_desc.input_rate {
                VertexInputRate::Vertex => {
                    let max_vertex_offset = (first_vertex as DeviceSize
                        + vertex_count as DeviceSize)
                        * binding_desc.stride as DeviceSize;

                    if max_vertex_offset > vertex_buffer.size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the size of the vertex buffer bound to binding {} is less \
                                than the minimum size required, for the provided \
                                `first_vertex` and `vertex_count` values, and the vertex \
                                input state bindings of the currently bound graphics pipeline",
                                binding_num
                            )
                            .into(),
                            vuids: &["VUID-vkCmdDraw-None-02721"],
                            ..Default::default()
                        }));
                    }
                }
                VertexInputRate::Instance { divisor } => {
                    let max_vertex_offset = if divisor == 0 {
                        (first_instance as DeviceSize + 1) * binding_desc.stride as DeviceSize
                    } else {
                        (first_instance as DeviceSize
                            + instance_count as DeviceSize / divisor as DeviceSize)
                            * binding_desc.stride as DeviceSize
                    };

                    if max_vertex_offset > vertex_buffer.size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the size of the vertex buffer bound to binding {} is less \
                                than the minimum size required, for the provided \
                                `first_instance` and `instance_count` values, and the vertex \
                                input state bindings of the currently bound graphics pipeline",
                                binding_num
                            )
                            .into(),
                            vuids: &["VUID-vkCmdDraw-None-02721"],
                            ..Default::default()
                        }));
                    }
                }
            };
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_unchecked(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);

        self.add_command(
            "draw",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance)
                };
            },
        );

        self
    }

    /// Perform multiple draw operations using a primitive shading graphics pipeline.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) limit.
    /// This limit is 1 unless the
    /// [`multi_draw_indirect`](DeviceFeatures::multi_draw_indirect) feature has been
    /// enabled.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets, vertex buffers and dynamic state, must have
    /// been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// vertex and instance ranges of each `DrawIndirectCommand` in the indirect buffer must be
    /// in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    /// - The [safety requirements for `DrawIndirectCommand`](DrawIndirectCommand#safety) apply.
    pub unsafe fn draw_indirect(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawIndirectCommand>() as u32;
        self.validate_draw_indirect(indirect_buffer.as_bytes(), draw_count, stride)?;

        Ok(unsafe { self.draw_indirect_unchecked(indirect_buffer, draw_count, stride) })
    }

    fn validate_draw_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_indirect(
            indirect_buffer.buffer(),
            indirect_buffer.offset(),
            draw_count,
            stride,
        )?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDrawIndirect-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawIndirect-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DrawIndirect;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_primitive_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indirect_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());

        self.add_command(
            "draw_indirect",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.draw_indirect_unchecked(
                        indirect_buffer.buffer(),
                        indirect_buffer.offset(),
                        draw_count,
                        stride,
                    )
                };
            },
        );

        self
    }

    /// Perform multiple draw operations using a primitive shading graphics pipeline,
    /// reading the number of draw operations from a separate buffer.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct that is read from
    /// `indirect_buffer`. The number of draws to perform is read from `count_buffer`, or
    /// specified by `max_draw_count`, whichever is lower.
    /// This number is limited by the
    /// [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) limit.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets, vertex buffers and dynamic state, must have
    /// been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// vertex and instance ranges of each `DrawIndirectCommand` in the indirect buffer must be
    /// in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    /// - The [safety requirements for `DrawIndirectCommand`](DrawIndirectCommand#safety) apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) device limit.
    /// - The count stored in `count_buffer` must fall within the range of `indirect_buffer`.
    pub unsafe fn draw_indirect_count(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndirectCommand]>,
        count_buffer: Subbuffer<u32>,
        max_draw_count: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let stride = size_of::<DrawIndirectCommand>() as u32;
        self.validate_draw_indirect_count(
            indirect_buffer.as_bytes(),
            count_buffer.as_bytes(),
            max_draw_count,
            stride,
        )?;

        Ok(unsafe {
            self.draw_indirect_count_unchecked(
                indirect_buffer,
                count_buffer,
                max_draw_count,
                stride,
            )
        })
    }

    fn validate_draw_indirect_count(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        count_buffer: &Subbuffer<[u8]>,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_indirect_count(
            indirect_buffer.buffer(),
            indirect_buffer.offset(),
            count_buffer.buffer(),
            count_buffer.offset(),
            max_draw_count,
            stride,
        )?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDrawIndirectCount-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawIndirectCount-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DrawIndirectCount;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_primitive_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indirect_count_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndirectCommand]>,
        count_buffer: Subbuffer<u32>,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());
        self.add_indirect_buffer_resources(&mut used_resources, count_buffer.as_bytes());

        self.add_command(
            "draw_indirect_count",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.draw_indirect_count_unchecked(
                        indirect_buffer.buffer(),
                        indirect_buffer.offset(),
                        count_buffer.buffer(),
                        count_buffer.offset(),
                        max_draw_count,
                        stride,
                    )
                };
            },
        );

        self
    }

    /// Perform a single draw operation using a primitive shading graphics pipeline,
    /// using an index buffer.
    ///
    /// The parameters specify the first index and the number of indices in the index buffer that
    /// should be used, and the first instance and number of instances. For non-instanced drawing,
    /// specify `instance_count` as 1 and `first_instance` as 0. The `vertex_offset` is a constant
    /// value that should be added to each index in the index buffer to produce the final vertex
    /// number to be used.
    ///
    /// An index buffer must have been bound using
    /// [`bind_index_buffer`](Self::bind_index_buffer), and the provided index range must be in
    /// range of the bound index buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets, vertex buffers and dynamic state, must have
    /// been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// provided instance range must be in range of the bound vertex buffers. The vertex
    /// indices in the index buffer must be in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    /// - Every vertex number that is retrieved from the index buffer must fall within the range of
    ///   the bound vertex-rate vertex buffers.
    /// - Every vertex number that is retrieved from the index buffer, if it is not the special
    ///   primitive restart value, must be no greater than the
    ///   [`max_draw_indexed_index_value`](DeviceProperties::max_draw_indexed_index_value) device
    ///   limit.
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )?;

        Ok(unsafe {
            self.draw_indexed_unchecked(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        })
    }

    fn validate_draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDrawIndexed-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawIndexed-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DrawIndexed;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_primitive_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        let index_buffer = self.builder_state.index_buffer.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "no index buffer is currently bound".into(),
                vuids: &["VUID-vkCmdDrawIndexed-None-07312"],
                ..Default::default()
            })
        })?;

        let index_buffer_bytes = index_buffer.as_bytes();

        if !self.device().enabled_features().robust_buffer_access2 {
            if index_buffer.index_type().size()
                * (first_index as DeviceSize + index_count as DeviceSize)
                > index_buffer_bytes.size()
            {
                return Err(Box::new(ValidationError {
                    problem: "`first_index + index_count`, \
                        multiplied by the size of the indices in the bound index buffer, \
                        is greater than the size of the bound index buffer"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "robust_buffer_access2",
                    )])]),
                    vuids: &["VUID-vkCmdDrawIndexed-robustBufferAccess2-07825"],
                    ..Default::default()
                }));
            }
        }

        let view_mask = match pipeline.subpass() {
            PipelineSubpassType::BeginRenderPass(subpass) => subpass.render_pass().views_used(),
            PipelineSubpassType::BeginRendering(rendering_info) => rendering_info.view_mask,
        };

        if view_mask != 0 {
            let properties = self.device().physical_device().properties();

            if (first_instance + instance_count).saturating_sub(1)
                > properties.max_multiview_instance_index.unwrap_or(0)
            {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance has a nonzero view mask, but \
                        `first_instance + instance_count - 1` is greater than the \
                        `max_multiview_instance_index` limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawIndexed-maxMultiviewInstanceIndex-02688"],
                    ..Default::default()
                }));
            }
        }

        let vertex_input_state = if pipeline
            .dynamic_state()
            .contains(&DynamicState::VertexInput)
        {
            self.builder_state.vertex_input.as_ref().unwrap()
        } else {
            pipeline.vertex_input_state().unwrap()
        };

        for (&binding_num, binding_desc) in &vertex_input_state.bindings {
            let vertex_buffer = &self.builder_state.vertex_buffers[&binding_num];

            // Per spec:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap22.html#fxvertex-input-address-calculation
            match binding_desc.input_rate {
                VertexInputRate::Vertex => (),
                VertexInputRate::Instance { divisor } => {
                    let max_vertex_offset = if divisor == 0 {
                        (first_instance as DeviceSize + 1) * binding_desc.stride as DeviceSize
                    } else {
                        (first_instance as DeviceSize
                            + instance_count as DeviceSize / divisor as DeviceSize)
                            * binding_desc.stride as DeviceSize
                    };

                    if max_vertex_offset > vertex_buffer.size() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the size of the vertex buffer bound to binding {} is less \
                                than the minimum size required, for the provided \
                                `first_instance` and `instance_count` values, and the vertex \
                                input state bindings of the currently bound graphics pipeline",
                                binding_num
                            )
                            .into(),
                            vuids: &["VUID-vkCmdDrawIndexed-None-02721"],
                            ..Default::default()
                        }));
                    }
                }
            };
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_unchecked(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);
        self.add_index_buffer_resources(&mut used_resources);

        self.add_command(
            "draw_indexed",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.draw_indexed_unchecked(
                        index_count,
                        instance_count,
                        first_index,
                        vertex_offset,
                        first_instance,
                    )
                };
            },
        );

        self
    }

    /// Perform multiple draw operations using a primitive shading graphics pipeline,
    /// using an index buffer.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct in `indirect_buffer`.
    /// The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) limit.
    /// This limit is 1 unless the
    /// [`multi_draw_indirect`](DeviceFeatures::multi_draw_indirect) feature has been
    /// enabled.
    ///
    /// An index buffer must have been bound using
    /// [`bind_index_buffer`](Self::bind_index_buffer), and the index ranges of each
    /// `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound index
    /// buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets, vertex buffers and dynamic state, must have
    /// been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// instance ranges of each `DrawIndexedIndirectCommand` in the indirect buffer must be in
    /// range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    /// - The [safety requirements for
    ///   `DrawIndexedIndirectCommand`](DrawIndexedIndirectCommand#safety) apply.
    pub unsafe fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawIndexedIndirectCommand>() as u32;
        self.validate_draw_indexed_indirect(indirect_buffer.as_bytes(), draw_count, stride)?;

        Ok(unsafe { self.draw_indexed_indirect_unchecked(indirect_buffer, draw_count, stride) })
    }

    fn validate_draw_indexed_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_indexed_indirect(
            indirect_buffer.buffer(),
            indirect_buffer.offset(),
            draw_count,
            stride,
        )?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirect-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DrawIndexedIndirect;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_primitive_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        let _index_buffer = self.builder_state.index_buffer.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "no index buffer is currently bound".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirect-None-07312"],
                ..Default::default()
            })
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_indirect_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);
        self.add_index_buffer_resources(&mut used_resources);
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());

        self.add_command(
            "draw_indexed_indirect",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.draw_indexed_indirect_unchecked(
                        indirect_buffer.buffer(),
                        indirect_buffer.offset(),
                        draw_count,
                        stride,
                    )
                };
            },
        );

        self
    }

    /// Perform multiple draw operations using a primitive shading graphics pipeline,
    /// using an index buffer, and reading the number of draw operations from a separate buffer.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct that is read from
    /// `indirect_buffer`. The number of draws to perform is read from `count_buffer`, or
    /// specified by `max_draw_count`, whichever is lower.
    /// This number is limited by the
    /// [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) limit.
    ///
    /// An index buffer must have been bound using
    /// [`bind_index_buffer`](Self::bind_index_buffer), and the index ranges of each
    /// `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound index
    /// buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets, vertex buffers and dynamic state, must have
    /// been set beforehand. If the bound graphics pipeline uses vertex buffers, then the
    /// instance ranges of each `DrawIndexedIndirectCommand` in the indirect buffer must be in
    /// range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    /// - The [safety requirements for
    ///   `DrawIndexedIndirectCommand`](DrawIndexedIndirectCommand#safety) apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) device limit.
    /// - The count stored in `count_buffer` must fall within the range of `indirect_buffer`.
    pub unsafe fn draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
        count_buffer: Subbuffer<u32>,
        max_draw_count: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let stride = size_of::<DrawIndexedIndirectCommand>() as u32;
        self.validate_draw_indexed_indirect_count(
            indirect_buffer.as_bytes(),
            count_buffer.as_bytes(),
            max_draw_count,
            stride,
        )?;

        Ok(unsafe {
            self.draw_indexed_indirect_count_unchecked(
                indirect_buffer,
                count_buffer,
                max_draw_count,
                stride,
            )
        })
    }

    fn validate_draw_indexed_indirect_count(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        count_buffer: &Subbuffer<[u8]>,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_indexed_indirect_count(
            indirect_buffer.buffer(),
            indirect_buffer.offset(),
            count_buffer.buffer(),
            count_buffer.offset(),
            max_draw_count,
            stride,
        )?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawIndexedIndirectCount-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DrawIndexedIndirectCount;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_primitive_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        let _index_buffer = self.builder_state.index_buffer.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "no index buffer is currently bound".into(),
                vuids: &["VUID-vkCmdDrawIndexedIndirectCount-None-07312"],
                ..Default::default()
            })
        })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_indexed_indirect_count_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
        count_buffer: Subbuffer<u32>,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_vertex_buffers_resources(&mut used_resources, pipeline);
        self.add_index_buffer_resources(&mut used_resources);
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());
        self.add_indirect_buffer_resources(&mut used_resources, count_buffer.as_bytes());

        self.add_command(
            "draw_indexed_indirect_count",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.draw_indexed_indirect_count_unchecked(
                        indirect_buffer.buffer(),
                        indirect_buffer.offset(),
                        count_buffer.buffer(),
                        count_buffer.offset(),
                        max_draw_count,
                        stride,
                    )
                };
            },
        );

        self
    }

    /// Perform a single draw operation using a mesh shading graphics pipeline.
    ///
    /// A mesh shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets and dynamic state, must have been set
    /// beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    pub unsafe fn draw_mesh_tasks(
        &mut self,
        group_counts: [u32; 3],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_draw_mesh_tasks(group_counts)?;

        Ok(unsafe { self.draw_mesh_tasks_unchecked(group_counts) })
    }

    fn validate_draw_mesh_tasks(&self, group_counts: [u32; 3]) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_mesh_tasks(group_counts)?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksEXT-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DrawMeshTasks;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_mesh_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        if pipeline.mesh_is_nv() {
            return Err(Box::new(ValidationError {
                problem: "the currently bound graphics pipeline uses NV mesh shaders instead of \
                    EXT mesh shaders"
                    .into(),
                vuids: &["VUID-vkCmdDrawMeshTasksEXT-MeshEXT-07087"],
                ..Default::default()
            }));
        }

        let properties = self.device().physical_device().properties();
        let group_counts_product = group_counts.into_iter().try_fold(1, u32::checked_mul);

        if pipeline.shader_stages().intersects(ShaderStages::TASK) {
            if group_counts[0] > properties.max_task_work_group_count.unwrap_or_default()[0] {
                return Err(Box::new(ValidationError {
                    context: "group_counts[0]".into(),
                    problem: "is greater than the `max_task_work_group_count[0]` device limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-TaskEXT-07322"],
                    ..Default::default()
                }));
            }

            if group_counts[1] > properties.max_task_work_group_count.unwrap_or_default()[1] {
                return Err(Box::new(ValidationError {
                    context: "group_counts[1]".into(),
                    problem: "is greater than the `max_task_work_group_count[1]` device limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-TaskEXT-07323"],
                    ..Default::default()
                }));
            }

            if group_counts[2] > properties.max_task_work_group_count.unwrap_or_default()[2] {
                return Err(Box::new(ValidationError {
                    context: "group_counts[2]".into(),
                    problem: "is greater than the `max_task_work_group_count[2]` device limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-TaskEXT-07324"],
                    ..Default::default()
                }));
            }

            if group_counts_product.is_none_or(|size| {
                size > properties
                    .max_task_work_group_total_count
                    .unwrap_or_default()
            }) {
                return Err(Box::new(ValidationError {
                    context: "group_counts".into(),
                    problem: "the product is greater than the `max_task_work_group_total_count` \
                        device limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-TaskEXT-07325"],
                    ..Default::default()
                }));
            }
        } else {
            if group_counts[0] > properties.max_mesh_work_group_count.unwrap_or_default()[0] {
                return Err(Box::new(ValidationError {
                    context: "group_counts[0]".into(),
                    problem: "is greater than the `max_mesh_work_group_count[0]` device limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-TaskEXT-07326"],
                    ..Default::default()
                }));
            }

            if group_counts[1] > properties.max_mesh_work_group_count.unwrap_or_default()[1] {
                return Err(Box::new(ValidationError {
                    context: "group_counts[1]".into(),
                    problem: "is greater than the `max_mesh_work_group_count[1]` device limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-TaskEXT-07327"],
                    ..Default::default()
                }));
            }

            if group_counts[2] > properties.max_mesh_work_group_count.unwrap_or_default()[2] {
                return Err(Box::new(ValidationError {
                    context: "group_counts[2]".into(),
                    problem: "is greater than the `max_mesh_work_group_count[2]` device limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-TaskEXT-07328"],
                    ..Default::default()
                }));
            }

            if group_counts_product.is_none_or(|size| {
                size > properties
                    .max_mesh_work_group_total_count
                    .unwrap_or_default()
            }) {
                return Err(Box::new(ValidationError {
                    context: "group_counts".into(),
                    problem: "the product is greater than the `max_mesh_work_group_total_count` \
                        device limit"
                        .into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksEXT-TaskEXT-07329"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_mesh_tasks_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());

        self.add_command(
            "draw_mesh_tasks",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.draw_mesh_tasks_unchecked(group_counts) };
            },
        );

        self
    }

    /// Perform multiple draw operations using a mesh shading graphics pipeline.
    ///
    /// One draw is performed for each [`DrawMeshTasksIndirectCommand`] struct in
    /// `indirect_buffer`. The maximum number of draw commands in the buffer is limited by the
    /// [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) limit.
    /// This limit is 1 unless the
    /// [`multi_draw_indirect`](DeviceFeatures::multi_draw_indirect) feature has been
    /// enabled.
    ///
    /// A mesh shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets and dynamic state, must have been set
    /// beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    /// - The [safety requirements for
    ///   `DrawMeshTasksIndirectCommand`](DrawMeshTasksIndirectCommand#safety) apply.
    pub unsafe fn draw_mesh_tasks_indirect(
        &mut self,
        indirect_buffer: Subbuffer<[DrawMeshTasksIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let draw_count = indirect_buffer.len() as u32;
        let stride = size_of::<DrawMeshTasksIndirectCommand>() as u32;
        self.validate_draw_mesh_tasks_indirect(indirect_buffer.as_bytes(), draw_count, stride)?;

        Ok(unsafe { self.draw_mesh_tasks_indirect_unchecked(indirect_buffer, draw_count, stride) })
    }

    fn validate_draw_mesh_tasks_indirect(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_mesh_tasks_indirect(
            indirect_buffer.buffer(),
            indirect_buffer.offset(),
            draw_count,
            stride,
        )?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DrawMeshTasksIndirect;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_mesh_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        if pipeline.mesh_is_nv() {
            return Err(Box::new(ValidationError {
                problem: "the currently bound graphics pipeline uses NV mesh shaders instead of \
                    EXT mesh shaders"
                    .into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectEXT-MeshEXT-07091"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_mesh_tasks_indirect_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DrawMeshTasksIndirectCommand]>,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());

        self.add_command(
            "draw_mesh_tasks_indirect",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.draw_mesh_tasks_indirect_unchecked(
                        indirect_buffer.buffer(),
                        indirect_buffer.offset(),
                        draw_count,
                        stride,
                    )
                };
            },
        );

        self
    }

    /// Perform multiple draw operations using a mesh shading graphics pipeline,
    /// reading the number of draw operations from a separate buffer.
    ///
    /// One draw is performed for each [`DrawMeshTasksIndirectCommand`] struct that is read from
    /// `indirect_buffer`. The number of draws to perform is read from `count_buffer`, or
    /// specified by `max_draw_count`, whichever is lower.
    /// This number is limited by the
    /// [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) limit.
    ///
    /// A mesh shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`](Self::bind_pipeline_graphics). Any resources used by the
    /// graphics pipeline, such as descriptor sets and dynamic state, must have been set
    /// beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements](crate::shader#safety) apply.
    /// - The [safety requirements for
    ///   `DrawMeshTasksIndirectCommand`](DrawMeshTasksIndirectCommand#safety) apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`](DeviceProperties::max_draw_indirect_count) device limit.
    /// - The count stored in `count_buffer` must fall within the range of `indirect_buffer`.
    pub unsafe fn draw_mesh_tasks_indirect_count(
        &mut self,
        indirect_buffer: Subbuffer<[DrawMeshTasksIndirectCommand]>,
        count_buffer: Subbuffer<u32>,
        max_draw_count: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        let stride = size_of::<DrawMeshTasksIndirectCommand>() as u32;
        self.validate_draw_mesh_tasks_indirect_count(
            indirect_buffer.as_bytes(),
            count_buffer.as_bytes(),
            max_draw_count,
            stride,
        )?;

        Ok(unsafe {
            self.draw_mesh_tasks_indirect_count_unchecked(
                indirect_buffer,
                count_buffer,
                max_draw_count,
                stride,
            )
        })
    }

    fn validate_draw_mesh_tasks_indirect_count(
        &self,
        indirect_buffer: &Subbuffer<[u8]>,
        count_buffer: &Subbuffer<[u8]>,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_draw_mesh_tasks_indirect_count(
            indirect_buffer.buffer(),
            indirect_buffer.offset(),
            count_buffer.buffer(),
            count_buffer.offset(),
            max_draw_count,
            stride,
        )?;

        let render_pass_state = self.builder_state.render_pass.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                problem: "a render pass instance is not active".into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-renderpass"],
                ..Default::default()
            })
        })?;

        let pipeline = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "no graphics pipeline is currently bound".into(),
                    vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-None-08606"],
                    ..Default::default()
                })
            })?;

        const VUID_TYPE: VUIDType = VUIDType::DrawMeshTasksIndirectCount;
        self.validate_pipeline_descriptor_sets(VUID_TYPE, pipeline.into())?;
        self.validate_pipeline_push_constants(VUID_TYPE, pipeline.layout())?;
        self.validate_pipeline_graphics_mesh_shading(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_dynamic_state(VUID_TYPE, pipeline)?;
        self.validate_pipeline_graphics_render_pass(VUID_TYPE, pipeline, render_pass_state)?;

        if pipeline.mesh_is_nv() {
            return Err(Box::new(ValidationError {
                problem: "the currently bound graphics pipeline uses NV mesh shaders instead of \
                    EXT mesh shaders"
                    .into(),
                vuids: &["VUID-vkCmdDrawMeshTasksIndirectCountEXT-MeshEXT-07100"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn draw_mesh_tasks_indirect_count_unchecked(
        &mut self,
        indirect_buffer: Subbuffer<[DrawMeshTasksIndirectCommand]>,
        count_buffer: Subbuffer<u32>,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        if let RenderPassStateType::BeginRendering(state) =
            &mut self.builder_state.render_pass.as_mut().unwrap().render_pass
        {
            state.pipeline_used = true;
        }

        let pipeline = self.builder_state.pipeline_graphics.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());
        self.add_indirect_buffer_resources(&mut used_resources, count_buffer.as_bytes());

        self.add_command(
            "draw_mesh_tasks_indirect_count",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.draw_mesh_tasks_indirect_count_unchecked(
                        indirect_buffer.buffer(),
                        indirect_buffer.offset(),
                        count_buffer.buffer(),
                        count_buffer.offset(),
                        max_draw_count,
                        stride,
                    )
                };
            },
        );

        self
    }

    /// Performs a single ray tracing operation using a ray tracing pipeline.
    ///
    /// A ray tracing pipeline must have been bound using [`bind_pipeline_ray_tracing`]. Any
    /// resources used by the ray tracing pipeline, such as descriptor sets, must have been set
    /// beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// [`bind_pipeline_ray_tracing`]: Self::bind_pipeline_ray_tracing
    /// [shader safety requirements]: vulkano::shader#safety
    pub unsafe fn trace_rays(
        &mut self,
        shader_binding_table_addresses: ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.inner
            .validate_trace_rays(&shader_binding_table_addresses, dimensions)?;

        Ok(unsafe { self.trace_rays_unchecked(shader_binding_table_addresses, dimensions) })
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn trace_rays_unchecked(
        &mut self,
        shader_binding_table_addresses: ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> &mut Self {
        let pipeline = self.builder_state.pipeline_ray_tracing.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());

        self.add_command("trace_rays", used_resources, move |out| {
            unsafe { out.trace_rays_unchecked(&shader_binding_table_addresses, dimensions) };
        });

        self
    }

    /// Performs a single ray tracing operation using a ray tracing pipeline. One ray tracing  
    /// operation is performed for the [`TraceRaysIndirectCommand`] struct that is read from  
    /// `indirect_buffer`.
    ///
    /// A ray tracing pipeline must have been bound using [`bind_pipeline_ray_tracing`]. Any
    /// resources used by the ray tracing pipeline, such as descriptor sets, must have been set
    /// beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `TraceRaysIndirectCommand`] apply.
    ///
    /// [`bind_pipeline_ray_tracing`]: Self::bind_pipeline_ray_tracing
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `TraceRaysIndirectCommand`]: TraceRaysIndirectCommand#safety
    pub unsafe fn trace_rays_indirect(
        &mut self,
        shader_binding_table_addresses: ShaderBindingTableAddresses,
        indirect_buffer: Subbuffer<[TraceRaysIndirectCommand]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_trace_rays_indirect(
            &shader_binding_table_addresses,
            indirect_buffer.as_bytes(),
        )?;

        Ok(unsafe {
            self.trace_rays_indirect_unchecked(shader_binding_table_addresses, indirect_buffer)
        })
    }

    fn validate_trace_rays_indirect(
        &self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        indirect_buffer: &Subbuffer<[u8]>,
    ) -> Result<(), Box<ValidationError>> {
        let buffer = indirect_buffer.buffer();
        let offset = indirect_buffer.offset();
        let indirect_device_address = buffer.device_address().get() + offset;

        self.inner.validate_trace_rays_indirect(
            shader_binding_table_addresses,
            indirect_device_address,
        )?;

        if !buffer.usage().intersects(BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "indirect_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-vkCmdTraceRaysIndirectKHR-indirectDeviceAddress-03633"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn trace_rays_indirect_unchecked(
        &mut self,
        shader_binding_table_addresses: ShaderBindingTableAddresses,
        indirect_buffer: Subbuffer<[TraceRaysIndirectCommand]>,
    ) -> &mut Self {
        let buffer = indirect_buffer.buffer();
        let offset = indirect_buffer.offset();
        let indirect_device_address = buffer.device_address().get() + offset;

        let pipeline = self.builder_state.pipeline_ray_tracing.as_ref().unwrap();

        let mut used_resources = Vec::new();
        self.add_descriptor_sets_resources(&mut used_resources, pipeline.into());
        self.add_indirect_buffer_resources(&mut used_resources, indirect_buffer.as_bytes());

        self.add_command("trace_rays_indirect", used_resources, move |out| {
            unsafe {
                out.trace_rays_indirect_unchecked(
                    &shader_binding_table_addresses,
                    indirect_device_address,
                )
            };
        });

        self
    }

    fn validate_pipeline_descriptor_sets(
        &self,
        vuid_type: VUIDType,
        pipeline: Pipeline<'_>,
    ) -> Result<(), Box<ValidationError>> {
        fn validate_resources<T>(
            vuid_type: VUIDType,
            set_num: u32,
            binding_num: u32,
            binding_reqs: &DescriptorBindingRequirements,
            elements: &[Option<T>],
            mut extra_check: impl FnMut(u32, u32, u32, &T) -> Result<(), Box<ValidationError>>,
        ) -> Result<(), Box<ValidationError>> {
            let elements_to_check = if let Some(descriptor_count) = binding_reqs.descriptor_count {
                // The shader has a fixed-sized array, so it will never access more than
                // the first `descriptor_count` elements.
                elements.get(..descriptor_count as usize).ok_or_else(|| {
                    // There are less than `descriptor_count` elements in `elements`
                    Box::new(ValidationError {
                        problem: format!(
                            "the currently bound pipeline accesses the resource bound to \
                            descriptor set {set_num}, binding {binding_num}, \
                            descriptor index {}, but no descriptor was written to the \
                            descriptor set currently bound to set {set_num}",
                            elements.len()
                        )
                        .into(),
                        vuids: vuids!(vuid_type, "None-02699"),
                        ..Default::default()
                    })
                })?
            } else {
                // The shader has a runtime-sized array, so any element could potentially
                // be accessed. We must check them all.
                elements
            };

            for (index, element) in elements_to_check.iter().enumerate() {
                let index = index as u32;

                let element = match element {
                    Some(x) => x,
                    None => {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound pipeline accesses the resource bound to \
                                descriptor set {set_num}, binding {binding_num}, \
                                descriptor index {index}, but no descriptor was written to the \
                                descriptor set currently bound to set {set_num}"
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-02699"),
                            ..Default::default()
                        }));
                    }
                };

                extra_check(set_num, binding_num, index, element)?;
            }

            Ok(())
        }

        if pipeline.num_used_descriptor_sets() == 0 {
            return Ok(());
        }

        let descriptor_set_state = self
            .builder_state
            .descriptor_sets
            .get(&pipeline.bind_point())
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "the currently bound pipeline accesses descriptor sets, but no \
                    descriptor sets were previously bound"
                        .into(),
                    vuids: vuids!(vuid_type, "None-02697"),
                    ..Default::default()
                })
            })?;

        if !pipeline.layout().is_compatible_with(
            &descriptor_set_state.pipeline_layout,
            pipeline.num_used_descriptor_sets(),
        ) {
            return Err(Box::new(ValidationError {
                problem: "the currently bound pipeline accesses descriptor sets, but the \
                    pipeline layouts that were used to bind the descriptor sets are \
                    not compatible with the pipeline layout of the currently bound pipeline"
                    .into(),
                vuids: vuids!(vuid_type, "None-02697"),
                ..Default::default()
            }));
        }

        for (&(set_num, binding_num), binding_reqs) in pipeline.descriptor_binding_requirements() {
            let layout_binding = pipeline.layout().set_layouts()[set_num as usize]
                .binding(binding_num)
                .unwrap();

            let check_buffer =
                |_set_num: u32,
                 _binding_num: u32,
                 _index: u32,
                 _buffer_info: &OwnedDescriptorBufferInfo| Ok(());

            let check_buffer_view =
                |set_num: u32,
                 binding_num: u32,
                 index: u32,
                 buffer_view: &Option<Arc<BufferView>>| {
                    let Some(buffer_view) = buffer_view else {
                        return Ok(());
                    };

                    for desc_reqs in binding_reqs
                        .descriptors
                        .get(&Some(index))
                        .into_iter()
                        .chain(binding_reqs.descriptors.get(&None))
                    {
                        if layout_binding.descriptor_type == DescriptorType::StorageTexelBuffer {
                            if binding_reqs.image_format.is_none()
                                && !desc_reqs.memory_write.is_empty()
                                && !buffer_view
                                    .format_features()
                                    .intersects(FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT)
                            {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline writes to the buffer view \
                                        bound to descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, without specifying a format, \
                                        but the format features of the buffer view's format do \
                                        not contain `FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT`"
                                    )
                                    .into(),
                                    vuids: vuids!(vuid_type, "OpTypeImage-06423"),
                                    ..Default::default()
                                }));
                            }

                            if binding_reqs.image_format.is_none()
                                && !desc_reqs.memory_read.is_empty()
                                && !buffer_view
                                    .format_features()
                                    .intersects(FormatFeatures::STORAGE_READ_WITHOUT_FORMAT)
                            {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline reads from the buffer view \
                                        bound to descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, without specifying a format, \
                                        but the format features of the buffer view's format do \
                                        not contain `FormatFeatures::STORAGE_READ_WITHOUT_FORMAT`"
                                    )
                                    .into(),
                                    vuids: vuids!(vuid_type, "OpTypeImage-06424"),
                                    ..Default::default()
                                }));
                            }
                        }
                    }

                    Ok(())
                };

            let check_image_view_common =
                |set_num: u32, binding_num: u32, index: u32, image_view: &Arc<ImageView>| {
                    for desc_reqs in binding_reqs
                        .descriptors
                        .get(&Some(index))
                        .into_iter()
                        .chain(binding_reqs.descriptors.get(&None))
                    {
                        if desc_reqs.storage_image_atomic
                            && !image_view
                                .format_features()
                                .intersects(FormatFeatures::STORAGE_IMAGE_ATOMIC)
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline performs atomic operations on \
                                    the image view bound to descriptor set {set_num}, \
                                    binding {binding_num}, descriptor index {index}, but \
                                    the format features of the image view's format do \
                                    not contain `FormatFeatures::STORAGE_IMAGE_ATOMIC`"
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-02691"),
                                ..Default::default()
                            }));
                        }

                        if layout_binding.descriptor_type == DescriptorType::StorageImage {
                            if binding_reqs.image_format.is_none()
                                && !desc_reqs.memory_write.is_empty()
                                && !image_view
                                    .format_features()
                                    .intersects(FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT)
                            {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline writes to the image view \
                                        bound to descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, without specifying a format, \
                                        but the format features of the image view's format do \
                                        not contain `FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT`"
                                    )
                                    .into(),
                                    vuids: vuids!(vuid_type, "OpTypeImage-06423"),
                                    ..Default::default()
                                }));
                            }

                            if binding_reqs.image_format.is_none()
                                && !desc_reqs.memory_read.is_empty()
                                && !image_view
                                    .format_features()
                                    .intersects(FormatFeatures::STORAGE_READ_WITHOUT_FORMAT)
                            {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline reads from the image view \
                                        bound to descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, without specifying a format, \
                                        but the format features of the image view's format do \
                                        not contain `FormatFeatures::STORAGE_READ_WITHOUT_FORMAT`"
                                    )
                                    .into(),
                                    vuids: vuids!(vuid_type, "OpTypeImage-06424"),
                                    ..Default::default()
                                }));
                            }
                        }
                    }

                    /*
                       Instruction/Sampler/Image View Validation
                       https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
                    */

                    // The SPIR-V Image Format is not compatible with the image view’s format.
                    if let Some(required_format) = binding_reqs.image_format {
                        let format = image_view.format();
                        if format != required_format {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline requires the image view \
                                    bound to descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index} to have a format of `{required_format:?}`, \
                                    but the actual format is `{format:?}`"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }

                    // Rules for viewType
                    if let Some(required_image_view_type) = binding_reqs.image_view_type {
                        let image_view_type = image_view.view_type();
                        if image_view_type != required_image_view_type {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline requires the image view \
                                    bound to descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index} to have a view type of `{required_image_view_type:?}`, \
                                    but the actual view type is `{image_view_type:?}`
                                    "
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "viewType-07752"),
                                ..Default::default()
                            }));
                        }
                    }

                    // - If the image was created with VkImageCreateInfo::samples equal to
                    //   VK_SAMPLE_COUNT_1_BIT, the instruction must have MS = 0.
                    // - If the image was created with VkImageCreateInfo::samples not equal to
                    //   VK_SAMPLE_COUNT_1_BIT, the instruction must have MS = 1.
                    if binding_reqs.image_multisampled
                        && image_view.image().samples() == SampleCount::Sample1
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound pipeline accesses the image view \
                                bound to descriptor set {set_num}, binding {binding_num}, \
                                descriptor index {index}, and the pipeline requires a \
                                multisampled image, but the image view has only one sample"
                            )
                            .into(),
                            // vuids?
                            ..Default::default()
                        }));
                    } else if !binding_reqs.image_multisampled
                        && image_view.image().samples() != SampleCount::Sample1
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound pipeline accesses the image view \
                                bound to descriptor set {set_num}, binding {binding_num}, \
                                descriptor index {index}, and the pipeline requires a non-\
                                multisampled image, but the image view has more than one sample"
                            )
                            .into(),
                            // vuids?
                            ..Default::default()
                        }));
                    }

                    // - If the Sampled Type of the OpTypeImage does not match the numeric format
                    //   of the image, as shown in the SPIR-V Sampled Type column of the
                    //   Interpretation of Numeric Format table.
                    // - If the signedness of any read or sample operation does not match the
                    //   signedness of the image’s format.
                    if let Some(shader_numeric_type) = binding_reqs.image_scalar_type {
                        let aspects = image_view.subresource_range().aspects;
                        let view_numeric_type = NumericType::from(
                            if aspects.intersects(
                                ImageAspects::COLOR
                                    | ImageAspects::PLANE_0
                                    | ImageAspects::PLANE_1
                                    | ImageAspects::PLANE_2,
                            ) {
                                image_view.format().numeric_format_color().unwrap()
                            } else if aspects.intersects(ImageAspects::DEPTH) {
                                image_view.format().numeric_format_depth().unwrap()
                            } else if aspects.intersects(ImageAspects::STENCIL) {
                                image_view.format().numeric_format_stencil().unwrap()
                            } else {
                                // Per `ImageViewBuilder::aspects` and
                                // VUID-VkDescriptorImageInfo-imageView-01976
                                unreachable!()
                            },
                        );

                        if shader_numeric_type != view_numeric_type {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the image view \
                                    bound to descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, and the pipeline requires an \
                                    image view whose format has a `{shader_numeric_type:?}` \
                                    numeric type, but the format of the image view has a \
                                    `{view_numeric_type:?}` numeric type"
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "format-07753"),
                                ..Default::default()
                            }));
                        }
                    }

                    Ok(())
                };

            let check_sampler_common =
                |set_num: u32, binding_num: u32, index: u32, sampler: &Arc<Sampler>| {
                    for desc_reqs in binding_reqs
                        .descriptors
                        .get(&Some(index))
                        .into_iter()
                        .chain(binding_reqs.descriptors.get(&None))
                    {
                        if desc_reqs.sampler_no_unnormalized_coordinates
                            && sampler.unnormalized_coordinates()
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the sampler bound to \
                                    descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, in a way that does not support \
                                    samplers with unnormalized coordinates, but \
                                    the sampler currently bound to that descriptor \
                                    uses unnormalized coordinates"
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-02703", "None-02704"),
                                ..Default::default()
                            }));
                        }

                        // - OpImageFetch, OpImageSparseFetch, OpImage*Gather, and
                        //   OpImageSparse*Gather must not be used with a sampler that enables
                        //   sampler Y′CBCR conversion.
                        // - The ConstOffset and Offset operands must not be used with a sampler
                        //   that enables sampler Y′CBCR conversion.
                        if desc_reqs.sampler_no_ycbcr_conversion
                            && sampler.sampler_ycbcr_conversion().is_some()
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the sampler bound to \
                                    descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, in a way that does not support \
                                    samplers with a sampler YCbCr conversion, but the sampler \
                                    currently bound to that descriptor has a \
                                    sampler YCbCr conversion"
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-06550", "ConstOffset-06551"),
                                ..Default::default()
                            }));
                        }

                        /*
                            Instruction/Sampler/Image View Validation
                            https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap16.html#textures-input-validation
                        */

                        if desc_reqs.sampler_compare && sampler.compare().is_none() {
                            // - The SPIR-V instruction is one of the OpImage*Dref* instructions
                            //   and the sampler compareEnable is VK_FALSE
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the sampler bound to \
                                    descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, in a way that requires a sampler \
                                    with a compare operation, but the sampler currently bound to \
                                    that descriptor does not have a compare operation"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        } else if !desc_reqs.sampler_compare && sampler.compare().is_some() {
                            // - The SPIR-V instruction is not one of the OpImage*Dref*
                            //   instructions and the sampler compareEnable is VK_TRUE
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound pipeline accesses the sampler bound to \
                                    descriptor set {set_num}, binding {binding_num}, \
                                    descriptor index {index}, in a way that requires a sampler \
                                    without a compare operation, but the sampler currently bound \
                                    to that descriptor has a compare operation"
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }

                    Ok(())
                };

            let check_sampler =
                |set_num: u32, binding_num: u32, index: u32, sampler: &Arc<Sampler>| {
                    check_sampler_common(set_num, binding_num, index, sampler)?;

                    for desc_reqs in binding_reqs
                        .descriptors
                        .get(&Some(index))
                        .into_iter()
                        .chain(binding_reqs.descriptors.get(&None))
                    {
                        // Check sampler-image compatibility. Only done for separate samplers;
                        // combined image samplers are checked when updating the descriptor set.
                        for id in &desc_reqs.sampler_with_images {
                            let DescriptorIdentifier {
                                set: iset_num,
                                binding: ibinding_num,
                                index: iindex,
                            } = id;

                            // If the image view isn't actually present in the resources, then just
                            // skip it. It will be caught later by check_resources.
                            let Some(set) = descriptor_set_state.descriptor_sets.get(iset_num)
                            else {
                                continue;
                            };
                            let resources = set.resources();
                            let Some(DescriptorBindingResources::Image(elements)) =
                                resources.binding(*ibinding_num)
                            else {
                                continue;
                            };
                            let Some(Some(OwnedDescriptorImageInfo {
                                sampler: _,
                                image_view: Some(image_view),
                                image_layout: _,
                            })) = elements.get(*iindex as usize)
                            else {
                                continue;
                            };

                            if let Err(error) = sampler.check_can_sample(image_view.as_ref()) {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound pipeline uses the sampler bound to \
                                        descriptor set {set_num}, binding {binding_num}, \
                                        descriptor index {index}, to sample the image bound to \
                                        descriptor set {iset_num}, binding {ibinding_num}, \
                                        descriptor index {iindex}, but the sampler is not \
                                        compatible with the image: {}",
                                        error,
                                    )
                                    .into(),
                                    // vuids?
                                    ..Default::default()
                                }));
                            }
                        }
                    }

                    Ok(())
                };

            let check_image =
                |set_num: u32,
                 binding_num: u32,
                 index: u32,
                 image_info: &OwnedDescriptorImageInfo| {
                    if let Some(sampler) = &image_info.sampler {
                        check_sampler(set_num, binding_num, index, sampler)?;
                    } else if let Some(&sampler) =
                        layout_binding.immutable_samplers.get(index as usize)
                    {
                        check_sampler(set_num, binding_num, index, sampler)?;
                    }

                    if let Some(image_view) = &image_info.image_view {
                        check_image_view_common(set_num, binding_num, index, image_view)?;
                    }

                    Ok(())
                };

            let check_acceleration_structure = |_set_num: u32,
                                                _binding_num: u32,
                                                _index: u32,
                                                _acceleration_structure: &Option<
                Arc<AccelerationStructure>,
            >| Ok(());

            let flags_skip_binding_validation =
                DescriptorBindingFlags::UPDATE_AFTER_BIND | DescriptorBindingFlags::PARTIALLY_BOUND;
            let requires_binding_validation =
                (layout_binding.binding_flags & flags_skip_binding_validation).is_empty();
            if requires_binding_validation {
                let set_resources = descriptor_set_state
                    .descriptor_sets
                    .get(&set_num)
                    .ok_or_else(|| {
                        Box::new(ValidationError {
                            problem: format!(
                                "the currently bound pipeline accesses descriptor set {set_num}, \
                                 but no descriptor set was previously bound"
                            )
                            .into(),
                            // vuids?
                            ..Default::default()
                        })
                    })?
                    .resources();

                let binding_resources = set_resources.binding(binding_num).unwrap();

                match binding_resources {
                    DescriptorBindingResources::Image(elements) => {
                        validate_resources(
                            vuid_type,
                            set_num,
                            binding_num,
                            binding_reqs,
                            elements,
                            check_image,
                        )?;
                    }
                    DescriptorBindingResources::Buffer(elements) => {
                        validate_resources(
                            vuid_type,
                            set_num,
                            binding_num,
                            binding_reqs,
                            elements,
                            check_buffer,
                        )?;
                    }
                    DescriptorBindingResources::BufferView(elements) => {
                        validate_resources(
                            vuid_type,
                            set_num,
                            binding_num,
                            binding_reqs,
                            elements,
                            check_buffer_view,
                        )?;
                    }
                    // Spec:
                    // Descriptor bindings with descriptor type of
                    // VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK can be undefined when the descriptor
                    // set is consumed; though values in that block will be undefined.
                    //
                    // TODO: We *may* still want to validate this?
                    DescriptorBindingResources::InlineUniformBlock => (),
                    DescriptorBindingResources::AccelerationStructure(elements) => {
                        validate_resources(
                            vuid_type,
                            set_num,
                            binding_num,
                            binding_reqs,
                            elements,
                            check_acceleration_structure,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_pipeline_push_constants(
        &self,
        vuid_type: VUIDType,
        pipeline_layout: &PipelineLayout,
    ) -> Result<(), Box<ValidationError>> {
        if pipeline_layout.push_constant_ranges().is_empty()
            || self.device().enabled_features().maintenance4
        {
            return Ok(());
        }

        let constants_pipeline_layout = self
            .builder_state
            .push_constants_pipeline_layout
            .as_ref()
            .ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "the currently bound pipeline accesses push constants, but no \
                    push constants were previously set"
                        .into(),
                    vuids: vuids!(vuid_type, "maintenance4-06425"),
                    ..Default::default()
                })
            })?;

        if pipeline_layout.handle() != constants_pipeline_layout.handle()
            && pipeline_layout.push_constant_ranges()
                != constants_pipeline_layout.push_constant_ranges()
        {
            return Err(Box::new(ValidationError {
                problem: "the currently bound pipeline accesses push constants, but the \
                    pipeline layouts that were used to set the push constants are \
                    not compatible with the pipeline layout of the currently bound pipeline"
                    .into(),
                vuids: vuids!(vuid_type, "maintenance4-06425"),
                ..Default::default()
            }));
        }

        let set_bytes = &self.builder_state.push_constants;

        if !pipeline_layout
            .push_constant_ranges()
            .iter()
            .all(|pc_range| set_bytes.contains(pc_range.offset..pc_range.offset + pc_range.size))
        {
            return Err(Box::new(ValidationError {
                problem: "the currently bound pipeline accesses push constants, but \
                    not all bytes in the push constant ranges of the pipeline layout of the \
                    currently bound pipeline have been set"
                    .into(),
                vuids: vuids!(vuid_type, "maintenance4-06425"),
                ..Default::default()
            }));
        }

        Ok(())
    }

    fn validate_pipeline_graphics_primitive_shading(
        &self,
        vuid_type: VUIDType,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), Box<ValidationError>> {
        if pipeline
            .shader_stages()
            .intersects(ShaderStages::MESH | ShaderStages::TASK)
        {
            return Err(Box::new(ValidationError {
                problem: "the currently bound graphics pipeline uses mesh shading".into(),
                vuids: vuids!(vuid_type, "stage-06481"),
                ..Default::default()
            }));
        }

        if self
            .builder_state
            .queries
            .contains_key(&QueryType::MeshPrimitivesGenerated)
        {
            return Err(Box::new(ValidationError {
                problem: "a `MeshPrimitivesGenerated` query is currently active".into(),
                vuids: vuids!(vuid_type, "stage-07073"),
                ..Default::default()
            }));
        }

        if let Some(query_state) = self
            .builder_state
            .queries
            .get(&QueryType::PipelineStatistics)
        {
            if query_state
                .query_pool
                .pipeline_statistics()
                .is_mesh_shading_graphics()
            {
                return Err(Box::new(ValidationError {
                    problem: "a `PipelineStatistics` query is currently active, and its \
                        pipeline statistics flags include statistics for mesh shading"
                        .into(),
                    vuids: vuids!(vuid_type, "stage-07073"),
                    ..Default::default()
                }));
            }
        }

        let vertex_input_state = if pipeline
            .dynamic_state()
            .contains(&DynamicState::VertexInput)
        {
            self.builder_state.vertex_input.as_ref().unwrap()
        } else {
            pipeline.vertex_input_state().unwrap()
        };

        for &binding_num in vertex_input_state.bindings.keys() {
            if !self.builder_state.vertex_buffers.contains_key(&binding_num) {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the currently bound graphics pipeline uses \
                        vertex buffer binding {0}, but \
                        no vertex buffer is currently bound to binding {0}",
                        binding_num
                    )
                    .into(),
                    vuids: vuids!(vuid_type, "None-04007"),
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    fn validate_pipeline_graphics_mesh_shading(
        &self,
        vuid_type: VUIDType,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), Box<ValidationError>> {
        if pipeline.shader_stages().intersects(
            ShaderStages::VERTEX
                | ShaderStages::TESSELLATION_CONTROL
                | ShaderStages::TESSELLATION_EVALUATION
                | ShaderStages::GEOMETRY,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the currently bound graphics pipeline uses primitive shading".into(),
                vuids: vuids!(vuid_type, "stage-06480"),
                ..Default::default()
            }));
        }

        if let Some(query_state) = self
            .builder_state
            .queries
            .get(&QueryType::PipelineStatistics)
        {
            if query_state
                .query_pool
                .pipeline_statistics()
                .is_primitive_shading_graphics()
            {
                return Err(Box::new(ValidationError {
                    problem: "a `PipelineStatistics` query is currently active, and its \
                        pipeline statistics flags include statistics for primitive shading"
                        .into(),
                    vuids: vuids!(vuid_type, "pipelineStatistics-07076"),
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    fn validate_pipeline_graphics_dynamic_state(
        &self,
        vuid_type: VUIDType,
        pipeline: &GraphicsPipeline,
    ) -> Result<(), Box<ValidationError>> {
        let device = pipeline.device();

        for dynamic_state in pipeline.dynamic_state().iter().copied() {
            match dynamic_state {
                DynamicState::BlendConstants => {
                    if self.builder_state.blend_constants.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07835"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::ColorWriteEnable => {
                    let enables = if let Some(enables) = &self.builder_state.color_write_enable {
                        enables
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07749"),
                            ..Default::default()
                        }));
                    };

                    if enables.len() < pipeline.color_blend_state().unwrap().attachments.len() {
                        return Err(Box::new(ValidationError {
                            problem: "the currently bound graphics pipeline requires the \
                                `DynamicState::ColorWriteEnable` dynamic state, but \
                                the number of enable values that were set is less than the number \
                                of color attachments in the color blend state of the \
                                graphics pipeline"
                                .into(),
                            vuids: vuids!(vuid_type, "attachmentCount-07750"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::CullMode => {
                    if self.builder_state.cull_mode.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07840"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthBias => {
                    if self.builder_state.depth_bias.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07834"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthBiasEnable => {
                    if self.builder_state.depth_bias_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-04877"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthBounds => {
                    if self.builder_state.depth_bounds.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07836"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthBoundsTestEnable => {
                    if self.builder_state.depth_bounds_test_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07846"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthCompareOp => {
                    if self.builder_state.depth_compare_op.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07845"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthTestEnable => {
                    if self.builder_state.depth_test_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07843"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DepthWriteEnable => {
                    if self.builder_state.depth_write_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07844"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::DiscardRectangle => {
                    for num in
                        0..pipeline.discard_rectangle_state().unwrap().rectangles.len() as u32
                    {
                        if !self.builder_state.discard_rectangle.contains_key(&num) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, but \
                                    this state was either not set for discard rectangle {1}, or \
                                    it was overwritten by a more recent \
                                    `bind_pipeline_graphics` command",
                                    dynamic_state, num,
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-07751"),
                                ..Default::default()
                            }));
                        }
                    }
                }
                // DynamicState::ExclusiveScissor => todo!(),
                DynamicState::FragmentShadingRate => {
                    if self.builder_state.fragment_shading_rate.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(
                                vuid_type,
                                "VUID-vkCmdDrawIndexed-pipelineFragmentShadingRate-09238"
                            ),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::FrontFace => {
                    if self.builder_state.front_face.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-0784"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::LineStipple => {
                    if self.builder_state.line_stipple.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07849"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::LineWidth => {
                    if self.builder_state.line_width.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07833"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::LogicOp => {
                    if self.builder_state.logic_op.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "logicOp-04878"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::PatchControlPoints => {
                    if self.builder_state.patch_control_points.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-04875"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::PrimitiveRestartEnable => {
                    let primitive_restart_enable =
                        if let Some(enable) = self.builder_state.primitive_restart_enable {
                            enable
                        } else {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, but \
                                    this state was either not set, or it was overwritten by a \
                                    more recent `bind_pipeline_graphics` command",
                                    dynamic_state
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-04879"),
                                ..Default::default()
                            }));
                        };

                    if primitive_restart_enable {
                        let topology = if pipeline
                            .dynamic_state()
                            .contains(&DynamicState::PrimitiveTopology)
                        {
                            if let Some(topology) = self.builder_state.primitive_topology {
                                topology
                            } else {
                                return Err(Box::new(ValidationError {
                                    problem: "the currently bound graphics pipeline requires \
                                        the `DynamicState::PrimitiveTopology` dynamic state, \
                                        but this state was either not set, or it was \
                                        overwritten by a more recent `bind_pipeline_graphics` \
                                        command"
                                        .into(),
                                    vuids: vuids!(vuid_type, "None-07842"),
                                    ..Default::default()
                                }));
                            }
                        } else if let Some(input_assembly_state) = pipeline.input_assembly_state() {
                            input_assembly_state.topology
                        } else {
                            unreachable!("PrimitiveRestartEnable can only occur on primitive shading pipelines")
                        };

                        match topology {
                            PrimitiveTopology::PointList
                            | PrimitiveTopology::LineList
                            | PrimitiveTopology::TriangleList
                            | PrimitiveTopology::LineListWithAdjacency
                            | PrimitiveTopology::TriangleListWithAdjacency => {
                                if !device.enabled_features().primitive_topology_list_restart {
                                    return Err(Box::new(ValidationError {
                                        problem: "primitive restart is enabled for the currently \
                                            bound graphics pipeline, but the currently set \
                                            dynamic primitive topology is \
                                            `PrimitiveTopology::*List`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature(
                                                "primitive_topology_list_restart",
                                            ),
                                        ])]),
                                        // vuids?
                                        ..Default::default()
                                    }));
                                }
                            }
                            PrimitiveTopology::PatchList => {
                                if !device
                                    .enabled_features()
                                    .primitive_topology_patch_list_restart
                                {
                                    return Err(Box::new(ValidationError {
                                        problem: "primitive restart is enabled for the currently \
                                            bound graphics pipeline, but the currently set \
                                            dynamic primitive topology is \
                                            `PrimitiveTopology::PatchList`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature(
                                                "primitive_topology_patch_list_restart",
                                            ),
                                        ])]),
                                        // vuids?
                                        ..Default::default()
                                    }));
                                }
                            }
                            _ => (),
                        }
                    }
                }
                DynamicState::PrimitiveTopology => {
                    let topology = if let Some(topology) = self.builder_state.primitive_topology {
                        topology
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07842"),
                            ..Default::default()
                        }));
                    };

                    if pipeline.shader_stages().intersects(
                        ShaderStages::TESSELLATION_CONTROL | ShaderStages::TESSELLATION_EVALUATION,
                    ) {
                        if !matches!(topology, PrimitiveTopology::PatchList) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, and the graphics pipeline \
                                    includes tessellation shader stages, but the dynamic \
                                    primitive topology is not `PrimitiveTopology::PatchList`",
                                    dynamic_state
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    } else {
                        if matches!(topology, PrimitiveTopology::PatchList) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, and the graphics pipeline \
                                    does not include tessellation shader stages, but the dynamic \
                                    primitive topology is `PrimitiveTopology::PatchList`",
                                    dynamic_state
                                )
                                .into(),
                                // vuids?
                                ..Default::default()
                            }));
                        }
                    }

                    let properties = device.physical_device().properties();

                    if !properties
                        .dynamic_primitive_topology_unrestricted
                        .unwrap_or(false)
                    {
                        if let Some(input_assembly_state) = pipeline.input_assembly_state() {
                            let is_same_topology_class = matches!(
                                (topology, input_assembly_state.topology),
                                (PrimitiveTopology::PointList, PrimitiveTopology::PointList)
                                    | (
                                        PrimitiveTopology::LineList
                                            | PrimitiveTopology::LineStrip
                                            | PrimitiveTopology::LineListWithAdjacency
                                            | PrimitiveTopology::LineStripWithAdjacency,
                                        PrimitiveTopology::LineList
                                            | PrimitiveTopology::LineStrip
                                            | PrimitiveTopology::LineListWithAdjacency
                                            | PrimitiveTopology::LineStripWithAdjacency,
                                    )
                                    | (
                                        PrimitiveTopology::TriangleList
                                            | PrimitiveTopology::TriangleStrip
                                            | PrimitiveTopology::TriangleFan
                                            | PrimitiveTopology::TriangleListWithAdjacency
                                            | PrimitiveTopology::TriangleStripWithAdjacency,
                                        PrimitiveTopology::TriangleList
                                            | PrimitiveTopology::TriangleStrip
                                            | PrimitiveTopology::TriangleFan
                                            | PrimitiveTopology::TriangleListWithAdjacency
                                            | PrimitiveTopology::TriangleStripWithAdjacency,
                                    )
                                    | (PrimitiveTopology::PatchList, PrimitiveTopology::PatchList)
                            );

                            if !is_same_topology_class {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound graphics pipeline requires the \
                                        `DynamicState::{:?}` dynamic state, and the \
                                        `dynamic_primitive_topology_unrestricted` device property is \
                                        `false`, but the dynamic primitive topology does not belong \
                                        to the same topology class as the topology that the \
                                        graphics pipeline was created with",
                                        dynamic_state
                                    )
                                    .into(),
                                    vuids: vuids!(
                                        vuid_type,
                                        "dynamicPrimitiveTopologyUnrestricted-07500"
                                    ),
                                    ..Default::default()
                                }));
                            }
                        }
                    }

                    // TODO: check that the topology matches the geometry shader
                }
                DynamicState::RasterizerDiscardEnable => {
                    if self.builder_state.rasterizer_discard_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-04876"),
                            ..Default::default()
                        }));
                    }
                }
                // DynamicState::RayTracingPipelineStackSize => unreachable!(
                //     "RayTracingPipelineStackSize dynamic state should not occur on a graphics \
                //     pipeline",
                // ),
                // DynamicState::SampleLocations => todo!(),
                DynamicState::Scissor => {
                    let viewport_state = pipeline.viewport_state().unwrap();

                    for num in 0..viewport_state.scissors.len() as u32 {
                        if !self.builder_state.scissor.contains_key(&num) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, but \
                                    this state was either not set, or it was overwritten by a \
                                    more recent `bind_pipeline_graphics` command",
                                    dynamic_state
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-07832"),
                                ..Default::default()
                            }));
                        }
                    }
                }
                DynamicState::ScissorWithCount => {
                    if let Some(scissors) = &self.builder_state.scissor_with_count {
                        let viewport_state = pipeline.viewport_state().unwrap();
                        let viewport_count = viewport_state.viewports.len() as u32;
                        let scissor_count = scissors.len() as u32;

                        if viewport_count != 0 {
                            // Check if the counts match, but only if the viewport count is fixed.
                            // If the viewport count is also dynamic, then the
                            // DynamicState::ViewportWithCount match arm will handle it.
                            if viewport_count != scissor_count {
                                return Err(Box::new(ValidationError {
                                    problem: "the currently bound graphics pipeline requires the \
                                        `DynamicState::ScissorWithCount` dynamic state, and \
                                        not the `DynamicState::ViewportWithCount` dynamic state, \
                                        but the dynamic scissor count is not equal to the \
                                        viewport count specified when creating the pipeline"
                                        .into(),
                                    vuids: vuids!(vuid_type, "scissorCount-03418"),
                                    ..Default::default()
                                }));
                            }
                        }
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "scissorCount-03418", "viewportCount-03419"),
                            ..Default::default()
                        }));
                    };
                }
                DynamicState::StencilCompareMask => {
                    let state = self.builder_state.stencil_compare_mask;

                    if state.front.is_none() || state.back.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07837"),
                            ..Default::default() //
                        }));
                    }
                }
                DynamicState::StencilOp => {
                    let state = self.builder_state.stencil_op;

                    if state.front.is_none() || state.back.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07848"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::StencilReference => {
                    let state = self.builder_state.stencil_reference;

                    if state.front.is_none() || state.back.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07839"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::StencilTestEnable => {
                    if self.builder_state.stencil_test_enable.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07847"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::StencilWriteMask => {
                    let state = self.builder_state.stencil_write_mask;

                    if state.front.is_none() || state.back.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07838"),
                            ..Default::default()
                        }));
                    }
                }
                DynamicState::VertexInput => {
                    if let Some(vertex_input_state) = &self.builder_state.vertex_input {
                        vertex_input_state
                            .validate_required_vertex_inputs(
                                pipeline.required_vertex_inputs().unwrap(),
                                RequiredVertexInputsVUIDs {
                                    not_present: vuids!(vuid_type, "Input-07939"),
                                    numeric_type: vuids!(vuid_type, "Input-08734"),
                                    requires32: vuids!(vuid_type, "format-08936"),
                                    requires64: vuids!(vuid_type, "format-08937"),
                                    requires_second_half: vuids!(vuid_type, "None-09203"),
                                },
                            )
                            .map_err(|mut err| {
                                err.problem = format!(
                                    "the currently bound graphics pipeline requires the \
                                `DynamicState::VertexInput` dynamic state, but \
                                the dynamic vertex input does not meet the requirements of the \
                                vertex shader in the pipeline: {}",
                                    err.problem,
                                )
                                .into();
                                err
                            })?;
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-04914"),
                            ..Default::default()
                        }));
                    }
                }
                // DynamicState::VertexInputBindingStride => todo!(),
                DynamicState::Viewport => {
                    let viewport_state = pipeline.viewport_state().unwrap();

                    for num in 0..viewport_state.viewports.len() as u32 {
                        if !self.builder_state.viewport.contains_key(&num) {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the currently bound graphics pipeline requires the \
                                    `DynamicState::{:?}` dynamic state, but \
                                    this state was either not set, or it was overwritten by a \
                                    more recent `bind_pipeline_graphics` command",
                                    dynamic_state
                                )
                                .into(),
                                vuids: vuids!(vuid_type, "None-07831"),
                                ..Default::default()
                            }));
                        }
                    }
                }
                // DynamicState::ViewportCoarseSampleOrder => todo!(),
                // DynamicState::ViewportShadingRatePalette => todo!(),
                DynamicState::ViewportWithCount => {
                    if let Some(viewports) = &self.builder_state.viewport_with_count {
                        let viewport_state = pipeline.viewport_state().unwrap();
                        let scissor_count = viewport_state.scissors.len() as u32;
                        let viewport_count = viewports.len() as u32;

                        if scissor_count != 0 {
                            if viewport_count != scissor_count {
                                return Err(Box::new(ValidationError {
                                    problem: "the currently bound graphics pipeline requires the \
                                        `DynamicState::ViewportWithCount` dynamic state, and \
                                        not the `DynamicState::ScissorWithCount` dynamic state, \
                                        but the dynamic scissor count is not equal to the scissor \
                                        count specified when creating the pipeline"
                                        .into(),
                                    vuids: vuids!(vuid_type, "viewportCount-03417"),
                                    ..Default::default()
                                }));
                            }
                        } else {
                            if let Some(scissors) = &self.builder_state.scissor_with_count {
                                if viewport_count != scissors.len() as u32 {
                                    return Err(Box::new(ValidationError {
                                        problem:
                                            "the currently bound graphics pipeline requires both \
                                            the `DynamicState::ViewportWithCount` and the \
                                            `DynamicState::ScissorWithCount` dynamic states, but \
                                            the dynamic scissor count is not equal to the \
                                            dynamic scissor count "
                                                .into(),
                                        vuids: vuids!(vuid_type, "viewportCount-03419"),
                                        ..Default::default()
                                    }));
                                }
                            } else {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "the currently bound graphics pipeline requires the \
                                        `DynamicState::{:?}` dynamic state, but \
                                        this state was either not set, or it was overwritten by a \
                                        more recent `bind_pipeline_graphics` command",
                                        dynamic_state
                                    )
                                    .into(),
                                    vuids: vuids!(
                                        vuid_type,
                                        "scissorCount-03418",
                                        "viewportCount-03419"
                                    ),
                                    ..Default::default()
                                }));
                            }
                        }
                    } else {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "viewportCount-03417", "viewportCount-03419"),
                            ..Default::default()
                        }));
                    }

                    // TODO: VUID-vkCmdDrawIndexed-primitiveFragmentShadingRateWithMultipleViewports-04552
                    // If the primitiveFragmentShadingRateWithMultipleViewports limit is not
                    // supported, the bound graphics pipeline was created with
                    // the VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT dynamic
                    // state enabled, and any of the shader stages of the bound
                    // graphics pipeline write to the PrimitiveShadingRateKHR
                    // built-in, then vkCmdSetViewportWithCountEXT must have been called in the
                    // current command buffer prior to this drawing command, and
                    // the viewportCount parameter of
                    // vkCmdSetViewportWithCountEXT must be 1
                }
                DynamicState::ConservativeRasterizationMode => {
                    if self.builder_state.conservative_rasterization_mode.is_none() {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07631"),
                            ..Default::default()
                        }));
                    }
                    // TODO: VUID-vkCmdDraw-conservativePointAndLineRasterization-07499
                }
                DynamicState::ExtraPrimitiveOverestimationSize => {
                    if self
                        .builder_state
                        .extra_primitive_overestimation_size
                        .is_none()
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "the currently bound graphics pipeline requires the \
                                `DynamicState::{:?}` dynamic state, but \
                                this state was either not set, or it was overwritten by a \
                                more recent `bind_pipeline_graphics` command",
                                dynamic_state
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "None-07632"),
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_pipeline_graphics_render_pass(
        &self,
        vuid_type: VUIDType,
        pipeline: &GraphicsPipeline,
        render_pass_state: &RenderPassState,
    ) -> Result<(), Box<ValidationError>> {
        if render_pass_state.contents != SubpassContents::Inline {
            return Err(Box::new(ValidationError {
                problem: "the contents of the current subpass instance is not \
                    `SubpassContents::Inline`"
                    .into(),
                // vuids?
                ..Default::default()
            }));
        }

        match (&render_pass_state.render_pass, pipeline.subpass()) {
            (
                RenderPassStateType::BeginRenderPass(state),
                PipelineSubpassType::BeginRenderPass(pipeline_subpass),
            ) => {
                if !pipeline_subpass
                    .render_pass()
                    .is_compatible_with(state.subpass.render_pass())
                {
                    return Err(Box::new(ValidationError {
                        problem: "the current render pass instance is not compatible with the \
                            render pass that the currently bound graphics pipeline was \
                            created with"
                            .into(),
                        vuids: vuids!(vuid_type, "renderPass-02684"),
                        ..Default::default()
                    }));
                }

                if pipeline_subpass.index() != state.subpass.index() {
                    return Err(Box::new(ValidationError {
                        problem: "the subpass index of the current render pass instance is not \
                            equal to the index of the subpass that the bound graphics pipeline was \
                            created with"
                            .into(),
                        vuids: vuids!(vuid_type, "subpass-02685"),
                        ..Default::default()
                    }));
                }
            }
            (
                RenderPassStateType::BeginRendering(_),
                PipelineSubpassType::BeginRendering(pipeline_rendering_info),
            ) => {
                if pipeline_rendering_info.view_mask
                    != render_pass_state.rendering_info.as_ref().view_mask
                {
                    return Err(Box::new(ValidationError {
                        problem: "the `view_mask` of the current render pass instance is not \
                            equal to the `view_mask` the bound graphics pipeline was created with"
                            .into(),
                        vuids: vuids!(vuid_type, "viewMask-06178"),
                        ..Default::default()
                    }));
                }

                if pipeline_rendering_info.color_attachment_formats.len()
                    != render_pass_state
                        .rendering_info
                        .as_ref()
                        .color_attachment_formats
                        .len()
                {
                    return Err(Box::new(ValidationError {
                        problem: "`color_attachments.len()` of the current render pass \
                            instance is not equal to `color_attachment_formats.len()` that the
                            currently bound graphics pipeline was created with"
                            .into(),
                        vuids: vuids!(vuid_type, "colorAttachmentCount-06179"),
                        ..Default::default()
                    }));
                }

                for (color_attachment_index, required_format, pipeline_format) in render_pass_state
                    .rendering_info
                    .as_ref()
                    .color_attachment_formats
                    .iter()
                    .zip(
                        pipeline_rendering_info
                            .color_attachment_formats
                            .iter()
                            .copied(),
                    )
                    .enumerate()
                    .filter_map(|(i, (r, p))| r.map(|r| (i as u32, r, p)))
                {
                    if Some(required_format) != pipeline_format {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`color_attachments[{0}].image_view.format()` of the current \
                                render pass instance is not equal to \
                                `color_attachment_formats[{0}]` that the currently bound \
                                graphics pipeline was created with",
                                color_attachment_index,
                            )
                            .into(),
                            vuids: vuids!(vuid_type, "colorAttachmentCount-06180"),
                            ..Default::default()
                        }));
                    }
                }

                if let Some((required_format, pipeline_format)) = render_pass_state
                    .rendering_info
                    .as_ref()
                    .depth_attachment_format
                    .map(|r| (r, pipeline_rendering_info.depth_attachment_format))
                {
                    if Some(required_format) != pipeline_format {
                        return Err(Box::new(ValidationError {
                            problem: "`depth_attachment.image_view.format()` of the current \
                                render pass instance is not equal to \
                                `depth_attachment_format` that the currently bound \
                                graphics pipeline was created with"
                                .into(),
                            vuids: vuids!(vuid_type, "pDepthAttachment-06181"),
                            ..Default::default()
                        }));
                    }
                }

                if let Some((required_format, pipeline_format)) = render_pass_state
                    .rendering_info
                    .as_ref()
                    .stencil_attachment_format
                    .map(|r| (r, pipeline_rendering_info.stencil_attachment_format))
                {
                    if Some(required_format) != pipeline_format {
                        return Err(Box::new(ValidationError {
                            problem: "`stencil_attachment.image_view.format()` of the current \
                                render pass instance is not equal to \
                                `stencil_attachment_format` that the currently bound \
                                graphics pipeline was created with"
                                .into(),
                            vuids: vuids!(vuid_type, "pStencilAttachment-06182"),
                            ..Default::default()
                        }));
                    }
                }

                // VUID-vkCmdDraw-imageView-06172
                // VUID-vkCmdDraw-imageView-06173
                // VUID-vkCmdDraw-imageView-06174
                // VUID-vkCmdDraw-imageView-06175
                // VUID-vkCmdDraw-imageView-06176
                // VUID-vkCmdDraw-imageView-06177
                // TODO:
            }
            (RenderPassStateType::BeginRenderPass(_), PipelineSubpassType::BeginRendering(_)) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was begun with \
                        `begin_render_pass`, but the currently bound graphics pipeline requires \
                        a render pass instance begun with `begin_rendering`"
                        .into(),
                    // vuids?
                    ..Default::default()
                }));
            }
            (RenderPassStateType::BeginRendering(_), PipelineSubpassType::BeginRenderPass(_)) => {
                return Err(Box::new(ValidationError {
                    problem: "the current render pass instance was begun with \
                        `begin_rendering`, but the currently bound graphics pipeline requires \
                        a render pass instance begun with `begin_render_pass`"
                        .into(),
                    // vuids?
                    ..Default::default()
                }));
            }
        }

        // VUID-vkCmdDraw-None-02686
        // TODO:

        Ok(())
    }

    fn add_descriptor_sets_resources(
        &self,
        used_resources: &mut Vec<(ResourceUseRef2, Resource)>,
        pipeline: Pipeline<'_>,
    ) {
        let descriptor_sets_state = match self
            .builder_state
            .descriptor_sets
            .get(&pipeline.bind_point())
        {
            Some(x) => x,
            None => return,
        };

        for (&(set, binding), binding_reqs) in pipeline.descriptor_binding_requirements() {
            let descriptor_type = descriptor_sets_state.pipeline_layout.set_layouts()[set as usize]
                .binding(binding)
                .unwrap()
                .descriptor_type;

            // TODO: Should input attachments be handled here or in attachment access?
            if descriptor_type == DescriptorType::InputAttachment {
                continue;
            }

            let default_image_layout = descriptor_type.default_image_layout();

            let use_iter = move |index: u32| {
                let (stages_read, stages_write) = [Some(index), None]
                    .into_iter()
                    .filter_map(|index| binding_reqs.descriptors.get(&index))
                    .fold(
                        (ShaderStages::empty(), ShaderStages::empty()),
                        |(stages_read, stages_write), desc_reqs| {
                            (
                                stages_read | desc_reqs.memory_read,
                                stages_write | desc_reqs.memory_write,
                            )
                        },
                    );
                let use_ref = ResourceUseRef2::from(ResourceInCommand::DescriptorSet {
                    set,
                    binding,
                    index,
                });
                let memory_access = PipelineStageAccess::iter_descriptor_stages(
                    descriptor_type,
                    stages_read,
                    stages_write,
                )
                .fold(PipelineStageAccessFlags::empty(), |total, val| {
                    total | val.into()
                });
                (use_ref, memory_access)
            };

            let descriptor_set_state = &descriptor_sets_state.descriptor_sets[&set];

            match descriptor_set_state.resources().binding(binding).unwrap() {
                DescriptorBindingResources::Image(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if let Some(OwnedDescriptorImageInfo {
                            sampler: _,
                            image_view: Some(image_view),
                            image_layout,
                        }) = element
                        {
                            let image_layout = if *image_layout == ImageLayout::Undefined {
                                default_image_layout
                            } else {
                                *image_layout
                            };

                            let (use_ref, memory_access) = use_iter(index as u32);

                            used_resources.push((
                                use_ref,
                                Resource::Image {
                                    image: image_view.image().clone(),
                                    subresource_range: *image_view.subresource_range(),
                                    memory_access,
                                    start_layout: image_layout,
                                    end_layout: image_layout,
                                },
                            ));
                        }
                    }
                }
                DescriptorBindingResources::Buffer(elements) => {
                    if matches!(
                        descriptor_type,
                        DescriptorType::UniformBufferDynamic | DescriptorType::StorageBufferDynamic
                    ) {
                        let dynamic_offsets = descriptor_set_state.dynamic_offsets();

                        for (index, element) in elements.iter().enumerate() {
                            if let Some(OwnedDescriptorBufferInfo {
                                buffer: Some(buffer),
                                offset,
                                range,
                            }) = element
                            {
                                let dynamic_offset = dynamic_offsets[index] as DeviceSize;
                                let (use_ref, memory_access) = use_iter(index as u32);

                                let range = dynamic_offset + offset
                                    ..dynamic_offset
                                        + range.map_or(buffer.size(), |range| offset + range);

                                used_resources.push((
                                    use_ref,
                                    Resource::Buffer {
                                        buffer: buffer.clone().into(),
                                        range,
                                        memory_access,
                                    },
                                ));
                            }
                        }
                    } else {
                        for (index, element) in elements.iter().enumerate() {
                            if let &Some(OwnedDescriptorBufferInfo {
                                buffer: Some(ref buffer),
                                offset,
                                range,
                            }) = element
                            {
                                let (use_ref, memory_access) = use_iter(index as u32);

                                used_resources.push((
                                    use_ref,
                                    Resource::Buffer {
                                        buffer: buffer.clone().into(),
                                        range: offset
                                            ..range.map_or(buffer.size(), |range| offset + range),
                                        memory_access,
                                    },
                                ));
                            }
                        }
                    }
                }
                DescriptorBindingResources::BufferView(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if let Some(Some(buffer_view)) = element {
                            let buffer = Subbuffer::new(buffer_view.buffer().clone());
                            let (use_ref, memory_access) = use_iter(index as u32);
                            let offset = buffer_view.offset();
                            let size = buffer_view.range().unwrap_or(buffer.size() - offset);

                            used_resources.push((
                                use_ref,
                                Resource::Buffer {
                                    buffer,
                                    range: offset..(offset + size),
                                    memory_access,
                                },
                            ));
                        }
                    }
                }
                DescriptorBindingResources::InlineUniformBlock => (),
                DescriptorBindingResources::AccelerationStructure(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if let Some(Some(acceleration_structure)) = element {
                            let buffer = acceleration_structure.buffer();
                            let offset = acceleration_structure.offset();
                            let size = acceleration_structure.size();
                            let (use_ref, memory_access) = use_iter(index as u32);

                            used_resources.push((
                                use_ref,
                                Resource::Buffer {
                                    buffer: buffer.clone().into(),
                                    range: offset..offset + size,
                                    memory_access,
                                },
                            ));
                        }
                    }
                }
            }
        }
    }

    fn add_vertex_buffers_resources(
        &self,
        used_resources: &mut Vec<(ResourceUseRef2, Resource)>,
        pipeline: &GraphicsPipeline,
    ) {
        let vertex_input_state = if pipeline
            .dynamic_state()
            .contains(&DynamicState::VertexInput)
        {
            self.builder_state.vertex_input.as_ref().unwrap()
        } else {
            pipeline.vertex_input_state().unwrap()
        };

        used_resources.extend(vertex_input_state.bindings.iter().map(|(&binding, _)| {
            let vertex_buffer = &self.builder_state.vertex_buffers[&binding];
            (
                ResourceInCommand::VertexBuffer { binding }.into(),
                Resource::Buffer {
                    buffer: vertex_buffer.clone(),
                    range: 0..vertex_buffer.size(), // TODO:
                    memory_access:
                        PipelineStageAccessFlags::VertexAttributeInput_VertexAttributeRead,
                },
            )
        }));
    }

    fn add_index_buffer_resources(&self, used_resources: &mut Vec<(ResourceUseRef2, Resource)>) {
        let index_buffer_bytes = self.builder_state.index_buffer.as_ref().unwrap().as_bytes();
        used_resources.push((
            ResourceInCommand::IndexBuffer.into(),
            Resource::Buffer {
                buffer: index_buffer_bytes.clone(),
                range: 0..index_buffer_bytes.size(), // TODO:
                memory_access: PipelineStageAccessFlags::IndexInput_IndexRead,
            },
        ));
    }

    fn add_indirect_buffer_resources(
        &self,
        used_resources: &mut Vec<(ResourceUseRef2, Resource)>,
        indirect_buffer: &Subbuffer<[u8]>,
    ) {
        used_resources.push((
            ResourceInCommand::IndirectBuffer.into(),
            Resource::Buffer {
                buffer: indirect_buffer.clone(),
                range: 0..indirect_buffer.size(), // TODO:
                memory_access: PipelineStageAccessFlags::DrawIndirect_IndirectCommandRead,
            },
        ));
    }
}

#[derive(Clone, Copy)]
enum VUIDType {
    Dispatch,
    DispatchIndirect,
    Draw,
    DrawIndirect,
    DrawIndirectCount,
    DrawIndexed,
    DrawIndexedIndirect,
    DrawIndexedIndirectCount,
    DrawMeshTasks,
    DrawMeshTasksIndirect,
    DrawMeshTasksIndirectCount,
}
