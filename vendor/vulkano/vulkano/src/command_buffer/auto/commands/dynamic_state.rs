use crate::{
    command_buffer::{sys::RecordingCommandBuffer, AutoCommandBufferBuilder},
    pipeline::{
        graphics::{
            color_blend::LogicOp,
            depth_stencil::{CompareOp, StencilFaces, StencilOp, StencilOps},
            fragment_shading_rate::{FragmentShadingRateCombinerOp, FragmentShadingRateState},
            input_assembly::PrimitiveTopology,
            rasterization::{
                ConservativeRasterizationMode, CullMode, DepthBiasState, FrontFace, LineStipple,
            },
            vertex_input::VertexInputState,
            viewport::{Scissor, Viewport},
        },
        DynamicState,
    },
    ValidationError,
};
use ash::vk;
use smallvec::SmallVec;
use std::ops::RangeInclusive;

/// # Commands to set dynamic state for pipelines.
///
/// These commands require a queue with a pipeline type that uses the given state.
impl<L> AutoCommandBufferBuilder<L> {
    // Helper function for dynamic state setting.
    fn validate_graphics_pipeline_fixed_state(
        &self,
        state: DynamicState,
    ) -> Result<(), Box<ValidationError>> {
        if self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .is_some_and(|pipeline| pipeline.fixed_state().contains(&state))
        {
            return Err(Box::new(ValidationError {
                problem: "the state for this value in the currently bound graphics pipeline \
                    is fixed, and cannot be set"
                    .into(),
                vuids: &["VUID-vkCmdDispatch-None-08608", "VUID-vkCmdDraw-None-08608"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    /// Sets the dynamic blend constants for future draw calls.
    pub fn set_blend_constants(
        &mut self,
        constants: [f32; 4],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_blend_constants(constants)?;

        Ok(unsafe { self.set_blend_constants_unchecked(constants) })
    }

    fn validate_set_blend_constants(
        &self,
        constants: [f32; 4],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_blend_constants(constants)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::BlendConstants)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_blend_constants_unchecked(&mut self, constants: [f32; 4]) -> &mut Self {
        self.builder_state.blend_constants = Some(constants);
        self.add_command(
            "set_blend_constants",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_blend_constants_unchecked(constants) };
            },
        );

        self
    }

    /// Sets whether dynamic color writes should be enabled for each attachment in the
    /// framebuffer.
    pub fn set_color_write_enable(
        &mut self,
        enables: SmallVec<[bool; 4]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_color_write_enable(&enables)?;

        Ok(unsafe { self.set_color_write_enable_unchecked(enables) })
    }

    fn validate_set_color_write_enable(
        &self,
        enables: &[bool],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_color_write_enable(enables)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::ColorWriteEnable)?;

        if let Some(color_blend_state) = self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .and_then(|pipeline| pipeline.color_blend_state())
        {
            // Indirectly checked
            if enables.len() != color_blend_state.attachments.len() {
                return Err(Box::new(ValidationError {
                    problem: "the length of `enables` does not match the number of \
                        color attachments in the subpass of the currently bound graphics pipeline"
                        .into(),
                    vuids: &["VUID-vkCmdSetColorWriteEnableEXT-attachmentCount-06656"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_color_write_enable_unchecked(
        &mut self,
        enables: SmallVec<[bool; 4]>,
    ) -> &mut Self {
        self.builder_state.color_write_enable = Some(enables.clone());
        self.add_command(
            "set_color_write_enable",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_color_write_enable_unchecked(&enables) };
            },
        );

        self
    }

    /// Sets the dynamic cull mode for future draw calls.
    pub fn set_cull_mode(
        &mut self,
        cull_mode: CullMode,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_cull_mode(cull_mode)?;

        Ok(unsafe { self.set_cull_mode_unchecked(cull_mode) })
    }

    fn validate_set_cull_mode(&self, cull_mode: CullMode) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_cull_mode(cull_mode)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::CullMode)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_cull_mode_unchecked(&mut self, cull_mode: CullMode) -> &mut Self {
        self.builder_state.cull_mode = Some(cull_mode);
        self.add_command(
            "set_cull_mode",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_cull_mode_unchecked(cull_mode) };
            },
        );

        self
    }

    /// Sets the dynamic depth bias values for future draw calls.
    pub fn set_depth_bias(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bias(constant_factor, clamp, slope_factor)?;

        Ok(unsafe { self.set_depth_bias_unchecked(constant_factor, clamp, slope_factor) })
    }

    fn validate_set_depth_bias(
        &self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_depth_bias(constant_factor, clamp, slope_factor)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthBias)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bias_unchecked(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        self.builder_state.depth_bias = Some(DepthBiasState {
            constant_factor,
            clamp,
            slope_factor,
        });
        self.add_command(
            "set_depth_bias",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_depth_bias_unchecked(constant_factor, clamp, slope_factor) };
            },
        );

        self
    }

    /// Sets whether dynamic depth bias is enabled for future draw calls.
    pub fn set_depth_bias_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bias_enable(enable)?;

        Ok(unsafe { self.set_depth_bias_enable_unchecked(enable) })
    }

    fn validate_set_depth_bias_enable(&self, enable: bool) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_bias_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthBiasEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bias_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_bias_enable = Some(enable);
        self.add_command(
            "set_depth_bias_enable",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_depth_bias_enable_unchecked(enable) };
            },
        );

        self
    }

    /// Sets the dynamic depth bounds for future draw calls.
    pub fn set_depth_bounds(
        &mut self,
        bounds: RangeInclusive<f32>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bounds(bounds.clone())?;

        Ok(unsafe { self.set_depth_bounds_unchecked(bounds) })
    }

    fn validate_set_depth_bounds(
        &self,
        bounds: RangeInclusive<f32>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_depth_bounds(*bounds.start(), *bounds.end())?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthBounds)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bounds_unchecked(&mut self, bounds: RangeInclusive<f32>) -> &mut Self {
        self.builder_state.depth_bounds = Some(bounds.clone());
        self.add_command(
            "set_depth_bounds",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_depth_bounds_unchecked(*bounds.start(), *bounds.end()) };
            },
        );

        self
    }

    /// Sets whether dynamic depth bounds testing is enabled for future draw calls.
    pub fn set_depth_bounds_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_bounds_test_enable(enable)?;

        Ok(unsafe { self.set_depth_bounds_test_enable_unchecked(enable) })
    }

    fn validate_set_depth_bounds_test_enable(
        &self,
        enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_bounds_test_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthBoundsTestEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bounds_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_bounds_test_enable = Some(enable);
        self.add_command(
            "set_depth_bounds_test_enable",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_depth_bounds_test_enable_unchecked(enable) };
            },
        );

        self
    }

    /// Sets the dynamic depth compare op for future draw calls.
    pub fn set_depth_compare_op(
        &mut self,
        compare_op: CompareOp,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_compare_op(compare_op)?;

        Ok(unsafe { self.set_depth_compare_op_unchecked(compare_op) })
    }

    fn validate_set_depth_compare_op(
        &self,
        compare_op: CompareOp,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_compare_op(compare_op)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthCompareOp)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_compare_op_unchecked(&mut self, compare_op: CompareOp) -> &mut Self {
        self.builder_state.depth_compare_op = Some(compare_op);
        self.add_command(
            "set_depth_compare_op",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_depth_compare_op_unchecked(compare_op) };
            },
        );

        self
    }

    /// Sets whether dynamic depth testing is enabled for future draw calls.
    pub fn set_depth_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_test_enable(enable)?;

        Ok(unsafe { self.set_depth_test_enable_unchecked(enable) })
    }

    fn validate_set_depth_test_enable(&self, enable: bool) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_test_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthTestEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_test_enable = Some(enable);
        self.add_command(
            "set_depth_test_enable",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_depth_test_enable_unchecked(enable) };
            },
        );

        self
    }

    /// Sets whether dynamic depth write is enabled for future draw calls.
    pub fn set_depth_write_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_depth_write_enable(enable)?;

        Ok(unsafe { self.set_depth_write_enable_unchecked(enable) })
    }

    fn validate_set_depth_write_enable(&self, enable: bool) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_depth_write_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DepthWriteEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_write_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_write_enable = Some(enable);
        self.add_command(
            "set_depth_write_enable",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_depth_write_enable_unchecked(enable) };
            },
        );

        self
    }

    /// Sets the dynamic discard rectangles for future draw calls.
    pub fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: SmallVec<[Scissor; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_discard_rectangle(first_rectangle, &rectangles)?;

        Ok(unsafe { self.set_discard_rectangle_unchecked(first_rectangle, rectangles) })
    }

    fn validate_set_discard_rectangle(
        &self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_discard_rectangle(first_rectangle, rectangles)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::DiscardRectangle)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_discard_rectangle_unchecked(
        &mut self,
        first_rectangle: u32,
        rectangles: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        for (num, rectangle) in rectangles.iter().enumerate() {
            let num = num as u32 + first_rectangle;
            self.builder_state.discard_rectangle.insert(num, *rectangle);
        }

        self.add_command(
            "set_discard_rectangle",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_discard_rectangle_unchecked(first_rectangle, &rectangles) };
            },
        );

        self
    }

    /// Sets the dynamic front face for future draw calls.
    pub fn set_front_face(&mut self, face: FrontFace) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_front_face(face)?;

        Ok(unsafe { self.set_front_face_unchecked(face) })
    }

    fn validate_set_front_face(&self, face: FrontFace) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_front_face(face)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::FrontFace)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_front_face_unchecked(&mut self, face: FrontFace) -> &mut Self {
        self.builder_state.front_face = Some(face);
        self.add_command(
            "set_front_face",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_front_face_unchecked(face) };
            },
        );

        self
    }

    /// Sets the dynamic line stipple values for future draw calls.
    pub fn set_line_stipple(
        &mut self,
        factor: u32,
        pattern: u16,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_line_stipple(factor, pattern)?;

        Ok(unsafe { self.set_line_stipple_unchecked(factor, pattern) })
    }

    fn validate_set_line_stipple(
        &self,
        factor: u32,
        pattern: u16,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_line_stipple(factor, pattern)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::LineStipple)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_line_stipple_unchecked(&mut self, factor: u32, pattern: u16) -> &mut Self {
        self.builder_state.line_stipple = Some(LineStipple { factor, pattern });
        self.add_command(
            "set_line_stipple",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_line_stipple_unchecked(factor, pattern) };
            },
        );

        self
    }

    /// Sets the dynamic line width for future draw calls.
    pub fn set_line_width(&mut self, line_width: f32) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_line_width(line_width)?;

        Ok(unsafe { self.set_line_width_unchecked(line_width) })
    }

    fn validate_set_line_width(&self, line_width: f32) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_line_width(line_width)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::LineWidth)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_line_width_unchecked(&mut self, line_width: f32) -> &mut Self {
        self.builder_state.line_width = Some(line_width);
        self.add_command(
            "set_line_width",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_line_width_unchecked(line_width) };
            },
        );

        self
    }

    /// Sets the dynamic logic op for future draw calls.
    pub fn set_logic_op(&mut self, logic_op: LogicOp) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_logic_op(logic_op)?;

        Ok(unsafe { self.set_logic_op_unchecked(logic_op) })
    }

    fn validate_set_logic_op(&self, logic_op: LogicOp) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_logic_op(logic_op)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::LogicOp)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_logic_op_unchecked(&mut self, logic_op: LogicOp) -> &mut Self {
        self.builder_state.logic_op = Some(logic_op);
        self.add_command(
            "set_logic_op",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_logic_op_unchecked(logic_op) };
            },
        );

        self
    }

    /// Sets the dynamic number of patch control points for future draw calls.
    pub fn set_patch_control_points(
        &mut self,
        num: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_patch_control_points(num)?;

        Ok(unsafe { self.set_patch_control_points_unchecked(num) })
    }

    fn validate_set_patch_control_points(&self, num: u32) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_patch_control_points(num)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::PatchControlPoints)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_patch_control_points_unchecked(&mut self, num: u32) -> &mut Self {
        self.builder_state.patch_control_points = Some(num);
        self.add_command(
            "set_patch_control_points",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_patch_control_points_unchecked(num) };
            },
        );

        self
    }

    /// Sets whether dynamic primitive restart is enabled for future draw calls.
    pub fn set_primitive_restart_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_primitive_restart_enable(enable)?;

        Ok(unsafe { self.set_primitive_restart_enable_unchecked(enable) })
    }

    fn validate_set_primitive_restart_enable(
        &self,
        enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_primitive_restart_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::PrimitiveRestartEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_primitive_restart_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.primitive_restart_enable = Some(enable);
        self.add_command(
            "set_primitive_restart_enable",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_primitive_restart_enable_unchecked(enable) };
            },
        );

        self
    }

    /// Sets the dynamic primitive topology for future draw calls.
    pub fn set_primitive_topology(
        &mut self,
        topology: PrimitiveTopology,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_primitive_topology(topology)?;

        Ok(unsafe { self.set_primitive_topology_unchecked(topology) })
    }

    fn validate_set_primitive_topology(
        &self,
        topology: PrimitiveTopology,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_primitive_topology(topology)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::PrimitiveTopology)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_primitive_topology_unchecked(
        &mut self,
        topology: PrimitiveTopology,
    ) -> &mut Self {
        self.builder_state.primitive_topology = Some(topology);
        self.add_command(
            "set_primitive_topology",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_primitive_topology_unchecked(topology) };
            },
        );

        self
    }

    /// Sets whether dynamic rasterizer discard is enabled for future draw calls.
    pub fn set_rasterizer_discard_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_rasterizer_discard_enable(enable)?;

        Ok(unsafe { self.set_rasterizer_discard_enable_unchecked(enable) })
    }

    fn validate_set_rasterizer_discard_enable(
        &self,
        enable: bool,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_rasterizer_discard_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::RasterizerDiscardEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_rasterizer_discard_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.rasterizer_discard_enable = Some(enable);
        self.add_command(
            "set_rasterizer_discard_enable",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_rasterizer_discard_enable_unchecked(enable) };
            },
        );

        self
    }

    /// Sets the dynamic scissors for future draw calls.
    pub fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_scissor(first_scissor, &scissors)?;

        Ok(unsafe { self.set_scissor_unchecked(first_scissor, scissors) })
    }

    fn validate_set_scissor(
        &self,
        first_scissor: u32,
        scissors: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_scissor(first_scissor, scissors)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::Scissor)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_scissor_unchecked(
        &mut self,
        first_scissor: u32,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();

        for (num, scissor) in scissors.iter().enumerate() {
            let num = num as u32 + first_scissor;
            self.builder_state.scissor.insert(num, *scissor);
        }

        self.add_command(
            "set_scissor",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_scissor_unchecked(first_scissor, &scissors) };
            },
        );

        self
    }

    /// Sets the dynamic scissors with count for future draw calls.
    pub fn set_scissor_with_count(
        &mut self,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_scissor_with_count(&scissors)?;

        Ok(unsafe { self.set_scissor_with_count_unchecked(scissors) })
    }

    fn validate_set_scissor_with_count(
        &self,
        scissors: &[Scissor],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_scissor_with_count(scissors)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::ScissorWithCount)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_scissor_with_count_unchecked(
        &mut self,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        self.builder_state.scissor_with_count = Some(scissors.clone());
        self.add_command(
            "set_scissor_with_count",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_scissor_with_count_unchecked(&scissors) };
            },
        );

        self
    }

    /// Sets the dynamic stencil compare mask on one or both faces for future draw calls.
    pub fn set_stencil_compare_mask(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_compare_mask(faces, compare_mask)?;

        Ok(unsafe { self.set_stencil_compare_mask_unchecked(faces, compare_mask) })
    }

    fn validate_set_stencil_compare_mask(
        &self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_stencil_compare_mask(faces, compare_mask)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilCompareMask)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_compare_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> &mut Self {
        let faces_vk = vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_compare_mask.front = Some(compare_mask);
        }

        if faces_vk.intersects(vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_compare_mask.back = Some(compare_mask);
        }

        self.add_command(
            "set_stencil_compare_mask",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_stencil_compare_mask_unchecked(faces, compare_mask) };
            },
        );

        self
    }

    /// Sets the dynamic stencil ops on one or both faces for future draw calls.
    pub fn set_stencil_op(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op)?;

        Ok(unsafe {
            self.set_stencil_op_unchecked(faces, fail_op, pass_op, depth_fail_op, compare_op)
        })
    }

    fn validate_set_stencil_op(
        &self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilOp)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_op_unchecked(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> &mut Self {
        let faces_vk = vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_op.front = Some(StencilOps {
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            });
        }

        if faces_vk.intersects(vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_op.back = Some(StencilOps {
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            });
        }

        self.add_command(
            "set_stencil_op",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.set_stencil_op_unchecked(faces, fail_op, pass_op, depth_fail_op, compare_op)
                };
            },
        );

        self
    }

    /// Sets the dynamic stencil reference on one or both faces for future draw calls.
    pub fn set_stencil_reference(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_reference(faces, reference)?;

        Ok(unsafe { self.set_stencil_reference_unchecked(faces, reference) })
    }

    fn validate_set_stencil_reference(
        &self,
        faces: StencilFaces,
        reference: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_stencil_reference(faces, reference)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilReference)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_reference_unchecked(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> &mut Self {
        let faces_vk = vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_reference.front = Some(reference);
        }

        if faces_vk.intersects(vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_reference.back = Some(reference);
        }

        self.add_command(
            "set_stencil_reference",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_stencil_reference_unchecked(faces, reference) };
            },
        );

        self
    }

    /// Sets whether dynamic stencil testing is enabled for future draw calls.
    pub fn set_stencil_test_enable(
        &mut self,
        enable: bool,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_test_enable(enable)?;

        Ok(unsafe { self.set_stencil_test_enable_unchecked(enable) })
    }

    fn validate_set_stencil_test_enable(&self, enable: bool) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_stencil_test_enable(enable)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilTestEnable)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.stencil_test_enable = Some(enable);
        self.add_command(
            "set_stencil_test_enable",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_stencil_test_enable_unchecked(enable) };
            },
        );

        self
    }

    /// Sets the dynamic stencil write mask on one or both faces for future draw calls.
    pub fn set_stencil_write_mask(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_stencil_write_mask(faces, write_mask)?;

        Ok(unsafe { self.set_stencil_write_mask_unchecked(faces, write_mask) })
    }

    fn validate_set_stencil_write_mask(
        &self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_stencil_write_mask(faces, write_mask)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::StencilWriteMask)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_write_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> &mut Self {
        let faces_vk = vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_write_mask.front = Some(write_mask);
        }

        if faces_vk.intersects(vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_write_mask.back = Some(write_mask);
        }

        self.add_command(
            "set_stencil_write_mask",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_stencil_write_mask_unchecked(faces, write_mask) };
            },
        );

        self
    }

    /// Sets the dynamic vertex input for future draw calls.
    #[inline]
    pub fn set_vertex_input(
        &mut self,
        vertex_input_state: VertexInputState,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_vertex_input(&vertex_input_state)?;

        Ok(unsafe { self.set_vertex_input_unchecked(vertex_input_state) })
    }

    fn validate_set_vertex_input(
        &self,
        vertex_input_state: &VertexInputState,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_vertex_input(vertex_input_state)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::VertexInput)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_vertex_input_unchecked(
        &mut self,
        vertex_input_state: VertexInputState,
    ) -> &mut Self {
        self.builder_state.vertex_input = Some(vertex_input_state.clone());

        self.add_command(
            "set_vertex_input",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_vertex_input_unchecked(&vertex_input_state) };
            },
        );

        self
    }

    /// Sets the dynamic viewports for future draw calls.
    pub fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_viewport(first_viewport, &viewports)?;

        Ok(unsafe { self.set_viewport_unchecked(first_viewport, viewports) })
    }

    fn validate_set_viewport(
        &self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_viewport(first_viewport, viewports)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::Viewport)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_viewport_unchecked(
        &mut self,
        first_viewport: u32,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> &mut Self {
        for (num, viewport) in viewports.iter().enumerate() {
            let num = num as u32 + first_viewport;
            self.builder_state.viewport.insert(num, viewport.clone());
        }

        self.add_command(
            "set_viewport",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_viewport_unchecked(first_viewport, &viewports) };
            },
        );

        self
    }

    /// Sets the dynamic viewports with count for future draw calls.
    pub fn set_viewport_with_count(
        &mut self,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_viewport_with_count(&viewports)?;

        Ok(unsafe { self.set_viewport_with_count_unchecked(viewports) })
    }

    fn validate_set_viewport_with_count(
        &self,
        viewports: &[Viewport],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_viewport_with_count(viewports)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::ViewportWithCount)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_viewport_with_count_unchecked(
        &mut self,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> &mut Self {
        self.builder_state.viewport_with_count = Some(viewports.clone());
        self.add_command(
            "set_viewport",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_viewport_with_count_unchecked(&viewports) };
            },
        );

        self
    }

    /// Sets the dynamic conservative rasterization mode for future draw calls.
    #[inline]
    pub fn set_conservative_rasterization_mode(
        &mut self,
        conservative_rasterization_mode: ConservativeRasterizationMode,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_conservative_rasterization_mode()?;

        Ok(unsafe {
            self.set_conservative_rasterization_mode_unchecked(conservative_rasterization_mode)
        })
    }

    fn validate_set_conservative_rasterization_mode(&self) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_conservative_rasterization_mode()?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::ConservativeRasterizationMode)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_conservative_rasterization_mode_unchecked(
        &mut self,
        conservative_rasterization_mode: ConservativeRasterizationMode,
    ) -> &mut Self {
        self.builder_state.conservative_rasterization_mode = Some(conservative_rasterization_mode);

        self.add_command(
            "set_conservative_rasterization_mode",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.set_conservative_rasterization_mode_unchecked(
                        conservative_rasterization_mode,
                    )
                };
            },
        );

        self
    }

    /// Sets the dynamic extra primitive overestimation size for future draw calls.
    #[inline]
    pub fn set_extra_primitive_overestimation_size(
        &mut self,
        extra_primitive_overestimation_size: f32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_extra_primitive_overestimation_size()?;

        Ok(unsafe {
            self.set_extra_primitive_overestimation_size_unchecked(
                extra_primitive_overestimation_size,
            )
        })
    }

    fn validate_set_extra_primitive_overestimation_size(&self) -> Result<(), Box<ValidationError>> {
        self.inner.validate_set_conservative_rasterization_mode()?;

        self.validate_graphics_pipeline_fixed_state(
            DynamicState::ExtraPrimitiveOverestimationSize,
        )?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_extra_primitive_overestimation_size_unchecked(
        &mut self,
        extra_primitive_overestimation_size: f32,
    ) -> &mut Self {
        self.builder_state.extra_primitive_overestimation_size =
            Some(extra_primitive_overestimation_size);

        self.add_command(
            "set_extra_primitive_overestimation_size",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.set_extra_primitive_overestimation_size_unchecked(
                        extra_primitive_overestimation_size,
                    )
                };
            },
        );

        self
    }

    /// Sets the dynamic fragment shading rate for future draw calls.
    #[inline]
    pub fn set_fragment_shading_rate(
        &mut self,
        fragment_size: [u32; 2],
        combiner_ops: [FragmentShadingRateCombinerOp; 2],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_fragment_shading_rate(fragment_size, combiner_ops)?;

        Ok(unsafe { self.set_fragment_shading_rate_unchecked(fragment_size, combiner_ops) })
    }

    fn validate_set_fragment_shading_rate(
        &self,
        fragment_size: [u32; 2],
        combiner_ops: [FragmentShadingRateCombinerOp; 2],
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_set_fragment_shading_rate(fragment_size, combiner_ops)?;

        self.validate_graphics_pipeline_fixed_state(DynamicState::FragmentShadingRate)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_fragment_shading_rate_unchecked(
        &mut self,
        fragment_size: [u32; 2],
        combiner_ops: [FragmentShadingRateCombinerOp; 2],
    ) -> &mut Self {
        self.builder_state.fragment_shading_rate = Some(FragmentShadingRateState {
            fragment_size,
            combiner_ops,
            ..FragmentShadingRateState::default()
        });

        self.add_command(
            "set_fragment_shading_rate",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.set_fragment_shading_rate_unchecked(fragment_size, combiner_ops) };
            },
        );

        self
    }
}
