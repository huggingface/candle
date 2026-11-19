use crate::{App, RenderContext, ShaderLanguage};
use std::{slice, sync::Arc};
use vulkano::{
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};
use vulkano_taskgraph::{command_buffer::RecordingCommandBuffer, Task, TaskContext, TaskResult};

const EXPOSURE: f32 = 1.0;

pub struct TonemapTask {
    pipeline: Option<Arc<GraphicsPipeline>>,
}

impl TonemapTask {
    pub fn new(_app: &App) -> Self {
        TonemapTask { pipeline: None }
    }

    pub fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let vs = unsafe { vs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = match app.shader_language {
                ShaderLanguage::Glsl => unsafe { fs_glsl::load(&app.device) }
                    .unwrap()
                    .entry_point("main"),
                ShaderLanguage::Slang => unsafe { fs_slang::load(&app.device) }
                    .unwrap()
                    .entry_point("main"),
            }
            .unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];
            let layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

            GraphicsPipeline::new(
                &app.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    vertex_input_state: Some(&VertexInputState::default()),
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    viewport_state: Some(&ViewportState::default()),
                    rasterization_state: Some(&RasterizationState::default()),
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState::default()],
                        ..Default::default()
                    }),
                    dynamic_state: &[DynamicState::Viewport],
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        self.pipeline = Some(pipeline);
    }
}

impl Task for TonemapTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        cbf.set_viewport(0, slice::from_ref(&rcx.viewport));
        cbf.bind_pipeline(self.pipeline.as_ref().unwrap());
        cbf.push_constants(
            self.pipeline.as_ref().unwrap().layout(),
            0,
            // We use the GLSL shader's push constants regardless of which shader is used for
            // simplicity because we know that they have the same representation.
            &fs_glsl::PushConstants {
                sampler_id: rcx.bloom_sampler_id,
                texture_id: rcx.bloom_sampled_image_id,
                exposure: EXPOSURE,
            },
        );

        unsafe { cbf.draw(3, 1, 0, 0) };

        Ok(())
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            const vec2[3] POSITIONS = {
                vec2(-1.0, -1.0),
                vec2(-1.0,  3.0),
                vec2( 3.0, -1.0),
            };

            const vec2[3] TEX_COORDS = {
                vec2(0.0, 0.0),
                vec2(0.0, 2.0),
                vec2(2.0, 0.0),
            };

            layout(location = 0) out vec2 v_tex_coords;

            void main() {
                gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
                v_tex_coords = TEX_COORDS[gl_VertexIndex];
            }
        ",
    }
}

mod fs_glsl {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/tonemap.glsl",
    }
}

mod fs_slang {
    vulkano_shaders::shader! {
        ty: "fragment",
        lang: "slang",
        path: "shaders/tonemap.slang",
    }
}
