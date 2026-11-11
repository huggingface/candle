use crate::{App, RenderContext};
use std::{slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    image::Image,
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer, resource::HostAccessType, ClearValues, Id, Task,
    TaskContext, TaskResult,
};

pub struct SceneTask {
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer_id: Id<Buffer>,
    diffuse_image_id: Id<Image>,
    normals_image_id: Id<Image>,
    depth_image_id: Id<Image>,
}

impl SceneTask {
    pub fn new(
        app: &App,
        virtual_diffuse_image_id: Id<Image>,
        virtual_normals_image_id: Id<Image>,
        virtual_depth_image_id: Id<Image>,
    ) -> Self {
        let vertices = [
            TriangleVertex {
                position: [-0.5, -0.25],
            },
            TriangleVertex {
                position: [0.0, 0.5],
            },
            TriangleVertex {
                position: [0.25, -0.1],
            },
        ];
        let vertex_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(vertices.as_slice()).unwrap(),
            )
            .unwrap();

        // FIXME(taskgraph): sane initialization
        app.resources.flight(app.flight_id).wait(None).unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &app.queue,
                &app.resources,
                app.flight_id,
                |_cbf, tcx| {
                    tcx.write_buffer::<[TriangleVertex]>(vertex_buffer_id, ..)
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        SceneTask {
            pipeline: None,
            vertex_buffer_id,
            diffuse_image_id: virtual_diffuse_image_id,
            normals_image_id: virtual_normals_image_id,
            depth_image_id: virtual_depth_image_id,
        }
    }

    pub fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let vs = unsafe { vs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { fs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = TriangleVertex::per_vertex().definition(&vs).unwrap();
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
                    vertex_input_state: Some(&vertex_input_state),
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    viewport_state: Some(&ViewportState::default()),
                    rasterization_state: Some(&RasterizationState::default()),
                    depth_stencil_state: Some(&DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &vec![
                            ColorBlendAttachmentState::default();
                            subpass.num_color_attachments() as usize
                        ],
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

impl Task for SceneTask {
    type World = RenderContext;

    fn clear_values(&self, clear_values: &mut ClearValues<'_>, _world: &Self::World) {
        clear_values.set(self.diffuse_image_id, [0.0; 4]);
        clear_values.set(self.normals_image_id, [0.0; 4]);
        clear_values.set(self.depth_image_id, 1.0);
    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        cbf.set_viewport(0, slice::from_ref(&rcx.viewport));
        cbf.bind_pipeline(self.pipeline.as_ref().unwrap());
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]);

        unsafe { cbf.draw(3, 1, 0, 0) };

        Ok(())
    }
}

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct TriangleVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) out vec4 f_color;
            layout(location = 1) out vec4 f_normal;

            void main() {
                f_color = vec4(1.0, 1.0, 1.0, 1.0);
                f_normal = vec4(0.0, 0.0, 1.0, 0.0);
            }
        ",
    }
}
