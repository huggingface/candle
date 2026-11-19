use crate::{App, RenderContext};
use glam::{Mat4, Vec3};
use std::{slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
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
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer, resource::HostAccessType, ClearValues, Id, Task,
    TaskContext, TaskResult,
};

pub struct DeferredTask {
    swapchain_id: Id<Swapchain>,
    ambient_lighting_pipeline: AmbientLightingPipeline,
    directional_lighting_pipeline: DirectionalLightingPipeline,
    point_lighting_pipeline: PointLightingPipeline,
}

impl DeferredTask {
    pub fn new(app: &App, virtual_swapchain_id: Id<Swapchain>) -> Self {
        DeferredTask {
            swapchain_id: virtual_swapchain_id,
            ambient_lighting_pipeline: AmbientLightingPipeline::new(app),
            directional_lighting_pipeline: DirectionalLightingPipeline::new(app),
            point_lighting_pipeline: PointLightingPipeline::new(app),
        }
    }

    pub fn create_pipelines(&mut self, app: &App, subpass: &Subpass) {
        self.ambient_lighting_pipeline.create_pipeline(app, subpass);
        self.directional_lighting_pipeline
            .create_pipeline(app, subpass);
        self.point_lighting_pipeline.create_pipeline(app, subpass);
    }
}

impl Task for DeferredTask {
    type World = RenderContext;

    fn clear_values(&self, clear_values: &mut ClearValues<'_>, _world: &Self::World) {
        clear_values.set(self.swapchain_id.current_image_id(), [0.0; 4]);
    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        cbf.set_viewport(0, slice::from_ref(&rcx.viewport));

        let world_to_screen = Mat4::IDENTITY;
        let screen_to_world = world_to_screen.inverse();

        self.ambient_lighting_pipeline.draw(cbf, [0.1, 0.1, 0.1])?;
        self.directional_lighting_pipeline.draw(
            cbf,
            Vec3::new(0.2, -0.1, -0.7),
            [0.6, 0.6, 0.6],
        )?;
        self.point_lighting_pipeline.draw(
            cbf,
            screen_to_world,
            Vec3::new(0.5, -0.5, -0.1),
            [1.0, 0.0, 0.0],
        )?;
        self.point_lighting_pipeline.draw(
            cbf,
            screen_to_world,
            Vec3::new(-0.9, 0.2, -0.15),
            [0.0, 1.0, 0.0],
        )?;
        self.point_lighting_pipeline.draw(
            cbf,
            screen_to_world,
            Vec3::new(0.0, 0.5, -0.05),
            [0.0, 0.0, 1.0],
        )?;

        Ok(())
    }
}

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct LightingVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

/// Allows applying an ambient lighting to a scene.
struct AmbientLightingPipeline {
    vertex_buffer_id: Id<Buffer>,
    pipeline: Option<Arc<GraphicsPipeline>>,
}

impl AmbientLightingPipeline {
    fn new(app: &App) -> Self {
        let vertices = [
            LightingVertex {
                position: [-1.0, -1.0],
            },
            LightingVertex {
                position: [-1.0, 3.0],
            },
            LightingVertex {
                position: [3.0, -1.0],
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
                    tcx.write_buffer::<[LightingVertex]>(vertex_buffer_id, ..)
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        AmbientLightingPipeline {
            vertex_buffer_id,
            pipeline: None,
        }
    }

    fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let vs = unsafe { ambient_lighting_vs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { ambient_lighting_fs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = LightingVertex::per_vertex().definition(&vs).unwrap();
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
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                            }),
                            ..Default::default()
                        }],
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

    /// Records a draw command that applies ambient lighting.
    ///
    /// This draw will read the diffuse image, multiply it with `ambient_color` and write the
    /// output to the current swapchain image with additive blending (in other words the value will
    /// be added to the existing value in the swapchain image, and not replace the existing value).
    ///
    /// The viewport must have been set beforehand.
    ///
    /// # Arguments
    ///
    /// - `cbf` is the command buffer to record to.
    /// - `viewport_dimensions` contains the dimensions of the current framebuffer.
    /// - `color_input` is an image containing the albedo of each object of the scene. It is the
    ///   result of the deferred pass.
    /// - `ambient_color` is the color to apply.
    unsafe fn draw(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        ambient_color: [f32; 3],
    ) -> TaskResult {
        cbf.bind_pipeline(self.pipeline.as_ref().unwrap());
        cbf.push_constants(
            self.pipeline.as_ref().unwrap().layout(),
            0,
            &ambient_lighting_fs::PushConstants {
                color: [ambient_color[0], ambient_color[1], ambient_color[2], 1.0],
            },
        );
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]);

        unsafe { cbf.draw(3, 1, 0, 0) };

        Ok(())
    }
}

mod ambient_lighting_vs {
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

mod ambient_lighting_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            #define VKO_INPUT_ATTACHMENT_ENABLED 1
            #include <vulkano.glsl>

            #define u_diffuse vko_subpassInput(0)

            layout(push_constant) uniform PushConstants {
                // The `ambient_color` parameter of the `draw` method.
                vec4 color;
            } push_constants;

            layout(location = 0) out vec4 f_color;

            void main() {
                // Load the value at the current pixel.
                vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
                f_color.rgb = push_constants.color.rgb * in_diffuse;
                f_color.a = 1.0;
            }
        ",
    }
}

/// Allows applying a directional light source to a scene.
struct DirectionalLightingPipeline {
    vertex_buffer_id: Id<Buffer>,
    pipeline: Option<Arc<GraphicsPipeline>>,
}

impl DirectionalLightingPipeline {
    fn new(app: &App) -> Self {
        let vertices = [
            LightingVertex {
                position: [-1.0, -1.0],
            },
            LightingVertex {
                position: [-1.0, 3.0],
            },
            LightingVertex {
                position: [3.0, -1.0],
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
                    tcx.write_buffer::<[LightingVertex]>(vertex_buffer_id, ..)
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        DirectionalLightingPipeline {
            vertex_buffer_id,
            pipeline: None,
        }
    }

    fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let vs = unsafe { directional_lighting_vs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { directional_lighting_fs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = LightingVertex::per_vertex().definition(&vs).unwrap();
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
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                            }),
                            ..Default::default()
                        }],
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

    /// Records a draw command that applies directional lighting.
    ///
    /// This draw will read the diffuse image and normals image, and multiply the color with
    /// `color` and the dot product of the `direction` with the normal. It then writes the output
    /// to the current swapchain image with additive blending (in other words the value will be
    /// added to the existing value in the swapchain image, and not replace the existing value).
    ///
    /// Since the normals image contains normals in world coordinates, `direction` should also be
    /// in world coordinates.
    ///
    /// The viewport must have been set beforehand.
    ///
    /// # Arguments
    ///
    /// - `cbf` is the command buffer to record to.
    /// - `direction` is the direction of the light in world coordinates.
    /// - `color` is the color to apply.
    unsafe fn draw(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        direction: Vec3,
        color: [f32; 3],
    ) -> TaskResult {
        cbf.bind_pipeline(self.pipeline.as_ref().unwrap());
        cbf.push_constants(
            self.pipeline.as_ref().unwrap().layout(),
            0,
            &directional_lighting_fs::PushConstants {
                color: [color[0], color[1], color[2], 1.0],
                direction: direction.extend(0.0).into(),
            },
        );
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]);

        unsafe { cbf.draw(3, 1, 0, 0) };

        Ok(())
    }
}

mod directional_lighting_vs {
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

mod directional_lighting_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            #define VKO_INPUT_ATTACHMENT_ENABLED 1
            #include <vulkano.glsl>

            #define u_diffuse vko_subpassInput(0)
            #define u_normals vko_subpassInput(1)

            layout(push_constant) uniform PushConstants {
                // The `color` parameter of the `draw` method.
                vec4 color;
                // The `direction` parameter of the `draw` method.
                vec4 direction;
            } push_constants;

            layout(location = 0) out vec4 f_color;

            void main() {
                vec3 in_normal = normalize(subpassLoad(u_normals).rgb);

                // If the normal is perpendicular to the direction of the lighting, then 
                // `light_percent` will be 0. If the normal is parallel to the direction of the 
                // lighting, then `light_percent` will be 1. Any other angle will yield an 
                // intermediate value.
                float light_percent = -dot(push_constants.direction.xyz, in_normal);
                // `light_percent` must not go below 0.0. There's no such thing as negative
                // lighting.
                light_percent = max(light_percent, 0.0);

                vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
                f_color.rgb = light_percent * push_constants.color.rgb * in_diffuse;
                f_color.a = 1.0;
            }
        ",
    }
}

/// Allows applying a point light to a scene.
struct PointLightingPipeline {
    vertex_buffer_id: Id<Buffer>,
    pipeline: Option<Arc<GraphicsPipeline>>,
}

impl PointLightingPipeline {
    fn new(app: &App) -> Self {
        let vertices = [
            LightingVertex {
                position: [-1.0, -1.0],
            },
            LightingVertex {
                position: [-1.0, 3.0],
            },
            LightingVertex {
                position: [3.0, -1.0],
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
                    tcx.write_buffer::<[LightingVertex]>(vertex_buffer_id, ..)
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        PointLightingPipeline {
            vertex_buffer_id,
            pipeline: None,
        }
    }

    fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let vs = unsafe { point_lighting_vs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { point_lighting_fs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = LightingVertex::per_vertex().definition(&vs).unwrap();
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
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                            }),
                            ..Default::default()
                        }],
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

    /// Records a draw command that applies a point lighting.
    ///
    /// This draw will read the depth image and rebuild the world position of the pixel currently
    /// being processed (modulo rounding errors). It will then compare this position with
    /// `position`, and process the lighting based on the distance and orientation (similar to the
    /// directional lighting pipeline).
    ///
    /// It then writes the output to the current swapchain image with additive blending (in other
    /// words the value will be added to the existing value in the swapchain image, and not replace
    /// the existing value).
    ///
    /// Note that in a real-world application, you probably want to pass additional parameters
    /// such as some way to indicate the distance at which the lighting decrease. In this example
    /// this value is hardcoded in the shader.
    ///
    /// The viewport must have been set beforehand.
    ///
    /// # Arguments
    ///
    /// - `cbf` is the command buffer to record to.
    /// - `screen_to_world` is a matrix that turns coordinates from framebuffer space into world
    ///   space. This matrix is used alongside with `depth_input` to determine the world
    ///   coordinates of each pixel being processed.
    /// - `position` is the position of the spot light in world coordinates.
    /// - `color` is the color of the light.
    unsafe fn draw(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        screen_to_world: Mat4,
        position: Vec3,
        color: [f32; 3],
    ) -> TaskResult {
        cbf.bind_pipeline(self.pipeline.as_ref().unwrap());
        cbf.push_constants(
            self.pipeline.as_ref().unwrap().layout(),
            0,
            &point_lighting_fs::PushConstants {
                screen_to_world: screen_to_world.to_cols_array_2d(),
                color: [color[0], color[1], color[2], 1.0],
                position: position.extend(0.0).into(),
            },
        );
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]);

        unsafe { cbf.draw(3, 1, 0, 0) };

        Ok(())
    }
}

mod point_lighting_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;
            layout(location = 0) out vec2 v_screen_coords;

            void main() {
                v_screen_coords = position;
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod point_lighting_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            #define VKO_INPUT_ATTACHMENT_ENABLED 1
            #include <vulkano.glsl>

            #define u_diffuse vko_subpassInput(0)
            #define u_normals vko_subpassInput(1)
            #define u_depth vko_subpassInput(2)

            layout(push_constant) uniform PushConstants {
                // The `screen_to_world` parameter of the `draw` method.
                mat4 screen_to_world;
                // The `color` parameter of the `draw` method.
                vec4 color;
                // The `position` parameter of the `draw` method.
                vec4 position;
            } push_constants;

            layout(location = 0) in vec2 v_screen_coords;
            layout(location = 0) out vec4 f_color;

            void main() {
                float in_depth = subpassLoad(u_depth).x;

                // Any depth superior or equal to 1.0 means that the pixel has been untouched by 
                // the deferred pass. We don't want to deal with them.
                if (in_depth >= 1.0) {
                    discard;
                }

                // Find the world coordinates of the current pixel.
                vec4 world = push_constants.screen_to_world * vec4(v_screen_coords, in_depth, 1.0);
                world /= world.w;

                vec3 in_normal = normalize(subpassLoad(u_normals).rgb);
                vec3 light_direction = normalize(push_constants.position.xyz - world.xyz);

                // Calculate the percent of lighting that is received based on the orientation of 
                // the normal and the direction of the light.
                float light_percent = max(-dot(light_direction, in_normal), 0.0);

                float light_distance = length(push_constants.position.xyz - world.xyz);
                // Further decrease light_percent based on the distance with the light position.
                light_percent *= 1.0 / exp(light_distance);

                vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
                f_color.rgb = push_constants.color.rgb * light_percent * in_diffuse;
                f_color.a = 1.0;
            }
        ",
    }
}
