// Welcome to the triangle-util example!
//
// This is almost exactly the same as the triangle example, except that it uses utility functions
// to make life easier.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

use std::{error::Error, sync::Arc, time::Duration};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::VulkanoWindows,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
}

impl App {
    fn new(_event_loop: &EventLoop<()>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig::default());

        // Manages any windows and their rendering.
        let windows = VulkanoWindows::default();

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            context.device().physical_device().properties().device_name,
            context.device().physical_device().properties().device_type,
        );

        // Before we can start creating and recording command buffers, we need a way of allocating
        // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command
        // pools underneath and provides a safe interface for them.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device(),
            &Default::default(),
        ));

        // We now create a buffer that will store the shape of our triangle.
        let vertices = [
            MyVertex {
                position: [-0.5, -0.25],
            },
            MyVertex {
                position: [0.0, 0.5],
            },
            MyVertex {
                position: [0.25, -0.1],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            context.memory_allocator(),
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        App {
            context,
            windows,
            command_buffer_allocator,
            vertex_buffer,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(primary_window_id) = self.windows.primary_window_id() {
            self.windows.remove_renderer(primary_window_id);
        }

        self.windows
            .create_window(event_loop, &self.context, &Default::default(), |_| {});
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let window_size = window_renderer.window().inner_size();

        // The next step is to create the shaders.
        //
        // The raw shader creation API provided by the vulkano library is unsafe for various
        // reasons, so The `shader!` macro provides a way to generate a Rust module from GLSL
        // source - in the example below, the source is provided as a string input directly to the
        // shader, but a path to a source file can be provided as well. Note that the user must
        // specify the type of shader (e.g. "vertex", "fragment", etc.) using the `ty` option of
        // the macro.
        //
        // The items generated by the `shader!` macro include a `load` function which loads the
        // shader using an input logical device. The module also includes type definitions for
        // layout structures defined in the shader source, for example uniforms and push constants.
        //
        // A more detailed overview of what the `shader!` macro generates can be found in the
        // vulkano-shaders crate docs. You can view them at https://docs.rs/vulkano-shaders/
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

                    void main() {
                        f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                ",
            }
        }

        // The next step is to create a *render pass*, which is an object that describes where the
        // output of the graphics pipeline will go. It describes the layout of the images where the
        // colors, depth and/or stencil information will be written.
        let render_pass = vulkano::single_pass_renderpass!(
            self.context.device(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `format: <ty>` indicates the type of the format of the image. This has to be
                    // one of the types of the `vulkano::format` module (or alternatively one of
                    // your structs that implements the `FormatDesc` trait). Here we use the same
                    // format as the swapchain.
                    format: window_renderer.swapchain_format(),
                    // `samples: 1` means that we ask the GPU to use one sample to determine the
                    // value of each pixel in the color attachment. We could use a larger value
                    // (multisampling) for antialiasing. An example of this can be found in
                    // msaa-renderpass.rs.
                    samples: 1,
                    // `load_op: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load_op: Clear,
                    // `store_op: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store_op: Store,
                },
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {},
            },
        )
        .unwrap();

        // The render pass we created above only describes the layout of our framebuffers. Before
        // we can draw we also need to create the actual framebuffers.
        //
        // Since we need to draw to multiple images, we are going to create a different framebuffer
        // for each image.
        let framebuffers =
            window_size_dependent_setup(window_renderer.swapchain_image_views(), &render_pass);

        // Before we draw, we have to create what is called a **pipeline**. A pipeline describes
        // how a GPU operation is to be performed. It is similar to an OpenGL program, but it also
        // contains many settings for customization, all baked into a single object. For drawing,
        // we create a **graphics** pipeline, but there are also other types of pipeline.
        let pipeline = {
            // First, we load the shaders that the pipeline will use: the vertex shader and the
            // fragment shader.
            //
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.
            let vs = unsafe { vs::load(self.context.device()) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { fs::load(self.context.device()) }
                .unwrap()
                .entry_point("main")
                .unwrap();

            // Automatically generate a vertex input state from the vertex shader's input
            // interface, that takes a single vertex buffer containing `Vertex` structs.
            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];

            // We must now create a **pipeline layout** object, which describes the locations and
            // types of descriptor sets and push constants used by the shaders in the pipeline.
            //
            // Multiple pipelines can share a common layout object, which is more efficient. The
            // shaders in a pipeline must use a subset of the resources described in its pipeline
            // layout, but the pipeline layout is allowed to contain resources that are not present
            // in the shaders; they can be used by shaders in other pipelines that share the same
            // layout. Thus, it is a good idea to design shaders so that many pipelines have common
            // resource locations, which allows them to share pipeline layouts.
            //
            // Since we only have one pipeline in this example, and thus one pipeline layout, we
            // automatically generate the layout from the resources used in the shaders. In a real
            // application, you would specify this information manually so that you can re-use one
            // layout in multiple pipelines.
            let layout = PipelineLayout::from_stages(self.context.device(), &stages).unwrap();

            // We have to indicate which subpass of which render pass this pipeline is going to be
            // used in. The pipeline will only be usable from this particular subpass.
            let subpass = Subpass::new(&render_pass, 0).unwrap();

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                self.context.device(),
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(&vertex_input_state),
                    // How vertices are arranged into primitive shapes. The default primitive shape
                    // is a triangle.
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    // How primitives are transformed and clipped to fit the framebuffer. We use a
                    // resizable viewport, set to draw over the entire window.
                    viewport_state: Some(&ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels. The default
                    // value does not perform any culling.
                    rasterization_state: Some(&RasterizationState::default()),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(&MultisampleState::default()),
                    // How pixel values are combined with the values already present in the
                    // framebuffer. The default value overwrites the old value with the new one,
                    // without any blending.
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState::default()],
                        ..Default::default()
                    }),
                    // Dynamic states allows us to specify parts of the pipeline settings when
                    // recording the command buffer, before we perform drawing. Here, we specify
                    // that the viewport should be dynamic.
                    dynamic_state: &[DynamicState::Viewport],
                    subpass: Some((&subpass).into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        // In the `window_event` handler below we are going to submit commands to the GPU.
        // Submitting a command produces an object that implements the `GpuFuture` trait, which
        // holds the resources for as long as they are in use by the GPU.

        self.rcx = Some(RenderContext {
            render_pass,
            framebuffers,
            pipeline,
            viewport,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                window_renderer.resize();
            }
            WindowEvent::RedrawRequested => {
                let window_size = window_renderer.window().inner_size();

                // Do not draw the frame when the screen size is zero. On Windows, this can
                // occur when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // Begin rendering by acquiring the gpu future from the window renderer.
                let previous_frame_end = window_renderer
                    .acquire(Some(Duration::from_millis(1000)), |swapchain_images| {
                        // Whenever the window resizes we need to recreate everything dependent
                        // on the window size. In this example that
                        // includes the swapchain, the framebuffers
                        // and the dynamic state viewport.
                        rcx.framebuffers =
                            window_size_dependent_setup(swapchain_images, &rcx.render_pass);
                        rcx.viewport.extent = window_size.into();
                    })
                    .unwrap();

                // In order to draw, we have to record a *command buffer*. The command buffer
                // object holds the list of commands that are going to be executed.
                //
                // Recording a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.context.graphics_queue().queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Before we can draw, we have to *enter a render pass*.
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            // A list of values to clear the attachments with. This list contains
                            // one item for each attachment in the render pass. In this case, there
                            // is only one attachment, and we clear it with a blue color.
                            //
                            // Only attachments that have `AttachmentLoadOp::Clear` are provided
                            // with clear values, any others should use `None` as the clear value.
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[window_renderer.image_index() as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            // The contents of the first (and only) subpass. This can be either
                            // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more
                            // advanced and is not covered here.
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    // We are now inside the first subpass of the render pass.
                    //
                    // TODO: Document state setting and how it affects subsequent draw commands.
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap();

                // We add a draw command.
                unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

                builder
                    // We leave the render pass. Note that if we had multiple subpasses we could
                    // have called `next_subpass` to jump to the next subpass.
                    .end_render_pass(Default::default())
                    .unwrap();

                // Finish recording the command buffer by calling `end`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .then_execute(self.context.graphics_queue().clone(), command_buffer)
                    .unwrap()
                    .boxed();

                // The color output is now expected to contain our triangle. But in order to show
                // it on the screen, we have to *present* the image by calling `present` on the
                // window renderer.
                //
                // This function does not actually present the image immediately. Instead it
                // submits a present command at the end of the queue. This means that it will only
                // be presented once the GPU has finished executing the command buffer that draws
                // the triangle.
                window_renderer.present(future, false);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        window_renderer.window().request_redraw();
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    swapchain_images: &[Arc<ImageView>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    swapchain_images
        .iter()
        .map(|swapchain_image| {
            Framebuffer::new(
                render_pass,
                &FramebufferCreateInfo {
                    attachments: &[swapchain_image],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
