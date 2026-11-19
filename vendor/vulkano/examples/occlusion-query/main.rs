// This is a modification of the triangle example, that demonstrates the basics of occlusion
// queries. Occlusion queries allow you to query whether, and sometimes how many, pixels pass the
// depth test in a range of draw calls.

use std::{error::Error, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    query::{QueryControlFlags, QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    triangle1: Subbuffer<[MyVertex]>,
    triangle2: Subbuffer<[MyVertex]>,
    triangle3: Subbuffer<[MyVertex]>,
    query_pool: Arc<QueryPool>,
    query_results: [u32; 3],
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = unsafe { VulkanLibrary::new() }.unwrap();
        let required_extensions = Surface::required_extensions(event_loop);
        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                enabled_extensions: &device_extensions,
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let [queue] = queues.try_into().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));

        let vertices = [
            // The first triangle (red) is the same one as in the triangle example.
            MyVertex {
                position: [-0.5, -0.25, 0.5],
                color: [1.0, 0.0, 0.0],
            },
            MyVertex {
                position: [0.0, 0.5, 0.5],
                color: [1.0, 0.0, 0.0],
            },
            MyVertex {
                position: [0.25, -0.1, 0.5],
                color: [1.0, 0.0, 0.0],
            },
            // The second triangle (cyan) is the same shape and position as the first, but smaller,
            // and moved behind a bit. It should be completely occluded by the first triangle. (You
            // can lower its z value to put it in front.)
            MyVertex {
                position: [-0.25, -0.125, 0.6],
                color: [0.0, 1.0, 1.0],
            },
            MyVertex {
                position: [0.0, 0.25, 0.6],
                color: [0.0, 1.0, 1.0],
            },
            MyVertex {
                position: [0.125, -0.05, 0.6],
                color: [0.0, 1.0, 1.0],
            },
            // The third triangle (green) is the same shape and size as the first, but moved to the
            // left and behind the second. It is partially occluded by the first two.
            MyVertex {
                position: [-0.25, -0.25, 0.7],
                color: [0.0, 1.0, 0.0],
            },
            MyVertex {
                position: [0.25, 0.5, 0.7],
                color: [0.0, 1.0, 0.0],
            },
            MyVertex {
                position: [0.5, -0.1, 0.7],
                color: [0.0, 1.0, 0.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            &memory_allocator,
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

        // Create three buffer slices, one for each triangle.
        let triangle1 = vertex_buffer.clone().slice(0..3);
        let triangle2 = vertex_buffer.clone().slice(3..6);
        let triangle3 = vertex_buffer.slice(6..9);

        // Create a query pool for occlusion queries, with 3 slots.
        let query_pool = QueryPool::new(
            &device,
            &QueryPoolCreateInfo {
                query_count: 3,
                ..QueryPoolCreateInfo::new(QueryType::Occlusion)
            },
        )
        .unwrap();

        // Create a buffer on the CPU to hold the results of the three queries. Query results are
        // always represented as either `u32` or `u64`. For occlusion queries, you always need one
        // element per query. You can ask for the number of elements needed at runtime by calling
        // `QueryType::result_len`. If you retrieve query results with `with_availability` enabled,
        // then this array needs to be 6 elements long instead of 3.
        let query_results = [0u32; 3];

        App {
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            triangle1,
            triangle2,
            triangle3,
            query_pool,
            query_results,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(&self.instance, &window).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();
            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            Swapchain::new(
                &self.device,
                &surface,
                &SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = vulkano::single_pass_renderpass!(
            &self.device,
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth_stencil},
            },
        )
        .unwrap();

        let framebuffers =
            window_size_dependent_setup(&images, &render_pass, &self.memory_allocator);

        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec3 position;
                    layout(location = 1) in vec3 color;

                    layout(location = 0) out vec3 v_color;

                    void main() {
                        v_color = color;
                        gl_Position = vec4(position, 1.0);
                    }
                ",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450

                    layout(location = 0) in vec3 v_color;
                    layout(location = 0) out vec4 f_color;

                    void main() {
                        f_color = vec4(v_color, 1.0);
                    }
                ",
            }
        }

        let pipeline = {
            let vs = unsafe { vs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { fs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];
            let layout = PipelineLayout::from_stages(&self.device, &stages).unwrap();
            let subpass = Subpass::new(&render_pass, 0).unwrap();

            GraphicsPipeline::new(
                &self.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    vertex_input_state: Some(&vertex_input_state),
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    viewport_state: Some(&ViewportState::default()),
                    rasterization_state: Some(&RasterizationState::default()),
                    multisample_state: Some(&MultisampleState::default()),
                    // Enable depth testing, which is needed for occlusion queries to make sense at
                    // all. If you disable depth testing, every pixel is considered to pass the
                    // depth test, so every query will return a nonzero result.
                    depth_stencil_state: Some(&DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState::default()],
                        ..Default::default()
                    }),
                    dynamic_state: &[DynamicState::Viewport],
                    subpass: Some((&subpass).into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            recreate_swapchain: false,
            previous_frame_end,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(&SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rcx.swapchain = new_swapchain;
                    rcx.framebuffers = window_size_dependent_setup(
                        &new_images,
                        &rcx.render_pass,
                        &self.memory_allocator,
                    );
                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // A query must be reset before each use, including the first use. This
                // must be done outside a render pass.
                // Beginning or resetting a query is unsafe for now.
                unsafe { builder.reset_query_pool(self.query_pool.clone(), 0..3) }.unwrap();

                builder
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap()
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into()), Some(1.0.into())],
                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap();

                // Begin query 0, then draw the red triangle. Enabling the
                // `QueryControlFlags::PRECISE` flag would give exact numeric results. This
                // needs the `occlusion_query_precise` feature to be enabled on the device.
                unsafe {
                    builder.begin_query(
                        self.query_pool.clone(),
                        0,
                        QueryControlFlags::empty(),
                        // QueryControlFlags::PRECISE,
                    )
                }
                .unwrap();
                builder
                    .bind_vertex_buffers(0, self.triangle1.clone())
                    .unwrap();

                unsafe { builder.draw(self.triangle1.len() as u32, 1, 0, 0) }.unwrap();
                // End query 0.
                builder.end_query(self.query_pool.clone(), 0).unwrap();

                // Begin query 1 for the cyan triangle.
                unsafe {
                    builder.begin_query(self.query_pool.clone(), 1, QueryControlFlags::empty())
                }
                .unwrap();
                builder
                    .bind_vertex_buffers(0, self.triangle2.clone())
                    .unwrap();
                unsafe { builder.draw(self.triangle2.len() as u32, 1, 0, 0) }.unwrap();
                builder.end_query(self.query_pool.clone(), 1).unwrap();

                // Finally, query 2 for the green triangle.
                unsafe {
                    builder.begin_query(self.query_pool.clone(), 2, QueryControlFlags::empty())
                }
                .unwrap();
                builder
                    .bind_vertex_buffers(0, self.triangle3.clone())
                    .unwrap();
                unsafe { builder.draw(self.triangle3.len() as u32, 1, 0, 0) }.unwrap();
                builder
                    .end_query(self.query_pool.clone(), 2)
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::new(rcx.swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }

                // Retrieve the query results. This copies the results to a variable on the CPU.
                // You can also use the `copy_query_pool_results` function on a command buffer to
                // write results to a Vulkano buffer. This could then be used to influence draw
                // operations further down the line, either in the same frame or a future frame.
                #[rustfmt::skip]
                self.query_pool.get_results(
                    0,
                    3,
                    &mut self.query_results,
                    // Block the function call until the results are available.
                    // NOTE: If not all the queries have actually been executed, then this will
                    // wait forever for something that never happens!
                    QueryResultFlags::WAIT

                    // Enable this flag to give partial results if available, instead of waiting
                    // for the full results.
                    // | QueryResultFlags::PARTIAL

                    // Blocking and waiting will ensure the results are always available after the
                    // function returns.
                    //
                    // If you disable waiting, then this flag can be enabled to include the
                    // availability of each query's results. You need one extra element per query
                    // in your `query_results` buffer for this. This element will be filled with a
                    // zero/nonzero value indicating availability.
                    // | QueryResultFlags::WITH_AVAILABILITY
                )
                .unwrap();

                // If the `precise` bit was not enabled, then you're only guaranteed to get a
                // boolean result here: zero if all pixels were occluded, nonzero if only some were
                // occluded. Enabling `precise` will give the exact number of pixels.

                // Query 0 (red triangle) will always succeed, because the depth buffer starts
                // empty and will never occlude anything.
                assert_ne!(self.query_results[0], 0);

                // Query 1 (cyan triangle) will fail, because it's drawn completely behind the
                // first.
                assert_eq!(self.query_results[1], 0);

                // Query 2 (green triangle) will succeed, because it's only partially occluded.
                assert_ne!(self.query_results[2], 0);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Vec<Arc<Framebuffer>> {
    let depth_attachment = ImageView::new_default(
        &Image::new(
            memory_allocator,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image).unwrap();

            Framebuffer::new(
                render_pass,
                &FramebufferCreateInfo {
                    attachments: &[&view, &depth_attachment],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
