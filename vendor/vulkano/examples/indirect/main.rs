// Indirect draw example
//
// Indirect draw calls allow us to issue a draw without needing to know the number of vertices
// until later when the draw is executed by the GPU.
//
// This is used in situations where vertices are being generated on the GPU, such as a GPU particle
// simulation, and the exact number of output vertices cannot be known until the compute shader has
// run.
//
// In this example the compute shader is trivial and the number of vertices does not change.
// However is does demonstrate that each compute instance atomically updates the vertex counter
// before filling the vertex buffer.
//
// For an explanation of how the rendering of the triangles takes place see the `triangle.rs`
// example.

use std::{error::Error, slice, sync::Arc};
use vulkano::{
    buffer::{
        allocator::{BufferAllocator, BufferAllocatorCreateInfo},
        BufferContents, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        DrawIndirectCommand, RenderPassBeginInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorBufferInfo, DescriptorSet,
        WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{BumpAllocator, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    single_pass_renderpass,
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
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    compute_pipeline: Arc<ComputePipeline>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    buffer_allocators: Vec<BufferAllocator<BumpAllocator>>,
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
            khr_storage_buffer_storage_class: true,
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
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &device,
            &Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));

        // A simple compute shader that generates vertices. It has two buffers bound: the first is
        // where we output the vertices, the second is the `IndirectDrawArgs` struct we passed the
        // `draw_indirect` so we can set the number to vertices to draw.
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r"
                    #version 450

                    layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

                    layout(set = 0, binding = 0) buffer Output {
                        vec2 pos[];
                    } triangles;

                    layout(set = 0, binding = 1) buffer IndirectDrawArgs {
                        uint vertices;
                        uint unused0;
                        uint unused1;
                        uint unused2;
                    };

                    void main() {
                        uint idx = gl_GlobalInvocationID.x;

                        // Each invocation of the compute shader is going to increment the counter,
                        // so we need to use atomic operations for safety. The previous value of
                        // the counter is returned so that gives us the offset into the vertex
                        // buffer this thread can write it's vertices into.
                        uint offset = atomicAdd(vertices, 6);

                        vec2 center = vec2(-0.8, -0.8) + idx * vec2(0.1, 0.1);
                        triangles.pos[offset + 0] = center + vec2(0.0, 0.0375);
                        triangles.pos[offset + 1] = center + vec2(0.025, -0.01725);
                        triangles.pos[offset + 2] = center + vec2(-0.025, -0.01725);
                        triangles.pos[offset + 3] = center + vec2(0.0, -0.0375);
                        triangles.pos[offset + 4] = center + vec2(0.025, 0.01725);
                        triangles.pos[offset + 5] = center + vec2(-0.025, 0.01725);
                    }
                ",
            }
        }

        let compute_pipeline = {
            let cs = unsafe { cs::load(&device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = PipelineLayout::from_stages(&device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                &device,
                None,
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        App {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            compute_pipeline,
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

        let render_pass = single_pass_renderpass!(
            &self.device,
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let framebuffers = window_size_dependent_setup(&images, &render_pass);

        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    // The triangle vertex positions.
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

        // Each frame we generate a new set of vertices and each frame we need a new
        // `DrawIndirectCommand` struct to set the number of vertices to draw. To allocate these
        // buffers, we use one buffer allocator per frame in flight. This allows us to reset the
        // allocator at the beginning of the frame, which is more efficient than allocating and
        // deallocating individually. Addidionally, it allows us to use the most efficient
        // allocation algorithm: bump allocation.
        //
        // We use a single allocator for both needs even though they need different buffer usage
        // flags. We simply combine the usage flags, which is the most efficient way. There are no
        // Vulkan implementations which care about buffer usage flags, and they never affect
        // performance, no matter how many you have. The exception is `UNIFORM_BUFFER`, which has
        // stricter limits, and the `SHADER_DEVICE_ADDRESS` and `ACCELERATION_STRUCTURE_STORAGE`,
        // which may influence the available memory types.
        let buffer_allocators = (0..swapchain.image_count())
            .map(|_| {
                BufferAllocator::new(
                    &self.memory_allocator,
                    &BufferAllocatorCreateInfo {
                        block_size: 1024,
                        buffer_usage: BufferUsage::VERTEX_BUFFER
                            | BufferUsage::INDIRECT_BUFFER
                            | BufferUsage::STORAGE_BUFFER,
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                )
            })
            .collect();

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            buffer_allocators,
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
                    rcx.framebuffers = window_size_dependent_setup(&new_images, &rcx.render_pass);
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

                let buffer_allocator = &mut rcx.buffer_allocators[image_index as usize];

                buffer_allocator.reset();

                // Allocate a buffer to hold the arguments for this frame's draw call. The compute
                // shader will only update `vertex_count`, so set the other parameters correctly
                // here.
                let indirect_commands = [DrawIndirectCommand {
                    vertex_count: 0,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                }];
                let indirect_buffer_allocation = buffer_allocator
                    .allocate(DeviceLayout::for_value(&indirect_commands).unwrap())
                    .unwrap();
                let indirect_buffer = Subbuffer::from(indirect_buffer_allocation.buffer.clone())
                    .slice(indirect_buffer_allocation.as_range())
                    .reinterpret::<[DrawIndirectCommand]>();
                indirect_buffer
                    .write()
                    .unwrap()
                    .copy_from_slice(&indirect_commands);

                // Allocate a buffer to hold this frame's vertices. This needs to be large enough
                // to hold the worst case number of vertices generated by the compute shader.
                let iter = (0..(6 * 16)).map(|_| MyVertex { position: [0.0; 2] });
                let vertex_buffer_allocation = buffer_allocator
                    .allocate(DeviceLayout::new_unsized::<[MyVertex]>(iter.len() as _).unwrap())
                    .unwrap();
                let vertex_buffer = Subbuffer::from(vertex_buffer_allocation.buffer.clone())
                    .slice(vertex_buffer_allocation.as_range())
                    .reinterpret::<[MyVertex]>();
                for (o, i) in vertex_buffer.write().unwrap().iter_mut().zip(iter) {
                    *o = i;
                }

                // Pass the two buffers to the compute shader.
                let layout = &self.compute_pipeline.layout().set_layouts()[0];
                let cs_descriptor_set = DescriptorSet::new(
                    &self.descriptor_set_allocator,
                    layout,
                    &[
                        WriteDescriptorSet::buffer(
                            0,
                            &DescriptorBufferInfo {
                                buffer: Some(vertex_buffer.buffer()),
                                ..Default::default()
                            },
                        ),
                        WriteDescriptorSet::buffer(
                            1,
                            &DescriptorBufferInfo {
                                buffer: Some(indirect_buffer.buffer()),
                                ..Default::default()
                            },
                        ),
                    ],
                    &[],
                )
                .unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // First in the command buffer we dispatch the compute shader to generate the
                // vertices and fill out the draw call arguments.
                builder
                    .bind_pipeline_compute(self.compute_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.compute_pipeline.layout().clone(),
                        0,
                        cs_descriptor_set,
                    )
                    .unwrap();
                unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer)
                    .unwrap();

                // The indirect draw call is placed in the command buffer with a reference to
                // the buffer that will contain the arguments for the draw.
                unsafe { builder.draw_indirect(indirect_buffer) }.unwrap();

                builder.end_render_pass(Default::default()).unwrap();

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
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

// `MyVertex` is the vertex type that will be output from the compute shader and be input to the
// vertex shader.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image).unwrap();

            Framebuffer::new(
                render_pass,
                &FramebufferCreateInfo {
                    attachments: &[&view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
