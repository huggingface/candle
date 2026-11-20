fn main() -> Result<(), winit::error::EventLoopError> {
    #[cfg(target_os = "linux")]
    {
        linux::main()
    }
    #[cfg(not(target_os = "linux"))]
    {
        println!("Not Implemented");
        Ok(())
    }
}

// TODO: Can this be demonstrated for other platforms as well?
#[cfg(target_os = "linux")]
mod linux {
    use glium::{
        backend::Facade,
        debug::DebugCallbackBehavior,
        glutin::{
            config::{Config, ConfigSurfaceTypes, ConfigTemplateBuilder, GlConfig},
            context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, Version},
            display::{GetGlDisplay, GlDisplay},
            surface::WindowSurface,
        },
        CapabilitiesSource, Display,
    };
    use glutin_winit::{DisplayBuilder, GlWindow};
    use std::{
        fs::File,
        os::fd::FromRawFd,
        sync::{Arc, Barrier},
        time::Instant,
    };
    use vulkano::{
        buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
        command_buffer::{
            allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
            CommandBufferUsage, RenderPassBeginInfo, SemaphoreSubmitInfo, SubmitInfo,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator, DescriptorImageInfo, DescriptorSet,
            WriteDescriptorSet,
        },
        device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
            QueueCreateInfo, QueueFlags,
        },
        format::Format,
        image::{
            sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
            sys::RawImage,
            view::ImageView,
            Image, ImageCreateFlags, ImageCreateInfo, ImageType, ImageUsage,
        },
        instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
        memory::{
            allocator::{
                AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
            },
            DedicatedAllocation, DeviceMemory, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
            MemoryAllocateInfo, ResourceMemory,
        },
        pipeline::{
            graphics::{
                color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
                input_assembly::{InputAssemblyState, PrimitiveTopology},
                multisample::MultisampleState,
                rasterization::RasterizationState,
                vertex_input::{Vertex, VertexDefinition},
                viewport::{Viewport, ViewportState},
                GraphicsPipelineCreateInfo,
            },
            DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
            PipelineShaderStageCreateInfo,
        },
        render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
        swapchain::{
            acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
        },
        sync::{
            self,
            semaphore::{
                ExternalSemaphoreHandleType, ExternalSemaphoreHandleTypes, Semaphore,
                SemaphoreCreateInfo,
            },
            GpuFuture,
        },
        Validated, VulkanError, VulkanLibrary,
    };
    use winit::{
        application::ApplicationHandler,
        error::EventLoopError,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, EventLoop},
        raw_window_handle::HasWindowHandle,
        window::{Window, WindowId},
    };

    pub fn main() -> Result<(), EventLoopError> {
        let event_loop = EventLoop::new().unwrap();
        let mut app = App::new(&event_loop);

        event_loop.run_app(&mut app)
    }

    struct App {
        instance: Arc<Instance>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        vertex_buffer: Subbuffer<[MyVertex]>,
        image_view: Arc<ImageView>,
        sampler: Arc<Sampler>,
        barrier: Arc<Barrier>,
        barrier_2: Arc<Barrier>,
        acquire_sem: Semaphore,
        release_sem: Semaphore,
        rcx: Option<RenderContext>,
    }

    struct RenderContext {
        window: Arc<Window>,
        swapchain: Arc<Swapchain>,
        render_pass: Arc<RenderPass>,
        framebuffers: Vec<Arc<Framebuffer>>,
        pipeline: Arc<GraphicsPipeline>,
        viewport: Viewport,
        descriptor_set: Arc<DescriptorSet>,
        recreate_swapchain: bool,
        previous_frame_end: Option<Box<dyn GpuFuture>>,
    }

    impl App {
        fn new(event_loop: &EventLoop<()>) -> Self {
            // A requirement for sharing memory between OpenGL and Vulkan is that both instances
            // need to use the same graphics device and driver. Historically, choosing which device
            // to use has not been a feature available in OpenGL (depending on the platform, it's
            // now possible to do this via environment variables, driver hints, driver extensions
            // or EGL).
            //
            // But since Vulkan allows us to explicitly choose a graphics device, we'll first
            // create an OpenGL context to query what device and driver UUIDs got chosen. Then
            // we'll make sure our Vulkan instance uses the same ones.
            let (gl_driver_uuid, gl_device_uuids) = {
                let (window, config) = create_window_with_opengl_support(event_loop);
                let gl_surface = create_opengl_surface(&window.unwrap(), config);
                (
                    gl_surface.get_context().driver_uuid().unwrap(),
                    gl_surface.get_context().device_uuids().unwrap(),
                )
            };

            let library = unsafe { VulkanLibrary::new() }.unwrap();
            let required_extensions = Surface::required_extensions(event_loop);
            let instance = Instance::new(
                &library,
                &InstanceCreateInfo {
                    flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                    enabled_extensions: &InstanceExtensions {
                        khr_get_physical_device_properties2: true,
                        khr_external_memory_capabilities: true,
                        khr_external_semaphore_capabilities: true,
                        khr_external_fence_capabilities: true,
                        ext_debug_utils: true,
                        ..required_extensions
                    },
                    ..Default::default()
                },
            )
            .unwrap();

            let device_extensions = DeviceExtensions {
                khr_external_semaphore: true,
                khr_external_semaphore_fd: true,
                khr_external_memory: true,
                khr_external_memory_fd: true,
                khr_external_fence: true,
                khr_external_fence_fd: true,
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
                .filter(|(p, _)| p.properties().driver_uuid.unwrap() == gl_driver_uuid)
                .filter(|(p, _)| gl_device_uuids.contains(&p.properties().device_uuid.unwrap()))
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

            let memory_allocator =
                Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
            let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
                &device,
                &Default::default(),
            ));
            let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
                &device,
                &Default::default(),
            ));

            let vertices = [
                MyVertex {
                    position: [-0.5, -0.5],
                },
                MyVertex {
                    position: [-0.5, 0.5],
                },
                MyVertex {
                    position: [0.5, -0.5],
                },
                MyVertex {
                    position: [0.5, 0.5],
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

            let raw_image = RawImage::new(
                &device,
                &ImageCreateInfo {
                    flags: ImageCreateFlags::MUTABLE_FORMAT,
                    image_type: ImageType::Dim2d,
                    format: Format::R16G16B16A16_UNORM,
                    extent: [200, 200, 1],
                    usage: ImageUsage::TRANSFER_SRC
                        | ImageUsage::TRANSFER_DST
                        | ImageUsage::SAMPLED,
                    external_memory_handle_types: ExternalMemoryHandleTypes::OPAQUE_FD,
                    ..Default::default()
                },
            )
            .unwrap();

            let image_requirements = raw_image.memory_requirements()[0];

            let image_memory = DeviceMemory::allocate(
                &device,
                &MemoryAllocateInfo {
                    allocation_size: image_requirements.layout.size(),
                    memory_type_index: memory_allocator
                        .find_memory_type_index(
                            image_requirements.memory_type_bits,
                            MemoryTypeFilter::PREFER_DEVICE,
                        )
                        .unwrap(),
                    dedicated_allocation: Some(DedicatedAllocation::Image(&raw_image)),
                    export_handle_types: ExternalMemoryHandleTypes::OPAQUE_FD,
                    ..Default::default()
                },
            )
            .unwrap();

            let allocation_size = image_memory.allocation_size();
            let image_fd = image_memory
                .export_fd(ExternalMemoryHandleType::OpaqueFd)
                .unwrap();

            let image = Arc::new(
                raw_image
                    .bind_memory([ResourceMemory::new_dedicated(image_memory)])
                    .map_err(|(err, _, _)| err)
                    .unwrap(),
            );

            let image_view = ImageView::new_default(&image).unwrap();

            let sampler = Sampler::new(
                &device,
                &SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::Repeat; 3],
                    ..Default::default()
                },
            )
            .unwrap();

            let barrier = Arc::new(Barrier::new(2));
            let barrier_2 = Arc::new(Barrier::new(2));

            let acquire_sem = Semaphore::new(
                &device,
                &SemaphoreCreateInfo {
                    export_handle_types: ExternalSemaphoreHandleTypes::OPAQUE_FD,
                    ..Default::default()
                },
            )
            .unwrap();
            let release_sem = Semaphore::new(
                &device,
                &SemaphoreCreateInfo {
                    export_handle_types: ExternalSemaphoreHandleTypes::OPAQUE_FD,
                    ..Default::default()
                },
            )
            .unwrap();

            let acquire_fd =
                unsafe { acquire_sem.export_fd(ExternalSemaphoreHandleType::OpaqueFd) }.unwrap();
            let release_fd =
                unsafe { release_sem.export_fd(ExternalSemaphoreHandleType::OpaqueFd) }.unwrap();

            let barrier_clone = barrier.clone();
            let barrier_2_clone = barrier_2.clone();

            let (window, window_config) = create_window_with_opengl_support(event_loop);
            std::thread::spawn(move || {
                let gl_surface = create_opengl_surface(&window.unwrap(), window_config);
                let gl_context = gl_surface.get_context().clone();

                let image_file = unsafe { File::from_raw_fd(image_fd) };
                let gl_tex = unsafe {
                    glium::texture::Texture2d::new_from_fd(
                        &gl_context,
                        glium::texture::UncompressedFloatFormat::U16U16U16U16,
                        glium::texture::MipmapsOption::NoMipmap,
                        glium::texture::Dimensions::Texture2d {
                            width: 200,
                            height: 200,
                        },
                        glium::texture::ImportParameters {
                            dedicated_memory: true,
                            size: allocation_size,
                            offset: 0,
                            tiling: glium::texture::ExternalTilingMode::Optimal,
                        },
                        image_file,
                    )
                }
                .unwrap();

                let gl_acquire_sem = unsafe {
                    glium::semaphore::Semaphore::new_from_fd(
                        &gl_context,
                        File::from_raw_fd(acquire_fd),
                    )
                }
                .unwrap();
                let gl_release_sem = unsafe {
                    glium::semaphore::Semaphore::new_from_fd(
                        &gl_context,
                        File::from_raw_fd(release_fd),
                    )
                }
                .unwrap();

                let rotation_start = Instant::now();

                loop {
                    barrier_clone.wait();
                    gl_acquire_sem.wait_textures(Some(&[(
                        &gl_tex,
                        glium::semaphore::TextureLayout::General,
                    )]));

                    gl_context.flush();

                    let elapsed = rotation_start.elapsed();
                    let rotation = elapsed.as_nanos() as f64 / 2_000_000_000.0;

                    use glium::Surface;
                    {
                        let mut fb = gl_tex.as_surface();

                        fb.clear_color(
                            0.0,
                            (((rotation as f32).sin() + 1.) / 2.).powf(2.2),
                            0.0,
                            1.0,
                        );
                    }
                    gl_release_sem.signal_textures(Some(&[(
                        &gl_tex,
                        glium::semaphore::TextureLayout::General,
                    )]));
                    barrier_2_clone.wait();

                    gl_context.finish();

                    gl_context.assert_no_error(Some("err"));
                }
            });

            App {
                instance,
                device,
                queue,
                descriptor_set_allocator,
                command_buffer_allocator,
                vertex_buffer,
                sampler,
                image_view,
                barrier,
                barrier_2,
                acquire_sem,
                release_sem,
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
                },
                pass: {
                    color: [color],
                    depth_stencil: {},
                },
            )
            .unwrap();

            let framebuffers = window_size_dependent_setup(&images, &render_pass);

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
                        input_assembly_state: Some(&InputAssemblyState {
                            topology: PrimitiveTopology::TriangleStrip,
                            ..Default::default()
                        }),
                        viewport_state: Some(&ViewportState::default()),
                        rasterization_state: Some(&RasterizationState::default()),
                        multisample_state: Some(&MultisampleState::default()),
                        color_blend_state: Some(&ColorBlendState {
                            attachments: &[ColorBlendAttachmentState {
                                blend: Some(AttachmentBlend::alpha()),
                                ..Default::default()
                            }],
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

            let layout = &pipeline.layout().set_layouts()[0];

            let descriptor_set = DescriptorSet::new(
                &self.descriptor_set_allocator,
                layout,
                &[
                    WriteDescriptorSet::image(
                        0,
                        &DescriptorImageInfo {
                            sampler: Some(&self.sampler),
                            ..Default::default()
                        },
                    ),
                    WriteDescriptorSet::image(
                        1,
                        &DescriptorImageInfo {
                            image_view: Some(&self.image_view),
                            ..Default::default()
                        },
                    ),
                ],
                &[],
            )
            .unwrap();

            let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

            self.rcx = Some(RenderContext {
                window,
                swapchain,
                render_pass,
                framebuffers,
                pipeline,
                viewport,
                descriptor_set,
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
                    self.queue
                        .with(|mut q| unsafe {
                            q.submit(
                                &[SubmitInfo {
                                    signal_semaphores: &[SemaphoreSubmitInfo::new(
                                        &self.acquire_sem,
                                    )],
                                    ..Default::default()
                                }],
                                None,
                            )
                        })
                        .unwrap();

                    self.barrier.wait();
                    self.barrier_2.wait();

                    self.queue
                        .with(|mut q| unsafe {
                            q.submit(
                                &[SubmitInfo {
                                    wait_semaphores: &[SemaphoreSubmitInfo::new(&self.release_sem)],
                                    ..Default::default()
                                }],
                                None,
                            )
                        })
                        .unwrap();

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
                        rcx.framebuffers =
                            window_size_dependent_setup(&new_images, &rcx.render_pass);
                        rcx.viewport.extent = window_size.into();
                        rcx.recreate_swapchain = false;
                    }

                    let (image_index, suboptimal, acquire_future) =
                        match acquire_next_image(rcx.swapchain.clone(), None)
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
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            rcx.pipeline.layout().clone(),
                            0,
                            rcx.descriptor_set.clone(),
                        )
                        .unwrap()
                        .bind_vertex_buffers(0, self.vertex_buffer.clone())
                        .unwrap();
                    unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

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
                            future.wait(None).unwrap();
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
                    };
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
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

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

    fn create_window_with_opengl_support(event_loop: &EventLoop<()>) -> (Option<Window>, Config) {
        DisplayBuilder::new()
            .with_window_attributes(Some(Window::default_attributes()))
            .build(
                event_loop,
                ConfigTemplateBuilder::default(),
                |mut available_configs| {
                    available_configs
                        .find(|config| {
                            config
                                .config_surface_types()
                                .intersects(ConfigSurfaceTypes::WINDOW)
                        })
                        .unwrap()
                },
            )
            .unwrap()
    }

    fn create_opengl_surface(window: &Window, window_config: Config) -> Display<WindowSurface> {
        // This example uses OpenGL 4.5 or OpenGL ES 3.2. Earlier versions can also be
        // supported given that additional extension requirements are met. See the dependency
        // section of
        // https://registry.khronos.org/OpenGL/extensions/EXT/EXT_external_objects.txt
        let raw_window_handle = window.window_handle().unwrap().as_raw();
        let gl_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(4, 5))))
            .build(Some(raw_window_handle));
        let gles_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(Some(Version::new(3, 2))))
            .build(Some(raw_window_handle));
        let context = unsafe {
            window_config
                .display()
                .create_context(&window_config, &gl_context_attributes)
                .unwrap_or_else(|_| {
                    window_config
                        .display()
                        .create_context(&window_config, &gles_context_attributes)
                        .unwrap()
                })
        };

        let window_attributes = window.build_surface_attributes(Default::default()).unwrap();
        let surface = unsafe {
            window_config
                .display()
                .create_window_surface(&window_config, &window_attributes)
                .unwrap()
        };
        let display = Display::with_debug(
            context.make_current(&surface).unwrap(),
            surface,
            DebugCallbackBehavior::PrintAll,
        )
        .unwrap();

        let supported_extensions = display.get_context().get_extensions();
        assert!(
            supported_extensions.gl_ext_memory_object
                && supported_extensions.gl_ext_memory_object_fd
                && supported_extensions.gl_ext_semaphore
                && supported_extensions.gl_ext_semaphore_fd,
            "Missing one or more of these required OpenGL extensions: GL_EXT_memory_object, \
            GL_EXT_memory_object_fd, GL_EXT_semaphore, GL_EXT_semaphore_fd"
        );

        display
    }

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450
                layout(location = 0) in vec2 position;
                layout(location = 0) out vec2 tex_coords;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    tex_coords = position + vec2(0.5);
                }
            ",
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450
                layout(location = 0) in vec2 tex_coords;
                layout(location = 0) out vec4 f_color;

                layout(set = 0, binding = 0) uniform sampler s;
                layout(set = 0, binding = 1) uniform texture2D tex;

                void main() {
                    f_color = texture(sampler2D(tex, s), tex_coords);
                }
            ",
        }
    }
}
