use std::{collections::HashMap, error::Error, slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        Queue, QueueCreateInfo, QueueFlags,
    },
    image::ImageUsage,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
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
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanError, VulkanLibrary,
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer,
    graph::{AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph},
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources},
    resource_map, Id, QueueFamilyType, Task, TaskContext, TaskResult,
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MIN_SWAPCHAIN_IMAGES: u32 = 3;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    println!("\nPress space to open a new window");

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    vertex_buffer_id: Id<Buffer>,
    rcxs: HashMap<WindowId, RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    viewport: Viewport,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<Self>,
    virtual_swapchain_id: Id<Swapchain>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = unsafe { VulkanLibrary::new() }.unwrap();

        let mut instance_extensions = Surface::required_extensions(event_loop);

        if library.supported_extensions().ext_surface_maintenance1 {
            instance_extensions.ext_surface_maintenance1 = true;
        }

        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &instance_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let mut device_extensions = DeviceExtensions {
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

        let mut device_features = DeviceFeatures::empty();

        if physical_device
            .supported_extensions()
            .ext_swapchain_maintenance1
        {
            device_extensions.ext_swapchain_maintenance1 = true;

            if physical_device.supported_features().swapchain_maintenance1 {
                device_features.swapchain_maintenance1 = true;
            }
        }

        let (device, queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                enabled_extensions: &device_extensions,
                enabled_features: &device_features,
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let [queue] = queues.try_into().unwrap();

        let resources = Resources::new(&device, &Default::default()).unwrap();

        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

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
        let vertex_buffer_id = resources
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

        unsafe {
            vulkano_taskgraph::execute(
                &queue,
                &resources,
                flight_id,
                |_cbf, tcx| {
                    tcx.write_buffer::<[MyVertex]>(vertex_buffer_id, ..)
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        // A hashmap that contains all of our created windows and their resources.
        let rcxs = HashMap::new();

        App {
            instance,
            device,
            queue,
            resources,
            flight_id,
            vertex_buffer_id,
            rcxs,
        }
    }

    fn create_rcx(&self, window: Window) -> RenderContext {
        let window = Arc::new(window);
        let surface = Surface::from_window(&self.instance, &window).unwrap();
        let window_size = window.inner_size();

        // The swapchain for this particular window.
        let swapchain_format;
        let swapchain_id = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();
            (swapchain_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            self.resources
                .create_swapchain(
                    &surface,
                    &SwapchainCreateInfo {
                        min_image_count: surface_capabilities
                            .min_image_count
                            .max(MIN_SWAPCHAIN_IMAGES),
                        image_format: swapchain_format,
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

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let mut task_graph = TaskGraph::new(&self.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: swapchain_format,
            ..Default::default()
        });
        let virtual_framebuffer_id = task_graph.add_framebuffer();

        let scene_node_id = task_graph
            .create_task_node(
                "Scene",
                QueueFamilyType::Graphics,
                SceneTask {
                    vertex_buffer_id: self.vertex_buffer_id,
                    pipeline: None,
                },
            )
            .framebuffer(virtual_framebuffer_id)
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .buffer_access(self.vertex_buffer_id, AccessTypes::VERTEX_ATTRIBUTE_READ)
            .build();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.queue],
                present_queue: Some(&self.queue),
                flight_id: self.flight_id,
                ..Default::default()
            })
        }
        .unwrap();

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

        let scene_node = task_graph.task_node_mut(scene_node_id).unwrap();

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
            let subpass = scene_node.subpass().unwrap();

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
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        scene_node
            .task_mut()
            .downcast_mut::<SceneTask>()
            .unwrap()
            .pipeline = Some(pipeline);

        RenderContext {
            window,
            swapchain_id,
            viewport,
            recreate_swapchain: false,
            task_graph,
            virtual_swapchain_id,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        // Use the window's id as a means to access it from the hashmap.
        let window_id = window.id();

        let rcx = self.create_rcx(window);

        self.rcxs.insert(window_id, rcx);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                let rcx = self.rcxs.remove(&window_id).unwrap();

                self.resources.remove_swapchain(rcx.swapchain_id);

                // Unfortunately, the only way to guarantee that a swapchain is no longer being
                // used by the presentation engine is to do a device-wide wait for idle. Without
                // this, the swapchain would never get cleaned up.
                self.resources.wait_idle().unwrap();

                if self.rcxs.is_empty() {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(_) => {
                self.rcxs.get_mut(&window_id).unwrap().recreate_swapchain = true;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Space),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let window = event_loop
                    .create_window(Window::default_attributes())
                    .unwrap();
                let window_id = window.id();
                let rcx = self.create_rcx(window);

                self.rcxs.insert(window_id, rcx);
            }
            WindowEvent::RedrawRequested => {
                let rcx = self.rcxs.get_mut(&window_id).unwrap();

                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                let flight = self.resources.flight(self.flight_id);

                if rcx.recreate_swapchain {
                    rcx.swapchain_id = self
                        .resources
                        .recreate_swapchain(rcx.swapchain_id, |create_info| SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..*create_info
                        })
                        .expect("failed to recreate swapchain");

                    if !self.device.enabled_features().swapchain_maintenance1 {
                        self.resources.wait_idle().unwrap();
                    }

                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                }

                let resource_map = resource_map!(
                    &rcx.task_graph,
                    rcx.virtual_swapchain_id => rcx.swapchain_id,
                )
                .unwrap();

                flight.wait(None).unwrap();

                match unsafe {
                    rcx.task_graph
                        .execute(resource_map, rcx, || rcx.window.pre_present_notify())
                } {
                    Ok(()) => {}
                    Err(ExecuteError::Swapchain {
                        error: VulkanError::OutOfDate,
                        ..
                    }) => {
                        rcx.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to execute next frame: {e:?}");
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.rcxs.values().for_each(|s| s.window.request_redraw());
    }
}

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

struct SceneTask {
    vertex_buffer_id: Id<Buffer>,
    pipeline: Option<Arc<GraphicsPipeline>>,
}

impl Task for SceneTask {
    type World = RenderContext;

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
