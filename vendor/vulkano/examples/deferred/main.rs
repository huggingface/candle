// Welcome to the deferred lighting example!
//
// The idea behind deferred lighting is to render the scene in two steps.
//
// First you draw all the objects of the scene. But instead of calculating the color they will have
// on the screen, you output their characteristics such as their diffuse color and their normals,
// and write this to images.
//
// After all the objects are drawn, you should obtain several images that contain the
// characteristics of each pixel.
//
// Then you apply lighting to the scene. In other words you draw to the final image by taking these
// intermediate images and the various lights of the scene as input.
//
// This technique allows you to apply tons of light sources to a scene, which would be too
// expensive otherwise. It has some drawbacks, which are the fact that transparent objects must be
// drawn after the lighting, and that the whole process consumes more memory.

use deferred::DeferredTask;
use scene::SceneTask;
use std::{error::Error, sync::Arc};
use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::AllocationCreateInfo,
    pipeline::graphics::viewport::Viewport,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanError, VulkanLibrary,
};
use vulkano_taskgraph::{
    descriptor_set::{BindlessContext, BindlessContextCreateInfo, LocalDescriptorSetCreateInfo},
    graph::{AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph},
    resource::{AccessTypes, Flight, ImageLayoutType, Resources, ResourcesCreateInfo},
    resource_map, Id, QueueFamilyType,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod deferred;
mod scene;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;

fn main() -> Result<(), impl Error> {
    // Basic initialization. See the triangle example if you want more details about this.

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    rcx: Option<RenderContext>,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    diffuse_image_id: Id<Image>,
    normals_image_id: Id<Image>,
    depth_image_id: Id<Image>,
    viewport: Viewport,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<Self>,
    virtual_swapchain_id: Id<Swapchain>,
    virtual_diffuse_image_id: Id<Image>,
    virtual_normals_image_id: Id<Image>,
    virtual_depth_image_id: Id<Image>,
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
            ..BindlessContext::required_extensions(&instance)
        };
        let mut device_features = BindlessContext::required_features(&instance);
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
                    && p.supported_features().contains(&device_features)
            })
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

        let resources = Resources::new(
            &device,
            &ResourcesCreateInfo {
                bindless_context: Some(&BindlessContextCreateInfo {
                    local_set: Some(&LocalDescriptorSetCreateInfo::default()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();

        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

        App {
            instance,
            device,
            queue,
            resources,
            flight_id,
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

        let (diffuse_image_id, normals_image_id, depth_image_id) =
            window_size_dependent_setup(&self.resources, swapchain_id);

        let mut task_graph = TaskGraph::new(&self.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: swapchain_format,
            ..Default::default()
        });
        // We create the virtual images with the `TRANSIENT_ATTACHMENT` usage to allow the task
        // graph compiler to optimize based on it. The physical images that are later assigned must
        // have been created with this usage.
        let virtual_diffuse_image_id = task_graph.add_image(&ImageCreateInfo {
            format: Format::A2B10G10R10_UNORM_PACK32,
            usage: ImageUsage::TRANSIENT_ATTACHMENT,
            ..Default::default()
        });
        let virtual_normals_image_id = task_graph.add_image(&ImageCreateInfo {
            format: Format::R16G16B16A16_SFLOAT,
            usage: ImageUsage::TRANSIENT_ATTACHMENT,
            ..Default::default()
        });
        let virtual_depth_image_id = task_graph.add_image(&ImageCreateInfo {
            format: Format::D16_UNORM,
            usage: ImageUsage::TRANSIENT_ATTACHMENT,
            ..Default::default()
        });
        let virtual_framebuffer_id = task_graph.add_framebuffer();

        let scene_node_id = task_graph
            .create_task_node(
                "Scene",
                QueueFamilyType::Graphics,
                SceneTask::new(
                    self,
                    virtual_diffuse_image_id,
                    virtual_normals_image_id,
                    virtual_depth_image_id,
                ),
            )
            .framebuffer(virtual_framebuffer_id)
            // We only need `COLOR_ATTACHMENT_WRITE` for the color attachments here because the
            // scene pipeline has color blending disabled, which would otherwise be a read as well.
            .color_attachment(
                virtual_diffuse_image_id,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    index: 0,
                    clear: true,
                    ..Default::default()
                },
            )
            .color_attachment(
                virtual_normals_image_id,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    index: 1,
                    clear: true,
                    ..Default::default()
                },
            )
            // We need both `DEPTH_STENCIL_ATTACHMENT_READ` and `DEPTH_STENCIL_ATTACHMENT_WRITE`
            // for the depth/stencil attachment, as the scene pipeline has both depth tests (read)
            // and depth writes (write) enabled.
            .depth_stencil_attachment(
                virtual_depth_image_id,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .build();
        let deferred_node_id = task_graph
            .create_task_node(
                "Deferred",
                QueueFamilyType::Graphics,
                DeferredTask::new(self, virtual_swapchain_id),
            )
            .framebuffer(virtual_framebuffer_id)
            // The deferred lighting pipelines have color blending enabled, so we need both
            // `COLOR_ATTACHMENT_READ` and `COLOR_ATTACHMENT_WRITE`.
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_READ | AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .input_attachment(
                virtual_diffuse_image_id,
                AccessTypes::FRAGMENT_SHADER_COLOR_INPUT_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    index: 0,
                    ..Default::default()
                },
            )
            .input_attachment(
                virtual_normals_image_id,
                AccessTypes::FRAGMENT_SHADER_COLOR_INPUT_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    index: 1,
                    ..Default::default()
                },
            )
            .input_attachment(
                virtual_depth_image_id,
                AccessTypes::FRAGMENT_SHADER_DEPTH_STENCIL_INPUT_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    index: 2,
                    ..Default::default()
                },
            )
            .build();

        task_graph
            .add_edge(scene_node_id, deferred_node_id)
            .unwrap();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.queue],
                present_queue: Some(&self.queue),
                flight_id: self.flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        let scene_node = task_graph.task_node_mut(scene_node_id).unwrap();
        let subpass = scene_node.subpass().unwrap().clone();
        scene_node
            .task_mut()
            .downcast_mut::<SceneTask>()
            .unwrap()
            .create_pipeline(self, &subpass);
        let deferred_node = task_graph.task_node_mut(deferred_node_id).unwrap();
        let subpass = deferred_node.subpass().unwrap().clone();
        deferred_node
            .task_mut()
            .downcast_mut::<DeferredTask>()
            .unwrap()
            .create_pipelines(self, &subpass);

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            diffuse_image_id,
            normals_image_id,
            depth_image_id,
            viewport,
            recreate_swapchain: false,
            task_graph,
            virtual_swapchain_id,
            virtual_diffuse_image_id,
            virtual_normals_image_id,
            virtual_depth_image_id,
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

                let flight = self.resources.flight(self.flight_id);

                if rcx.recreate_swapchain {
                    rcx.swapchain_id = self
                        .resources
                        .recreate_swapchain(rcx.swapchain_id, |create_info| SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..*create_info
                        })
                        .expect("failed to recreate swapchain");

                    self.resources
                        .create_deferred_batch()
                        .destroy_image(rcx.diffuse_image_id)
                        .destroy_image(rcx.normals_image_id)
                        .destroy_image(rcx.depth_image_id);

                    if !self.device.enabled_features().swapchain_maintenance1 {
                        self.resources.wait_idle().unwrap();
                    }

                    (
                        rcx.diffuse_image_id,
                        rcx.normals_image_id,
                        rcx.depth_image_id,
                    ) = window_size_dependent_setup(&self.resources, rcx.swapchain_id);

                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                }

                flight.wait(None).unwrap();

                let resource_map = resource_map!(
                    &rcx.task_graph,
                    rcx.virtual_swapchain_id => rcx.swapchain_id,
                    rcx.virtual_diffuse_image_id => rcx.diffuse_image_id,
                    rcx.virtual_normals_image_id => rcx.normals_image_id,
                    rcx.virtual_depth_image_id => rcx.depth_image_id,
                )
                .unwrap();

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
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    resources: &Resources,
    swapchain_id: Id<Swapchain>,
) -> (Id<Image>, Id<Image>, Id<Image>) {
    let swapchain_state = resources.swapchain(swapchain_id);
    let images = swapchain_state.images();
    let extent = images[0].extent();

    // Note that we create "transient" images here. This means that the content of the image is
    // only defined when within a render pass. In other words you can draw to them in a subpass
    // then read them in another subpass, but as soon as you leave the render pass their content
    // becomes undefined.
    let diffuse_image_id = resources
        .create_image(
            &ImageCreateInfo {
                extent,
                format: Format::A2B10G10R10_UNORM_PACK32,
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .unwrap();
    let normals_image_id = resources
        .create_image(
            &ImageCreateInfo {
                extent,
                format: Format::R16G16B16A16_SFLOAT,
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .unwrap();
    let depth_image_id = resources
        .create_image(
            &ImageCreateInfo {
                extent,
                format: Format::D16_UNORM,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .unwrap();

    (diffuse_image_id, normals_image_id, depth_image_id)
}
