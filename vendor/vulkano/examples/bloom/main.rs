// TODO: document

use bloom::BloomTask;
use scene::SceneTask;
use std::{array, cmp, env, error::Error, sync::Arc};
use tonemap::TonemapTask;
use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        Queue, QueueCreateInfo, QueueFlags,
    },
    format::{Format, NumericFormat},
    image::{
        max_mip_levels,
        sampler::{Filter, SamplerCreateInfo, SamplerMipmapMode},
        view::ImageViewCreateInfo,
        Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageSubresourceRange,
        ImageType, ImageUsage,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::AllocationCreateInfo,
    pipeline::graphics::viewport::Viewport,
    swapchain::{ColorSpace, Surface, Swapchain, SwapchainCreateInfo},
    Version, VulkanError, VulkanLibrary,
};
use vulkano_taskgraph::{
    descriptor_set::{BindlessContext, SampledImageId, SamplerId, StorageImageId},
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

mod bloom;
mod scene;
mod tonemap;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;
const MAX_BLOOM_MIP_LEVELS: u32 = 6;

fn main() -> Result<(), impl Error> {
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
    shader_language: ShaderLanguage,
    rcx: Option<RenderContext>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShaderLanguage {
    Glsl,
    Slang,
}

pub struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    bloom_image_id: Id<Image>,
    viewport: Viewport,
    recreate_swapchain: bool,
    bloom_sampler_id: SamplerId,
    bloom_sampled_image_id: SampledImageId,
    bloom_storage_image_ids: [StorageImageId; MAX_BLOOM_MIP_LEVELS as usize],
    task_graph: ExecutableTaskGraph<Self>,
    virtual_swapchain_id: Id<Swapchain>,
    virtual_bloom_image_id: Id<Image>,
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
                p.api_version() >= Version::V1_1 || p.supported_extensions().khr_maintenance2
            })
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
                    && p.supported_features().contains(&device_features)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags
                            .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
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

        if physical_device.api_version() < Version::V1_1 {
            device_extensions.khr_maintenance2 = true;
        }

        if physical_device.api_version() < Version::V1_2
            && physical_device.supported_extensions().khr_image_format_list
        {
            device_extensions.khr_image_format_list = true;
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
                bindless_context: Some(&Default::default()),
                ..Default::default()
            },
        )
        .unwrap();

        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

        let shader_language = match env::args().nth(1) {
            Some(arg) if arg == "slang" => ShaderLanguage::Slang,
            _ => ShaderLanguage::Glsl,
        };

        App {
            instance,
            device,
            queue,
            resources,
            flight_id,
            shader_language,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let bcx = self.resources.bindless_context().unwrap();

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
                .unwrap()
                .into_iter()
                .find(|&(format, color_space)| {
                    format.numeric_format_color() == Some(NumericFormat::SRGB)
                        && color_space == ColorSpace::SrgbNonLinear
                })
                .unwrap();

            self.resources
                .create_swapchain(
                    &surface,
                    &SwapchainCreateInfo {
                        min_image_count: surface_capabilities
                            .min_image_count
                            .max(MIN_SWAPCHAIN_IMAGES),
                        image_format: swapchain_format,
                        image_extent: window.inner_size().into(),
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

        let bloom_sampler_id = bcx
            .global_set()
            .create_sampler(&SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Nearest,
                ..Default::default()
            })
            .unwrap();

        let (bloom_image_id, bloom_sampled_image_id, bloom_storage_image_ids) =
            window_size_dependent_setup(&self.resources, swapchain_id);

        let mut task_graph = TaskGraph::new(&self.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: swapchain_format,
            ..Default::default()
        });
        let virtual_bloom_image_id = task_graph.add_image(&ImageCreateInfo {
            format: Format::E5B9G9R9_UFLOAT_PACK32,
            ..Default::default()
        });
        let virtual_framebuffer_id = task_graph.add_framebuffer();

        let scene_node_id = task_graph
            .create_task_node(
                "Scene",
                QueueFamilyType::Graphics,
                SceneTask::new(self, virtual_bloom_image_id),
            )
            .framebuffer(virtual_framebuffer_id)
            .color_attachment(
                virtual_bloom_image_id,
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    format: Format::R32_UINT,
                    ..Default::default()
                },
            )
            .build();
        let bloom_node_id = task_graph
            .create_task_node(
                "Bloom",
                QueueFamilyType::Compute,
                BloomTask::new(self, virtual_bloom_image_id),
            )
            .image_access(
                virtual_bloom_image_id,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ
                    | AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            )
            .build();
        let tonemap_node_id = task_graph
            .create_task_node("Tonemap", QueueFamilyType::Graphics, TonemapTask::new(self))
            .framebuffer(virtual_framebuffer_id)
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo::default(),
            )
            .image_access(
                virtual_bloom_image_id,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            )
            .build();

        task_graph.add_edge(scene_node_id, bloom_node_id).unwrap();
        task_graph.add_edge(bloom_node_id, tonemap_node_id).unwrap();

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
        let tonemap_node = task_graph.task_node_mut(tonemap_node_id).unwrap();
        let subpass = tonemap_node.subpass().unwrap().clone();
        tonemap_node
            .task_mut()
            .downcast_mut::<TonemapTask>()
            .unwrap()
            .create_pipeline(self, &subpass);

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            bloom_image_id,
            viewport,
            recreate_swapchain: false,
            bloom_sampler_id,
            bloom_sampled_image_id,
            bloom_storage_image_ids,
            task_graph,
            virtual_swapchain_id,
            virtual_bloom_image_id,
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

                    let mut batch = self.resources.create_deferred_batch();
                    batch.destroy_image(rcx.bloom_image_id);
                    batch.destroy_sampled_image(rcx.bloom_sampled_image_id);

                    for &id in &rcx.bloom_storage_image_ids {
                        batch.destroy_storage_image(id);
                    }

                    batch.enqueue();

                    if !self.device.enabled_features().swapchain_maintenance1 {
                        self.resources.wait_idle().unwrap();
                    }

                    (
                        rcx.bloom_image_id,
                        rcx.bloom_sampled_image_id,
                        rcx.bloom_storage_image_ids,
                    ) = window_size_dependent_setup(&self.resources, rcx.swapchain_id);

                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                }

                flight.wait(None).unwrap();

                let resource_map = resource_map!(
                    &rcx.task_graph,
                    rcx.virtual_swapchain_id => rcx.swapchain_id,
                    rcx.virtual_bloom_image_id => rcx.bloom_image_id,
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
) -> (
    Id<Image>,
    SampledImageId,
    [StorageImageId; MAX_BLOOM_MIP_LEVELS as usize],
) {
    let device = resources.device();
    let bcx = resources.bindless_context().unwrap();
    let swapchain_state = resources.swapchain(swapchain_id);
    let images = swapchain_state.images();
    let extent = images[0].extent();

    let bloom_image_mip_levels = cmp::min(MAX_BLOOM_MIP_LEVELS, max_mip_levels(extent));

    let bloom_image_id = {
        let view_formats = if device.api_version() >= Version::V1_2
            || device.enabled_extensions().khr_image_format_list
        {
            &[Format::R32_UINT, Format::E5B9G9R9_UFLOAT_PACK32] as &[_]
        } else {
            &[]
        };

        resources
            .create_image(
                &ImageCreateInfo {
                    flags: ImageCreateFlags::MUTABLE_FORMAT,
                    image_type: ImageType::Dim2d,
                    format: Format::R32_UINT,
                    view_formats,
                    extent,
                    mip_levels: bloom_image_mip_levels,
                    usage: ImageUsage::TRANSFER_DST
                        | ImageUsage::SAMPLED
                        | ImageUsage::STORAGE
                        | ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
            )
            .unwrap()
    };

    let bloom_image_state = resources.image(bloom_image_id);
    let bloom_image = bloom_image_state.image();

    let bloom_sampled_image_id = bcx
        .global_set()
        .create_sampled_image(
            bloom_image_id,
            &ImageViewCreateInfo {
                format: Format::E5B9G9R9_UFLOAT_PACK32,
                subresource_range: bloom_image.subresource_range(),
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
            ImageLayout::General,
        )
        .unwrap();

    let bloom_storage_image_ids = array::from_fn(|mip_level| {
        bcx.global_set()
            .create_storage_image(
                bloom_image_id,
                &ImageViewCreateInfo {
                    format: Format::R32_UINT,
                    subresource_range: ImageSubresourceRange {
                        aspects: ImageAspects::COLOR,
                        base_mip_level: cmp::min(mip_level as u32, max_mip_levels(extent) - 1),
                        ..Default::default()
                    },
                    usage: ImageUsage::STORAGE,
                    ..Default::default()
                },
                ImageLayout::General,
            )
            .unwrap()
    });

    (
        bloom_image_id,
        bloom_sampled_image_id,
        bloom_storage_image_ids,
    )
}
