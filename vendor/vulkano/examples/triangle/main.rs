// Welcome to the triangle example!
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

use std::{error::Error, slice, sync::Arc};
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
    render_pass::Subpass,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanError, VulkanLibrary,
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer,
    graph::{AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph},
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources},
    resource_map, ClearValues, Id, QueueFamilyType, Task, TaskContext,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;

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
    rcx: Option<RenderContext>,
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

        // The first step of any Vulkan program is to create an instance.
        //
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need to
        // enable manually. To do so, we ask `Surface` for the list of extensions required to draw
        // to a window.
        let mut instance_extensions = Surface::required_extensions(event_loop);

        // The `ext_surface_maintenance1` instance extension is a dependency of the
        // `ext_swapchain_maintenance1` device extension. See below for why we enable the latter
        // when it's available.
        if library.supported_extensions().ext_surface_maintenance1 {
            instance_extensions.ext_surface_maintenance1 = true;
        }

        // Now creating the instance.
        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations (e.g.,
                // MoltenVK).
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &instance_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        // Choose the device extensions that we're going to use. In order to present images to a
        // surface, we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        // We then choose which physical device to use. First, we enumerate all the available
        // physical devices, then apply filters to narrow them down to those that can support our
        // needs.
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .filter(|p| {
                // Some devices may not support the extensions or features that your application
                // requires, or report properties and limits that are not sufficient. These are
                // filtered out here.
                p.supported_extensions().contains(&device_extensions)
            })
            .filter_map(|p| {
                // For each physical device, we try to find a suitable queue family that will
                // execute our draw commands.
                //
                // Devices can provide multiple queues to run commands in parallel (for example, a
                // draw queue and a compute queue), similar to CPU threads. This is something you
                // have to manage manually in Vulkan. Queues of the same family have the same
                // properties.
                //
                // Here, we look for a single queue family that is suitable for our purposes. In a
                // real-world application, you may want to use a separate dedicated transfer queue
                // to handle data transfers in parallel with graphics operations.
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // We select a queue family that supports graphics operations. When drawing
                        // to a window surface, as we do in this example, we also need to check
                        // that queues in this queue family are capable of presenting images to a
                        // surface.
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop)
                    })
                    // The code here searches for the first queue family that is suitable. If none
                    // is found, `None` is returned to `filter_map`, which disqualifies this
                    // physical device.
                    .map(|i| (p, i as u32))
            })
            // All the physical devices that pass the filters above are suitable for the
            // application. However, not every device is equal; some are preferred over others.
            // Now, we assign each physical device a score, and pick the device with the lowest
            // ("best") score.
            //
            // In this example, we simply select the best-scoring device to use in the application.
            // In a real-world setting, you may want to use the best-scoring device only as a
            // "default" or "recommended" device, and let the user choose the device themself.
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let mut device_features = DeviceFeatures::empty();

        // We don't *need* the `ext_swapchain_maintenance1` device extension in the sense that
        // vulkano-taskgraph has a fallback for when this extension isn't present. However, it is
        // impossible to have sensible synchronization without this extension. vulkano-taskgraph
        // does the best that's possible, but it can't do miracles. The Vulkan specification is
        // hilariously broken when it comes to swapchains. That's why you should enable this
        // extension, which fixes those specification bugs, whenever possible.
        if physical_device
            .supported_extensions()
            .ext_swapchain_maintenance1
        {
            device_extensions.ext_swapchain_maintenance1 = true;

            // Some device extensions also have associated device features, and you have to enable
            // the device feature(s) in order to use the extension. `ext_swapchain_maintenance1` is
            // one such extension. From this arises an "amusing" possibility of a device supporting
            // the extension but none of its features, so we have to check for feature support.
            if physical_device.supported_features().swapchain_maintenance1 {
                device_features.swapchain_maintenance1 = true;
            }
        }

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // A vector of created queues is returned by the function alongside the device.
        let (device, queues) = Device::new(
            // Which physical device to connect to.
            &physical_device,
            &DeviceCreateInfo {
                // A list of optional features and extensions that our program needs to work
                // correctly. Some parts of the Vulkan specs are optional and must be enabled
                // manually at device creation. In this example, the only thing we are going to
                // need is the `khr_swapchain` extension that allows us to draw to a window.
                enabled_extensions: &device_extensions,
                enabled_features: &device_features,

                // The list of queues that we are going to use. Here we only use one queue from the
                // previously chosen queue family.
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();

        // Since we can request multiple queues, the `queues` variable is in fact a vector. We only
        // use one queue in this example, so we just retrieve the one and only element of the
        // vector.
        let [queue] = queues.try_into().unwrap();

        // We will use vulkano's "task graph", which is available through the vulkano-taskgraph
        // crate.
        //
        // The task graph is an optional abstraction built on top of vulkano. It simplifies parts
        // of the Vulkan API by providing a modular, node-based approach to structure and execute
        // GPU work.
        //
        // In order to use the task graph, we need to create a `Resources` collection. This will be
        // the container for our GPU resources, allowing the task graph to track their lifetime and
        // usage.
        let resources = Resources::new(&device, &Default::default()).unwrap();

        // Lastly, a "flight" is created.
        //
        // This is where the concept of "pipelining" comes into play. Rather than waiting for the
        // GPU to finish drawing each individual frame, it's preferred to start preparing the next
        // frame right away. This allows us to overlap CPU and GPU work, thus maximizing throughput
        // at the cost of some latency. Flights are the task graph's mechanism to do just that.
        //
        // The number of frames in flight puts a hard limit on how far the CPU side is allowed to
        // advance ahead of the GPU's execution before it needs to wait. Higher numbers increase
        // the size of this buffer, while a value of 1 effectively disables pipelining.
        //
        // We choose to use 2 frames in flight, which is the go-to for desktops. For mobile
        // devices, Arm recommends 3 frames in flight.
        //
        // One reason to choose a higher number is making the application more resilient to spikes
        // in frame time. A longer buffer allows the application to make up for missed frames, but
        // may increase latency noticeably. Memory usage is also increased since data transferred
        // to the GPU needs to be available for longer.
        //
        // On the other hand, a single frame in flight might be attractive for applications that
        // have very low workloads and thus don't benefit much from pipelining.
        //
        // A common misconception is that frames in flight and swapchain images are the same thing,
        // or that they serve the same purpose. This is not the case as there wouldn't be a need
        // for this separation then. You can use no swapchain, one swapchain, or however many
        // swapchains you need with the same flight. The swapchains can each have a different
        // amount of swapchain images. Swapchain images don't correspond to frames in flight (more
        // on this further down).
        //
        // What a flight *does* roughly correspond to is a thread. If you have 2 threads both
        // executing their own task graph, they should use different flights so as to not interfere
        // with each other. However, you can also use more than one flight on the same thread.
        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

        // The "render context" is left uninitialized for now. In order to set it up, we need a
        // window and swapchain first, which can be created once winit's event loop has started.
        let rcx = None;

        App {
            instance,
            device,
            queue,
            resources,
            flight_id,
            rcx,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // The objective of this example is to draw a triangle on a window. To do so, we first need
        // to create the window.
        //
        // Before we can render to a window, we must first create a `Surface` object from it, which
        // represents the drawable surface of a window. For that, we must wrap the `Window` in an
        // `Arc`.
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(&self.instance, &window).unwrap();
        let window_size = window.inner_size();

        // In order to draw on a surface, we need to create a "swapchain".
        //
        // Creating a swapchain allocates the swapchain images that will contain the image that
        // will ultimately be visible on the screen.
        let swapchain_format;
        let swapchain_id = {
            // Querying the capabilities of the surface. When we create the swapchain, we can only
            // pass values that are allowed by the capabilities.
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();

            // Choosing the internal format that the images will have.
            (swapchain_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            self.resources
                .create_swapchain(
                    &surface,
                    &SwapchainCreateInfo {
                        // We choose the lowest image count that the surface supports in order to
                        // minimize memory usage. However, we also take our minimum value of
                        // `MAX_FRAMES_IN_FLIGHT + 1` into account.
                        //
                        // We need at least as many images as frames in flight, otherwise not all
                        // frames would actually be in flight since there wouldn't be enough images
                        // for the device to work on at the same time. At least, that would be the
                        // case if only the host and device were involved.
                        //
                        // When it comes to presentation, there is a third party involved: the
                        // "Presentation Engine". It can be working on one swapchain image at a
                        // time, and its work is not necessarily in sync with the device, so we
                        // need one more swapchain image. With fewer swapchain images, we could be
                        // blocking on the host while acquiring the next image.
                        //
                        // Note that even if you had the same amount of swapchain images as frames
                        // in flight, they would still not be interchangeable. The order in which
                        // swapchain image indices are acquired is unpredictable, whereas the
                        // current frame index always cycles in the same way. Say for example that
                        // you have 3 frames and 3 swapchain images. Then this cycle might happen:
                        //
                        // frame_index 0 1 2 0 1 2 0 1 2
                        // image_index 0 1 0 1 2 0 2 0 1
                        //
                        // This means that the image index can only be used to index resources that
                        // are swapchain image-local, such as image views made from the swapchain
                        // image. This is especially obvious if you add a second swapchain to the
                        // mix because the image indices of the two will likely be acquired in a
                        // different order. The current frame index can be used to index everything
                        // that's frame-local, such as uniform buffers.
                        min_image_count: surface_capabilities
                            .min_image_count
                            .max(MIN_SWAPCHAIN_IMAGES),

                        image_format: swapchain_format,

                        // The size of the window, only used to initially setup the swapchain.
                        //
                        // NOTE:
                        // On some drivers, the swapchain extent is specified by
                        // `surface_capabilities.current_extent`, and the swapchain size must use
                        // this extent. This extent is always the same as the window size.
                        //
                        // However, other drivers don't specify a value, i.e.
                        // `surface_capabilities.current_extent` is `None`. These drivers will
                        // allow anything, but the only sensible value is the window size.
                        //
                        // Both of these cases need the swapchain to use the window size, so we
                        // just use that.
                        image_extent: window_size.into(),

                        image_usage: ImageUsage::COLOR_ATTACHMENT,

                        // The alpha mode indicates how the alpha value of the final image will
                        // behave. For example, you can choose whether the window will be opaque or
                        // transparent.
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

        // We will use a dynamic viewport, which allows us to recreate just the viewport when the
        // window is resized. Otherwise, we would have to recreate the whole graphics pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        // Now let's define what we want the GPU to do each frame by creating a task graph.
        //
        // This graph lets us structure our GPU work while taking care of resource synchronization
        // and resource cleanup.
        //
        // The task graph has a generic "world" type parameter. It can be used to pass shared data
        // to all nodes when executing the graph. We use the `RenderContext` as world, which
        // contains everything we need.
        let mut task_graph = TaskGraph::new(&self.resources);

        // Here we add a "virtual" swapchain.
        //
        // Virtual resources allow us to declare resources ahead of time and reference them in the
        // task graph. These can be thought of as placeholders that are filled in once the graph
        // is executed.
        //
        // Another way to think about the difference between a virtual and physical ID is that a
        // physical ID can always be used to retrieve the thing it's referencing. In the case of a
        // physical resource ID for instance, you can use it to get a resource from the `Resources`
        // collection. However, you can't use a virtual resource ID to retrieve the resource it's
        // referencing until task graph execution.
        //
        // In the case of our swapchain, this means that we can recreate the swapchain whenever
        // the window is resized without having to recreate and recompile the entire task graph.
        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: swapchain_format,
            ..Default::default()
        });

        // We also create a virtual framebuffer, which, unlike virtual resources, is not a stand-in
        // for a physical framebuffer.
        //
        // For one, the task graph creates its framebuffers and render passes internally, so you
        // don't need to specify these yourself.
        //
        // Also, a single virtual framebuffer doesn't necessarily correspond to a single physical
        // framebuffer. Instead, all a virtual framebuffer is for is a way to inform the task graph
        // that different nodes' attachments have the same dimensions. This allows the task graph
        // to combine different nodes into the same render pass, or even the same subpass.
        // Therefore, you should use the same virtual framebuffer across nodes that share the same
        // framebuffer dimensions.
        let virtual_framebuffer_id = task_graph.add_framebuffer();

        // Next, we instantiate our `TriangleTask`.
        //
        // A "task" defines some work that we want the GPU to perform. A task is only useful when
        // it is inserted into the task graph as part of a task node.
        //
        // A "task node" contains a task alongside information about how it needs to be
        // synchronized. It is a unit of work that can be independently scheduled and synchronized
        // by the task graph.
        //
        // When creating a task node, we need to be explicit about how each resource is accessed.
        // This way, the task graph can ensure that accesses are correctly synchronized and images
        // are transitioned into the layout that we expect.
        let triangle_node_id = task_graph
            .create_task_node(
                // The name of the node.
                "Triangle",
                // Which type of queue family should this task run on?
                QueueFamilyType::Graphics,
                // The task to execute.
                TriangleTask::new(self, virtual_swapchain_id),
            )
            // We bind the framebuffer that we want to use...
            .framebuffer(virtual_framebuffer_id)
            // ...and add the current swapchain image as a color attachment.
            .color_attachment(
                // `current_image_id()` means that this color attachment will always use the
                // currently acquired swapchain image.
                virtual_swapchain_id.current_image_id(),
                // We only need `COLOR_ATTACHMENT_WRITE` for the color attachment because our
                // graphics pipeline has color blending disabled, which would otherwise be a read
                // as well.
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    // We want to clear the color attachment before drawing.
                    clear: true,
                    ..Default::default()
                },
            )
            .build();

        // Note that we only need to reason about the accesses of one task node in isolation. This
        // is one of the benefits of using the task graph to structure our work.
        //
        // The graph in this example has only a single node. However, most real world use cases
        // will have multiple instead. In that case, you can specify dependencies by adding edges
        // between nodes.

        // Once the graph is built, it's time to compile it.
        //
        // This step turns the graph into an executable form, producing a linear sequence of
        // instructions to execute. During compilation, the task graph chooses the order in which
        // nodes are executed such that synchronization overhead is minimized.
        //
        // All of this is done ahead of time, so executing the graph becomes as efficient as
        // possible. This does not necessarily mean that compiling the task graph each time it's
        // executed is inefficient. The AOT (ahead-of-time) compilation can squeeze out the most
        // performance out of the device regardless of how often it is done; it just results in
        // slightly more host-side overhead. Compiling the task graph each time it's executed, or
        // however often you need, is a perfectly valid strategy if that's what you need.
        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                // We need to provide all queues that we want to use for executing the graph. The
                // queue family types that were specified in the task nodes must be compatible with
                // these queues.
                //
                // In this example, we only have a single graphics queue.
                queues: &[&self.queue],
                // We use the same queue for presentation. You must specify a present queue if your
                // task graph uses any swapchains.
                present_queue: Some(&self.queue),
                // The flight that we use to track each execution of this task graph.
                flight_id: self.flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        // The triangle node requires a subpass for its graphics pipeline. We can only access the
        // subpass after the task graph is compiled.
        let triangle_node = task_graph.task_node_mut(triangle_node_id).unwrap();
        let subpass = triangle_node.subpass().unwrap().clone();
        triangle_node
            .task_mut()
            .downcast_mut::<TriangleTask>()
            .unwrap()
            .create_pipeline(self, &subpass);

        // In some situations, the swapchain will become invalid by itself. This includes for
        // example when the window is resized (as the images of the swapchain will no longer match
        // the window's).
        //
        // In this situation, acquiring a swapchain image or presenting it will return an error.
        // Rendering to an image of that swapchain will not produce any error, but may or may not
        // work. To continue rendering, we need to recreate the swapchain by creating a new
        // swapchain. Here, we remember that we need to do this for the next loop iteration.
        let recreate_swapchain = false;

        // We finish by setting the render context.
        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            viewport,
            recreate_swapchain,
            task_graph,
            virtual_swapchain_id,
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
                // We want the swapchain to be the same size as the window. Rather than resizing it
                // immediately, we set a flag to resize it during the next frame. This prevents
                // resizing it for every `Resized` event, which may be multiple times per frame.
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                // Do not draw the frame when the screen size is zero. On Windows, this can occur
                // when minimizing the application. In Vulkan, it is not allowed to create images
                // that have a width, height (or depth) of zero.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // Whenever the window resizes, we need to recreate everything dependent on the
                // window size. In this example, that includes the swapchain and the viewport.
                if rcx.recreate_swapchain {
                    // Use the new dimensions of the window.

                    rcx.swapchain_id = self
                        .resources
                        .recreate_swapchain(rcx.swapchain_id, |create_info| SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..*create_info
                        })
                        .expect("failed to recreate swapchain");

                    // Without the `swapchain_maintenance1` device feature, the task graph does the
                    // best fallback available, which doesn't involve any blocking on the CPU just
                    // like with `swapchain_maintenance1`. However, the drawback is that when the
                    // user resizes the window fast enough so that there is only one frame drawn
                    // per resize event, the old swapchains accumulate and can't be destroyed until
                    // the user stops. This is impossible to avoid without `swapchain_maintenance1`
                    // or a wait for device idle. So you have the choice of using
                    // `swapchain_maintenance1`, using the fallback with a wait for device idle, or
                    // using the fallback and ignoring the possible degenerate scenario. All of
                    // them work, but only `swapchain_maintenance1` has no drawbacks.
                    if !self.device.enabled_features().swapchain_maintenance1 {
                        self.resources.wait_idle().unwrap();
                    }

                    rcx.viewport.extent = window_size.into();

                    rcx.recreate_swapchain = false;
                }

                // Wait for the oldest frame in flight to finish executing on the GPU.
                let flight = self.resources.flight(self.flight_id);
                flight.wait(None).unwrap();

                // Remember that we have used a virtual swapchain in our task graph. Now that we
                // want to execute the graph, we need to map each virtual ID to the ID of an
                // existing resource.
                //
                // The `resource_map!` macro is a convenient way to map one or more IDs.
                let resource_map =
                    resource_map!(&rcx.task_graph, rcx.virtual_swapchain_id => rcx.swapchain_id)
                        .unwrap();

                // Finally, it is time to execute the graph.
                match unsafe {
                    rcx.task_graph
                        .execute(resource_map, rcx, || rcx.window.pre_present_notify())
                } {
                    Ok(()) => {}
                    // Since the task graph also handles presenting to the swapchain, it may return
                    // a swapchain error. When the swapchain is "out of date", we set a flag to
                    // recreate it during the next frame.
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

// This type represents the logic of our "draw a triangle" task and the data associated with it.
struct TriangleTask {
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer_id: Id<Buffer>,
    swapchain_id: Id<Swapchain>,
}

impl TriangleTask {
    fn new(app: &mut App, swapchain_id: Id<Swapchain>) -> Self {
        // This is the array of vertices we are going to draw.
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

        // Allocate the Vulkan buffer that will hold the vertices.
        //
        // Since we are using vulkano's task graph, the buffer is created using the `Resources`
        // collection.
        let vertex_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    // We are going to bind this buffer as a vertex buffer.
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    // We want the buffer to be located on the device (GPU) so it is fast to access
                    // from shaders. It must also be writable from the host side (CPU) to initially
                    // upload the data.
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                // The device layout determines the size and alignment of the buffer.
                DeviceLayout::for_value(vertices.as_slice()).unwrap(),
            )
            .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &app.queue,
                &app.resources,
                app.flight_id,
                |_cbf, tcx| {
                    tcx.try_write_buffer::<[MyVertex]>(vertex_buffer_id, ..)?
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        // As mentioned earlier, the pipeline depends on the subpass, which is only created once
        // the task graph is compiled. The pipeline field is initialized below.
        let pipeline = None;

        Self {
            pipeline,
            vertex_buffer_id,
            swapchain_id,
        }
    }

    pub fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        // The next step is to create the shaders.
        //
        // The raw shader creation API provided by the vulkano library is unsafe for various
        // reasons, so the `shader!` macro provides a way to generate a Rust module from shader
        // source. In the example below, the source is provided as a string input directly to the
        // shader, but a path to a source file can be provided as well. Note that the user must
        // specify the type of shader (e.g., "vertex", "fragment", etc.) using the `ty` option of
        // the macro.
        //
        // The items generated by the `shader!` macro include a `load` function which loads the
        // shader using a logical device. The module also includes structs compatible with the ones
        // defined in the shader source, such as uniforms and push constants for example.
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
                        f_color = vec4(251.0 / 255.0, 113.0 / 255.0, 133.0 / 255.0, 1.0);
                    }
                ",
            }
        }

        // Before we draw, we have to create what is called a "pipeline". A pipeline describes how
        // a GPU operation is to be performed. It is similar to an OpenGL program, but it also
        // contains many settings for customization, all baked into a single object. For drawing
        // triangles, we create a graphics pipeline, but there are also other types of pipelines.
        let pipeline = {
            // First, we load the shaders that the pipeline will use: the vertex shader and the
            // fragment shader.
            //
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one to use.
            let vs = unsafe { vs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { fs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();

            // Automatically generate a vertex input state from the vertex shader's input
            // interface that takes a single vertex buffer containing `Vertex` structs.
            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];

            // We must now create a "pipeline layout" object, which describes the locations and
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
            let layout = PipelineLayout::from_stages(&app.device, &stages).unwrap();

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                &app.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(&vertex_input_state),
                    // How vertices are arranged into primitive shapes. The default primitive shape
                    // is a triangle.
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    // How primitives are transformed and clipped to fit the framebuffer. We use a
                    // resizable viewport set to draw over the entire window.
                    viewport_state: Some(&ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels. The default
                    // value does not perform any culling.
                    rasterization_state: Some(&RasterizationState::default()),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(&MultisampleState::default()),
                    // How pixel values are combined with the values already present in the
                    // framebuffer. The default value overwrites the old value with the new one
                    // without any blending.
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState::default()],
                        ..Default::default()
                    }),
                    // Dynamic state allows us to specify parts of the pipeline settings when
                    // recording the command buffer, before we perform drawing. Here, we specify
                    // that the viewport should be dynamic.
                    dynamic_state: &[DynamicState::Viewport],
                    // We have to indicate which subpass of which render pass this pipeline is
                    // going to be used in. The pipeline will only be usable from this particular
                    // subpass.
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        self.pipeline = Some(pipeline);
    }
}

// The `Task` trait defines the logic of a task in the task graph.
impl Task for TriangleTask {
    type World = RenderContext;

    fn clear_values(&self, clear_values: &mut ClearValues<'_>, _world: &Self::World) {
        // Earlier, we requested that the color attachment of the task node should be cleared. This
        // method is where we specify the clear values that should be used.

        clear_values.set(
            self.swapchain_id.current_image_id(),
            [2.0 / 255.0, 6.0 / 255.0, 24.0 / 255.0, 1.0],
        );
    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        // This method is called when the task graph executes the task node. Here, we record all
        // GPU commands to execute as part of this task.

        // Update the dynamic viewport, which is set to the current window and swapchain size.
        cbf.set_viewport(0, slice::from_ref(&rcx.viewport));

        // Bind the graphics pipeline and vertex buffer.
        cbf.bind_pipeline(self.pipeline.as_ref().unwrap());
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]);

        // Draw the triangle using one instance of our three vertices.
        unsafe { cbf.draw(3, 1, 0, 0) };

        // If you are familiar with Vulkan, you will notice that we have performed no manual
        // synchronization here. This is handled entirely by the task graph as long as we have
        // specified all resources that we want to access when creating the task node.

        Ok(())
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data as the default
// representation has *no guarantees*.
#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    // We need to set a GPU compatible format for each vertex attribute.
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
