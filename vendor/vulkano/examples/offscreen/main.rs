// Offscreen rendering example, renders a red triangle on a blue background to a buffer in memory
// then exports to a PNG. No swapchains here!

use std::{default::Default, fs::File, io::BufWriter, path::Path, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyImageToBufferInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    device::{physical::PhysicalDeviceType, Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
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
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    sync::GpuFuture,
    VulkanLibrary,
};

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the `triangle`
    // example if you haven't done so yet.

    let library = unsafe { VulkanLibrary::new() }.unwrap();

    let instance = Instance::new(
        &library,
        &InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .into_iter()
        // No need for swapchain extension support.
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::GRAPHICS))
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
        .expect("no suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, queues) = Device::new(
        &physical_device,
        &DeviceCreateInfo {
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

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    let vertices = [
        Vertex {
            position: [-0.5, -0.25],
        },
        Vertex {
            position: [0.0, 0.5],
        },
        Vertex {
            position: [0.25, -0.1],
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

    let format = Format::R8G8B8A8_UNORM;

    let render_pass = vulkano::single_pass_renderpass!(
        &device,
        attachments: {
            color: {
                format: format,
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

    // Create an offscreen image for rendering into.
    let render_output_image = Image::new(
        &memory_allocator,
        &ImageCreateInfo {
            format,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            extent: [1920, 1080, 1],
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
    )
    .unwrap();

    let render_output_image_view = ImageView::new_default(&render_output_image).unwrap();

    let framebuffer = Framebuffer::new(
        &render_pass,
        &FramebufferCreateInfo {
            // Attach the offscreen image to the framebuffer.
            attachments: &[&render_output_image_view],
            ..Default::default()
        },
    )
    .unwrap();

    let pipeline = {
        let vs = unsafe { vs::load(&device) }
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = unsafe { fs::load(&device) }
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(&vs),
            PipelineShaderStageCreateInfo::new(&fs),
        ];

        let layout = PipelineLayout::from_stages(&device, &stages).unwrap();

        let subpass = Subpass::new(&render_pass, 0).unwrap();

        GraphicsPipeline::new(
            &device,
            None,
            &GraphicsPipelineCreateInfo {
                stages: &stages,
                vertex_input_state: Some(&vertex_input_state),
                input_assembly_state: Some(&InputAssemblyState::default()),
                viewport_state: Some(&ViewportState {
                    viewports: &[Viewport {
                        offset: [0.0, 0.0],
                        extent: [1920.0, 1080.0],
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                    ..Default::default()
                }),
                rasterization_state: Some(&RasterizationState::default()),
                multisample_state: Some(&MultisampleState::default()),
                color_blend_state: Some(&ColorBlendState {
                    attachments: &[ColorBlendAttachmentState::default()],
                    ..Default::default()
                }),
                subpass: Some((&subpass).into()),
                ..GraphicsPipelineCreateInfo::new(&layout)
            },
        )
        .unwrap()
    };

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        &device,
        &Default::default(),
    ));

    // Host-accessible buffer where the offscreen image's contents are copied to after rendering.
    let render_output_buf = Buffer::from_iter(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..(1920 * 1080 * 4)).map(|_| 0u8),
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                // This framebuffer has the offscreen image attached to it.
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .unwrap()
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .unwrap();
    unsafe { builder.draw(vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

    builder.end_render_pass(Default::default()).unwrap();

    // The output image stores information in an unknown, non-linear layout, optimized for usage on
    // the device. This step copies the output image into a host-readable linear output buffer
    // where consecutive pixels in the image are laid out consecutively in memory.
    builder
        .copy_image_to_buffer(CopyImageToBufferInfo::new(
            render_output_image,
            render_output_buf.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let finished = command_buffer.clone().execute(queue.clone()).unwrap();

    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    // Access the bytes copied into the host-accessible output buffer by reference.
    let buffer_content = render_output_buf.read().unwrap();

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("triangle.png");
    let file = File::create(&path).unwrap();
    let w = &mut BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, 1920, 1080);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&buffer_content).unwrap();

    if let Ok(path) = path.canonicalize() {
        println!("Saved to {}", path.display());
    }
}
