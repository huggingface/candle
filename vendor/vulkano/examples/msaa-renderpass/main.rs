// Multisampling anti-aliasing example, using a render pass resolve.
//
// # Introduction to multisampling
//
// When you draw an object on an image, this object occupies a certain set of pixels. Each pixel of
// the image is either fully covered by the object, or not covered at all. There is no such thing
// as a pixel that is half-covered by the object that you're drawing. What this means is that you
// will sometimes see a "staircase effect" at the border of your object, also called aliasing.
//
// The root cause of aliasing is that the resolution of the image is not high enough. If you
// increase the size of the image you're drawing to, this effect will still exist but will be much
// less visible.
//
// In order to decrease aliasing, some games and programs use what we call "SuperSample Anti-
// Aliasing" (SSAA). For example instead of drawing to an image of size 1024x1024, you draw to an
// image of size 2048x2048. Then at the end, you scale down your image to 1024x1024 by merging
// nearby pixels. Since the intermediate image is 4 times larger than the destination, this would
// be 4x SSAA.
//
// However this technique is very expensive in terms of GPU power. The fragment shader and all its
// calculations has to run four times more often.
//
// So instead of SSAA, a common alternative is MSAA (MultiSample Anti-Aliasing). The base principle
// is more or less the same: you draw to an image of a larger dimension, and then at the end you
// scale it down to the final size. The difference is that the fragment shader is only run once per
// pixel of the final size, and its value is duplicated to fill to all the pixels of the
// intermediate image that are covered by the object.
//
// For example, let's say that you use 4x MSAA, you draw to an intermediate image of size
// 2048x2048, and your object covers the whole image. With MSAA, the fragment shader will only be
// run 1,048,576 times (1024 * 1024), compared to 4,194,304 times (2048 * 2048) with 4x SSAA. Then
// the output of each fragment shader invocation is copied in each of the four pixels of the
// intermediate image that correspond to each pixel of the final image.
//
// Now, let's say that your object doesn't cover the whole image. In this situation, only the
// pixels of the intermediate image that are covered by the object will receive the output of the
// fragment shader.
//
// Because of the way it works, this technique requires direct support from the hardware, contrary
// to SSAA which can be done on any machine.
//
// # Multisampled images
//
// Using MSAA with Vulkan is done by creating a regular image, but with a number of samples per
// pixel different from 1. For example if you want to use 4x MSAA, you should create an image with
// 4 samples per pixel. Internally this image will have 4 times as many pixels as its extent
// would normally require, but this is handled transparently for you. Drawing to a multisampled
// image is exactly the same as drawing to a regular image.
//
// However multisampled images have some restrictions, for example you can't show them on the
// screen (swapchain images are always single-sampled), and you can't copy them into a buffer.
// Therefore when you have finished drawing, you have to blit your multisampled image to a
// non-multisampled image. This operation is not a regular blit (blitting a multisampled image is
// an error), instead it is called *resolving* the image.

use std::{fs::File, io::BufWriter, path::Path, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyImageToBufferInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount},
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
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    sync::GpuFuture,
    VulkanLibrary,
};

fn main() {
    // The usual Vulkan initialization.
    let library = unsafe { VulkanLibrary::new() }.unwrap();
    let instance = Instance::new(
        &library,
        &InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
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

    // Creating our intermediate multisampled image.
    //
    // As explained in the introduction, we pass the same extent and format as for the final
    // image. But we also pass the number of samples-per-pixel, which is 4 here.
    let intermediary = ImageView::new_default(
        &Image::new(
            &memory_allocator,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [1024, 1024, 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                samples: SampleCount::Sample4,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    // This is the final image that will receive the anti-aliased triangle.
    let image = Image::new(
        &memory_allocator,
        &ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::TRANSFER_SRC
                | ImageUsage::TRANSFER_DST
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::STORAGE,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
    )
    .unwrap();

    let view = ImageView::new_default(&image).unwrap();

    // In this example, we are going to perform the *resolve* (ie. turning a multisampled image
    // into a non-multisampled one) as part of the render pass. This is the preferred method of
    // doing so, as it the advantage that the Vulkan implementation doesn't have to write the
    // content of the multisampled image back to memory at the end.
    let render_pass = vulkano::single_pass_renderpass!(
        &device,
        attachments: {
            // The first framebuffer attachment is the intermediary image.
            intermediary: {
                format: Format::R8G8B8A8_UNORM,
                // This has to match the image definition.
                samples: 4,
                load_op: Clear,
                store_op: DontCare,
            },
            // The second framebuffer attachment is the final image.
            color: {
                format: Format::R8G8B8A8_UNORM,
                // Same here, this has to match.
                samples: 1,
                load_op: DontCare,
                store_op: Store,
            },
        },
        pass: {
            // When drawing, we have only one output which is the intermediary image.
            //
            // At the end of the pass, each color attachment will be *resolved* into the image
            // given under `color_resolve`. In other words, here, at the end of the pass, the
            // `intermediary` attachment will be copied to the attachment named `color`.
            //
            // For depth/stencil attachments, there is also a `depth_stencil_resolve` field.
            // When you specify this, you must also specify at least one of the
            // `depth_resolve_mode` and `stencil_resolve_mode` fields.
            // We don't need that here, so it's skipped.
            color: [intermediary],
            color_resolve: [color],
            depth_stencil: {},
        },
    )
    .unwrap();

    // Creating the framebuffer, the calls to `add` match the list of attachments in order.
    let framebuffer = Framebuffer::new(
        &render_pass,
        &FramebufferCreateInfo {
            attachments: &[&intermediary, &view],
            ..Default::default()
        },
    )
    .unwrap();

    // Here is the "end" of the multisampling example, as starting from here everything is the same
    // as in any other example. The pipeline, vertex buffer, and command buffer are created in
    // exactly the same way as without multisampling. At the end of the example, we copy the
    // content of `image` (ie. the final image) to a buffer, then read the content of that buffer
    // and save it to a PNG file.

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

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    let vertices = [
        Vertex {
            position: [-0.5, -0.5],
        },
        Vertex {
            position: [0.0, 0.5],
        },
        Vertex {
            position: [0.5, -0.25],
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
                viewport_state: Some(&ViewportState::default()),
                rasterization_state: Some(&RasterizationState::default()),
                multisample_state: Some(&MultisampleState {
                    rasterization_samples: subpass.num_samples().unwrap(),
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
        extent: [1024.0, 1024.0],
        min_depth: 0.0,
        max_depth: 1.0,
    };

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        &device,
        &Default::default(),
    ));

    let buf = Buffer::from_iter(
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
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into()), None],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            Default::default(),
        )
        .unwrap()
        .set_viewport(0, [viewport].into_iter().collect())
        .unwrap()
        .bind_pipeline_graphics(pipeline)
        .unwrap()
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .unwrap();
    unsafe { builder.draw(vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

    builder
        .end_render_pass(Default::default())
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::new(image, buf.clone()))
        .unwrap();

    let command_buffer = builder.build().unwrap();
    let finished = command_buffer.execute(queue).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let buffer_content = buf.read().unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("triangle.png");
    let file = File::create(&path).unwrap();
    let w = &mut BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, 1024, 1024); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&buffer_content).unwrap();

    if let Ok(path) = path.canonicalize() {
        println!("Saved to {}", path.display());
    }
}
