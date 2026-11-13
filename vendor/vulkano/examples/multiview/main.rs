// This example demonstrates using the `VK_KHR_multiview` extension to render to multiple layers of
// the framebuffer in one render pass. This can significantly improve performance in cases where
// multiple perspectives or cameras are very similar like in virtual reality or other types of
// stereoscopic rendering where the left and right eye only differ in a small position offset.

use std::{fs::File, io::BufWriter, path::Path, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BufferImageCopy,
        CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{
        view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageType,
        ImageUsage, SampleCount,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
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
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass,
        SubpassDescription,
    },
    sync::{self, GpuFuture},
    VulkanLibrary,
};

fn main() {
    let library = unsafe { VulkanLibrary::new() }.unwrap();
    let instance = Instance::new(
        &library,
        &InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: &InstanceExtensions {
                // Required to get multiview limits.
                khr_get_physical_device_properties2: true,
                ..InstanceExtensions::empty()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        ..DeviceExtensions::empty()
    };
    let device_features = DeviceFeatures {
        // enabling the `multiview` feature will use the `VK_KHR_multiview` extension on Vulkan 1.0
        // and the device feature on Vulkan 1.1+.
        multiview: true,
        ..DeviceFeatures::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .into_iter()
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
                && p.supported_features().contains(&device_features)
        })
        .filter(|p| {
            // This example renders to two layers of the framebuffer using the multiview extension
            // so we check that at least two views are supported by the device. Not checking this
            // on a device that doesn't support two views will lead to a runtime error when
            // creating the `RenderPass`. The `max_multiview_view_count` function will return
            // `None` when the `VK_KHR_get_physical_device_properties2` instance extension has not
            // been enabled.
            p.properties().max_multiview_view_count.unwrap_or(0) >= 2
        })
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
        // A real application should probably fall back to rendering the framebuffer layers in
        // multiple passes when multiview isn't supported.
        .expect(
            "no device supports two multiview views or the \
            `VK_KHR_get_physical_device_properties2` instance extension has not been loaded",
        );

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
            enabled_extensions: &device_extensions,
            enabled_features: &device_features,
            ..Default::default()
        },
    )
    .unwrap();

    let [queue] = queues.try_into().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

    let image = Image::new(
        &memory_allocator,
        &ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::B8G8R8A8_SRGB,
            extent: [512, 512, 1],
            array_layers: 2,
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::COLOR_ATTACHMENT,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
    )
    .unwrap();

    let image_view = ImageView::new_default(&image).unwrap();

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

    // Note the `#extension GL_EXT_multiview : enable` that enables the multiview extension for the
    // shader and the use of `gl_ViewIndex` which contains a value based on which view the shader
    // is being invoked for. In this example `gl_ViewIndex` is used to toggle a hardcoded offset
    // for vertex positions but in a VR application you could easily use it as an index to a
    // uniform array that contains the transformation matrices for the left and right eye.
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450
                #extension GL_EXT_multiview : enable

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0) + gl_ViewIndex * vec4(0.25, 0.25, 0.0, 0.0);
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

    let render_pass = RenderPass::new(
        &device,
        &RenderPassCreateInfo {
            attachments: &[AttachmentDescription {
                format: image.format(),
                samples: SampleCount::Sample1,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                initial_layout: ImageLayout::ColorAttachmentOptimal,
                final_layout: ImageLayout::ColorAttachmentOptimal,
                ..Default::default()
            }],
            subpasses: &[SubpassDescription {
                // The view mask indicates which layers of the framebuffer should be rendered for
                // each subpass.
                view_mask: 0b11,
                color_attachments: &[Some(AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::ColorAttachmentOptimal,
                    ..Default::default()
                })],
                ..Default::default()
            }],
            // The correlated view masks indicate sets of views that may be more efficient to render
            // concurrently.
            correlated_view_masks: &[0b11],
            ..Default::default()
        },
    )
    .unwrap();

    let framebuffer = Framebuffer::new(
        &render_pass,
        &FramebufferCreateInfo {
            attachments: &[&image_view],
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
                        extent: [image.extent()[0] as f32, image.extent()[1] as f32],
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

    let create_buffer = || {
        Buffer::from_iter(
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
            (0..image.extent()[0] * image.extent()[1] * 4).map(|_| 0u8),
        )
        .unwrap()
    };

    let buffer1 = create_buffer();
    let buffer2 = create_buffer();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            Default::default(),
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline)
        .unwrap()
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .unwrap();

    // Drawing commands are broadcast to each view in the view mask of the active renderpass
    // which means only a single draw call is needed to draw to multiple layers of the
    // framebuffer.
    unsafe { builder.draw(vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

    builder.end_render_pass(Default::default()).unwrap();

    // Copy the image layers to different buffers to save them as individual images to disk.
    builder
        .copy_image_to_buffer(CopyImageToBufferInfo {
            regions: [BufferImageCopy {
                image_subresource: ImageSubresourceLayers {
                    base_array_layer: 0,
                    layer_count: Some(1),
                    ..image.subresource_layers()
                },
                image_extent: image.extent(),
                ..Default::default()
            }]
            .into(),
            ..CopyImageToBufferInfo::new(image.clone(), buffer1.clone())
        })
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo {
            regions: [BufferImageCopy {
                image_subresource: ImageSubresourceLayers {
                    base_array_layer: 1,
                    layer_count: Some(1),
                    ..image.subresource_layers()
                },
                image_extent: image.extent(),
                ..Default::default()
            }]
            .into(),
            ..CopyImageToBufferInfo::new(image.clone(), buffer2.clone())
        })
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    // Write each layer to its own file.
    write_image_buffer_to_file(
        buffer1,
        "multiview1.png",
        image.extent()[0],
        image.extent()[1],
    );
    write_image_buffer_to_file(
        buffer2,
        "multiview2.png",
        image.extent()[0],
        image.extent()[1],
    );
}

fn write_image_buffer_to_file(buffer: Subbuffer<[u8]>, path: &str, width: u32, height: u32) {
    let buffer_content = buffer.read().unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    let file = File::create(&path).unwrap();
    let w = &mut BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&buffer_content).unwrap();

    if let Ok(path) = path.canonicalize() {
        println!("Saved to {}", path.display());
    }
}
