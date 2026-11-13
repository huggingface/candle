// This example demonstrates how to define the compute shader local size layout at runtime through
// specialization constants while considering the physical device properties.
//
// Workgroup parallelism capabilities vary between GPUs and setting them properly is important to
// achieve the maximal performance that particular device can provide.

use std::{fs::File, io::BufWriter, path::Path, slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyImageToBufferInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorImageInfo, DescriptorSet,
        WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, ComputePipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
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
                // This extension is required to obtain physical device metadata about the device
                // workgroup size limits.
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
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .into_iter()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
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

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 450

                // We set `local_size_x` and `local_size_y` to be variables configurable values 
                // through specialization constants. Values `1` and `2` both define a constant ID 
                // as well as a default value of 1 and 2 of the constants respectively. The 
                // `local_size_z = 1` here is an ordinary constant of the local size on the Z axis.
                //
                // Unfortunately current GLSL language capabilities doesn't let us define exact 
                // names of the constants so we will have to use anonymous constants instead. See 
                // below for how to provide their values at runtime.
                //
                // NOTE: The constant ID in `local_size` layout must be positive values. Zeros lead 
                // to runtime failure on NVIDIA devices due to a known bug in the driver.
                layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z = 1) in;

                // We can still define more constants in the Shader
                layout(constant_id = 0) const float red = 0.0;
                layout(constant_id = 3) const float green = 0.0;
                layout(constant_id = 4) const float blue = 0.0;

                layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

                void main() {
                    // Colorful Mandelbrot fractal.

                    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
                    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

                    vec2 z = vec2(0.0, 0.0);
                    float i;
                    for (i = 0.0; i < 1.0; i += 0.005) {
                        z = vec2(
                            z.x * z.x - z.y * z.y + c.x,
                            z.y * z.x + z.x * z.y + c.y
                        );

                        if (length(z) > 4.0) {
                            break;
                        }
                    }

                    vec4 to_write = vec4(vec3(red, green, blue) * i, 1.0);

                    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
                }
            ",
        }
    }

    // Fetching subgroup size from the physical device properties to determine an appropriate
    // compute shader local size.
    //
    // Most of the drivers provide this property, but some of the drivers don't. In that case we
    // can find an appropriate value using this tool: https://vulkan.gpuinfo.org, or just use a
    // fallback constant for simplicity, but failure to set a proper local size can lead to a
    // significant performance penalty.
    let (local_size_x, local_size_y) = match device.physical_device().properties().subgroup_size {
        Some(subgroup_size) => {
            println!("Subgroup size is {subgroup_size}");

            // Most of the subgroup values are divisors of 8.
            (8, subgroup_size / 8)
        }
        None => {
            println!("This Vulkan driver doesn't provide physical device Subgroup information");

            // Using a fallback constant.
            (8, 8)
        }
    };

    println!("Local size will be set to: ({local_size_x}, {local_size_y}, 1)");

    let pipeline = {
        let cs = unsafe { cs::load(&device) }
            .unwrap()
            .specialize(&[
                (0, 0.2f32.into()),
                (1, local_size_x.into()),
                (2, local_size_y.into()),
                (3, 0.5f32.into()),
                (4, 1.0f32.into()),
            ])
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

    let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        &device,
        &Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        &device,
        &Default::default(),
    ));

    let image = Image::new(
        &memory_allocator,
        &ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::STORAGE,
            ..Default::default()
        },
        &AllocationCreateInfo::default(),
    )
    .unwrap();
    let view = ImageView::new_default(&image).unwrap();

    let layout = &pipeline.layout().set_layouts()[0];
    let set = DescriptorSet::new(
        &descriptor_set_allocator,
        layout,
        &[WriteDescriptorSet::image(
            0,
            &DescriptorImageInfo {
                image_view: Some(&view),
                ..Default::default()
            },
        )],
        &[],
    )
    .unwrap();

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
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap();

    // Note that dispatch dimensions must be proportional to the local size.
    unsafe { builder.dispatch([1024 / local_size_x, 1024 / local_size_y, 1]) }.unwrap();

    builder
        .copy_image_to_buffer(CopyImageToBufferInfo::new(image, buf.clone()))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    println!("Success");

    let buffer_content = buf.read().unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("mandelbrot.png");
    let file = File::create(&path).unwrap();
    let w = &mut BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, 1024, 1024);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&buffer_content).unwrap();

    if let Ok(path) = path.canonicalize() {
        println!("Saved to {}", path.display());
    }
}
