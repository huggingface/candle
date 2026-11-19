// This example demonstrates how to use dynamic uniform buffers.
//
// Dynamic uniform and storage buffers store buffer data for different calls in one large buffer.
// Each draw or dispatch call can specify an offset into the buffer to read object data from,
// without having to rebind descriptor sets.

use std::{iter, slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::{push_constant_ranges_from_stages, PipelineLayoutCreateInfo},
        ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::ShaderStages,
    sync::{self, GpuFuture},
    DeviceSize, VulkanLibrary,
};

fn main() {
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

                layout(local_size_x = 12) in;

                // Uniform buffer.
                layout(set = 0, binding = 0) uniform InData {
                    uint index;
                } ub;

                // Output buffer.
                layout(set = 0, binding = 1) buffer OutData {
                    uint data[];
                };

                // Toy shader that only runs for the index specified in `ub`.
                void main() {
                    uint index = gl_GlobalInvocationID.x;
                    if (index == ub.index) {
                        data[index] = index;
                    }
                }
            ",
        }
    }

    let pipeline = {
        let cs = unsafe { cs::load(&device) }
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(&cs);
        let layout = PipelineLayout::new(
            &device,
            &PipelineLayoutCreateInfo {
                set_layouts: &[&DescriptorSetLayout::new(
                    &device,
                    &DescriptorSetLayoutCreateInfo {
                        bindings: &[
                            DescriptorSetLayoutBinding {
                                binding: 0,
                                descriptor_count: 1,
                                stages: ShaderStages::COMPUTE,
                                ..DescriptorSetLayoutBinding::new(
                                    DescriptorType::UniformBufferDynamic,
                                )
                            },
                            DescriptorSetLayoutBinding {
                                binding: 1,
                                descriptor_count: 1,
                                stages: ShaderStages::COMPUTE,
                                ..DescriptorSetLayoutBinding::new(DescriptorType::StorageBuffer)
                            },
                        ],
                        ..Default::default()
                    },
                )
                .unwrap()],
                push_constant_ranges: &push_constant_ranges_from_stages(slice::from_ref(&stage)),
                ..Default::default()
            },
        )
        .unwrap();

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

    // Create the input buffer. Data in a dynamic buffer **MUST** be aligned to
    // `min_uniform_buffer_offset_align` or `min_storage_buffer_offset_align`, depending on the
    // type of buffer.
    let data: Vec<u32> = vec![3, 11, 7];
    let min_dynamic_align = device
        .physical_device()
        .properties()
        .min_uniform_buffer_offset_alignment
        .as_devicesize() as usize;

    println!("Minimum uniform buffer offset alignment: {min_dynamic_align}");
    println!("Input: {data:?}");

    // Round size up to the next multiple of align.
    let align = (size_of::<u32>() + min_dynamic_align - 1) & !(min_dynamic_align - 1);
    let aligned_data = {
        let mut aligned_data = Vec::with_capacity(align * data.len());

        for elem in data {
            let bytes = elem.to_ne_bytes();
            // Fill up the buffer with data.
            aligned_data.extend(bytes);
            // Zero out any padding needed for alignment.
            aligned_data.extend(iter::repeat_n(0, align - bytes.len()));
        }

        aligned_data
    };

    let input_buffer = Buffer::from_iter(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        aligned_data,
    )
    .unwrap();

    let output_buffer = Buffer::from_iter(
        &memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..12).map(|_| 0u32),
    )
    .unwrap();

    let layout = &pipeline.layout().set_layouts()[0];
    let set = DescriptorSet::new(
        &descriptor_set_allocator,
        layout,
        &[
            // When writing to the dynamic buffer binding, the range of the buffer that the shader
            // will access must also be provided. We specify the size of the `InData` struct here.
            // When dynamic offsets are provided later, they get added to the start and end of this
            // range.
            WriteDescriptorSet::buffer(
                0,
                &DescriptorBufferInfo {
                    buffer: Some(input_buffer.buffer()),
                    range: Some(size_of::<cs::InData>() as DeviceSize),
                    ..Default::default()
                },
            ),
            WriteDescriptorSet::buffer(
                1,
                &DescriptorBufferInfo {
                    buffer: Some(output_buffer.buffer()),
                    ..Default::default()
                },
            ),
        ],
        &[],
    )
    .unwrap();

    // Build the command buffer, using different offsets for each call.
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder.bind_pipeline_compute(pipeline.clone()).unwrap();

    for index in 0..3 {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone().offsets([index * align as u32]),
            )
            .unwrap();
        unsafe { builder.dispatch([12, 1, 1]) }.unwrap();
    }

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let output_content = output_buffer.read().unwrap();
    println!("Output: {:?}", &*output_content);
}
