// This example demonstrates how to use the standard and relative include directives within shader
// source code. The boilerplate is taken from the "basic-compute-shader.rs" example, where most of
// the boilerplate is explained.

use std::{slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorBufferInfo, DescriptorSet,
        WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
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

    let pipeline = {
        mod cs {
            vulkano_shaders::shader! {
                 ty: "compute",
                 // We declare what directories to search for when using the `#include <...>`
                 // syntax. Specified directories have descending priorities based on their order.
                 include: ["standard-shaders"],
                 src: r#"
                    #version 450

                    // Substitutes this line with the contents of the file `common.glsl` found in 
                    // one of the standard `include` directories specified above. Note that
                    // relative inclusion (`#include "..."`), although it falls back to standard
                    // inclusion, should not be used if you're not going to use the relative
                    // inclusion as it may be misleading and/or confusing.
                    #include <common.glsl>

                    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                    layout(set = 0, binding = 0) buffer Data {
                       uint data[];
                    };

                    void main() {
                       uint idx = gl_GlobalInvocationID.x;
                       data[idx] = multiply_by_12(data[idx]);
                    }
                "#,
            }
        }

        let cs = unsafe { cs::load(&device) }
            .unwrap()
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

    let data_buffer = Buffer::from_iter(
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
        0..65536u32,
    )
    .unwrap();

    let layout = &pipeline.layout().set_layouts()[0];
    let set = DescriptorSet::new(
        &descriptor_set_allocator,
        layout,
        &[WriteDescriptorSet::buffer(
            0,
            &DescriptorBufferInfo {
                buffer: Some(data_buffer.buffer()),
                ..Default::default()
            },
        )],
        &[],
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
    unsafe { builder.dispatch([1024, 1, 1]) }.unwrap();

    let command_buffer = builder.build().unwrap();
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], n * 12);
    }

    println!("Success");
}
