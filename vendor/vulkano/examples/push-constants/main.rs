// Push constants are a small bank of values written directly to the command buffer and accessible
// in shaders. They allow the application to set values used in shaders without creating buffers or
// modifying and binding descriptor sets for each update. As a result, they are expected to
// outperform such memory-backed resource updates.

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

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 450

                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                layout(push_constant) uniform PushConstantData {
                    int multiple;
                    float addend;
                    bool enable;
                } pc;

                layout(set = 0, binding = 0) buffer Data {
                    uint data[];
                };

                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    if (pc.enable) {
                        data[idx] *= pc.multiple;
                        data[idx] += uint(pc.addend);
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
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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

    // The `vulkano_shaders::shaders!` macro generates a struct with the correct representation of
    // the push constants struct specified in the shader. Here we create an instance of the
    // generated struct.
    let push_constants = cs::PushConstantData {
        multiple: 1,
        addend: 1.0,
        enable: 1,
    };

    // For a compute pipeline, push constants are passed to the `dispatch` method. For a graphics
    // pipeline, push constants are passed to the `draw` and `draw_indexed` methods.
    //
    // Note that there is no type safety for the push constant data, so be careful to only pass an
    // instance of the struct generated by the `vulkano_shaders::shaders!` macro.
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
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, push_constants)
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
        assert_eq!(data_buffer_content[n as usize], n + 1);
    }
    println!("Success");
}
