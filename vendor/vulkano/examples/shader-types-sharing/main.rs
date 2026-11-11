// This example demonstrates how to compile several shaders together using the `shader!` macro,
// such that the macro doesn't generate unique shader types for each compiled shader, but generates
// a common shareable set of Rust structs for the corresponding structs in the shaders.
//
// Normally, each `shader!` macro invocation among other things generates all Rust types for each
// `struct` declaration of the GLSL code. Using these the user can organize type-safe
// interoperability between Rust code and the shader input/output interface tied to these structs.
// However, if the user compiles several shaders in independent Rust modules, each of these modules
// would contain an independent set of Rust types. So, even if both shaders contain the same (or
// partially intersecting) GLSL structs they will be duplicated by each macro invocation and
// treated by Rust as independent types. As such it would be tricky to organize interoperability
// between shader interfaces in Rust.
//
// To solve this problem the user can use "shared" generation mode of the macro. In this mode the
// user declares all shaders that possibly share common layout interfaces in a single macro
// invocation. The macro will check that there is no inconsistency between declared GLSL structs
// with the same names, and it will not generate duplicates.

use std::{slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorBufferInfo, DescriptorSet,
        WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
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
        physical_device.properties().device_type
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

    mod shaders {
        vulkano_shaders::shader! {
            // We declaring two simple compute shaders with push and specialization constants in
            // their layout interfaces.
            //
            // First one is just multiplying each value from the input array of ints to provided
            // value in push constants struct. And the second one in turn adds this value instead
            // of multiplying.
            //
            // However both shaders declare glsl struct `Parameters` for push constants in each
            // shader. Since each of the struct has exactly the same interface, they will be
            // treated by the macro as "shared".
            //
            // Also, note that GLSL code duplications between shader sources is not necessary too.
            // In a more complex system the user may want to declare an independent GLSL file with
            // such types, and include it in each shader entry-point file using the `#include`
            // directive.
            shaders: {
                mult: {
                    ty: "compute",
                    src: r"
                        #version 450

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        layout(constant_id = 0) const bool enabled = true;

                        layout(push_constant) uniform Parameters {
                            int value;
                        } pc;

                        layout(set = 0, binding = 0) buffer Data {
                            uint data[];
                        };

                        void main() {
                            if (!enabled) {
                                return;
                            }
                            uint idx = gl_GlobalInvocationID.x;
                            data[idx] *= pc.value;
                        }
                    ",
                },
                add: {
                    ty: "compute",
                    src: r"
                        #version 450

                        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                        layout(constant_id = 0) const bool enabled = true;

                        layout(push_constant) uniform Parameters {
                            int value;
                        } pc;

                        layout(set = 0, binding = 0) buffer Data {
                            uint data[];
                        };

                        void main() {
                            if (!enabled) {
                                return;
                            }
                            uint idx = gl_GlobalInvocationID.x;
                            data[idx] += pc.value;
                        }
                    ",
                },
            },
        }

        // The macro will create the following things in this module:
        // - `load_mult` for the first shader loader/entry-point.
        // - `load_add` for the second shader loader/entry-point.
        // - `Parameters` struct common for both shaders.
    }

    /// We are introducing a generic function responsible for running any of the shaders above with
    /// the provided push constants parameter. Note that the shaders' interface `parameters` here
    /// are shader-independent.
    fn run_shader(
        pipeline: &Arc<ComputePipeline>,
        queue: &Arc<Queue>,
        data_buffer: &Subbuffer<[u32]>,
        parameters: shaders::Parameters,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    ) {
        let layout = &pipeline.layout().set_layouts()[0];
        let set = DescriptorSet::new(
            descriptor_set_allocator,
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
            command_buffer_allocator.clone(),
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
            .push_constants(pipeline.layout().clone(), 0, parameters)
            .unwrap();
        unsafe { builder.dispatch([1024, 1, 1]) }.unwrap();

        let command_buffer = builder.build().unwrap();
        let future = sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();
    }

    let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        &device,
        &Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        &device,
        &Default::default(),
    ));

    // Prepare test array `[0, 1, 2, 3....]`.
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

    // Load the first shader, and create a pipeline for the shader.
    let mult_pipeline = {
        let cs = unsafe { shaders::load_mult(&device) }
            .unwrap()
            .specialize(&[(0, true.into())])
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

    // Load the second shader, and create a pipeline for the shader.
    let add_pipeline = {
        let cs = unsafe { shaders::load_add(&device) }
            .unwrap()
            .specialize(&[(0, true.into())])
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

    // Multiply each value by 2.
    run_shader(
        &mult_pipeline,
        &queue,
        &data_buffer,
        shaders::Parameters { value: 2 },
        &descriptor_set_allocator,
        &command_buffer_allocator,
    );

    // Then add 1 to each value.
    run_shader(
        &add_pipeline,
        &queue,
        &data_buffer,
        shaders::Parameters { value: 1 },
        &descriptor_set_allocator,
        &command_buffer_allocator,
    );

    // Then multiply each value by 3.
    run_shader(
        &mult_pipeline,
        &queue,
        &data_buffer,
        shaders::Parameters { value: 3 },
        &descriptor_set_allocator,
        &command_buffer_allocator,
    );

    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], (n * 2 + 1) * 3);
    }
    println!("Success");
}
