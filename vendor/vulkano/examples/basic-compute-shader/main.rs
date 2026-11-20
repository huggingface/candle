// This example demonstrates how to use the compute capabilities of Vulkan.
//
// While graphics cards have traditionally been used for graphical operations, over time they have
// been more or more used for general-purpose operations as well. This is called "General-Purpose
// GPU", or *GPGPU*. This is what this example demonstrates.

use std::slice;
use vulkano::{
    buffer::{BufferCreateInfo, BufferUsage},
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        compute::ComputePipelineCreateInfo, ComputePipeline, PipelineShaderStageCreateInfo,
    },
    VulkanLibrary,
};
use vulkano_taskgraph::{
    descriptor_set::{BindlessContext, BindlessContextCreateInfo},
    resource::{AccessTypes, HostAccessType, Resources, ResourcesCreateInfo},
};

// The compute shader we are going to run.
mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            #include <vulkano.glsl>

            VKO_DECLARE_STORAGE_BUFFER(buffer, Buffer {
                uint data[];
            })

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(push_constant) uniform PushConstants {
                StorageBufferId buffer_id;
                uint buffer_len;
            };

            #define buffer vko_buffer(buffer, buffer_id)

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                // Because we dispatch a multiple of 64 threads (work group size) it's usually
                // required to guard against accessing buffers or storage images out of bounds.
                if (idx >= buffer_len) {
                    return;
                }
                buffer.data[idx] *= 12;
            }
        ",
    }
}

fn main() {
    // As with other examples, the first step is to create an instance.
    let library = unsafe { VulkanLibrary::new() }.unwrap();
    let instance = Instance::new(
        &library,
        &InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    // Choose which physical device to use.
    //
    // We make use of vulkano's bindless feature to handle resource bindings. This way we don't
    // need to worry about managing descriptors. Bindless requires additional extensions and
    // features.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..BindlessContext::required_extensions(&instance)
    };
    let device_features = BindlessContext::required_features(&instance);
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .into_iter()
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
                && p.supported_features().contains(&device_features)
        })
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one
            // queue that supports compute operations.
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

    // Now initializing the device.
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

    // Since we can request multiple queues, the `queues` variable is in fact a vector. We only use
    // one queue in this example, so we just retrieve the one and only element of the vector.
    let [queue] = queues.try_into().unwrap();

    // Now let's get to the actual example.
    //
    // What we are going to do is very basic: we are going to fill a buffer with 64k integers and
    // ask the GPU to multiply each of them by 12.
    //
    // GPUs are very good at parallel computations (SIMD-like operations), and thus will do this
    // much more quickly than a CPU would do. While a CPU would typically multiply them one by one
    // or four by four, a GPU will do it by groups of 32 or 64.
    //
    // Note however that in a real-life situation for such a simple operation the cost of accessing
    // memory usually outweighs the benefits of a faster calculation. Since both the CPU and the
    // GPU will need to access data, there is no other choice but to transfer the data through the
    // slow PCI express bus.

    // The `Resources` type is used in conjunction with the task graph and tracks available
    // resources for automatic synchronization and cleanup.
    let resources = Resources::new(
        &device,
        &ResourcesCreateInfo {
            bindless_context: Some(&BindlessContextCreateInfo::default()),
            ..Default::default()
        },
    )
    .unwrap();

    // We'll use the bindless context to bind our buffer.
    let bcx = resources.bindless_context().unwrap();

    // We need to create the compute pipeline that describes our operation.
    //
    // If you are familiar with graphics pipeline, the principle is the same except that compute
    // pipelines are much simpler to create.
    let pipeline = {
        let module = unsafe { compute_shader::load(&device) }
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(&module);
        let layout = bcx
            .pipeline_layout_from_stages(slice::from_ref(&stage))
            .unwrap();

        ComputePipeline::new(
            &device,
            None,
            &ComputePipelineCreateInfo::new(stage, &layout),
        )
        .unwrap()
    };

    // Create a flight.
    //
    // We're going to execute a one-shot compute shader, so a single frame in flight is enough.
    let flight_id = resources.create_flight(1).unwrap();

    // Create a storage buffer.
    //
    // This example reads and writes the same buffer from a compute shader. The buffer also needs
    // to be accessible from the host to copy the initial data into it and read the result back
    // later.
    const BUFFER_LEN: u32 = 65536;
    let buffer_id = resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            DeviceLayout::new_unsized::<[u32]>(BUFFER_LEN.into()).unwrap(),
        )
        .unwrap();

    // Add the storage buffer to the bindless context.
    //
    // We can send this ID to our compute shader and use it to access the buffer.
    let buffer_bindless_id = bcx
        .global_set()
        .create_storage_buffer(buffer_id, 0, None)
        .unwrap();

    // We can use `vulkano_taskgraph::execute` to run our simple workload as a single "task node".
    //
    // It's important to specify all resource accesses we want to do within the task.
    unsafe {
        vulkano_taskgraph::execute(
            &queue,
            &resources,
            flight_id,
            |cbf, tcx| {
                // Initialize the buffer with our data.
                for (i, value) in (0..).zip(tcx.write_buffer::<[u32]>(buffer_id, ..)) {
                    *value = i;
                }

                cbf.bind_pipeline(&pipeline);
                cbf.push_constants(
                    pipeline.layout(),
                    0,
                    &compute_shader::PushConstants {
                        buffer_id: buffer_bindless_id,
                        buffer_len: BUFFER_LEN,
                    },
                );

                // We have set the local size of the shader to (64, 1, 1). Each thread processes
                // one item in the buffer, so ceil(n / 64) groups are required.
                let groups_x = BUFFER_LEN.div_ceil(64);
                cbf.dispatch([groups_x, 1, 1]);

                Ok(())
            },
            [(buffer_id, HostAccessType::Write)],
            [(
                buffer_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_READ
                    | AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
            )],
            [],
        )
    }
    .unwrap();

    // Wait for the compute work to finish on the GPU before proceeding.
    resources.flight(flight_id).wait_idle().unwrap();

    // Read our data back from the GPU and verify the result.
    let mut data_buffer_content: Vec<u32> = Vec::new();
    unsafe {
        vulkano_taskgraph::execute(
            &queue,
            &resources,
            flight_id,
            |_cbf, tcx| {
                data_buffer_content = tcx.read_buffer::<[u32]>(buffer_id, ..).to_vec();

                Ok(())
            },
            [(buffer_id, HostAccessType::Read)],
            [],
            [],
        )
    }
    .unwrap();

    assert_eq!(data_buffer_content.len(), BUFFER_LEN.try_into().unwrap());

    for (i, value) in (0..).zip(data_buffer_content) {
        assert_eq!(value, i * 12);
    }

    println!("Success");
}
