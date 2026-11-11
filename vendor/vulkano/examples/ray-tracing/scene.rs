use crate::{App, RenderContext};
use glam::{Mat4, Vec3};
use std::{slice, sync::Arc};
use vulkano::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometry,
        AccelerationStructureGeometryData, AccelerationStructureGeometryInstancesData,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    device::{DeviceOwned, Queue},
    format::Format,
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        graphics::vertex_input::Vertex,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
        PipelineShaderStageCreateInfo,
    },
    swapchain::Swapchain,
    DeviceSize,
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer,
    descriptor_set::{AccelerationStructureId, StorageBufferId},
    resource::{AccessTypes, Flight, HostAccessType, Resources},
    Id, Task, TaskContext, TaskResult,
};

pub struct SceneTask {
    swapchain_id: Id<Swapchain>,
    acceleration_structure_id: AccelerationStructureId,
    camera_storage_buffer_id: StorageBufferId,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    // The bottom-level acceleration structure is required to be kept alive as we reference it in
    // the top-level acceleration structure.
    _blas: Arc<AccelerationStructure>,
}

impl SceneTask {
    pub fn new(app: &App, virtual_swapchain_id: Id<Swapchain>) -> (Self, Id<Buffer>, Id<Buffer>) {
        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let raygen = unsafe { raygen::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let closest_hit = unsafe { closest_hit::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let miss = unsafe { miss::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(&raygen),
                PipelineShaderStageCreateInfo::new(&miss),
                PipelineShaderStageCreateInfo::new(&closest_hit),
            ];

            // Define the shader groups that will eventually turn into the shader binding table.
            // The numbers are the indices of the stages in the `stages` array.
            let groups = [
                RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
                RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
                RayTracingShaderGroupCreateInfo::TrianglesHit {
                    closest_hit_shader: Some(2),
                    any_hit_shader: None,
                },
            ];

            let layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

            RayTracingPipeline::new(
                &app.device,
                None,
                &RayTracingPipelineCreateInfo {
                    stages: &stages,
                    groups: &groups,
                    max_pipeline_ray_recursion_depth: 1,
                    ..RayTracingPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        let vertices = [
            MyVertex {
                position: [-0.5, -0.25, 0.0],
            },
            MyVertex {
                position: [0.0, 0.5, 0.0],
            },
            MyVertex {
                position: [0.25, -0.1, 0.0],
            },
        ];
        let vertex_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(vertices.as_slice()).unwrap(),
            )
            .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &app.queue,
                &app.resources,
                app.flight_id,
                |_cbf, tcx| {
                    tcx.write_buffer::<[MyVertex]>(vertex_buffer_id, ..)
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        // FIXME(taskgraph): sane initialization
        app.resources.flight(app.flight_id).wait(None).unwrap();

        // Build the bottom-level acceleration structure and then the top-level acceleration
        // structure. Acceleration structures are used to accelerate ray tracing. The bottom-level
        // acceleration structure contains the geometry data. The top-level acceleration structure
        // contains the instances of the bottom-level acceleration structures. In our shader, we
        // will trace rays against the top-level acceleration structure.
        let (blas, blas_buffer_id) = unsafe {
            build_acceleration_structure_triangles(
                &app.queue,
                &app.resources,
                app.flight_id,
                vertex_buffer_id,
            )
        };
        let (tlas, tlas_buffer_id) = unsafe {
            build_acceleration_structure_instances(
                &app.queue,
                &app.resources,
                app.flight_id,
                &[AccelerationStructureInstance {
                    acceleration_structure_reference: blas.device_address().into(),
                    ..Default::default()
                }],
            )
        };

        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 4.0 / 3.0, 0.01, 100.0);
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );

        let camera_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<raygen::Camera>(),
            )
            .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &app.queue,
                &app.resources,
                app.flight_id,
                |_cbf, tcx| {
                    *tcx.write_buffer(camera_buffer_id, ..) = raygen::Camera {
                        view_proj: (proj * view).to_cols_array_2d(),
                        view_inverse: view.inverse().to_cols_array_2d(),
                        proj_inverse: proj.inverse().to_cols_array_2d(),
                    };

                    Ok(())
                },
                [(camera_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        let acceleration_structure_id = bcx.global_set().add_acceleration_structure(tlas);

        let camera_storage_buffer_id = bcx
            .global_set()
            .create_storage_buffer(camera_buffer_id, 0, None)
            .unwrap();

        let shader_binding_table =
            ShaderBindingTable::new(app.resources.memory_allocator(), &pipeline).unwrap();

        let task = SceneTask {
            swapchain_id: virtual_swapchain_id,
            acceleration_structure_id,
            camera_storage_buffer_id,
            shader_binding_table,
            pipeline,
            _blas: blas,
        };

        (task, tlas_buffer_id, blas_buffer_id)
    }
}

impl Task for SceneTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.swapchain_id);
        let image_index = swapchain_state.current_image_index().unwrap();
        let extent = swapchain_state.images()[0].extent();

        cbf.push_constants(
            self.pipeline.layout(),
            0,
            &raygen::PushConstants {
                image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                acceleration_structure_id: self.acceleration_structure_id,
                camera_buffer_id: self.camera_storage_buffer_id,
            },
        );
        cbf.bind_pipeline(&self.pipeline);

        unsafe { cbf.trace_rays(self.shader_binding_table.addresses(), extent) };

        Ok(())
    }
}

mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "rgen.glsl",
        vulkan_version: "1.2",
    }
}

mod closest_hit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "rchit.glsl",
        vulkan_version: "1.2",
    }
}

mod miss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "rmiss.glsl",
        vulkan_version: "1.2",
    }
}

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

/// A helper function to build an acceleration structure and wait for its completion.
///
/// # Safety
///
/// - If you are referencing a bottom-level acceleration structure in a top-level acceleration
///   structure, you must ensure that the bottom-level acceleration structure is kept alive.
unsafe fn build_acceleration_structure_common(
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
    ty: AccelerationStructureType,
    geometries: &[AccelerationStructureGeometry<'_>],
    geometry_buffer_id: Id<Buffer>,
    primitive_count: u32,
) -> (Arc<AccelerationStructure>, Id<Buffer>) {
    let mut build_geometry_info = AccelerationStructureBuildGeometryInfo {
        ty,
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        geometries,
        ..Default::default()
    };

    let build_sizes_info = resources.device().acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        &build_geometry_info,
        &[primitive_count],
    );

    let buffer_id = resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
            DeviceLayout::new_unsized::<[u8]>(build_sizes_info.acceleration_structure_size)
                .unwrap(),
        )
        .unwrap();

    let acceleration_structure = unsafe {
        AccelerationStructure::new(
            resources.device(),
            &AccelerationStructureCreateInfo {
                size: build_sizes_info.acceleration_structure_size,
                ty,
                ..AccelerationStructureCreateInfo::new(resources.buffer(buffer_id).buffer())
            },
        )
    }
    .unwrap();

    let device_properties = resources.device().physical_device().properties();
    let min_scratch_alignment = device_properties
        .min_acceleration_structure_scratch_offset_alignment
        .unwrap();

    // We create a new scratch buffer for each acceleration structure build for simplicity. You may
    // want to reuse scratch buffers if you need to build many acceleration structures.
    let scratch_buffer_id = resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
            DeviceLayout::from_size_alignment(
                build_sizes_info.build_scratch_size,
                DeviceSize::from(min_scratch_alignment),
            )
            .unwrap(),
        )
        .unwrap();
    let scratch_buffer_state = resources.buffer(scratch_buffer_id);
    let scratch_buffer = scratch_buffer_state.buffer();

    build_geometry_info.dst_acceleration_structure = Some(&acceleration_structure);
    build_geometry_info.scratch_data = scratch_buffer.device_address().get();

    let build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    // For simplicity, we execute a one-off task graph that builds the acceleration structure, and
    // wait for its execution to complete.
    unsafe {
        vulkano_taskgraph::execute(
            queue,
            resources,
            flight_id,
            |cbf, _tcx| {
                cbf.as_raw().build_acceleration_structure(
                    &build_geometry_info,
                    slice::from_ref(&build_range_info),
                );

                Ok(())
            },
            [],
            [
                (
                    buffer_id,
                    AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE,
                ),
                (
                    scratch_buffer_id,
                    AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_READ
                        | AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE,
                ),
                (
                    geometry_buffer_id,
                    AccessTypes::ACCELERATION_STRUCTURE_BUILD_SHADER_READ,
                ),
            ],
            [],
        )
    }
    .unwrap();

    resources.flight(flight_id).wait(None).unwrap();

    (acceleration_structure, buffer_id)
}

unsafe fn build_acceleration_structure_triangles(
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
    vertex_buffer_id: Id<Buffer>,
) -> (Arc<AccelerationStructure>, Id<Buffer>) {
    let vertex_buffer_state = resources.buffer(vertex_buffer_id);
    let vertex_buffer = vertex_buffer_state.buffer();
    let vertex_count = vertex_buffer.size() / size_of::<MyVertex>() as DeviceSize;
    let triangles_data = AccelerationStructureGeometryTrianglesData {
        vertex_format: Format::R32G32B32_SFLOAT,
        vertex_data: vertex_buffer.device_address().get(),
        vertex_stride: size_of::<MyVertex>() as u32,
        max_vertex: u32::try_from(vertex_count).unwrap() - 1,
        ..Default::default()
    };
    let geometry = AccelerationStructureGeometry::new(
        AccelerationStructureGeometryData::Triangles(triangles_data),
    );

    let primitive_count = vertex_count as u32 / 3;

    build_acceleration_structure_common(
        queue,
        resources,
        flight_id,
        AccelerationStructureType::BottomLevel,
        slice::from_ref(&geometry),
        vertex_buffer_id,
        primitive_count,
    )
}

unsafe fn build_acceleration_structure_instances(
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
    instances: &[AccelerationStructureInstance],
) -> (Arc<AccelerationStructure>, Id<Buffer>) {
    let primitive_count = instances.len() as u32;

    let instance_buffer_id = resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::for_value(instances).unwrap(),
        )
        .unwrap();
    let instance_buffer_state = resources.buffer(instance_buffer_id);
    let instance_buffer = instance_buffer_state.buffer();

    unsafe {
        vulkano_taskgraph::execute(
            queue,
            resources,
            flight_id,
            |_cbf, tcx| {
                tcx.write_buffer::<[AccelerationStructureInstance]>(instance_buffer_id, ..)
                    .copy_from_slice(instances);

                Ok(())
            },
            [(instance_buffer_id, HostAccessType::Write)],
            [],
            [],
        )
    }
    .unwrap();

    // FIXME(taskgraph): sane initialization
    resources.flight(flight_id).wait(None).unwrap();

    let instances_data = AccelerationStructureGeometryInstancesData {
        data: instance_buffer.device_address().get(),
        ..Default::default()
    };
    let geometry = AccelerationStructureGeometry::new(
        AccelerationStructureGeometryData::Instances(instances_data),
    );

    build_acceleration_structure_common(
        queue,
        resources,
        flight_id,
        AccelerationStructureType::TopLevel,
        slice::from_ref(&geometry),
        instance_buffer_id,
        primitive_count,
    )
}
