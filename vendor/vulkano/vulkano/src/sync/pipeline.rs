use crate::{
    buffer::Buffer,
    descriptor_set::layout::DescriptorType,
    device::{Device, QueueFlags},
    image::{Image, ImageLayout, ImageSubresourceRange},
    macros::{vulkan_bitflags, vulkan_bitflags_enum},
    shader::ShaderStages,
    DeviceSize,
};
use ash::vk;
use foldhash::HashMap;
use smallvec::SmallVec;
use std::sync::{Arc, LazyLock};

vulkan_bitflags_enum! {
    #[non_exhaustive]
    /// A set of [`PipelineStage`] values.
    PipelineStages impl {
        /// Returns whether `self` contains stages that are only available in
        /// `VkPipelineStageFlagBits2`.
        pub(crate) fn contains_flags2(self) -> bool {
            !(self
                - (PipelineStages::TOP_OF_PIPE
                    | PipelineStages::DRAW_INDIRECT
                    | PipelineStages::VERTEX_INPUT
                    | PipelineStages::VERTEX_SHADER
                    | PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER
                    | PipelineStages::GEOMETRY_SHADER
                    | PipelineStages::FRAGMENT_SHADER
                    | PipelineStages::EARLY_FRAGMENT_TESTS
                    | PipelineStages::LATE_FRAGMENT_TESTS
                    | PipelineStages::COLOR_ATTACHMENT_OUTPUT
                    | PipelineStages::COMPUTE_SHADER
                    | PipelineStages::ALL_TRANSFER
                    | PipelineStages::BOTTOM_OF_PIPE
                    | PipelineStages::HOST
                    | PipelineStages::ALL_GRAPHICS
                    | PipelineStages::ALL_COMMANDS
                    | PipelineStages::TRANSFORM_FEEDBACK
                    | PipelineStages::CONDITIONAL_RENDERING
                    | PipelineStages::ACCELERATION_STRUCTURE_BUILD
                    | PipelineStages::RAY_TRACING_SHADER
                    | PipelineStages::FRAGMENT_DENSITY_PROCESS
                    | PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT
                    | PipelineStages::COMMAND_PREPROCESS
                    | PipelineStages::TASK_SHADER
                    | PipelineStages::MESH_SHADER))
                .is_empty()
        }

        /// Replaces and unsets flags that are equivalent to multiple other flags.
        ///
        /// This may set flags that are not supported by the device, so this is for internal use only
        /// and should not be passed on to Vulkan.
        pub(crate) fn expand(mut self, queue_flags: QueueFlags) -> Self {
            if self.intersects(PipelineStages::ALL_COMMANDS) {
                self -= PipelineStages::ALL_COMMANDS;
                self |= queue_flags.into();
            }

            if self.intersects(PipelineStages::ALL_GRAPHICS) {
                self -= PipelineStages::ALL_GRAPHICS;
                self |= QueueFlags::GRAPHICS.into();
            }

            if self.intersects(PipelineStages::VERTEX_INPUT) {
                self -= PipelineStages::VERTEX_INPUT;
                self |= PipelineStages::INDEX_INPUT | PipelineStages::VERTEX_ATTRIBUTE_INPUT;
            }

            if self.intersects(PipelineStages::PRE_RASTERIZATION_SHADERS) {
                self -= PipelineStages::PRE_RASTERIZATION_SHADERS;
                self |= PipelineStages::VERTEX_SHADER
                    | PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER
                    | PipelineStages::GEOMETRY_SHADER
                    | PipelineStages::TASK_SHADER
                    | PipelineStages::MESH_SHADER;
            }

            if self.intersects(PipelineStages::ALL_TRANSFER) {
                self -= PipelineStages::ALL_TRANSFER;
                self |= PipelineStages::COPY
                    | PipelineStages::RESOLVE
                    | PipelineStages::BLIT
                    | PipelineStages::CLEAR
                    | PipelineStages::ACCELERATION_STRUCTURE_COPY;
            }

            self
        }

        /// Replaces unsupported flags with more general supported ones.
        pub(crate) fn into_supported(mut self, device: &Device) -> Self {
            if !device.enabled_features().ray_tracing_maintenance1 {
                if self.intersects(PipelineStages::ACCELERATION_STRUCTURE_COPY) {
                    self -= PipelineStages::ACCELERATION_STRUCTURE_COPY;
                    self |= PipelineStages::ACCELERATION_STRUCTURE_BUILD;
                }
            }

            if !device.enabled_features().synchronization2 {
                if self.intersects(
                    PipelineStages::INDEX_INPUT
                    | PipelineStages::VERTEX_ATTRIBUTE_INPUT
                ) {
                    self -= PipelineStages::INDEX_INPUT | PipelineStages::VERTEX_ATTRIBUTE_INPUT;
                    self |= PipelineStages::VERTEX_INPUT;
                }

                if self.intersects(
                    PipelineStages::COPY
                    | PipelineStages::RESOLVE
                    | PipelineStages::BLIT
                    | PipelineStages::CLEAR
                ) {
                    self -= PipelineStages::COPY
                        | PipelineStages::RESOLVE
                        | PipelineStages::BLIT
                        | PipelineStages::CLEAR;
                    self |= PipelineStages::ALL_TRANSFER;
                }

                if self.contains_flags2() {
                    return if (self - PipelineStages::from(QueueFlags::GRAPHICS)).is_empty() {
                        PipelineStages::ALL_GRAPHICS
                    } else {
                        PipelineStages::ALL_COMMANDS
                    };
                }
            }

            match self.validate_device(device) {
                Ok(_) => self,
                Err(_) if (self - PipelineStages::from(QueueFlags::GRAPHICS)).is_empty() => PipelineStages::ALL_GRAPHICS,
                Err(_) => PipelineStages::ALL_COMMANDS,
            }
        }
    },

    /// A single stage in the device's processing pipeline.
    PipelineStage,

    = PipelineStageFlags2(u64);

    /// A pseudo-stage representing the start of the pipeline.
    TOP_OF_PIPE, TopOfPipe = TOP_OF_PIPE,

    /// Indirect buffers are read.
    DRAW_INDIRECT, DrawIndirect = DRAW_INDIRECT,

    /// Vertex and index buffers are read.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `index_input`
    /// - `vertex_attribute_input`
    VERTEX_INPUT, VertexInput = VERTEX_INPUT,

    /// Vertex shaders are executed.
    VERTEX_SHADER, VertexShader = VERTEX_SHADER,

    /// Tessellation control shaders are executed.
    TESSELLATION_CONTROL_SHADER, TessellationControlShader = TESSELLATION_CONTROL_SHADER,

    /// Tessellation evaluation shaders are executed.
    TESSELLATION_EVALUATION_SHADER, TessellationEvaluationShader = TESSELLATION_EVALUATION_SHADER,

    /// Geometry shaders are executed.
    GEOMETRY_SHADER, GeometryShader = GEOMETRY_SHADER,

    /// Fragment shaders are executed.
    FRAGMENT_SHADER, FragmentShader = FRAGMENT_SHADER,

    /// Early fragment tests (depth and stencil tests before fragment shading) are performed.
    /// Subpass load operations for framebuffer attachments with a depth/stencil format are
    /// performed.
    EARLY_FRAGMENT_TESTS, EarlyFragmentTests = EARLY_FRAGMENT_TESTS,

    /// Late fragment tests (depth and stencil tests after fragment shading) are performed.
    /// Subpass store operations for framebuffer attachments with a depth/stencil format are
    /// performed.
    LATE_FRAGMENT_TESTS, LateFragmentTests = LATE_FRAGMENT_TESTS,

    /// The final color values are output from the pipeline after blending.
    /// Subpass load and store operations, multisample resolve operations for framebuffer
    /// attachments with a color or depth/stencil format, and `clear_attachments` are performed.
    COLOR_ATTACHMENT_OUTPUT, ColorAttachmentOutput = COLOR_ATTACHMENT_OUTPUT,

    /// Compute shaders are executed.
    COMPUTE_SHADER, ComputeShader = COMPUTE_SHADER,

    /// The set of all current and future transfer pipeline stages.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `copy`
    /// - `blit`
    /// - `resolve`
    /// - `clear`
    /// - `acceleration_structure_copy`
    ALL_TRANSFER, AllTransfer = ALL_TRANSFER,

    /// A pseudo-stage representing the end of the pipeline.
    BOTTOM_OF_PIPE, BottomOfPipe = BOTTOM_OF_PIPE,

    /// A pseudo-stage representing reads and writes to device memory on the host.
    HOST, Host = HOST,

    /// The set of all current and future graphics pipeline stages.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `draw_indirect`
    /// - `task_shader`
    /// - `mesh_shader`
    /// - `vertex_input`
    /// - `vertex_shader`
    /// - `tessellation_control_shader`
    /// - `tessellation_evaluation_shader`
    /// - `geometry_shader`
    /// - `fragment_shader`
    /// - `early_fragment_tests`
    /// - `late_fragment_tests`
    /// - `color_attachment_output`
    /// - `conditional_rendering`
    /// - `transform_feedback`
    /// - `fragment_shading_rate_attachment`
    /// - `fragment_density_process`
    /// - `invocation_mask`
    ALL_GRAPHICS, AllGraphics = ALL_GRAPHICS,

    /// The set of all current and future pipeline stages of all types.
    ///
    /// It is currently equivalent to setting all flags in `PipelineStages`, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    ALL_COMMANDS, AllCommands = ALL_COMMANDS,

    /// The `copy_buffer`, `copy_image`, `copy_buffer_to_image`, `copy_image_to_buffer` and
    /// `copy_query_pool_results` commands are executed.
    COPY, Copy = COPY
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// The `resolve_image` command is executed.
    RESOLVE, Resolve = RESOLVE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// The `blit_image` command is executed.
    BLIT, Blit = BLIT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// The `clear_color_image`, `clear_depth_stencil_image`, `fill_buffer` and `update_buffer`
    /// commands are executed.
    CLEAR, Clear = CLEAR
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Index buffers are read.
    INDEX_INPUT, IndexInput = INDEX_INPUT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Vertex buffers are read.
    VERTEX_ATTRIBUTE_INPUT, VertexAttributeInput = VERTEX_ATTRIBUTE_INPUT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// The various pre-rasterization shader types are executed.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `vertex_shader`
    /// - `tessellation_control_shader`
    /// - `tessellation_evaluation_shader`
    /// - `geometry_shader`
    /// - `task_shader`
    /// - `mesh_shader`
    PRE_RASTERIZATION_SHADERS, PreRasterizationShaders = PRE_RASTERIZATION_SHADERS
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Video decode operations are performed.
    VIDEO_DECODE, VideoDecode = VIDEO_DECODE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),

    /// Video encode operations are performed.
    VIDEO_ENCODE, VideoEncode = VIDEO_ENCODE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),

    /// Vertex attribute output values are written to the transform feedback buffers.
    TRANSFORM_FEEDBACK, TransformFeedback = TRANSFORM_FEEDBACK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),

    /// The predicate of conditional rendering is read.
    CONDITIONAL_RENDERING, ConditionalRendering = CONDITIONAL_RENDERING_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_conditional_rendering)]),
    ]),

    /// Acceleration_structure commands are executed.
    ACCELERATION_STRUCTURE_BUILD, AccelerationStructureBuild = ACCELERATION_STRUCTURE_BUILD_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /// The various ray tracing shader types are executed.
    RAY_TRACING_SHADER, RayTracingShader = RAY_TRACING_SHADER_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /// The fragment density map is read to generate the fragment areas.
    FRAGMENT_DENSITY_PROCESS, FragmentDensityProcess = FRAGMENT_DENSITY_PROCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_fragment_density_map)]),
    ]),

    /// The fragment shading rate attachment or shading rate image is read to determine the
    /// fragment shading rate for portions of a rasterized primitive.
    FRAGMENT_SHADING_RATE_ATTACHMENT, FragmentShadingRateAttachment = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),

    /// Device-side preprocessing for generated commands via the `preprocess_generated_commands`
    /// command is handled.
    COMMAND_PREPROCESS, CommandPreprocess = COMMAND_PREPROCESS_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_device_generated_commands)]),
    ]),

    /// Task shaders are executed.
    TASK_SHADER, TaskShader = TASK_SHADER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
        RequiresAllOf([DeviceExtension(nv_mesh_shader)]),
    ]),

    /// Mesh shaders are executed.
    MESH_SHADER, MeshShader = MESH_SHADER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
        RequiresAllOf([DeviceExtension(nv_mesh_shader)]),
    ]),

    /// Subpass shaders are executed.
    SUBPASS_SHADER, SubpassShader = SUBPASS_SHADER_HUAWEI
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(huawei_subpass_shading),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(huawei_subpass_shading),
        ]),
    ]),

    /// The invocation mask image is read to optimize ray dispatch.
    INVOCATION_MASK, InvocationMask = INVOCATION_MASK_HUAWEI
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(huawei_invocation_mask),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(huawei_invocation_mask),
        ]),
    ]),

    /// The `copy_acceleration_structure` command is executed.
    ACCELERATION_STRUCTURE_COPY, AccelerationStructureCopy = ACCELERATION_STRUCTURE_COPY_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_maintenance1)]),
    ]),

    /// Micromap commands are executed.
    MICROMAP_BUILD, MicromapBuild = MICROMAP_BUILD_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]),

    /// Optical flow operations are performed.
    OPTICAL_FLOW, OpticalFlow = OPTICAL_FLOW_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_optical_flow)]),
    ]),
}

impl From<PipelineStage> for vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStage) -> Self {
        Self::from_raw(val as u32)
    }
}

impl From<PipelineStages> for vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStages) -> Self {
        Self::from_raw(vk::PipelineStageFlags2::from(val).as_raw() as u32)
    }
}

impl From<QueueFlags> for PipelineStages {
    /// Corresponds to the table "[Supported pipeline stage flags]" in the Vulkan specification.
    ///
    /// [Supported pipeline stage flags]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-stages-supported
    #[inline]
    fn from(val: QueueFlags) -> Self {
        let mut result = PipelineStages::TOP_OF_PIPE
            | PipelineStages::BOTTOM_OF_PIPE
            | PipelineStages::HOST
            | PipelineStages::ALL_COMMANDS;

        if val.intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER) {
            result |= PipelineStages::ALL_TRANSFER
                | PipelineStages::COPY
                | PipelineStages::RESOLVE
                | PipelineStages::BLIT
                | PipelineStages::CLEAR
                | PipelineStages::ACCELERATION_STRUCTURE_COPY;
        }

        if val.intersects(QueueFlags::GRAPHICS) {
            result |= PipelineStages::DRAW_INDIRECT
                | PipelineStages::VERTEX_INPUT
                | PipelineStages::VERTEX_SHADER
                | PipelineStages::TESSELLATION_CONTROL_SHADER
                | PipelineStages::TESSELLATION_EVALUATION_SHADER
                | PipelineStages::GEOMETRY_SHADER
                | PipelineStages::FRAGMENT_SHADER
                | PipelineStages::EARLY_FRAGMENT_TESTS
                | PipelineStages::LATE_FRAGMENT_TESTS
                | PipelineStages::COLOR_ATTACHMENT_OUTPUT
                | PipelineStages::ALL_GRAPHICS
                | PipelineStages::INDEX_INPUT
                | PipelineStages::VERTEX_ATTRIBUTE_INPUT
                | PipelineStages::PRE_RASTERIZATION_SHADERS
                | PipelineStages::CONDITIONAL_RENDERING
                | PipelineStages::TRANSFORM_FEEDBACK
                | PipelineStages::COMMAND_PREPROCESS
                | PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT
                | PipelineStages::TASK_SHADER
                | PipelineStages::MESH_SHADER
                | PipelineStages::FRAGMENT_DENSITY_PROCESS
                | PipelineStages::SUBPASS_SHADER
                | PipelineStages::INVOCATION_MASK;
        }

        if val.intersects(QueueFlags::COMPUTE) {
            result |= PipelineStages::DRAW_INDIRECT
                | PipelineStages::COMPUTE_SHADER
                | PipelineStages::CONDITIONAL_RENDERING
                | PipelineStages::COMMAND_PREPROCESS
                | PipelineStages::ACCELERATION_STRUCTURE_BUILD
                | PipelineStages::RAY_TRACING_SHADER
                | PipelineStages::MICROMAP_BUILD;
        }

        if val.intersects(QueueFlags::VIDEO_DECODE) {
            result |= PipelineStages::VIDEO_DECODE;
        }

        if val.intersects(QueueFlags::VIDEO_ENCODE) {
            result |= PipelineStages::VIDEO_ENCODE;
        }

        if val.intersects(QueueFlags::OPTICAL_FLOW) {
            result |= PipelineStages::OPTICAL_FLOW;
        }

        result
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// A set of memory access types that are included in a memory dependency.
    AccessFlags impl {
        const WRITES: AccessFlags = AccessFlags::SHADER_WRITE
            .union(AccessFlags::COLOR_ATTACHMENT_WRITE)
            .union(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .union(AccessFlags::TRANSFER_WRITE)
            .union(AccessFlags::HOST_WRITE)
            .union(AccessFlags::MEMORY_WRITE)
            .union(AccessFlags::SHADER_STORAGE_WRITE)
            .union(AccessFlags::VIDEO_DECODE_WRITE)
            .union(AccessFlags::VIDEO_ENCODE_WRITE)
            .union(AccessFlags::TRANSFORM_FEEDBACK_WRITE)
            .union(AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE)
            .union(AccessFlags::COMMAND_PREPROCESS_WRITE)
            .union(AccessFlags::ACCELERATION_STRUCTURE_WRITE);

        /// Returns whether `self` contains any read flags.
        #[inline]
        pub const fn contains_reads(self) -> bool {
            !self.difference(Self::WRITES).is_empty()
        }

        /// Returns whether `self` contains any write flags.
        #[inline]
        pub const fn contains_writes(self) -> bool {
            self.intersects(Self::WRITES)
        }

        /// Returns whether `self` contains flags that are only available in `VkAccessFlagBits2`.
        pub(crate) fn contains_flags2(self) -> bool {
            !(self
                - (AccessFlags::INDIRECT_COMMAND_READ
                    | AccessFlags::INDEX_READ
                    | AccessFlags::VERTEX_ATTRIBUTE_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::INPUT_ATTACHMENT_READ
                    | AccessFlags::SHADER_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::COLOR_ATTACHMENT_READ
                    | AccessFlags::COLOR_ATTACHMENT_WRITE
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | AccessFlags::TRANSFER_READ
                    | AccessFlags::TRANSFER_WRITE
                    | AccessFlags::HOST_READ
                    | AccessFlags::HOST_WRITE
                    | AccessFlags::MEMORY_READ
                    | AccessFlags::MEMORY_WRITE
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::VIDEO_DECODE_READ
                    | AccessFlags::VIDEO_DECODE_WRITE
                    | AccessFlags::VIDEO_ENCODE_READ
                    | AccessFlags::VIDEO_ENCODE_WRITE
                    | AccessFlags::TRANSFORM_FEEDBACK_WRITE
                    | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ
                    | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE
                    | AccessFlags::CONDITIONAL_RENDERING_READ
                    | AccessFlags::COMMAND_PREPROCESS_READ
                    | AccessFlags::COMMAND_PREPROCESS_WRITE
                    | AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ
                    | AccessFlags::ACCELERATION_STRUCTURE_READ
                    | AccessFlags::ACCELERATION_STRUCTURE_WRITE
                    | AccessFlags::FRAGMENT_DENSITY_MAP_READ
                    | AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT
                    | AccessFlags::INVOCATION_MASK_READ))
                .is_empty()
        }

        /// Replaces and unsets flags that are equivalent to multiple other flags.
        ///
        /// This may set flags that are not supported by the device, so this is for internal use
        /// only and should not be passed on to Vulkan.
        #[allow(dead_code)] // TODO: use this function
        pub(crate) fn expand(mut self) -> Self {
            if self.intersects(AccessFlags::SHADER_READ) {
                self -= AccessFlags::SHADER_READ;
                self |= AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_BINDING_TABLE_READ;
            }

            if self.intersects(AccessFlags::SHADER_WRITE) {
                self -= AccessFlags::SHADER_WRITE;
                self |= AccessFlags::SHADER_STORAGE_WRITE;
            }

            self
        }

        /// Replaces unsupported flags with more general supported ones.
        pub(crate) fn into_supported(mut self, device: &Device) -> Self {
            if !device.enabled_features().synchronization2 {
                if self.intersects(
                    AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                ) {
                    self -= AccessFlags::SHADER_SAMPLED_READ | AccessFlags::SHADER_STORAGE_READ;
                    self |= AccessFlags::SHADER_READ;
                }

                if self.intersects(
                    AccessFlags::SHADER_STORAGE_WRITE
                ) {
                    self -= AccessFlags::SHADER_STORAGE_WRITE;
                    self |= AccessFlags::SHADER_WRITE;
                }

                if self.contains_flags2() {
                    return match (self.contains_reads(), self.contains_writes()) {
                        (true, true) => AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                        (true, false) => AccessFlags::MEMORY_READ,
                        (false, true) => AccessFlags::MEMORY_WRITE,
                        (false, false) => unreachable!(),
                    };
                }
            }

            if self.validate_device(device).is_err() {
                return match (self.contains_reads(), self.contains_writes()) {
                    (true, true) => AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                    (true, false) => AccessFlags::MEMORY_READ,
                    (false, true) => AccessFlags::MEMORY_WRITE,
                    (false, false) => unreachable!(),
                };
            }

            self
        }
    }
    = AccessFlags2(u64);

    /// Read access to an indirect buffer.
    INDIRECT_COMMAND_READ = INDIRECT_COMMAND_READ,

    /// Read access to an index buffer.
    INDEX_READ = INDEX_READ,

    /// Read access to a vertex buffer.
    VERTEX_ATTRIBUTE_READ = VERTEX_ATTRIBUTE_READ,

    /// Read access to a uniform buffer in a shader.
    UNIFORM_READ = UNIFORM_READ,

    /// Read access to an input attachment in a fragment shader, within a render pass.
    INPUT_ATTACHMENT_READ = INPUT_ATTACHMENT_READ,

    /// Read access to a buffer or image in a shader.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `uniform_read`
    /// - `shader_sampled_read`
    /// - `shader_storage_read`
    /// - `shader_binding_table_read`
    SHADER_READ = SHADER_READ,

    /// Write access to a buffer or image in a shader.
    ///
    /// It is currently equivalent to `shader_storage_write`. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    SHADER_WRITE = SHADER_WRITE,

    /// Read access to a color attachment during blending, logic operations or
    /// subpass load operations.
    COLOR_ATTACHMENT_READ = COLOR_ATTACHMENT_READ,

    /// Write access to a color, resolve or depth/stencil resolve attachment during a render pass
    /// or subpass store operations.
    COLOR_ATTACHMENT_WRITE = COLOR_ATTACHMENT_WRITE,

    /// Read access to a depth/stencil attachment during depth/stencil operations or
    /// subpass load operations.
    DEPTH_STENCIL_ATTACHMENT_READ = DEPTH_STENCIL_ATTACHMENT_READ,

    /// Write access to a depth/stencil attachment during depth/stencil operations or
    /// subpass store operations.
    DEPTH_STENCIL_ATTACHMENT_WRITE = DEPTH_STENCIL_ATTACHMENT_WRITE,

    /// Read access to a buffer or image during a copy, blit or resolve command.
    TRANSFER_READ = TRANSFER_READ,

    /// Write access to a buffer or image during a copy, blit, resolve or clear command.
    TRANSFER_WRITE = TRANSFER_WRITE,

    /// Read access performed by the host.
    HOST_READ = HOST_READ,

    /// Write access performed by the host.
    HOST_WRITE = HOST_WRITE,

    /// Any type of read access.
    ///
    /// This is equivalent to setting all `_read` flags that are allowed in the given context.
    MEMORY_READ = MEMORY_READ,

    /// Any type of write access.
    ///
    /// This is equivalent to setting all `_write` flags that are allowed in the given context.
    MEMORY_WRITE = MEMORY_WRITE,

    /// Read access to a uniform texel buffer or sampled image in a shader.
    SHADER_SAMPLED_READ = SHADER_SAMPLED_READ
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Read access to a storage buffer, storage texel buffer or storage image in a shader.
    SHADER_STORAGE_READ = SHADER_STORAGE_READ
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Write access to a storage buffer, storage texel buffer or storage image in a shader.
    SHADER_STORAGE_WRITE = SHADER_STORAGE_WRITE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Read access to an image or buffer as part of a video decode operation.
    VIDEO_DECODE_READ = VIDEO_DECODE_READ_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_video_decode_queue),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_video_decode_queue),
        ]),
    ]),

    /// Write access to an image or buffer as part of a video decode operation.
    VIDEO_DECODE_WRITE = VIDEO_DECODE_WRITE_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_video_decode_queue),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_video_decode_queue),
        ]),
    ]),

    /// Read access to an image or buffer as part of a video encode operation.
    VIDEO_ENCODE_READ = VIDEO_ENCODE_READ_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_video_encode_queue),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_video_encode_queue),
        ]),
    ]),

    /// Write access to an image or buffer as part of a video encode operation.
    VIDEO_ENCODE_WRITE = VIDEO_ENCODE_WRITE_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_video_encode_queue),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_video_encode_queue),
        ]),
    ]),

    /// Write access to a transform feedback buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_WRITE = TRANSFORM_FEEDBACK_WRITE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),

    /// Read access to a transform feedback counter buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_COUNTER_READ = TRANSFORM_FEEDBACK_COUNTER_READ_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),

    /// Write access to a transform feedback counter buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_COUNTER_WRITE = TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),

    /// Read access to a predicate during conditional rendering.
    CONDITIONAL_RENDERING_READ = CONDITIONAL_RENDERING_READ_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_conditional_rendering)]),
    ]),

    /// Read access to preprocess buffers input to `preprocess_generated_commands`.
    COMMAND_PREPROCESS_READ = COMMAND_PREPROCESS_READ_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_device_generated_commands)]),
    ]),

    /// Read access to sequences buffers output by `preprocess_generated_commands`.
    COMMAND_PREPROCESS_WRITE = COMMAND_PREPROCESS_WRITE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_device_generated_commands)]),
    ]),

    /// Read access to a fragment shading rate attachment during rasterization.
    FRAGMENT_SHADING_RATE_ATTACHMENT_READ = FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),

    /// Read access to an acceleration structure or acceleration structure scratch buffer during
    /// trace, build or copy commands.
    ACCELERATION_STRUCTURE_READ = ACCELERATION_STRUCTURE_READ_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /// Write access to an acceleration structure or acceleration structure scratch buffer during
    /// trace, build or copy commands.
    ACCELERATION_STRUCTURE_WRITE = ACCELERATION_STRUCTURE_WRITE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /// Read access to a fragment density map attachment during dynamic fragment density map
    /// operations.
    FRAGMENT_DENSITY_MAP_READ = FRAGMENT_DENSITY_MAP_READ_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_fragment_density_map)]),
    ]),

    /// Read access to color attachments when performing advanced blend operations.
    COLOR_ATTACHMENT_READ_NONCOHERENT = COLOR_ATTACHMENT_READ_NONCOHERENT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),

    /// Read access to an invocation mask image.
    INVOCATION_MASK_READ = INVOCATION_MASK_READ_HUAWEI
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(huawei_invocation_mask),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(huawei_invocation_mask),
        ]),
    ]),

    /// Read access to a shader binding table.
    SHADER_BINDING_TABLE_READ = SHADER_BINDING_TABLE_READ_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_ray_tracing_pipeline),
            DeviceExtension(khr_ray_tracing_maintenance1),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_ray_tracing_pipeline),
            DeviceExtension(khr_ray_tracing_maintenance1),
        ]),
    ]),

    /// Read access to a micromap object.
    MICROMAP_READ = MICROMAP_READ_EXT
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(ext_opacity_micromap),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(ext_opacity_micromap),
        ]),
    ]),

    /// Write access to a micromap object.
    MICROMAP_WRITE = MICROMAP_WRITE_EXT
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(ext_opacity_micromap),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(ext_opacity_micromap),
        ]),
    ]),

    /// Read access to a buffer or image during optical flow operations.
    OPTICAL_FLOW_READ = OPTICAL_FLOW_READ_NV
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(nv_optical_flow),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(nv_optical_flow),
        ]),
    ]),

    /// Write access to a buffer or image during optical flow operations.
    OPTICAL_FLOW_WRITE = OPTICAL_FLOW_WRITE_NV
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(nv_optical_flow),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(nv_optical_flow),
        ]),
    ]),
}

impl From<AccessFlags> for vk::AccessFlags {
    #[inline]
    fn from(val: AccessFlags) -> Self {
        Self::from_raw(vk::AccessFlags2::from(val).as_raw() as u32)
    }
}

impl From<PipelineStages> for AccessFlags {
    /// Corresponds to the table "[Supported access types]" in the Vulkan specification.
    ///
    /// [Supported access types]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported
    #[inline]
    fn from(mut val: PipelineStages) -> Self {
        if val.is_empty() {
            return AccessFlags::empty();
        }

        val = val.expand(QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER);
        let mut result = AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE;

        if val.intersects(PipelineStages::DRAW_INDIRECT) {
            result |=
                AccessFlags::INDIRECT_COMMAND_READ | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ;
        }

        if val.intersects(
            PipelineStages::VERTEX_SHADER
                | PipelineStages::TESSELLATION_CONTROL_SHADER
                | PipelineStages::TESSELLATION_EVALUATION_SHADER
                | PipelineStages::GEOMETRY_SHADER
                | PipelineStages::FRAGMENT_SHADER
                | PipelineStages::COMPUTE_SHADER
                | PipelineStages::RAY_TRACING_SHADER
                | PipelineStages::TASK_SHADER
                | PipelineStages::MESH_SHADER,
        ) {
            result |= AccessFlags::SHADER_READ
                | AccessFlags::UNIFORM_READ
                | AccessFlags::SHADER_SAMPLED_READ
                | AccessFlags::SHADER_STORAGE_READ
                | AccessFlags::SHADER_WRITE
                | AccessFlags::SHADER_STORAGE_WRITE
                | AccessFlags::ACCELERATION_STRUCTURE_READ;
        }

        if val.intersects(PipelineStages::FRAGMENT_SHADER | PipelineStages::SUBPASS_SHADER) {
            result |= AccessFlags::INPUT_ATTACHMENT_READ;
        }

        if val
            .intersects(PipelineStages::EARLY_FRAGMENT_TESTS | PipelineStages::LATE_FRAGMENT_TESTS)
        {
            result |= AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }

        if val.intersects(PipelineStages::COLOR_ATTACHMENT_OUTPUT) {
            result |= AccessFlags::COLOR_ATTACHMENT_READ
                | AccessFlags::COLOR_ATTACHMENT_WRITE
                | AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT;
        }

        if val.intersects(PipelineStages::HOST) {
            result |= AccessFlags::HOST_READ | AccessFlags::HOST_WRITE;
        }

        if val.intersects(
            PipelineStages::COPY
                | PipelineStages::RESOLVE
                | PipelineStages::BLIT
                | PipelineStages::ACCELERATION_STRUCTURE_COPY,
        ) {
            result |= AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE;
        }

        if val.intersects(PipelineStages::CLEAR) {
            result |= AccessFlags::TRANSFER_WRITE;
        }

        if val.intersects(PipelineStages::INDEX_INPUT) {
            result |= AccessFlags::INDEX_READ;
        }

        if val.intersects(PipelineStages::VERTEX_ATTRIBUTE_INPUT) {
            result |= AccessFlags::VERTEX_ATTRIBUTE_READ;
        }

        if val.intersects(PipelineStages::VIDEO_DECODE) {
            result |= AccessFlags::VIDEO_DECODE_READ | AccessFlags::VIDEO_DECODE_WRITE;
        }

        if val.intersects(PipelineStages::VIDEO_ENCODE) {
            result |= AccessFlags::VIDEO_ENCODE_READ | AccessFlags::VIDEO_ENCODE_WRITE;
        }

        if val.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
            result |= AccessFlags::TRANSFORM_FEEDBACK_WRITE
                | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE
                | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ;
        }

        if val.intersects(PipelineStages::CONDITIONAL_RENDERING) {
            result |= AccessFlags::CONDITIONAL_RENDERING_READ;
        }

        if val.intersects(PipelineStages::ACCELERATION_STRUCTURE_BUILD) {
            result |= AccessFlags::INDIRECT_COMMAND_READ
                | AccessFlags::SHADER_READ
                | AccessFlags::SHADER_SAMPLED_READ
                | AccessFlags::SHADER_STORAGE_READ
                | AccessFlags::SHADER_STORAGE_WRITE
                | AccessFlags::TRANSFER_READ
                | AccessFlags::TRANSFER_WRITE
                | AccessFlags::ACCELERATION_STRUCTURE_READ
                | AccessFlags::ACCELERATION_STRUCTURE_WRITE
                | AccessFlags::MICROMAP_READ;
        }

        if val.intersects(PipelineStages::RAY_TRACING_SHADER) {
            result |= AccessFlags::SHADER_BINDING_TABLE_READ;
        }

        if val.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
            result |= AccessFlags::FRAGMENT_DENSITY_MAP_READ;
        }

        if val.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
            result |= AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ;
        }

        if val.intersects(PipelineStages::COMMAND_PREPROCESS) {
            result |= AccessFlags::COMMAND_PREPROCESS_READ | AccessFlags::COMMAND_PREPROCESS_WRITE;
        }

        if val.intersects(PipelineStages::INVOCATION_MASK) {
            result |= AccessFlags::INVOCATION_MASK_READ;
        }

        if val.intersects(PipelineStages::MICROMAP_BUILD) {
            result |= AccessFlags::MICROMAP_READ | AccessFlags::MICROMAP_WRITE;
        }

        if val.intersects(PipelineStages::OPTICAL_FLOW) {
            result |= AccessFlags::OPTICAL_FLOW_READ | AccessFlags::OPTICAL_FLOW_WRITE;
        }

        result
    }
}

macro_rules! pipeline_stage_access {
    (
        $(
            $val:ident, $stage:ident, $access:ident
        );*
        $(;)?
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[allow(non_camel_case_types, dead_code)]
        #[repr(u8)]
        pub(crate) enum PipelineStageAccess {
            $($val,)*

            // If there are ever more than 128 preceding values, then there will be a compile error:
            // "discriminant value `128` assigned more than once"
            __MAX_VALUE__ = 128,
        }

        impl From<PipelineStageAccess> for PipelineStages {
            #[inline]
            fn from(val: PipelineStageAccess) -> Self {
                match val {
                    $(
                        PipelineStageAccess::$val => PipelineStages::$stage,
                    )*
                    PipelineStageAccess::__MAX_VALUE__ => unreachable!(),
                }
            }
        }

        impl From<PipelineStageAccess> for AccessFlags {
            #[inline]
            fn from(val: PipelineStageAccess) -> Self {
                match val {
                    $(
                        PipelineStageAccess::$val => AccessFlags::$access,
                    )*
                    PipelineStageAccess::__MAX_VALUE__ => unreachable!(),
                }
            }
        }

        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub(crate) struct PipelineStageAccessFlags(u128);

        #[allow(non_upper_case_globals, dead_code)]
        impl PipelineStageAccessFlags {
            $(
                pub(crate) const $val: Self = Self(1u128 << PipelineStageAccess::$val as u8);
            )*

            #[inline]
            pub(crate) const fn empty() -> Self {
                Self(0)
            }

            #[inline]
            pub(crate) const fn count(self) -> u32 {
                self.0.count_ones()
            }

            #[inline]
            pub(crate) const fn is_empty(self) -> bool {
                self.0 == 0
            }

            #[inline]
            pub(crate) const fn intersects(self, other: Self) -> bool {
                self.0 & other.0 != 0
            }

            #[inline]
            pub(crate) const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            #[inline]
            pub(crate) const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            #[inline]
            pub(crate) const fn intersection(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }

            #[inline]
            pub(crate) const fn difference(self, other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            #[inline]
            pub(crate) const fn symmetric_difference(self, other: Self) -> Self {
                Self(self.0 ^ other.0)
            }

            #[inline]
            pub(crate) fn contains_enum(self, val: PipelineStageAccess) -> bool {
                self.intersects(val.into())
            }
        }

        impl std::ops::BitAnd for PipelineStageAccessFlags {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                self.intersection(rhs)
            }
        }

        impl std::ops::BitAndAssign for PipelineStageAccessFlags {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.intersection(rhs);
            }
        }

        impl std::ops::BitOr for PipelineStageAccessFlags {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl std::ops::BitOrAssign for PipelineStageAccessFlags {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.union(rhs);
            }
        }

        impl std::ops::BitXor for PipelineStageAccessFlags {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                self.symmetric_difference(rhs)
            }
        }

        impl std::ops::BitXorAssign for PipelineStageAccessFlags {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.symmetric_difference(rhs);
            }
        }

        impl std::ops::Sub for PipelineStageAccessFlags {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.difference(rhs)
            }
        }

        impl std::ops::SubAssign for PipelineStageAccessFlags {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(rhs);
            }
        }

        impl From<PipelineStageAccess> for PipelineStageAccessFlags {
            #[inline]
            fn from(val: PipelineStageAccess) -> Self {
                debug_assert!(val != PipelineStageAccess::__MAX_VALUE__); // You did something very dumb...
                Self(1u128 << val as u8)
            }
        }

        impl From<PipelineStageAccessFlags> for PipelineStages {
            #[inline]
            fn from(val: PipelineStageAccessFlags) -> Self {
                let mut result = PipelineStages::empty();

                $(
                    if val.intersects(PipelineStageAccessFlags::$val) {
                        result |= PipelineStages::$stage;
                    }
                )*

                result
            }
        }

        impl From<PipelineStageAccessFlags> for AccessFlags {
            #[inline]
            fn from(val: PipelineStageAccessFlags) -> Self {
                let mut result = AccessFlags::empty();

                $(
                    if val.intersects(PipelineStageAccessFlags::$val) {
                        result |= AccessFlags::$access;
                    }
                )*

                result
            }
        }
    };
}

pipeline_stage_access! {
    DrawIndirect_IndirectCommandRead, DRAW_INDIRECT, INDIRECT_COMMAND_READ;
    DrawIndirect_TransformFeedbackCounterRead, DRAW_INDIRECT, TRANSFORM_FEEDBACK_COUNTER_READ;
    VertexShader_UniformRead, VERTEX_SHADER, UNIFORM_READ;
    VertexShader_ShaderSampledRead, VERTEX_SHADER, SHADER_SAMPLED_READ;
    VertexShader_ShaderStorageRead, VERTEX_SHADER, SHADER_STORAGE_READ;
    VertexShader_ShaderStorageWrite, VERTEX_SHADER, SHADER_STORAGE_WRITE;
    VertexShader_AccelerationStructureRead, VERTEX_SHADER, ACCELERATION_STRUCTURE_READ;
    TessellationControlShader_UniformRead, TESSELLATION_CONTROL_SHADER, UNIFORM_READ;
    TessellationControlShader_ShaderSampledRead, TESSELLATION_CONTROL_SHADER, SHADER_SAMPLED_READ;
    TessellationControlShader_ShaderStorageRead, TESSELLATION_CONTROL_SHADER, SHADER_STORAGE_READ;
    TessellationControlShader_ShaderStorageWrite, TESSELLATION_CONTROL_SHADER, SHADER_STORAGE_WRITE;
    TessellationControlShader_AccelerationStructureRead, TESSELLATION_CONTROL_SHADER, ACCELERATION_STRUCTURE_READ;
    TessellationEvaluationShader_UniformRead, TESSELLATION_EVALUATION_SHADER, UNIFORM_READ;
    TessellationEvaluationShader_ShaderSampledRead, TESSELLATION_EVALUATION_SHADER, SHADER_SAMPLED_READ;
    TessellationEvaluationShader_ShaderStorageRead, TESSELLATION_EVALUATION_SHADER, SHADER_STORAGE_READ;
    TessellationEvaluationShader_ShaderStorageWrite, TESSELLATION_EVALUATION_SHADER, SHADER_STORAGE_WRITE;
    TessellationEvaluationShader_AccelerationStructureRead, TESSELLATION_EVALUATION_SHADER, ACCELERATION_STRUCTURE_READ;
    GeometryShader_UniformRead, GEOMETRY_SHADER, UNIFORM_READ;
    GeometryShader_ShaderSampledRead, GEOMETRY_SHADER, SHADER_SAMPLED_READ;
    GeometryShader_ShaderStorageRead, GEOMETRY_SHADER, SHADER_STORAGE_READ;
    GeometryShader_ShaderStorageWrite, GEOMETRY_SHADER, SHADER_STORAGE_WRITE;
    GeometryShader_AccelerationStructureRead, GEOMETRY_SHADER, ACCELERATION_STRUCTURE_READ;
    FragmentShader_UniformRead, FRAGMENT_SHADER, UNIFORM_READ;
    FragmentShader_InputAttachmentRead, FRAGMENT_SHADER, INPUT_ATTACHMENT_READ;
    FragmentShader_ShaderSampledRead, FRAGMENT_SHADER, SHADER_SAMPLED_READ;
    FragmentShader_ShaderStorageRead, FRAGMENT_SHADER, SHADER_STORAGE_READ;
    FragmentShader_ShaderStorageWrite, FRAGMENT_SHADER, SHADER_STORAGE_WRITE;
    FragmentShader_AccelerationStructureRead, FRAGMENT_SHADER, ACCELERATION_STRUCTURE_READ;
    EarlyFragmentTests_DepthStencilAttachmentRead, EARLY_FRAGMENT_TESTS, DEPTH_STENCIL_ATTACHMENT_READ;
    EarlyFragmentTests_DepthStencilAttachmentWrite, EARLY_FRAGMENT_TESTS, DEPTH_STENCIL_ATTACHMENT_WRITE;
    LateFragmentTests_DepthStencilAttachmentRead, LATE_FRAGMENT_TESTS, DEPTH_STENCIL_ATTACHMENT_READ;
    LateFragmentTests_DepthStencilAttachmentWrite, LATE_FRAGMENT_TESTS, DEPTH_STENCIL_ATTACHMENT_WRITE;
    ColorAttachmentOutput_ColorAttachmentRead, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_READ;
    ColorAttachmentOutput_ColorAttachmentWrite, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_WRITE;
    ColorAttachmentOutput_ColorAttachmentReadNoncoherent, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_READ_NONCOHERENT;
    ComputeShader_UniformRead, COMPUTE_SHADER, UNIFORM_READ;
    ComputeShader_ShaderSampledRead, COMPUTE_SHADER, SHADER_SAMPLED_READ;
    ComputeShader_ShaderStorageRead, COMPUTE_SHADER, SHADER_STORAGE_READ;
    ComputeShader_ShaderStorageWrite, COMPUTE_SHADER, SHADER_STORAGE_WRITE;
    ComputeShader_AccelerationStructureRead, COMPUTE_SHADER, ACCELERATION_STRUCTURE_READ;
    Host_HostRead, HOST, HOST_READ;
    Host_HostWrite, HOST, HOST_WRITE;
    Copy_TransferRead, COPY, TRANSFER_READ;
    Copy_TransferWrite, COPY, TRANSFER_WRITE;
    Resolve_TransferRead, RESOLVE, TRANSFER_READ;
    Resolve_TransferWrite, RESOLVE, TRANSFER_WRITE;
    Blit_TransferRead, BLIT, TRANSFER_READ;
    Blit_TransferWrite, BLIT, TRANSFER_WRITE;
    Clear_TransferWrite, CLEAR, TRANSFER_WRITE;
    IndexInput_IndexRead, INDEX_INPUT, INDEX_READ;
    VertexAttributeInput_VertexAttributeRead, VERTEX_ATTRIBUTE_INPUT, VERTEX_ATTRIBUTE_READ;
    VideoDecode_VideoDecodeRead, VIDEO_DECODE, VIDEO_DECODE_READ;
    VideoDecode_VideoDecodeWrite, VIDEO_DECODE, VIDEO_DECODE_WRITE;
    VideoEncode_VideoEncodeRead, VIDEO_ENCODE, VIDEO_ENCODE_READ;
    VideoEncode_VideoEncodeWrite, VIDEO_ENCODE, VIDEO_ENCODE_WRITE;
    TransformFeedback_TransformFeedbackWrite, TRANSFORM_FEEDBACK, TRANSFORM_FEEDBACK_WRITE;
    TransformFeedback_TransformFeedbackCounterRead, TRANSFORM_FEEDBACK, TRANSFORM_FEEDBACK_COUNTER_READ;
    TransformFeedback_TransformFeedbackCounterWrite, TRANSFORM_FEEDBACK, TRANSFORM_FEEDBACK_COUNTER_WRITE;
    ConditionalRendering_ConditionalRenderingRead, CONDITIONAL_RENDERING, CONDITIONAL_RENDERING_READ;
    AccelerationStructureBuild_IndirectCommandRead, ACCELERATION_STRUCTURE_BUILD, INDIRECT_COMMAND_READ;
    AccelerationStructureBuild_UniformRead, ACCELERATION_STRUCTURE_BUILD, UNIFORM_READ;
    AccelerationStructureBuild_TransferRead, ACCELERATION_STRUCTURE_BUILD, TRANSFER_READ;
    AccelerationStructureBuild_TransferWrite, ACCELERATION_STRUCTURE_BUILD, TRANSFER_WRITE;
    AccelerationStructureBuild_ShaderSampledRead, ACCELERATION_STRUCTURE_BUILD, SHADER_SAMPLED_READ;
    AccelerationStructureBuild_ShaderStorageRead, ACCELERATION_STRUCTURE_BUILD, SHADER_STORAGE_READ;
    AccelerationStructureBuild_AccelerationStructureRead, ACCELERATION_STRUCTURE_BUILD, ACCELERATION_STRUCTURE_READ;
    AccelerationStructureBuild_AccelerationStructureWrite, ACCELERATION_STRUCTURE_BUILD, ACCELERATION_STRUCTURE_WRITE;
    AccelerationStructureBuild_MicromapRead, ACCELERATION_STRUCTURE_BUILD, MICROMAP_READ;
    RayTracingShader_UniformRead, RAY_TRACING_SHADER, UNIFORM_READ;
    RayTracingShader_ShaderSampledRead, RAY_TRACING_SHADER, SHADER_SAMPLED_READ;
    RayTracingShader_ShaderStorageRead, RAY_TRACING_SHADER, SHADER_STORAGE_READ;
    RayTracingShader_ShaderStorageWrite, RAY_TRACING_SHADER, SHADER_STORAGE_WRITE;
    RayTracingShader_AccelerationStructureRead, RAY_TRACING_SHADER, ACCELERATION_STRUCTURE_READ;
    RayTracingShader_ShaderBindingTableRead, RAY_TRACING_SHADER, SHADER_BINDING_TABLE_READ;
    FragmentDensityProcess_FragmentDensityMapRead, FRAGMENT_DENSITY_PROCESS, FRAGMENT_DENSITY_MAP_READ;
    FragmentShadingRateAttachment_FragmentShadingRateAttachmentRead, FRAGMENT_SHADING_RATE_ATTACHMENT, FRAGMENT_SHADING_RATE_ATTACHMENT_READ;
    CommandPreprocess_CommandPreprocessRead, COMMAND_PREPROCESS, COMMAND_PREPROCESS_READ;
    CommandPreprocess_CommandPreprocessWrite, COMMAND_PREPROCESS, COMMAND_PREPROCESS_WRITE;
    TaskShader_UniformRead, TASK_SHADER, UNIFORM_READ;
    TaskShader_ShaderSampledRead, TASK_SHADER, SHADER_SAMPLED_READ;
    TaskShader_ShaderStorageRead, TASK_SHADER, SHADER_STORAGE_READ;
    TaskShader_ShaderStorageWrite, TASK_SHADER, SHADER_STORAGE_WRITE;
    TaskShader_AccelerationStructureRead, TASK_SHADER, ACCELERATION_STRUCTURE_READ;
    MeshShader_UniformRead, MESH_SHADER, UNIFORM_READ;
    MeshShader_ShaderSampledRead, MESH_SHADER, SHADER_SAMPLED_READ;
    MeshShader_ShaderStorageRead, MESH_SHADER, SHADER_STORAGE_READ;
    MeshShader_ShaderStorageWrite, MESH_SHADER, SHADER_STORAGE_WRITE;
    MeshShader_AccelerationStructureRead, MESH_SHADER, ACCELERATION_STRUCTURE_READ;
    SubpassShading_InputAttachmentRead, SUBPASS_SHADER, INPUT_ATTACHMENT_READ;
    InvocationMask_InvocationMaskRead, INVOCATION_MASK, INVOCATION_MASK_READ;
    AccelerationStructureCopy_AccelerationStructureRead, ACCELERATION_STRUCTURE_COPY, ACCELERATION_STRUCTURE_READ;
    AccelerationStructureCopy_AccelerationStructureWrite, ACCELERATION_STRUCTURE_COPY, ACCELERATION_STRUCTURE_WRITE;
    AccelerationStructureCopy_TransferRead, ACCELERATION_STRUCTURE_COPY, TRANSFER_READ;
    AccelerationStructureCopy_TransferWrite, ACCELERATION_STRUCTURE_COPY, TRANSFER_WRITE;
    OpticalFlow_OpticalFlowRead, OPTICAL_FLOW, OPTICAL_FLOW_READ;
    OpticalFlow_OpticalFlowWrite, OPTICAL_FLOW, OPTICAL_FLOW_WRITE;
    MicromapBuild_MicromapRead, MICROMAP_BUILD, MICROMAP_READ;
    MicromapBuild_MicromapWrite, MICROMAP_BUILD, MICROMAP_WRITE;
}

impl PipelineStageAccess {
    pub(crate) fn iter_descriptor_stages(
        descriptor_type: DescriptorType,
        stages_read: ShaderStages,
        stages_write: ShaderStages,
    ) -> impl Iterator<Item = Self> + use<> {
        static MAP_READ: LazyLock<
            HashMap<DescriptorType, HashMap<PipelineStage, PipelineStageAccess>>,
        > = LazyLock::new(|| {
            let uniform_read = [
                DescriptorType::UniformBuffer,
                DescriptorType::UniformBufferDynamic,
            ]
            .into_iter()
            .map(|descriptor_type| {
                (
                    descriptor_type,
                    [
                        (
                            PipelineStage::VertexShader,
                            PipelineStageAccess::VertexShader_UniformRead,
                        ),
                        (
                            PipelineStage::TessellationControlShader,
                            PipelineStageAccess::TessellationControlShader_UniformRead,
                        ),
                        (
                            PipelineStage::TessellationEvaluationShader,
                            PipelineStageAccess::TessellationControlShader_UniformRead,
                        ),
                        (
                            PipelineStage::GeometryShader,
                            PipelineStageAccess::GeometryShader_UniformRead,
                        ),
                        (
                            PipelineStage::FragmentShader,
                            PipelineStageAccess::FragmentShader_UniformRead,
                        ),
                        (
                            PipelineStage::ComputeShader,
                            PipelineStageAccess::ComputeShader_UniformRead,
                        ),
                        (
                            PipelineStage::RayTracingShader,
                            PipelineStageAccess::RayTracingShader_UniformRead,
                        ),
                        (
                            PipelineStage::TaskShader,
                            PipelineStageAccess::TaskShader_UniformRead,
                        ),
                        (
                            PipelineStage::MeshShader,
                            PipelineStageAccess::MeshShader_UniformRead,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            });

            let shader_sampled_read = [
                DescriptorType::CombinedImageSampler,
                DescriptorType::SampledImage,
                DescriptorType::UniformTexelBuffer,
            ]
            .into_iter()
            .map(|descriptor_type| {
                (
                    descriptor_type,
                    [
                        (
                            PipelineStage::VertexShader,
                            PipelineStageAccess::VertexShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::TessellationControlShader,
                            PipelineStageAccess::TessellationControlShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::TessellationEvaluationShader,
                            PipelineStageAccess::TessellationControlShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::GeometryShader,
                            PipelineStageAccess::GeometryShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::FragmentShader,
                            PipelineStageAccess::FragmentShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::ComputeShader,
                            PipelineStageAccess::ComputeShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::RayTracingShader,
                            PipelineStageAccess::RayTracingShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::TaskShader,
                            PipelineStageAccess::TaskShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::MeshShader,
                            PipelineStageAccess::MeshShader_ShaderSampledRead,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            });

            let shader_storage_read = [
                DescriptorType::StorageImage,
                DescriptorType::StorageTexelBuffer,
                DescriptorType::StorageBuffer,
                DescriptorType::StorageBufferDynamic,
            ]
            .into_iter()
            .map(|descriptor_type| {
                (
                    descriptor_type,
                    [
                        (
                            PipelineStage::VertexShader,
                            PipelineStageAccess::VertexShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::TessellationControlShader,
                            PipelineStageAccess::TessellationControlShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::TessellationEvaluationShader,
                            PipelineStageAccess::TessellationControlShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::GeometryShader,
                            PipelineStageAccess::GeometryShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::FragmentShader,
                            PipelineStageAccess::FragmentShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::ComputeShader,
                            PipelineStageAccess::ComputeShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::RayTracingShader,
                            PipelineStageAccess::RayTracingShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::TaskShader,
                            PipelineStageAccess::TaskShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::MeshShader,
                            PipelineStageAccess::MeshShader_ShaderStorageRead,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            });

            let input_attachment_read =
                [DescriptorType::InputAttachment]
                    .into_iter()
                    .map(|descriptor_type| {
                        (
                            descriptor_type,
                            [(
                                PipelineStage::FragmentShader,
                                PipelineStageAccess::FragmentShader_InputAttachmentRead,
                            )]
                            .into_iter()
                            .collect(),
                        )
                    });

            uniform_read
                .chain(shader_sampled_read)
                .chain(shader_storage_read)
                .chain(input_attachment_read)
                .collect()
        });
        static MAP_WRITE: LazyLock<
            HashMap<DescriptorType, HashMap<PipelineStage, PipelineStageAccess>>,
        > = LazyLock::new(|| {
            let shader_storage_write = [
                DescriptorType::StorageImage,
                DescriptorType::StorageTexelBuffer,
                DescriptorType::StorageBuffer,
                DescriptorType::StorageBufferDynamic,
            ]
            .into_iter()
            .map(|descriptor_type| {
                (
                    descriptor_type,
                    [
                        (
                            PipelineStage::VertexShader,
                            PipelineStageAccess::VertexShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::TessellationControlShader,
                            PipelineStageAccess::TessellationControlShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::TessellationEvaluationShader,
                            PipelineStageAccess::TessellationControlShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::GeometryShader,
                            PipelineStageAccess::GeometryShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::FragmentShader,
                            PipelineStageAccess::FragmentShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::ComputeShader,
                            PipelineStageAccess::ComputeShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::RayTracingShader,
                            PipelineStageAccess::RayTracingShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::TaskShader,
                            PipelineStageAccess::TaskShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::MeshShader,
                            PipelineStageAccess::MeshShader_ShaderStorageWrite,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            });

            shader_storage_write.collect()
        });

        [
            (stages_read, &*MAP_READ, "read"),
            (stages_write, &*MAP_WRITE, "write"),
        ]
        .into_iter()
        .filter(|(stages, _, _)| !stages.is_empty())
        .flat_map(move |(stages, descriptor_map, access)| {
            let stages_map = descriptor_map.get(&descriptor_type).unwrap_or_else(|| {
                panic!(
                    "DescriptorType::{:?} does not {} memory",
                    descriptor_type, access,
                )
            });

            PipelineStages::from(stages).into_iter().map(move |stage| {
                *stages_map.get(&stage).unwrap_or_else(|| {
                    panic!(
                        "DescriptorType::{:?} does not {} memory in PipelineStage::{:?}",
                        descriptor_type, access, stage,
                    )
                })
            })
        })
    }
}

impl PipelineStageAccessFlags {
    #[inline]
    pub(crate) fn contains_write(self) -> bool {
        self.intersects(
            PipelineStageAccessFlags::VertexShader_ShaderStorageWrite
                | PipelineStageAccessFlags::TessellationControlShader_ShaderStorageWrite
                | PipelineStageAccessFlags::TessellationEvaluationShader_ShaderStorageWrite
                | PipelineStageAccessFlags::GeometryShader_ShaderStorageWrite
                | PipelineStageAccessFlags::FragmentShader_ShaderStorageWrite
                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite
                | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentWrite
                | PipelineStageAccessFlags::ComputeShader_ShaderStorageWrite
                | PipelineStageAccessFlags::Host_HostWrite
                | PipelineStageAccessFlags::Copy_TransferWrite
                | PipelineStageAccessFlags::Resolve_TransferWrite
                | PipelineStageAccessFlags::Blit_TransferWrite
                | PipelineStageAccessFlags::Clear_TransferWrite
                | PipelineStageAccessFlags::VideoDecode_VideoDecodeWrite
                | PipelineStageAccessFlags::VideoEncode_VideoEncodeWrite
                | PipelineStageAccessFlags::TransformFeedback_TransformFeedbackWrite
                | PipelineStageAccessFlags::TransformFeedback_TransformFeedbackCounterWrite
                | PipelineStageAccessFlags::AccelerationStructureBuild_TransferWrite
                | PipelineStageAccessFlags::AccelerationStructureBuild_AccelerationStructureWrite
                | PipelineStageAccessFlags::RayTracingShader_ShaderStorageWrite
                | PipelineStageAccessFlags::CommandPreprocess_CommandPreprocessWrite
                | PipelineStageAccessFlags::TaskShader_ShaderStorageWrite
                | PipelineStageAccessFlags::MeshShader_ShaderStorageWrite
                | PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureWrite
                | PipelineStageAccessFlags::AccelerationStructureCopy_TransferWrite
                | PipelineStageAccessFlags::OpticalFlow_OpticalFlowWrite
                | PipelineStageAccessFlags::MicromapBuild_MicromapWrite,
        )
    }
}

/// Dependency info for barriers in a pipeline barrier or event command.
///
/// A pipeline barrier creates a dependency between commands submitted before the barrier (the
/// source scope) and commands submitted after it (the destination scope). An event command acts
/// like a split pipeline barrier: the source scope and destination scope are defined
/// relative to different commands. Each `DependencyInfo` consists of multiple individual barriers
/// that concern a either single resource or operate globally.
///
/// Each barrier has a set of source/destination pipeline stages and source/destination memory
/// access types. The pipeline stages create an *execution dependency*: the `src_stages` of
/// commands submitted before the barrier must be completely finished before before any of the
/// `dst_stages` of commands after the barrier are allowed to start. The memory access types
/// create a *memory dependency*: in addition to the execution dependency, any `src_access`
/// performed before the barrier must be made available and visible before any `dst_access`
/// are made after the barrier.
#[derive(Clone, Debug)]
pub struct DependencyInfo {
    /// Flags to modify how the execution and memory dependencies are formed.
    ///
    /// The default value is empty.
    pub dependency_flags: DependencyFlags,

    /// Memory barriers for global operations and accesses, not limited to a single resource.
    ///
    /// The default value is empty.
    pub memory_barriers: SmallVec<[MemoryBarrier; 2]>,

    /// Memory barriers for individual buffers.
    ///
    /// The default value is empty.
    pub buffer_memory_barriers: SmallVec<[BufferMemoryBarrier; 8]>,

    /// Memory barriers for individual images.
    ///
    /// The default value is empty.
    pub image_memory_barriers: SmallVec<[ImageMemoryBarrier; 8]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for DependencyInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyInfo {
    /// Returns a default `DependencyInfo`.
    // TODO: make const
    #[inline]
    pub fn new() -> Self {
        Self {
            dependency_flags: DependencyFlags::empty(),
            memory_barriers: SmallVec::new(),
            buffer_memory_barriers: SmallVec::new(),
            image_memory_barriers: SmallVec::new(),
            _ne: crate::NE,
        }
    }

    /// Returns whether `self` contains any barriers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.memory_barriers.is_empty()
            && self.buffer_memory_barriers.is_empty()
            && self.image_memory_barriers.is_empty()
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that modify how execution and memory dependencies are formed.
    DependencyFlags = DependencyFlags(u32);

    /// For framebuffer-space pipeline stages, specifies that the dependency is framebuffer-local.
    /// The implementation can start the destination operation for some given pixels as long as the
    /// source operation is finished for these given pixels.
    ///
    /// Framebuffer-local dependencies are usually more efficient, especially on tile-based
    /// architectures.
    BY_REGION = BY_REGION,

    /// For devices that consist of multiple physical devices, specifies that the dependency is
    /// device-local. The dependency will only apply to the operations on each physical device
    /// individually, rather than applying to all physical devices as a whole. This allows each
    /// physical device to operate independently of the others.
    ///
    /// The device API version must be at least 1.1, or the [`khr_device_group`] extension must be
    /// enabled on the device.
    ///
    /// [`khr_device_group`]: crate::device::DeviceExtensions::khr_device_group
    DEVICE_GROUP = DEVICE_GROUP
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),


    /// For subpass dependencies, and pipeline barriers executing within a render pass instance,
    /// if the render pass uses multiview rendering, specifies that the dependency is view-local.
    /// Each view in the destination subpass will only depend on a single view in the destination
    /// subpass, instead of all views.
    ///
    /// The device API version must be at least 1.1, or the [`khr_multiview`] extension must be
    /// enabled on the device.
    ///
    /// [`khr_multiview`]: crate::device::DeviceExtensions::khr_multiview
    VIEW_LOCAL = VIEW_LOCAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_multiview)]),
    ]),
}

/// A memory barrier that is applied globally.
#[derive(Clone, Debug)]
pub struct MemoryBarrier {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    pub dst_access: AccessFlags,

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for MemoryBarrier {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBarrier {
    /// Returns a default `MemoryBarrier`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            _ne: crate::NE,
        }
    }
}

/// A memory barrier that is applied to a single buffer.
#[derive(Clone, Debug)]
pub struct BufferMemoryBarrier {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    pub dst_access: AccessFlags,

    /// For resources created with [`Sharing::Exclusive`](crate::sync::Sharing), transfers
    /// ownership of a resource from one queue family to another.
    pub queue_family_ownership_transfer: Option<QueueFamilyOwnershipTransfer>,

    /// The buffer to apply the barrier to.
    pub buffer: Arc<Buffer>,

    /// The byte offset from `buffer` to apply the barrier to.
    pub offset: DeviceSize,

    /// The byte size to apply the barrier to.
    pub size: DeviceSize,

    pub _ne: crate::NonExhaustive<'static>,
}

impl BufferMemoryBarrier {
    /// Returns a default `BufferMemoryBarrier` with the provided `buffer`.
    #[inline]
    pub const fn new(buffer: Arc<Buffer>) -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            queue_family_ownership_transfer: None,
            buffer,
            offset: 0,
            size: 0,
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn buffer(buffer: Arc<Buffer>) -> Self {
        Self::new(buffer)
    }
}

/// A memory barrier that is applied to a single image.
#[derive(Clone, Debug)]
pub struct ImageMemoryBarrier {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub dst_access: AccessFlags,

    /// The layout that the specified `subresource_range` of `image` is expected to be in when the
    /// source scope completes.
    ///
    /// The default value is [`ImageLayout::Undefined`].
    pub old_layout: ImageLayout,

    /// The layout that the specified `subresource_range` of `image` will be transitioned to before
    /// the destination scope begins.
    ///
    /// The default value is [`ImageLayout::Undefined`], which must be overridden.
    pub new_layout: ImageLayout,

    /// For resources created with [`Sharing::Exclusive`](crate::sync::Sharing), transfers
    /// ownership of a resource from one queue family to another.
    ///
    /// The default value is `None`.
    pub queue_family_ownership_transfer: Option<QueueFamilyOwnershipTransfer>,

    /// The image to apply the barrier to.
    ///
    /// There is no default value.
    pub image: Arc<Image>,

    /// The subresource range of `image` to apply the barrier to.
    ///
    /// The default value is [`ImageSubresourceRange::default()`].
    pub subresource_range: ImageSubresourceRange,

    pub _ne: crate::NonExhaustive<'static>,
}

impl ImageMemoryBarrier {
    /// Returns a default `ImageMemoryBarrier` with the provided `image`.
    #[inline]
    pub const fn new(image: Arc<Image>) -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::Undefined,
            queue_family_ownership_transfer: None,
            image,
            subresource_range: ImageSubresourceRange::new(),
            _ne: crate::NE,
        }
    }

    #[inline]
    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    pub fn image(image: Arc<Image>) -> Self {
        Self::new(image)
    }
}

/// Specifies a queue family ownership transfer for a resource.
///
/// There are three classes of queues that can be used in an ownership transfer:
/// - A **local** queue exists on the current [`Instance`] and [`Device`].
/// - An **external** queue does not exist on the current [`Instance`], but has the same
///   [`device_uuid`] and [`driver_uuid`] as the current [`Device`].
/// - A **foreign** queue can be an external queue, or any queue on another device for which the
///   mentioned parameters do not match.
///
/// [`Instance`]: crate::instance::Instance
/// [`device_uuid`]: crate::device::DeviceProperties::device_uuid
/// [`driver_uuid`]: crate::device::DeviceProperties::driver_uuid
#[derive(Clone, Copy, Debug)]
pub enum QueueFamilyOwnershipTransfer {
    /// For a resource with [`Sharing::Exclusive`], transfers ownership between two local queues.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    ExclusiveBetweenLocal {
        /// The queue family that currently owns the resource.
        src_index: u32,

        /// The queue family to transfer ownership to.
        dst_index: u32,
    },

    /// For a resource with [`Sharing::Exclusive`], transfers ownership from a local queue to an
    /// external queue.
    ///
    /// The device API version must be at least 1.1, or the [`khr_external_memory`] extension must
    /// be enabled on the device.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    ExclusiveToExternal {
        /// The queue family that currently owns the resource.
        src_index: u32,
    },

    /// For a resource with [`Sharing::Exclusive`], transfers ownership from an external queue to a
    /// local queue.
    ///
    /// The device API version must be at least 1.1, or the [`khr_external_memory`] extension must
    /// be enabled on the device.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    ExclusiveFromExternal {
        /// The queue family to transfer ownership to.
        dst_index: u32,
    },

    /// For a resource with [`Sharing::Exclusive`], transfers ownership from a local queue to a
    /// foreign queue.
    ///
    /// The [`ext_queue_family_foreign`] extension must be enabled on the device.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    /// [`ext_queue_family_foreign`]: crate::device::DeviceExtensions::ext_queue_family_foreign
    ExclusiveToForeign {
        /// The queue family that currently owns the resource.
        src_index: u32,
    },

    /// For a resource with [`Sharing::Exclusive`], transfers ownership from a foreign queue to a
    /// local queue.
    ///
    /// The [`ext_queue_family_foreign`] extension must be enabled on the device.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    /// [`ext_queue_family_foreign`]: crate::device::DeviceExtensions::ext_queue_family_foreign
    ExclusiveFromForeign {
        /// The queue family to transfer ownership to.
        dst_index: u32,
    },

    /// For a resource with [`Sharing::Concurrent`], transfers ownership from its local queues to
    /// an external queue.
    ///
    /// The device API version must be at least 1.1, or the [`khr_external_memory`] extension must
    /// be enabled on the device.
    ///
    /// [`Sharing::Concurrent`]: crate::sync::Sharing::Concurrent
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    ConcurrentToExternal,

    /// For a resource with [`Sharing::Concurrent`], transfers ownership from an external queue to
    /// its local queues.
    ///
    /// The device API version must be at least 1.1, or the [`khr_external_memory`] extension must
    /// be enabled on the device.
    ///
    /// [`Sharing::Concurrent`]: crate::sync::Sharing::Concurrent
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    ConcurrentFromExternal,

    /// For a resource with [`Sharing::Concurrent`], transfers ownership from its local queues to
    /// a foreign queue.
    ///
    /// The [`ext_queue_family_foreign`] extension must be enabled on the device.
    ///
    /// [`Sharing::Concurrent`]: crate::sync::Sharing::Concurrent
    /// [`ext_queue_family_foreign`]: crate::device::DeviceExtensions::ext_queue_family_foreign
    ConcurrentToForeign,

    /// For a resource with [`Sharing::Concurrent`], transfers ownership from a foreign queue to
    /// its local queues.
    ///
    /// The [`ext_queue_family_foreign`] extension must be enabled on the device.
    ///
    /// [`Sharing::Concurrent`]: crate::sync::Sharing::Concurrent
    /// [`ext_queue_family_foreign`]: crate::device::DeviceExtensions::ext_queue_family_foreign
    ConcurrentFromForeign,
}

impl QueueFamilyOwnershipTransfer {
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_vk(&self) -> (u32, u32) {
        match *self {
            Self::ExclusiveBetweenLocal {
                src_index,
                dst_index,
            } => (src_index, dst_index),
            Self::ExclusiveToExternal { src_index } => (src_index, vk::QUEUE_FAMILY_EXTERNAL),
            Self::ExclusiveFromExternal { dst_index } => (vk::QUEUE_FAMILY_EXTERNAL, dst_index),
            Self::ExclusiveToForeign { src_index } => (src_index, vk::QUEUE_FAMILY_FOREIGN_EXT),
            Self::ExclusiveFromForeign { dst_index } => (vk::QUEUE_FAMILY_FOREIGN_EXT, dst_index),
            Self::ConcurrentToExternal => (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_EXTERNAL),
            Self::ConcurrentFromExternal => (vk::QUEUE_FAMILY_EXTERNAL, vk::QUEUE_FAMILY_IGNORED),
            Self::ConcurrentToForeign => (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_FOREIGN_EXT),
            Self::ConcurrentFromForeign => (vk::QUEUE_FAMILY_FOREIGN_EXT, vk::QUEUE_FAMILY_IGNORED),
        }
    }
}
