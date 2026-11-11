//! Gather information about rendering, held in query pools.
//!
//! In Vulkan, queries are not created individually. Instead you manipulate **query pools**, which
//! represent a collection of queries. Whenever you use a query, you have to specify both the query
//! pool and the slot id within that query pool.

use crate::{
    buffer::BufferContents,
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_enum},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError, VulkanObject,
};
use ash::vk;
use std::{
    mem::{size_of_val, MaybeUninit},
    num::NonZero,
    ptr,
    sync::Arc,
};

/// A collection of one or more queries of a particular type.
#[derive(Debug)]
pub struct QueryPool {
    handle: vk::QueryPool,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    query_type: QueryType,
    query_count: u32,
    pipeline_statistics: QueryPipelineStatisticFlags,
}

impl QueryPool {
    /// Creates a new `QueryPool`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        create_info: &QueryPoolCreateInfo<'_>,
    ) -> Result<Arc<QueryPool>, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `QueryPool`.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &QueryPoolCreateInfo<'_>,
    ) -> Result<Arc<QueryPool>, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &QueryPoolCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &QueryPoolCreateInfo<'_>,
    ) -> Result<Arc<QueryPool>, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let fns = device.fns();
        let handle = {
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_query_pool)(
                    device.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle(device, handle, create_info) })
    }

    /// Creates a new `QueryPool` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::QueryPool,
        create_info: &QueryPoolCreateInfo<'_>,
    ) -> Arc<QueryPool> {
        let &QueryPoolCreateInfo {
            query_type,
            query_count,
            pipeline_statistics,
            _ne: _,
        } = create_info;

        Arc::new(QueryPool {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),
            query_type,
            query_count,
            pipeline_statistics,
        })
    }

    /// Returns the query type of the pool.
    #[inline]
    pub fn query_type(&self) -> QueryType {
        self.query_type
    }

    /// Returns the number of query slots of this query pool.
    #[inline]
    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    /// Returns the pipeline statistics flags of this query pool.
    #[inline]
    pub fn pipeline_statistics(&self) -> QueryPipelineStatisticFlags {
        self.pipeline_statistics
    }

    /// Returns the number of [`QueryResultElement`]s that are needed to hold the result of a
    /// single query of this type.
    #[inline]
    pub const fn result_len(&self, result_flags: QueryResultFlags) -> DeviceSize {
        (match self.query_type {
            QueryType::Occlusion
            | QueryType::Timestamp
            | QueryType::AccelerationStructureCompactedSize
            | QueryType::AccelerationStructureSerializationSize
            | QueryType::AccelerationStructureSerializationBottomLevelPointers
            | QueryType::AccelerationStructureSize
            | QueryType::MeshPrimitivesGenerated => 1,
            QueryType::PipelineStatistics => self.pipeline_statistics.count() as DeviceSize,
        }) + result_flags.intersects(QueryResultFlags::WITH_AVAILABILITY) as DeviceSize
    }

    /// Copies the results of a range of queries to a buffer on the CPU, panicking on a validation
    /// error.
    ///
    /// [`self.result_len(flags)`] elements will be written for each query in the range, plus 1
    /// extra element per query if [`WITH_AVAILABILITY`] is enabled. The provided buffer must be
    /// large enough to hold the data.
    ///
    /// `true` is returned if every result was available and written to the buffer. `false` is
    /// returned if some results were not yet available; these will not be written to the buffer.
    ///
    /// See also [`copy_query_pool_results`].
    ///
    /// This is a shortcut for `try_get_results().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_get_results`] returns a [`ValidationError`].
    ///
    /// [`self.result_len(flags)`]: QueryPool::result_len
    /// [`WITH_AVAILABILITY`]: QueryResultFlags::WITH_AVAILABILITY
    /// [`copy_query_pool_results`]: crate::command_buffer::AutoCommandBufferBuilder::copy_query_pool_results
    /// [`try_get_results`]: Self::try_get_results
    #[inline]
    #[track_caller]
    pub fn get_results<T>(
        &self,
        first_query: u32,
        query_count: u32,
        destination: &mut [T],
        flags: QueryResultFlags,
    ) -> Result<bool, VulkanError>
    where
        T: QueryResultElement,
    {
        match self.try_get_results(first_query, query_count, destination, flags) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Copies the results of a range of queries to a buffer on the CPU.
    ///
    /// [`self.result_len(flags)`] elements will be written for each query in the range, plus 1
    /// extra element per query if [`WITH_AVAILABILITY`] is enabled. The provided buffer must be
    /// large enough to hold the data.
    ///
    /// `true` is returned if every result was available and written to the buffer. `false` is
    /// returned if some results were not yet available; these will not be written to the buffer.
    ///
    /// See also [`copy_query_pool_results`].
    ///
    /// [`self.result_len(flags)`]: QueryPool::result_len
    /// [`WITH_AVAILABILITY`]: QueryResultFlags::WITH_AVAILABILITY
    /// [`copy_query_pool_results`]: crate::command_buffer::AutoCommandBufferBuilder::copy_query_pool_results
    #[inline]
    pub fn try_get_results<T>(
        &self,
        first_query: u32,
        query_count: u32,
        destination: &mut [T],
        flags: QueryResultFlags,
    ) -> Result<bool, Validated<VulkanError>>
    where
        T: QueryResultElement,
    {
        self.validate_get_results(first_query, query_count, destination, flags)?;

        Ok(unsafe { self.get_results_unchecked(first_query, query_count, destination, flags) }?)
    }

    fn validate_get_results<T>(
        &self,
        first_query: u32,
        query_count: u32,
        destination: &[T],
        flags: QueryResultFlags,
    ) -> Result<(), Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        flags.validate_device(&self.device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-vkGetQueryPoolResults-flags-parameter"])
        })?;

        if destination.is_empty() {
            return Err(Box::new(ValidationError {
                context: "destination".into(),
                problem: "is empty".into(),
                vuids: &["VUID-vkGetQueryPoolResults-dataSize-arraylength"],
                ..Default::default()
            }));
        }

        if !first_query
            .checked_add(query_count)
            .is_some_and(|end| end <= self.query_count)
        {
            return Err(Box::new(ValidationError {
                problem: "`first_query + query_count` is greater than `self.query_count()`".into(),
                vuids: &[
                    "VUID-vkGetQueryPoolResults-firstQuery-00813",
                    "VUID-vkGetQueryPoolResults-firstQuery-00816",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkGetQueryPoolResults-flags-02828
        // VUID-vkGetQueryPoolResults-flags-00815
        // Ensured by the Rust alignment of T.

        // VUID-vkGetQueryPoolResults-stride-08993
        // Ensured by choosing the stride ourselves.

        let per_query_len = self.result_len(flags);
        let required_len = per_query_len * query_count as DeviceSize;

        if (destination.len() as DeviceSize) < required_len {
            return Err(Box::new(ValidationError {
                context: "destination.len()".into(),
                problem: "is less than the number of elements required for the query type, and \
                    the provided range and flags"
                    .into(),
                vuids: &["VUID-vkGetQueryPoolResults-dataSize-00817"],
                ..Default::default()
            }));
        }

        if self.query_type == QueryType::Timestamp {
            if flags.intersects(QueryResultFlags::PARTIAL) {
                return Err(Box::new(ValidationError {
                    problem: "`self.query_type()` is `QueryType::Timestamp`, but \
                        `flags` contains `QueryResultFlags::PARTIAL`"
                        .into(),
                    vuids: &["VUID-vkGetQueryPoolResults-queryType-00818"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn get_results_unchecked<T>(
        &self,
        first_query: u32,
        query_count: u32,
        destination: &mut [T],
        flags: QueryResultFlags,
    ) -> Result<bool, VulkanError>
    where
        T: QueryResultElement,
    {
        let per_query_len = self.result_len(flags);
        let stride = per_query_len * size_of::<T>() as DeviceSize;

        let fns = self.device.fns();
        let result = unsafe {
            (fns.v1_0.get_query_pool_results)(
                self.device.handle(),
                self.handle(),
                first_query,
                query_count,
                size_of_val(destination),
                destination.as_mut_ptr().cast(),
                stride,
                vk::QueryResultFlags::from(flags) | T::FLAG,
            )
        };

        match result {
            vk::Result::SUCCESS => Ok(true),
            vk::Result::NOT_READY => Ok(false),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Resets a range of queries, panicking on a validation error.
    ///
    /// The [`host_query_reset`] feature must be enabled on the device.
    ///
    /// This is a shortcut for `try_reset().unwrap()`.
    ///
    /// # Safety
    ///
    /// For the queries indicated by `first_query` and `query_count`:
    /// - There must be no operations pending or executing on the device.
    /// - There must be no calls to `reset*` or `get_results*` executing concurrently on another
    ///   thread.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_reset`] returns a [`ValidationError`].
    ///
    /// [`host_query_reset`]: crate::device::DeviceFeatures::host_query_reset
    /// [`try_reset`]: Self::try_reset
    #[inline]
    #[track_caller]
    pub unsafe fn reset(&self, first_query: u32, query_count: u32) {
        unsafe { self.try_reset(first_query, query_count) }.unwrap()
    }

    /// Resets a range of queries.
    ///
    /// The [`host_query_reset`] feature must be enabled on the device.
    ///
    /// # Safety
    ///
    /// For the queries indicated by `first_query` and `query_count`:
    /// - There must be no operations pending or executing on the device.
    /// - There must be no calls to `reset*` or `get_results*` executing concurrently on another
    ///   thread.
    ///
    /// [`host_query_reset`]: crate::device::DeviceFeatures::host_query_reset
    #[inline]
    pub unsafe fn try_reset(
        &self,
        first_query: u32,
        query_count: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.validate_reset(first_query, query_count)?;

        unsafe { self.reset_unchecked(first_query, query_count) };

        Ok(())
    }

    fn validate_reset(
        &self,
        first_query: u32,
        query_count: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_features().host_query_reset {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "host_query_reset",
                )])]),
                vuids: &["VUID-vkResetQueryPool-None-02665"],
                ..Default::default()
            }));
        }

        if query_count == 0 {
            return Err(Box::new(ValidationError {
                context: "query_count".into(),
                problem: "is zero".into(),
                // vuids?
                ..Default::default()
            }));
        }

        if !first_query
            .checked_add(query_count)
            .is_some_and(|end| end <= self.query_count)
        {
            return Err(Box::new(ValidationError {
                problem: "`first_query + query_count` is greater than `self.query_count()`".into(),
                vuids: &[
                    "VUID-vkResetQueryPool-firstQuery-09436",
                    "VUID-vkResetQueryPool-firstQuery-09437",
                ],
                ..Default::default()
            }));
        }

        // unsafe
        // VUID-vkResetQueryPool-firstQuery-02741
        // VUID-vkResetQueryPool-firstQuery-02742

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_unchecked(&self, first_query: u32, query_count: u32) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_2 {
            unsafe {
                (fns.v1_2.reset_query_pool)(
                    self.device.handle(),
                    self.handle(),
                    first_query,
                    query_count,
                )
            };
        } else {
            debug_assert!(self.device.enabled_extensions().ext_host_query_reset);
            unsafe {
                (fns.ext_host_query_reset.reset_query_pool_ext)(
                    self.device.handle(),
                    self.handle(),
                    first_query,
                    query_count,
                )
            };
        }
    }
}

impl Drop for QueryPool {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.destroy_query_pool)(self.device.handle(), self.handle, ptr::null()) };
    }
}

unsafe impl VulkanObject for QueryPool {
    type Handle = vk::QueryPool;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for QueryPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(QueryPool);

/// Parameters to create a new `QueryPool`.
#[derive(Clone, Debug)]
pub struct QueryPoolCreateInfo<'a> {
    /// The type of query that the pool should be for.
    ///
    /// There is no default value.
    pub query_type: QueryType,

    /// The number of queries to create in the pool.
    ///
    /// The default value is `0`, which must be overridden.
    pub query_count: u32,

    /// If `query_type` is [`QueryType::PipelineStatistics`], the statistics to query.
    ///
    /// For any other value of `query_type`, this must be empty.
    ///
    /// The default value is empty.
    pub pipeline_statistics: QueryPipelineStatisticFlags,

    pub _ne: crate::NonExhaustive<'a>,
}

impl QueryPoolCreateInfo<'_> {
    /// Returns a default `QueryPoolCreateInfo` with the provided `query_type`.
    #[inline]
    pub const fn new(query_type: QueryType) -> Self {
        Self {
            query_type,
            query_count: 0,
            pipeline_statistics: QueryPipelineStatisticFlags::empty(),
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn query_type(query_type: QueryType) -> Self {
        Self::new(query_type)
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            query_type,
            query_count,
            pipeline_statistics,
            _ne: _,
        } = self;

        query_type.validate_device(device).map_err(|err| {
            err.add_context("query_type")
                .set_vuids(&["VUID-VkQueryPoolCreateInfo-queryType-parameter"])
        })?;

        if query_count == 0 {
            return Err(Box::new(ValidationError {
                context: "query_count".into(),
                problem: "is 0".into(),
                vuids: &["VUID-VkQueryPoolCreateInfo-queryCount-02763"],
                ..Default::default()
            }));
        }

        if query_type == QueryType::PipelineStatistics {
            if !device.enabled_features().pipeline_statistics_query {
                return Err(Box::new(ValidationError {
                    context: "query_type".into(),
                    problem: "is `QueryType::PipelineStatistics`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "pipeline_statistics_query",
                    )])]),
                    vuids: &["VUID-VkQueryPoolCreateInfo-queryType-00791"],
                }));
            }

            pipeline_statistics.validate_device(device).map_err(|err| {
                err.add_context("pipeline_statistics")
                    .set_vuids(&["VUID-VkQueryPoolCreateInfo-queryType-00792"])
            })?;

            if pipeline_statistics.intersects(
                QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS
                    | QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS,
            ) && !device.enabled_features().mesh_shader_queries
            {
                return Err(Box::new(ValidationError {
                    context: "pipeline_statistics".into(),
                    problem: "contains `TASK_SHADER_INVOCATIONS` or \
                        `MESH_SHADER_INVOCATIONS`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader_queries",
                    )])]),
                    vuids: &["VUID-VkQueryPoolCreateInfo-meshShaderQueries-07069"],
                }));
            }
        } else if !pipeline_statistics.is_empty() {
            return Err(Box::new(ValidationError {
                problem: "`query_type` is not `QueryType::PipelineStatistics`, but \
                    `pipeline_statistics` is not empty"
                    .into(),
                ..Default::default()
            }));
        }

        if query_type == QueryType::MeshPrimitivesGenerated
            && !device.enabled_features().mesh_shader_queries
        {
            return Err(Box::new(ValidationError {
                context: "query_type".into(),
                problem: "is `QueryType::MeshPrimitivesGenerated`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "mesh_shader_queries",
                )])]),
                vuids: &["VUID-VkQueryPoolCreateInfo-meshShaderQueries-07068"],
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::QueryPoolCreateInfo<'static> {
        let &Self {
            query_type,
            query_count,
            pipeline_statistics,
            _ne: _,
        } = self;

        vk::QueryPoolCreateInfo::default()
            .flags(vk::QueryPoolCreateFlags::empty())
            .query_type(query_type.into())
            .query_count(query_count)
            .pipeline_statistics(pipeline_statistics.into())
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The type of query that a query pool should perform.
    QueryType = QueryType(i32);

    /// Tracks the number of samples that pass per-fragment tests (e.g. the depth test).
    ///
    /// Used with the [`begin_query`] and [`end_query`] commands.
    ///
    /// [`begin_query`]: crate::command_buffer::AutoCommandBufferBuilder::begin_query
    /// [`end_query`]: crate::command_buffer::AutoCommandBufferBuilder::end_query
    Occlusion = OCCLUSION,

    /// Tracks statistics on pipeline invocations and their input data.
    ///
    /// Used with the [`begin_query`] and [`end_query`] commands.
    ///
    /// [`begin_query`]: crate::command_buffer::AutoCommandBufferBuilder::begin_query
    /// [`end_query`]: crate::command_buffer::AutoCommandBufferBuilder::end_query
    PipelineStatistics = PIPELINE_STATISTICS,

    /// Writes timestamps at chosen points in a command buffer.
    ///
    /// Used with the [`write_timestamp`] command.
    ///
    /// [`write_timestamp`]: crate::command_buffer::AutoCommandBufferBuilder::write_timestamp
    Timestamp = TIMESTAMP,

    /// Queries the size of data resulting from a
    /// [`CopyAccelerationStructureMode::Compact`] operation.
    ///
    /// Used with the [`write_acceleration_structures_properties`] command.
    ///
    /// [`CopyAccelerationStructureMode::Compact`]: crate::acceleration_structure::CopyAccelerationStructureMode::Compact
    /// [`write_acceleration_structures_properties`]: crate::command_buffer::AutoCommandBufferBuilder::write_acceleration_structures_properties
    AccelerationStructureCompactedSize = ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
    ]),

    /// Queries the size of data resulting from a
    /// [`CopyAccelerationStructureMode::Serialize`] operation.
    ///
    /// Used with the [`write_acceleration_structures_properties`] command.
    ///
    /// [`CopyAccelerationStructureMode::Serialize`]: crate::acceleration_structure::CopyAccelerationStructureMode::Serialize
    /// [`write_acceleration_structures_properties`]: crate::command_buffer::AutoCommandBufferBuilder::write_acceleration_structures_properties
    AccelerationStructureSerializationSize = ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
    ]),

    /// For a top-level acceleration structure, queries the number of bottom-level acceleration
    /// structure handles that will be written during a
    /// [`CopyAccelerationStructureMode::Serialize`] operation.
    ///
    /// Used with the [`write_acceleration_structures_properties`] command.
    ///
    /// [`CopyAccelerationStructureMode::Serialize`]: crate::acceleration_structure::CopyAccelerationStructureMode::Serialize
    /// [`write_acceleration_structures_properties`]: crate::command_buffer::AutoCommandBufferBuilder::write_acceleration_structures_properties
    AccelerationStructureSerializationBottomLevelPointers = ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_maintenance1)]),
    ]),

    /// Queries the total size of an acceleration structure.
    ///
    /// Used with the [`write_acceleration_structures_properties`] command.
    ///
    /// [`write_acceleration_structures_properties`]: crate::command_buffer::AutoCommandBufferBuilder::write_acceleration_structures_properties
    AccelerationStructureSize = ACCELERATION_STRUCTURE_SIZE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_maintenance1)]),
    ]),

    /// Queries the number of primitives emitted from a mesh shader that reach the fragment shader.
    ///
    /// Used with the [`begin_query`] and [`end_query`] commands.
    ///

    /// [`begin_query`]: crate::command_buffer::AutoCommandBufferBuilder::begin_query
    /// [`end_query`]: crate::command_buffer::AutoCommandBufferBuilder::end_query
    MeshPrimitivesGenerated = MESH_PRIMITIVES_GENERATED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
    ]),
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that control how a query is to be executed.
    QueryControlFlags = QueryControlFlags(u32);

    /// For occlusion queries, specifies that the result must reflect the exact number of
    /// tests passed. If not enabled, the query may return a result of 1 even if more fragments
    /// passed the test.
    PRECISE = PRECISE,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// For `PipelineStatistics` queries, the statistics that should be gathered.
    QueryPipelineStatisticFlags impl {
        /// Returns `true` if `self` contains any flags referring to compute operations.
        #[inline]
        pub const fn is_compute(self) -> bool {
            self.intersects(QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS)
        }

        /// Returns `true` if `self` contains any flags referring to graphics operations.
        #[inline]
        pub const fn is_graphics(self) -> bool {
            self.is_primitive_shading_graphics() || self.is_mesh_shading_graphics() ||
            self.intersects(QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS)
        }

        /// Returns `true` if `self` contains any flags referring to primitive shading graphics
        /// operations.
        #[inline]
        pub const fn is_primitive_shading_graphics(self) -> bool {
            self.intersects(
                QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES
                    .union(QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES)
                    .union(QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS)
                    .union(QueryPipelineStatisticFlags::GEOMETRY_SHADER_INVOCATIONS)
                    .union(QueryPipelineStatisticFlags::GEOMETRY_SHADER_PRIMITIVES)
                    .union(QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS)
                    .union(QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES)
                    .union(QueryPipelineStatisticFlags::TESSELLATION_CONTROL_SHADER_PATCHES)
                    .union(QueryPipelineStatisticFlags::TESSELLATION_EVALUATION_SHADER_INVOCATIONS),
            )
        }

        /// Returns `true` if `self` contains any flags referring to mesh shading graphics
        /// operations.
        #[inline]
        pub const fn is_mesh_shading_graphics(self) -> bool {
            self.intersects(
                QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS
                    .union(QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS),
            )
        }
    }
    = QueryPipelineStatisticFlags(u32);

    /// Count the number of vertices processed by the input assembly.
    INPUT_ASSEMBLY_VERTICES = INPUT_ASSEMBLY_VERTICES,

    /// Count the number of primitives processed by the input assembly.
    INPUT_ASSEMBLY_PRIMITIVES = INPUT_ASSEMBLY_PRIMITIVES,

    /// Count the number of times a vertex shader is invoked.
    VERTEX_SHADER_INVOCATIONS = VERTEX_SHADER_INVOCATIONS,

    /// Count the number of times a geometry shader is invoked.
    GEOMETRY_SHADER_INVOCATIONS = GEOMETRY_SHADER_INVOCATIONS,

    /// Count the number of primitives generated by geometry shaders.
    GEOMETRY_SHADER_PRIMITIVES = GEOMETRY_SHADER_PRIMITIVES,

    /// Count the number of times the clipping stage is invoked on a primitive.
    CLIPPING_INVOCATIONS = CLIPPING_INVOCATIONS,

    /// Count the number of primitives that are output by the clipping stage.
    CLIPPING_PRIMITIVES = CLIPPING_PRIMITIVES,

    /// Count the number of times a fragment shader is invoked.
    FRAGMENT_SHADER_INVOCATIONS = FRAGMENT_SHADER_INVOCATIONS,

    /// Count the number of patches processed by a tessellation control shader.
    TESSELLATION_CONTROL_SHADER_PATCHES = TESSELLATION_CONTROL_SHADER_PATCHES,

    /// Count the number of times a tessellation evaluation shader is invoked.
    TESSELLATION_EVALUATION_SHADER_INVOCATIONS = TESSELLATION_EVALUATION_SHADER_INVOCATIONS,

    /// Count the number of times a compute shader is invoked.
    COMPUTE_SHADER_INVOCATIONS = COMPUTE_SHADER_INVOCATIONS,

    /// Count the number of times a task shader is invoked.
    TASK_SHADER_INVOCATIONS = TASK_SHADER_INVOCATIONS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
    ]),

    /// Count the number of times a mesh shader is invoked.
    MESH_SHADER_INVOCATIONS = MESH_SHADER_INVOCATIONS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
    ]),
}

/// A trait for elements of buffers that can be used as a destination for query results.
///
/// # Safety
/// This is implemented for `u32` and `u64`. Unless you really know what you're doing, you should
/// not implement this trait for any other type.
pub unsafe trait QueryResultElement: BufferContents + Sized {
    const FLAG: vk::QueryResultFlags;
}

unsafe impl QueryResultElement for u32 {
    const FLAG: vk::QueryResultFlags = vk::QueryResultFlags::empty();
}

unsafe impl QueryResultElement for u64 {
    const FLAG: vk::QueryResultFlags = vk::QueryResultFlags::TYPE_64;
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags to control how the results of a query should be retrieved.
    ///
    /// `VK_QUERY_RESULT_64_BIT` is not included, as it is determined automatically via the
    /// [`QueryResultElement`] trait.
    QueryResultFlags = QueryResultFlags(u32);

    /// Wait for the results to become available before writing the results.
    WAIT = WAIT,

    /// Write an additional element to the end of each query's results, indicating the availability
    /// of the results:
    /// - Nonzero: The results are available, and have been written to the element(s) preceding.
    /// - Zero: The results are not yet available, and have not been written.
    WITH_AVAILABILITY = WITH_AVAILABILITY,

    /// Allow writing partial results to the buffer, instead of waiting until they are fully
    /// available.
    PARTIAL = PARTIAL,

    /* TODO: enable
    // TODO: document
    WITH_STATUS = WITH_STATUS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_queue)]),
    ]),*/
}

#[cfg(test)]
mod tests {
    use super::QueryPoolCreateInfo;
    use crate::{
        query::{QueryPool, QueryType},
        Validated,
    };

    #[test]
    fn pipeline_statistics_feature() {
        let (device, _) = gfx_dev_and_queue!();
        assert!(matches!(
            QueryPool::try_new(
                &device,
                &QueryPoolCreateInfo {
                    query_count: 256,
                    ..QueryPoolCreateInfo::new(QueryType::PipelineStatistics)
                },
            ),
            Err(Validated::ValidationError(_)),
        ));
    }
}
