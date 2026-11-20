use crate::{
    buffer::Subbuffer,
    command_buffer::{
        auto::{QueryState, Resource},
        sys::RecordingCommandBuffer,
        AutoCommandBufferBuilder, ResourceInCommand,
    },
    query::{QueryControlFlags, QueryPool, QueryResultElement, QueryResultFlags},
    sync::{PipelineStage, PipelineStageAccessFlags},
    ValidationError,
};
use ash::vk::DeviceSize;
use std::{ops::Range, sync::Arc};

/// # Commands related to queries.
impl<L> AutoCommandBufferBuilder<L> {
    /// Begins a query.
    ///
    /// The query will be active until [`end_query`](Self::end_query) is called for the same query.
    ///
    /// # Safety
    ///
    /// The query must be unavailable, ensured by calling
    /// [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn begin_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_begin_query(&query_pool, query, flags)?;

        Ok(unsafe { self.begin_query_unchecked(query_pool, query, flags) })
    }

    fn validate_begin_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
        flags: QueryControlFlags,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_begin_query(query_pool, query, flags)?;

        if self
            .builder_state
            .queries
            .contains_key(&query_pool.query_type())
        {
            return Err(Box::new(ValidationError {
                problem: "a query with the same type as `query_pool.query_type()` is \
                    already active"
                    .into(),
                vuids: &["VUID-vkCmdBeginQuery-queryPool-01922"],
                ..Default::default()
            }));
        }

        if let Some(render_pass_state) = &self.builder_state.render_pass {
            let view_mask = render_pass_state.rendering_info.as_ref().view_mask;

            if query + view_mask.count_ones() > query_pool.query_count() {
                return Err(Box::new(ValidationError {
                    problem: "a render subpass with a non-zero `view_mask` is active, but \
                        `query` + the number of views in `view_mask` is greater than \
                        `query_pool.query_count()`"
                        .into(),
                    vuids: &["VUID-vkCmdBeginQuery-query-00808"],
                    ..Default::default()
                }));
            }
        }

        // VUID-vkCmdBeginQuery-None-00807
        // Not checked, therefore unsafe.
        // TODO: add check.

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_query_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        flags: QueryControlFlags,
    ) -> &mut Self {
        self.builder_state.queries.insert(
            query_pool.query_type(),
            QueryState {
                query_pool: query_pool.clone(),
                query,
                flags,
                in_subpass: self.builder_state.render_pass.is_some(),
            },
        );

        self.add_command(
            "begin_query",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.begin_query_unchecked(&query_pool, query, flags) };
            },
        );

        self
    }

    /// Ends an active query.
    pub fn end_query(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_end_query(&query_pool, query)?;

        Ok(unsafe { self.end_query_unchecked(query_pool, query) })
    }

    fn validate_end_query(
        &self,
        query_pool: &QueryPool,
        query: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_end_query(query_pool, query)?;

        if !self
            .builder_state
            .queries
            .get(&query_pool.query_type())
            .is_some_and(|state| *state.query_pool == *query_pool && state.query == query)
        {
            return Err(Box::new(ValidationError {
                problem: "no query with the same type as `query_pool.query_type()` is active"
                    .into(),
                vuids: &["VUID-vkCmdEndQuery-None-01923"],
                ..Default::default()
            }));
        }

        if let Some(render_pass_state) = &self.builder_state.render_pass {
            let view_mask = render_pass_state.rendering_info.as_ref().view_mask;

            if query + view_mask.count_ones() > query_pool.query_count() {
                return Err(Box::new(ValidationError {
                    problem: "a render subpass with a non-zero `view_mask` is active, but \
                        `query` + the number of views in `view_mask` is greater than \
                        `query_pool.query_count()`"
                        .into(),
                    vuids: &["VUID-vkCmdEndQuery-query-00812"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_query_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
    ) -> &mut Self {
        self.builder_state.queries.remove(&query_pool.query_type());

        self.add_command(
            "end_query",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.end_query_unchecked(&query_pool, query) };
            },
        );

        self
    }

    /// Writes a timestamp to a timestamp query.
    ///
    /// # Safety
    ///
    /// The query must be unavailable, ensured by calling
    /// [`reset_query_pool`](Self::reset_query_pool).
    pub unsafe fn write_timestamp(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_write_timestamp(&query_pool, query, stage)?;

        Ok(unsafe { self.write_timestamp_unchecked(query_pool, query, stage) })
    }

    fn validate_write_timestamp(
        &self,
        query_pool: &QueryPool,
        query: u32,
        stage: PipelineStage,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_write_timestamp(query_pool, query, stage)?;

        if let Some(render_pass_state) = &self.builder_state.render_pass {
            let view_mask = render_pass_state.rendering_info.as_ref().view_mask;

            if query + view_mask.count_ones() > query_pool.query_count() {
                return Err(Box::new(ValidationError {
                    problem: "a render subpass with a non-zero `view_mask` is active, but \
                        `query` + the number of views in `view_mask` is greater than \
                        `query_pool.query_count()`"
                        .into(),
                    vuids: &["VUID-vkCmdWriteTimestamp2-query-03865"],
                    ..Default::default()
                }));
            }
        }

        // VUID-vkCmdWriteTimestamp2-queryPool-03862
        // VUID-vkCmdWriteTimestamp2-None-03864
        // Not checked, therefore unsafe.
        // TODO: add check.

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_timestamp_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> &mut Self {
        self.add_command(
            "write_timestamp",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.write_timestamp_unchecked(&query_pool, query, stage) };
            },
        );

        self
    }

    /// Copies the results of a range of queries to a buffer on the GPU.
    ///
    /// [`query_pool.ty().result_len()`] elements will be written for each query in the range, plus
    /// 1 extra element per query if [`QueryResultFlags::WITH_AVAILABILITY`] is enabled.
    /// The provided buffer must be large enough to hold the data.
    ///
    /// See also [`get_results`].
    ///
    /// [`query_pool.ty().result_len()`]: QueryPool::result_len
    /// [`get_results`]: QueryPool::get_results
    pub fn copy_query_pool_results<T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> Result<&mut Self, Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        self.validate_copy_query_pool_results(&query_pool, queries.clone(), &destination, flags)?;

        Ok(unsafe {
            self.copy_query_pool_results_unchecked(query_pool, queries, destination, flags)
        })
    }

    fn validate_copy_query_pool_results<T>(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
        dst: &Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> Result<(), Box<ValidationError>>
    where
        T: QueryResultElement,
    {
        let stride = query_pool.result_len(flags) * size_of::<T>() as DeviceSize;

        self.inner.validate_copy_query_pool_results::<T>(
            query_pool,
            queries.start,
            queries.end - queries.start,
            dst.buffer(),
            dst.offset(),
            stride,
            flags,
        )?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyQueryPoolResults-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_query_pool_results_unchecked<T>(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
        destination: Subbuffer<[T]>,
        flags: QueryResultFlags,
    ) -> &mut Self
    where
        T: QueryResultElement,
    {
        let stride = query_pool.result_len(flags) * size_of::<T>() as DeviceSize;

        self.add_command(
            "copy_query_pool_results",
            [(
                ResourceInCommand::Destination.into(),
                Resource::Buffer {
                    buffer: destination.as_bytes().clone(),
                    range: 0..destination.size(), // TODO:
                    memory_access: PipelineStageAccessFlags::Copy_TransferWrite,
                },
            )]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.copy_query_pool_results_unchecked::<T>(
                        &query_pool,
                        queries.start,
                        queries.end - queries.start,
                        destination.buffer(),
                        destination.offset(),
                        stride,
                        flags,
                    )
                };
            },
        );

        self
    }

    /// Resets a range of queries on a query pool.
    ///
    /// The affected queries will be marked as "unavailable" after this command runs, and will no
    /// longer return any results. They will be ready to have new results recorded for them.
    ///
    /// # Safety
    /// The queries in the specified range must not be active in another command buffer.
    // TODO: Do other command buffers actually matter here? Not sure on the Vulkan spec.
    pub unsafe fn reset_query_pool(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_reset_query_pool(&query_pool, queries.clone())?;

        Ok(unsafe { self.reset_query_pool_unchecked(query_pool, queries) })
    }

    fn validate_reset_query_pool(
        &self,
        query_pool: &QueryPool,
        queries: Range<u32>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_reset_query_pool(
            query_pool,
            queries.start,
            queries.end - queries.start,
        )?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdResetQueryPool-renderpass"],
                ..Default::default()
            }));
        }

        if self
            .builder_state
            .queries
            .values()
            .any(|state| *state.query_pool == *query_pool && queries.contains(&state.query))
        {
            return Err(Box::new(ValidationError {
                problem: "one of the `queries` in `query_pool` is currently active".into(),
                vuids: &["VUID-vkCmdResetQueryPool-None-02841"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_query_pool_unchecked(
        &mut self,
        query_pool: Arc<QueryPool>,
        queries: Range<u32>,
    ) -> &mut Self {
        self.add_command(
            "reset_query_pool",
            Default::default(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.reset_query_pool_unchecked(
                        &query_pool,
                        queries.start,
                        queries.end - queries.start,
                    )
                };
            },
        );

        self
    }
}
