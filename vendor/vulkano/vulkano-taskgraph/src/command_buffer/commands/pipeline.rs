use crate::{
    command_buffer::{RecordingCommandBuffer, Result},
    Id,
};
#[cfg(doc)]
use vulkano::command_buffer::{
    DispatchIndirectCommand, DrawIndexedIndirectCommand, DrawIndirectCommand,
    DrawMeshTasksIndirectCommand,
};
use vulkano::{
    buffer::Buffer, device::DeviceOwned, pipeline::ray_tracing::ShaderBindingTableAddresses,
    DeviceAddress, DeviceSize, Version, VulkanObject,
};

/// # Commands to execute a bound pipeline
///
/// Dispatch commands require a compute queue, draw commands require a graphics queue.
impl RecordingCommandBuffer<'_> {
    /// Performs a single compute operation using a compute pipeline, panicking on a validation
    /// error.
    ///
    /// A compute pipeline must have been bound using [`bind_pipeline_compute`]. Any resources used
    /// by the compute pipeline, such as descriptor sets, must have been set beforehand.
    ///
    /// This is a shortcut for `try_dispatch().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_dispatch`] returns a [`ValidationError`].
    ///
    /// [`bind_pipeline_compute`]: Self::bind_pipeline_compute
    /// [shader safety requirements]: vulkano::shader#safety
    /// [`try_dispatch`]: Self::try_dispatch
    #[track_caller]
    pub unsafe fn dispatch(&mut self, group_counts: [u32; 3]) -> &mut Self {
        unsafe { self.try_dispatch(group_counts) }.unwrap()
    }

    /// Performs a single compute operation using a compute pipeline.
    ///
    /// A compute pipeline must have been bound using [`bind_pipeline_compute`]. Any resources used
    /// by the compute pipeline, such as descriptor sets, must have been set beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// [`bind_pipeline_compute`]: Self::bind_pipeline_compute
    /// [shader safety requirements]: vulkano::shader#safety
    pub unsafe fn try_dispatch(&mut self, group_counts: [u32; 3]) -> Result<&mut Self> {
        Ok(unsafe { self.dispatch_unchecked(group_counts) })
    }

    pub unsafe fn dispatch_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_dispatch)(
                self.handle(),
                group_counts[0],
                group_counts[1],
                group_counts[2],
            )
        };

        self
    }

    /// Performs a single compute operation using a compute pipeline, panicking on a validation
    /// error. One dispatch is performed for the [`DispatchIndirectCommand`] struct that is read
    /// from `buffer` starting at `offset`.
    ///
    /// A compute pipeline must have been bound using [`bind_pipeline_compute`]. Any resources used
    /// by the compute pipeline, such as descriptor sets, must have been set beforehand.
    ///
    /// This is a shortcut for `try_dispatch_indirect().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DispatchIndirectCommand`] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_dispatch_indirect`] returns a [`ValidationError`].
    ///
    /// [`bind_pipeline_compute`]: Self::bind_pipeline_compute
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DispatchIndirectCommand`]: DispatchIndirectCommand#safety
    /// [`try_dispatch_indirect`]: Self::try_dispatch_indirect
    #[track_caller]
    pub unsafe fn dispatch_indirect(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
    ) -> &mut Self {
        unsafe { self.try_dispatch_indirect(buffer, offset) }.unwrap()
    }

    /// Performs a single compute operation using a compute pipeline. One dispatch is performed
    /// for the [`DispatchIndirectCommand`] struct that is read from `buffer` starting at `offset`.
    ///
    /// A compute pipeline must have been bound using [`bind_pipeline_compute`]. Any resources used
    /// by the compute pipeline, such as descriptor sets, must have been set beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DispatchIndirectCommand`] apply.
    ///
    /// [`bind_pipeline_compute`]: Self::bind_pipeline_compute
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DispatchIndirectCommand`]: DispatchIndirectCommand#safety
    pub unsafe fn try_dispatch_indirect(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.dispatch_indirect_unchecked(buffer, offset) })
    }

    pub unsafe fn dispatch_indirect_unchecked(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
    ) -> &mut Self {
        let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };

        let fns = self.device().fns();
        unsafe { (fns.v1_0.cmd_dispatch_indirect)(self.handle(), buffer.handle(), offset) };

        self
    }

    /// Performs a single draw operation using a primitive shading graphics pipeline, panicking on
    /// a validation error.
    ///
    /// The parameters specify the first vertex and the number of vertices to draw, and the first
    /// instance and number of instances. For non-instanced drawing, specify `instance_count` as 1
    /// and `first_instance` as 0.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the provided vertex and instance ranges must be
    /// in range of the bound vertex buffers.
    ///
    /// This is a shortcut for `try_draw().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw`] returns a [`ValidationError`].
    ///
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [`try_draw`]: Self::try_draw
    #[track_caller]
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        unsafe { self.try_draw(vertex_count, instance_count, first_vertex, first_instance) }
            .unwrap()
    }

    /// Performs a single draw operation using a primitive shading graphics pipeline.
    ///
    /// The parameters specify the first vertex and the number of vertices to draw, and the first
    /// instance and number of instances. For non-instanced drawing, specify `instance_count` as 1
    /// and `first_instance` as 0.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the provided vertex and instance ranges must be
    /// in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    pub unsafe fn try_draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance)
        })
    }

    pub unsafe fn draw_unchecked(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_draw)(
                self.handle(),
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        };

        self
    }

    /// Performs multiple draw operations using a primitive shading graphics pipeline, panicking on
    /// a validation error.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct that is read from `buffer`
    /// starting at `offset`, with the offset increasing by `stride` bytes for each successive
    /// draw. `draw_count` draw commands are performed. The maximum number of draw commands in the
    /// buffer is limited by the [`max_draw_indirect_count`] limit. This limit is 1 unless the
    /// [`multi_draw_indirect`] feature has been enabled.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the vertex and instance ranges of each
    /// `DrawIndirectCommand` in the indirect buffer must be in range of the bound vertex buffers.
    ///
    /// This is a shortcut for `try_draw_indirect().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawIndirectCommand`] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw_indirect`] returns a [`ValidationError`].
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`multi_draw_indirect`]: vulkano::device::DeviceFeatures::multi_draw_indirect
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawIndirectCommand`]: DrawIndirectCommand#safety
    /// [`try_draw_indirect`]: Self::try_draw_indirect
    #[track_caller]
    pub unsafe fn draw_indirect(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe { self.try_draw_indirect(buffer, offset, draw_count, stride) }.unwrap()
    }

    /// Performs multiple draw operations using a primitive shading graphics pipeline.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct that is read from `buffer`
    /// starting at `offset`, with the offset increasing by `stride` bytes for each successive
    /// draw. `draw_count` draw commands are performed. The maximum number of draw commands in the
    /// buffer is limited by the [`max_draw_indirect_count`] limit. This limit is 1 unless the
    /// [`multi_draw_indirect`] feature has been enabled.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the vertex and instance ranges of each
    /// `DrawIndirectCommand` in the indirect buffer must be in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawIndirectCommand`] apply.
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`multi_draw_indirect`]: vulkano::device::DeviceFeatures::multi_draw_indirect
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawIndirectCommand`]: DrawIndirectCommand#safety
    pub unsafe fn try_draw_indirect(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.draw_indirect_unchecked(buffer, offset, draw_count, stride) })
    }

    pub unsafe fn draw_indirect_unchecked(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_draw_indirect)(self.handle(), buffer.handle(), offset, draw_count, stride)
        };

        self
    }

    /// Performs multiple draw operations using a primitive shading graphics pipeline, reading the
    /// number of draw operations from a separate buffer, panicking on a validation error.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct that is read from `buffer`
    /// starting at `offset`, with the offset increasing by `stride` bytes for each successive
    /// draw. The number of draws to perform is read from `count_buffer` at `count_buffer_offset`,
    /// or specified by `max_draw_count`, whichever is lower. This number is limited by the
    /// [`max_draw_indirect_count`] limit.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the vertex and instance ranges of each
    /// `DrawIndirectCommand` in the indirect buffer must be in range of the bound vertex buffers.
    ///
    /// This is a shortcut for `try_draw_indirect_count().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawIndirectCommand`] apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`] device limit.
    /// - The count stored in `count_buffer` must fall within the range of `buffer` starting at
    ///   `offset`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw_indirect_count`] returns a [`ValidationError`].
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawIndirectCommand`]: DrawIndirectCommand#safety
    /// [`try_draw_indirect_count`]: Self::try_draw_indirect_count
    #[track_caller]
    pub unsafe fn draw_indirect_count(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe {
            self.try_draw_indirect_count(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        }
        .unwrap()
    }

    /// Performs multiple draw operations using a primitive shading graphics pipeline, reading the
    /// number of draw operations from a separate buffer.
    ///
    /// One draw is performed for each [`DrawIndirectCommand`] struct that is read from `buffer`
    /// starting at `offset`, with the offset increasing by `stride` bytes for each successive
    /// draw. The number of draws to perform is read from `count_buffer` at `count_buffer_offset`,
    /// or specified by `max_draw_count`, whichever is lower. This number is limited by the
    /// [`max_draw_indirect_count`] limit.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the vertex and instance ranges of each
    /// `DrawIndirectCommand` in the indirect buffer must be in range of the bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawIndirectCommand`] apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`] device limit.
    /// - The count stored in `count_buffer` must fall within the range of `buffer` starting at
    ///   `offset`.
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawIndirectCommand`]: DrawIndirectCommand#safety
    pub unsafe fn try_draw_indirect_count(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.draw_indirect_count_unchecked(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        })
    }

    pub unsafe fn draw_indirect_count_unchecked(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };
        let count_buffer = unsafe { self.accesses.buffer_unchecked(count_buffer) };

        let device = self.device();
        let fns = device.fns();
        let cmd_draw_indirect_count = if device.api_version() >= Version::V1_2 {
            fns.v1_2.cmd_draw_indirect_count
        } else if device.enabled_extensions().khr_draw_indirect_count {
            fns.khr_draw_indirect_count.cmd_draw_indirect_count_khr
        } else if device.enabled_extensions().amd_draw_indirect_count {
            fns.amd_draw_indirect_count.cmd_draw_indirect_count_amd
        } else {
            std::process::abort();
        };

        unsafe {
            cmd_draw_indirect_count(
                self.handle(),
                buffer.handle(),
                offset,
                count_buffer.handle(),
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };

        self
    }

    /// Performs a single draw operation using a primitive shading graphics pipeline, using an
    /// index buffer, panicking on a validation error.
    ///
    /// The parameters specify the first index and the number of indices in the index buffer that
    /// should be used, and the first instance and number of instances. For non-instanced drawing,
    /// specify `instance_count` as 1 and `first_instance` as 0. The `vertex_offset` is a constant
    /// value that should be added to each index in the index buffer to produce the final vertex
    /// number to be used.
    ///
    /// An index buffer must have been bound using [`bind_index_buffer`], and the provided index
    /// range must be in range of the bound index buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the provided instance range must be in range of
    /// the bound vertex buffers. The vertex indices in the index buffer must be in range of the
    /// bound vertex buffers.
    ///
    /// This is a shortcut for `try_draw_indexed().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - Every vertex number that is retrieved from the index buffer must fall within the range of
    ///   the bound vertex-rate vertex buffers.
    /// - Every vertex number that is retrieved from the index buffer, if it is not the special
    ///   primitive restart value, must be no greater than the [`max_draw_indexed_index_value`]
    ///   device limit.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw_indexed`] returns a [`ValidationError`].
    ///
    /// [`bind_index_buffer`]: Self::bind_index_buffer
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [`max_draw_indexed_index_value`]: vulkano::device::DeviceProperties::max_draw_indexed_index_value
    /// [`try_draw_indexed`]: Self::try_draw_indexed
    #[track_caller]
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        unsafe {
            self.try_draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
        .unwrap()
    }

    /// Performs a single draw operation using a primitive shading graphics pipeline, using an
    /// index buffer.
    ///
    /// The parameters specify the first index and the number of indices in the index buffer that
    /// should be used, and the first instance and number of instances. For non-instanced drawing,
    /// specify `instance_count` as 1 and `first_instance` as 0. The `vertex_offset` is a constant
    /// value that should be added to each index in the index buffer to produce the final vertex
    /// number to be used.
    ///
    /// An index buffer must have been bound using [`bind_index_buffer`], and the provided index
    /// range must be in range of the bound index buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the provided instance range must be in range of
    /// the bound vertex buffers. The vertex indices in the index buffer must be in range of the
    /// bound vertex buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - Every vertex number that is retrieved from the index buffer must fall within the range of
    ///   the bound vertex-rate vertex buffers.
    /// - Every vertex number that is retrieved from the index buffer, if it is not the special
    ///   primitive restart value, must be no greater than the [`max_draw_indexed_index_value`]
    ///   device limit.
    ///
    /// [`bind_index_buffer`]: Self::bind_index_buffer
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [`max_draw_indexed_index_value`]: vulkano::device::DeviceProperties::max_draw_indexed_index_value
    pub unsafe fn try_draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.draw_indexed_unchecked(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        })
    }

    pub unsafe fn draw_indexed_unchecked(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_draw_indexed)(
                self.handle(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };

        self
    }

    /// Performs multiple draw operations using a primitive shading graphics pipeline, using an
    /// index buffer, panicking on a validation error.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct that is read from
    /// `buffer` starting at `offset`, with the offset increasing by `stride` bytes with each
    /// successive draw. `draw_count` draw commands are performed. The maximum number of draw
    /// commands in the buffer is limited by the [`max_draw_indirect_count`] limit. This limit is 1
    /// unless the [`multi_draw_indirect`] feature has been enabled.
    ///
    /// An index buffer must have been bound using [`bind_index_buffer`], and the index ranges of
    /// each `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound
    /// index buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the instance ranges of each
    /// `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound vertex
    /// buffers.
    ///
    /// This is a shortcut for `try_draw_indexed_indirect().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawIndexedIndirectCommand`] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw_indexed_indirect`] returns a [`ValidationError`].
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`multi_draw_indirect`]: vulkano::device::DeviceFeatures::multi_draw_indirect
    /// [`bind_index_buffer`]: Self::bind_index_buffer
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawIndexedIndirectCommand`]: DrawIndexedIndirectCommand#safety
    /// [`try_draw_indexed_indirect`]: Self::try_draw_indexed_indirect
    #[track_caller]
    pub unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe { self.try_draw_indexed_indirect(buffer, offset, draw_count, stride) }.unwrap()
    }

    /// Performs multiple draw operations using a primitive shading graphics pipeline, using an
    /// index buffer.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct that is read from
    /// `buffer` starting at `offset`, with the offset increasing by `stride` bytes with each
    /// successive draw. `draw_count` draw commands are performed. The maximum number of draw
    /// commands in the buffer is limited by the [`max_draw_indirect_count`] limit. This limit is 1
    /// unless the [`multi_draw_indirect`] feature has been enabled.
    ///
    /// An index buffer must have been bound using [`bind_index_buffer`], and the index ranges of
    /// each `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound
    /// index buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the instance ranges of each
    /// `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound vertex
    /// buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawIndexedIndirectCommand`] apply.
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`multi_draw_indirect`]: vulkano::device::DeviceFeatures::multi_draw_indirect
    /// [`bind_index_buffer`]: Self::bind_index_buffer
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawIndexedIndirectCommand`]: DrawIndexedIndirectCommand#safety
    pub unsafe fn try_draw_indexed_indirect(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.draw_indexed_indirect_unchecked(buffer, offset, draw_count, stride) })
    }

    pub unsafe fn draw_indexed_indirect_unchecked(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };

        let fns = self.device().fns();
        unsafe {
            (fns.v1_0.cmd_draw_indexed_indirect)(
                self.handle(),
                buffer.handle(),
                offset,
                draw_count,
                stride,
            )
        };

        self
    }

    /// Performs multiple draw operations using a primitive shading graphics pipeline, using an
    /// index buffer, and reading the number of draw operations from a separate buffer, panicking
    /// on a validation error.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct that is read from
    /// `buffer` starting at `offset`, with the offset increasing by `stride` bytes for each
    /// successive draw. The number of draws to perform is read from `count_buffer` at
    /// `count_buffer_offset`, or specified by `max_draw_count`, whichever is lower. This number is
    /// limited by the [`max_draw_indirect_count`] limit.
    ///
    /// An index buffer must have been bound using [`bind_index_buffer`], and the index ranges of
    /// each `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound
    /// index buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the instance ranges of each
    /// `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound vertex
    /// buffers.
    ///
    /// This is a shortcut for `try_draw_indexed_indirect_count().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawIndexedIndirectCommand`] apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`] device limit.
    /// - The count stored in `count_buffer` must fall within the range of `buffer`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw_indexed_indirect_count`] returns a [`ValidationError`].
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`bind_index_buffer`]: Self::bind_index_buffer
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawIndexedIndirectCommand`]: DrawIndexedIndirectCommand#safety
    /// [`try_draw_indexed_indirect_count`]: Self::try_draw_indexed_indirect_count
    #[track_caller]
    pub unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe {
            self.try_draw_indexed_indirect_count(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        }
        .unwrap()
    }

    /// Performs multiple draw operations using a primitive shading graphics pipeline, using an
    /// index buffer, and reading the number of draw operations from a separate buffer.
    ///
    /// One draw is performed for each [`DrawIndexedIndirectCommand`] struct that is read from
    /// `buffer` starting at `offset`, with the offset increasing by `stride` bytes for each
    /// successive draw. The number of draws to perform is read from `count_buffer` at
    /// `count_buffer_offset`, or specified by `max_draw_count`, whichever is lower. This number is
    /// limited by the [`max_draw_indirect_count`] limit.
    ///
    /// An index buffer must have been bound using [`bind_index_buffer`], and the index ranges of
    /// each `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound
    /// index buffer.
    ///
    /// A primitive shading graphics pipeline must have been bound using
    /// [`bind_pipeline_graphics`]. Any resources used by the graphics pipeline, such as descriptor
    /// sets, vertex buffers and dynamic state, must have been set beforehand. If the bound
    /// graphics pipeline uses vertex buffers, then the instance ranges of each
    /// `DrawIndexedIndirectCommand` in the indirect buffer must be in range of the bound vertex
    /// buffers.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawIndexedIndirectCommand`] apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`] device limit.
    /// - The count stored in `count_buffer` must fall within the range of `buffer`.
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`bind_index_buffer`]: Self::bind_index_buffer
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawIndexedIndirectCommand`]: DrawIndexedIndirectCommand#safety
    pub unsafe fn try_draw_indexed_indirect_count(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.draw_indexed_indirect_count_unchecked(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        })
    }

    pub unsafe fn draw_indexed_indirect_count_unchecked(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };
        let count_buffer = unsafe { self.accesses.buffer_unchecked(count_buffer) };

        let device = self.device();
        let fns = device.fns();
        let cmd_draw_indexed_indirect_count = if device.api_version() >= Version::V1_2 {
            fns.v1_2.cmd_draw_indexed_indirect_count
        } else if device.enabled_extensions().khr_draw_indirect_count {
            fns.khr_draw_indirect_count
                .cmd_draw_indexed_indirect_count_khr
        } else if device.enabled_extensions().amd_draw_indirect_count {
            fns.amd_draw_indirect_count
                .cmd_draw_indexed_indirect_count_amd
        } else {
            std::process::abort();
        };

        unsafe {
            cmd_draw_indexed_indirect_count(
                self.handle(),
                buffer.handle(),
                offset,
                count_buffer.handle(),
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };

        self
    }

    /// Perform a single draw operation using a mesh shading graphics pipeline, panicking on a
    /// validation error.
    ///
    /// A mesh shading graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any
    /// resources used by the graphics pipeline, such as descriptor sets and dynamic state, must
    /// have been set beforehand.
    ///
    /// This is a shortcut for `try_draw_mesh_tasks().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw_mesh_tasks`] returns a [`ValidationError`].
    ///
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [`try_draw_mesh_tasks`]: Self::try_draw_mesh_tasks
    #[track_caller]
    pub unsafe fn draw_mesh_tasks(&mut self, group_counts: [u32; 3]) -> &mut Self {
        unsafe { self.try_draw_mesh_tasks(group_counts) }.unwrap()
    }

    /// Perform a single draw operation using a mesh shading graphics pipeline.
    ///
    /// A mesh shading graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any
    /// resources used by the graphics pipeline, such as descriptor sets and dynamic state, must
    /// have been set beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    pub unsafe fn try_draw_mesh_tasks(&mut self, group_counts: [u32; 3]) -> Result<&mut Self> {
        Ok(unsafe { self.draw_mesh_tasks_unchecked(group_counts) })
    }

    pub unsafe fn draw_mesh_tasks_unchecked(&mut self, group_counts: [u32; 3]) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.ext_mesh_shader.cmd_draw_mesh_tasks_ext)(
                self.handle(),
                group_counts[0],
                group_counts[1],
                group_counts[2],
            )
        };

        self
    }

    /// Perform multiple draw operations using a mesh shading graphics pipeline, panicking on a
    /// validation error.
    ///
    /// One draw is performed for each [`DrawMeshTasksIndirectCommand`] struct that is read from
    /// `buffer` starting at `offset`, with the offset increasing by `stride` bytes for each
    /// successive draw. `draw_count` draw commands are performed. The maximum number of draw
    /// commands in the buffer is limited by the [`max_draw_indirect_count`] limit. This limit is 1
    /// unless the [`multi_draw_indirect`] feature has been enabled.
    ///
    /// A mesh shading graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any
    /// resources used by the graphics pipeline, such as descriptor sets and dynamic state, must
    /// have been set beforehand.
    ///
    /// This is a shortcut for `try_draw_mesh_tasks_indirect().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawMeshTasksIndirectCommand`] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw_mesh_tasks_indirect`] returns a [`ValidationError`].
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`multi_draw_indirect`]: vulkano::device::DeviceFeatures::multi_draw_indirect
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawMeshTasksIndirectCommand`]: DrawMeshTasksIndirectCommand#safety
    /// [`try_draw_mesh_tasks_indirect`]: Self::try_draw_mesh_tasks_indirect
    #[track_caller]
    pub unsafe fn draw_mesh_tasks_indirect(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe { self.try_draw_mesh_tasks_indirect(buffer, offset, draw_count, stride) }.unwrap()
    }

    /// Perform multiple draw operations using a mesh shading graphics pipeline.
    ///
    /// One draw is performed for each [`DrawMeshTasksIndirectCommand`] struct that is read from
    /// `buffer` starting at `offset`, with the offset increasing by `stride` bytes for each
    /// successive draw. `draw_count` draw commands are performed. The maximum number of draw
    /// commands in the buffer is limited by the [`max_draw_indirect_count`] limit. This limit is 1
    /// unless the [`multi_draw_indirect`] feature has been enabled.
    ///
    /// A mesh shading graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any
    /// resources used by the graphics pipeline, such as descriptor sets and dynamic state, must
    /// have been set beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawMeshTasksIndirectCommand`] apply.
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`multi_draw_indirect`]: vulkano::device::DeviceFeatures::multi_draw_indirect
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawMeshTasksIndirectCommand`]: DrawMeshTasksIndirectCommand#safety
    pub unsafe fn try_draw_mesh_tasks_indirect(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.draw_mesh_tasks_indirect_unchecked(buffer, offset, draw_count, stride) })
    }

    pub unsafe fn draw_mesh_tasks_indirect_unchecked(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };

        let fns = self.device().fns();
        unsafe {
            (fns.ext_mesh_shader.cmd_draw_mesh_tasks_indirect_ext)(
                self.handle(),
                buffer.handle(),
                offset,
                draw_count,
                stride,
            )
        };

        self
    }

    /// Performs multiple draw operations using a mesh shading graphics pipeline, reading the
    /// number of draw operations from a separate buffer, panicking on a validation error.
    ///
    /// One draw is performed for each [`DrawMeshTasksIndirectCommand`] struct that is read from
    /// `buffer` starting at `offset`, with the offset increasing by `stride` bytes after each
    /// successive draw. The number of draws to perform is read from `count_buffer`, or specified
    /// by `max_draw_count`, whichever is lower. This number is limited by the
    /// [`max_draw_indirect_count`] limit.
    ///
    /// A mesh shading graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any
    /// resources used by the graphics pipeline, such as descriptor sets and dynamic state, must
    /// have been set beforehand.
    ///
    /// This is a shortcut for `try_draw_mesh_tasks_indirect_count().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawMeshTasksIndirectCommand`] apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`] device limit.
    /// - The count stored in `count_buffer` must fall within the range of `buffer`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_draw_mesh_tasks_indirect_count`] returns a [`ValidationError`].
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawMeshTasksIndirectCommand`]: DrawMeshTasksIndirectCommand#safety
    /// [`try_draw_mesh_tasks_indirect_count`]: Self::try_draw_mesh_tasks_indirect_count
    #[track_caller]
    pub unsafe fn draw_mesh_tasks_indirect_count(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        unsafe {
            self.try_draw_mesh_tasks_indirect_count(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        }
        .unwrap()
    }

    /// Performs multiple draw operations using a mesh shading graphics pipeline, reading the
    /// number of draw operations from a separate buffer.
    ///
    /// One draw is performed for each [`DrawMeshTasksIndirectCommand`] struct that is read from
    /// `buffer` starting at `offset`, with the offset increasing by `stride` bytes after each
    /// successive draw. The number of draws to perform is read from `count_buffer`, or specified
    /// by `max_draw_count`, whichever is lower. This number is limited by the
    /// [`max_draw_indirect_count`] limit.
    ///
    /// A mesh shading graphics pipeline must have been bound using [`bind_pipeline_graphics`]. Any
    /// resources used by the graphics pipeline, such as descriptor sets and dynamic state, must
    /// have been set beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `DrawMeshTasksIndirectCommand`] apply.
    /// - The count stored in `count_buffer` must not be greater than the
    ///   [`max_draw_indirect_count`] device limit.
    /// - The count stored in `count_buffer` must fall within the range of `buffer`.
    ///
    /// [`max_draw_indirect_count`]: vulkano::device::DeviceProperties::max_draw_indirect_count
    /// [`bind_pipeline_graphics`]: Self::bind_pipeline_graphics
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `DrawMeshTasksIndirectCommand`]: DrawMeshTasksIndirectCommand#safety
    pub unsafe fn try_draw_mesh_tasks_indirect_count(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.draw_mesh_tasks_indirect_count_unchecked(
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        })
    }

    pub unsafe fn draw_mesh_tasks_indirect_count_unchecked(
        &mut self,
        buffer: Id<Buffer>,
        offset: DeviceSize,
        count_buffer: Id<Buffer>,
        count_buffer_offset: DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> &mut Self {
        let buffer = unsafe { self.accesses.buffer_unchecked(buffer) };
        let count_buffer = unsafe { self.accesses.buffer_unchecked(count_buffer) };

        let fns = self.device().fns();
        unsafe {
            (fns.ext_mesh_shader.cmd_draw_mesh_tasks_indirect_count_ext)(
                self.handle(),
                buffer.handle(),
                offset,
                count_buffer.handle(),
                count_buffer_offset,
                max_draw_count,
                stride,
            )
        };

        self
    }

    /// Performs a single ray tracing operation using a ray tracing pipeline, panicking on a
    /// validation error.
    ///
    /// A ray tracing pipeline must have been bound using [`bind_pipeline_ray_tracing`]. Any
    /// resources used by the ray tracing pipeline, such as descriptor sets, must have been set
    /// beforehand.
    ///
    /// This is a shortcut for `try_trace_rays().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_trace_rays`] returns a [`ValidationError`].
    ///
    /// [`bind_pipeline_ray_tracing`]: Self::bind_pipeline_ray_tracing
    /// [shader safety requirements]: vulkano::shader#safety
    /// [`try_trace_rays`]: Self::try_trace_rays
    #[track_caller]
    pub unsafe fn trace_rays(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> &mut Self {
        unsafe { self.try_trace_rays(shader_binding_table_addresses, dimensions) }.unwrap()
    }

    /// Performs a single ray tracing operation using a ray tracing pipeline.
    ///
    /// A ray tracing pipeline must have been bound using [`bind_pipeline_ray_tracing`]. Any
    /// resources used by the ray tracing pipeline, such as descriptor sets, must have been set
    /// beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// [`bind_pipeline_ray_tracing`]: Self::bind_pipeline_ray_tracing
    /// [shader safety requirements]: vulkano::shader#safety
    pub unsafe fn try_trace_rays(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> Result<&mut Self> {
        Ok(unsafe { self.trace_rays_unchecked(shader_binding_table_addresses, dimensions) })
    }

    pub unsafe fn trace_rays_unchecked(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        dimensions: [u32; 3],
    ) -> &mut Self {
        let raygen = shader_binding_table_addresses.raygen.to_vk();
        let miss = shader_binding_table_addresses.miss.to_vk();
        let hit = shader_binding_table_addresses.hit.to_vk();
        let callable = shader_binding_table_addresses.callable.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_ray_tracing_pipeline.cmd_trace_rays_khr)(
                self.handle(),
                &raygen,
                &miss,
                &hit,
                &callable,
                dimensions[0],
                dimensions[1],
                dimensions[2],
            )
        };

        self
    }

    /// Performs a single ray tracing operation using a ray tracing pipeline, panicking on a  
    /// validation error. One ray tracing operation is performed for the  
    /// [`TraceRaysIndirectCommand`] struct that is read from `indirect_device_address`.
    ///
    /// A ray tracing pipeline must have been bound using [`bind_pipeline_ray_tracing`]. Any
    /// resources used by the ray tracing pipeline, such as descriptor sets, must have been set
    /// beforehand.
    ///
    /// This is a shortcut for `try_trace_rays_indirect().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `TraceRaysIndirectCommand`] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_trace_rays_indirect`] returns a [`ValidationError`].
    ///
    /// [`bind_pipeline_ray_tracing`]: Self::bind_pipeline_ray_tracing
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `TraceRaysIndirectCommand`]: TraceRaysIndirectCommand#safety
    /// [`try_trace_rays_indirect`]: Self::try_trace_rays_indirect
    #[track_caller]
    pub unsafe fn trace_rays_indirect(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        indirect_device_address: DeviceAddress,
    ) -> &mut Self {
        unsafe {
            self.try_trace_rays_indirect(shader_binding_table_addresses, indirect_device_address)
        }
        .unwrap()
    }

    /// Performs a single ray tracing operation using a ray tracing pipeline. One ray tracing  
    /// operation is performed for the [`TraceRaysIndirectCommand`] struct that is read from  
    /// `indirect_device_address`.
    ///
    /// A ray tracing pipeline must have been bound using [`bind_pipeline_ray_tracing`]. Any
    /// resources used by the ray tracing pipeline, such as descriptor sets, must have been set
    /// beforehand.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    /// - The [safety requirements for `TraceRaysIndirectCommand`] apply.
    ///
    /// [`bind_pipeline_ray_tracing`]: Self::bind_pipeline_ray_tracing
    /// [shader safety requirements]: vulkano::shader#safety
    /// [safety requirements for `TraceRaysIndirectCommand`]: TraceRaysIndirectCommand#safety
    pub unsafe fn try_trace_rays_indirect(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        indirect_device_address: DeviceAddress,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.trace_rays_indirect_unchecked(
                shader_binding_table_addresses,
                indirect_device_address,
            )
        })
    }

    pub unsafe fn trace_rays_indirect_unchecked(
        &mut self,
        shader_binding_table_addresses: &ShaderBindingTableAddresses,
        indirect_device_address: DeviceAddress,
    ) -> &mut Self {
        let raygen = shader_binding_table_addresses.raygen.to_vk();
        let miss = shader_binding_table_addresses.miss.to_vk();
        let hit = shader_binding_table_addresses.hit.to_vk();
        let callable = shader_binding_table_addresses.callable.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_ray_tracing_pipeline.cmd_trace_rays_indirect_khr)(
                self.handle(),
                &raygen,
                &miss,
                &hit,
                &callable,
                indirect_device_address,
            )
        };

        self
    }
}
