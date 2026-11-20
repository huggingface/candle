use crate::{
    acceleration_structure::{
        AccelerationStructure, CopyAccelerationStructureInfo, CopyAccelerationStructureMode,
    },
    command_buffer::{
        auto::Resource, sys::RecordingCommandBuffer, AutoCommandBufferBuilder, ResourceInCommand,
    },
    query::QueryPool,
    sync::PipelineStageAccessFlags,
    ValidationError,
};
use smallvec::SmallVec;
use std::sync::Arc;

/// # Commands to do operations on acceleration structures.
impl<L> AutoCommandBufferBuilder<L> {
    /// Copies the data of one acceleration structure to another.
    ///
    /// # Safety
    ///
    /// - `info.src` must have been built when this command is executed.
    /// - If `info.mode` is [`CopyAccelerationStructureMode::Compact`], then `info.src` must have
    ///   been built with [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`].
    ///
    /// [`CopyAccelerationStructureMode::Compact`]: crate::acceleration_structure::CopyAccelerationStructureMode::Compact
    /// [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`]: crate::acceleration_structure::BuildAccelerationStructureFlags::ALLOW_COMPACTION
    #[inline]
    pub unsafe fn copy_acceleration_structure(
        &mut self,
        info: &CopyAccelerationStructureInfo<'_>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_acceleration_structure(info)?;

        Ok(unsafe { self.copy_acceleration_structure_unchecked(info) })
    }

    fn validate_copy_acceleration_structure(
        &self,
        info: &CopyAccelerationStructureInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_copy_acceleration_structure(info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureKHR-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_acceleration_structure_unchecked(
        &mut self,
        info: &CopyAccelerationStructureInfo<'_>,
    ) -> &mut Self {
        struct OwnedCopyAccelerationStructureInfo {
            src: Arc<AccelerationStructure>,
            dst: Arc<AccelerationStructure>,
            mode: CopyAccelerationStructureMode,
        }

        let &CopyAccelerationStructureInfo {
            src,
            dst,
            mode,
            _ne: _,
        } = info;

        let src_buffer = src.buffer();
        let dst_buffer = dst.buffer();
        let info = OwnedCopyAccelerationStructureInfo {
            src: src.clone(),
            dst: dst.clone(),
            mode,
        };

        self.add_command(
            "copy_acceleration_structure",
            [
                (
                    ResourceInCommand::Source.into(),
                    Resource::Buffer {
                        buffer: src_buffer.clone().into(),
                        range: src.offset()..src.offset() + src.size(),
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureRead,
                    },
                ),
                (
                    ResourceInCommand::Destination.into(),
                    Resource::Buffer {
                        buffer: dst_buffer.clone().into(),
                        range: dst.offset()..dst.offset() + dst.size(),
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureWrite,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                let OwnedCopyAccelerationStructureInfo {
                    ref src,
                    ref dst,
                    mode,
                } = info;
                let info = CopyAccelerationStructureInfo {
                    src,
                    dst,
                    mode,
                    _ne: crate::NE,
                };
                unsafe { out.copy_acceleration_structure_unchecked(&info) };
            },
        );

        self
    }

    /// Writes the properties of one or more acceleration structures to a query.
    ///
    /// For each element in `acceleration_structures`, one query is written, in numeric order
    /// starting at `first_query`.
    ///
    /// # Safety
    ///
    /// - All elements of `acceleration_structures` must have been built when this command is
    ///   executed.
    /// - If `query_pool.query_type()` is [`QueryType::AccelerationStructureCompactedSize`], all
    ///   elements of `acceleration_structures` must have been built with
    ///   [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`].
    /// - The queries must be unavailable, ensured by calling [`reset_query_pool`].
    ///
    /// [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`]: crate::acceleration_structure::BuildAccelerationStructureFlags::ALLOW_COMPACTION
    /// [`reset_query_pool`]: Self::reset_query_pool
    #[inline]
    pub unsafe fn write_acceleration_structures_properties(
        &mut self,
        acceleration_structures: SmallVec<[Arc<AccelerationStructure>; 4]>,
        query_pool: Arc<QueryPool>,
        first_query: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_write_acceleration_structures_properties(
            &acceleration_structures,
            &query_pool,
            first_query,
        )?;

        Ok(unsafe {
            self.write_acceleration_structures_properties_unchecked(
                acceleration_structures,
                query_pool,
                first_query,
            )
        })
    }

    fn validate_write_acceleration_structures_properties(
        &self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_write_acceleration_structures_properties(
                acceleration_structures,
                query_pool,
                first_query,
            )?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_acceleration_structures_properties_unchecked(
        &mut self,
        acceleration_structures: SmallVec<[Arc<AccelerationStructure>; 4]>,
        query_pool: Arc<QueryPool>,
        first_query: u32,
    ) -> &mut Self {
        if acceleration_structures.is_empty() {
            return self;
        }

        self.add_command(
            "write_acceleration_structures_properties",
            acceleration_structures.iter().enumerate().map(|(index, acs)| {
                let index = index as u32;
                let buffer = acs.buffer();
                let offset = acs.offset();
                let size = acs.size();

                (
                    ResourceInCommand::AccelerationStructure { index }.into(),
                    Resource::Buffer {
                        buffer: buffer.clone().into(),
                        range: offset..offset + size,
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureRead,
                    },
                )
            })
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.write_acceleration_structures_properties_unchecked(
                        &acceleration_structures,
                        &query_pool,
                        first_query,
                    )
                };
            },
        );

        self
    }
}
