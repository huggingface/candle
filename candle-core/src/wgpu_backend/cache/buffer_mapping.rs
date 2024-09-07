use crate::wgpu_backend::{device::PipelineType, util::FixedSizeQueue};

use super::CachedBufferId;


///Cache, that stores previously Flushed Gpu Commands, we try to use the same buffers as the last time
#[derive(Debug)]
pub(crate) struct BufferMappingCache {
    pub(crate) last_buffer_mappings: FixedSizeQueue<CachedBufferMappings>,
    pub(crate) current_buffer_mapping: Option<CachedBufferMappings>,
    pub(crate) current_index: u32,
}

impl BufferMappingCache {
    pub (crate) fn new(size: u32) -> Self {
        Self {
            last_buffer_mappings: FixedSizeQueue::new(size as usize),
            current_buffer_mapping: None,
            current_index: 0,
        }
    }

    pub(crate) fn set_current_buffer_mapping(&mut self, hash: u64) {
        let index = self
            .last_buffer_mappings
            .iter()
            .position(|b| b.hash == hash);
        if let Some(index) = index {
            log::debug!(
                "reuse mapping: {index}, hash: {hash}, mappings: {}",
                self.last_buffer_mappings.deque.len()
            );
            self.current_buffer_mapping = self.last_buffer_mappings.deque.remove(index);
        } else {
            log::debug!(
                "create new mapping: hash: {hash}, mappings: {}",
                self.last_buffer_mappings.deque.len()
            );
            self.current_buffer_mapping = Some(CachedBufferMappings::new(hash));
        }
    }

    pub(crate) fn get_buffer(&mut self, pipeline: PipelineType) -> Option<CachedBufferId> {
        if let Some(mapping) = &self.current_buffer_mapping {
            if let Some(mapping) = &mapping.data.get(self.current_index as usize) {
                if mapping.pipeline == pipeline {
                    return Some(mapping.used_buffer.clone());
                }
            }
        } else {
            panic!("expected current buffer to be set");
        }
        return None;
    }

    ///Stores, that at the provided buffer was used
    pub(crate) fn add_buffer(&mut self, buffer: CachedBufferId, pipeline: PipelineType) {
        if let Some(mapping) = &mut self.current_buffer_mapping {
            let data = CachedBufferMapping::new(pipeline, buffer);
            if (self.current_index as usize) < mapping.data.len() {
                mapping.data[self.current_index as usize] = data;
            } else {
                mapping.data.push(data);
            }
        } else {
            panic!("expected current buffer to be set");
        }

        self.current_index += 1;
    }

    pub(crate) fn finish(&mut self) {
        if let Some(value) = self.current_buffer_mapping.take() {
            self.last_buffer_mappings.push(value);
        }
        self.current_index = 0;
    }
}

#[derive(Debug)]
pub(crate) struct CachedBufferMapping {
    pub(crate) pipeline: PipelineType,
    pub(crate) used_buffer: CachedBufferId,
}

impl CachedBufferMapping {
    fn new(pipeline: PipelineType, used_buffer: CachedBufferId) -> Self {
        Self {
            pipeline,
            used_buffer,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CachedBufferMappings {
    pub(crate) data: Vec<CachedBufferMapping>, //all shader calls, and there used BufferCache
    pub(crate) hash: u64,
}

impl CachedBufferMappings {
    fn new(hash: u64) -> Self {
        Self { data: vec![], hash }
    }
}