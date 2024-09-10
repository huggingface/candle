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

    pub(crate) fn set_current_buffer_mapping(&mut self, hash: u64){
        let index = self
            .last_buffer_mappings
            .iter()
            .position(|b| b.hash == hash);
        if let Some(index) = index {
            log::debug!(
                "reuse mapping: {index}, hash: {hash}, mappings: {}",
                self.last_buffer_mappings.deque.len()
            );
            let mut buffer_mapping = self.last_buffer_mappings.deque.remove(index).unwrap();
            buffer_mapping.count += 1;

            self.current_buffer_mapping = Some(buffer_mapping);
        } else {
            log::debug!(
                "create new mapping: hash: {hash}, mappings: {}",
                self.last_buffer_mappings.deque.len()
            );
            self.current_buffer_mapping = Some(CachedBufferMappings::new(hash));
        }
    }

    pub (crate) fn get_current_mapping_count(&self) -> u32{
        if let Some(mapping) = &self.current_buffer_mapping {
            return mapping.count;
        }
        return 0;
    }

    pub (crate) fn get_current_mapping(&mut self) -> &mut CachedBufferMappings{
        return self.current_buffer_mapping.as_mut().unwrap();
    }

 
    ///Stores, that at the provided buffer was used
    pub(crate) fn add_new_buffer(&mut self, buffer: CachedBufferId, pipeline: PipelineType, last_size : u64) {
        if let Some(mapping) = &mut self.current_buffer_mapping {
            mapping.add_new(buffer, pipeline, last_size, self.current_index);
        } else {
            panic!("expected current buffer to be set");
        }
      
        self.current_index += 1;
    }

    pub(crate) fn reuse_buffer(&mut self, last_size : u64) {
        if let Some(mapping) = &mut self.current_buffer_mapping {
            mapping.reuse_buffer(last_size, self.current_index);
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
    pub(crate) used_buffer: CachedBufferId, //index in cachedBUfferMappings
    pub(crate) last_size : u64, //size of the buffer at the last run(used to determine if this buffer is growing)
}

impl CachedBufferMapping {
    fn new(pipeline: PipelineType, used_buffer: CachedBufferId, last_size : u64) -> Self {
        Self {
            pipeline,
            used_buffer,
            last_size
        }   
    }
}

#[derive(Debug)]
pub(crate) struct CachedBufferMappings {
    pub(crate) data: Vec<CachedBufferMapping>, //all shader calls, and there used BufferCache
    pub(crate) hash: u64,
    pub(crate) count : u32,//how many times this mapping has been used (used to determine the size of increasing buffers,
                           //e.g. if this mapping has been used 100 times, we can allocate enough memory for 200 runs) 
}

impl CachedBufferMappings {
    fn new(hash: u64) -> Self {
        Self { data: vec![], hash, count : 0 }
    }

    fn add_new(&mut self, buffer: CachedBufferId, pipeline: PipelineType, last_size : u64, index : u32){
        let data = CachedBufferMapping::new(pipeline, buffer, last_size);
        if (index as usize) < self.data.len() {
            self.data[index as usize] = data;
        } else {
            self.data.push(data);
        }
    }

    fn reuse_buffer(&mut self, last_size : u64, index : u32){
        if (index as usize) < self.data.len() {
            let mapping = &mut self.data[index as usize];
            mapping.last_size = last_size;
        }   
    }

    pub(crate) fn get_buffer_mapping(&self, pipeline: &PipelineType, index : u32) -> Option<&CachedBufferMapping> {
        if let Some(mapping) = &self.data.get(index as usize) {
            if &mapping.pipeline == pipeline {
                return Some(mapping);
            }
            else{
                //if this pipeline is inplaceable the pipeline might be different based wheter the buffer was inplaced or not.
                if !pipeline.2.input1_inplaceable &&  !pipeline.2.input2_inplaceable {
                    panic!("expected: {pipeline:?} at index {index}, but got {:?}", mapping.pipeline);
                }
            }
        }
        return None;
    }
}