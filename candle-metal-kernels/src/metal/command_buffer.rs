use crate::{BlitCommandEncoder, ComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use std::{collections::HashMap, thread};

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self { raw }
    }

    pub fn compute_command_encoder(&self) -> ComputeCommandEncoder {
        self.as_ref()
            .computeCommandEncoder()
            .map(ComputeCommandEncoder::new)
            .unwrap()
    }

    pub fn blit_command_encoder(&self) -> BlitCommandEncoder {
        self.as_ref()
            .blitCommandEncoder()
            .map(BlitCommandEncoder::new)
            .unwrap()
    }

    pub fn commit(&self) {
        self.raw.commit()
    }

    pub fn enqueue(&self) {
        self.raw.enqueue()
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(label)))
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn wait_until_completed(&self) {
        unsafe { self.raw.waitUntilCompleted() }
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}

pub struct CommandBufferThreadMap {
    inner: HashMap<thread::ThreadId, CommandBuffer>,
}

impl CommandBufferThreadMap {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn get(&self) -> Option<&CommandBuffer> {
        self.inner.get(&thread::current().id())
    }

    pub fn get_mut(&mut self) -> Option<&mut CommandBuffer> {
        self.inner.get_mut(&thread::current().id())
    }

    pub fn insert(&mut self, command_buffer: CommandBuffer) -> Option<CommandBuffer> {
        self.inner.insert(thread::current().id(), command_buffer)
    }
}
