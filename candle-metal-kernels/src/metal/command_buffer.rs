use crate::{BlitCommandEncoder, ComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use std::borrow::Cow;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Clone, Debug, PartialEq)]
pub enum CommandStatus {
    Available,
    Encoding,
    Done,
}

#[derive(Debug)]
pub struct CommandSemaphore {
    pub cond: Condvar,
    pub status: Mutex<CommandStatus>,
}

impl CommandSemaphore {
    pub fn new() -> CommandSemaphore {
        CommandSemaphore {
            cond: Condvar::new(),
            status: Mutex::new(CommandStatus::Available),
        }
    }

    pub fn wait_until<F: FnMut(&mut CommandStatus) -> bool>(
        &self,
        mut f: F,
    ) -> MutexGuard<'_, CommandStatus> {
        self.cond
            .wait_while(self.status.lock().unwrap(), |s| !f(s))
            .unwrap()
    }

    pub fn set_status(&self, status: CommandStatus) {
        *self.status.lock().unwrap() = status;
        // We notify the condvar that the value has changed.
        self.cond.notify_one();
    }

    pub fn when<T, B: FnMut(&mut CommandStatus) -> bool, F: FnMut() -> T>(
        &self,
        b: B,
        mut f: F,
        next: Option<CommandStatus>,
    ) -> T {
        let mut guard = self.wait_until(b);
        let v = f();
        if let Some(status) = next {
            *guard = status;
            self.cond.notify_one();
        }
        v
    }
}

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    semaphore: Arc<CommandSemaphore>,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> Self {
        Self { raw, semaphore }
    }

    pub fn compute_command_encoder(&self) -> ComputeCommandEncoder {
        self.as_ref()
            .computeCommandEncoder()
            .map(|raw| ComputeCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
            .unwrap()
    }

    pub fn blit_command_encoder(&self) -> BlitCommandEncoder {
        self.as_ref()
            .blitCommandEncoder()
            .map(|raw| BlitCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
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

    pub fn error(&self) -> Option<Cow<'_, str>> {
        unsafe {
            self.raw.error().map(|error| {
                let description = error.localizedDescription();
                let c_str = core::ffi::CStr::from_ptr(description.UTF8String());
                c_str.to_string_lossy()
            })
        }
    }

    pub fn wait_until_completed(&self) {
        self.raw.waitUntilCompleted();
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}
