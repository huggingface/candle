use super::{Buffer, Device};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLAllocation, MTLDevice as _, MTLResidencySet, MTLResidencySetDescriptor};

/// Keeps Metal buffers resident in GPU memory, removing the need for `useResource` calls.
///
/// Add the set to the command queue once via `MTLCommandQueue::addResidencySet`. Then
/// register every buffer at allocation time and unregister at free time.
pub struct ResidencySet {
    raw: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
}

unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl ResidencySet {
    pub fn new(device: &Device) -> Self {
        let descriptor = MTLResidencySetDescriptor::new();
        let raw = device
            .as_ref()
            .newResidencySetWithDescriptor_error(&descriptor)
            .ok()
            .inspect(|set| set.requestResidency());
        ResidencySet { raw }
    }

    pub fn raw(&self) -> Option<&ProtocolObject<dyn MTLResidencySet>> {
        self.raw.as_deref()
    }

    pub fn insert(&self, buf: &Buffer) {
        self.insert_batch(std::iter::once(buf));
    }

    /// Adds many buffers with a single commit: each commit synchronously
    /// applies the pending changes, so batching a large set (e.g. weights at
    /// model load) avoids paying that cost once per buffer. An empty batch is
    /// a no-op.
    pub fn insert_batch<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        if let Some(set) = &self.raw {
            let mut any = false;
            for buf in bufs {
                set.addAllocation(as_allocation(buf));
                any = true;
            }
            if any {
                set.commit();
            }
        }
    }

    /// Removes many buffers with a single commit; the batch counterpart of
    /// `insert_batch`. An empty batch is a no-op.
    pub fn remove_batch<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        if let Some(set) = &self.raw {
            let mut any = false;
            for buf in bufs {
                set.removeAllocation(as_allocation(buf));
                any = true;
            }
            if any {
                set.commit();
            }
        }
    }

    pub fn remove(&self, buf: &Buffer) {
        self.remove_batch(std::iter::once(buf));
    }
}

/// Cast a `&Buffer` to `&ProtocolObject<dyn MTLAllocation>`.
///
/// Safe because `MTLBuffer: MTLResource: MTLAllocation`. All `ProtocolObject<P>` are
/// `repr(C)` thin ObjC pointers — the cast only changes the static protocol marker,
/// not the pointer value or runtime dispatch.
fn as_allocation(buf: &Buffer) -> &ProtocolObject<dyn MTLAllocation> {
    unsafe {
        &*(buf.as_ref() as *const ProtocolObject<dyn objc2_metal::MTLBuffer>
            as *const ProtocolObject<dyn MTLAllocation>)
    }
}
