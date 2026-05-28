use super::Device;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLFence;

/// Cross-encoder synchronization primitive for HazardTrackingModeUntracked resources.
///
/// memoryBarrierWithScope only works within a single encoder. When transitioning between
/// encoders (e.g. compute -> blit for to_cpu, blit -> compute for next dispatch), untracked
/// resources need explicit fence wait/update calls to ensure memory visibility.
pub struct Fence {
    raw: Retained<ProtocolObject<dyn MTLFence>>,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    pub fn new(device: &Device) -> Self {
        use objc2_metal::MTLDevice as _;
        let raw = device
            .as_ref()
            .newFence()
            .expect("failed to create MTLFence");
        Fence { raw }
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLFence> {
        &self.raw
    }
}
