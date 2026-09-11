use std::ops::Deref;

use rocm_rs::hip::Module;

/// A loaded HIP module that can be shared across threads.
///
/// The field is private and [`SendSyncModule::load_data`] is the only
/// constructor: the `unsafe impl`s below assert a property of *this* wrapper,
/// so safe code outside the crate must not be able to wrap an arbitrary
/// `Module` and get `Send + Sync` for free.
pub struct SendSyncModule(Module);

// SAFETY: the wrapper's only state is a `hipModule_t`, a process-wide handle
// owned by the primary context rather than by any one thread. HIP places no
// thread affinity on it — `hipModuleGetFunction` and `hipModuleLaunchKernel`
// accept it from any thread — and the wrapper exposes no interior mutability,
// so moving it between threads (`Send`) and sharing `&`-references across
// threads (`Sync`) cannot introduce a data race. `rocm-rs` leaves `Module`
// non-`Send`/`Sync` only because it holds a raw pointer.
unsafe impl Send for SendSyncModule {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for SendSyncModule {}

impl SendSyncModule {
    pub fn load_data(data: impl AsRef<[u8]>) -> Result<Self, rocm_rs::hip::error::Error> {
        Ok(Self(Module::load_data(data)?))
    }
}

impl Deref for SendSyncModule {
    type Target = Module;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
