use std::ops::Deref;

use rocm_rs::hip::Module;

pub struct SendSyncModule(pub Module);

unsafe impl Send for SendSyncModule {}
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
