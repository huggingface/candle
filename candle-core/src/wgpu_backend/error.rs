
#[derive(thiserror::Error, Debug)]
pub enum WebGpuError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for WebGpuError {
    fn from(e: String) -> Self {
        WebGpuError::Message(e)
    }
}

#[macro_export]
macro_rules! notImplemented {
    ($x:ident) => {{
        let name = String::from(stringify!($x));
        return Err(crate::Error::WebGpu(format!("WebGpu Function not yet Implemented{name}").to_owned().into()));
    }};
}
#[macro_export]
macro_rules! wrongType {
    ($x:ident, $ty:expr) => {{
        let name = String::from(stringify!($x));
        let ty = $ty;
        return Err(crate::Error::WebGpu(format!("Can not create webGpu Array of Type.{:?} (in {name})", ty).to_owned().into()));
    }};
}
