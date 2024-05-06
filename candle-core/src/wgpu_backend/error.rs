
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


