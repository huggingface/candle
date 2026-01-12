
impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Main library error type.
#[derive(thiserror::Error)]
pub enum Error {
    #[error("Wgpu error {0}")]
    Wgpu(#[from] WgpuError),

    /// Zip file format error.
    #[cfg(feature = "wgpu_debug")]
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;


#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for WgpuError {
    fn from(e: String) -> Self {
        WgpuError::Message(e)
    }
}



#[macro_export]
macro_rules! notImplemented {
    ($x:ident) => {{
        let name = String::from(stringify!($x));
        return Err($crate::Error::Wgpu(
            format!("Wgpu Function not yet Implemented {name}")
                .to_owned()
                .into(),
        ));
    }};
}
#[macro_export]
macro_rules! wrongType {
    ($x:ident, $ty:expr) => {{
        let name = String::from(stringify!($x));
        let ty = $ty;
        return Err($crate::Error::Wgpu(
            format!("Can not create wgpu Array of Type.{:?} (in {name})", ty)
                .to_owned()
                .into(),
        ));
    }};
}

#[macro_export]
macro_rules! wgpuError {
    ($x:expr) => {{
        return Err($crate::Error::Wgpu($x.to_owned().into()));
    }};
}
