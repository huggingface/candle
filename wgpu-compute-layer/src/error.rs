impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Main library error type.
#[derive(thiserror::Error)]
pub enum Error {
    #[error("Wgpu error {0}")]
    Msg(String),

    #[error(transparent)]
    PollError(#[from] wgpu::PollError),

    /// Zip file format error.
    #[cfg(feature = "wgpu_debug")]
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Common result type used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

impl From<String> for Error {
    fn from(e: String) -> Self {
        Error::Msg(e)
    }
}
