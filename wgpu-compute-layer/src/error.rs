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

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()))
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()))
    };
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Error::Msg(e)
    }
}
