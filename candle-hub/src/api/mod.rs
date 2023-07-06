/// The asynchronous version of the API
#[cfg(feature = "tokio")]
pub mod tokio;

/// The synchronous version of the API
pub mod sync;
