//! Error type for the SYCL backend. Kept deliberately small and in-tree — the
//! feasibility report (§6e) calls for an auditable FFI/error layer rather than a
//! dependency on an unvetted `sycl-rs`-style crate.

/// A SYCL backend error. Wrapped in [`crate::Error::Sycl`].
#[derive(Debug)]
pub struct SyclError(pub String);

impl SyclError {
    pub fn msg(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for SyclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sycl: {}", self.0)
    }
}

impl std::error::Error for SyclError {}
