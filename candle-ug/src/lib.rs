//! This crate is used to re-export the `ug` crate together with `ug-cuda` & `ug-metal` gated
//! behind the `cuda` and `metal` features respectively.

pub use ug::*;

#[cfg(feature = "cuda")]
pub mod cuda {
    pub use ug_cuda::*;
}

#[cfg(feature = "metal")]
pub mod metal {
    pub use ug_metal::*;
}
