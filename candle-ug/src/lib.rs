//! This crate is used to re-export the `ug` crate together with `ug-cuda` & `ug-metal` gated
//! behind the `cuda` and `metal` features respectively.
//!
//! The `rocm` feature is the odd one out: there is no `ug-rocm` crate to re-export, so the HIP
//! code generator is implemented here. See [`rocm::code_gen`].

pub use ug::*;

#[cfg(feature = "cuda")]
pub mod cuda {
    pub use ug_cuda::*;
}

#[cfg(feature = "metal")]
pub mod metal {
    pub use ug_metal::*;
}

#[cfg(feature = "rocm")]
pub mod rocm;
