//! HIP/ROCm support for `ug`.
//!
//! Unlike the `cuda` and `metal` modules, this one is not a re-export: there is
//! no `ug-rocm` crate upstream. See [`code_gen`] for why it has to live here.

pub mod code_gen;
