//! This is the rust-gpu source code used to generate the spv output files found in this directory.
//! A pre-release version of rust-gpu 0.10 was used, specifically git 3689d11a. Spirv-Builder was
//! configured like this:
//!
//! ```norun
//! SpirvBuilder::new(shader_crate, "spirv-unknown-vulkan1.2")
//!     .multimodule(true)
//!     .spirv_metadata(SpirvMetadata::Full)
//! ```

use crate::test_shader::some_module::Bla;
use glam::{vec4, Vec4};
use spirv_std::spirv;

mod some_module {
    use super::*;

    pub struct Bla {
        pub value: Vec4,
        pub value2: Vec4,
        pub decider: f32,
    }

    impl Bla {
        pub fn lerp(&self) -> Vec4 {
            self.value * (1. - self.decider) + self.value2 * self.decider
        }
    }
}

#[spirv(vertex)]
pub fn vertex(
    #[spirv(vertex_index)] vertex_id: u32,
    #[spirv(position)] position: &mut Vec4,
    vtx_color: &mut Bla,
) {
    let corners = [
        vec4(-0.5, -0.5, 0.1, 1.),
        vec4(0.5, -0.5, 0.1, 1.),
        vec4(0., 0.5, 0.1, 1.),
    ];
    *position = corners[(vertex_id % 3) as usize];
    *vtx_color = Bla {
        value: vec4(1., 1., 0., 1.),
        value2: vec4(0., 1., 1., 1.),
        decider: f32::max(vertex_id as f32, 1.),
    };
}

#[spirv(fragment)]
pub fn fragment(vtx_color: Bla, f_color: &mut Vec4) {
    *f_color = vtx_color.lerp();
}
