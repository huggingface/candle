use wgpu_compute_layer::cache::BindGroupReference;
use wgpu_compute_layer::cache::BindgroupAlignmentLayout;
use wgpu_compute_layer::cache::BindgroupInputBase;
use wgpu_compute_layer::cache::BindgroupReferenceInput;
use wgpu_compute_layer::create_loader;
use wgpu_compute_layer::InplaceRewriteDesc;
use wgpu_compute_layer::KernelConstId;
use wgpu_compute_layer::PipelineIndex;
use wgpu_compute_layer::ReplacedInput;
use wgpu_compute_layer::RewritePlan;
use wgpu_compute_layer::ShaderIndex;
use wgpu_compute_layer::ShaderLoader;
pub use wgpu_compute_layer_pwgsl::shader_loader::{DefineDefinition, DefinesDefinitions};
use wgpu_compute_layer_pwgsl::{shader_loader, ParseState, ShaderStore};
mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}

pub use wgpu_compute_layer::DType;
pub use wgpu_compute_layer::EntryPoint;
pub use wgpu_compute_layer::DTYPE_COUNT;

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::path::PathBuf;

pub use generated::kernels::*;
pub use generated::*;

impl Constants {
    pub fn get_const(i: usize) -> Constants {
        match i {
            0 => Constants::Constv0,
            1 => Constants::Constv1,
            2 => Constants::Constv2,
            3 => Constants::Constv3,
            4 => Constants::Constv4,
            5 => Constants::Constv5,
            6 => Constants::Constv6,
            7 => Constants::Constv7,
            8 => Constants::Constv8,
            9 => Constants::Constv9,
            //10 => Constants::Constv10,
            //11 => Constants::Constv11,
            _ => todo!(),
        }
    }
}

impl From<Constants> for KernelConstId {
    fn from(val: Constants) -> Self {
        KernelConstId(val.get_entry_point())
    }
}

#[derive(Debug)]
pub struct DefaultWgpuShader {
    shader_store: ShaderStore, // build-time embedded shaders
}

impl DefaultWgpuShader {
    pub fn new() -> Self {
        DefaultWgpuShader {
            shader_store: embedded_shader_store(),
        }
    }
}

impl Default for DefaultWgpuShader {
    fn default() -> Self {
        Self::new()
    }
}

create_loader!(DefaultWgpuShader);

impl ShaderLoader for DefaultWgpuShader {
    fn load(&self, index: ShaderIndex, defines: &[(&str, String)]) -> Cow<'_, str> {
        let shader: Shaders = index.into();
        let path = shader.load_shader();
        let typ: crate::DType = shader.get_type();

        let mut state_defines = HashMap::new();
        for define in defines {
            state_defines.insert(
                define.0.to_string(),
                DefineDefinition::new(define.1.clone()),
            );
        }
        let typ_define_str = match typ {
            DType::F32 => "f32",
            DType::U32 => "u32",
            DType::U8 => "u8",
            DType::I64 => "i64",
            DType::F64 => "f64",
            DType::F16 => "f16",
        };

        #[cfg(target_arch = "wasm32")]
        {
            state_defines.insert(
                "USE_IMMEDIATES".to_string(),
                DefineDefinition::new("0".to_string()),
            );
        }

        state_defines.insert(
            typ_define_str.to_string(),
            DefineDefinition::new(typ_define_str.to_string()),
        );

        let mut parse_state = ParseState::new();
        parse_state.set_path(PathBuf::from(path));
        parse_state.set_defines(state_defines.clone().into_iter().collect());

        let processed = shader_loader::load_shader(&mut parse_state, &self.shader_store);
        processed.into()
    }

    fn get_entry_point(&self, index: PipelineIndex) -> &str {
        let pipeline: Pipelines = index.into();
        pipeline.get_entry_point()
    }

    fn get_debug_name(&self, index: ShaderIndex) -> Option<String> {
        let shader: Shaders = index.into();
        Some(format!("{:?}", shader))
    }

    fn rewrite_plan(&self, desc: InplaceRewriteDesc<'_>) -> Option<RewritePlan> {
        let pipeline: Pipelines = desc.pipeline.into();

        match pipeline {
            Pipelines::Unary(dtype, unary::Functions::UnaryFromBufferContiguous)
                if desc.inplace_flags.input1_inplaceable =>
            {
                let BindgroupReferenceInput::Bindgroup1(v1, layout) = desc.bindgroup.get_input()
                else {
                    return None;
                };

                Some(RewritePlan::InplaceDispatch {
                    new_pipeline: Pipelines::Unary(dtype, unary::Functions::UnaryInplaceContiguous)
                        .into(),
                    new_bindgroup: BindGroupReference::new(
                        *v1,
                        BindgroupInputBase::Bindgroup0(BindgroupAlignmentLayout::Bindgroup0(
                            layout.get_dest(),
                        )),
                    ),
                    replaced_input: ReplacedInput::Input1,
                })
            }

            Pipelines::Binary(dtype, binary::Functions::BinaryBufferFromBufferContiguousBoth)
                if desc.inplace_flags.input1_inplaceable =>
            {
                let BindgroupReferenceInput::Bindgroup2(v1, v2, layout) =
                    desc.bindgroup.get_input()
                else {
                    return None;
                };

                Some(RewritePlan::InplaceDispatch {
                    new_pipeline: Pipelines::Binary(
                        dtype,
                        binary::Functions::BinaryBufferInplace1ContiguousBoth,
                    )
                    .into(),
                    new_bindgroup: BindGroupReference::new(
                        *v1,
                        BindgroupInputBase::Bindgroup1(
                            *v2,
                            BindgroupAlignmentLayout::Bindgroup1(
                                layout.get_dest(),
                                layout.get_dest(),
                            ),
                        ),
                    ),
                    replaced_input: ReplacedInput::Input1,
                })
            }
            Pipelines::Binary(dtype, binary::Functions::BinaryBufferFromBufferContiguousBoth)
                if desc.inplace_flags.input2_inplaceable =>
            {
                let BindgroupReferenceInput::Bindgroup2(v1, v2, layout) =
                    desc.bindgroup.get_input()
                else {
                    return None;
                };

                Some(RewritePlan::InplaceDispatch {
                    new_pipeline: Pipelines::Binary(
                        dtype,
                        binary::Functions::BinaryBufferInplace2ContiguousBoth,
                    )
                    .into(),
                    new_bindgroup: BindGroupReference::new(
                        *v2,
                        BindgroupInputBase::Bindgroup1(
                            *v1,
                            BindgroupAlignmentLayout::Bindgroup1(
                                layout.get_dest(),
                                layout.get_dest(),
                            ),
                        ),
                    ),
                    replaced_input: ReplacedInput::Input2,
                })
            }
            Pipelines::Copy(_, copy::Functions::Copy) if desc.inplace_flags.input1_inplaceable => {
                Some(RewritePlan::ElideDispatch {
                    replaced_input: ReplacedInput::Input1,
                })
            }
            _ => None,
        }
    }

    fn normalize_debug_meta(&self, pipeline: PipelineIndex, meta: &mut [u32]) {
        let pipeline: Pipelines = pipeline.into();
        //the scalar and randstate on unary should have no performance effect:
        if let Pipelines::Unary(_, unary::Functions::RandInplaceContiguous) = pipeline {
            meta[1] = f32::to_bits(1.0);
            meta[2] = f32::to_bits(1.0);
            meta[3] = 0; //rand state
        } else if let Pipelines::Unary(_, _) = pipeline {
            meta[1] = f32::to_bits(1.0);
            meta[2] = f32::to_bits(1.0);
        }
    }
}
