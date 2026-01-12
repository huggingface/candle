use wgpu_compute_engine::cache::BindGroupReference;
use wgpu_compute_engine::cache::BindgroupAlignmentLayout;
use wgpu_compute_engine::cache::BindgroupInputBase;
use wgpu_compute_engine::create_loader;
use wgpu_compute_engine::shader_loader::PipelineIndex;
use wgpu_compute_engine::shader_loader::ReplacedInput;
use wgpu_compute_engine::shader_loader::ShaderIndex;
use wgpu_compute_engine::shader_loader::ShaderLoader;
use wgpu_compute_engine::wgpu_functions::KernelConstId;
pub use wgpu_compute_engine_pwgsl::shader_loader::{DefineDefinition, DefinesDefinitions};
use wgpu_compute_engine_pwgsl::{shader_loader, ParseState, ShaderStore};
use wgpu_compute_engine::cache::BindgroupReferenceInput;
use wgpu_compute_engine::shader_loader::InplaceRewriteDesc;
use wgpu_compute_engine::shader_loader::RewritePlan;
mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}

pub use wgpu_compute_engine::DType;
pub use wgpu_compute_engine::EntryPoint;
pub use wgpu_compute_engine::DTYPE_COUNT;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::path::{Path, PathBuf};

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

impl Into<KernelConstId> for Constants {
    fn into(self) -> KernelConstId {
        KernelConstId(self.get_entry_point())
    }
}

#[derive(Debug)]
pub struct DefaultWgpuShader {}

create_loader!(DefaultWgpuShader);

impl ShaderLoader for DefaultWgpuShader {
    fn load(&self, index: ShaderIndex) -> &str {
        let shader: Shaders = index.into();
        shader.load_shader()
    }

    fn get_entry_point(&self, index: PipelineIndex) -> &str {
        let pipeline: Pipelines = index.into();
        pipeline.get_entry_point()
    }

    fn get_debug_name(&self, index: ShaderIndex) -> Option<String> {
        let shader: Shaders = index.into();
        Some(format!("{:?}", shader))
    }

    fn rewrite_plan(
        &self,
        desc: InplaceRewriteDesc<'_>,
    ) -> Option<RewritePlan> {
        let pipeline: Pipelines = desc.pipeline.into();

        match pipeline {
            Pipelines::Unary(dtype, unary::Functions::UnaryFromBufferContiguous)
                if desc.inplace_flags.input1_inplaceable =>
            {
                let BindgroupReferenceInput::Bindgroup1(v1, layout) =
                    desc.bindgroup.get_input() else { return None };

                Some(RewritePlan::InplaceDispatch {
                    new_pipeline: Pipelines::Unary(
                        dtype,
                        unary::Functions::UnaryInplaceContiguous,
                    ).into(),
                    new_bindgroup: BindGroupReference::new(
                        *v1,
                        BindgroupInputBase::Bindgroup0(BindgroupAlignmentLayout::Bindgroup0(layout.get_dest())),
                    ),
                    replaced_input: ReplacedInput::Input1,
                })
            }

            Pipelines::Binary(dtype, binary::Functions::BinaryBufferFromBufferContiguousBoth)
                if desc.inplace_flags.input1_inplaceable =>
            {
                let BindgroupReferenceInput::Bindgroup2(v1, v2, layout) =
                    desc.bindgroup.get_input() else { return None };

                Some(RewritePlan::InplaceDispatch {
                    new_pipeline: Pipelines::Binary(
                        dtype,
                        binary::Functions::BinaryBufferInplace1ContiguousBoth,
                    ).into(),
                    new_bindgroup: BindGroupReference::new(
                        *v1,
                        BindgroupInputBase::Bindgroup1(*v2, BindgroupAlignmentLayout::Bindgroup1(layout.get_dest(), layout.get_dest())),
                    ),
                    replaced_input: ReplacedInput::Input1,
                })
            }
            Pipelines::Binary(dtype, binary::Functions::BinaryBufferFromBufferContiguousBoth)
                if desc.inplace_flags.input2_inplaceable =>
            {
                let BindgroupReferenceInput::Bindgroup2(v1, v2, layout) =
                    desc.bindgroup.get_input() else { return None };

                Some(RewritePlan::InplaceDispatch {
                    new_pipeline: Pipelines::Binary(
                        dtype,
                        binary::Functions::BinaryBufferInplace2ContiguousBoth,
                    ).into(),
                    new_bindgroup: BindGroupReference::new(
                        *v2,
                        BindgroupInputBase::Bindgroup1(*v1, BindgroupAlignmentLayout::Bindgroup1(layout.get_dest(), layout.get_dest())),
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

    fn normalize_debug_meta(
            &self,
            pipeline: PipelineIndex,
            meta: &mut [u32],
        ) {
        let pipeline : Pipelines = pipeline.into();
        //the scalar and randstate on unary should have no performance effect:
        if let Pipelines::Unary(_, unary::Functions::RandInplaceContiguous) = pipeline {
            meta[1] = f32::to_bits(1.0);
            meta[2] = f32::to_bits(1.0);
            meta[3] = 0; //rand state
        }
        else if let Pipelines::Unary(_, _) = pipeline {
            meta[1] = f32::to_bits(1.0);
            meta[2] = f32::to_bits(1.0);
        }
    }
}

//Dynamic ShaderLoader Implementation
#[derive(Hash, PartialEq, Eq, Debug)]
struct ShaderKey {
    path: PathBuf,
    defines: Vec<(String, DefineDefinition)>,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone)]
struct ShaderKeyRef<'a> {
    path: &'a Path,
    defines: &'a [(&'a str, DefineDefinition)],
}

enum ShaderKeyRefOrOwned<'a> {
    Ref(ShaderKeyRef<'a>),
    Owned(&'a ShaderKey),
}

trait ShaderKeyTrait {
    fn key<'a>(&'a self) -> ShaderKeyRefOrOwned<'a>;
    fn key_eq(&self, other: &dyn ShaderKeyTrait) -> bool;
}

impl ShaderKeyTrait for ShaderKey {
    fn key<'a>(&'a self) -> ShaderKeyRefOrOwned<'a> {
        ShaderKeyRefOrOwned::Owned(self)
    }
    fn key_eq(&self, other: &dyn ShaderKeyTrait) -> bool {
        match other.key() {
            ShaderKeyRefOrOwned::Owned(o) => {
                self.path == o.path
                    && self.defines.len() == o.defines.len()
                    && self.defines.iter().zip(&o.defines).all(
                        |((a_name, a_def), (b_name, b_def))| a_name == b_name && a_def == b_def,
                    )
            }
            ShaderKeyRefOrOwned::Ref(r) => {
                self.path == r.path
                    && self.defines.len() == r.defines.len()
                    && self.defines.iter().zip(r.defines).all(
                        |((a_name, a_def), (b_name, b_def))| a_name == b_name && a_def == b_def,
                    )
            }
        }
    }
}

impl<'a> ShaderKeyTrait for ShaderKeyRef<'a> {
    fn key<'b>(&'b self) -> ShaderKeyRefOrOwned<'b> {
        ShaderKeyRefOrOwned::Ref(*self)
    }
    fn key_eq(&self, other: &dyn ShaderKeyTrait) -> bool {
        match other.key() {
            ShaderKeyRefOrOwned::Owned(o) => {
                self.path == o.path
                    && self.defines.len() == o.defines.len()
                    && self.defines.iter().zip(&o.defines).all(
                        |((a_name, a_def), (b_name, b_def))| a_name == b_name && a_def == b_def,
                    )
            }
            ShaderKeyRefOrOwned::Ref(r) => {
                self.path == r.path
                    && self.defines.len() == r.defines.len()
                    && self.defines.iter().zip(r.defines).all(
                        |((a_name, a_def), (b_name, b_def))| a_name == b_name && a_def == b_def,
                    )
            }
        }
    }
}

impl<'a> Borrow<dyn ShaderKeyTrait + 'a> for ShaderKey {
    fn borrow(&self) -> &(dyn ShaderKeyTrait + 'a) {
        self
    }
}

impl<'a> Eq for dyn ShaderKeyTrait + 'a {}

impl<'a> PartialEq for dyn ShaderKeyTrait + 'a {
    fn eq(&self, other: &dyn ShaderKeyTrait) -> bool {
        self.key_eq(other)
    }
}

impl<'a> Hash for dyn ShaderKeyTrait + 'a {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self.key() {
            ShaderKeyRefOrOwned::Ref(shader_key_ref) => shader_key_ref.hash(state),
            ShaderKeyRefOrOwned::Owned(shader_key) => shader_key.hash(state),
        }
    }
}

#[derive(Debug)]
struct DynamicShaderEntry {
    dynamic_shader: String,     //the shader content after the preprocessor run
    debug_name: Option<String>, //debug name of the shader
    entry_points: Vec<String>,  //PipelineIndex to EntryPoint Name
}

#[derive(Debug)]
pub struct DefaultWgpuDynamicShader {
    shader_store: ShaderStore, // build-time embedded shaders
    shader_index_mapping: HashMap<ShaderKey, ShaderIndex>,
    shader_cache: Vec<DynamicShaderEntry>,
}

create_loader!(DefaultWgpuDynamicShader);

impl DefaultWgpuDynamicShader {
    pub fn new() -> Self {
        DefaultWgpuDynamicShader {
            shader_store: embedded_shader_store(),
            shader_index_mapping: HashMap::new(),
            shader_cache: Vec::new(),
        }
    }

    pub fn get_shader_index(
        &mut self,
        path: PathBuf,
        defines: &[(&str, DefineDefinition)],
    ) -> ShaderIndex {
        let key_ref = ShaderKeyRef {
            path: path.as_path(),
            defines,
        };

        if let Some(&idx) = self
            .shader_index_mapping
            .get(&key_ref as &dyn ShaderKeyTrait)
        {
            return idx;
        }

        let key = ShaderKey {
            path: path.clone(),
            defines: defines
                .iter()
                .map(|c| (c.0.to_string(), c.1.clone()))
                .collect(),
        };

        let mut parse_state = ParseState::new();
        parse_state.set_path(path.clone());
        parse_state.set_defines(key.defines.clone().into_iter().collect());
        let processed = shader_loader::load_shader(&mut parse_state, &self.shader_store);

        let entry = DynamicShaderEntry {
            dynamic_shader: processed,
            debug_name: Some(path.display().to_string()),
            entry_points: parse_state
                .info()
                .global_functions
                .iter()
                .map(|c| c.1.clone())
                .collect(),
        };

        let shader_id = self.shader_cache.len() as u16;
        let shader_index = ShaderIndex::new(DefaultWgpuDynamicShader::LOADER_INDEX, shader_id);

        self.shader_cache.push(entry);
        self.shader_index_mapping.insert(key, shader_index);

        shader_index
    }

    pub fn get_pipeline_index(&self, shader_index: ShaderIndex, entry_name: &str) -> PipelineIndex {
        let entries = &self.shader_cache[shader_index.1 as usize].entry_points;

        let entry_idx = entries
            .iter()
            .position(|s| s == entry_name)
            .unwrap_or_else(|| {
                panic!(
                    r#"
    Failed to find shader entry point

    Shader index: {:?}
    Requested:    {}
    Available:   [{}]

    "#,
                    shader_index,
                    entry_name,
                    entries.join(", "),
                )
            });

        PipelineIndex::new(shader_index, entry_idx as u8)
    }
}

impl Default for DefaultWgpuDynamicShader {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderLoader for DefaultWgpuDynamicShader {
    fn load(&self, index: ShaderIndex) -> &str {
        let ShaderIndex(_, shader_id) = index;
        &self.shader_cache[shader_id as usize].dynamic_shader
    }

    fn get_entry_point(&self, index: PipelineIndex) -> &str {
        let PipelineIndex(ShaderIndex(_, shader_id), entry_id) = index;
        &self.shader_cache[shader_id as usize].entry_points[entry_id as usize]
    }

    fn get_debug_name(&self, index: ShaderIndex) -> Option<String> {
        let ShaderIndex(_, shader_id) = index;
        self.shader_cache[shader_id as usize].debug_name.clone()
    }
}
