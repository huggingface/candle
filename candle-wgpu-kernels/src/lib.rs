pub use candle_pwgsl::shader_loader::{DefineDefinition, DefinesDefinitions};
use candle_pwgsl::{shader_loader, ParseState, ShaderStore};
mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}

use std::any::Any;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::path::{Path, PathBuf};

use candle_wgpu_kernels_macro::create_loader_internal;
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(
    feature = "wgpu_debug_serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum DType {
    F32,
    U32,
    U8,
    I64,
    F64,
    F16,
}
const DTYPE_COUNT: u16 = 6;

impl DType {
    pub fn get_index(&self) -> u16 {
        match self {
            DType::F32 => 0,
            DType::U32 => 1,
            DType::U8 => 2,
            DType::I64 => 3,
            DType::F64 => 4,
            DType::F16 => 5,
        }
    }

    pub fn from_index(index: u16) -> Self {
        match index {
            0 => DType::F32,
            1 => DType::U32,
            2 => DType::U8,
            3 => DType::I64,
            4 => DType::F64,
            5 => DType::F16,
            _ => {
                todo!()
            }
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::U32 => 4,
            DType::U8 => 1,
            DType::I64 => 8,
            DType::F64 => 8,
            DType::F16 => 2,
        }
    }
}

pub trait EntryPoint {
    fn get_entry_point(&self) -> &'static str;
}

extern crate candle_wgpu_kernels_macro;
pub use candle_wgpu_kernels_macro::create_loader;

/// A struct representing the LoaderIndex
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct LoaderIndex(pub u8);

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ShaderIndex(LoaderIndex, u16); //loader index, shader index

impl ShaderIndex {
    pub fn new(loader: LoaderIndex, index: u16) -> Self {
        ShaderIndex(loader, index)
    }

    pub fn get_loader(&self) -> LoaderIndex {
        self.0
    }

    pub fn get_index(&self) -> u16 {
        self.1
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct PipelineIndex(ShaderIndex, u8); //Shader, entry point

impl PipelineIndex {
    pub fn new(shader: ShaderIndex, index: u8) -> Self {
        PipelineIndex(shader, index)
    }

    pub fn get_shader(&self) -> ShaderIndex {
        self.0
    }

    pub fn get_index(&self) -> u8 {
        self.1
    }
}

impl From<PipelineIndex> for ShaderIndex {
    fn from(val: PipelineIndex) -> Self {
        val.get_shader()
    }
}

impl From<ShaderIndex> for LoaderIndex {
    fn from(val: ShaderIndex) -> Self {
        val.get_loader()
    }
}

impl From<PipelineIndex> for LoaderIndex {
    fn from(val: PipelineIndex) -> Self {
        val.get_shader().get_loader()
    }
}

/// generic trait to load a shader
pub trait ShaderLoader: std::fmt::Debug + std::any::Any {
    /// loads the wgsl shader code
    /// a shader loader may handle multiple shader files, one can use the index to differentiate between shaders
    /// # Examples
    ///  ```
    /// use candle_wgpu_kernels::{ShaderLoader, ShaderIndex, PipelineIndex};
    ///
    /// #[derive(Debug)]
    /// struct Example{};
    ///
    /// impl ShaderLoader for Example{
    ///     fn load(&self, index : ShaderIndex) -> &str{
    ///         let index : u16 = index.get_index();
    ///         match index{
    ///             0 => "@compute fn compute1_i32(...){...} @compute fn compute2_i32(...){...}",
    ///             1 => "@compute fn compute1_f32(...){...} @compute fn compute2_f32(...){...}",
    ///             _ => {todo!()}
    ///         }
    ///     }
    ///     fn get_entry_point(&self, index : PipelineIndex) -> &str{todo!()}
    /// }
    ///  ```
    fn load(&self, index: ShaderIndex) -> &str;

    /// Returns the entry point, of this pipeline.
    /// a shader loader may handle multiple shader files with multiple entry points in each file,
    /// one can use the index to differentiate between shaders and entry points
    /// # Examples
    ///  ```
    /// use candle_wgpu_kernels::{ShaderLoader, ShaderIndex, PipelineIndex};
    /// #[derive(Debug)]
    /// struct Example{};
    ///
    /// impl ShaderLoader for Example{
    ///     fn get_entry_point(&self, index : PipelineIndex) -> &str{
    ///         let shader : ShaderIndex = index.get_shader(); //wich shader to load
    ///         let index : u8 = index.get_index();        //wich entry point to load
    ///         match (shader.get_index(), index){
    ///             (0,0) => "compute1_i32",
    ///             (0,1) => "compute2_i32",
    ///             (1,0) => "compute1_f32",
    ///             (1,1) => "compute2_f32",
    ///             _ => {todo!()}
    ///         }
    ///     }
    ///
    ///     fn load(&self, index : ShaderIndex) -> &str{todo!()}
    /// }
    ///  ```
    fn get_entry_point(&self, index: PipelineIndex) -> &str;

    /// Returns an optional debug name for the given shader.
    /// This name may be used for debugging purposes, such as labeling shader outputs in wgpu or logging.
    /// Returns `None` if no debug name is available for the given shader index.
    fn get_debug_name(&self, _index: ShaderIndex) -> Option<String> {
        None
    }
}

//Struct for storing a custom Shader
#[derive(Debug)]
pub struct ShaderLoaderCache {
    loader: Vec<Option<CustomLoader>>,
}

#[derive(Debug)]
struct CustomLoader {
    shader_loader: Box<dyn ShaderLoader + Send + Sync>,
}

impl ShaderLoaderCache {
    pub fn new() -> Self {
        Self { loader: Vec::new() }
    }

    pub fn add_wgpu_shader_loader<T: ShaderLoader + 'static + Send + Sync>(
        &mut self,
        index: LoaderIndex,
        shader_loader: impl Fn() -> T,
    ) {
        if self.loader.len() <= index.0 as usize {
            let new_items = 1 + index.0 as usize - self.loader.len();
            for _ in 0..new_items {
                self.loader.push(None);
            }
        }
        if self.loader[index.0 as usize].is_none() {
            let loader = shader_loader();
            self.loader[index.0 as usize] = Some(CustomLoader {
                shader_loader: Box::new(loader),
            });
        }
    }

    pub fn get_loader<T: ShaderLoader + 'static>(&self, index: LoaderIndex) -> Option<&T> {
        let loader = self.loader.get(index.0 as usize)?.as_ref()?;
        let any = loader.shader_loader.as_ref() as &dyn Any;
        any.downcast_ref::<T>()
    }

    pub fn get_loader_mut<T: ShaderLoader + 'static>(
        &mut self,
        index: LoaderIndex,
    ) -> Option<&mut T> {
        let loader = self.loader.get_mut(index.0 as usize)?.as_mut()?;
        let any = loader.shader_loader.as_mut() as &mut dyn Any;
        any.downcast_mut::<T>()
    }

    pub fn get_shader(&self, shader: impl Into<ShaderIndex>) -> &str {
        let shader: ShaderIndex = shader.into();
        let loader: LoaderIndex = shader.into();
        self.loader[loader.0 as usize]
            .as_ref()
            .expect("expected loader to be added")
            .shader_loader
            .load(shader)
    }

    pub fn get_shader_name(&self, shader: impl Into<ShaderIndex>) -> String {
        let shader: ShaderIndex = shader.into();
        let loader: LoaderIndex = shader.into();
        self.loader[loader.0 as usize]
            .as_ref()
            .expect("expected loader to be added")
            .shader_loader
            .get_debug_name(shader)
            .unwrap_or_else(|| format!("{:?}", shader))
    }

    pub fn get_entry_point(&self, shader: impl Into<PipelineIndex>) -> &str {
        let shader: PipelineIndex = shader.into();
        let loader: LoaderIndex = shader.into();
        self.loader[loader.0 as usize]
            .as_ref()
            .expect("expected loader to be added")
            .shader_loader
            .get_entry_point(shader)
    }
}

impl Default for ShaderLoaderCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct DefaultWgpuShader {}

create_loader_internal!(DefaultWgpuShader);

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

create_loader_internal!(DefaultWgpuDynamicShader);

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
        let shader_index = ShaderIndex(DefaultWgpuDynamicShader::LOADER_INDEX, shader_id);

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

        PipelineIndex(shader_index, entry_idx as u8)
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
