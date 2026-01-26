//! Shader loader abstractions and a small runtime cache.
//!
//! The `ShaderLoader` trait represents a source of WGSL shader modules and
//! their entry points. `ShaderLoaderCache` holds registered loaders and
//! provides helper accessors used by the rest of the crate.

use std::any::Any;

use crate::cache::BindGroupReference;
use crate::queue_buffer::OpIsInplaceable;
/// A struct representing the LoaderIndex
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct LoaderIndex(pub u8);

#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ShaderIndex(pub LoaderIndex, pub u16); //loader index, shader index

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
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct PipelineIndex(pub ShaderIndex, pub u8); //Shader, entry point

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

/// Generic trait to load a shader.
pub trait ShaderLoader: std::fmt::Debug + std::any::Any {
    /// Loads the WGSL shader code.
    /// A shader loader may handle multiple shader files; use the `index` to differentiate between shaders.
    /// # Examples
    ///  ```
    /// use wgpu_compute_layer::{ShaderLoader, ShaderIndex, PipelineIndex};
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

    /// Returns the entry point for this pipeline.
    /// A shader loader may handle multiple shader files with multiple entry points in each file;
    /// use the `index` to differentiate between shaders and entry points.
    /// # Examples
    ///  ```
    /// use wgpu_compute_layer::{ShaderLoader, ShaderIndex, PipelineIndex};
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
    /// This name may be used for debugging (for example labeling shader outputs in wgpu) or logging.
    /// Returns `None` when no debug name is available for the given shader index.
    fn get_debug_name(&self, _index: ShaderIndex) -> Option<String> {
        None
    }

    /// Try to rewrite a pipeline invocation into an in-place variant or elide it entirely.
    ///
    /// Returns a description of how to rewrite the call.
    fn rewrite_plan(&self, _: InplaceRewriteDesc<'_>) -> Option<RewritePlan> {
        None
    }

    /// Normalize (canonicalize) the meta buffer for debug and profiling purposes.
    ///
    /// This hook allows shader implementations to remove or neutralize metadata
    /// that does not affect execution performance so that multiple dispatches
    /// with identical performance characteristics can be grouped together.
    ///
    /// Typical use cases include:
    /// - Replacing scalar values with fixed constants when their magnitude does
    ///   not affect runtime (for example `x = 1.0` vs `x = 2.0`).
    /// - Clearing or fixing random seeds or counters for RNG-based kernels.
    /// - Ignoring flags or parameters that only affect numerical results but not
    ///   memory access patterns or control flow.
    ///
    /// Implementations should mutate `meta` in-place by overwriting the relevant
    /// entries with deterministic, canonical values. The modified `meta` slice
    /// is used only for debug/profiling aggregation and has no effect on actual
    /// kernel execution.
    ///
    /// The default implementation performs no normalization.
    fn normalize_debug_meta(&self, _pipeline: PipelineIndex, _meta: &mut [u32]) {}
}

/// Description of an inplace rewrite request
pub struct InplaceRewriteDesc<'a> {
    pub pipeline: PipelineIndex,
    pub bindgroup: &'a BindGroupReference,
    pub inplace_flags: OpIsInplaceable,
}

pub enum RewritePlan {
    /// Replace pipeline + bindgroup, then dispatch
    InplaceDispatch {
        new_pipeline: PipelineIndex,
        new_bindgroup: BindGroupReference,
        replaced_input: ReplacedInput,
    },

    /// Elide the dispatch entirely (copy inplace)
    ElideDispatch { replaced_input: ReplacedInput },
}

pub enum ReplacedInput {
    Input1,
    Input2,
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

    pub fn rewrite_plan(
        &self,
        loader: LoaderIndex,
        desc: InplaceRewriteDesc,
    ) -> Option<RewritePlan> {
        self.loader[loader.0 as usize]
            .as_ref()
            .expect("expected loader to be added")
            .shader_loader
            .rewrite_plan(desc)
    }

    #[cfg(feature = "wgpu_debug")]
    pub(crate) fn normalize_debug_meta(&self, pipeline: PipelineIndex, meta: &mut [u32]) {
        self.loader[pipeline.get_shader().get_loader().0 as usize]
            .as_ref()
            .expect("expected loader to be added")
            .shader_loader
            .normalize_debug_meta(pipeline, meta);
    }
}

impl Default for ShaderLoaderCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests
{
    use std::collections::HashMap;

    use crate::{LoaderIndex, PipelineIndex, ShaderIndex, ShaderLoader, shader_loader::ShaderLoaderCache};

    // A simple shader loader that stores WGSL sources in a HashMap and allows
    // mutation at runtime. This is used to test dynamic shader loader behavior
    // through the public `ShaderLoaderCache` API.
    #[derive(Debug)]
    struct DynamicLoader {
        map: HashMap<u16, String>,
    }

    impl DynamicLoader {
        fn new() -> Self {
            Self { map: HashMap::new() }
        }

        fn insert(&mut self, idx: u16, src: &str) {
            self.map.insert(idx, src.to_string());
        }
    }

    impl ShaderLoader for DynamicLoader {
        fn load(&self, index: ShaderIndex) -> &str {
            let i = index.get_index();
            self.map
                .get(&i)
                .map(|s| s.as_str())
                .unwrap_or("@compute @workgroup_size(1) fn main() {}")
        }

        fn get_entry_point(&self, _index: PipelineIndex) -> &str {
            "main"
        }
    }

    #[test]
    fn shader_loader_cache_dynamic_replace() {
        let mut cache = ShaderLoaderCache::new();

        // Register a dynamic loader at index 0
        cache.add_wgpu_shader_loader(LoaderIndex(0), DynamicLoader::new);

        // Populate the loader with a shader and verify retrieval
        {
            let str1 = "@compute @workgroup_size(1) fn main() { }";
            let loader = cache.get_loader_mut::<DynamicLoader>(LoaderIndex(0)).unwrap();
            loader.insert(0, str1);
            let shader_str = cache.get_shader(ShaderIndex::new(LoaderIndex(0), 0));
            assert_eq!(shader_str, str1);
        }

        // Replace the shader at runtime and verify the change is visible
        {
            let str2 = "@compute @workgroup_size(1) fn main2() { }";
            let loader = cache.get_loader_mut::<DynamicLoader>(LoaderIndex(0)).unwrap();
            loader.insert(0, str2);
            let shader_str = cache.get_shader(ShaderIndex::new(LoaderIndex(0), 0));
            assert_eq!(shader_str, str2);
        }
    }
}
