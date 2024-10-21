mod generated{
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}
use std::sync::Arc;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "wgpu_debug_serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum DType {
    F32,
    U32,
    U8,
    I64,
    F64
}
const DTYPE_COUNT : u16 = 5;

impl DType{
    pub fn get_index(&self) -> u16{
        match self {
            DType::F32 => return 0,
            DType::U32 => return 1,
            DType::U8 => return 2,
            DType::I64 => return 3,
            DType::F64 => return 4,
        }
    }

    pub fn from_index(index : u16) -> Self{
        match index {
            0 =>  DType::F32,
            1 => DType::U32,
            2 => DType::U8,
            3 => DType::I64,
            4 => DType::F64,
            _=> {todo!()}
        }
    }
}

pub trait EntryPoint {
    fn get_entry_point(&self) -> &'static str;
}





extern crate candle_wgpu_kernels_macro;
pub use candle_wgpu_kernels_macro::create_loader;

/// A struct representing the LoaderIndex
#[derive(Copy, Clone, Debug, PartialEq,Hash, Eq)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct LoaderIndex(pub u8);

#[derive(Copy, Clone, Debug, PartialEq,Hash, Eq)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ShaderIndex(LoaderIndex, u16); //loader index, shader index

impl ShaderIndex{
    pub fn new(loader : LoaderIndex, index : u16) -> Self{
        ShaderIndex(loader, index)
    }

    pub fn get_loader(&self) -> LoaderIndex{
        self.0
    }

    pub fn get_index(&self) -> u16{
        self.1
    }
}


#[derive(Copy, Clone, Debug, PartialEq,Hash, Eq)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct PipelineIndex(ShaderIndex, u8); //Shader, entry point

impl PipelineIndex{
    pub fn new(shader : ShaderIndex, index : u8) -> Self{
        PipelineIndex(shader, index)
    }

    pub fn get_shader(&self) -> ShaderIndex{
        self.0
    }

    pub fn get_index(&self) -> u8{
        self.1
    }
}

impl Into<ShaderIndex> for PipelineIndex{
    fn into(self) -> ShaderIndex {
       self.get_shader()
    }
}

impl Into<LoaderIndex> for ShaderIndex{
    fn into(self) -> LoaderIndex {
       self.get_loader()
    }
}

impl Into<LoaderIndex> for PipelineIndex{
    fn into(self) -> LoaderIndex {
       self.get_shader().get_loader()
    }
}




/// generic trait to load a shader
pub trait ShaderLoader : std::fmt::Debug{
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
    fn load(&self, index : ShaderIndex) -> &str;

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
    fn get_entry_point(&self, index : PipelineIndex) -> &str;
}


//Struct for storing a custom Shader
#[derive(Debug)]
pub struct ShaderLoaderCache{
    loader : Vec<Option<CustomLoader>>
}

#[derive(Debug)]
struct CustomLoader{
    shader_loader : Arc<dyn ShaderLoader + Send + Sync>,
}



impl ShaderLoaderCache{
    pub fn new() -> Self {
        Self { loader : Vec::new() }
    }
    
    pub fn add_wgpu_shader_loader<T : ShaderLoader + 'static + Send + Sync>(&mut self, index : LoaderIndex, shader_loader : impl Fn() -> T){
        if self.loader.len() <= index.0 as usize {
            let new_items = 1 + index.0 as usize - self.loader.len();
            for _ in 0..new_items{
                self.loader.push(None);
            }
        }
        if self.loader[index.0 as usize].is_none() {
            let loader = shader_loader();
            self.loader[index.0 as usize] = Some(CustomLoader{shader_loader : Arc::new(loader)});
        }
    }

    pub fn get_shader(&self, shader : impl Into<ShaderIndex>) -> &str{
        let shader : ShaderIndex = shader.into();
        let loader : LoaderIndex = shader.into();
        return self.loader[loader.0 as usize].as_ref().expect("expected loader to be added").shader_loader.load(shader);
    }

    pub fn get_entry_point(&self, shader : impl Into<PipelineIndex>) -> &str{
        let shader : PipelineIndex = shader.into();
        let loader : LoaderIndex = shader.into();
        return self.loader[loader.0 as usize].as_ref().expect("expected loader to be added").shader_loader.get_entry_point(shader);
    }


}

create_loader_internal!(DefaultWgpuShader);

impl ShaderLoader for DefaultWgpuShader{
    fn load(&self, index : ShaderIndex) -> &str {
        let shader : Shaders = index.into();
        shader.load_shader()
    }

    fn get_entry_point(&self, index : PipelineIndex) -> &str {
        let pipeline : Pipelines = index.into();
        return pipeline.get_entry_point();
    }
}