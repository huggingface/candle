use crate::{
    ComputePipeline, ConstantValues, Device, Function, Library, MTLCompileOptions, MTLMathMode,
    MetalKernelError, Source, AFFINE, BINARY, CAST, CONV, FILL, INDEXING, MLX_GEMM, MLX_SORT,
    QUANTIZED, RANDOM, REDUCE, SDPA, SORT, TERNARY, UNARY,
};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone)]
pub enum KernelName {
    Ref(&'static str),
    Value(String),
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        match self {
            Self::Ref(r) => r,
            Self::Value(v) => v.as_str(),
        }
    }
}

impl std::hash::Hash for KernelName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Ref(r) => r.hash(state),
            Self::Value(v) => v.hash(state),
        }
    }
}

impl PartialEq for KernelName {
    fn eq(&self, other: &Self) -> bool {
        let v1: &str = self.as_ref();
        let v2: &str = other.as_ref();
        v1 == v2
    }
}

impl Eq for KernelName {}

impl From<&'static str> for KernelName {
    fn from(value: &'static str) -> Self {
        Self::Ref(value)
    }
}

impl From<String> for KernelName {
    fn from(value: String) -> Self {
        Self::Value(value)
    }
}

type Libraries = HashMap<Source, Library>;
type Pipelines = HashMap<(KernelName, Option<ConstantValues>), ComputePipeline>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let pipelines = RwLock::new(Pipelines::new());
        Self {
            libraries,
            pipelines,
        }
    }

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::Affine => AFFINE,
            Source::Binary => BINARY,
            Source::Cast => CAST,
            Source::Conv => CONV,
            Source::Fill => FILL,
            Source::Gemm => MLX_GEMM,
            Source::Indexing => INDEXING,
            Source::MlxSort => MLX_SORT,
            Source::Quantized => QUANTIZED,
            Source::Random => RANDOM,
            Source::Reduce => REDUCE,
            Source::Sort => SORT,
            Source::Ternary => TERNARY,
            Source::Unary => UNARY,
            Source::Sdpa => SDPA,
        }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let mut libraries = self.libraries.write()?;
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let lib = {
                let source_content = self.get_library_source(source);
                let compile_options = MTLCompileOptions::new();
                //unsafe { compile_options.setEnableLogging(true) };
                unsafe { compile_options.setMathMode(MTLMathMode::Fast) };
                device
                    .new_library_with_source(source_content, Some(&compile_options))
                    .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?
            };
            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &str,
        constants: Option<&ConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(name, constants)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    pub fn load_pipeline_with_constants(
        &self,
        device: &Device,
        source: Source,
        name: impl Into<KernelName>,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipeline, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = (name.into(), constants);
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(device, source, name.as_ref(), constants.as_ref())?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert((name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: impl Into<KernelName>,
    ) -> Result<ComputePipeline, MetalKernelError> {
        self.load_pipeline_with_constants(device, source, name, None)
    }
}
