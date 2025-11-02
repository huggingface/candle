use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;
use tracing::instrument;

use super::PipelineReference;
use crate::wgpu_backend::wgpu_functions;
#[derive(Debug)]
pub struct ShaderModuleComputePipelines {
    pub(crate) shader: Arc<wgpu::ShaderModule>,
    pub(crate) pipelines: HashMap<PipelineReference, Arc<wgpu::ComputePipeline>>,
}

#[derive(Debug)]
pub(crate) struct ShaderCache {
    loader_cache: candle_wgpu_kernels::ShaderLoaderCache,
    pub(crate) shaders: HashMap<candle_wgpu_kernels::ShaderIndex, ShaderModuleComputePipelines>,
}

impl ShaderCache {
    pub(crate) fn new() -> Self {
        let mut loader_cache = candle_wgpu_kernels::ShaderLoaderCache::new();
        loader_cache
            .add_wgpu_shader_loader(candle_wgpu_kernels::DefaultWgpuShader::LOADER_INDEX, || {
                candle_wgpu_kernels::DefaultWgpuShader {}
            });
        Self {
            loader_cache,
            shaders: HashMap::default(),
        }
    }

    pub fn add_wgpu_shader_loader<T: candle_wgpu_kernels::ShaderLoader + 'static + Send + Sync>(
        &mut self,
        index: candle_wgpu_kernels::LoaderIndex,
        shader_loader: impl Fn() -> T,
    ) {
        self.loader_cache
            .add_wgpu_shader_loader(index, shader_loader);
    }

    #[instrument(skip(self, pipeline_layout, consts, pipeline))]
    pub(crate) fn get_pipeline(
        &mut self,
        device: &wgpu::Device,
        pipeline: &PipelineReference,
        pipeline_layout: &wgpu::PipelineLayout,
        consts: &[(&str, f64)],
    ) -> crate::Result<Arc<wgpu::ComputePipeline>> {
        let shader = pipeline.0.get_shader();
        let shaders = &mut self.shaders;

        let s = shaders.entry(shader).or_insert_with(|| {
            let shader_str = self.loader_cache.get_shader(shader);
            let s = wgpu_functions::get_shader(device, shader_str);
            ShaderModuleComputePipelines {
                shader: Arc::new(s),
                pipelines: HashMap::default(),
            }
        });

        let pipelines = &mut s.pipelines;
        if !pipelines.contains_key(pipeline) {
            let entry_point_str = self.loader_cache.get_entry_point(pipeline.0);
            let p = load_pipeline(
                device,
                s.shader.clone(),
                entry_point_str,
                pipeline,
                pipeline_layout,
                consts,
            );
            pipelines.insert(pipeline.clone(), Arc::new(p));
        }

        if let Some(p) = pipelines.get(pipeline) {
            return Ok(p.clone());
        } else {
            panic!("Not expected")
        }
    }
}

#[instrument(skip(device, shader, pipeline_layout))]
fn load_pipeline(
    device: &wgpu::Device,
    shader: Arc<wgpu::ShaderModule>,
    entry_point: &str,
    pipeline: &PipelineReference,
    pipeline_layout: &wgpu::PipelineLayout,
    consts: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    let compilation_options = if consts.is_empty() {
        wgpu::PipelineCompilationOptions::default()
    } else {
        wgpu::PipelineCompilationOptions {
            constants: consts,
            zero_initialize_workgroup_memory: true,
        }
    };
    return device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options,
        cache: None,
    });
}
