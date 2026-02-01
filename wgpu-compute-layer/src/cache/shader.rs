//! Shader cache that stores compiled shader modules and created compute pipelines.
//!
//! Pipelines are cached per `(shader, entry_point, constants)` to avoid
//! recompilation at runtime.
use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;
use tracing::instrument;

use super::PipelineReference;
use crate::wgpu_functions;
#[derive(Debug)]
pub struct ShaderModuleComputePipelines {
    pub(crate) shader: Arc<wgpu::ShaderModule>,
    pub(crate) pipelines: HashMap<PipelineReference, Arc<wgpu::ComputePipeline>>,
}

#[derive(Debug)]
pub(crate) struct ShaderCache {
    pub(crate) loader_cache: crate::shader_loader::ShaderLoaderCache,
    pub(crate) shaders: HashMap<(crate::shader_loader::ShaderIndex, usize), ShaderModuleComputePipelines>, //shader + define index
}

impl ShaderCache {
    pub(crate) fn new() -> Self {
        let loader_cache = crate::shader_loader::ShaderLoaderCache::new();
        Self {
            loader_cache,
            shaders: HashMap::default(),
        }
    }

    pub(crate) fn add_wgpu_shader_loader<
        T: crate::shader_loader::ShaderLoader + 'static + Send + Sync,
    >(
        &mut self,
        index: crate::shader_loader::LoaderIndex,
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
        define_index : usize,
        define_cache : &crate::queue_buffer::DefinesCache,
    ) -> crate::Result<Arc<wgpu::ComputePipeline>> {
        let shader = pipeline.index.get_shader();
       
        let shaders = &mut self.shaders;
        let s = shaders.entry((shader, define_index)).or_insert_with(|| {
            let defines = define_cache.get_define(define_index);
            let shader_str = self.loader_cache.get_shader(shader, &defines);
            
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                wgpu_functions::get_shader(device, &shader_str)
            }));
            
            let s = match result {
                Ok(module) => module,
                Err(payload) => {
                    let entry_point_str = self.loader_cache.get_entry_point(pipeline.index);
                    eprintln!("Shader compilation panic!");
                    eprintln!("EntryPoint: '{}'", entry_point_str);
                    eprintln!("Defines:");
                    for (k, v) in defines {
                        eprintln!("  {} = {}", k, v);
                    }

                    eprintln!("Consts:");
                    for (k, v) in consts {
                        eprintln!("  {} = {}", k, v);
                    }

                    // Optional: dump shader source
                    eprintln!("Shader source:\n{}", shader_str);

                    // Re-panic so behavior stays the same
                    std::panic::resume_unwind(payload);
                }
            };

            ShaderModuleComputePipelines {
                shader: Arc::new(s),
                pipelines: HashMap::default(),
            }
        });

        let pipelines = &mut s.pipelines;
        if !pipelines.contains_key(pipeline) {
            let entry_point_str = self.loader_cache.get_entry_point(pipeline.index);
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
            Ok(p.clone())
        } else {
            panic!("pipeline expected to be present after insertion")
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
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options,
        cache: None,
    })
}
