use std::{collections::HashMap, sync::Arc};

use tracing::instrument;

use crate::wgpu_backend::{device::PipelineType, wgpu_functions};
use candle_wgpu_kernels::EntryPoint;

#[derive(Debug)]
pub struct ShaderModuleComputePipelines {
    pub(crate) shader: Arc<wgpu::ShaderModule>,
    pub(crate) pipelines: HashMap<PipelineType, Arc<wgpu::ComputePipeline>>,
}

#[derive(Debug)]
pub(crate) struct ShaderCache {
    pub(crate) shaders: HashMap<candle_wgpu_kernels::Shaders, ShaderModuleComputePipelines>,
}

impl ShaderCache {
    pub(crate) fn new() -> Self {
        Self {
            shaders: HashMap::new(),
        }
    }

    #[instrument(skip(self, pipeline_layout, consts, pipeline))]
    pub(crate) fn get_pipeline(
        &mut self,
        device: &wgpu::Device,
        pipeline: &PipelineType,
        pipeline_layout: &wgpu::PipelineLayout,
        consts: &HashMap<String, f64>,
    ) -> crate::Result<Arc<wgpu::ComputePipeline>> {
        let shader = pipeline.0.get_shader();
        let shaders = &mut self.shaders;

        let s = shaders
            .entry(shader.clone())
            .or_insert_with(|| 
                {
                    let s = wgpu_functions::get_shader(device, shader.load_shader());
                    ShaderModuleComputePipelines {
                        shader: Arc::new(s),
                        pipelines: HashMap::new(),
                    }
                });
        
        let pipelines = &mut s.pipelines;
        if !pipelines.contains_key(&pipeline) {
            let p = load_pipeline(device, s.shader.clone(), pipeline, pipeline_layout, consts);
            pipelines.insert(pipeline.clone(), Arc::new(p));
        }

        if let Some(p) = pipelines.get(&pipeline) {
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
    pipeline: &PipelineType,
    pipeline_layout: &wgpu::PipelineLayout,
    consts: &HashMap<String, f64>,
) -> wgpu::ComputePipeline {
    let entry_point = pipeline.0.get_entry_point();
    let compilation_options = if consts.is_empty() {
        wgpu::PipelineCompilationOptions::default()
    } else {
        wgpu::PipelineCompilationOptions {
            constants: &consts,
            zero_initialize_workgroup_memory: true,
            vertex_pulling_transform: false,
        }
    };
    return device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        module: &shader,
        entry_point: entry_point,
        compilation_options: compilation_options,
        cache: None,
    });
}
