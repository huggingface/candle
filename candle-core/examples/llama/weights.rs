use memmap2::MmapOptions;
use candle::{Device, Result, Shape, Tensor, WithDType};
use std::fs::File;
use std::path::PathBuf;
use super::*;
use safetensors::{SafeTensors, tensor::{Dtype, TensorView}};
use half::f16;

fn convert<'a>(view: TensorView<'a>, device: &Device) -> Result<Tensor>{
    match view.dtype(){
        Dtype::F16 => {
            let v = view.data();
            if (v.as_ptr() as usize) % 2 == 0 {
                // SAFETY This is safe because we just checked that this
                // was correctly aligned.
                let data: &[f16] =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f16, v.len() / 2) };
                Tensor::from_slice(data, view.shape(), device)
            } else {
                let mut c = Vec::with_capacity(v.len() / 2);
                let mut i = 0;
                while i < v.len() {
                    c.push(f16::from_le_bytes([v[i], v[i + 1]]));
                    i += 2;
                }
                Tensor::from_slice(&c, view.shape(), device)
            }

        }
        dt => todo!("Unhandled dtype {dt:?}")
    }
}

pub struct VarBuilder<'a>{
    routing: HashMap<String, usize>,
    safetensors: Vec<SafeTensors<'a>>,
    device: Device,
}


impl<'a> VarBuilder<'a>{
    pub fn new(safetensors: Vec<SafeTensors<'a>>, device: Device) -> Self{
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate(){
            for k in sf.names(){
                routing.insert(k.to_string(), index);
            }
        }

        Self{
            safetensors,
            device,
            routing
        }
    }

    pub fn get(&self, tensor_name: &str) -> Result<Tensor>{
        // Unwrap or 0  just to let the proper error flow.
        let index = self.routing.get(tensor_name).unwrap_or(&0);
        let view = self.safetensors[*index].tensor(tensor_name).unwrap();
        let tensor = convert(view, &self.device)?;
        Ok(tensor)

    }
}

impl Linear{
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self>{
        let weight = vb.get(&format!("{prefix}.weight"))?;
        Ok(Self::new(weight))
    }

    fn load_multi(prefixes: &[&str], vb: &VarBuilder) -> Result<Self>{
        let weights: Vec<_> = prefixes.iter().map(|p| vb.get(&format!("{p}.weight")).unwrap()).collect();
        println!("shapes {:?}", weights.iter().map(|w| w.shape()).collect::<Vec<_>>());
        let weight = Tensor::cat(&weights, 0)?;
        Ok(Self::new(weight))
    }
}

impl RmsNorm{
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self>{
        let scale = vb.get(&format!("{prefix}.weight"))?;
        Ok(Self::new(scale))
    }
}

impl CausalSelfAttention{
    fn load(prefix: &str, vb: &VarBuilder, cache: &Cache, config: &Config) -> Result<Self>{
        let c_attn = Linear::load_multi(&[&format!("{prefix}.q_proj"), &format!("{prefix}.k_proj"), &format!("{prefix}.v_proj")], vb)?;
        let o_proj = Linear::load(&format!("{prefix}.o_proj"), vb)?;
        Ok(Self::new(c_attn,o_proj, config.n_head, cache))
    }
}

impl Mlp{
    fn load(prefix: &str, vb: &VarBuilder, config: &Config) -> Result<Self>{
        let c_fc1 = Linear::load(&format!("{prefix}.gate_proj"), vb)?;
        let c_fc2 = Linear::load(&format!("{prefix}.up_proj"), vb)?;
        let c_proj = Linear::load(&format!("{prefix}.down_proj"), vb)?;
        Ok(Self::new(c_fc1, c_fc2, c_proj))
    }
}

impl Block{
    fn load(prefix: &str, vb: &VarBuilder, cache: &Cache, config: &Config) -> Result<Self>{
        let attn = CausalSelfAttention::load(&format!("{prefix}.self_attn"), vb, cache, config)?;
        let mlp = Mlp::load(&format!("{prefix}.mlp"), vb, config)?;
        let input_layernorm = RmsNorm::load(&format!("{prefix}.input_layernorm"), vb)?;
        let post_attention_layernorm = RmsNorm::load(&format!("{prefix}.post_attention_layernorm"), vb)?;
        Ok(Self::new(input_layernorm, attn, post_attention_layernorm, mlp))
    }
}

impl Llama{
    pub fn load(device: &Device, filenames: &[PathBuf], cache: &Cache, config: &Config) -> Result<Self>{
        let handles: Vec<_> = filenames.iter().map(|f| {
            let file = File::open(f).unwrap();
            let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
            buffer
        }).collect();
        let tensors: Vec<_> = handles.iter().map(|h| {
            let tensors = SafeTensors::deserialize(h).unwrap();
            tensors
        }).collect();

        let vb = VarBuilder::new(tensors, device.clone());

        let embedding = vb.get("model.embed_tokens.weight")?;
        let wte = Embedding::new(embedding);
        let lm_head = Linear::load("lm_head", &vb)?;
        let norm = RmsNorm::load("model.norm", &vb)?;
        let blocks: Vec<_> = (0..config.n_layer).map(|i| Block::load(&format!("model.layers.{i}"), &vb, cache, config).unwrap()).collect();

        Ok(Self::new(
            wte,
            blocks,
            norm,
            lm_head
        ))
    }
}


