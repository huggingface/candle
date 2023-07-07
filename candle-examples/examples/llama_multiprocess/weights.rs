use super::*;
use candle::{safetensors::SafeTensors, Device, Result, Tensor};
use safetensors::slice::IndexOp;
use std::path::PathBuf;

pub struct VarBuilder<'a> {
    routing: HashMap<String, usize>,
    safetensors: Vec<SafeTensors<'a>>,
    device: Device,
    comm: Rc<Comm>,
}

impl<'a> VarBuilder<'a> {
    pub fn new(safetensors: Vec<SafeTensors<'a>>, device: Device, comm: Rc<Comm>) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }

        Self {
            safetensors,
            device,
            routing,
            comm,
        }
    }

    pub fn comm(&self) -> Rc<Comm> {
        self.comm.clone()
    }

    pub fn get(&self, tensor_name: &str) -> Result<Tensor> {
        // Unwrap or 0  just to let the proper error flow.
        let index = self.routing.get(tensor_name).unwrap_or(&0);
        self.safetensors[*index]
            .tensor(tensor_name, &self.device)?
            .to_dtype(DTYPE)
    }

    pub fn get_sharded(&self, tensor_name: &str, dim: usize) -> Result<Tensor> {
        // Unwrap or 0  just to let the proper error flow.
        let index = self.routing.get(tensor_name).unwrap_or(&0);

        let world_size = self.comm.world_size();
        let rank = self.comm.rank();
        let view = self.safetensors[*index].view(tensor_name)?;
        let mut shape = view.shape().to_vec();
        let size = shape[dim];
        let block_size = size / world_size;
        let start = rank * block_size;
        let stop = (rank + 1) * block_size;

        let iterator = if dim == 0 {
            view.slice(start..stop).unwrap()
        } else if dim == 1 {
            view.slice((.., start..stop)).unwrap()
        } else {
            unimplemented!("Get sharded on dimensions != 0 or 1");
        };

        shape[dim] = block_size;

        self.safetensors[*index]
            .slice(tensor_name, iterator, &shape, &self.device)?
            .to_dtype(DTYPE)
    }
}

impl Linear {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(&format!("{prefix}.weight"))?;
        Ok(Self::new(weight))
    }

    // fn load_multi(prefixes: &[&str], vb: &VarBuilder) -> Result<Self> {
    //     let weights: Vec<_> = prefixes
    //         .iter()
    //         .map(|p| vb.get(&format!("{p}.weight")).unwrap())
    //         .collect();
    //     let weight = Tensor::cat(&weights, 0)?;
    //     Ok(Self::new(weight))
    // }
}

impl TensorParallelColumnLinear {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get_sharded(&format!("{prefix}.weight"), 0)?;
        Ok(Self::new(Linear::new(weight)))
    }

    fn load_multi(prefixes: &[&str], vb: &VarBuilder) -> Result<Self> {
        let weights: Vec<_> = prefixes
            .iter()
            .map(|p| vb.get_sharded(&format!("{p}.weight"), 0).unwrap())
            .collect();
        let weight = Tensor::cat(&weights, 0)?;
        Ok(Self::new(Linear::new(weight)))
    }
}

impl TensorParallelRowLinear {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get_sharded(&format!("{prefix}.weight"), 1)?;
        let comm = vb.comm();
        Ok(Self::new(Linear::new(weight), comm))
    }
}

impl RmsNorm {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let scale = vb.get(&format!("{prefix}.weight"))?;
        Ok(Self::new(scale))
    }
}

impl CausalSelfAttention {
    fn load(prefix: &str, vb: &VarBuilder, cache: &Cache, config: &Config) -> Result<Self> {
        let c_attn = TensorParallelColumnLinear::load_multi(
            &[
                &format!("{prefix}.q_proj"),
                &format!("{prefix}.k_proj"),
                &format!("{prefix}.v_proj"),
            ],
            vb,
        )?;
        let o_proj = TensorParallelRowLinear::load(&format!("{prefix}.o_proj"), vb)?;
        Ok(Self::new(
            c_attn,
            o_proj,
            config.n_head / vb.comm.world_size(),
            cache,
        ))
    }
}

impl Mlp {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let c_fc1 = TensorParallelColumnLinear::load(&format!("{prefix}.gate_proj"), vb)?;
        let c_fc2 = TensorParallelColumnLinear::load(&format!("{prefix}.up_proj"), vb)?;
        let c_proj = TensorParallelRowLinear::load(&format!("{prefix}.down_proj"), vb)?;
        Ok(Self::new(c_fc1, c_fc2, c_proj))
    }
}

impl Block {
    fn load(prefix: &str, vb: &VarBuilder, cache: &Cache, config: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(&format!("{prefix}.self_attn"), vb, cache, config)?;
        let mlp = Mlp::load(&format!("{prefix}.mlp"), vb)?;
        let input_layernorm = RmsNorm::load(&format!("{prefix}.input_layernorm"), vb)?;
        let post_attention_layernorm =
            RmsNorm::load(&format!("{prefix}.post_attention_layernorm"), vb)?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

impl Llama {
    pub fn load(
        device: &Device,
        filenames: &[PathBuf],
        cache: &Cache,
        config: &Config,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let handles: Vec<_> = filenames
            .iter()
            .map(|f| unsafe { candle::safetensors::MmapedFile::new(f) })
            .collect::<Result<Vec<_>>>()?;
        let tensors: Vec<_> = handles
            .iter()
            .map(|h| h.deserialize())
            .collect::<Result<Vec<_>>>()?;

        let vb = VarBuilder::new(tensors, device.clone(), comm);

        let embedding = vb.get("model.embed_tokens.weight")?;
        let wte = Embedding::new(embedding);
        let lm_head = Linear::load("lm_head", &vb)?;
        let norm = RmsNorm::load("model.norm", &vb)?;
        let blocks: Vec<_> = (0..config.n_layer)
            .map(|i| Block::load(&format!("model.layers.{i}"), &vb, cache, config).unwrap())
            .collect();

        Ok(Self::new(wte, blocks, norm, lm_head))
    }
}
