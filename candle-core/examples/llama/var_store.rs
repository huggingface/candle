use super::*;
use candle::{DType, Device, Result, Shape, Tensor, WithDType};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[allow(dead_code)]
#[derive(Clone)]
struct NamedVar {
    path: String,
    dtype: DType,
    shape: Shape,
}

#[derive(Clone)]
pub struct VarBuilder {
    path: Vec<String>,
    vars: std::rc::Rc<std::cell::RefCell<Vec<NamedVar>>>,
    default_dtype: DType,
    default_device: Device,
    tensors: Arc<Option<HashMap<String, Tensor>>>,
}

#[allow(dead_code)]
pub struct VarStore {
    vars: Vec<NamedVar>,
}

impl VarBuilder {
    pub fn new<B: WithDType>(device: &Device, tensors: Option<HashMap<String, Tensor>>) -> Self {
        let vars = std::rc::Rc::new(std::cell::RefCell::new(vec![]));
        Self {
            path: vec![],
            vars,
            default_dtype: B::DTYPE,
            tensors: Arc::new(tensors),
            default_device: device.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.vars.borrow().len()
    }

    pub fn var(&self, s: &str) -> Result<Tensor> {
        let path = format!("{}.{s}", self.path.join("."));
        let parameter = match self.tensors.as_ref() {
            None => panic!("Cannot find tensors"),
            Some(tensors) => match tensors.get(&path) {
                Some(tensor) => tensor.to_device(&self.default_device)?,
                None => panic!("cannot find tensor for {path}"),
            },
        };
        Ok(parameter)
    }

    pub fn into_store(self) -> VarStore {
        let vars = self.vars.borrow();
        VarStore {
            vars: vars.to_vec(),
        }
    }
}

impl<S: ToString> std::ops::Div<S> for &VarBuilder {
    type Output = VarBuilder;

    fn div(self, rhs: S) -> VarBuilder {
        let mut path = self.path.clone();
        path.push(rhs.to_string());
        VarBuilder {
            path,
            vars: self.vars.clone(),
            default_dtype: self.default_dtype,
            default_device: self.default_device.clone(),
            tensors: self.tensors.clone(),
        }
    }
}

impl<S: ToString> std::ops::Div<S> for VarBuilder {
    type Output = VarBuilder;

    fn div(self, rhs: S) -> VarBuilder {
        &self / rhs
    }
}

impl Embedding {
    fn load_npy(vb: VarBuilder) -> Result<Self> {
        let embeddings = vb.var("weight")?;
        Ok(Self { embeddings })
    }
}

impl Linear {
    fn load_npy(vb: VarBuilder) -> Result<Self> {
        let weight = vb.var("weight")?.t()?;
        Ok(Self { weight })
    }
}

impl RmsNorm {
    fn load_npy(vb: VarBuilder) -> Result<Self> {
        let scale = vb.var("scale")?;
        Ok(Self::new(scale))
    }
}

impl CausalSelfAttention {
    fn load_npy(vb: VarBuilder, cache: &Cache, config: &Config) -> Result<Self> {
        let c_attn = Linear::load_npy(&vb / "c_attn")?;
        let c_proj = Linear::load_npy(&vb / "c_proj")?;
        Ok(Self::new(c_attn, c_proj, config.n_head, cache))
    }
}

impl Mlp {
    fn load_npy(vb: VarBuilder) -> Result<Self> {
        let c_fc1 = Linear::load_npy(&vb / "c_fc1")?;
        let c_fc2 = Linear::load_npy(&vb / "c_fc2")?;
        let c_proj = Linear::load_npy(&vb / "c_proj")?;
        Ok(Self::new(c_fc1, c_fc2, c_proj))
    }
}

impl Block {
    fn load_npy(vb: VarBuilder, cache: &Cache, config: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load_npy(&vb / "attn", cache, config)?;
        let mlp = Mlp::load_npy(&vb / "mlp")?;
        let input_layernorm = RmsNorm::load_npy(&vb / "rms_1")?;
        let post_attention_layernorm = RmsNorm::load_npy(&vb / "rms_2")?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

impl Llama {
    pub fn load_npy(
        device: &Device,
        _filenames: &[PathBuf],
        cache: &Cache,
        config: &Config,
    ) -> Result<Self> {
        let weight_path = std::path::Path::new("/data/llama.npz");
        let weights = if weight_path.exists() {
            println!("loading weights from {weight_path:?}");
            let start_load = std::time::Instant::now();
            let tensors = Tensor::read_npz(weight_path)?;
            println!("loaded weights in {:?}", start_load.elapsed());
            let tensors: std::collections::HashMap<String, Tensor> = tensors.into_iter().collect();
            Some(tensors)
        } else {
            println!("cannot find {weight_path:?}, using zero weights");
            None
        };
        let vb = VarBuilder::new::<f32>(device, weights);

        let wte = Embedding::load_npy(&vb / "transformer" / "wte")?;
        let lm_head = Linear::load_npy(&vb / "lm_head")?;
        let norm = RmsNorm::load_npy(&vb / "transformer" / "ln_f")?;
        let blocks: Vec<_> = (0..config.n_layer)
            .map(|i| Block::load_npy(&vb / "transformer" / "h" / i, cache, config).unwrap())
            .collect();

        Ok(Self::new(wte, blocks, norm, lm_head))
    }
}
