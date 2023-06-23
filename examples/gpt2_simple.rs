use anyhow::Result;
use candle::{Device, Tensor};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;

    //wget https://huggingface.co/gpt2/raw/main/tokenizer.json
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let encoded = tokenizer.encode("What is deep learning ?", true).unwrap();

    let ids = encoded.get_ids();
    assert_eq!(ids, &[2061, 318, 2769, 4673, 5633]);
    let input_ids = Tensor::from_slice(ids, (1, ids.len()), &device)?;

    let position_ids: Vec<_> = (0..ids.len() as u32).collect();
    let position_ids = Tensor::from_slice(&position_ids, (1, position_ids.len()), &device)?;

    // XXX: This is highly gpt2 specific, before we actually load configurations.
    let num_heads = 12;
    //wget https://huggingface.co/gpt2/resolve/main/model.safetensors
    let gpt2 = load::load("model.safetensors", &Device::Cpu, num_heads)?;
    let logits = gpt2.forward(&input_ids, &position_ids)?;
    // let id = logits.argmax(-1)?;
    todo!("Argmax {:?}", logits);
    // let token = tokenizer.decode(&[id], true).unwrap();
    // assert_eq!(token, "");
    // Ok(())
}

mod load {
    use super::*;
    use candle::nn::{
        layers::{Embedding, LayerNorm, LinearT, UnbiasedLinear},
        models::gpt2::{Gpt2, Gpt2Attention, Gpt2Layer, Gpt2Model, Mlp},
    };
    use memmap2::MmapOptions;
    use safetensors::tensor::{Dtype, SafeTensors, TensorView};
    use std::borrow::Cow;
    use std::fs::File;

    fn to_tensor(view: TensorView<'_>, device: &Device) -> Result<Tensor> {
        let shape = view.shape().to_vec();
        let data = to_f32(view);
        Ok(Tensor::from_slice(&data, shape.as_slice(), device)?)
    }

    fn to_f32(view: TensorView) -> Cow<'static, [f32]> {
        assert_eq!(view.dtype(), Dtype::F32);
        let v = view.data();
        if (v.as_ptr() as usize) % 4 == 0 {
            // SAFETY This is safe because we just checked that this
            // was correctly aligned.
            let data: &[f32] =
                unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) };
            Cow::Borrowed(data)
        } else {
            let mut c = Vec::with_capacity(v.len() / 4);
            let mut i = 0;
            while i < v.len() {
                c.push(f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]));
                i += 4;
            }
            Cow::Owned(c)
        }
    }

    fn embedding_from(weights: TensorView<'_>, device: &Device) -> Result<Embedding> {
        Ok(Embedding::new(to_tensor(weights, device)?))
    }

    fn unbiased_linear_from(weights: TensorView<'_>, device: &Device) -> Result<UnbiasedLinear> {
        Ok(UnbiasedLinear::new(to_tensor(weights, device)?))
    }

    fn layer_norm_from_prefix(
        prefix: &str,
        tensors: &SafeTensors<'_>,
        device: &Device,
    ) -> LayerNorm {
        let epsilon = 1e-5;
        if let (Ok(weight), Ok(bias)) = (
            tensors.tensor(&format!("{}.weight", prefix)),
            tensors.tensor(&format!("{}.bias", prefix)),
        ) {
            LayerNorm::new(
                to_tensor(weight, device).unwrap(),
                to_tensor(bias, device).unwrap(),
                epsilon,
            )
        } else {
            LayerNorm::new(
                to_tensor(
                    tensors.tensor(&format!("{}.gamma", prefix)).unwrap(),
                    device,
                )
                .unwrap(),
                to_tensor(tensors.tensor(&format!("{}.beta", prefix)).unwrap(), device).unwrap(),
                epsilon,
            )
        }
    }

    fn gpt2_layer_from_tensors(
        index: usize,
        tensors: &SafeTensors<'_>,
        device: &Device,
    ) -> Result<Gpt2Layer> {
        let ln_1 = layer_norm_from_prefix(&format!("h.{index}.ln_1"), tensors, device);
        let ln_2 = layer_norm_from_prefix(&format!("h.{index}.ln_2"), tensors, device);
        let attention = gpt2_attention_from_tensors(index, tensors, device)?;
        let mlp = gpt2_mlp_from_tensors(index, tensors, device)?;
        Ok(Gpt2Layer::new(attention, mlp, ln_1, ln_2))
    }
    fn gpt2_attention_from_tensors(
        index: usize,
        tensors: &SafeTensors<'_>,
        device: &Device,
    ) -> Result<Gpt2Attention> {
        let c_attn = linear_from_prefix(&format!("h.{index}.attn.c_attn"), tensors, device)?;
        let c_proj = linear_from_prefix(&format!("h.{index}.attn.c_proj"), tensors, device)?;
        Ok(Gpt2Attention::new(c_attn, c_proj, index))
    }

    fn gpt2_mlp_from_tensors(
        index: usize,
        tensors: &SafeTensors<'_>,
        device: &Device,
    ) -> Result<Mlp> {
        let c_fc = linear_from_prefix(&format!("h.{index}.mlp.c_fc"), tensors, device)?;
        let c_proj = linear_from_prefix(&format!("h.{index}.mlp.c_proj"), tensors, device)?;
        Ok(Mlp::new(c_fc, c_proj))
    }
    fn model_from_tensors(tensors: &SafeTensors<'_>, device: &Device) -> Result<Gpt2Model> {
        // TODO ! Count heads from tensors present
        let layers: Result<Vec<_>> = (0..12)
            .map(|i| gpt2_layer_from_tensors(i, tensors, device))
            .collect();
        Ok(Gpt2Model::new(layers?))
    }

    fn linear_from_prefix(
        prefix: &str,
        tensors: &SafeTensors<'_>,
        device: &Device,
    ) -> Result<LinearT> {
        let weights = tensors.tensor(&format!("{}.weight", prefix))?;
        let bias = tensors.tensor(&format!("{}.bias", prefix))?;
        Ok(LinearT::new(
            to_tensor(weights, device)?,
            to_tensor(bias, device)?,
        ))
    }

    pub fn load(filename: &str, device: &Device, num_heads: usize) -> Result<Gpt2> {
        let file = File::open(filename)?;
        let buffer = unsafe { MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&buffer)?;
        let wte = embedding_from(tensors.tensor("wte.weight")?, device)?;
        let wpe = embedding_from(tensors.tensor("wpe.weight")?, device)?;
        let h = model_from_tensors(&tensors, device)?;
        let ln_f = layer_norm_from_prefix("ln_f", &tensors, device);
        let lm_head = unbiased_linear_from(tensors.tensor("wte.weight")?, device)?;
        Ok(Gpt2::new(wte, wpe, h, ln_f, lm_head, num_heads))
    }
}
