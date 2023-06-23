use crate::nn::layers::{Embedding, LayerNorm, LinearT, UnbiasedLinear};
use crate::Result;
use crate::{DType, Device, Tensor};

/// A special structure handy for Past Key values for text-generation
pub struct PastKeyValue {
    /// The cached key tensor. Shape is `NUM_HEADS, PAST_SEQUENCE_LENGTH, HEAD_DIM`.
    pub key: Tensor,
    /// The cached value tensor. Shape is `NUM_HEADS, PAST_SEQUENCE_LENGTH, HEAD_DIM`.
    pub value: Tensor,
}

impl PastKeyValue {
    pub fn new(
        num_heads: usize,
        past_sequence_length: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let key = Tensor::zeros(
            (num_heads, past_sequence_length, head_dim),
            DType::F32,
            device,
        )?;
        let value = Tensor::zeros(
            (num_heads, past_sequence_length, head_dim),
            DType::F32,
            device,
        )?;
        Ok(Self { key, value })
    }
}

pub type PastKeyValues = Vec<PastKeyValue>;

pub struct Gpt2Attention {
    qkv: LinearT,
    dense: LinearT,
    i: usize,
}

impl Gpt2Attention {
    pub fn new(qkv: LinearT, dense: LinearT, i: usize) -> Self {
        Self { qkv, dense, i }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.qkv.forward(hidden_states)?;

        // TODO
        // attention(&self.qkv, self.i)?;
        //
        let dense = self.dense.forward(&hidden_states);
        hidden_states + dense
    }
}

pub struct Mlp {
    c_fc: LinearT,
    c_proj: LinearT,
}

impl Mlp {
    pub fn new(c_fc: LinearT, c_proj: LinearT) -> Self {
        Self { c_fc, c_proj }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let intermediary_states = self.c_fc.forward(hidden_states)?;
        // let intermediary_states = intermediary_states.gelu()?;
        todo!("MLP gelu");
        // let hidden_states = self.c_proj.forward(intermediary_states)?;
        // Ok(hidden_states)
    }
}

pub struct Gpt2Layer {
    attention: Gpt2Attention,
    mlp: Mlp,
    ln_1: LayerNorm,
    ln_2: LayerNorm,
}

impl Gpt2Layer {
    pub fn new(attention: Gpt2Attention, mlp: Mlp, ln_1: LayerNorm, ln_2: LayerNorm) -> Self {
        Self {
            attention,
            mlp,
            ln_1,
            ln_2,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let layered_states = self.ln_1.forward(hidden_states)?;
        let attention_weights = self.attention.forward(&layered_states)?;
        let hidden_states = (hidden_states + attention_weights)?;
        let layered_states = self.ln_2.forward(&hidden_states)?;
        let output = self.mlp.forward(&layered_states)?;
        let hidden_states = hidden_states.add(&output)?;
        Ok(hidden_states)
    }
}

pub struct Gpt2Model {
    layers: Vec<Gpt2Layer>,
}

impl Gpt2Model {
    pub fn new(layers: Vec<Gpt2Layer>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }
}

pub struct Gpt2 {
    wte: Embedding,
    wpe: Embedding,
    h: Gpt2Model,
    ln_f: LayerNorm,
    lm_head: UnbiasedLinear,
    num_heads: usize,
}

impl Gpt2 {
    pub fn new(
        wte: Embedding,
        wpe: Embedding,
        h: Gpt2Model,
        ln_f: LayerNorm,
        lm_head: UnbiasedLinear,
        num_heads: usize,
    ) -> Self {
        Self {
            h,
            ln_f,
            wte,
            wpe,
            lm_head,
            num_heads,
        }
    }

    pub fn forward(&self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // if input_ids.len() != position_ids.len() {
        //     return Err(Error::InvalidLength {
        //         expected: input_ids.len(),
        //         got: position_ids.len(),
        //     });
        // }
        let embeddings = self.wte.forward(input_ids)?;
        let position_embeddings = self.wpe.forward(position_ids)?;

        let hidden_states = embeddings.add(&position_embeddings)?;
        let hidden_states = self.h.forward(&hidden_states)?;
        let hidden_states = self.ln_f.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok(logits)
    }

    // pub fn run(&self, input_ids: Vec<usize>, new_tokens: usize) -> Result<Vec<usize>> {
    //     #[cfg(feature = "cuda")]
    //     profiler_start()?;
    //     let mut context = self.new_context(input_ids, self.num_heads)?;
    //     for _ in 0..new_tokens {
    //         let start = std::time::Instant::now();
    //         self.forward(&mut context)?;
    //         context.generate()?;
    //         // println!("past {:?}", context.past_key_values[0].key.shape());
    //         println!("Took {:?}", start.elapsed());
    //     }
    //     #[cfg(feature = "cuda")]
    //     profiler_stop()?;
    //     context.new_tokens()
    // }
}
