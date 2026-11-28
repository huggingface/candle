use candle::{DType, Device, Result, Tensor};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::gemma3::{Config, Model};

// Forwarding several tokens once the sliding-window kv-cache is full (a second prompt, or a
// chunked prefill) must give the same logits as feeding the same tokens one at a time.
#[test]
fn multi_token_forward_past_sliding_window() -> Result<()> {
    let device = Device::Cpu;
    // A single sliding layer with a 4-token window. With more layers the two paths differ: the
    // earlier tokens of a chunk lose the cache entries that the chunk itself overwrites.
    let cfg = Config {
        attention_bias: false,
        head_dim: 8,
        hidden_activation: Activation::GeluPytorchTanh,
        hidden_size: 16,
        intermediate_size: 32,
        num_attention_heads: 2,
        num_hidden_layers: 1,
        num_key_value_heads: 1,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.,
        rope_local_base_freq: 10000.,
        vocab_size: 32,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: 8,
        sliding_window: 4,
        sliding_window_pattern: 2,
        max_position_embeddings: 64,
    };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut chunked = Model::new(false, &cfg, vb.clone())?;
    let mut one_by_one = Model::new(false, &cfg, vb)?;
    let tokens = Tensor::new(&[[3u32, 8, 13, 18, 23, 28, 1, 6]], &device)?;
    // Two more tokens on a cache that is already full, or that wraps around during the chunk.
    for prompt_len in [6, 3] {
        chunked.clear_kv_cache();
        one_by_one.clear_kv_cache();
        let prompt = tokens.narrow(1, 0, prompt_len)?;
        chunked.forward(&prompt, 0)?;
        one_by_one.forward(&prompt, 0)?;
        let logits = chunked.forward(&tokens.narrow(1, prompt_len, 2)?, prompt_len)?;
        one_by_one.forward(&tokens.narrow(1, prompt_len, 1)?, prompt_len)?;
        let next = prompt_len + 1;
        let expected = one_by_one.forward(&tokens.narrow(1, next, 1)?, next)?;
        let diff = (logits - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "prompt of {prompt_len} tokens: diff {diff}");
    }
    Ok(())
}
