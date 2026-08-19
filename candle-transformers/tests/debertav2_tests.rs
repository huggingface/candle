// Regression test for #3750: F16 forward failed with a dtype mismatch from
// hardcoded F32 scalars.
use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{Config, DebertaV2Model};

fn small_config() -> Config {
    serde_json::from_value(serde_json::json!({
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 32,
        "type_vocab_size": 0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-7,
        "relative_attention": true,
        "max_relative_positions": -1,
        "pad_token_id": 0,
        "position_biased_input": true,
        "pos_att_type": ["p2c", "c2p"],
        "position_buckets": 8,
        "norm_rel_ebd": "layer_norm"
    }))
    .unwrap()
}

fn forward_with_dtype(dtype: DType) -> Result<DType> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(dtype, &device);
    let model = DebertaV2Model::load(vb, &small_config())?;
    let input_ids = Tensor::zeros((1, 6), DType::U32, &device)?;
    let attention_mask = Tensor::new(&[[1u32, 1, 1, 1, 0, 0]], &device)?;
    let output = model.forward(&input_ids, Some(attention_mask), None)?;
    let values = output
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(values.iter().all(|v| v.is_finite()));
    Ok(output.dtype())
}

#[test]
fn forward_f32() -> Result<()> {
    assert_eq!(forward_with_dtype(DType::F32)?, DType::F32);
    Ok(())
}

#[test]
fn forward_f16() -> Result<()> {
    assert_eq!(forward_with_dtype(DType::F16)?, DType::F16);
    Ok(())
}
