use candle::{DType, Device, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaModel};

/// A deliberately tiny config: the test exercises control flow, not numerics,
/// so the model is kept small enough to build and run in milliseconds on CPU.
fn tiny_config(num_hidden_layers: usize) -> Config {
    Config {
        hidden_size: 8,
        layer_norm_eps: 1e-5,
        attention_probs_dropout_prob: 0.0,
        hidden_dropout_prob: 0.0,
        num_attention_heads: 2,
        position_embedding_type: "absolute".to_string(),
        intermediate_size: 16,
        hidden_act: Activation::Gelu,
        num_hidden_layers,
        vocab_size: 32,
        max_position_embeddings: 16,
        type_vocab_size: 1,
        pad_token_id: 1,
    }
}

fn tiny_model(num_hidden_layers: usize) -> Result<(XLMRobertaModel, Tensor, Tensor, Tensor)> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let model = XLMRobertaModel::new(&tiny_config(num_hidden_layers), vb)?;

    let input_ids = Tensor::new(&[[2u32, 5, 7, 3]], &device)?;
    let attention_mask = Tensor::ones((1, 4), DType::F32, &device)?;
    let token_type_ids = Tensor::zeros((1, 4), DType::U32, &device)?;

    Ok((model, input_ids, attention_mask, token_type_ids))
}

/// A callback that never cancels must leave the pass untouched — this is the
/// path `forward` itself takes, so a regression here would break every caller.
#[test]
fn forward_with_cancel_runs_to_completion_when_not_cancelled() -> Result<()> {
    let (model, input_ids, attention_mask, token_type_ids) = tiny_model(2)?;

    let out = model.forward_with_cancel(
        &input_ids,
        &attention_mask,
        &token_type_ids,
        None,
        None,
        None,
        &|| false,
    )?;

    assert_eq!(out.dims3()?, (1, 4, 8));
    Ok(())
}

/// The point of the API: an already-started pass stops instead of running the
/// remaining layers.
#[test]
fn forward_with_cancel_stops_when_cancelled() -> Result<()> {
    let (model, input_ids, attention_mask, token_type_ids) = tiny_model(2)?;

    let out = model.forward_with_cancel(
        &input_ids,
        &attention_mask,
        &token_type_ids,
        None,
        None,
        None,
        &|| true,
    );

    assert!(out.is_err(), "a cancelled pass must not return a tensor");
    Ok(())
}

/// Cancellation is polled *between* layers, so it must be observed once per
/// layer — not once per call. Counting the polls is what proves the check sits
/// inside the loop rather than in front of it.
#[test]
fn cancellation_is_polled_once_per_layer() -> Result<()> {
    let layers = 5;
    let (model, input_ids, attention_mask, token_type_ids) = tiny_model(layers)?;

    let polls = std::sync::atomic::AtomicUsize::new(0);
    let count = || {
        polls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        false
    };

    model.forward_with_cancel(
        &input_ids,
        &attention_mask,
        &token_type_ids,
        None,
        None,
        None,
        &count,
    )?;

    assert_eq!(polls.load(std::sync::atomic::Ordering::Relaxed), layers);
    Ok(())
}

/// `forward` must keep its exact previous behaviour: same signature, same
/// result as the cancellable variant with a callback that never fires.
#[test]
fn forward_matches_forward_with_cancel_without_cancellation() -> Result<()> {
    let (model, input_ids, attention_mask, token_type_ids) = tiny_model(2)?;

    let plain = model.forward(
        &input_ids,
        &attention_mask,
        &token_type_ids,
        None,
        None,
        None,
    )?;
    let cancellable = model.forward_with_cancel(
        &input_ids,
        &attention_mask,
        &token_type_ids,
        None,
        None,
        None,
        &|| false,
    )?;

    assert_eq!(plain.dims(), cancellable.dims());
    let diff = (plain - cancellable)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert_eq!(diff, 0.0);
    Ok(())
}
