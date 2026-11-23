use candle::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::stable_diffusion::attention::CrossAttention;

/// The sliced attention path must give the same result as the full attention path.
/// Before the fix, `sliced_attention` stacked the slices along a new axis and the
/// following `reshape_batch_dim_to_heads` failed with "unexpected rank, expected: 3, got: 4".
#[test]
fn sliced_attention_matches_full_attention() -> Result<()> {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let (query_dim, heads, dim_head, seq_len) = (16, 4, 4, 6);
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let full = CrossAttention::new(vb.clone(), query_dim, None, heads, dim_head, None, false)?;
    let sliced = CrossAttention::new(vb, query_dim, None, heads, dim_head, Some(1), false)?;
    let xs = Tensor::randn(0f32, 1f32, (2, seq_len, query_dim), &dev)?;

    let a = full.forward(&xs, None)?;
    let b = sliced.forward(&xs, None)?;
    assert_eq!(a.dims(), b.dims());
    let diff = (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(diff < 1e-5, "sliced and full attention differ by {diff}");
    Ok(())
}
