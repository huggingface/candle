use candle_core::{ggml, Result};
use ggml::GgmlType;

#[test]
fn ggml_matmul() -> Result<()> {
    let (m, k, n) = (3, 32, 4);
    let lhs = (0..(m * k)).map(|v| v as f32).collect::<Vec<_>>();
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![ggml::BlockQ4_0::zeros(); 4];
    let rhs = (0..(k * n)).map(|v| v as f32).collect::<Vec<_>>();
    ggml::BlockQ4_0::from_float(&rhs, &mut rhs_t)?;
    ggml::matmul((m, k, n), &lhs, &rhs_t, &mut dst)?;
    assert_eq!(dst, &[0f32]);
    Ok(())
}
