use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};

fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

fn fa_acausal(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = att.softmax(D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

#[test]
fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 24, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 4))?;
    let k = (&q / 20.)?;
    let v = (&q / 25.)?;
    let q = (&q / 15.)?;

    let ys = fa_acausal(&q, &k, &v, 0.5)?;
    let ys = ys.i(0)?.to_dtype(DType::F32)?;
    assert_eq!(ys.dims(), &[3, 2, 4]);
    assert_eq!(
        to_vec3_round(ys, 4)?,
        &[
            [
                [0.0816, 0.1216, 0.1616, 0.2017],
                [0.0859, 0.1259, 0.1659, 0.2058]
            ],
            [
                [0.4102, 0.4502, 0.4902, 0.5303],
                [0.4143, 0.4543, 0.4944, 0.5342]
            ],
            [
                [0.7388, 0.7788, 0.8188, 0.8589],
                [0.7427, 0.7827, 0.8228, 0.8628]
            ]
        ]
    );

    let ys = candle_flash_attn::flash_attn(&q, &k, &v, 0.5, false)?;
    let ys = ys.i(0)?.to_dtype(DType::F32)?;
    assert_eq!(ys.dims(), &[3, 2, 4]);
    assert_eq!(to_vec3_round(ys, 4)?, &[[[0f32]]]);
    Ok(())
}
