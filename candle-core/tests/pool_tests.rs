mod test_utils;
use candle_core::{Device, Tensor};

// https://github.com/huggingface/candle/issues/364
#[test]
fn avg_pool2d() -> anyhow::Result<()> {
    let device = Device::Cpu;

    let data: Vec<f32> = vec![
        1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), &device)?;

    let pool = t.avg_pool2d((2, 2), (2, 2))?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2::<f32>()?, [[0.5f32, 1.], [1., 1.]]);
    Ok(())
}
