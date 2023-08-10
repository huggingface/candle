mod test_utils;
use candle_core::{Device, IndexOp, Tensor};

// https://github.com/huggingface/candle/issues/364
#[test]
fn avg_pool2d() -> anyhow::Result<()> {
    let data: Vec<f32> = vec![
        1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), &Device::Cpu)?;

    let pool = t.avg_pool2d((2, 2), (2, 2))?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2::<f32>()?, [[0.5f32, 1.], [1., 1.]]);
    Ok(())
}

#[test]
fn max_pool2d() -> anyhow::Result<()> {
    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), &Device::Cpu)?;

    let pool = t.max_pool2d((2, 2), (2, 2))?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2::<f32>()?, [[2f32, 3.], [5., 1.]]);
    Ok(())
}

/* This test corresponds to the following PyTorch script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 2, 4, 4))
print(t.flatten())
res = torch.nn.functional.avg_pool2d(t, 2)
print(res)
*/
#[test]
fn avg_pool2d_pytorch() -> anyhow::Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127,
        ],
        &Device::Cpu,
    )?
    .reshape((1, 2, 4, 4))?;
    let pool = t.avg_pool2d((2, 2), (2, 2))?.squeeze(0)?;
    assert_eq!(
        test_utils::to_vec3_round(pool, 4)?,
        [
            [[-1.1926, -0.0395], [0.2688, 0.1871]],
            [[0.1835, -0.1606], [0.6249, 0.3217]]
        ]
    );
    let pool = t.avg_pool2d((3, 3), (3, 3))?.squeeze(0)?;
    assert_eq!(test_utils::to_vec3_round(pool, 4)?, [[[0.085]], [[0.0078]]]);
    Ok(())
}

#[test]
fn upsample_nearest2d() -> anyhow::Result<()> {
    let t = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((1, 1, 2, 3))?;
    let upsampled = t.upsample_nearest2d(4, 6)?.i(0)?.i(0)?;
    assert_eq!(
        t.i(0)?.i(0)?.to_vec2::<f32>()?,
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    );
    assert_eq!(
        upsampled.to_vec2::<f32>()?,
        [
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            [3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
        ]
    );
    Ok(())
}

#[test]
fn downsample_nearest2d() -> anyhow::Result<()> {
    let t = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((1, 1, 2, 3))?;
    let upsampled = t.upsample_nearest2d(4, 6)?;
    let downsampled = upsampled.upsample_nearest2d(2, 3)?;
    assert_eq!(
        downsampled.i(0)?.i(0)?.to_vec2::<f32>()?,
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    );
    Ok(())
}