
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, test_utils, Device, IndexOp, Result, Tensor};
async fn avg_pool2d(dev: &Device) -> Result<()> {
    let data: Vec<f32> = vec![
        1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), dev)?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::< f32 > (). await ?, [[0.5f32, 1.], [1., 1.]]);
    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 2, 8), dev)?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(
        pool.to_vec2_async::< f32 > (). await ?, [[5. / 4., 6. / 4., 6. / 4., 1.]]
    );
    Ok(())
}
async fn max_pool2d(dev: &Device) -> Result<()> {
    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), dev)?;
    let pool = t.max_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::< f32 > (). await ?, [[2f32, 3.], [5., 1.]]);
    let t = t.reshape((1, 1, 2, 8))?;
    let pool = t.max_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::< f32 > (). await ?, [[2.0, 3.0, 5.0, 1.0]]);
    Ok(())
}
async fn avg_pool2d_pytorch(dev: &Device) -> Result<()> {
    if dev.is_metal() {
        return Ok(());
    }
    let t = Tensor::new(
            &[
                0.4056f32,
                -0.8689,
                -0.0773,
                -1.5630,
                -2.8012,
                -1.5059,
                0.3972,
                1.0852,
                0.4997,
                3.0616,
                1.6541,
                0.0964,
                -0.8338,
                -1.6523,
                -0.8323,
                -0.1699,
                0.0823,
                0.3526,
                0.6843,
                0.2395,
                1.2279,
                -0.9287,
                -1.7030,
                0.1370,
                0.6047,
                0.3770,
                -0.6266,
                0.3529,
                2.2013,
                -0.6836,
                0.2477,
                1.3127,
            ],
            dev,
        )?
        .reshape((1, 2, 4, 4))?;
    if !dev.is_wgpu() {
        let pool = t.avg_pool2d(2)?.squeeze(0)?;
        assert_eq!(
            to_vec3_round_async(& pool, 4). await ?, [[[- 1.1926, - 0.0395], [0.2688,
            0.1871]], [[0.1835, - 0.1606], [0.6249, 0.3217]]]
        );
    }
    let pool = t.avg_pool2d(3)?.squeeze(0)?;
    assert_eq!(to_vec3_round_async(& pool, 4). await ?, [[[0.085]], [[0.0078]]]);
    let t = t.reshape((1, 1, 4, 8))?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(
        to_vec2_round_async(& pool, 4). await ?, [[0.7745, 0.0276, - 1.6983, 0.12],
        [0.3542, 0.1625, 0.4542, - 0.0014]]
    );
    Ok(())
}
async fn upsample_nearest2d(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 6f32, dev)?.reshape((1, 1, 2, 3))?;
    let upsampled = t.upsample_nearest2d(4, 6)?.i(0)?.i(0)?;
    assert_eq!(
        t.i(0) ?.i(0) ?.to_vec2_async::< f32 > (). await ?, [[0.0, 1.0, 2.0], [3.0, 4.0,
        5.0]]
    );
    assert_eq!(
        upsampled.to_vec2_async::< f32 > (). await ?, [[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0, 5.0, 5.0], [3.0, 3.0, 4.0,
        4.0, 5.0, 5.0]]
    );
    Ok(())
}
async fn upsample_nearest1d(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 3f32, dev)?.reshape((1, 1, 3))?;
    let upsampled = t.upsample_nearest1d(6)?.i(0)?.i(0)?;
    assert_eq!(t.i(0) ?.i(0) ?.to_vec1_async::< f32 > (). await ?, [0.0, 1.0, 2.0]);
    assert_eq!(
        upsampled.to_vec1_async::< f32 > (). await ?, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    );
    Ok(())
}
candle_wasm_tests::test_device!(
    avg_pool2d, avg_pool2d_cpu, avg_pool2d_gpu, avg_pool2d_metal, avg_pool2d_wgpu
);
candle_wasm_tests::test_device!(
    avg_pool2d_pytorch, avg_pool2d_pytorch_cpu, avg_pool2d_pytorch_gpu,
    avg_pool2d_pytorch_metal, avg_pool2d_pytorch_wgpu
);
candle_wasm_tests::test_device!(
    max_pool2d, max_pool2d_cpu, max_pool2d_gpu, max_pool2d_metal, max_pool2d_wgpu
);
candle_wasm_tests::test_device!(
    upsample_nearest1d, upsample_nearest1d_cpu, upsample_nearest1d_gpu,
    upsample_nearest1d_metal, upsample_nearest1d_wgpu
);
candle_wasm_tests::test_device!(
    upsample_nearest2d, upsample_nearest2d_cpu, upsample_nearest2d_gpu,
    upsample_nearest2d_metal, upsample_nearest2d_wgpu
);
