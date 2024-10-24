#![allow(clippy::approx_constant)]

use candle::quantized::{self, k_quants, GgmlDType, GgmlType};
use candle::{bail, cpu_backend, CpuStorage, CustomOp1, DType, Device, IndexOp, Layout, Module, Shape, Tensor, Var, D};
use candle_test::{test_device, to_vec1_round, to_vec2_round, ToVecRound};
use candle_test as test_utils;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use candle::backend::BackendStorage;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use crate::Device::Cpu;

use candle::Result;

#[cfg(target_arch="wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[cfg(not(target_arch="wasm32"))]
use tokio::test as test;


/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5))
w = torch.randn((2, 4, 3))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv1d(t, w)
print(res.flatten())
res = torch.nn.functional.conv1d(t, w, padding=1)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose1d(t, w_t)
print(res.shape)
print(res)
res = torch.nn.functional.conv_transpose1d(t, w_t, groups=2)
print(res.shape)
print(res)
*/
async fn conv1d(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
            1.8025, -0.1536, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278, -1.0124, 0.5599,
        ],
        dev,
    )?
    .reshape((1, 4, 5))?;
    let w = Tensor::new(
        &[
            -0.8404f32, -0.3490, 0.0130, 1.3123, 0.1763, -1.9249, 1.4270, 0.9421, 0.8670, -0.7181,
            -1.1111, 0.8869, -1.2429, 1.8357, 1.6052, -1.3844, 0.3951, -1.2036, 0.6686, 1.6261,
            -0.6451, -0.0840, -1.4247, 0.5512,
        ],
        dev,
    )?
    .reshape((2, 4, 3))?;
    let res = t.conv1d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [2.6357, -1.3336, 4.1393, -1.1784, 3.5675, 0.5069]
    );
    let res = t.conv1d(&w, /*padding*/ 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 5]);
    // Same as pytorch default padding: use zeros.
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [2.4509, 2.6357, -1.3336, 4.1393, 0.5657, 1.8091, -1.1784, 3.5675, 0.5069, 3.3352]
    );

    let w = w.transpose(0, 1)?;
    // The CPU kernels applied in the contiguous and non contiguous cases are different.
    for w in [w.clone(), w.contiguous()?] {
        let res = t.conv_transpose1d(&w, 0, 0, 1, 1, 1)?;
        assert_eq!(res.dims(), [1, 2, 7]);
        assert_eq!(
            test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
            [
                0.0699, -1.2899, 8.3018, 5.5873, 2.4572, -2.6143, -0.0706, 1.8765, 4.8318, 1.1538,
                4.7076, -5.9745, -0.8276, 1.621
            ],
        );
        let res = t.conv_transpose1d(&w, 0, 0, 1, 1, 2)?;
        assert_eq!(res.dims(), [1, 4, 7]);
        assert_eq!(
            test_utils::to_vec2_round(&res.squeeze(0)?, 4).await?,
            [
                [-1.5596, -1.8099, 2.0407, 4.8764, -0.1743, -0.735, -0.7819],
                [0.7816, 3.8152, -0.5926, 2.2515, -5.1844, -0.3157, 1.4721],
                [1.6295, 0.52, 6.2611, 0.7109, 2.6315, -1.8793, 0.7113],
                [1.0949, 1.0166, 1.7464, 2.4561, -0.79, -0.5119, 0.1488]
            ]
        );
    }
    Ok(())
}

async fn conv1d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[0.4056f32, -0.8689, -0.0773, -1.5630], dev)?.reshape((1, 1, 4))?;
    let w = Tensor::new(&[1f32, 0., 0.], dev)?.reshape((1, 1, 3))?;
    let res = t.conv1d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [0.4056, -0.8689]
    );
    let res = t.conv1d(&w, /*padding*/ 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [0.0, 0.4056, -0.8689, -0.0773],
    );
    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5, 5))
w = torch.randn((2, 4, 3, 3))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t, w_t)
print(res.shape)
print(res)

res = torch.nn.functional.conv2d(t, w, dilation=2)
print(res.shape)
print(res[0])

res = torch.nn.functional.conv_transpose2d(t, w_t, dilation=2)
print(res.shape)
print(res)
*/
async fn conv2d(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130, 1.3123,
            1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071, 1.1586,
            0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090, 0.2049,
            0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323, -1.3712,
            0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742, 0.3790,
            -0.4431, -0.4720, -0.7890, 0.2620, 0.7875, 0.5377, -0.6779, -0.8088, 1.9098, 1.2006,
            -0.8, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
        ],
        dev,
    )?;
    let w = Tensor::new(
        &[
            -0.9325f32, 0.6451, -0.8537, 0.2378, 0.8764, -0.1832, 0.2987, -0.6488, -0.2273,
            -2.4184, -0.1192, -0.4821, -0.5079, -0.5766, -2.4729, 1.6734, 0.4558, 0.2851, 1.1514,
            -0.9013, 1.0662, -0.1817, -0.0259, 0.1709, 0.5367, 0.7513, 0.8086, -2.2586, -0.5027,
            0.9141, -1.3086, -1.3343, -1.5669, -0.1657, 0.7958, 0.1432, 0.3896, -0.4501, 0.1667,
            0.0714, -0.0952, 1.2970, -0.1674, -0.3178, 1.0677, 0.3060, 0.7080, 0.1914, 1.1679,
            -0.3602, 1.9265, -1.8626, -0.5112, -0.0982, 0.2621, 0.6565, 0.5908, 1.0089, -0.1646,
            1.8032, -0.6286, 0.2016, -0.3370, 1.2555, 0.8009, -0.6488, -0.4652, -1.5685, 1.5860,
            0.5583, 0.4623, 0.6026,
        ],
        dev,
    )?;
    let t = t.reshape((1, 4, 5, 5))?;
    let w = w.reshape((2, 4, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [
            -4.2812, 2.0923, 5.2187, 7.5184, 0.752, -14.9426, 10.0087, 4.391, 0.2918, 1.6715,
            10.389, 3.6023, -4.2808, 0.2672, 5.3646, -5.2023, -2.1955, -9.4075
        ]
    );

    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;

    assert_eq!(res.dims(), [1, 2, 7, 7]);
    assert_eq!(
        test_utils::to_vec3_round(&res.i(0)?, 4).await?,
        [
            [
                [-1.9918, 2.6797, -0.4599, -1.6037, 1.4131, -2.4012, 2.9277],
                [1.8016, -3.5361, 1.0757, 3.5395, -8.2168, -3.2023, 0.5375],
                [0.8243, 1.8675, 7.8929, -4.0746, -6.4415, 5.1139, 1.6889],
                [0.2722, 8.9679, 3.3477, 1.8514, -4.2896, -3.8228, -7.5632],
                [-8.5412, -5.8142, -7.1587, -1.6095, 0.4651, 0.2748, -2.0985],
                [2.0833, -0.6482, -12.1692, -4.1284, -2.9765, -0.0656, -4.5114],
                [5.307, 2.6957, 2.3087, 1.0478, 0.7808, -1.1519, -0.9579]
            ],
            [
                [1.089, 0.1872, -0.6408, -0.9897, 0.8503, 1.1019, -0.9211],
                [-0.1741, -0.2915, 4.2472, 1.9417, 1.65, 0.6303, -4.7131],
                [1.6555, 2.4026, -2.9293, 2.9953, 0.5328, 3.5873, -0.9621],
                [-1.4289, -3.2787, 4.1747, -6.0341, -4.6341, -5.7945, 4.142],
                [7.5973, 6.4431, 5.9872, 2.1639, -8.6566, 3.3143, -3.4059],
                [-0.8775, -3.048, 11.6543, 0.6442, 2.3218, -0.4765, 1.1516],
                [-5.5423, -2.5188, 1.0754, -0.0563, -2.9386, -1.1504, 1.0171]
            ]
        ]
    );

    // Dilations.
    let res = t.conv2d(&w, 0, 1, 2, 1)?;
    assert_eq!(res.dims(), [1, 2, 1, 1]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [2.45, -2.3504],
    );

    // Transpose and dilations.
    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 2)?;
    assert_eq!(res.dims(), [1, 2, 9, 9]);
    assert_eq!(
        test_utils::to_vec3_round(&res.i(0)?, 4).await?,
        [
            [
                [-1.9918, 3.1652, -0.6778, -4.3442, 4.4351, 0.6652, -3.0124, -0.6031, 2.9277],
                [2.7036, -1.7156, -0.3969, 1.0516, 1.6381, -2.8886, -0.205, 2.4682, -1.0499],
                [-0.9459, 3.1631, 3.707, -4.8369, -8.5166, -1.4496, -2.7559, -3.2698, 1.4376],
                [-0.2157, 3.7786, -2.0252, -4.2633, 3.6731, -1.5142, 5.9391, -0.2622, -0.141],
                [-6.8121, -3.1744, 1.5945, 3.0637, -9.6088, 1.4446, 2.9489, -3.0082, -7.3822],
                [0.2371, 3.3303, 0.3861, 2.2646, -4.6784, 4.1235, -0.0109, 0.3176, -0.03],
                [-2.5339, -2.9564, -3.4518, -4.4594, -9.1873, -1.9709, -0.4676, 0.51, -3.5024],
                [4.007, 0.3067, -2.2954, 1.1105, -0.1992, 1.6372, -2.9268, 0.2807, -1.2787],
                [5.307, 1.1317, 1.3518, 0.9049, 3.8116, -0.4075, -0.8874, -0.2241, -0.9579]
            ],
            [
                [1.089, -0.6483, 0.0726, -0.4752, -1.3283, 1.7103, 1.0703, 0.1076, -0.9211],
                [-0.8629, 0.1376, 0.3202, 2.0955, 0.9696, 2.8988, -1.0012, 1.5049, -0.1278],
                [1.9286, -1.5255, -2.9563, 2.4589, 3.3611, -0.6951, 0.3525, -1.7724, -5.9861],
                [1.1226, 2.1561, 3.6417, 4.7546, -0.692, 4.4126, -5.1902, 6.0805, 2.3185],
                [1.0111, 0.3604, 0.6432, -3.6605, 7.9517, -9.2955, -5.2988, -3.7803, -2.0642],
                [3.3172, -1.7967, -3.6576, -2.0942, 1.3158, 0.112, -1.7405, 2.9167, 0.7957],
                [5.1001, 1.8995, -1.8639, 1.1262, 9.9629, 2.683, -3.6319, -1.1607, 0.5856],
                [-4.8445, -0.5642, 4.2317, 0.0856, 1.2267, -0.5712, 1.736, 1.0997, 0.6908],
                [-5.5423, -1.1831, -1.2176, 0.0843, 0.0446, -0.7545, -2.4798, -0.0827, 1.0171]
            ]
        ]
    );

    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 2, 3, 3))
w = torch.randn((1, 2, 1, 1))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t, w_t)
print(res.shape)
print(res.flatten())

t_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t_t, w)
print(res.shape)
print(res.flatten())
*/
async fn conv2d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
            -0.6266, 0.3529, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278,
        ],
        dev,
    )?;
    let w = Tensor::new(&[-0.9259f32, 1.3017], dev)?;
    let t = t.reshape((1, 2, 3, 3))?;
    let w = w.reshape((1, 2, 1, 1))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [0.164, -0.0111, -0.1742, 2.6437, -2.0268, 1.1823, 3.2855, -1.0324, 0.2539]
    );
    let res = t.conv2d(&w, 2, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 7, 7]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1640,
            -0.0111, -0.1742, 0.0, 0.0, 0.0, 0.0, 2.6437, -2.0268, 1.1823, 0.0, 0.0, 0.0, 0.0,
            3.2855, -1.0324, 0.2539, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ]
    );

    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [0.164, -0.0111, -0.1742, 2.6437, -2.0268, 1.1823, 3.2855, -1.0324, 0.2539],
    );
    let res = t.transpose(0, 1)?.conv_transpose2d(&w, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [2, 2, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [
            -0.3755, 0.8045, -0.6336, -0.2218, -1.1369, 0.8599, 1.5768, -0.1268, -0.1728, 0.528,
            -1.131, 0.8908, 0.3118, 1.5984, -1.2089, -2.2168, 0.1783, 0.2429, -0.3838, 0.5802,
            -0.3268, -2.0382, 0.6329, -0.2293, -1.2154, 0.6441, -0.3035, 0.5396, -0.8156, 0.4594,
            2.8654, -0.8898, 0.3224, 1.7087, -0.9056, 0.4267
        ]
    );
    Ok(())
}

async fn conv2d_smaller(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866,
        ],
        dev,
    )?;
    let w = Tensor::new(&[1f32, 1., 1., 1., 1., 1., 1., 1., 1.], dev)?;
    let t = t.reshape((1, 1, 3, 3))?;
    let w = w.reshape((1, 1, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 1, 1]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [-0.6197]
    );
    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 2, 4, 2))
w = torch.randn((1, 2, 1, 1))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())
*/
async fn conv2d_non_square(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699,
        ],
        dev,
    )?;
    let w = Tensor::new(&[-1.1351f32, 1.3841], dev)?;
    let t = t.reshape((1, 2, 4, 2))?;
    let w = w.reshape((1, 2, 1, 1))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4).await?,
        [0.2312, 5.2238, 2.3772, 1.9076, 2.0256, -0.5776, -1.6028, -1.467]
    );
    Ok(())
}

/*
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5, 5), requires_grad=True)
w = torch.randn((2, 4, 3, 3), requires_grad=True)
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())
loss = (res ** 2).sum()
print(loss)
loss.backward()
print(t.grad.shape)
print(t.grad.flatten())
print(w.grad.shape)
print(w.grad.flatten())

t.grad.zero_()
w.grad.zero_()
res = torch.nn.functional.conv2d(t, w, stride=2)
print(res.flatten())
loss = (res ** 2).sum()
print(loss)
loss.backward()
print(t.grad.shape)
print(t.grad[0])
print(w.grad.shape)
print(w.grad[0])
*/
async fn conv2d_grad(dev: &Device) -> Result<()> {
    // conv-transposes are not implemented for metal
    use candle::Var;
    let t = Var::from_slice(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130, 1.3123,
            1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071, 1.1586,
            0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090, 0.2049,
            0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323, -1.3712,
            0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742, 0.3790,
            -0.4431, -0.4720, -0.7890, 0.2620, 0.7875, 0.5377, -0.6779, -0.8088, 1.9098, 1.2006,
            -0.8, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
        ],
        (1, 4, 5, 5),
        dev,
    )?;
    let w = Var::from_slice(
        &[
            -0.9325f32, 0.6451, -0.8537, 0.2378, 0.8764, -0.1832, 0.2987, -0.6488, -0.2273,
            -2.4184, -0.1192, -0.4821, -0.5079, -0.5766, -2.4729, 1.6734, 0.4558, 0.2851, 1.1514,
            -0.9013, 1.0662, -0.1817, -0.0259, 0.1709, 0.5367, 0.7513, 0.8086, -2.2586, -0.5027,
            0.9141, -1.3086, -1.3343, -1.5669, -0.1657, 0.7958, 0.1432, 0.3896, -0.4501, 0.1667,
            0.0714, -0.0952, 1.2970, -0.1674, -0.3178, 1.0677, 0.3060, 0.7080, 0.1914, 1.1679,
            -0.3602, 1.9265, -1.8626, -0.5112, -0.0982, 0.2621, 0.6565, 0.5908, 1.0089, -0.1646,
            1.8032, -0.6286, 0.2016, -0.3370, 1.2555, 0.8009, -0.6488, -0.4652, -1.5685, 1.5860,
            0.5583, 0.4623, 0.6026,
        ],
        (2, 4, 3, 3),
        dev,
    )?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2).await?, 741.12f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&grad_t.flatten_all()?, 2).await?,
        [
            9.29, -2.84, -5.71, 3.38, -7.71, -19.15, 7.02, 29.1, 9.34, 34.73, -22.87, 24.35,
            -39.88, -14.01, 21.08, 9.94, 13.63, -34.68, 11.21, -6.26, 7.72, -6.32, -16.64, -1.08,
            -20.22, 21.73, -0.37, -4.06, 5.82, -3.65, -30.73, 14.55, 87.7, 31.6, 4.53, -89.78,
            -75.37, -57.43, -7.56, 92.96, 18.79, -4.63, -159.75, -42.47, -47.26, 52.88, 37.32,
            49.0, 12.82, 2.01, -8.98, 20.18, 16.62, 12.06, 15.38, 20.0, 2.57, -15.22, 72.62,
            -10.75, 2.25, -31.2, 3.75, -0.2, 9.76, -0.68, 5.21, -40.44, -22.59, -61.61, 17.28,
            20.41, 37.55, 5.23, 6.81, 23.54, 23.62, -9.99, -9.13, 4.87, -35.06, -26.1, 63.48,
            25.81, -39.21, -70.68, -46.96, 2.33, 41.81, 82.42, -28.63, -11.78, -35.33, -10.28,
            -28.57, -9.13, 7.21, -9.05, -9.62, -11.25
        ]
    );
    assert_eq!(
        test_utils::to_vec1_round(&grad_w.flatten_all()?, 2).await?,
        [
            -28.92, -22.88, -141.23, 73.35, 61.07, 47.81, -20.0, -73.71, -41.82, -13.59, 21.5,
            28.72, 28.57, -46.85, -90.19, 143.61, 16.68, 7.43, 18.88, -90.81, -20.29, 54.79, 82.63,
            22.94, 77.81, -16.39, -13.2, 9.34, -40.39, -26.62, 5.33, -60.91, 9.09, -59.37, 7.08,
            58.64, 5.55, 20.52, 2.5, -17.25, -6.8, 22.21, 30.15, -7.52, -37.46, 5.67, 22.58, 9.03,
            47.05, 17.61, 37.31, -98.13, -14.61, -4.8, -6.36, 44.69, 23.34, 8.37, -13.52, 80.05,
            -34.24, -16.36, -12.31, 1.92, -33.62, -14.1, -49.23, -7.39, 11.5, -9.98, 9.66, 29.6
        ]
    );

    // Same as before but with stride.
    let res = t.conv2d(&w, 0, 2, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2).await?, 277.16f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        test_utils::to_vec3_round(&grad_t.i(0)?, 2).await?,
        [
            [
                [9.29, -7.03, 0.94, 3.49, -7.71],
                [-1.8, -7.82, 8.9, 8.46, 7.43],
                [-25.84, 22.09, -19.27, -0.22, 1.69],
                [4.02, 18.53, -18.37, 2.3, -24.51],
                [7.72, -9.68, -12.34, 5.6, -20.22]
            ],
            [
                [21.73, 3.39, -18.27, 3.86, -3.65],
                [8.25, 3.73, 30.73, -8.61, -11.93],
                [-72.15, -15.36, -17.53, -12.32, -1.61],
                [-22.32, -7.79, -91.82, 6.44, -37.69],
                [52.88, 14.44, 42.75, 9.88, 2.01]
            ],
            [
                [-8.98, 9.91, 6.75, -4.68, 15.38],
                [4.93, -0.33, 9.94, -1.46, 14.78],
                [13.62, -30.63, 3.96, -3.58, -4.48],
                [-14.13, 1.19, -34.43, 3.08, -33.83],
                [17.28, 12.94, 31.83, -3.35, 6.81]
            ],
            [
                [23.54, 6.98, -24.52, 0.52, 4.87],
                [9.65, 6.18, 1.71, -25.23, -4.93],
                [-54.99, -23.66, 3.19, -3.73, 18.58],
                [-21.35, -10.39, -39.88, 28.73, -30.76],
                [-9.13, 11.12, -14.0, -8.23, -11.25]
            ]
        ]
    );
    assert_eq!(
        test_utils::to_vec3_round(&grad_w.i(0)?, 2).await?,
        [
            [
                [28.34, -7.91, -45.75],
                [21.03, 3.86, 29.86],
                [0.72, -36.58, -35.28]
            ],
            [
                [-16.04, 11.53, -16.38],
                [29.62, -16.32, -48.35],
                [57.5, 28.29, 25.81]
            ],
            [
                [2.93, -19.6, 1.57],
                [27.15, 53.88, -24.64],
                [12.74, -22.6, -26.2]
            ],
            [
                [-0.18, -14.86, -6.82],
                [-19.55, -2.72, 45.9],
                [-2.54, 36.97, 27.11]
            ]
        ]
    );

    // Replicate the issue from https://github.com/huggingface/candle/issues/1212
    let res = t.i((.., .., 0..4, 0..4))?.conv2d(&w, 0, 2, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2).await?, 21.12f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        test_utils::to_vec3_round(&grad_t.i(0)?, 2).await?,
        [
            [
                [9.29, -7.03, 7.87, 0.0, 0.0],
                [-1.8, -7.82, 5.9, 0.0, 0.0],
                [-3.12, 4.49, 5.52, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [21.73, 3.39, 4.77, 0.0, 0.0],
                [8.25, 3.73, 27.61, 0.0, 0.0],
                [-20.55, -5.61, -2.77, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [-8.98, 9.91, -7.15, 0.0, 0.0],
                [4.93, -0.33, 4.56, 0.0, 0.0],
                [-6.7, -5.76, -8.05, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [23.54, 6.98, -10.0, 0.0, 0.0],
                [9.65, 6.18, 18.72, 0.0, 0.0],
                [3.29, -5.27, 0.79, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ]
    );
    assert_eq!(
        test_utils::to_vec3_round(&grad_w.i(0)?, 2).await?,
        [
            [
                [-3.47, 7.44, 0.66],
                [12.89, -3.4, -9.29],
                [-14.16, -0.83, 7.14]
            ],
            [
                [-3.23, 5.37, -3.02],
                [-2.12, -11.24, 1.94],
                [6.97, 7.2, 2.99]
            ],
            [
                [-4.04, -3.31, 4.87],
                [-6.68, -5.68, 1.73],
                [-5.54, 4.32, 0.52]
            ],
            [[-4.72, 1.5, 4.72], [3.79, 4.04, 6.76], [-4.6, 5.8, 6.93]]
        ]
    );

    // Conv Transpose 2d Test
    //tested against following python

    // import torch
    // torch.manual_seed(4242)
    // padding = 4
    // outpadding = 2
    // dilation = 3
    // stride = 3
    // input = torch.randn((1, 4, 7, 5), requires_grad=True)
    // kernel = torch.randn((4, 2, 3, 5), requires_grad=True)
    // print("input", input.flatten())
    // print("kernel", kernel.flatten())
    // res = torch.nn.functional.conv_transpose2d(
    //     input,
    //     kernel,
    //     stride=stride,
    //     padding=padding,
    //     dilation=dilation,
    //     output_padding=outpadding,
    // )
    // res.retain_grad()
    // print(res.shape)
    // loss = (res**2).sum()
    // print(loss)
    // loss.backward()
    // print(input.grad.shape)
    // print("input grad", torch.round(input.grad, decimals=1))
    // print(kernel.grad.shape)
    // print("kernel grad", torch.round(kernel.grad.flatten(), decimals=1))

    let padding = 4;
    let outpadding = 2;
    let dilation = 3;
    let stride = 3;

    let t = Var::from_slice(
        &[
            0.4056_f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997,
            3.0616, 1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843,
            0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013,
            -0.6836, 0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130,
            1.3123, 1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071,
            1.1586, 0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090,
            0.2049, 0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323,
            -1.3712, 0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742,
            0.3790, -0.4431, -0.4720, -0.7890, 0.2620, 0.5411, -1.1715, -2.4997, 2.3249, -0.8912,
            -0.4733, -0.5701, -2.8888, -1.4112, -0.5471, -0.9234, -1.1660, 0.4189, -0.7465,
            -0.6473, 0.1402, 0.7875, 0.5377, -0.6779, -0.8088, -0.4864, -0.2312, 0.9279, 0.1264,
            1.5480, 0.8265, -0.1025, 0.5138, -0.2512, 0.1576, 1.2705, 0.3641, -0.9325, 0.6451,
            -0.8537, 0.2378, 0.1794, 0.2752, -0.3687, -1.1149, -0.1410, -0.5829, -0.0892, 1.4258,
            -2.2789, 0.5270, 0.1825, 1.7007, -0.5263, -0.2954, 0.4440, 0.5537, 0.3492, 0.6186,
            1.6475, 0.2219,
        ],
        (1, 4, 7, 5),
        dev,
    )?;

    #[rustfmt::skip]
    let w = Var::from_slice(
        &[
            -1.1744_f32, 0.3266, 2.5893, 1.0142, 0.1763, 0.7752, 0.6604, 0.2029, -0.2145, 0.7234,
            -0.3441, -1.5400, -0.6333, 0.6613, 0.2083, 0.6230, -1.7002, 0.3393, 0.4049, 1.0762,
            0.2723, 1.4181, 0.0029, -0.2122, 1.7668, 1.4168, 0.3320, -0.2719, 0.7932, -0.7204,
            0.4447, 0.1211, 0.5908, 1.0089, -0.1646, 1.8033, -0.6286, 0.2016, -0.3370, 1.2555,
            0.8009, -0.6488, -0.4652, -1.5685, 1.5860, 0.5583, 0.4623, 0.6026, 0.8828, 2.4990,
            0.6811, -0.3369, 1.3320, 1.7669, -1.1067, 1.2958, -0.9415, -0.9655, -0.4462, 0.7181,
            0.5181, -1.1658, -1.8467, -0.7763, 1.2769, 0.8651, 0.9890, 1.5092, 0.7207, -0.8481,
            0.7417, 0.3375, -1.2685, 1.4572, 1.0915, 0.1093, -0.8550, -0.5831, -0.6309, -0.2509,
            0.5220, -0.0914, 0.7900, 0.1096, 0.3258, 0.2723, -1.0942, -0.3393, -0.1653, 0.5732,
            -0.8014, 1.8194, -1.9023, 0.2127, 1.8636, -0.8979, 0.1927, -0.2778, 0.3105, 0.0071,
            -1.1823, 0.2476, -0.7178, -1.3821, 1.0769, -0.4376, -0.9967, -0.1227, 1.6197, -1.0604,
            0.1372, 0.8141, -0.6163, 0.7304, -0.8285, 2.0636, -0.7176, 0.2495, -0.2581, -0.4478,
        ],
        (4, 2, 3, 5),
        dev,
    )?;
    let res = t.conv_transpose2d(&w, padding, outpadding, stride, dilation)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 0).await?, 2904.0);
    let grads = loss.backward()?;

    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 7, 5]);
    assert_eq!(grad_w.dims(), [4, 2, 3, 5]);

    assert_eq!(
        test_utils::to_vec1_round(&grad_w.flatten_all()?, 1).await?,
        [
            // torch gets 89.1
            -89.0, -135.3, 136.7, 102.0, -53.4, 117.9, 118.6, -43.9, -218.0, -58.5, -114.3, -150.0,
            -15.6, 172.1, 66.3, -64.3, -27.9, -19.8, 31.7, 62.1, 5.5, 92.6, 28.2, -29.6, 55.9,
            52.7, -72.7, -119.8, 53.8, -25.5, 128.8, 19.3, 68.0, 190.9, -64.1, -86.2, -111.2,
            106.6, -67.7, 37.8, 115.9, 50.4, -77.7, -54.9, 22.3, -4.6, 89.8, 61.7, 122.4, 192.6,
            -27.8, -104.6, 57.0, 166.4, 27.1, 6.1, 18.7, -93.2, 31.5, 168.2, -3.7, -99.5, -55.5,
            -10.8, 17.5, 20.8, 16.9, 43.8, 42.0, -89.2, 18.8, -9.6, -84.1, 212.6, 19.7, -50.0,
            -52.0, -40.0, -166.6, -73.2, -10.8, -73.3, 31.5, -23.4, -79.3, -27.0, -84.4, -42.9,
            -20.3, 51.8, -16.7, 76.3, -120.5, -65.8, 96.5, -10.7, -45.9, -88.1, 65.4, -7.0, -1.5,
            92.8, -25.1, -114.2, -5.8, -14.8, -51.2, -20.7, 54.2, -79.8, 47.7, -29.2, -8.8, 53.5,
            -28.4, 85.0, -18.3, 107.0, 28.3, -71.8
        ]
    );

    assert_eq!(
        test_utils::to_vec3_round(&grad_t.i(0)?, 1).await?,
        [
            [
                [32.3, -41.6, -24.0, 14.1, 17.6],
                [-11.8, 72.5, 87.6, 46.4, 61.5],
                [115.0, 108.5, -48.6, -63.4, -50.0],
                [51.3, 5.4, 31.3, 91.1, -30.9],
                [52.7, 92.8, -68.0, -47.0, 83.0],
                // pytorch gets -107.1
                [-10.2, -107.0, -5.4, 213.1, -31.4],
                [-2.4, 65.1, 9.2, -146.2, -24.2]
            ],
            [
                [-72.6, -63.9, -61.9, 45.3, 33.0],
                [79.3, -0.5, -26.2, 78.2, 42.7],
                [90.9, 141.6, 40.1, -62.7, 37.0],
                [32.8, 198.2, -0.8, -31.1, 27.3],
                // torch gets 48.0
                [34.5, 34.9, -47.9, 127.6, -12.3],
                [-61.4, -3.2, -2.9, -10.9, -16.6],
                [74.6, 60.1, -68.9, 34.5, -50.4]
            ],
            [
                [37.5, -56.9, -43.6, -13.5, -9.9],
                [40.0, 97.3, 28.6, 14.2, -30.1],
                [-22.3, -126.3, -68.8, -8.2, 26.1],
                [-32.9, 37.3, 108.5, -54.8, 29.6],
                [34.9, -176.9, -125.0, -28.3, -13.9],
                [-54.9, 142.6, 62.1, -80.4, -65.6],
                [7.4, -91.1, -67.6, 35.0, 39.7]
            ],
            [
                [-57.2, -40.9, -10.1, 32.6, 29.4],
                [18.7, -18.0, 29.5, -1.2, 59.2],
                [-14.0, -74.4, 19.8, -117.0, 58.2],
                [-21.8, 163.5, -71.1, -99.0, 80.9],
                [-58.9, -10.9, 93.8, -139.6, 98.0],
                // torch gets 54.5
                [-54.4, 135.3, 6.0, -79.1, 134.6],
                [27.5, -76.0, 43.4, -2.8, -7.8]
            ]
        ]
    );

    // Test the same, but then with the following properties, t & w are unmodified.
    let padding = 1;
    let outpadding = 1;
    let dilation = 1;
    let stride = 2;

    let res = t.conv_transpose2d(&w, padding, outpadding, stride, dilation)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 0).await?, 3627.0); // torch gives 3626.8560

    let grads = loss.backward()?;

    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 7, 5]);
    assert_eq!(grad_w.dims(), [4, 2, 3, 5]);

    #[rustfmt::skip]
    assert_eq!(
        test_utils::to_vec3_round(&grad_t.i(0)?, 1).await?,
        [
            [
                [  13.2,  -40.7,   -9.7,  -47.3,  -82.7],
                [ -98.2,    9.7,   57.7,   -6.2,  180.7],
                [ 100.2,   24.1,    3.7, -100.5,  -48.1],
                [  -0.3,   13.5,   -2.9,   80.0,  -49.8],
                [  47.2,  -25.6,  -74.4,   61.2,  -18.4],
                [   4.6,  -69.5,   27.9,   66.5,  -88.1],
                 // 4th column on next row; torch is 4.2
                [ -12.0,   79.2,  -40.0,    4.1,  -97.1],
            ],
            [
                [ -42.2,  -36.5,  -51.1,    7.5,   32.3],
                [  74.1,  -44.6,  -68.8,   19.5,    7.7],
                [ 137.1,   54.2,  153.8,  -58.0,   45.5],
                [  24.4,  -56.8,    9.7,  -41.0,  -14.5],
                [  -3.7,   72.6,    8.3,  134.8,   40.5],
                [  43.2,  -56.9,  -47.5,  -89.4,  -95.4],
                [  68.2,  108.1,  -80.0,   57.0, -121.1]
            ],
            [
                [  31.1,  -11.4,  -34.8,   33.1,  -44.2],
                [  29.4,  -31.6,  -40.2,   13.7,   13.1],
                [  -0.8,  -83.8,   -7.8,  -17.3,   78.2],
                [  12.0, -118.7,  137.5,  -76.7,   50.8],
                [ -28.7, -114.2,   -3.7,  -96.3,  -13.8],
                [ -31.8,   28.5,  -14.3,    4.6,   13.4],
                [  28.0,   -0.2,  -38.9,  -29.7,  -59.0]
            ],
            [
                [ -16.8,   38.5,   15.5,   26.6,   48.9],
                [  14.5,   49.6,  -24.8,   65.6,   61.7],
                [  22.1,  -64.7,   -4.3,  -51.0,   36.3],
                [  31.0,  -88.9,   47.1, -123.5,   -3.8],
                [ -14.8,  -39.8,  128.2, -110.3,   42.6],
                // 1st column on next row; torch is -7.2
                [  -7.1,   95.3,  -21.3,  -58.7,  -13.9], 
                [  26.9,   21.3,   16.1,   70.3,   32.1]
            ]
        ]
    );

    #[rustfmt::skip]
    assert_eq!(
        test_utils::to_vec1_round(&grad_w.flatten_all()?, 1).await?,
        [
            // 2nd value; torch gets -3.2, 3rd value; torch gets 221.8
           -2.460e+01, -3.100e+00,  2.219e+02,  7.400e+00,  5.620e+01,
            7.420e+01,  7.830e+01,  8.900e+00,  1.050e+01,  2.810e+01,
            5.100e+00, -1.046e+02, -1.572e+02,  8.710e+01, -9.840e+01,
           -4.230e+01, -1.898e+02,  1.860e+01, -3.570e+01,  9.810e+01,
            4.680e+01,  1.182e+02,  4.020e+01, -1.900e+00,  1.508e+02,
            1.094e+02,  1.018e+02, -4.620e+01,  1.591e+02, -2.320e+01,
            // 5th value; torch gets 7.1
           -8.450e+01, -4.600e+00,  6.330e+01,  1.123e+02, -7.000e+00,
            1.101e+02, -6.620e+01,  2.090e+01, -5.120e+01,  8.990e+01,
            9.050e+01, -6.990e+01,  6.800e+01, -9.250e+01,  1.380e+02,
            4.720e+01,  4.710e+01,  6.210e+01,  8.870e+01,  2.098e+02,
            3.870e+01, -1.390e+01,  6.270e+01,  1.484e+02, -9.920e+01,
           -4.200e+01, -1.505e+02, -1.480e+01, -2.620e+01,  8.220e+01,
           -3.350e+01, -2.260e+01, -1.198e+02, -5.080e+01,  1.259e+02,
            5.600e+01,  9.270e+01,  1.209e+02,  6.590e+01, -8.330e+01,
            7.000e+00, -2.600e+01, -1.133e+02,  3.870e+01,  4.020e+01,
           -6.300e+00, -8.710e+01, -5.150e+01, -8.510e+01,  2.000e-01,
            3.640e+01, -6.100e+00,  6.590e+01, -2.700e+00,  6.550e+01,
            // 4th value; torch gets 3.8
            5.300e+00, -6.760e+01, -4.270e+01, -3.900e+00,  2.880e+01,
            5.260e+01,  6.170e+01, -1.203e+02, -1.610e+01,  7.740e+01,
           -1.008e+02, -1.070e+01, -9.900e+00,  3.300e+00, -2.620e+01,
           -4.440e+01,  2.580e+01, -6.920e+01, -4.220e+01,  1.108e+02,
            1.240e+01, -3.440e+01, -2.800e+00,  7.880e+01, -6.690e+01,
            1.480e+01,  2.310e+01, -4.260e+01, -1.500e+00, -4.760e+01,
            5.350e+01, -2.260e+01,  8.000e-01, -3.840e+01, -2.500e+00
        ]
    );

    Ok(())
}

test_device!(conv1d, conv1d_cpu, conv1d_gpu, conv1d_metal, conv1d_wgpu);
test_device!(
    conv1d_small,
    conv1d_small_cpu,
    conv1d_small_gpu,
    conv1d_small_metal,
    conv1d_small_wgpu
);
test_device!(conv2d, conv2d_cpu, conv2d_gpu, conv2d_metal,conv2d_wgpu);
test_device!(
    conv2d_non_square,
    conv2d_non_square_cpu,
    conv2d_non_square_gpu,
    conv2d_non_square_metal,
    conv2d_non_square_wgpu
);
test_device!(
    conv2d_small,
    conv2d_small_cpu,
    conv2d_small_gpu,
    conv2d_small_metal,
    conv2d_small_wgpu
);
test_device!(
    conv2d_smaller,
    conv2d_smaller_cpu,
    conv2d_smaller_gpu,
    conv2d_smaller_metal,
    conv2d_smaller_wgpu
);
test_device!(
    conv2d_grad,
    conv2d_grad_cpu,
    conv2d_grad_gpu,
    conv2_grad_metal,
    conv2_grad_wgpu
);

async fn convert(device: &Device) -> Result<()> {
    let vf32 = Tensor::arange(0f32, 4f32, device)?;

    let vf32_u32 : Vec<u32> = vf32.to_dtype(candle::DType::U32)?.to_vec1_async().await?;
    assert_eq!(vf32_u32, [0u32, 1u32, 2u32, 3u32]);

    let vu32 = Tensor::new(vf32_u32, device)?;
    let vu32_f32 : Vec<f32> = vu32.to_dtype(candle::DType::F32)?.to_vec1_round(3).await?;
    assert_eq!(vu32_f32, [0f32, 1f32, 2f32, 3f32]);


    let vu32_u8 : Vec<u8> = vu32.to_dtype(candle::DType::U8)?.to_vec1_async().await?;
    assert_eq!(vu32_u8, [0, 1, 2, 3]);

    let vf32_u8 : Vec<u8> = vf32.to_dtype(candle::DType::U8)?.to_vec1_async().await?;
    assert_eq!(vf32_u8, [0, 1, 2, 3]);

    let vu8 = vu32.to_dtype(candle::DType::U8)?;
    let vu8_f32 : Vec<f32> = vu8.to_dtype(candle::DType::F32)?.to_vec1_async().await?;
    assert_eq!(vu8_f32, [0f32, 1f32, 2f32, 3f32]);

    Ok(())
}

async fn alloc(device: &Device) -> Result<()> {
    let t = 5.0f64;
    let ratio = (Tensor::ones(1, candle::DType::F32, device)? * t)?;
    assert_eq!(ratio.to_vec1_round(3).await?, [5f32]);

    let ratio = (Tensor::ones(1, candle::DType::U32, device)? * t)?;

    assert_eq!(ratio.to_vec1_async::<u32>().await?, [5u32]);

    Ok(())
}

test_device!(convert, convert_cpu, convert_gpu, convert_metal, convert_wgpu);
test_device!(alloc, alloc_cpu, alloc_gpu, alloc_metal, alloc_wgpu);



fn fwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        v
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        (v.exp() - T::one()) * alpha
    }
}

struct Elu {
    alpha: f64,
}

impl CustomOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = candle::map_dtype!(
            "elu",
            s,
            |s| cpu_backend::unary_map(s, l, |v| fwd(v, self.alpha)),
            (BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }
}

#[test]
async fn custom_op1_no_backward() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    let elu_t = t.apply_op1_no_bwd(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&elu_t, 4).await?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

// Define a similar struct as Elu but with backward support.
fn bwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        T::one()
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        v.exp() * alpha
    }
}

use candle::Error;

struct EluBackward {
    alpha: f64,
}

impl CustomOp1 for EluBackward {
    fn name(&self) -> &'static str {
        "elu-bwd"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = candle::map_dtype!(
            "elu-bwd",
            s,
            |s| cpu_backend::unary_map(s, l, |v| bwd(v, self.alpha)),
            (BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }
}

struct EluWithBackward(Elu);

impl EluWithBackward {
    fn new(alpha: f64) -> Self {
        Self(Elu { alpha })
    }
}

impl CustomOp1 for EluWithBackward {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        self.0.cpu_fwd(s, l)
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let alpha = self.0.alpha;
        let bwd = arg.apply_op1(EluBackward { alpha })?;
        Ok(Some(grad_res.mul(&bwd)?))
    }
}

#[test]
async fn custom_op1_with_backward() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = candle::Var::new(&[-2f32, 0f32, 2f32], cpu)?;
    let elu_t = t.apply_op1(EluWithBackward::new(2.))?;
    assert_eq!(to_vec1_round(&elu_t, 4).await?, &[-1.7293, 0.0, 2.0]);

    let grads = elu_t.backward()?;
    let grad_x = grads.get(&t).unwrap();
    assert_eq!(to_vec1_round(grad_x, 4).await?, [0.2707, 1.0, 1.0]);

    Ok(())
}

impl candle::InplaceOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        let alpha = self.alpha;
        match s {
            CpuStorage::BF16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F32(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F64(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            _ => candle::bail!("unsupported dtype for inplace elu"),
        }
        Ok(())
    }
}

#[test]
async fn inplace_op1() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    t.inplace_op1(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&t, 4).await?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

#[test]
async fn display_scalar() -> Result<()> {
    let t = Tensor::new(1234u32, &Cpu)?;
    let s = format!("{t}");
    assert_eq!(&s, "[1234]\nTensor[[], u32]");
    let t = t.to_dtype(DType::F32)?.neg()?;
    let s = format!("{}", (&t / 10.0)?);
    assert_eq!(&s, "[-123.4000]\nTensor[[], f32]");
    let s = format!("{}", (&t / 1e8)?);
    assert_eq!(&s, "[-1.2340e-5]\nTensor[[], f32]");
    let s = format!("{}", (&t * 1e8)?);
    assert_eq!(&s, "[-1.2340e11]\nTensor[[], f32]");
    let s = format!("{}", (&t * 0.)?);
    assert_eq!(&s, "[0.]\nTensor[[], f32]");
    Ok(())
}

#[test]
async fn display_vector() -> Result<()> {
    let t = Tensor::new::<&[u32; 0]>(&[], &Cpu)?;
    let s = format!("{t}");
    assert_eq!(&s, "[]\nTensor[[0], u32]");
    let t = Tensor::new(&[0.1234567, 1.0, -1.2, 4.1, f64::NAN], &Cpu)?;
    let s = format!("{t}");
    assert_eq!(
        &s,
        "[ 0.1235,  1.0000, -1.2000,  4.1000,     NaN]\nTensor[[5], f64]"
    );
    let t = (Tensor::ones(50, DType::F32, &Cpu)? * 42.)?;
    let s = format!("\n{t}");
    let expected = r#"
[42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42.]
Tensor[[50], f32]"#;
    assert_eq!(&s, expected);
    let t = (Tensor::ones(11000, DType::F32, &Cpu)? * 42.)?;
    let s = format!("{t}");
    assert_eq!(
        &s,
        "[42., 42., 42., ..., 42., 42., 42.]\nTensor[[11000], f32]"
    );
    Ok(())
}

#[test]
async fn display_multi_dim() -> Result<()> {
    let t = (Tensor::ones((200, 100), DType::F32, &Cpu)? * 42.)?;
    let s = format!("\n{t}");
    let expected = r#"
[[42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 ...
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.]]
Tensor[[200, 100], f32]"#;
    assert_eq!(&s, expected);
    let t = t.reshape(&[2, 1, 1, 100, 100])?;
    let t = format!("\n{t}");
    let expected = r#"
[[[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]],
 [[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]]]
Tensor[[2, 1, 1, 100, 100], f32]"#;
    assert_eq!(&t, expected);
    Ok(())
}





pub mod grads{

    use super::*;
    use anyhow::{Result, Context};
    #[cfg(target_arch="wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[cfg(not(target_arch="wasm32"))]
    use tokio::test as test;

    async fn simple_grad(device: &Device) -> Result<()> {
        let x = Var::new(&[3f32, 1., 4.], device)?;
        let x = x.as_tensor();
        let y = (((x * x)? + x * 5f64)? + 4f64)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(x.to_vec1_async::<f32>().await?, [3., 1., 4.]);
        // y = x^2 + 5.x + 4
        assert_eq!(y.to_vec1_async::<f32>().await?, [28., 10., 40.]);
        // dy/dx = 2.x + 5
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [11., 7., 13.]);
        Ok(())
    }
    
    async fn sum_grad(device: &Device) -> Result<()> {
        let x = Var::new(&[3f32, 1., 4.], device)?;
        let x = x.as_tensor();
        let y = (x.sqr()?.sum_keepdim(0)? * 2.)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_vec1_async::<f32>().await?, [52.]);
        // y = 2.x^2 so dy/dx = 4.x
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, &[12., 4., 16.]);
    
        // Same test as before but squeezing on the last dimension.
        let y = (x.sqr()?.sum_keepdim(0)? * 2.)?.squeeze(0)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_scalar_async::<f32>().await?, 52.);
        // y = 2.x^2 so dy/dx = 4.x
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, &[12., 4., 16.]);
        Ok(())
    }
    
    async fn matmul_grad(device: &Device) -> Result<()> {
        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let x = Var::from_slice(&data, (2, 2, 3), device)?;
        let data: Vec<_> = (0..12).map(|i| i as f32).collect();
        let y = Var::from_slice(&data, (2, 3, 2), device)?;
        let c = x.matmul(&y)?;
        let grads = c.backward()?;
        let grad_x = grads.get(&x).context("no grad for x")?;
        let grad_y = grads.get(&y).context("no grad for y")?;
        assert_eq!(grad_x.shape(), &Shape::from((2, 2, 3)));
        assert_eq!(grad_y.shape(), &Shape::from((2, 3, 2)));
        assert_eq!(
            &*grad_x.to_vec3_async::<f32>().await?,
            &[
                [[1., 5., 9.], [1., 5., 9.]],
                [[13., 17., 21.], [13., 17., 21.]]
            ]
        );
        assert_eq!(
            &*grad_y.to_vec3_async::<f32>().await?,
            &[
                [[3., 3.], [5., 5.], [7., 7.]],
                [[15., 15.], [17., 17.], [19., 19.]]
            ]
        );
        Ok(())
    }
    
    // The simplest gradient descent, using scalar variable.
    async fn grad_descent(device: &Device) -> Result<()> {
        let x = Var::new(0f32, device)?;
        let learning_rate = 0.1;
        for _step in 0..100 {
            let xt = x.as_tensor();
            let c = ((xt - 4.2)? * (xt - 4.2)?)?;
            let grads = c.backward()?;
            let x_grad = grads.get(&x).context("no grad for x")?;
            x.set(&(xt - x_grad * learning_rate)?)?
        }
        assert_eq!(x.to_scalar_async::<f32>().await?, 4.199999);
        Ok(())
    }
    
    async fn unary_grad(device: &Device) -> Result<()> {
        let x = Var::new(&[3f32, 1., 4., 0.15], device)?;
        let x = x.as_tensor();
        let y = (x.log()? + 1.)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [2.0986, 1.0, 2.3863, -0.8971]
        );
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [0.3333, 1.0, 0.25, 6.6667]
        );
        let y = x.exp()?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [20.0855, 2.7183, 54.5982, 1.1618]
        );
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [20.0855, 2.7183, 54.5982, 1.1618]
        );
        let y = x.exp()?.sqr()?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 3).await?,
            [403.429, 7.389, 2980.958, 1.35]
        );
        // exp(x)^2 = exp(2*x)
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 2).await?,
            [806.86, 14.78, 5961.92, 2.7]
        );
        let y = x.sin()?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [0.1411, 0.8415, -0.7568, 0.1494],
        );
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [-0.99, 0.5403, -0.6536, 0.9888],
        );
        let y = x.cos()?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [-0.99, 0.5403, -0.6536, 0.9888],
        );
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [-0.1411, -0.8415, 0.7568, -0.1494],
        );
        let y = x.sqr()?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_vec1_async::<f32>().await?, [9.0, 1.0, 16.0, 0.0225]);
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [6.0, 2.0, 8.0, 0.3]);
        let y = x.sqr()?.sqrt()?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_vec1_async::<f32>().await?, [3.0, 1.0, 4.0, 0.15]);
        assert_eq!(test_utils::to_vec1_round(grad_x, 4).await?, [1.0, 1.0, 1.0, 1.0]);
        let y = x.neg()?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_vec1_async::<f32>().await?, [-3.0, -1.0, -4.0, -0.15]);
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [-1.0, -1.0, -1.0, -1.0]);
        let y = x.affine(0.2, 1.)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_vec1_async::<f32>().await?, [1.6, 1.2, 1.8, 1.03]);
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [0.2, 0.2, 0.2, 0.2]);
        let y = Tensor::new(1f32, device)?.broadcast_div(x)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [0.3333, 1.0, 0.25, 6.6667]
        );
        assert_eq!(
            grad_x.to_vec1_async::<f32>().await?,
            [-0.11111111, -1.0, -0.0625, -44.444443],
        );
        let y = x.broadcast_div(&Tensor::new(0.5f32, device)?)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_vec1_async::<f32>().await?, [6., 2., 8., 0.3]);
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [2., 2., 2., 2.]);
    
        let x = Var::new(&[3f32, 1., 4., 0.15], device)?;
        let y = x.powf(2.5)?;
        let grads = y.backward()?;
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(test_utils::to_vec1_round(&y, 2).await?, [15.59, 1.0, 32.0, 0.01]);
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 2).await?,
            [12.99, 2.5, 20.0, 0.15]
        );
    
        let y = x.tanh()?;
        let grads = y.backward()?;
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(test_utils::to_vec1_round(&y, 2).await?, [1.0, 0.76, 1.0, 0.15]);
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 2).await?,
            [0.01, 0.42, 0.0, 0.98],
        );
    
        // testing compared to pytorch nn.GELU(approximate = 'tanh')
        let y = x.gelu()?;
        let grads = y.backward()?;
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [2.9964, 0.8412, 3.9999, 0.0839]
        );
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [1.0116, 1.0830, 1.0003, 0.6188],
        );
    
        // Testing compared to pytorch torch.erf
        //
        // import torch
        // x = torch.tensor([3.0, 1.0, 4.0, 0.15], requires_grad=True)
        // y = x.erf()
        // print(y)
        // loss = y.sum()
        // loss.backward()
        // print(x.grad)
        let y = x.erf()?;
        let grads = y.backward()?;
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(test_utils::to_vec1_round(&y, 4).await?, [1.0, 0.8427, 1.0, 0.168]);
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [0.0001, 0.4151, 0.0, 1.1033],
        );
    
        // Testing compared to pytorch nn.GELU(approximate = 'none')
        //
        // import torch
        // import torch.nn.functional as F
        // x = torch.tensor([3.0, 1.0, 4.0, 0.15], requires_grad=True)
        // y = F.gelu(x, approximate='none')
        // print(y)
        // loss = y.sum()
        // loss.backward()
        // print(x.grad)
        let y = x.gelu_erf()?;
        let grads = y.backward()?;
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [2.9960, 0.8413, 3.9999, 0.0839]
        );
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [1.0119, 1.0833, 1.0005, 0.6188],
        );
    
        // Testing compared to pytorch elu
        //
        // import torch
        // import torch.nn.functional as F
        // x = torch.tensor([-1.0, 0.0, -2.0, 3.0], requires_grad=True)
        // y = F.elu(x, alpha=2.0)
        // print(y)
        // loss = y.min
        // loss = y.sum()
        // loss.backward()
        // print(x.grad)
        let elu_x = Var::new(&[-1.0f32, 0., -2., 3.], device)?;
        let y = elu_x.elu(2.)?;
        let grads = y.backward()?;
        let grad_x = grads.get(&elu_x).context("no grad for x")?;
    
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [-1.2642, 0.0000, -1.7293, 3.0000]
        );
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [0.7358, 2.0000, 0.2707, 1.0000]
        );
    
        // testing compared to pytorch nn.Silu()
        let y = x.silu()?;
        let grads = y.backward()?;
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec1_round(&y, 4).await?,
            [2.8577, 0.7311, 3.9281, 0.0806]
        );
        assert_eq!(
            test_utils::to_vec1_round(grad_x, 4).await?,
            [1.0881, 0.9277, 1.0527, 0.5747],
        );
    
        if device.is_cpu() {
            let x = Var::new(&[[[1f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]]], device)?;
            let y = x.interpolate1d(12)?.reshape(36)?;
    
            let z = Tensor::new(
                &[
                    1_f32, 02., 03., 04., 05., 06., 07., 08., 09., 10., 11., 12., 13., 14., 15., 16.,
                    17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
                    33., 34., 35., 36.,
                ],
                device,
            )?;
    
            let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
            let grads = loss.backward()?;
            let grad_x = grads.get(&x).context("no grad for x")?;
    
            assert_eq!(
                test_utils::to_vec3_round(grad_x, 4).await?,
                [[[10_f32, 26., 42.], [58., 74., 90.], [106., 122., 138.]]]
            );
        }
    
        // manually checked: see comments
        let x = Var::new(&[[[[1f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]]]], device)?;
        let y = x.interpolate2d(6, 6)?.reshape(36)?;
    
        let z = Tensor::new(
            &[
                1_f32, 02., 03., 04., 05., 06., 07., 08., 09., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
                35., 36.,
            ],
            device,
        )?;
        // gradient should be
        // row 1
        // 1+2+7+8 = 18
        // 3+4+9+10 = 26
        // 5+6+11+12 = 34
        // row 2
        // 13+14+19+20 = 66
        // 15+16+21+22 = 74
        // 17+18+23+24 = 82
        // row 3
        // 25+26+31+32 = 114
        // 27+28+33+34 = 122
        // 29+30+35+36 = 130
        let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
    
        let grads = loss.backward()?;
    
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec2_round(&grad_x.flatten(0, 2)?, 4).await?,
            [[18_f32, 26., 34.], [66., 74., 82.], [114., 122., 130.]]
        );
    
        // manually checked: see comments
        let x = Var::new(&[[[[1f32, 2.], [4., 5.]]]], device)?;
        let y = x.interpolate2d(6, 6)?.reshape(36)?;
    
        let z = Tensor::new(
            &[
                1_f32, 02., 03., 04., 05., 06., 07., 08., 09., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
                35., 36.,
            ],
            device,
        )?;
        // gradient should be
        // row 1
        // 1+2+3+7+8+9+13+14+15 = 72
        // 4+5+6+10+11+12+16+17+18 = 99
        // row 2
        // 19+20+21+25+26+27+31+32+33 = 234
        // 22+23+24+28+29+30+34+35+36 = 243
        let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
    
        let grads = loss.backward()?;
    
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(
            test_utils::to_vec2_round(&grad_x.flatten(0, 2)?, 4).await?,
            [[72_f32, 99.], [234., 261.]]
        );
    
        // manually checked: see comments
        let x = Var::new(&[[[[1f32, 2.], [4., 5.]], [[6f32, 7.], [8., 9.]]]], device)?;
    
        let y = x.interpolate2d(4, 4)?.reshape(32)?;
    
        #[rustfmt::skip]
        let z = Tensor::new(
            &[
                1_f32, 02., 03., 04.,
                05.,   06., 07., 08.,
                09.,   10., 11., 12.,
                13.,   14., 15., 16.,
                17.,   18., 19., 20.,
                21.,   22., 23., 24.,
                25.,   26., 27., 28.,
                29.,   30., 31., 32.
            ],
            device,
        )?;
        // gradient should be
        // m1r1
        // 1+2+5+6=14
        // 3+4+7+8=22
        // m1r2
        // 9+10+13+14=46
        // 11+12+15+16=54
        // m2r1
        // 17+18+21+22=78
        // 19+20+23+24=86
        // m2r2
        // 25+26+29+30=110
        // 27+28+31+32=118
        let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
    
        let grads = loss.backward()?;
    
        let grad_x = grads.get(&x).context("no grad for x")?;
    
        assert_eq!(
            test_utils::to_vec3_round(&grad_x.flatten(0, 1)?, 4).await?,
            [[[14_f32, 22.], [46., 54.]], [[78., 86.], [110., 118.]]]
        );
    
        // manually checked: see comments
        let x = Var::new(
            &[[[[1f32, 2.], [4., 5.]]], [[[6f32, 7.], [8., 9.]]]],
            device,
        )?;
    
        let y = x.interpolate2d(4, 4)?.reshape(32)?;
    
        #[rustfmt::skip]
           let z = Tensor::new(
               &[
                   1_f32, 02., 03., 04.,
                   05.,   06., 07., 08.,
                   09.,   10., 11., 12.,
                   13.,   14., 15., 16.,
                   17.,   18., 19., 20.,
                   21.,   22., 23., 24.,
                   25.,   26., 27., 28.,
                   29.,   30., 31., 32.
               ],
               device,
           )?;
        // gradient should be
        // m1r1
        // 1+2+5+6=14
        // 3+4+7+8=22
        // m1r2
        // 9+10+13+14=46
        // 11+12+15+16=54
        // m2r1
        // 17+18+21+22=78
        // 19+20+23+24=86
        // m2r2
        // 25+26+29+30=110
        // 27+28+31+32=118
        let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
    
        let grads = loss.backward()?;
    
        let grad_x = grads.get(&x).context("no grad for x")?;
    
        assert_eq!(
            test_utils::to_vec3_round(&grad_x.flatten(0, 1)?, 4).await?,
            [[[14_f32, 22.], [46., 54.]], [[78., 86.], [110., 118.]]]
        );
        Ok(())
    }
    
    async fn binary_grad(device: &Device) -> Result<()> {
        let x = Var::new(&[3f32, 1., -4., -1.], device)?;
        let x = x.as_tensor();
        // leaky relu
        let y = x.maximum(&(x * 0.1)?)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(x.to_vec1_async::<f32>().await?, [3., 1., -4., -1.]);
        assert_eq!(y.to_vec1_async::<f32>().await?, [3., 1., -0.4, -0.1]);
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [1., 1., 0.1, 0.1]);
    
        let y = x.minimum(&(x * 0.1)?)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_vec1_async::<f32>().await?, [0.3, 0.1, -4., -1.]);
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [0.1, 0.1, 1., 1.]);
    
        // This one is easy to mess up, we want the gradient to be one as it is the identity function.
        let y = x.minimum(x)?;
        let grads = y.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        assert_eq!(y.to_vec1_async::<f32>().await?, [3., 1., -4., -1.]);
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [1., 1., 1., 1.]);
    
        let x_var = Var::new(&[3f32, 1., -4., -1., 5., 9.], device)?;
        let x = x_var.as_tensor();
        let y_var = Var::new(&[2f32, 7., 1.], device)?;
        let y = y_var.as_tensor();
    
        let ss = x
            .reshape((2, 3))?
            .slice_scatter0(&y.reshape((1, 3))?, 1)?
            .sqr()?;
        let grads = ss.backward()?;
        let grad_x = grads.get(x).context("no grad for x")?;
        let grad_y = grads.get(y).context("no grad for y")?;
        assert_eq!(ss.to_vec2_async::<f32>().await?, [[9., 1., 16.], [4., 49., 1.]]);
        assert_eq!(grad_x.to_vec1_async::<f32>().await?, [6.0, 2.0, -8.0, 0.0, 0.0, 0.0]);
        assert_eq!(grad_y.to_vec1_async::<f32>().await?, [4.0, 14.0, 2.0]);
        Ok(())
    }
    
    test_device!(
        simple_grad,
        simple_grad_cpu,
        simple_grad_gpu,
        simple_grad_metal,
        simple_grad_wgpu
    );
    test_device!(sum_grad, sum_grad_cpu, sum_grad_gpu, sum_grad_metal,sum_grad_wgpu);
    test_device!(
        matmul_grad,
        matmul_grad_cpu,
        matmul_grad_gpu,
        matmul_grad_metal,
        matmul_grad_wgpu
    );
    test_device!(
        grad_descent,
        grad_descent_cpu,
        grad_descent_gpu,
        grad_descent_metal,
        grad_descent_wgpu
    );
    test_device!(unary_grad, unary_grad_cpu, unary_grad_gpu, unary_grad_metal,unary_grad_wgpu);
    test_device!(
        binary_grad,
        binary_grad_cpu,
        binary_grad_gpu,
        binary_grad_metal,
        binary_grad_wgpu
    );
    


}


#[test]
async fn integer_index() -> Result<()> {
    let dev = Device::Cpu;

    let tensor = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((2, 3))?;
    let result = tensor.i(1)?;
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.to_vec1_async::<u32>().await?, &[3, 4, 5]);

    let result = tensor.i((.., 2))?;
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.to_vec1_async::<u32>().await?, &[2, 5]);

    Ok(())
}

#[test]
async fn range_index() -> Result<()> {
    let dev = Device::Cpu;
    // RangeFull
    let tensor = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((2, 3))?;
    let result = tensor.i(..)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2_async::<u32>().await?, &[[0, 1, 2], [3, 4, 5]]);

    // Range
    let tensor = Tensor::arange(0u32, 4 * 3, &dev)?.reshape((4, 3))?;
    let result = tensor.i(1..3)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2_async::<u32>().await?, &[[3, 4, 5], [6, 7, 8]]);

    // RangeFrom
    let result = tensor.i(2..)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2_async::<u32>().await?, &[[6, 7, 8], [9, 10, 11]]);

    // RangeTo
    let result = tensor.i(..2)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2_async::<u32>().await?, &[[0, 1, 2], [3, 4, 5]]);

    // RangeInclusive
    let result = tensor.i(1..=2)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2_async::<u32>().await?, &[[3, 4, 5], [6, 7, 8]]);

    // RangeTo
    let result = tensor.i(..1)?;
    assert_eq!(result.dims(), &[1, 3]);
    assert_eq!(result.to_vec2_async::<u32>().await?, &[[0, 1, 2]]);

    // RangeToInclusive
    let result = tensor.i(..=1)?;
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.to_vec2_async::<u32>().await?, &[[0, 1, 2], [3, 4, 5]]);

    // Empty range
    let result = tensor.i(1..1)?;
    assert_eq!(result.dims(), &[0, 3]);
    let empty: [[u32; 3]; 0] = [];
    assert_eq!(result.to_vec2_async::<u32>().await?, &empty);

    // Similar to PyTorch, allow empty ranges when the computed length is negative.
    #[allow(clippy::reversed_empty_ranges)]
    let result = tensor.i(1..0)?;
    assert_eq!(result.dims(), &[0, 3]);
    let empty: [[u32; 3]; 0] = [];
    assert_eq!(result.to_vec2_async::<u32>().await?, &empty);
    Ok(())
}

#[test]
async fn index_3d() -> Result<()> {
    let tensor = Tensor::from_iter(0..24u32, &Device::Cpu)?.reshape((2, 3, 4))?;
    assert_eq!(tensor.i((0, 0, 0))?.to_scalar_async::<u32>().await?, 0);
    assert_eq!(tensor.i((1, 0, 0))?.to_scalar_async::<u32>().await?, 12);
    assert_eq!(tensor.i((0, 1, 0))?.to_scalar_async::<u32>().await?, 4);
    assert_eq!(tensor.i((0, 1, 3))?.to_scalar_async::<u32>().await?, 7);
    assert_eq!(tensor.i((0..2, 0, 0))?.to_vec1_async::<u32>().await?, &[0, 12]);
    assert_eq!(
        tensor.i((0..2, .., 0))?.to_vec2_async::<u32>().await?,
        &[[0, 4, 8], [12, 16, 20]]
    );
    assert_eq!(
        tensor.i((..2, .., 3))?.to_vec2_async::<u32>().await?,
        &[[3, 7, 11], [15, 19, 23]]
    );
    assert_eq!(tensor.i((1, .., 3))?.to_vec1_async::<u32>().await?, &[15, 19, 23]);
    Ok(())
}

#[test]
async fn slice_assign() -> Result<()> {
    let dev = Device::Cpu;

    let tensor = Tensor::arange(0u32, 4 * 5, &dev)?.reshape((4, 5))?;
    let src = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((3, 2))?;
    let out = tensor.slice_assign(&[1..4, 3..5], &src)?;
    assert_eq!(
        out.to_vec2_async::<u32>().await?,
        &[
            [0, 1, 2, 3, 4],
            [5, 6, 7, 0, 1],
            [10, 11, 12, 2, 3],
            [15, 16, 17, 4, 5]
        ]
    );
    let out = tensor.slice_assign(&[0..3, 0..2], &src)?;
    assert_eq!(
        out.to_vec2_async::<u32>().await?,
        &[
            [0, 1, 2, 3, 4],
            [2, 3, 7, 8, 9],
            [4, 5, 12, 13, 14],
            [15, 16, 17, 18, 19]
        ]
    );
    Ok(())
}



async fn contiguous(device: &Device) -> Result<()> {
    let tensor = Tensor::arange(0u32, 24u32, device)?.reshape((2, 3, 4))?;
    assert_eq!(
        tensor.to_vec3_async::<u32>().await?,
        &[
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
        ]
    );
    assert_eq!(
        tensor.t()?.contiguous()?.to_vec3_async::<u32>().await?,
        &[
            [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
            [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]]
        ]
    );
    assert_eq!(
        tensor.transpose(0, 1)?.contiguous()?.to_vec3_async::<u32>().await?,
        &[
            [[0, 1, 2, 3], [12, 13, 14, 15]],
            [[4, 5, 6, 7], [16, 17, 18, 19]],
            [[8, 9, 10, 11], [20, 21, 22, 23]]
        ]
    );
    assert_eq!(
        tensor.transpose(0, 1)?.flatten_all()?.to_vec1_async::<u32>().await?,
        &[0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23]
    );
    assert_eq!(
        tensor
            .i(1..)?
            .transpose(0, 1)?
            .contiguous()?
            .to_vec3_async::<u32>().await?,
        &[[[12, 13, 14, 15]], [[16, 17, 18, 19]], [[20, 21, 22, 23]]]
    );
    assert_eq!(
        tensor.transpose(0, 2)?.contiguous()?.to_vec3_async::<u32>().await?,
        &[
            [[0, 12], [4, 16], [8, 20]],
            [[1, 13], [5, 17], [9, 21]],
            [[2, 14], [6, 18], [10, 22]],
            [[3, 15], [7, 19], [11, 23]]
        ]
    );
    Ok(())
}

test_device!(contiguous, contiguous_cpu, contiguous_gpu, contiguous_metal, contiguous_wgpu);

#[test]
async fn strided_blocks() -> Result<()> {
    use candle::Device::Cpu;
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 0);
            assert_eq!(len, 24);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 26u32, &Cpu)?
        .i(2..)?
        .reshape((2, 3, 4))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 2);
            assert_eq!(len, 24);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i(1)?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 12);
            assert_eq!(len, 12);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i((.., 1))?.contiguous()?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 0);
            assert_eq!(len, 8);
            assert_eq!(tensor.to_vec2_async::<u32>().await?, &[[4, 5, 6, 7], [16, 17, 18, 19]]);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i((.., 1))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => {
            panic!("unexpected block structure")
        }
        candle::StridedBlocks::MultipleBlocks {
            block_len,
            block_start_index,
        } => {
            assert_eq!(block_len, 4);
            assert_eq!(block_start_index.collect::<Vec<_>>(), &[4, 16])
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.t()?.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => {
            panic!("unexpected block structure")
        }
        candle::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            assert_eq!(block_len, 1);
            assert_eq!(
                block_start_index.collect::<Vec<_>>(),
                &[
                    0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15,
                    19, 23
                ]
            )
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.transpose(0, 1)?.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => {
            panic!("unexpected block structure")
        }
        candle::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            assert_eq!(block_len, 4);
            assert_eq!(
                block_start_index.collect::<Vec<_>>(),
                &[0, 12, 4, 16, 8, 20]
            )
        }
    };
    Ok(())
}



async fn layout(device: &Device) -> Result<()> {

    let rs : usize = 14;

    let a : usize = 12;  
    let b : usize = 13;

    let data1 = Tensor::ones((1,b,a,rs), candle::DType::U32, &Device::Cpu)?;
    let data1 = data1.reshape((1, b, a, rs))?;
    let data2 = data1.to_device_async(device).await?;

    let index1 = data1.i((..,..,3..6,..4))?;
    let index2 = data2.i((..,..,3..6,..4))?;

    let result1 = index1.reshape((b, 3,4))?;
    let result2 = index2.reshape((b, 3,4))?;

    assert_eq!(result1.to_vec3_async::<u32>().await?, result2.to_vec3_async::<u32>().await?);

    let copy1 = index1.copy()?;
    let copy2 = index2.copy()?;

    let result1 = copy1.reshape((b, 3,4))?;
    let result2 = copy2.reshape((b, 3,4))?;

    assert_eq!(result1.to_vec3_async::<u32>().await?, result2.to_vec3_async::<u32>().await?);

    let result1 = index1.sum_all()?.to_vec0_async::<u32>().await?;
    let result2 = index2.sum_all()?.to_vec0_async::<u32>().await?;
    
    assert_eq!(result1, result2);
    
    Ok(())
}
test_device!(layout, layout_cpu, layout_gpu, layout_metal, layout_wgpu);


async fn matmul(device: &Device) -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?;

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2_async::<f32>().await?, &[[7.0f32, 10.0], [15.0, 22.0]]);

    let data = vec![1.0f32, 2.0];
    let a = Tensor::from_slice(&data, (2, 1), device)?;
    let data = vec![3.0f32, 4.0];
    let b = Tensor::from_slice(&data, (1, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2_async::<f32>().await?, &[&[3.0, 4.0], &[6.0, 8.0]]);

    let data: Vec<_> = (0..6).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 3), device)?;
    let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (3, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2_async::<f32>().await?, &[&[16., 19.], &[52., 64.]]);

    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (2, 3, 2), device)?;
    let expected = [[[16., 19.], [52., 64.]], [[214., 235.], [304., 334.]]];

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec3_async::<f32>().await?, &expected);

    // Also perform the matmul on contiguous transposed versions.
    let a_tt = a.t()?.contiguous()?.t()?;
    assert!(!a_tt.is_contiguous());
    assert_eq!(a.dims(), a_tt.dims());
    assert_eq!(a_tt.stride(), &[6, 1, 2]);

    let b_tt = b.t()?.contiguous()?.t()?;
    assert!(!b_tt.is_contiguous());
    assert_eq!(b.dims(), b_tt.dims());
    assert_eq!(b_tt.stride(), &[6, 1, 3]);

    assert_eq!(a_tt.matmul(&b)?.to_vec3_async::<f32>().await?, &expected);
    assert_eq!(a.matmul(&b_tt)?.to_vec3_async::<f32>().await?, &expected);
    assert_eq!(a_tt.matmul(&b_tt)?.to_vec3_async::<f32>().await?, &expected);
    Ok(())
}

async fn matmul_bf16(device: &Device) -> Result<()> {
    if !device.supports_bf16() {
        return Ok(());
    }
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;

    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    assert_eq!(c.to_vec2_async::<f32>().await?, &[[7.0f32, 10.0], [15.0, 22.0]]);
    Ok(())
}

async fn broadcast_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::randn(0f32, 1f32, (3, 1, 4, 5), device)?;
    let rhs = Tensor::randn(0f32, 1f32, (6, 5, 2), device)?;
    let out = lhs.broadcast_matmul(&rhs)?;
    assert_eq!(out.dims(), &[3, 6, 4, 2]);
    for idx1 in 0..3 {
        for idx2 in 0..6 {
            let out = out.i((idx1, idx2))?;
            let lhs = lhs.i((idx1, 0))?;
            let rhs = rhs.i(idx2)?;
            let out2 = lhs.matmul(&rhs);
            let sum_diff2 = (out - out2)?.sqr()?.sum_all()?;
            // With cuda, we see errors of up to ~1e-12.
            assert!(sum_diff2.to_vec0_async::<f32>().await? < 1e-6)
        }
    }
    Ok(())
}

// https://github.com/huggingface/candle/issues/1948
async fn squeeze_mm(device: &Device) -> Result<()> {
    let seq_len = 8_usize;
    let a = Tensor::zeros((1, seq_len, 16), DType::F32, device)?;
    let x = a.i((.., seq_len - 1, ..))?;
    let w = Tensor::zeros((32, 16), DType::F32, device)?.t()?;
    let x = x.matmul(&w)?;
    assert_eq!(x.dims(), &[1, 32]);
    Ok(())
}

// https://github.com/huggingface/candle/issues/1992
async fn mm_layout(device: &Device) -> Result<()> {
    let a = Tensor::arange(0f32, 16f32, device)?.reshape((1, 1, 4, 4))?;
    let b = Tensor::arange(0f32, 8f32, device)?.reshape((1, 1, 4, 2))?;
    let mm1 = a.matmul(&b)?;
    // Forces the layout to be:
    // shape: [1, 1, 4, 2], stride: [8, 2, 2, 1], start_offset: 0
    // This is still a contiguous matrix but matmul checks are only the two last dimensions have
    // non 1 sizes but matmul check may be reluctant to handle it.
    let b = b.transpose(1, 2)?.force_contiguous()?.transpose(1, 2)?;
    let mm2 = a.matmul(&b)?;
    let diff = (mm1 - mm2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    Ok(())
}

test_device!(matmul, matmul_cpu, matmul_gpu, matmul_metal, matmul_wgpu);
test_device!(
    matmul_bf16,
    matmul_bf16_cpu,
    matmul_bf16_gpu,
    matmul_bf16_metal,
    matmul_bf16_wgpu
);
test_device!(
    broadcast_matmul,
    broadcast_matmul_cpu,
    broadcast_matmul_gpu,
    broadcast_matmul_metal,
    broadcast_matmul_wgpu
);
test_device!(squeeze_mm, squeeze_mm_cpu, squeeze_mm_gpu, squeeze_mm_metal,squeeze_mm_wgpu);
test_device!(mm_layout, mm_layout_cpu, mm_layout_gpu, mm_layout_metal, mm_layout_wgpu);


#[cfg(feature="wgpu")]
#[test]
//test different wgpu matmul shaders, compares results with cpu impl
async fn test_matmul_kernels_wgpu()-> Result<()> {
    use candle::wgpu::MatmulAlgorithm;
    
    let algs = vec![
        MatmulAlgorithm::Matmul32_64,
        MatmulAlgorithm::Matmul32_64B,
        MatmulAlgorithm::Matmul1_64B,
        MatmulAlgorithm::Matmul7,
        MatmulAlgorithm::Matmul1,
        MatmulAlgorithm::MatmulX,
        MatmulAlgorithm::Matmul16_16,
        MatmulAlgorithm::Matmul32_32,
        MatmulAlgorithm::Matmul64_64,
        MatmulAlgorithm::Matmul64_64_8_8,
        //MatmulAlgorithm::Matmul1_64,
        MatmulAlgorithm::Matmul24_24,
        MatmulAlgorithm::Matmul24_48,
        MatmulAlgorithm::Matmul24_24B,
        MatmulAlgorithm::Matmul24_48B
    ];

    let device = Device::new_wgpu(0).await?;

    if let Device::Wgpu(wgpu) = &device{
        for alg in algs{
            (*wgpu.matmul_alg.lock().unwrap()) = alg.clone();

            for tpa in [true, false]{
                for tpb in [true, false]{
                    for use_start_offset in [true, false]{
                        for tpb_batch in [true, false]{
                            for tpa_batch in [true, false]{
                                big_matmul_wgpu(&device, tpa, tpb, use_start_offset, tpb_batch, tpa_batch).await?;
                            }
                        }
                    }
                }
            }

            matmul(&device).await?;
            broadcast_matmul(&device).await?;
            squeeze_mm(&device).await?;
            mm_layout(&device).await?;
        }
    }

    Ok(())
}


//compares wgpu matmul impl, with cpu impl
#[cfg(feature="wgpu")]
async fn big_matmul_wgpu(device: &Device, tpa : bool, tpb : bool, use_start_offset : bool, tpb_batch : bool, tpa_batch : bool)-> Result<()> {
    use candle::D;
    let b = 1;
    let m = 63;
    let n = 63;
    let k = 63;

    let start_offset = if use_start_offset {100} else {0};
    let lhs1 = Tensor::rand(0f32, 100f32, b * k * m + start_offset, &Device::Cpu)?.to_dtype(DType::U32)?.to_dtype(DType::F32)?.i(start_offset..)?;
    let rhs1 = Tensor::rand(0f32, 100f32, b * k * n + start_offset, &Device::Cpu)?.to_dtype(DType::U32)?.to_dtype(DType::F32)?.i(start_offset..)?;

    let lhs;
    if tpa_batch{
        if tpa{
            lhs = lhs1.reshape((m,k,b))?.transpose(D::Minus1, D::Minus2)?.transpose(0, 1)?;
        }
        else{
            lhs = lhs1.reshape((k,m,b))?.transpose(0, 2)?;
        }
    }
    else if tpa{
        lhs = lhs1.reshape((b,k,m))?.transpose(D::Minus1, D::Minus2)?;
    }
    else{
        lhs = lhs1.reshape((b,m,k))?;
    }
    

    let rhs;
    if tpb_batch {
        if tpb{
            rhs = rhs1.reshape((k,n,b))?.transpose(D::Minus1, D::Minus2)?.transpose(0, 1)?;
        }
        else{
            rhs = rhs1.reshape((n,k,b))?.transpose(0, 2)?;
        }
    }
    else if tpb{
        rhs = rhs1.reshape((b,n,k))?.transpose(D::Minus1, D::Minus2)?;
    }
    else{
        rhs = rhs1.reshape((b,k,n))?;
    }
   
    

    let t1 = lhs.matmul(&rhs)?.reshape((b,m,n))?;


    let lhs = lhs.to_device_async(device).await?;
    let rhs = rhs.to_device_async(device).await?;

    let t2 = lhs.matmul(&rhs)?.reshape((b,m,n))?;

    let m = candle_test::to_vec3_round(&t1, 3).await?;
    let m2 = candle_test::to_vec3_round(&t2, 3).await?;

    assert_eq!(m, m2);
    Ok(())
}


// https://github.com/huggingface/candle/issues/364
async fn avg_pool2d(dev: &Device) -> Result<()> {
    let data: Vec<f32> = vec![
        1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), dev)?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::<f32>().await?, [[0.5f32, 1.], [1., 1.]]);

    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 2, 8), dev)?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::<f32>().await?, [[5. / 4., 6. / 4., 6. / 4., 1.]]);
    Ok(())
}

async fn max_pool2d(dev: &Device) -> Result<()> {
    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), dev)?;

    let pool = t.max_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::<f32>().await?, [[2f32, 3.], [5., 1.]]);

    let t = t.reshape((1, 1, 2, 8))?;
    let pool = t.max_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::<f32>().await?, [[2.0, 3.0, 5.0, 1.0]]);
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
async fn avg_pool2d_pytorch(dev: &Device) -> Result<()> {
    if dev.is_metal() {
        return Ok(());
    }
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127,
        ],
        dev,
    )?
    .reshape((1, 2, 4, 4))?;
    if !dev.is_wgpu(){ //-0.16055 rounds to -0.1605 for wgpu
        let pool = t.avg_pool2d(2)?.squeeze(0)?;
        assert_eq!(
            test_utils::to_vec3_round(&pool, 4).await?,
            [
                [[-1.1926, -0.0395], [0.2688, 0.1871]],
                [[0.1835, -0.1606], [0.6249, 0.3217]]
            ]
        );
    }
    
    let pool = t.avg_pool2d(3)?.squeeze(0)?;
    assert_eq!(
        test_utils::to_vec3_round(&pool, 4).await?,
        [[[0.085]], [[0.0078]]]
    );

    let t = t.reshape((1, 1, 4, 8))?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(
        test_utils::to_vec2_round(&pool, 4).await?,
        [
            [0.7745, 0.0276, -1.6983, 0.12],
            [0.3542, 0.1625, 0.4542, -0.0014]
        ]
    );
    Ok(())
}

async fn upsample_nearest2d(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 6f32, dev)?.reshape((1, 1, 2, 3))?;
    let upsampled = t.upsample_nearest2d(4, 6)?.i(0)?.i(0)?;
    assert_eq!(
        t.i(0)?.i(0)?.to_vec2_async::<f32>().await?,
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    );
    assert_eq!(
        upsampled.to_vec2_async::<f32>().await?,
        [
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            [3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
        ]
    );
    Ok(())
}

async fn upsample_nearest1d(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 3f32, dev)?.reshape((1,1,3))?;
    let upsampled = t.upsample_nearest1d(6)?.i(0)?.i(0)?;
    assert_eq!(
        t.i(0)?.i(0)?.to_vec1_async::<f32>().await?,
        [0.0, 1.0, 2.0]
    );
    assert_eq!(
        upsampled.to_vec1_async::<f32>().await?,[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],);
    Ok(())
}


test_device!(avg_pool2d, avg_pool2d_cpu, avg_pool2d_gpu, avg_pool2d_metal, avg_pool2d_wgpu);
test_device!(
    avg_pool2d_pytorch,
    avg_pool2d_pytorch_cpu,
    avg_pool2d_pytorch_gpu,
    avg_pool2d_pytorch_metal,
    avg_pool2d_pytorch_wgpu
);
test_device!(max_pool2d, max_pool2d_cpu, max_pool2d_gpu, max_pool2d_metal, max_pool2d_wgpu);


test_device!(
    upsample_nearest1d,
    upsample_nearest1d_cpu,
    upsample_nearest1d_gpu,
    upsample_nearest1d_metal,
    upsample_nearest1d_wgpu
);


test_device!(
    upsample_nearest2d,
    upsample_nearest2d_cpu,
    upsample_nearest2d_gpu,
    upsample_nearest2d_metal,
    upsample_nearest2d_wgpu
);

/// Regression test for pth files not loading on Windows.
// #[test]
// async fn test_pth() {
//     let tensors = candle::pickle::PthTensors::new("tests/test.pt", None).unwrap();
//     tensors.get("test").unwrap().unwrap();
// }

// #[test]
// async fn test_pth_with_key() {
//     let tensors =
//         candle::pickle::PthTensors::new("tests/test_with_key.pt", Some("model_state_dict"))
//             .unwrap();
//     tensors.get("test").unwrap().unwrap();
// }

// #[test]
// async fn test_pth_fortran_congiguous() {
//     let tensors =
//         candle::pickle::PthTensors::new("tests/fortran_tensor_3d.pth", None).unwrap();
//     let tensor = tensors.get("tensor_fortran").unwrap().unwrap();

//     assert_eq!(tensor.dims3().unwrap(), (2, 3, 4));

//     assert_eq!(
//         tensor.to_vec3_async::<i64>().await.unwrap(),
//         [
//             [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
//             [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
//         ]
//     );
// }




const GGML_TEST_SIZE: usize = 32 * 128;

const GGML_MAX_QUANTIZATION_TOTAL_ERROR: f32 = 0.002;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS: f32 = 0.0075;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS: f32 = 0.0040;
const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;

async fn test_matmul(
    device: &Device,
    (b, m, n, k): (usize, usize, usize, usize),
    dtype: GgmlDType,
) -> Result<()> {
    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), device)?;
    let rhs = Tensor::from_slice(&rhs, (k, n), device)?;
    let mm = lhs.matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&rhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar_async().await?;
    let error = error / (b * m * n) as f32;
    assert!(
        error <= 0.02,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

async fn quantized_matmul(device: &Device) -> Result<()> {
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k)).map(|v| v as f32).collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..(k * n)).map(|v| v as f32).collect::<Vec<_>>();
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t)?;
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            85120.0, 214562.0, 345455.0, 474748.0, 213475.0, 604465.0, 1000686.0, 1388317.0,
            341876.0, 994283.0, 1655709.0, 2301518.0
        ]
    );
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        mm.to_vec2_async::<f32>().await?,
        &[
            [85344.0, 214368.0, 343392.0, 472416.0],
            [214368.0, 605536.0, 996704.0, 1387872.0],
            [343392.0, 996704.0, 1650016.0, 2303328.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0).await?,
            &[
                [84946.0, 214126.0, 344757.0, 473798.0],
                [213458.0, 604350.0, 1000469.0, 1387990.0],
                [341970.0, 994574.0, 1656181.0, 2302182.0]
            ]
        ),
        Device::Cuda(_) => assert_eq!(
            to_vec2_round(&res, 0).await?,
            &[
                [84866.0, 214045.0, 344676.0, 473707.0],
                [213425.0, 604313.0, 1000431.0, 1387960.0],
                [342030.0, 994630.0, 1656248.0, 2302250.0]
            ]
        ),
        Device::Cpu => assert_eq!(
            to_vec2_round(&res, 0).await?,
            &[
                [85120.0, 214562.0, 345455.0, 474748.0],
                [213475.0, 604465.0, 1000686.0, 1388317.0],
                [341876.0, 994283.0, 1655709.0, 2301518.0]
            ]
        ),
        Device::Wgpu(_) => panic!("not supported for wgpu")
    }
    test_matmul(device, (1, 3, 4, 256), GgmlDType::Q4_0).await?;
    Ok(())
}

async fn quantized_matmul_neg(device: &Device) -> Result<()> {
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k))
        .map(|v| v as f32 - (m * k) as f32 / 2.0)
        .collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..k * n)
        .map(|v| v as f32 - (k * n) as f32 / 3.0)
        .collect::<Vec<_>>();
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t)?;
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            243524.0, -19596.0, -285051.0, -549815.0, 23777.0, 21651.0, 19398.0, 18367.0,
            -196472.0, 63012.0, 324585.0, 587902.0
        ]
    );
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        to_vec2_round(&mm, 0).await?,
        &[
            [244064.0, -20128.0, -284320.0, -548512.0],
            [23563.0, 21515.0, 19467.0, 17419.0],
            [-196939.0, 63157.0, 323253.0, 583349.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0).await?,
            &[
                [243666.0, -19714.0, -285433.0, -550453.0],
                [23782.0, 21654.0, 19400.0, 18369.0],
                [-196102.0, 63022.0, 324233.0, 587191.0]
            ]
        ),
        Device::Cuda(_) => assert_eq!(
            to_vec2_round(&res, 0).await?,
            &[
                [243740.0, -19762.0, -285476.0, -550498.0],
                [23774.0, 21645.0, 19395.0, 18364.0],
                [-196045.0, 63030.0, 324120.0, 587079.0]
            ]
        ),
        Device::Cpu => assert_eq!(
            to_vec2_round(&res, 0).await?,
            &[
                [243524.0, -19596.0, -285051.0, -549815.0],
                [23777.0, 21651.0, 19398.0, 18367.0],
                [-196472.0, 63012.0, 324585.0, 587902.0]
            ]
        ),
        Device::Wgpu(_) => panic!("not supported for wgpu")
    }
    let lhs2 = Tensor::stack(&[&lhs, &lhs], 0)?;
    let res2 = matmul.forward(&lhs2)?;
    let res2 = res2.i(1)?;
    let diff = (res - res2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    if device.is_cuda() {
        assert!(diff < 0.1);
    } else {
        assert_eq!(diff, 0.);
    }
    Ok(())
}

async fn qmm_batch(dev: &Device) -> Result<()> {
    let (lhs, rhs, _mm) = get_random_tensors(2, 256, 6, dev).await?;
    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;
    assert_eq!(mm.shape().dims(), [2, 6]);
    let lhs2 = Tensor::cat(&[&lhs, &lhs], 0)?;
    let mm2 = rhs.forward(&lhs2)?;
    assert_eq!(mm2.shape().dims(), [4, 6]);
    let diff2 = (mm2.i(2..)? - &mm)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff2, 0.0);
    let lhs3 = Tensor::cat(&[&lhs2, &lhs], 0)?;
    let mm3 = rhs.forward(&lhs3)?;
    assert_eq!(mm3.shape().dims(), [6, 6]);
    let diff3 = (mm3.i(2..4)? - &mm)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff3, 0.0);
    let diff3 = (mm3.i(4..)? - &mm)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff3, 0.0);
    let lhs4 = Tensor::cat(&[&lhs3, &lhs3], 0)?;
    let mm4 = rhs.forward(&lhs4)?;
    assert_eq!(mm4.shape().dims(), [12, 6]);
    let diff4 = (mm4.i(..6)? - &mm3)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    if dev.is_cuda() {
        // We use a different kernel for sizes from 1 to 8 on cuda which explains
        // the difference here.
        assert!(0. < diff4 && diff4 < 1e-4)
    } else {
        assert_eq!(diff4, 0.0)
    };
    let diff4 = (mm4.i(6..)? - &mm4.i(..6)?)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff4, 0.0);
    Ok(())
}

test_device!(quantized_matmul, qmm_cpu, qmm_cuda, qmm_metal);
test_device!(quantized_matmul_neg, qmm_n_cpu, qmm_n_cuda, qmm_n_metal);
test_device!(qmm_batch, qmm_b_cpu, qmm_b_cuda, qmm_b_metal);

async fn quantize_q4_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();

    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    assert_eq!(
        dst.to_vec1_async::<f32>().await?,
        &[
            -0.0, -0.0, 3.875, 3.875, 3.875, 3.875, 7.75, 7.75, 7.75, 7.75, 11.625, 11.625, 11.625,
            11.625, 15.5, 15.5, 15.5, 15.5, 19.375, 19.375, 19.375, 19.375, 23.25, 23.25, 23.25,
            23.25, 27.125, 27.125, 27.125, 27.125, 31.0, 31.0, 31.5, 31.5, 31.5, 31.5, 39.375,
            39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 47.25, 47.25, 47.25, 47.25,
            47.25, 47.25, 47.25, 47.25, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125,
            55.125, 63.0, 63.0, 63.0, 63.0, 59.375, 59.375, 71.25, 71.25, 71.25, 71.25, 71.25,
            71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 83.125, 83.125, 83.125, 83.125,
            83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 95.0, 95.0, 95.0, 95.0,
            95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 127.0, 127.0, 127.0, 127.0, 127.0, 127.0,
            127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR).await?;
    Ok(())
}

async fn quantize_q4_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1_async::<f32>().await?).await,
        &[
            0.0, 0.0, 2.066, 2.066, 4.133, 4.133, 6.199, 6.199, 8.266, 8.266, 10.332, 10.332,
            12.398, 12.398, 14.465, 14.465, 16.531, 16.531, 18.598, 18.598, 20.664, 20.664, 22.73,
            22.73, 24.797, 24.797, 26.863, 26.863, 28.93, 28.93, 30.996, 30.996, 32.0, 32.0,
            34.066, 34.066, 36.133, 36.133, 38.199, 38.199, 40.266, 40.266, 42.332, 42.332, 44.398,
            44.398, 46.465, 46.465, 48.531, 48.531, 50.598, 50.598, 52.664, 52.664, 54.73, 54.73,
            56.797, 56.797, 58.863, 58.863, 60.93, 60.93, 62.996, 62.996, 64.0, 64.0, 66.066,
            66.066, 68.133, 68.133, 70.199, 70.199, 72.266, 72.266, 74.332, 74.332, 76.398, 76.398,
            78.465, 78.465, 80.531, 80.531, 82.598, 82.598, 84.664, 84.664, 86.73, 86.73, 88.797,
            88.797, 90.863, 90.863, 92.93, 92.93, 94.996, 94.996, 96.0, 96.0, 98.066, 98.066,
            100.133, 100.133, 102.199, 102.199, 104.266, 104.266, 106.332, 106.332, 108.398,
            108.398, 110.465, 110.465, 112.531, 112.531, 114.598, 114.598, 116.664, 116.664,
            118.73, 118.73, 120.797, 120.797, 122.863, 122.863, 124.93, 124.93, 126.996, 126.996
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR).await?;
    Ok(())
}

async fn quantize_q5_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1_async::<f32>().await?).await,
        &[
            -0.0, 1.938, 1.938, 3.875, 3.875, 5.813, 5.813, 7.75, 7.75, 9.688, 9.688, 11.625,
            11.625, 13.563, 13.563, 15.5, 15.5, 17.438, 17.438, 19.375, 19.375, 21.313, 21.313,
            23.25, 23.25, 25.188, 25.188, 27.125, 27.125, 29.063, 29.063, 31.0, 31.5, 31.5, 35.438,
            35.438, 35.438, 35.438, 39.375, 39.375, 39.375, 39.375, 43.313, 43.313, 43.313, 43.313,
            47.25, 47.25, 47.25, 47.25, 51.188, 51.188, 51.188, 51.188, 55.125, 55.125, 55.125,
            55.125, 59.063, 59.063, 59.063, 59.063, 63.0, 63.0, 65.313, 65.313, 65.313, 65.313,
            65.313, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 77.188, 77.188, 77.188, 77.188,
            77.188, 77.188, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 89.063, 89.063, 89.063,
            89.063, 89.063, 89.063, 95.0, 95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 103.188, 103.188,
            103.188, 103.188, 103.188, 103.188, 103.188, 103.188, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 119.063, 119.063, 119.063, 119.063,
            119.063, 119.063, 119.063, 119.063, 127.0, 127.0, 127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR).await?;
    Ok(())
}

async fn quantize_q5_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1_async::<f32>().await?).await,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
            44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
            58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
            72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
            86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
            124.0, 125.0, 126.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR).await?;
    Ok(())
}

async fn get_test_vector2(bound: f32, size: usize, device: &Device) -> Result<Tensor> {
    assert!(
        size % crate::quantized::k_quants::QK_K == 0,
        "size must be a multiple of {}",
        crate::quantized::k_quants::QK_K
    );

    let src = (0..size)
        .map(|v| (v as f32 - size as f32 / 2.) * bound / (size as f32 / 2.))
        .collect::<Vec<_>>();
    assert_eq!([src[0], src[size / 2]], [-bound, 0.0]);
    Tensor::from_vec(src, (size,), device)
}

/// Round a vector
async fn round_vector(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|x| (1000. * x).round() / 1000.)
        .collect::<Vec<_>>()
}

async fn compare_with_error(values: &[f32], expected: &[f32], tolerance: f32) {
    for (i, (value, expected_value)) in values.iter().zip(expected.iter()).enumerate() {
        let difference = (value - expected_value).abs();

        assert!(
            difference < tolerance,
            "Error at index {}: value = {}, expected = {}. Difference = {} exceeds tolerance = {}.",
            i,
            value,
            expected_value,
            difference,
            tolerance
        );
    }
}

/// Creates a vector similar to the ones used in GGML unit tests:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L26-L30
async fn create_ggml_like_vector(offset: f32) -> Vec<f32> {
    (0..GGML_TEST_SIZE)
        .map(|i| 0.1 + 2.0 * (i as f32 + offset).cos())
        .collect()
}

/// Calculates the root mean square error between two vectors
async fn calculate_rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    sum / a.len() as f32
}

/// Similar to the GGML quantization unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L43-L50
async fn ggml_quantization_error_test(dtype: GgmlDType, device: &Device, max_error: f32) -> Result<()> {
    let src = create_ggml_like_vector(0.0).await;
    let src = Tensor::from_slice(&src, (GGML_TEST_SIZE,), device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    let error = calculate_rmse(&src.to_vec1_async::<f32>().await?, &dst.to_vec1_async::<f32>().await?).await;
    if error > max_error {
        bail!(
            "Quantization error {} exceeds max error {}",
            error,
            max_error
        );
    }
    Ok(())
}

async fn quantize_q2k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q2K;

    let src = get_test_vector2(0.5, 1024, device).await?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1_async::<f32>().await?;
    let dst = dst.to_vec1_async::<f32>().await?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.1).await;

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst).await;
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.499, -0.366, -0.249, 0.0, 0.295, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device).await?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1_async::<f32>().await?;
    let dst_big = dst_big.to_vec1_async::<f32>().await?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 6.0).await;

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS).await?;
    Ok(())
}

async fn quantize_q3k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q3K;
    let src = get_test_vector2(0.5, 1024, device).await?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1_async::<f32>().await?;
    let dst = dst.to_vec1_async::<f32>().await?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.03).await;

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst).await;
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.493, -0.37, -0.243, -0.0, 0.292, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device).await?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1_async::<f32>().await?;
    let dst_big = dst_big.to_vec1_async::<f32>().await?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 3.5).await;

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS).await?;
    Ok(())
}

async fn quantize_q4k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q4K;
    let src = get_test_vector2(0.5, 1024, device).await?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1_async::<f32>().await?;
    let dst = dst.to_vec1_async::<f32>().await?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.017).await;

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst).await;
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.288, 0.498]
    );

    let src_big = get_test_vector2(128.0, 1024, device).await?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1_async::<f32>().await?;
    let dst_big = dst_big.to_vec1_async::<f32>().await?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 4.5).await;

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR).await?;
    Ok(())
}

async fn quantize_q5k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q5K;
    let src = get_test_vector2(0.5, 1024, device).await?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1_async::<f32>().await?;
    let dst = dst.to_vec1_async::<f32>().await?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.009).await;

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst).await;
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.279, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device).await?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1_async::<f32>().await?;
    let dst_big = dst_big.to_vec1_async::<f32>().await?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.5).await;

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR).await?;
    Ok(())
}

async fn quantize_q6k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q6K;
    let src = get_test_vector2(0.5, 1024, device).await?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1_async::<f32>().await?;
    let dst = dst.to_vec1_async::<f32>().await?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008).await;

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst).await;
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.497, -0.372, -0.25, -0.0, 0.284, 0.5]
    );

    let src_big = get_test_vector2(128.0, 1024, device).await?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1_async::<f32>().await?;
    let dst_big = dst_big.to_vec1_async::<f32>().await?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.0).await;

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR).await?;
    Ok(())
}

async fn quantize_q8k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q8K;
    let src = get_test_vector2(0.5, 1024, device).await?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1_async::<f32>().await?;
    let dst = dst.to_vec1_async::<f32>().await?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008).await;

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst).await;
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.375, -0.25, -0.0, 0.281, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device).await?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1_async::<f32>().await?;
    let dst_big = dst_big.to_vec1_async::<f32>().await?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 0.6).await;

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR).await?;
    Ok(())
}

test_device!(
    quantize_q4_0,
    quantize_q4_0_cpu,
    quantize_q4_0_cuda,
    quantize_q4_0_metal
);
test_device!(
    quantize_q4_1,
    quantize_q4_1_cpu,
    quantize_q4_1_cuda,
    quantize_q4_1_metal
);
test_device!(
    quantize_q5_0,
    quantize_q5_0_cpu,
    quantize_q5_0_cuda,
    quantize_q5_0_metal
);
test_device!(
    quantize_q5_1,
    quantize_q5_1_cpu,
    quantize_q5_1_cuda,
    quantize_q5_1_metal
);
test_device!(
    quantize_q2k,
    quantize_q2k_cpu,
    quantize_q2k_cuda,
    quantize_q2k_metal
);
test_device!(
    quantize_q3k,
    quantize_q3k_cpu,
    quantize_q3k_cuda,
    quantize_q3k_metal
);
test_device!(
    quantize_q4k,
    quantize_q4k_cpu,
    quantize_q4k_cuda,
    quantize_q4k_metal
);
test_device!(
    quantize_q5k,
    quantize_q5k_cpu,
    quantize_q5k_cuda,
    quantize_q5k_metal
);
test_device!(
    quantize_q6k,
    quantize_q6k_cpu,
    quantize_q6k_cuda,
    quantize_q6k_metal
);
test_device!(
    quantize_q8k,
    quantize_q8k_cpu,
    quantize_q8k_cuda,
    quantize_q8k_metal
);

/// Very simple dot product implementation
async fn vec_dot_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Returns the error achieved by the GGML matmul unit test.
async fn ggml_reference_matmul_error(dtype: GgmlDType) -> Result<f32> {
    let err = match dtype {
        GgmlDType::F16 => 0.000010,
        GgmlDType::Q2K => 0.004086,
        GgmlDType::Q3K => 0.016148,
        GgmlDType::Q4K => 0.002425,
        GgmlDType::Q5K => 0.000740,
        GgmlDType::Q6K => 0.000952,
        GgmlDType::Q4_0 => 0.001143,
        GgmlDType::Q4_1 => 0.008,
        GgmlDType::Q5_0 => 0.001353,
        GgmlDType::Q5_1 => 0.00149,
        GgmlDType::Q8_0 => 0.000092,

        // Not from the ggml repo.
        GgmlDType::Q8K => 0.00065,
        _ => bail!("No GGML results for quantization type {dtype:?}",),
    };
    Ok(err)
}

/// Similar to the GGML matmul unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L76-L91
async fn ggml_matmul_error_test<T: GgmlType>() -> Result<()> {
    let a = create_ggml_like_vector(0.0).await;
    let b = create_ggml_like_vector(1.0).await;
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 1.0).await?;
    // Another example that is more likely to trigger the overflow reported in #1526
    let a = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    let b = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 2.0).await?;
    Ok(())
}

async fn ggml_matmul_error_test_<T: GgmlType>(a: &[f32], b: &[f32], err_m: f32) -> Result<()> {
    let length = a.len();

    let mut a_quant = vec![T::zeros(); length / T::BLCK_SIZE];
    let mut b_quant = vec![T::VecDotType::zeros(); length / T::VecDotType::BLCK_SIZE];
    T::from_float(a, &mut a_quant)?;
    T::VecDotType::from_float(b, &mut b_quant)?;

    let result = T::vec_dot(length, &a_quant, &b_quant)?;
    let result_unopt = T::vec_dot_unopt(length, &a_quant, &b_quant)?;
    let reference_result = vec_dot_reference(a, b).await;

    if (result - result_unopt).abs() / length as f32 > 1e-6 {
        bail!(
            "the opt and unopt vec-dot returned different values, opt {result}, unopt {result_unopt}"
        )
    }

    let error = (result - reference_result).abs() / length as f32;

    let ggml_error = ggml_reference_matmul_error(T::DTYPE).await? * err_m;

    if !error.is_finite() || error > GGML_MAX_DOT_PRODUCT_ERROR {
        bail!("Dot product error {error} exceeds max error {GGML_MAX_DOT_PRODUCT_ERROR}",);
    }

    // We diverge slightly due to different rounding behavior / f16 to f32 conversions in GGML
    // => we use a slightly higher error threshold
    const ERROR_LENIENCY: f32 = 0.00001;
    if error - ERROR_LENIENCY > ggml_error {
        bail!(
            "Dot product error {} exceeds ggml reference error {}",
            error,
            ggml_error
        );
    }
    Ok(())
}

#[test]
async fn quantized_mm() -> Result<()> {
    ggml_matmul_error_test::<k_quants::BlockQ4_0>().await?;
    ggml_matmul_error_test::<k_quants::BlockQ4_1>().await?;
    ggml_matmul_error_test::<k_quants::BlockQ5_0>().await?;
    ggml_matmul_error_test::<k_quants::BlockQ5_1>().await?;
    ggml_matmul_error_test::<k_quants::BlockQ8_0>().await?;
    Ok(())
}

/// generates random tensors of size `m x k` and `n x k` and calculates their expected matrix multiplication result.
async fn get_random_tensors(
    m: usize,
    k: usize,
    n: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(314159265358979);

    let lhs = (0..m * k)
        .map(|_| rng.gen::<f32>() - 0.5)
        .collect::<Vec<_>>();
    let rhs = (0..n * k)
        .map(|_| rng.gen::<f32>() - 0.5)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_vec(lhs, (m, k), device)?;
    let rhs = Tensor::from_vec(rhs, (n, k), device)?;

    let mm = lhs.matmul(&rhs.t()?)?;
    Ok((lhs, rhs, mm))
}

#[macro_export]
macro_rules! quantized_matmul {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $fn_name_cpu: ident, $fn_name_cuda: ident, $fn_name_metal: ident, $dtype: expr) => {
        async fn $fn_name(device: &Device) -> Result<()> {
            test_matmul(device, (1, 3, 4, 256), $dtype).await?;
            Ok(())
        }

        test_device!($fn_name, $fn_name_cpu, $fn_name_cuda, $fn_name_metal);
    };
}

quantized_matmul!(
    quantized_matmul_q4_0_bis,
    quantized_matmul_q4_0_cpu,
    quantized_matmul_q4_0_cuda,
    quantized_matmul_q4_0_metal,
    GgmlDType::Q4_0
);
quantized_matmul!(
    quantized_matmul_q4_1_bis,
    quantized_matmul_q4_1_cpu,
    quantized_matmul_q4_1_cuda,
    quantized_matmul_q4_1_metal,
    GgmlDType::Q4_1
);
quantized_matmul!(
    quantized_matmul_q5_0_bis,
    quantized_matmul_q5_0_cpu,
    quantized_matmul_q5_0_cuda,
    quantized_matmul_q5_0_metal,
    GgmlDType::Q5_0
);
quantized_matmul!(
    quantized_matmul_q5_1_bis,
    quantized_matmul_q5_1_cpu,
    quantized_matmul_q5_1_cuda,
    quantized_matmul_q5_1_metal,
    GgmlDType::Q5_1
);
quantized_matmul!(
    quantized_matmul_q8_0_bis,
    quantized_matmul_q8_0_cpu,
    quantized_matmul_q8_0_cuda,
    quantized_matmul_q8_0_metal,
    GgmlDType::Q8_0
);
// Not implemented in Ggml
// quantized_matmul!(
//     quantized_matmul_q8_1_bis,
//     quantized_matmul_q8_1_cpu,
//     quantized_matmul_q8_1_cuda,
//     quantized_matmul_q8_1_metal,
//     GgmlDType::Q8_1
// );
// TODO This is bugged (also bugged in GGML
quantized_matmul!(
    quantized_matmul_q2k_bis,
    quantized_matmul_q2k_cpu,
    quantized_matmul_q2k_cuda,
    quantized_matmul_q2k_metal,
    GgmlDType::Q2K
);
quantized_matmul!(
    quantized_matmul_q3k_bis,
    quantized_matmul_q3k_cpu,
    quantized_matmul_q3k_cuda,
    quantized_matmul_q3k_metal,
    GgmlDType::Q3K
);
quantized_matmul!(
    quantized_matmul_q4k_bis,
    quantized_matmul_q4k_cpu,
    quantized_matmul_q4k_cuda,
    quantized_matmul_q4k_metal,
    GgmlDType::Q4K
);
quantized_matmul!(
    quantized_matmul_q5k_bis,
    quantized_matmul_q5k_cpu,
    quantized_matmul_q5k_cuda,
    quantized_matmul_q5k_metal,
    GgmlDType::Q5K
);
quantized_matmul!(
    quantized_matmul_q6k_bis,
    quantized_matmul_q6k_cpu,
    quantized_matmul_q6k_cuda,
    quantized_matmul_q6k_metal,
    GgmlDType::Q6K
);
// Not implemented on metal
// quantized_matmul!(
//     quantized_matmul_q8k_bis,
//     quantized_matmul_q8k_cpu,
//     quantized_matmul_q8k_cuda,
//     quantized_matmul_q8k_metal,
//     GgmlDType::Q8K
// );

#[test]
async fn quantized_matmul_q2k() -> Result<()> {
    use k_quants::BlockQ2K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu).await?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [0.916, 0.422, 0.215, 1.668]);

    ggml_matmul_error_test::<BlockQ2K>().await?;

    Ok(())
}

#[test]
async fn quantized_matmul_q3k() -> Result<()> {
    use k_quants::BlockQ3K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu).await?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q3K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.029, 1.418, -0.314, 1.495]);

    ggml_matmul_error_test::<BlockQ3K>().await?;

    Ok(())
}

#[test]
async fn quantized_matmul_q4k() -> Result<()> {
    use k_quants::BlockQ4K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu).await?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q4K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.125, 1.435, -0.201, 1.589]);

    ggml_matmul_error_test::<BlockQ4K>().await?;

    Ok(())
}

#[test]
async fn quantized_matmul_q5k() -> Result<()> {
    use k_quants::BlockQ5K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu).await?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q5K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.192, 1.491, -0.18, 1.743]);

    //Expected: 0.000740408897
    ggml_matmul_error_test::<BlockQ5K>().await?;

    Ok(())
}

#[test]
async fn quantized_matmul_q6k() -> Result<()> {
    use k_quants::BlockQ6K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu).await?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q6K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.324, 1.49, -0.164, 1.741]);

    ggml_matmul_error_test::<BlockQ6K>().await?;
    Ok(())
}

#[test]
async fn quantized_matmul_q8k() -> Result<()> {
    use k_quants::BlockQ8K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu).await?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q8K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1_async::<f32>().await?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]).await;
    assert_eq!(dst, [1.266, 1.504, -0.204, 1.7]);

    ggml_matmul_error_test::<BlockQ8K>().await?;
    Ok(())
}




async fn zeros(device: &Device) -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, device)?;
    let (dim1, dim2) = tensor.dims2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 2);
    Ok(())
}

async fn ones(device: &Device) -> Result<()> {
    if device.is_dtype_available(DType::U8){
        assert_eq!(
            Tensor::ones((2, 3), DType::U8, device)?.to_vec2_async::<u8>().await?,
            [[1, 1, 1], [1, 1, 1]],
        );
    }
    if device.is_dtype_available(DType::U32){
        assert_eq!(
            Tensor::ones((2, 3), DType::U32, device)?.to_vec2_async::<u32>().await?,
            [[1, 1, 1], [1, 1, 1]],
        );
    }
    if device.is_dtype_available(DType::I64){
        assert_eq!(
            Tensor::ones((2, 3), DType::I64, device)?.to_vec2_async::<i64>().await?,
            [[1, 1, 1], [1, 1, 1]],
        );
    }
    assert_eq!(
        Tensor::ones((2, 3), DType::F32, device)?.to_vec2_async::<f32>().await?,
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    );
    if device.is_dtype_available(DType::F64){
        assert_eq!(
            Tensor::ones((2, 3), DType::F64, device)?.to_vec2_async::<f64>().await?,
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        );
    }
    Ok(())
}

async fn full(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::full(42u32, (2, 3), device)?.to_vec2_async::<u32>().await?,
        [[42, 42, 42], [42, 42, 42]],
    );
    Ok(())
}

async fn arange(device: &Device) -> Result<()> {
    if device.is_dtype_available(DType::U8){
        assert_eq!(
            Tensor::arange(0u8, 5u8, device)?.to_vec1_async::<u8>().await?,
            [0, 1, 2, 3, 4],
        );
        assert_eq!(
            Tensor::arange_step(0u8, 5u8, 2, device)?.to_vec1_async::<u8>().await?,
            [0, 2, 4],
        );
        assert_eq!(
            Tensor::arange_step(0u8, 5u8, 3, device)?.to_vec1_async::<u8>().await?,
            [0, 3],
        );
    }

    if device.is_dtype_available(DType::I64){
        assert_eq!(
            Tensor::arange_step(5i64, 0i64, -1, device)?.to_vec1_async::<i64>().await?,
            [5, 4, 3, 2, 1],
        );
    }
    Ok(())
}

async fn add_mul(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[3f32, 1., 4.], device)?;
    let dim1 = tensor.dims1()?;
    assert_eq!(dim1, 3);
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [3., 1., 4.]);
    let tensor = Tensor::add(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [6., 2., 8.]);
    let tensor = Tensor::mul(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [36., 4., 64.]);
    Ok(())
}

async fn tensor_2d(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content, data);
    Ok(())
}

async fn clamp(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let tensor = tensor.clamp(1.5, 6.2)?;
    assert_eq!(
        tensor.to_vec2_async::<f32>().await?,
        [[3.0, 1.5, 4.0, 1.5, 5.0], [2.0, 1.5, 6.2, 6.2, 2.0]],
    );
    Ok(())
}

async fn asort(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1.1, 5.], [2.1, 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let indexes = tensor.arg_sort_last_dim(true)?;
    assert_eq!(
        indexes.to_vec2_async::<u32>().await?,
        [[1, 3, 0, 2, 4], [1, 4, 0, 2, 3]],
    );
    let indexes = tensor.arg_sort_last_dim(false)?;
    assert_eq!(
        indexes.to_vec2_async::<u32>().await?,
        [[4, 2, 0, 3, 1], [3, 2, 0, 4, 1]],
    );
    let (sorted, indexes) = tensor.sort_last_dim(true)?;
    assert_eq!(
        indexes.to_vec2_async::<u32>().await?,
        [[1, 3, 0, 2, 4], [1, 4, 0, 2, 3]],
    );
    assert_eq!(
        sorted.to_vec2_async::<f32>().await?,
        [[1.0, 1.1, 3.0, 4.0, 5.0], [1.0, 2.0, 2.1, 7.0, 8.0]]
    );
    let (sorted, indexes) = tensor.sort_last_dim(false)?;
    assert_eq!(
        indexes.to_vec2_async::<u32>().await?,
        [[4, 2, 0, 3, 1], [3, 2, 0, 4, 1]],
    );
    assert_eq!(
        sorted.to_vec2_async::<f32>().await?,
        [[5.0, 4.0, 3.0, 1.1, 1.0], [8.0, 7.0, 2.1, 2.0, 1.0]]
    );
    Ok(())
}

async fn unary_op(device: &Device) -> Result<()> {
    let data = &[[-3f32, 1., 4., -0.1, 0.5], [2.7, -1.8, -0.28, 1.8, 2.8]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        test_utils::to_vec2_round(&tensor.gelu()?, 4).await?,
        [
            [-0.0036, 0.8412, 3.9999, -0.046, 0.3457],
            [2.6911, -0.0647, -0.1091, 1.7353, 2.7933]
        ]
    );
    if device.is_dtype_available(DType::F16){
        let t_f16 = tensor.to_dtype(DType::F16)?.gelu()?.to_dtype(DType::F32)?;
        let max_diff = (tensor.gelu()? - t_f16)?.flatten_all()?.max(0)?;
        assert!(max_diff.to_vec0_async::<f32>().await? < 5e-3);
        assert_eq!(
            test_utils::to_vec2_round(&tensor.gelu_erf()?, 4).await?,
            [
                [-0.004, 0.8413, 3.9999, -0.046, 0.3457],
                [2.6906, -0.0647, -0.1091, 1.7353, 2.7928]
            ]
        );
    }
   
    assert_eq!(
        test_utils::to_vec2_round(&tensor.erf()?, 4).await?,
        [
            [-1.0, 0.8427, 1.0, -0.1125, 0.5205],
            [0.9999, -0.9891, -0.3079, 0.9891, 0.9999]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.silu()?, 4).await?,
        [
            [-0.1423, 0.7311, 3.9281, -0.0475, 0.3112],
            [2.53, -0.2553, -0.1205, 1.5447, 2.6395]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.ceil()?, 4).await?,
        [[-3.0, 1.0, 4.0, -0.0, 1.0], [3.0, -1.0, -0.0, 2.0, 3.0]]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.floor()?, 4).await?,
        [[-3.0, 1.0, 4.0, -1.0, 0.0], [2.0, -2.0, -1.0, 1.0, 2.0]]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.round()?, 4).await?,
        [[-3.0, 1.0, 4.0, -0.0, 1.0], [3.0, -2.0, -0.0, 2.0, 3.0]]
    );
    let tensor = Tensor::new(&[2997.9246, 314.15926f32], device)?;
    assert_eq!(
        test_utils::to_vec1_round(&tensor.round_to(2)?, 4).await?,
        [2997.92, 314.16]
    );
    assert_eq!(
        test_utils::to_vec1_round(&tensor.round_to(-2)?, 4).await?,
        [3000.0, 300.]
    );
    let tensor = Tensor::new(
        &[-1.01f32, -0.9, -0.1, 0.0, -0.0, 0.1, 0.9, 1.0, 1.1],
        device,
    )?;
    assert_eq!(
        tensor.sign()?.to_vec1_async::<f32>().await?,
        [-1., -1., -1., 0., 0., 1., 1., 1., 1.]
    );
    let tensor = Tensor::new(&[-1.0f32, 0., -2., 3.], device)?;
    let y = tensor.elu(2.)?;
    assert_eq!(
        test_utils::to_vec1_round(&y, 4).await?,
        [-1.2642, 0.0000, -1.7293, 3.0000]
    );
    // This test failed on metal prior to the following PR:
    // https://github.com/huggingface/candle/pull/2490
    let y = tensor.reshape((2, 2))?.t()?.elu(2.)?.flatten_all()?;
    assert_eq!(
        test_utils::to_vec1_round(&y, 4).await?,
        [-1.2642, -1.7293, 0.0000, 3.0000]
    );
    Ok(())
}

async fn binary_op(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, device)?;
    let tensor = (&tensor1 + (&tensor1 * &tensor1)? / (&tensor1 + &tensor2))?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);
    #[allow(clippy::eq_op)]
    let tensor = (&tensor - &tensor)?;
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);

    let min = tensor1.minimum(&(&tensor2 * 0.5)?)?;
    let max = tensor1.maximum(&(&tensor2 * 0.5)?)?;
    assert_eq!(
        min.to_vec2_async::<f32>().await?,
        [[2.5, 1.0, 2.5, 1.0, 2.5], [1.0, 0.5, 3.5, 4.0, 1.0]],
    );
    assert_eq!(
        max.to_vec2_async::<f32>().await?,
        [[3.0, 2.5, 4.0, 2.5, 5.0], [2.0, 1.0, 7.0, 8.0, 2.0]]
    );
    Ok(())
}

async fn transpose(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?.t()?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (5, 2));
    assert_eq!(
        tensor.to_vec2_async::<f32>().await?,
        &[[3f32, 2.], [1., 1.], [4., 7.], [1., 8.], [5., 2.]]
    );
    assert_eq!(tensor.t()?.to_vec2_async::<f32>().await?, data);
    assert_eq!(tensor.contiguous()?.t()?.to_vec2_async::<f32>().await?, data);
    assert_eq!(((tensor + 1.)?.t()? - 1.)?.to_vec2_async::<f32>().await?, data);
    Ok(())
}

async fn var(device: &Device) -> Result<()> {
    // Values taken from https://pytorch.org/docs/stable/generated/torch.var.html
    let data = &[
        [0.2035f32, 1.2959, 1.8101, -0.4644],
        [1.5027, -0.3270, 0.5905, 0.6538],
        [-1.5745, 1.3330, -0.5596, -0.6548],
        [0.1264, -0.5080, 1.6420, 0.1992],
    ];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        test_utils::to_vec2_round(&tensor.var_keepdim(1)?, 4).await?,
        &[[1.0631], [0.559], [1.4893], [0.8258]]
    );
    Ok(())
}

async fn sum(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.sum_keepdim(2)?.to_vec3_async::<u32>().await?,
        &[[[8], [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec3_async::<u32>().await?,
        &[[[5, 2, 11], [9, 7, 17]]],
    );
    assert_eq!(tensor.sum_keepdim((0, 2, 1))?.to_vec3_async::<u32>().await?, &[[[51]]],);
    assert_eq!(
        tensor.t()?.sum_keepdim(1)?.t()?.to_vec3_async::<u32>().await?,
        &[[[8], [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim((2, 1))?.to_vec3_async::<u32>().await?,
        &[[[8 + 15]], [[10 + 18]]]
    );
    let data: Vec<u32> = (0..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.sum_keepdim(0)?.to_vec1_async::<u32>().await?, &[7998000]);
    let tensor = tensor.reshape((2000, 2))?;
    assert_eq!(tensor.sum_keepdim((0, 1))?.to_vec2_async::<u32>().await?, &[[7998000]]);
    assert_eq!(
        tensor.sum_keepdim(0)?.sum_keepdim(1)?.to_vec2_async::<u32>().await?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1)?.sum_keepdim(0)?.to_vec2_async::<u32>().await?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec2_async::<u32>().await?,
        &[[3998000, 4000000]]
    );

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(tensor.sum_keepdim((0, 1))?.to_vec2_async::<u32>().await?, &[[7998000]]);
    assert_eq!(
        tensor.sum_keepdim(0)?.sum_keepdim(1)?.to_vec2_async::<u32>().await?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1)?.sum_keepdim(0)?.to_vec2_async::<u32>().await?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec2_async::<u32>().await?,
        &[[3998000, 4000000]]
    );

    let t1 = tensor.reshape((200, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.sum_keepdim((0, 1, 2))?.to_vec3_async::<u32>().await?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(0)?
                .sum_keepdim(2)?
                .sum_keepdim(1)?
                .to_vec3_async::<u32>().await?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(0)?
                .sum_keepdim((1, 2))?
                .to_vec3_async::<u32>().await?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(1)?
                .sum_keepdim((0, 2))?
                .to_vec3_async::<u32>().await?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0)?.to_vec3_async::<u32>().await?,
            &[[
                [398000, 398200, 398400, 398600],
                [398800, 399000, 399200, 399400],
                [399600, 399800, 400000, 400200],
                [400400, 400600, 400800, 401000],
                [401200, 401400, 401600, 401800]
            ]]
        );
    }
    Ok(())
}

async fn min(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.min_keepdim(2)?.to_vec3_async::<u32>().await?,
        &[[[1], [1]], [[1], [2]]]
    );
    assert_eq!(
        tensor.min_keepdim(0)?.to_vec3_async::<u32>().await?,
        &[[[2, 1, 4], [1, 2, 8]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.min_keepdim(0)?.to_vec1_async::<u32>().await?, &[200]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.min_keepdim(0)?.min_keepdim(1)?.to_vec2_async::<u32>().await?,
        &[[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1)?.min_keepdim(0)?.to_vec2_async::<u32>().await?,
        &[[200]]
    );
    assert_eq!(tensor.min_keepdim(0)?.to_vec2_async::<u32>().await?, &[[200, 201]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.min_keepdim(0)?.min_keepdim(1)?.to_vec2_async::<u32>().await?,
        &[[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1)?.min_keepdim(0)?.to_vec2_async::<u32>().await?,
        &[[200]]
    );
    assert_eq!(tensor.min_keepdim(0)?.to_vec2_async::<u32>().await?, &[[200, 201]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .min_keepdim(0)?
                .min_keepdim(2)?
                .min_keepdim(1)?
                .to_vec3_async::<u32>().await?,
            &[[[200]]]
        );
        assert_eq!(
            tensor.min_keepdim(0)?.to_vec3_async::<u32>().await?,
            &[[
                [200, 201, 202, 203],
                [204, 205, 206, 207],
                [208, 209, 210, 211],
                [212, 213, 214, 215],
                [216, 217, 218, 219]
            ]]
        );
    }
    Ok(())
}

async fn max(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.max_keepdim(2)?.to_vec3_async::<u32>().await?,
        &[[[4], [9]], [[7], [8]]]
    );
    assert_eq!(
        tensor.max_keepdim(0)?.to_vec3_async::<u32>().await?,
        &[[[3, 1, 7], [8, 5, 9]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.max_keepdim(0)?.to_vec1_async::<u32>().await?, &[3999]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.max_keepdim(0)?.max_keepdim(1)?.to_vec2_async::<u32>().await?,
        &[[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1)?.max_keepdim(0)?.to_vec2_async::<u32>().await?,
        &[[3999]]
    );
    assert_eq!(tensor.max_keepdim(0)?.to_vec2_async::<u32>().await?, &[[3998, 3999]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.max_keepdim(0)?.max_keepdim(1)?.to_vec2_async::<u32>().await?,
        &[[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1)?.max_keepdim(0)?.to_vec2_async::<u32>().await?,
        &[[3999]]
    );
    assert_eq!(tensor.max_keepdim(0)?.to_vec2_async::<u32>().await?, &[[3998, 3999]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .max_keepdim(0)?
                .max_keepdim(2)?
                .max_keepdim(1)?
                .to_vec3_async::<u32>().await?,
            &[[[3999]]]
        );
        assert_eq!(
            tensor.max_keepdim(0)?.to_vec3_async::<u32>().await?,
            &[[
                [3980, 3981, 3982, 3983],
                [3984, 3985, 3986, 3987],
                [3988, 3989, 3990, 3991],
                [3992, 3993, 3994, 3995],
                [3996, 3997, 3998, 3999]
            ]]
        );
    }
    Ok(())
}

async fn argmin(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmin_keepdim(2)?.to_vec3_async::<u32>().await?,
        &[[[1], [0]], [[1], [1]]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0)?.to_vec3_async::<u32>().await?,
        &[[[1, 0, 0], [0, 1, 1]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec1_async::<u32>().await?, &[0]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor
            .argmin_keepdim(0)?
            .argmin_keepdim(1)?
            .to_vec2_async::<u32>().await?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmin_keepdim(1)?
            .argmin_keepdim(0)?
            .to_vec2_async::<u32>().await?,
        &[[0]]
    );
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec2_async::<u32>().await?, &[[0, 0]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor
            .argmin_keepdim(0)?
            .argmin_keepdim(1)?
            .to_vec2_async::<u32>().await?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmin_keepdim(1)?
            .argmin_keepdim(0)?
            .to_vec2_async::<u32>().await?,
        &[[0]]
    );
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec2_async::<u32>().await?, &[[0, 0]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .argmin_keepdim(0)?
                .argmin_keepdim(2)?
                .argmin_keepdim(1)?
                .to_vec3_async::<u32>().await?,
            &[[[0]]]
        );
        assert_eq!(
            tensor.argmin_keepdim(0)?.to_vec3_async::<u32>().await?,
            &[[
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]]
        );
    }
    Ok(())
}

async fn argmax(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmax_keepdim(2)?.to_vec3_async::<u32>().await?,
        &[[[2], [2]], [[2], [0]]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0)?.to_vec3_async::<u32>().await?,
        &[[[0, 0, 1], [1, 0, 0]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec1_async::<u32>().await?, &[3799]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor
            .argmax_keepdim(0)?
            .argmax_keepdim(1)?
            .to_vec2_async::<u32>().await?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmax_keepdim(1)?
            .argmax_keepdim(0)?
            .to_vec2_async::<u32>().await?,
        &[[0]]
    );
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec2_async::<u32>().await?, &[[1899, 1899]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor
            .argmax_keepdim(0)?
            .argmax_keepdim(1)?
            .to_vec2_async::<u32>().await?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmax_keepdim(1)?
            .argmax_keepdim(0)?
            .to_vec2_async::<u32>().await?,
        &[[0]]
    );
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec2_async::<u32>().await?, &[[1899, 1899]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .argmax_keepdim(0)?
                .argmax_keepdim(2)?
                .argmax_keepdim(1)?
                .to_vec3_async::<u32>().await?,
            &[[[0]]]
        );
        assert_eq!(
            tensor.argmax_keepdim(0)?.to_vec3_async::<u32>().await?,
            &[[
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
            ]]
        );
    }
    Ok(())
}

async fn narrow(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.narrow(2, 1, 2)?.to_vec3_async::<f32>().await?,
        &[[[1.0, 4.0], [5.0, 9.0]], [[1.0, 7.0], [2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(1, 1, 1)?.to_vec3_async::<f32>().await?,
        &[[[1.0, 5.0, 9.0]], [[8.0, 2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 0, 1)?.to_vec3_async::<f32>().await?,
        &[[[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 1, 1)?.to_vec3_async::<f32>().await?,
        &[[[2.0, 1.0, 7.0], [8.0, 2.0, 8.0]]],
    );
    // The following has been checked against PyTorch via:
    //   import torch
    //   t = torch.tensor([[[3., 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]])
    //   t.transpose(-1, -2).narrow(1, 1, 2)
    assert_eq!(
        tensor.t()?.narrow(1, 1, 2)?.to_vec3_async::<f32>().await?,
        &[[[1.0, 5.0], [4.0, 9.0]], [[1.0, 2.0], [7.0, 8.0]]],
    );
    Ok(())
}

async fn broadcast(device: &Device) -> Result<()> {
    let data = &[3f32, 1., 4.];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.broadcast_left((3, 1))?.to_vec3_async::<f32>().await?,
        &[[[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]]]
    );
    Ok(())
}

async fn slice_set(device: &Device) -> Result<()> {
    let (b, h, max_t, d) = (2, 4, 7, 3);
    let cache = Tensor::zeros((b, h, max_t, d), DType::F32, device)?;
    let tensor = Tensor::randn(0f32, 1f32, (b, h, 4, d), device)?;
    cache.slice_set(&tensor, 2, 0)?;
    let cache_t = cache.narrow(2, 0, 4)?;
    let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    cache.slice_set(&tensor, 2, 1)?;
    let cache_t = cache.narrow(2, 1, 4)?;
    let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    let ones = Tensor::ones((b, h, 1, d), DType::F32, device)?;
    cache.slice_set(&ones, 2, 6)?;
    let diff = cache.narrow(2, 5, 1)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    let diff = (cache.narrow(2, 6, 1)? - 1.)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    Ok(())
}

async fn cat(device: &Device) -> Result<()> {
    // 1D
    let t1 = Tensor::new(&[3f32, 1., 4.], device)?;
    let t2 = Tensor::new(&[1f32, 5., 9., 2.], device)?;
    let t3 = Tensor::new(&[6f32, 5., 3., 5., 8., 9.], device)?;
    assert_eq!(Tensor::cat(&[&t1], 0)?.to_vec1_async::<f32>().await?, [3f32, 1., 4.],);
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 0)?.to_vec1_async::<f32>().await?,
        [3f32, 1., 4., 1., 5., 9., 2.],
    );
    assert_eq!(
        Tensor::cat(&[&t1, &t2, &t3], 0)?.to_vec1_async::<f32>().await?,
        [3f32, 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8., 9.],
    );

    // 2D
    let data = &[[3f32, 1., 4., 1., 5.], [2., 7., 1., 8., 2.]];
    let t1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 7., 1., 8., 2.]];
    let t2 = Tensor::new(data2, device)?;
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 0)?.to_vec2_async::<f32>().await?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    // PyTorch equivalent:
    //     import torch
    //     t1 = torch.tensor([[3, 1, 4, 1, 5], [2, 7, 1, 8, 2]])
    //     t2 = torch.tensor([[5]*5, [2, 7, 1, 8, 2]])
    //     torch.cat([t1.t(), t2.t()], dim=1).t()
    assert_eq!(
        Tensor::cat(&[&t1.t()?, &t2.t()?], 1)?
            .t()?
            .to_vec2_async::<f32>().await?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 1)?.to_vec2_async::<f32>().await?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0, 2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );

    if device.is_dtype_available(DType::I64){
        // 3D
        let t1 = Tensor::arange(0, 48i64, device)?.reshape((2, 6, 4))?;
        let t2 = Tensor::arange(100, 124i64, device)?.reshape((2, 3, 4))?;
        let t3 = Tensor::arange(10000, 10032i64, device)?.reshape((2, 4, 4))?;

        let t_cat = Tensor::cat(&[&t1, &t2, &t3], 1)?;

        let t1 = t1.t()?.contiguous()?.t()?;
        let t2 = t2.t()?.contiguous()?.t()?;
        let t3 = t3.t()?.contiguous()?.t()?;
        let t_cat2 = Tensor::cat(&[&t1, &t2, &t3], 1)?;

        let diff = t_cat.eq(&t_cat2)?.to_dtype(DType::F32)?.sum_all()?;
        assert_eq!(diff.to_vec0_async::<f32>().await?, 104.0);
        assert_eq!(t_cat.i((0, 0, 0))?.to_vec0_async::<i64>().await?, 0);
        assert_eq!(t_cat.i((0, 4, 0))?.to_vec0_async::<i64>().await?, 16);
        assert_eq!(t_cat.i((0, 5, 0))?.to_vec0_async::<i64>().await?, 20);
        assert_eq!(t_cat.i((1, 5, 0))?.to_vec0_async::<i64>().await?, 44);
        assert_eq!(t_cat.i((0, 6, 0))?.to_vec0_async::<i64>().await?, 100);
        assert_eq!(t_cat.i((1, 6, 0))?.to_vec0_async::<i64>().await?, 112);
        assert_eq!(t_cat.i((0, 6, 1))?.to_vec0_async::<i64>().await?, 101);
        assert_eq!(t_cat.i((0, 7, 1))?.to_vec0_async::<i64>().await?, 105);
        assert_eq!(t_cat.i((0, 12, 1))?.to_vec0_async::<i64>().await?, 10013);
        assert_eq!(t_cat.i((1, 12, 3))?.to_vec0_async::<i64>().await?, 10031);                                       
    }
    
    Ok(())
}

async fn embeddings(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let hs = t.embedding(&ids)?;
    assert_eq!(hs.to_vec2_async::<f32>().await?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(hs.to_vec2_async::<f32>().await?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    if device.is_dtype_available(DType::I64){
        let hs = t.index_select(&ids.to_dtype(DType::I64)?, 0)?;
        assert_eq!(hs.to_vec2_async::<f32>().await?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    }
    Ok(())
}

async fn cmp(device: &Device) -> Result<()> {
    let t1 = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let t2 = Tensor::new(&[[1f32, 0f32], [3f32, 3f32], [4f32, 7f32]], device)?;
    assert_eq!(t1.eq(&t2)?.to_vec2_async::<u8>().await?, &[[0, 0], [0, 1], [1, 0]]);
    assert_eq!(t1.ne(&t2)?.to_vec2_async::<u8>().await?, &[[1, 1], [1, 0], [0, 1]]);
    assert_eq!(t1.le(&t2)?.to_vec2_async::<u8>().await?, &[[1, 0], [1, 1], [1, 1]]);
    assert_eq!(t1.lt(&t2)?.to_vec2_async::<u8>().await?, &[[1, 0], [1, 0], [0, 1]]);
    assert_eq!(t1.gt(&t2)?.to_vec2_async::<u8>().await?, &[[0, 1], [0, 0], [0, 0]]);
    assert_eq!(t1.ge(&t2)?.to_vec2_async::<u8>().await?, &[[0, 1], [0, 1], [1, 0]]);
    Ok(())
}

async fn index_select(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    for dtype in [DType::U8, DType::U32, DType::I64] {
        if device.is_dtype_available(dtype){
            let ids = ids.to_dtype(dtype)?;
            let hs = t.index_select(&ids, 1)?;
            assert_eq!(
                hs.to_vec2_async::<f32>().await?,
                &[
                    [0.0, 2.0, 1.0],
                    [3.0, 5.0, 4.0],
                    [6.0, 8.0, 7.0],
                    [9.0, 11.0, 10.0]
                ]
            );
            let hs = t.index_select(&ids, 0)?;
            assert_eq!(
                hs.to_vec2_async::<f32>().await?,
                &[[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [3.0, 4.0, 5.0]]
            );
        }
    }

    // Prior to https://github.com/huggingface/candle/pull/1022
    // There would be a bug where the last values in the result tensor would be set to 0.
    let ids = Tensor::new(&[0u32, 2u32, 1u32, 0u32, 2u32, 1u32], device)?;
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(
        hs.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [6.0, 7.0, 8.0],
            [3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0],
            [6.0, 7.0, 8.0],
            [3.0, 4.0, 5.0],
        ]
    );

    // Test when selecting dim > 0 with ids size different from elem count of
    // target dim in source/input.
    let ids = Tensor::new(&[1u32, 0u32, 1u32], device)?;
    let t = Tensor::arange(1f32, 5f32, device)?.reshape((2, 2))?;
    assert_eq!(t.to_vec2_async::<f32>().await?, &[[1.0, 2.0], [3.0, 4.0]]);
    let hs = t.index_select(&ids, 1)?;
    assert_eq!(hs.to_vec2_async::<f32>().await?, &[[2.0, 1.0, 2.0], [4.0, 3.0, 4.0]]);

    Ok(())
}

async fn index_add(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 1u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let init = Tensor::ones((4, 2), DType::F32, device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::<f32>().await?,
        &[[1.0, 4.0], [4.0, 10.0], [7.0, 16.0], [10.0, 22.0]],
    );
    let init = Tensor::zeros((4, 2), DType::F32, device)?;
    let ids = Tensor::new(&[1u32, 0u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::<f32>().await?,
        &[[3.0, 0.0], [9.0, 3.0], [15.0, 6.0], [21.0, 9.0]],
    );

    let init = Tensor::zeros((6, 3), DType::F32, device)?;
    let ids = Tensor::new(&[5u32, 0u32, 1u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2_async::<f32>().await?,
        &[
            [12.0, 14.0, 16.0],
            [6.0, 7.0, 8.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0]
        ]
    );
    Ok(())
}

async fn slice_scatter(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let src = Tensor::arange(100f32, 106f32, device)?.reshape((2, 3))?;
    assert_eq!(
        t.slice_scatter0(&src, 0)?.to_vec2_async::<f32>().await?,
        &[
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    assert_eq!(
        t.slice_scatter0(&src, 1)?.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
            [9.0, 10.0, 11.0]
        ]
    );
    assert_eq!(
        t.slice_scatter0(&src, 2)?.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
        ]
    );
    Ok(())
}

async fn scatter_add(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let ids = Tensor::new(&[[0u32, 1, 2], [3, 4, 0], [3, 3, 1], [2, 0, 4]], device)?;
    let init = Tensor::ones((4, 5), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::<f32>().await?,
        &[
            [1.0, 2.0, 3.0, 1.0, 1.0],
            [6.0, 1.0, 1.0, 4.0, 5.0],
            [1.0, 9.0, 1.0, 14.0, 1.0],
            [11.0, 1.0, 10.0, 1.0, 12.0]
        ]
    );

    let init = Tensor::ones((6, 3), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2_async::<f32>().await?,
        &[
            [1.0, 11.0, 6.0],
            [1.0, 2.0, 9.0],
            [10.0, 1.0, 3.0],
            [10.0, 8.0, 1.0],
            [1.0, 5.0, 12.0],
            [1.0, 1.0, 1.0]
        ]
    );
    Ok(())
}

async fn gather(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[[0u32], [2u32], [1u32], [0u32]], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let hs = t.gather(&ids, 1)?;
    assert_eq!(hs.to_vec2_async::<f32>().await?, &[[0.0], [5.0], [7.0], [9.0]]);
    let ids = Tensor::new(
        &[[0u32, 0u32], [2u32, 0u32], [1u32, 1u32], [0u32, 2u32]],
        device,
    )?;
    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec2_async::<f32>().await?,
        &[[0.0, 0.0], [5.0, 3.0], [7.0, 7.0], [9.0, 11.0]]
    );
    let ids = Tensor::new(&[[0u32, 2u32, 0u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(hs.to_vec2_async::<f32>().await?, &[[0.0, 7.0, 2.0]]);
    let ids = Tensor::new(&[[0u32, 2u32, 0u32], [0u32, 1u32, 1u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(hs.to_vec2_async::<f32>().await?, &[[0.0, 7.0, 2.0], [0.0, 4.0, 5.0]]);

    // Random data

    // Dim: 0
    let t = Tensor::new(
        &[
            [
                [108_f32, -47., 16., -56., -83., -130., 210.],
                [253., 95., 151., 228., -210., -123., -127.],
                [-9., -217., 2., -78., 163., 245., -204.],
                [-246., 79., -238., 88., -226., -184., 171.],
                [8., -48., -153., 234., -34., 166., -153.],
                [124., 0., -10., -61., -242., -15., -238.],
            ],
            [
                [12., -64., -199., 244., -240., 156., -128.],
                [173., -57., 4., -198., 233., -110., 238.],
                [95., 82., 0., 240., 53., -211., 209.],
                [-122., 167., -212., 227., -144., 61., 118.],
                [-63., -146., 200., 244., 168., -167., 116.],
                [-125., -147., 110., -253., -178., -250., -18.],
            ],
            [
                [57., 86., -50., 56., 92., 205., -78.],
                [-137., -156., -18., 248., -61., -239., 14.],
                [-248., -30., -50., -70., -251., 250., -83.],
                [-221., 67., 72., 59., -24., -154., 232.],
                [-144., -23., -74., 5., 93., 171., 205.],
                [46., -77., -38., -226., 246., 161., -17.],
            ],
            [
                [-153., -231., -236., 161., 126., 2., -22.],
                [-229., -41., 209., 164., 234., 160., 57.],
                [223., 254., -186., -162., -46., -160., -102.],
                [65., 30., 213., -253., 59., 224., -154.],
                [-82., -203., -177., 17., 31., -256., -246.],
                [176., -135., -65., 54., -56., 210., 76.],
            ],
            [
                [-10., -245., 168., 124., -14., -33., -178.],
                [25., -43., -39., 132., -89., 169., 179.],
                [187., -215., 32., -133., 87., -7., -168.],
                [-224., -215., -5., -230., -58., -162., 128.],
                [158., -137., -122., -100., -202., -83., 136.],
                [30., -185., -144., 250., 209., -40., 127.],
            ],
            [
                [-196., 108., -245., 122., 146., -228., 62.],
                [-1., -66., 160., 137., 13., -172., -21.],
                [244., 199., -164., 28., 119., -175., 198.],
                [-62., 253., -162., 195., -95., -230., -211.],
                [123., -72., -26., -107., -139., 64., 245.],
                [11., -126., -182., 108., -12., 184., -127.],
            ],
            [
                [-159., 126., 176., 161., 73., -111., -138.],
                [-187., 214., -217., -33., -223., -201., -212.],
                [-61., -120., -166., -172., -95., 53., 196.],
                [-33., 86., 134., -152., 154., -53., 74.],
                [186., -28., -154., -174., 141., -109., 217.],
                [82., 35., 252., 145., 181., 74., -87.],
            ],
        ],
        device,
    )?;

    let ids = Tensor::new(
        &[
            [
                [6_u32, 6, 4, 3, 4, 4, 6],
                [3, 3, 2, 4, 4, 4, 6],
                [3, 3, 0, 2, 4, 6, 4],
                [2, 5, 1, 2, 6, 6, 1],
                [2, 1, 6, 5, 3, 2, 3],
                [6, 1, 0, 1, 0, 2, 6],
            ],
            [
                [4, 6, 4, 3, 3, 3, 2],
                [4, 3, 2, 4, 4, 4, 6],
                [2, 3, 0, 2, 4, 6, 4],
                [6, 5, 1, 2, 6, 6, 1],
                [4, 1, 6, 5, 3, 2, 3],
                [1, 1, 0, 1, 0, 2, 6],
            ],
            [
                [3, 6, 4, 3, 3, 3, 2],
                [2, 3, 2, 4, 4, 4, 6],
                [4, 3, 0, 2, 4, 6, 4],
                [0, 5, 1, 2, 6, 6, 1],
                [6, 1, 6, 5, 3, 2, 3],
                [4, 1, 0, 1, 0, 2, 6],
            ],
            [
                [0, 6, 4, 3, 3, 3, 2],
                [5, 3, 2, 4, 4, 4, 6],
                [0, 3, 0, 2, 4, 6, 4],
                [3, 5, 1, 2, 6, 6, 1],
                [0, 1, 6, 5, 3, 2, 3],
                [3, 1, 0, 1, 0, 2, 6],
            ],
        ],
        device,
    )?;

    let hs = t.gather(&ids, 0)?;
    assert_eq!(
        hs.to_vec3_async::<f32>().await?,
        &[
            [
                [-159_f32, 126., 168., 161., -14., -33., -138.],
                [-229., -41., -18., 132., -89., 169., -212.],
                [223., 254., 2., -70., 87., 53., -168.],
                [-221., 253., -212., 59., 154., -53., 118.],
                [-144., -146., -154., -107., 31., 171., -246.],
                [82., -147., -10., -253., -242., 161., -87.]
            ],
            [
                [-10., 126., 168., 161., 126., 2., -78.],
                [25., -41., -18., 132., -89., 169., -212.],
                [-248., 254., 2., -70., 87., 53., -168.],
                [-33., 253., -212., 59., 154., -53., 118.],
                [158., -146., -154., -107., 31., 171., -246.],
                [-125., -147., -10., -253., -242., 161., -87.]
            ],
            [
                [-153., 126., 168., 161., 126., 2., -78.],
                [-137., -41., -18., 132., -89., 169., -212.],
                [187., 254., 2., -70., 87., 53., -168.],
                [-246., 253., -212., 59., 154., -53., 118.],
                [186., -146., -154., -107., 31., 171., -246.],
                [30., -147., -10., -253., -242., 161., -87.]
            ],
            [
                [108., 126., 168., 161., 126., 2., -78.],
                [-1., -41., -18., 132., -89., 169., -212.],
                [-9., 254., 2., -70., 87., 53., -168.],
                [65., 253., -212., 59., 154., -53., 118.],
                [8., -146., -154., -107., 31., 171., -246.],
                [176., -147., -10., -253., -242., 161., -87.]
            ]
        ]
    );

    // Dim: 1
    let t = Tensor::new(
        &[
            [
                [-117_f32, -175., 69., -163.],
                [200., 242., -21., -67.],
                [179., 150., -126., -75.],
                [-118., 38., -138., -13.],
                [-221., 136., -185., 180.],
                [58., 182., -204., -149.],
            ],
            [
                [3., -148., -58., -154.],
                [-43., 45., -108., 4.],
                [-69., -249., -71., -21.],
                [80., 110., -152., -235.],
                [-88., 7., 92., -250.],
                [-186., 207., -242., 98.],
            ],
            [
                [238., 19., 64., -242.],
                [-150., -97., 218., 58.],
                [111., -233., 204., -212.],
                [-242., -232., 83., 42.],
                [153., 62., -251., 219.],
                [-117., 36., -119., 10.],
            ],
            [
                [215., 159., -169., -27.],
                [-83., 101., -88., 169.],
                [-205., 93., 225., -64.],
                [-162., 240., 214., 23.],
                [-112., 6., 21., 245.],
                [-38., 113., 93., 215.],
            ],
            [
                [91., -188., -148., 101.],
                [74., 203., -35., 55.],
                [-116., -130., -153., -96.],
                [58., 22., -45., -194.],
                [-221., -134., 73., 159.],
                [-203., -254., 31., 235.],
            ],
            [
                [105., -53., 61., 186.],
                [-195., 234., 75., -1.],
                [51., 139., 160., -108.],
                [-173., -167., 161., 19.],
                [83., -246., 156., -222.],
                [109., 39., -149., 137.],
            ],
        ],
        device,
    )?;

    let ids = Tensor::new(
        &[
            [[4_u32, 4, 4, 2]],
            [[0, 4, 4, 3]],
            [[1, 5, 3, 4]],
            [[0, 3, 3, 2]],
            [[1, 1, 5, 2]],
            [[1, 4, 5, 4]],
        ],
        device,
    )?;

    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec3_async::<f32>().await?,
        &[
            [[-221., 136., -185., -75.]],
            [[3., 7., 92., -235.]],
            [[-150., 36., 83., 219.]],
            [[215., 240., 214., -64.]],
            [[74., 203., 31., -96.]],
            [[-195., -246., -149., -222.]]
        ]
    );

    // Dim: 2
    let t = Tensor::new(
        &[
            [[-162_f32, 202.], [-126., -39.], [35., -65.], [1., 80.]],
            [[37., 248.], [-191., 89.], [117., -40.], [-217., 220.]],
        ],
        device,
    )?;

    let ids = Tensor::new(&[[[1_u32], [0], [1], [1]], [[0], [1], [0], [1]]], device)?;

    let hs = t.gather(&ids, 2)?;
    assert_eq!(
        hs.to_vec3_async::<f32>().await?,
        &[
            [[202.], [-126.], [-65.], [80.]],
            [[37.], [89.], [117.], [220.]]
        ]
    );

    let t = Tensor::new(
        &[
            [[-21_f32, -197.], [194., 122.]],
            [[255., -106.], [-191., 250.]],
            [[33., -117.], [43., 10.]],
            [[-130., 238.], [-217., -92.]],
        ],
        device,
    )?;

    let ids = Tensor::new(
        &[
            [[0_u32, 1], [1, 0]],
            [[1, 0], [0, 1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
        ],
        device,
    )?;

    let hs = t.gather(&ids, 2)?;
    assert_eq!(
        hs.to_vec3_async::<f32>().await?,
        &[
            [[-21., -197.], [122., 194.]],
            [[-106., 255.], [-191., 250.]],
            [[33., -117.], [43., 10.]],
            [[238., -130.], [-92., -217.]]
        ]
    );

    Ok(())
}

async fn broadcasting(device: &Device) -> Result<()> {
    let t1 = Tensor::arange(0f32, 24f32, device)?.reshape((4, 2, 3))?;
    let t2 = Tensor::new(&[100f32, 200f32], device)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[100.0, 101.0, 102.0], [203.0, 204.0, 205.0]],
            [[106.0, 107.0, 108.0], [209.0, 210.0, 211.0]],
            [[112.0, 113.0, 114.0], [215.0, 216.0, 217.0]],
            [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]
        ]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[100.0, 203.0], [101.0, 204.0], [102.0, 205.0]],
            [[106.0, 209.0], [107.0, 210.0], [108.0, 211.0]],
            [[112.0, 215.0], [113.0, 216.0], [114.0, 217.0]],
            [[118.0, 221.0], [119.0, 222.0], [120.0, 223.0]]
        ]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[-100.0, -99.0, -98.0], [-197.0, -196.0, -195.0]],
            [[-94.0, -93.0, -92.0], [-191.0, -190.0, -189.0]],
            [[-88.0, -87.0, -86.0], [-185.0, -184.0, -183.0]],
            [[-82.0, -81.0, -80.0], [-179.0, -178.0, -177.0]]
        ]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[-100.0, -197.0], [-99.0, -196.0], [-98.0, -195.0]],
            [[-94.0, -191.0], [-93.0, -190.0], [-92.0, -189.0]],
            [[-88.0, -185.0], [-87.0, -184.0], [-86.0, -183.0]],
            [[-82.0, -179.0], [-81.0, -178.0], [-80.0, -177.0]]
        ]
    );
    // Test a narrowed version as this uses a layout start_offset.
    let t1 = t1.i(2..)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[112.0, 113.0, 114.0], [215.0, 216.0, 217.0]],
            [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]
        ]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[112.0, 215.0], [113.0, 216.0], [114.0, 217.0]],
            [[118.0, 221.0], [119.0, 222.0], [120.0, 223.0]]
        ]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[-88.0, -87.0, -86.0], [-185.0, -184.0, -183.0]],
            [[-82.0, -81.0, -80.0], [-179.0, -178.0, -177.0]]
        ]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[-88.0, -185.0], [-87.0, -184.0], [-86.0, -183.0]],
            [[-82.0, -179.0], [-81.0, -178.0], [-80.0, -177.0]]
        ]
    );
    let t3 = Tensor::new(1f32, device)?.broadcast_div(&t2)?;
    let s = t1.broadcast_mul(&t2.reshape((2, 1))?)?;
    let s_div = t1.broadcast_div(&t3.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[1200.0, 1300.0, 1400.0], [3000.0, 3200.0, 3400.0]],
            [[1800.0, 1900.0, 2000.0], [4200.0, 4400.0, 4600.0]]
        ]
    );
    assert_eq!(s.to_vec3_async::<f32>().await?, s_div.to_vec3_async::<f32>().await?,);
    let s = t1.t()?.broadcast_mul(&t2)?;
    let s_div = t1.t()?.broadcast_div(&t3)?;
    assert_eq!(
        s.to_vec3_async::<f32>().await?,
        &[
            [[1200.0, 3000.0], [1300.0, 3200.0], [1400.0, 3400.0]],
            [[1800.0, 4200.0], [1900.0, 4400.0], [2000.0, 4600.0]]
        ]
    );
    assert_eq!(s.to_vec3_async::<f32>().await?, s_div.to_vec3_async::<f32>().await?,);
    Ok(())
}

async fn randn(device: &Device) -> Result<()> {
    let tensor = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    // Check that the seed gets updated by checking that
    // a new series of numbers is generated each time
    let tensor2 = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_ne!(tensor.to_vec2_async::<f32>().await?, tensor2.to_vec2_async::<f32>().await?);
    let tensor = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    // Check that the seed gets updated by checking that
    // a new series of numbers is generated each time
    let tensor2 = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_ne!(tensor.to_vec2_async::<f32>().await?, tensor2.to_vec2_async::<f32>().await?);
    // We do not expect deterministic elements at any index.
    // There once was a bug that had a deterministic zero element in evenly sized tensors.
    const N: usize = 2;

    let mut v = Vec::new();
    for _ in 0..100 {
        let t = Tensor::randn(0f32, 1f32, N, device)?;
        let vec = t.to_vec1_async::<f32>().await?;
        v.push(vec);
    }
   
    assert!(
        (0..N).all(|i| v.windows(2).any(|pair| pair[0][i] != pair[1][i])),
        "There are deterministic values in the randn tensors"
    );

    let mut v = Vec::new();
    for _ in 0..100 {
        let t = Tensor::randn(0f32, 1f32, N, device)?;
        let vec = t.to_vec1_async::<f32>().await?;
        v.push(vec);
    }

    assert!(
        (0..N).all(|i| v.windows(2).any(|pair| pair[0][i] != pair[1][i])),
        "There are deterministic values in the rand tensors"
    );
    Ok(())
}



async fn where_cond(device: &Device) -> Result<()> {
    let cond = Tensor::new(&[0u32, 2u32, 1u32, 0, 0, 0, 35, 255, 53, 0, 29 ,0], device)?.reshape((4,3))?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );

    let t_f = Tensor::arange(12f32, 24f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t_f.to_vec2_async::<f32>().await?,
        &[
            [12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0],
            [18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0]
        ]
    );

    for dtype in [DType::U8, DType::U32, DType::I64] {
        if device.is_dtype_available(dtype){
            let cond = cond.to_dtype(dtype)?;
            let hs = cond.where_cond(&t, &t_f)?;
            assert_eq!(
                hs.to_vec2_async::<f32>().await?,
                &[
                    [12.0, 1.0, 2.0],
                    [15.0, 16.0, 17.0],
                    [6.0, 7.0, 8.0],
                    [21.0, 10.0, 23.0]
                ]
            );
        }
    }
    Ok(())
}


async fn zero_dim(device: &Device) -> Result<()> {
    let t = Tensor::zeros((4, 0, 1), DType::F32, device)?;
    assert_eq!(t.dims3()?, (4, 0, 1));
    let t2 = Tensor::zeros((4, 3, 1), DType::F32, device)?;
    let t_cat = Tensor::cat(&[&t, &t2], 1)?;
    assert_eq!(t_cat.dims3()?, (4, 3, 1));
    let t_cat = Tensor::cat(&[&t, &t], 1)?;
    assert_eq!(t_cat.dims3()?, (4, 0, 1));
    let t_unary = t.sqrt()?;
    assert_eq!(t_unary.dims3()?, (4, 0, 1));
    let t_plus = (&t + 1.)?;
    assert_eq!(t_plus.dims3()?, (4, 0, 1));
    let t_mm = t2.matmul(&t.t()?)?;
    assert_eq!(t_mm.dims3()?, (4, 3, 0));
    let t_mm = t.matmul(&t2.t()?)?;
    assert_eq!(t_mm.dims3()?, (4, 0, 3));
    let t_mm = t.t()?.matmul(&t)?;
    assert_eq!(t_mm.dims3()?, (4, 1, 1));
    Ok(())
}

test_device!(zeros, zeros_cpu, zeros_gpu, zeros_metal,zeros_wgpu);
test_device!(ones, ones_cpu, ones_gpu, ones_metal,ones_wgpu);
test_device!(full, full_cpu, full_gpu, full_metal,full_wgpu);
test_device!(arange, arange_cpu, arange_gpu, arange_metal,arange_wgpu);
test_device!(add_mul, add_mul_cpu, add_mul_gpu, add_mul_metal,add_mul_wgpu);
test_device!(tensor_2d, tensor_2d_cpu, tensor_2d_gpu, tensor_2d_metal,tensor_2d_wgpu);
test_device!(narrow, narrow_cpu, narrow_gpu, narrow_metal,narrow_wgpu);
test_device!(broadcast, broadcast_cpu, broadcast_gpu, broadcast_metal,broadcast_wgpu);
test_device!(slice_set, ss_cpu, ss_gpu, ss_metal, ss_wgpu);
test_device!(cat, cat_cpu, cat_gpu, cat_metal,cat_wgpu);
test_device!(sum, sum_cpu, sum_gpu, sum_metal,sum_wgpu);
test_device!(min, min_cpu, min_gpu, min_metal,min_wgpu);
test_device!(max, max_cpu, max_gpu, max_metal,max_wgpu);
test_device!(argmax, argmax_cpu, argmax_gpu, argmax_metal,argmax_wgpu);
test_device!(argmin, argmin_cpu, argmin_gpu, argmin_metal,argmin_wgpu);
test_device!(transpose, transpose_cpu, transpose_gpu, transpose_metal,transpose_wgpu);
test_device!(unary_op, unary_op_cpu, unary_op_gpu, unary_op_metal,unary_op_wgpu);
test_device!(binary_op, binary_op_cpu, binary_op_gpu, binary_op_metal,binary_op_wgpu);
test_device!(embeddings, embeddings_cpu, embeddings_gpu, embeddings_metal,embeddings_wgpu);
test_device!(cmp, cmp_cpu, cmp_gpu, cmp_metal,cmp_wgpu);
test_device!(
    broadcasting,
    broadcasting_cpu,
    broadcasting_gpu,
    broadcasting_metal,
    broadcasting_wgpu
);
test_device!(
    index_select,
    index_select_cpu,
    index_select_gpu,
    index_select_metal,
    index_select_wgpu
);

test_device!(
    where_cond,
    where_cond_cpu,
    where_cond_gpu,
    where_cond_metal,
    where_cond_wgpu
);
test_device!(index_add, index_add_cpu, index_add_gpu, index_add_metal,index_add_wgpu);
test_device!(gather, gather_cpu, gather_gpu, gather_metal,gather_wgpu);
test_device!(
    scatter_add,
    scatter_add_cpu,
    scatter_add_gpu,
    scatter_add_metal,
    scatter_add_wgpu
);
test_device!(
    slice_scatter,
    slice_scatter_cpu,
    slice_scatter_gpu,
    slice_scatter_metal,
    slice_scatter_wgpu
);
test_device!(randn, randn_cpu, randn_gpu, randn_metal,randn_wgpu);
test_device!(clamp, clamp_cpu, clamp_gpu, clamp_metal,clamp_wgpu);
test_device!(asort, asort_cpu, asort_gpu, asort_metal);
test_device!(var, var_cpu, var_gpu, var_metal,var_wgpu);
test_device!(zero_dim, zero_dim_cpu, zero_dim_gpu, zero_dim_metal,zero_dim_wgpu);

// There was originally a bug on the CPU implementation for randn
// https://github.com/huggingface/candle/issues/381
#[test]
async fn randn_hasneg() -> Result<()> {
    let t = Tensor::randn(0f32, 1f32, 200, &Device::Cpu)?.to_vec1_async::<f32>().await?;
    if t.iter().all(|&v| v >= 0.) {
        candle::bail!("all values in tensors are non-negative")
    }
    Ok(())
}

#[test]
async fn pad_with_same() -> Result<()> {
    let t = Tensor::arange(1f32, 5f32, &Device::Cpu)?.reshape((2, 2))?;
    let t0 = t.pad_with_same(0, 1, 2)?;
    assert_eq!(
        t0.to_vec2_async::<f32>().await?,
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]
    );
    let t1 = t.pad_with_same(1, 1, 2)?;
    assert_eq!(
        t1.to_vec2_async::<f32>().await?,
        [[1.0, 1.0, 2.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0, 4.0]]
    );
    Ok(())
}

#[test]
async fn i64_abs() -> Result<()> {
    let t = Tensor::new(&[-42i64, 1337], &Device::Cpu)?;
    let t = t.abs()?;
    assert_eq!(t.to_vec1_async::<i64>().await?, [42, 1337]);
    Ok(())
}

#[test]
async fn tril_triu_eye() -> Result<()> {
    let t = Tensor::tril2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0]
        ],
    );
    let t = Tensor::triu2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    let t = Tensor::eye(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::<f32>().await?,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    Ok(())
}

#[test]
async fn cumsum() -> Result<()> {
    let t = &[3f32, 1., 4., 1., 5.];
    let t = Tensor::new(t, &Device::Cpu)?;
    assert_eq!(t.cumsum(0)?.to_vec1_async::<f32>().await?, [3., 4., 8., 9., 14.]);
    let t = t.unsqueeze(1)?;
    assert_eq!(
        t.cumsum(0)?.to_vec2_async::<f32>().await?,
        [[3.0], [4.0], [8.0], [9.0], [14.0]]
    );
    assert_eq!(
        t.cumsum(1)?.to_vec2_async::<f32>().await?,
        [[3.0], [1.0], [4.0], [1.0], [5.0]]
    );
    let t = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let t = Tensor::new(t, &Device::Cpu)?;
    assert_eq!(
        t.cumsum(1)?.to_vec2_async::<f32>().await?,
        [[3.0, 4.0, 8.0, 9.0, 14.0], [2.0, 3.0, 10.0, 18.0, 20.0]],
    );
    assert_eq!(
        t.cumsum(0)?.to_vec2_async::<f32>().await?,
        [[3.0, 1.0, 4.0, 1.0, 5.0], [5.0, 2.0, 11.0, 9.0, 7.0]]
    );
    Ok(())
}

/// A helper function for floating point comparison. Both a and b must be 1D Tensor and contains the same amount of data.
/// Assertion passes if the difference of all pairs of a and b is smaller than epsilon.
async fn assert_close(a: &Tensor, b: &Tensor, epsilon: f64) -> Result<()> {
    let a_vec: Vec<f64> = a.to_vec1_async().await?;
    let b_vec: Vec<f64> = b.to_vec1_async().await?;

    assert_eq!(a_vec.len(), b_vec.len());
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        assert!((a - b).abs() < epsilon);
    }
    Ok(())
}

#[test]
async fn log_sum_exp() -> Result<()> {
    let input = Tensor::new(
        &[
            [[1f64, 2., 3.], [4., 5., 6.]],
            [[-1000.0, -999.0, -1001.0], [1000.0, 999.0, 1001.0]],
        ],
        &Device::Cpu,
    )?;

    let output = input.log_sum_exp(D::Minus1)?;
    // The expectations obtained from pytorch.
    let expected = Tensor::new(&[[3.4076, 6.4076], [-998.5924, 1001.4076]], &Device::Cpu)?;
    assert_eq!(output.dims(), expected.dims());
    assert_close(&output.flatten_all()?, &expected.flatten_all()?, 0.00001).await?;

    assert_eq!(
        input.log_sum_exp((0, 1))?.to_vec1_async::<f64>().await?,
        [1000.0, 999.0, 1001.0]
    );
    assert_eq!(
        input.log_sum_exp(())?.to_vec3_async::<f64>().await?,
        input.to_vec3_async::<f64>().await?
    );

    Ok(())
}

#[test]
async fn pow() -> Result<()> {
    let lhs = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let rhs = (&lhs - 2.)?;
    let res = lhs.pow(&rhs)?;
    assert_eq!(
        test_utils::to_vec2_round(&res, 3).await?,
        [[1.0, 1.0, 3.0], [16.0, 125.0, 1296.0]]
    );
    Ok(())
}
