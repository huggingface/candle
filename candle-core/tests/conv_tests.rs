mod test_utils;
use anyhow::Result;
use candle_core::{Device, Tensor};

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
*/
fn conv1d(dev: &Device) -> Result<()> {
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
    let res = t.conv1d(&w, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [2.6357, -1.3336, 4.1393, -1.1784, 3.5675, 0.5069]
    );
    let res = t.conv1d(&w, /*padding*/ 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 5]);
    // Same as pytorch default padding: use zeros.
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [2.4509, 2.6357, -1.3336, 4.1393, 0.5657, 1.8091, -1.1784, 3.5675, 0.5069, 3.3352]
    );
    Ok(())
}

fn conv1d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[0.4056f32, -0.8689, -0.0773, -1.5630], dev)?.reshape((1, 1, 4))?;
    let w = Tensor::new(&[1f32, 0., 0.], dev)?.reshape((1, 1, 3))?;
    let res = t.conv1d(&w, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.4056, -0.8689]
    );
    let res = t.conv1d(&w, /*padding*/ 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
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
*/
fn conv2d(dev: &Device) -> Result<()> {
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
            -0.8000, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
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
    let res = t.conv2d(&w, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            -4.2812, 2.0923, 5.2187, 7.5184, 0.752, -14.9426, 10.0087, 4.391, 0.2918, 1.6715,
            10.389, 3.6023, -4.2808, 0.2672, 5.3646, -5.2023, -2.1955, -9.4075
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
*/
fn conv2d_small(dev: &Device) -> Result<()> {
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
    let res = t.conv2d(&w, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.164, -0.0111, -0.1742, 2.6437, -2.0268, 1.1823, 3.2855, -1.0324, 0.2539]
    );
    let res = t.conv2d(&w, 2, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 7, 7]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1640, -0.0111, -0.1742, 0.0000, 0.0000,
            0.0000, 0.0000, 2.6437, -2.0268, 1.1823, 0.0000, 0.0000, 0.0000, 0.0000, 3.2855,
            -1.0324, 0.2539, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ]
    );
    Ok(())
}

fn conv2d_smaller(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866,
        ],
        dev,
    )?;
    let w = Tensor::new(&[1f32, 1., 1., 1., 1., 1., 1., 1., 1.], dev)?;
    let t = t.reshape((1, 1, 3, 3))?;
    let w = w.reshape((1, 1, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 1, 1]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [-0.6197]
    );
    Ok(())
}

test_device!(conv1d, conv1d_cpu, conv1d_gpu);
test_device!(conv1d_small, conv1d_small_cpu, conv1d_small_gpu);
test_device!(conv2d, conv2d_cpu, conv2d_gpu);
test_device!(conv2d_small, conv2d_small_cpu, conv2d_small_gpu);
test_device!(conv2d_smaller, conv2d_smaller_cpu, conv2d_smaller_gpu);
