#![allow(unused_imports, unexpected_cfgs)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, Device, IndexOp, Result, Tensor};
use candle as candle;
async fn contiguous(device: &Device) -> Result<()> {
    let tensor = Tensor::arange(0u32, 24u32, device)?.reshape((2, 3, 4))?;
    assert_eq!(
        tensor.to_vec3_async::< u32 > (). await ?, & [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9,
        10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
    );
    assert_eq!(
        tensor.t() ?.contiguous() ?.to_vec3_async::< u32 > (). await ?, & [[[0, 4, 8],
        [1, 5, 9], [2, 6, 10], [3, 7, 11]], [[12, 16, 20], [13, 17, 21], [14, 18, 22],
        [15, 19, 23]]]
    );
    assert_eq!(
        tensor.transpose(0, 1) ?.contiguous() ?.to_vec3_async::< u32 > (). await ?, &
        [[[0, 1, 2, 3], [12, 13, 14, 15]], [[4, 5, 6, 7], [16, 17, 18, 19]], [[8, 9, 10,
        11], [20, 21, 22, 23]]]
    );
    assert_eq!(
        tensor.transpose(0, 1) ?.flatten_all() ?.to_vec1_async::< u32 > (). await ?, &
        [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21,
        22, 23]
    );
    assert_eq!(
        tensor.i(1..) ? .transpose(0, 1) ? .contiguous() ? .to_vec3_async::< u32 > ().
        await ?, & [[[12, 13, 14, 15]], [[16, 17, 18, 19]], [[20, 21, 22, 23]]]
    );
    assert_eq!(
        tensor.transpose(0, 2) ?.contiguous() ?.to_vec3_async::< u32 > (). await ?, &
        [[[0, 12], [4, 16], [8, 20]], [[1, 13], [5, 17], [9, 21]], [[2, 14], [6, 18],
        [10, 22]], [[3, 15], [7, 19], [11, 23]]]
    );
    Ok(())
}
candle_wasm_tests::test_device!(
    contiguous, contiguous_cpu, contiguous_gpu, contiguous_metal, contiguous_wgpu
);
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
    let tensor = Tensor::arange(0u32, 26u32, &Cpu)?.i(2..)?.reshape((2, 3, 4))?;
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
            assert_eq!(
                tensor.to_vec2_async::< u32 > (). await ?, & [[4, 5, 6, 7], [16, 17, 18,
                19]]
            );
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i((.., 1))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => panic!("unexpected block structure"),
        candle::StridedBlocks::MultipleBlocks { block_len, block_start_index } => {
            assert_eq!(block_len, 4);
            assert_eq!(block_start_index.collect::< Vec < _ >> (), & [4, 16])
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.t()?.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => panic!("unexpected block structure"),
        candle::StridedBlocks::MultipleBlocks { block_start_index, block_len } => {
            assert_eq!(block_len, 1);
            assert_eq!(
                block_start_index.collect::< Vec < _ >> (), & [0, 4, 8, 1, 5, 9, 2, 6,
                10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23]
            )
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.transpose(0, 1)?.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => panic!("unexpected block structure"),
        candle::StridedBlocks::MultipleBlocks { block_start_index, block_len } => {
            assert_eq!(block_len, 4);
            assert_eq!(
                block_start_index.collect::< Vec < _ >> (), & [0, 12, 4, 16, 8, 20]
            )
        }
    };
    Ok(())
}
async fn layout(device: &Device) -> Result<()> {
    let rs: usize = 14;
    let a: usize = 12;
    let b: usize = 13;
    let data1 = Tensor::ones((1, b, a, rs), candle::DType::U32, &Device::Cpu)?;
    let data1 = data1.reshape((1, b, a, rs))?;
    let data2 = data1.to_device_async(device).await?;
    let index1 = data1.i((.., .., 3..6, ..4))?;
    let index2 = data2.i((.., .., 3..6, ..4))?;
    let result1 = index1.reshape((b, 3, 4))?;
    let result2 = index2.reshape((b, 3, 4))?;
    assert_eq!(
        result1.to_vec3_async::< u32 > (). await ?, result2.to_vec3_async::< u32 > ().
        await ?
    );
    let copy1 = index1.copy()?;
    let copy2 = index2.copy()?;
    let result1 = copy1.reshape((b, 3, 4))?;
    let result2 = copy2.reshape((b, 3, 4))?;
    assert_eq!(
        result1.to_vec3_async::< u32 > (). await ?, result2.to_vec3_async::< u32 > ().
        await ?
    );
    let result1 = index1.sum_all()?.to_vec0_async::<u32>().await?;
    let result2 = index2.sum_all()?.to_vec0_async::<u32>().await?;
    assert_eq!(result1, result2);
    Ok(())
}
candle_wasm_tests::test_device!(
    layout, layout_cpu, layout_gpu, layout_metal, layout_wgpu
);
