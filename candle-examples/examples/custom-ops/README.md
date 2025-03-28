# candle-custom-ops

 This example illustrates how to implement forward and backward passes for custom operations on the CPU and GPU.
 The custom op in this example implements RMS normalization for the CPU and CUDA.
 
## Running an example

```bash
$ cargo run --example custom-ops

> [[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
>  [ 7.,  8.,  9., 10., 11., 12., 13.]]
> Tensor[[2, 7], f32]
> [[0.0000, 0.2773, 0.5547, 0.8320, 1.1094, 1.3867, 1.6641],
>  [0.6864, 0.7845, 0.8825, 0.9806, 1.0786, 1.1767, 1.2748]]
> Tensor[[2, 7], f32]
```