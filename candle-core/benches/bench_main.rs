mod benchmarks;

use criterion::criterion_main;

criterion_main!(
    benchmarks::affine::benches,
    benchmarks::broadcast::benches,
    benchmarks::copy::benches,
    benchmarks::conv_transpose2d::benches,
    benchmarks::matmul::benches,
    benchmarks::qmatmul::benches,
    benchmarks::matmul_wgpu::benches,
    benchmarks::matmul_quantized::benches,
    benchmarks::random::benches,
    benchmarks::reduce::benches,
    benchmarks::unary::benches,
    benchmarks::where_cond::benches,
    benchmarks::conv_transpose2d::benches,
    benchmarks::conv2d::benches,
    benchmarks::qmatmul::benches,
    benchmarks::unary::benches,
    benchmarks::binary::benches,
    benchmarks::copy::benches
);
