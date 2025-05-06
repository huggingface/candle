mod benchmarks;

use criterion::criterion_main;

criterion_main!(
    benchmarks::matmul::benches,
    benchmarks::affine::benches,
    benchmarks::random::benches,
    benchmarks::reduce::benches,
    benchmarks::where_cond::benches,
    benchmarks::conv_transpose2d::benches,
    benchmarks::qmatmul::benches,
    benchmarks::unary::benches
);
