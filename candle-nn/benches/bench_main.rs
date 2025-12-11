mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::layer_norm::benches,
    benchmarks::softmax::benches,
    benchmarks::conv::benches
);
