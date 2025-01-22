mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::layer_norm::benches,
    benchmarks::conv::benches,
    benchmarks::attention::benches_fast,
    benchmarks::attention::benches_naive
);
