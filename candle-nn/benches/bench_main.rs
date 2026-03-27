mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::norm::benches,
    benchmarks::softmax::benches,
    benchmarks::conv::benches
);
