// Keep the benchmark-only reporter outside candle-core's public library while still
// compiling and exercising its pure parsing and rendering logic under `cargo test`.
#[allow(dead_code)]
#[path = "../benches/benchmarks/qmatmul/report.rs"]
mod report;
