use candle_kernels_build::NvccBuilder;

fn main() {
    NvccBuilder::new()
        .arg("--expt-relaxed-constexpr")
        .build("src/kernels.rs");
}
