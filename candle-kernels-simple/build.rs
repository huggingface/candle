use candle_kernels_build::NvccBuilder;

fn main() {
    NvccBuilder::new().build("src/kernels.rs");
}
