use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType};
use candle_core::Device;

fn main() {
    let raw_data: [u8; 1] = [0x1; 1];
    let dim: Vec<usize> = Vec::from([1usize, 2, 3, 4]);
    let res = qtensor_from_ggml(GgmlDType::F32, &raw_data, dim, &Device::Cpu);
    println!("{:?}", res.unwrap().data());
}
