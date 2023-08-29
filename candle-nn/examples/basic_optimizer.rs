#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Tensor};
use candle_nn::{conv2d, Conv2dConfig, Module, VarBuilder, VarMap};

fn main() {
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &Device::Cpu);
    let c = Conv2dConfig {
        stride: 16,
        ..Default::default()
    };
    let projection = conv2d(3, 192, 3, c, vb.pp("projection")).unwrap();
    let xs = Tensor::ones(&[1, 3, 224, 224], DType::F32, &Device::Cpu).unwrap();
    let out = projection.forward(&xs).unwrap();
    out.backward().unwrap();
}
