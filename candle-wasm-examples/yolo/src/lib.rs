mod app;
pub mod coco_classes;
pub mod model;
pub mod worker;
pub use app::App;
pub use worker::Worker;

use candle::CpuStorage;
type Tensor = candle::Tensor<CpuStorage>;
type Conv2d = candle_nn::Conv2d<CpuStorage>;
type BatchNorm = candle_nn::BatchNorm<CpuStorage>;
type VarBuilder<'a> = candle_nn::VarBuilder<'a, CpuStorage>;
