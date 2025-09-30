use candle::{CpuStorage, Tensor};

type CpuTensor = Tensor<CpuStorage>;

pub struct Dataset {
    pub train_images: CpuTensor,
    pub train_labels: CpuTensor,
    pub test_images: CpuTensor,
    pub test_labels: CpuTensor,
    pub labels: usize,
}

pub mod cifar;
pub mod fashion_mnist;
pub mod mnist;
