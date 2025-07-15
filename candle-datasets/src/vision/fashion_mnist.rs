//! Zalando Fashion MNIST dataset.
//! A slightly more difficult dataset that is drop-in compatible with MNIST.
//!
//! Taken from here: https://huggingface.co/datasets/zalando-datasets/fashion_mnist
use candle::Result;

pub fn load() -> Result<crate::vision::Dataset> {
    crate::vision::mnist::load_mnist_like(
        "zalando-datasets/fashion_mnist",
        "refs/convert/parquet",
        "fashion_mnist/test/0000.parquet",
        "fashion_mnist/train/0000.parquet",
    )
}
