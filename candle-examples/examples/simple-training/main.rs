// This should rearch 91.5% accuracy.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle::{DType, Var, D};

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

pub fn main() -> Result<()> {
    let dev = candle::Device::cuda_if_available(0)?;
    let m = candle_nn::vision::mnist::load_dir("data")?;
    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());
    let ws = Var::zeros((IMAGE_DIM, LABELS), DType::F32, &dev)?;
    let bs = Var::zeros(LABELS, DType::F32, &dev)?;
    let sgd = candle_nn::SGD::new(&[&ws, &bs], 0.1);
    for epoch in 1..200 {
        let logits = m.train_images.matmul(&ws)?.broadcast_add(&bs)?;
        let loss = logits.softmax(D::Minus1)?;
        // TODO: let loss = loss.nll_loss(&m.train_labels);
        sgd.backward_step(&loss)?;

        let _test_logits = m.test_images.matmul(&ws)?.broadcast_add(&bs)?;
        /* TODO
        let test_accuracy = test_logits
            .argmax(Some(-1), false)
            .eq_tensor(&m.test_labels)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        */
        let test_accuracy = 0.;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        )
    }
    Ok(())
}
