// This should rearch 91.5% accuracy.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle::{DType, Tensor, Var, D};

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

fn log_softmax<D: candle::shape::Dim>(xs: &Tensor, d: D) -> candle::Result<Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

fn nll_loss(inp: &Tensor, target: &Tensor) -> candle::Result<Tensor> {
    let b_sz = target.dim(0)?;
    inp.gather(target, 1)?
        .sum_all()?
        .affine(-1f64 / b_sz as f64, 0.)
}

pub fn main() -> Result<()> {
    let dev = candle::Device::cuda_if_available(0)?;
    let m = candle_nn::vision::mnist::load_dir("data")?;
    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());
    let train_labels = m.train_labels;
    let train_images = m.train_images;
    let train_labels_32 = train_labels.to_dtype(DType::U32)?.unsqueeze(1)?;
    let train_labels = train_labels.to_vec1::<u8>()?;
    let train_label_mask = train_labels
        .iter()
        .flat_map(|l| (0..LABELS).map(|i| f32::from(i == *l as usize)))
        .collect::<Vec<_>>();
    let train_label_mask = Tensor::from_vec(train_label_mask, (train_labels.len(), LABELS), &dev)?;
    let ws = Var::zeros((IMAGE_DIM, LABELS), DType::F32, &dev)?;
    let bs = Var::zeros(LABELS, DType::F32, &dev)?;
    let sgd = candle_nn::SGD::new(&[&ws, &bs], 1.0);
    let test_images = m.test_images;
    let test_labels = m.test_labels.to_dtype(DType::U32)?;
    for epoch in 1..200 {
        let logits = train_images.matmul(&ws)?.broadcast_add(&bs)?;
        let log_sm = log_softmax(&logits, D::Minus1)?;
        let loss = (&log_sm * &train_label_mask)?
            .sum_all()?
            .affine(-1f64 / train_images.dim(0)? as f64, 0f64)?;
        sgd.backward_step(&loss)?;

        let test_logits = test_images.matmul(&ws)?.broadcast_add(&bs)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.shape().r1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
        let nll_loss = nll_loss(&log_sm, &train_labels_32)?.to_vec0::<f32>()?;
        println!("{nll_loss}");
    }
    Ok(())
}
