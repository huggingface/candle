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
    let b_sz = target.shape().r1()?;
    inp.index_select(target, 0)?.sum_all()? / b_sz as f64
}

pub fn main() -> Result<()> {
    let dev = candle::Device::cuda_if_available(0)?;
    let m = candle_nn::vision::mnist::load_dir("data")?;
    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());
    let train_labels = m.train_labels.to_dtype(DType::U32)?;
    let ws = Var::zeros((IMAGE_DIM, LABELS), DType::F32, &dev)?;
    let bs = Var::zeros(LABELS, DType::F32, &dev)?;
    let sgd = candle_nn::SGD::new(&[&ws, &bs], 0.1);
    for epoch in 1..200 {
        let logits = m.train_images.matmul(&ws)?.broadcast_add(&bs)?;
        let loss = nll_loss(&log_softmax(&logits, D::Minus1)?, &train_labels)?;
        println!("{loss:?}");
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
