// This should rearch 91.5% accuracy.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle::{DType, IndexOp, Tensor, Var, D};

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

// TODO: Once the index_select backprop is efficient enough, switch to using this.
fn _nll_loss(inp: &Tensor, target: &Tensor) -> candle::Result<Tensor> {
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
    let train_labels = m.train_labels.i(0..100)?;
    let train_images = m.train_images.i(0..100)?;
    let train_labels = train_labels.to_vec1::<u8>()?;
    let train_label_mask = train_labels
        .iter()
        .flat_map(|l| (0..LABELS).map(|i| f32::from(i == *l as usize)))
        .collect::<Vec<_>>();
    let train_label_mask = Tensor::from_vec(train_label_mask, (train_labels.len(), LABELS), &dev)?;
    let ws = Var::zeros((IMAGE_DIM, LABELS), DType::F32, &dev)?;
    let bs = Var::zeros(LABELS, DType::F32, &dev)?;
    let sgd = candle_nn::SGD::new(&[&ws, &bs], 0.1);
    let test_labels = m.test_labels.to_vec1::<u8>()?;
    for epoch in 1..200 {
        let logits = train_images.matmul(&ws)?.broadcast_add(&bs)?;
        let log_sm = log_softmax(&logits, D::Minus1)?;
        let loss = (log_sm * &train_label_mask)?.sum_all()?;
        println!("{loss:?}");
        sgd.backward_step(&loss)?;

        let test_logits = m.test_images.matmul(&ws)?.broadcast_add(&bs)?;
        /* TODO: Add argmax so that the following can be computed within candle.
        let test_accuracy = test_logits
            .argmax(Some(-1), false)
            .eq_tensor(&m.test_labels)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        */
        let test_logits = test_logits.to_vec2::<f32>()?;
        let sum_ok = test_logits
            .iter()
            .zip(test_labels.iter())
            .map(|(logits, label)| {
                let arg_max = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, v1), (_, v2)| v1.total_cmp(v2))
                    .map(|(idx, _)| idx);
                f64::from(arg_max == Some(*label as usize))
            })
            .sum::<f64>();
        let test_accuracy = sum_ok / test_labels.len() as f64;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        )
    }
    Ok(())
}
