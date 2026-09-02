use crate::model::{Cache, Config, Llama};
use candle::{DType, Device, Result};
use candle_datasets::nlp::tinystories::{Dataset, DatasetRandomIter};
use candle_nn::Optimizer;

fn valid_loss(
    dataset: &Dataset,
    model: &Llama,
    args: &crate::TrainingCmd,
    device: &Device,
    cache: &mut Cache,
) -> Result<f64> {
    let iter = DatasetRandomIter::new(dataset, true, model.config.seq_len, device.clone());
    let batch_iter = candle_datasets::Batcher::new_r2(iter).batch_size(args.batch_size);
    let mut sum_ce = 0f64;
    let mut cnt = 0usize;
    for inp_tgt in batch_iter.take(50) {
        let (inp, tgt) = inp_tgt?;
        let logits = model.forward(&inp, 0, cache)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        sum_ce += loss.to_vec0::<f32>()? as f64;
        cnt += 1;
    }
    Ok(sum_ce / cnt as f64)
}

pub fn run(args: &crate::TrainingCmd, common_args: &crate::Args) -> Result<()> {
    let device = candle_examples::device(common_args.cpu)?;
    let dataset = Dataset::new(&args.pretokenized_dir)?;
    println!(
        "loaded dataset, train: {} files, valid: {} files",
        dataset.train_tokens(),
        dataset.valid_tokens()
    );
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = Config::tiny_15m();
    let iter = DatasetRandomIter::new(&dataset, false, config.seq_len, device.clone());
    let batch_iter = candle_datasets::Batcher::new_r2(iter).batch_size(args.batch_size);

    let mut cache = Cache::new(false, &config, vb.pp("rot"))?;
    let model = Llama::load(vb, config)?;
    let params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), params)?;
    for (batch_index, batch) in batch_iter.enumerate() {
        let (inp, tgt) = batch?;
        let logits = model.forward(&inp, 0, &mut cache)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        opt.backward_step(&loss)?;

        if batch_index > 0 && batch_index % 100 == 0 {
            // TODO: Add a way to deactivate the backprop graph tracking when computing the
            // validation loss.
            let loss = valid_loss(&dataset, &model, args, &device, &mut cache)?;
            println!("{batch_index} {loss}");
        }
        if batch_index > 0 && batch_index % 1000 == 0 {
            varmap.save("checkpoint.safetensors")?
        }
    }
    Ok(())
}
