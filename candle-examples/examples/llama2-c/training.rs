#![allow(dead_code)]
#![allow(unused)]
use crate::model::{Cache, Config, Llama};
use candle::{DType, Device, Result, Tensor};

pub struct Dataset {
    valid_tokens: Vec<memmap2::Mmap>,
    train_tokens: Vec<memmap2::Mmap>,
}

fn mmap_file(p: &std::path::PathBuf) -> Result<memmap2::Mmap> {
    let file = std::fs::File::open(p)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    Ok(mmap)
}

impl Dataset {
    pub fn new<P: AsRef<std::path::Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let mut bin_files = vec![];
        for file in std::fs::read_dir(dir)?.flatten() {
            let file = file.path();
            if let Some(extension) = file.extension() {
                if extension == "bin" {
                    bin_files.push(file)
                }
            }
        }
        if bin_files.len() < 2 {
            candle::bail!("found less than two bin files in {:?}", dir)
        }
        bin_files.sort();
        let valid_tokens = mmap_file(&bin_files[0])?;
        let train_tokens = bin_files[1..]
            .iter()
            .map(mmap_file)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            valid_tokens: vec![valid_tokens],
            train_tokens,
        })
    }
}

struct DatasetRandomIter<'a> {
    all_tokens: &'a [memmap2::Mmap],
    tokens: Vec<&'a memmap2::Mmap>,
    current_tokens: &'a memmap2::Mmap,
    indexes_in_bytes: Vec<usize>,
    seq_len: usize,
    device: Device,
}

impl<'a> DatasetRandomIter<'a> {
    pub fn new(ds: &'a Dataset, valid: bool, seq_len: usize, device: Device) -> Self {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let all_tokens = if valid {
            &ds.valid_tokens
        } else {
            &ds.train_tokens
        };
        let mut tokens = all_tokens.iter().collect::<Vec<_>>();
        tokens.shuffle(&mut thread_rng());
        let current_tokens = tokens.pop().unwrap();
        let seq_len_in_bytes = seq_len * 2;
        let mut indexes_in_bytes = (0..current_tokens.len() - seq_len_in_bytes)
            .step_by(seq_len_in_bytes)
            .collect::<Vec<_>>();
        indexes_in_bytes.shuffle(&mut thread_rng());
        Self {
            all_tokens,
            tokens,
            current_tokens,
            indexes_in_bytes,
            seq_len,
            device,
        }
    }
}

impl<'a> Iterator for DatasetRandomIter<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let seq_len = self.seq_len;
        if self.indexes_in_bytes.is_empty() {
            if self.tokens.is_empty() {
                self.tokens = self.all_tokens.iter().collect();
                self.tokens.shuffle(&mut thread_rng());
            }
            self.current_tokens = self.tokens.pop().unwrap();
            let seq_len_in_bytes = self.seq_len * 2;
            self.indexes_in_bytes = (0..self.current_tokens.len() - seq_len_in_bytes)
                .step_by(seq_len_in_bytes)
                .collect::<Vec<_>>();
            self.indexes_in_bytes.shuffle(&mut thread_rng());
        }
        let start_idx = self.indexes_in_bytes.pop().unwrap();
        let bytes = &self.current_tokens[start_idx..start_idx + 2 * (seq_len + 1)];
        let mut tokens = vec![0u16; bytes.len() / 2];
        if let Err(err) = std::io::Cursor::new(bytes).read_u16_into::<LittleEndian>(&mut tokens) {
            return Some(Err(err.into()));
        }
        let tokens = tokens.into_iter().map(|v| v as u32).collect::<Vec<_>>();
        let inputs = Tensor::new(&tokens[..seq_len], &self.device);
        let targets = Tensor::new(&tokens[1..], &self.device);
        Some(candle::error::zip(inputs, targets))
    }
}

fn valid_loss(
    dataset: &Dataset,
    model: &Llama,
    args: &crate::TrainingCmd,
    device: &Device,
) -> Result<f64> {
    let iter = DatasetRandomIter::new(dataset, true, model.config.seq_len, device.clone());
    let batch_iter = candle_nn::dataset::Batcher::new_r2(iter).batch_size(args.batch_size);
    let mut sum_ce = 0f64;
    let mut cnt = 0usize;
    for inp_tgt in batch_iter.take(50) {
        let (inp, tgt) = inp_tgt?;
        let logits = model.forward(&inp, 0)?;
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
        dataset.train_tokens.len(),
        dataset.valid_tokens.len()
    );
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = Config::tiny();
    let iter = DatasetRandomIter::new(&dataset, false, config.seq_len, device.clone());
    let batch_iter = candle_nn::dataset::Batcher::new_r2(iter).batch_size(args.batch_size);

    let cache = Cache::new(false, &config, vb.pp("rot"))?;
    let model = Llama::load(vb, &cache, config)?;
    let params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), params)?;
    for (batch_index, batch) in batch_iter.enumerate() {
        let (inp, tgt) = batch?;
        let logits = model.forward(&inp, 0)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        opt.backward_step(&loss)?;

        if batch_index > 0 && batch_index % 100 == 0 {
            // TODO: Add a way to deactivate the backprop graph tracking when computing the
            // validation loss.
            let loss = valid_loss(&dataset, &model, args, &device)?;
            println!("{batch_index} {loss}");
        }
        if batch_index > 0 && batch_index % 1000 == 0 {
            varmap.save("checkpoint.safetensors")?
        }
    }
    Ok(())
}
