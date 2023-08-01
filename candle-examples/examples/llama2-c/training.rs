#![allow(dead_code)]
#![allow(unused)]
use crate::model::Llama;
use candle::{Device, Result, Tensor};

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

        let mut tokens: Vec<_> = if valid {
            ds.valid_tokens.iter().collect()
        } else {
            ds.train_tokens.iter().collect()
        };
        tokens.shuffle(&mut thread_rng());
        let current_tokens = tokens.pop().unwrap();
        let seq_len_in_bytes = seq_len * 2;
        let mut indexes_in_bytes = (0..current_tokens.len() - seq_len_in_bytes)
            .step_by(seq_len_in_bytes)
            .collect::<Vec<_>>();
        indexes_in_bytes.shuffle(&mut thread_rng());
        Self {
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

        let seq_len = self.seq_len;
        if self.indexes_in_bytes.is_empty() {}
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

fn _eval(_dataset: &Dataset, _model: &Llama) -> Result<()> {
    // use std::io::BufRead;

    // let mut tokens = vec![0u16; bytes.len() / 2];
    // std::io::Cursor::new(bytes).read_u16_into::<LittleEndian>(&mut tokens)?;
    // tokens.into_iter().map(|u| u as u32).collect::<Vec<u32>>();
    // println!("dataset loaded and encoded: {} tokens", tokens.len());

    // let seq_len = model.config.seq_len;
    // let iter = (0..tokens.len()).step_by(seq_len).flat_map(|start_idx| {
    //     if start_idx + seq_len + 1 > tokens.len() {
    //         None
    //     } else {
    //         let tokens = &tokens[start_idx..start_idx + seq_len + 1];
    //         let inputs = Tensor::new(&tokens[..seq_len], &device);
    //         let targets = Tensor::new(&tokens[1..], &device);
    //         Some(inputs.and_then(|inputs| targets.map(|targets| (inputs, targets))))
    //     }
    // });
    // let batch_iter = candle_nn::dataset::Batcher::new_r2(iter).batch_size(args.batch_size);
    // for inp_tgt in batch_iter {
    //     let (inp, tgt) = inp_tgt?;
    //     let logits = model.forward(&inp, 0)?;
    //     let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
    //     println!("{}", loss.to_vec0::<f32>()?);
    // }
    Ok(())
}

pub fn run(args: &crate::TrainingCmd, common_args: &crate::Args) -> Result<()> {
    let _device = candle_examples::device(common_args.cpu)?;
    let dataset = Dataset::new(&args.pretokenized_dir)?;
    println!(
        "loaded dataset, train: {} files, valid: {} files",
        dataset.train_tokens.len(),
        dataset.valid_tokens.len()
    );
    Ok(())
}
