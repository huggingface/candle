//! Helper functions for the tinystories dataset. This uses the pre-tokenized version as generated
//! by the tools from https://github.com/karpathy/llama2.c
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

    pub fn train_tokens(&self) -> usize {
        self.train_tokens.len()
    }

    pub fn valid_tokens(&self) -> usize {
        self.valid_tokens.len()
    }
}

pub struct DatasetRandomIter<'a> {
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

impl Iterator for DatasetRandomIter<'_> {
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
