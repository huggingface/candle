extern crate rand;

use rand::seq::SliceRandom;

use candle::{error, Device, Result, Tensor, WithDType};

pub struct LLMDataset<T> {
    data: Vec<Vec<T>>,
    device: Device,
}

impl<T> LLMDataset<T> {
    /// Creata a new LLM dataset from a set of tokens.
    pub fn new(data: Vec<Vec<T>>, device: Device) -> Self {
        Self { data, device }
    }

    /// Add a line of data to this dataset.
    pub fn add_line(&mut self, line: Vec<T>) {
        self.data.push(line);
    }
}

pub struct LLMDatasetIter<'a, T> {
    data: &'a LLMDataset<T>,
    indices: Box<dyn Iterator<Item = usize>>,
}

impl<'a, T> LLMDatasetIter<'a, T> {
    /// Create a LLM dataset iterator, which can be shuffled.
    pub fn new(dataset: &'a LLMDataset<T>, shuffle: bool) -> Self {
        let mut indices = (0..dataset.data.len()).collect::<Vec<_>>();
        if shuffle {
            indices.shuffle(&mut rand::thread_rng());
        }
        Self {
            data: dataset,
            indices: Box::new(indices.into_iter()),
        }
    }
}

impl<'a, T: WithDType> Iterator for LLMDatasetIter<'a, T> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.indices.next()?;
        let toks = self.data.data.get(next)?;
        let len = toks.len();
        let inputs = Tensor::from_slice(&toks[..toks.len() - 1], len, &self.data.device);
        let targets = Tensor::from_slice(&toks[1..], len, &self.data.device);
        Some(error::zip(inputs, targets))
    }
}
