extern crate rand;

use rand::seq::SliceRandom;

use candle::{error, Device, Result, Tensor, WithDType};

/// A general dataset for all LLMs that automates the task of shifting tokens.
pub struct LLMDataset<T: WithDType> {
    data: Vec<Vec<T>>,
    device: Device,
}

impl<T: WithDType> LLMDataset<T> {
    /// Creata a new LLM dataset from a set of (any) tokens.
    pub fn new(data: Vec<Vec<T>>, device: Device) -> Self {
        Self { data, device }
    }

    /// Add a line of data to this dataset.
    pub fn add_line(&mut self, line: Vec<T>) {
        self.data.push(line);
    }

    /// Apply a given function over an entire dataset of tokens T.
    pub fn map(&mut self, f: impl Fn(&T) -> T) {
        for line in self.data.iter_mut() {
            for tok in line {
                *tok = f(tok);
            }
        }
    }

    /// Apply a given function over an entire dataset of tokens T which will create a new dataset of type N.
    /// The dataset is reconstructed, but will have the same number of elements. Therefore, it may be a costly
    /// operation.
    pub fn copy_mapped<N: WithDType>(&mut self, f: impl Fn(&T) -> N) -> LLMDataset<N> {
        let mut data = Vec::new();

        for line in self.data.iter() {
            let mut new_lines = Vec::new();
            for tok in line {
                new_lines.push(f(tok));
            }
            data.push(new_lines);
        }

        LLMDataset::new(data, self.device.clone())
    }

    /// Gets the total number of elements in the dataset. This is calculated lazily.
    pub fn elements(&self) -> usize {
        let mut n = 0;
        self.data.iter().for_each(|l| n += l.len());
        n
    }
}

/// A LLMDatasetIter is tied to the lifetime of the LLMDataset and has an immutable reference,
/// so it is not possible to add rows while the LLMDatasetIter is in scope.
pub struct LLMDatasetIter<'a, T: WithDType> {
    data: &'a LLMDataset<T>,
    indices: Box<dyn Iterator<Item = usize>>,
}

impl<'a, T: WithDType> LLMDatasetIter<'a, T> {
    /// Create a LLM dataset iterator, which will iterate in the exact order of it's internal data.
    ///
    /// A LLM dataset iter will return a 2-tuple of (input, target). They are automatically shifted.
    ///
    /// The input and target Tensors are automatically created from each token sequence by
    /// truncating the rightmost token to create the input, and by truncating
    /// the tokens to remove the leftmost token to create the target.
    pub fn new(dataset: &'a LLMDataset<T>) -> Self {
        Self {
            data: dataset,
            indices: Box::new((0..dataset.data.len()).collect::<Vec<_>>().into_iter()),
        }
    }

    /// Create a shuffled LLM dataset iterator.
    ///
    /// A LLM dataset iter will return a 2-tuple of (input, target). They are automatically shifted.
    ///
    /// The input and target Tensors are automatically created from each token sequence by
    /// truncating the rightmost token to create the input, and by truncating
    /// the tokens to remove the leftmost token to create the target.
    pub fn new_shuffled(dataset: &'a LLMDataset<T>) -> Self {
        let mut indices = (0..dataset.data.len()).collect::<Vec<_>>();
        indices.shuffle(&mut rand::thread_rng());
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
        let inputs = Tensor::from_slice(&toks[..len - 1], len, &self.data.device);
        let targets = Tensor::from_slice(&toks[1..], len, &self.data.device);
        Some(error::zip(inputs, targets))
    }
}
