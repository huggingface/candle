//! LLMDataset is a dataset for training LLMs. It implements automatic token shifting
//! and provides a method to give the attention mask.

extern crate rand;

use rand::seq::SliceRandom;

use candle::{Device, Tensor};
use tokenizers::{Encoding, Result as TokenizerResult, Tokenizer};

pub struct DatasetLine {
    pub ids: Vec<u32>,
    pub attention_mask: Option<Vec<u32>>,
}

pub struct InputTensors {
    pub ids: Tensor,
    pub attention_mask: Option<Tensor>,
}

pub struct TargetTensors {
    pub ids: Tensor,
    pub attention_mask: Option<Tensor>,
}

pub struct DatasetOutput {
    pub input: InputTensors,
    pub target: TargetTensors,
}

/// A general dataset for all LLMs that automates the task of shifting tokens.
pub struct LLMDataset {
    data: Vec<DatasetLine>,
    device: Device,
    tokenizer: Tokenizer,
}

impl LLMDataset {
    /// Creata a new LLM dataset from a set of tokens and attention masks.
    pub fn new(data: Vec<DatasetLine>, device: Device, tokenizer: Tokenizer) -> Self {
        Self {
            data,
            device,
            tokenizer,
        }
    }

    /// Add a line of data to this dataset. If either bos or eos are None, then they are automatically converted to "".
    pub fn add_line(
        &mut self,
        mut line: String,
        add_special_toks: bool,
        bos_tok: Option<String>,
        eos_tok: Option<String>,
    ) -> TokenizerResult<()> {
        if !add_special_toks {
            line.insert_str(0, &bos_tok.unwrap_or_default());
            line.push_str(&eos_tok.unwrap_or_default());
        }
        let encoded = self.tokenizer.encode(line, add_special_toks)?;
        self.data.push(DatasetLine {
            ids: encoded.get_ids().to_vec(),
            attention_mask: Some(encoded.get_attention_mask().to_vec()),
        });
        Ok(())
    }

    /// Add a line of pre-tokenized data to this dataset.
    pub fn add_line_tokenized(&mut self, encoded: Encoding) -> TokenizerResult<()> {
        self.data.push(DatasetLine {
            ids: encoded.get_ids().to_vec(),
            attention_mask: Some(encoded.get_attention_mask().to_vec()),
        });
        Ok(())
    }

    /// Gets the total number of elements in the dataset. This is calculated lazily.
    pub fn elements(&self) -> usize {
        let mut n = 0;
        self.data.iter().for_each(|l| n += l.ids.len());
        n
    }

    /// Gets the number of lines in the dataset.
    pub fn length(&self) -> usize {
        self.data.len()
    }

    /// Sets the device of this dataset, returning the old device.
    pub fn set_device(&mut self, device: Device) -> Device {
        let old = self.device.clone();
        self.device = device;
        old
    }

    /// Sets the tokenizer for this dataset, returning the old tokenizer.
    pub fn set_tokenizer(&mut self, tokenizer: Tokenizer) -> Tokenizer {
        let old = self.tokenizer.clone();
        self.tokenizer = tokenizer;
        old
    }

    /// Add all the lines from an iterator to this dataset. The iterator must have an item type of `String`.
    ///
    /// When adding lines, add_special_toks is set to true.
    pub fn from_iter<I: IntoIterator<Item = String>>(
        iter: I,
        device: Device,
        tokenizer: Tokenizer,
    ) -> Self {
        let mut this = Self::new(Vec::new(), device, tokenizer);
        for line in iter {
            let _ = this.add_line(line, true, None, None);
        }
        this
    }

    /// Add all the lines from an iterator to this dataset. The iterator must have an item type of `String`.
    ///
    /// When adding lines, add_special_toks is set to false and bos/eos tokens should be specified. If either are None, they are automatically
    /// converted to "".
    pub fn from_iter_bos_eos<I: IntoIterator<Item = String>>(
        iter: I,
        device: Device,
        tokenizer: Tokenizer,
        bos_tok: Option<String>,
        eos_tok: Option<String>,
    ) -> Self {
        let mut this = Self::new(Vec::new(), device, tokenizer);
        for line in iter {
            let _ = this.add_line(line, false, bos_tok.clone(), eos_tok.clone());
        }
        this
    }

    /// Add all the lines from an iterator to this dataset. The iterator must have an item type of `Encoding`.
    pub fn from_iter_tokenized<I: IntoIterator<Item = Encoding>>(
        iter: I,
        device: Device,
        tokenizer: Tokenizer,
    ) -> Self {
        let mut this = Self::new(Vec::new(), device, tokenizer);
        for line in iter {
            let _ = this.add_line_tokenized(line);
        }
        this
    }
}

/// A LLMDatasetIter is tied to the lifetime of the LLMDataset and has an immutable reference,
/// so it is not possible to add rows while the LLMDatasetIter is in scope.
///
/// If any of the attention masks in a batch are None, the attention mask field will be None.
pub struct LLMDatasetIter<'a> {
    data: &'a LLMDataset,
    indices: Box<dyn Iterator<Item = Vec<usize>>>,
}

impl<'a> LLMDatasetIter<'a> {
    /// Create a non-shuffled LLM dataset iterator.
    ///
    /// A LLM dataset iter will return the inputs, targets, and their attention masks. All tokens
    /// are automatically shifted.
    pub fn new(dataset: &'a LLMDataset, batch_size: usize) -> Self {
        let indices = (0..dataset.data.len()).collect::<Vec<_>>();

        let indices = indices[..]
            .chunks(batch_size)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        Self {
            data: dataset,
            indices: Box::new(indices.into_iter()),
        }
    }

    /// Create a shuffled LLM dataset iterator.
    ///
    /// A LLM dataset iter will return the inputs, targets, and their attention masks. All tokens
    /// are automatically shifted.
    pub fn new_shuffled(dataset: &'a LLMDataset, batch_size: usize) -> Self {
        let mut indices = (0..dataset.data.len()).collect::<Vec<_>>();
        indices.shuffle(&mut rand::thread_rng());

        let indices = indices[..]
            .chunks(batch_size)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        Self {
            data: dataset,
            indices: Box::new(indices.into_iter()),
        }
    }
}

impl<'a> Iterator for LLMDatasetIter<'a> {
    type Item = DatasetOutput;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.indices.next()?;
        let mut inputs = Vec::new();
        let mut input_masks = Vec::new();
        let mut targets = Vec::new();
        let mut target_masks = Vec::new();

        for line in next {
            let line = self.data.data.get(line)?;
            let ids = &line.ids;
            let attention_mask = &line.attention_mask;

            let len = ids.len() - 1;
            let input = Tensor::from_slice(&ids[..len], len, &self.data.device).ok()?;
            let input_mask = if let Some(mask) = attention_mask {
                Some(
                    Tensor::from_slice(&mask[..len], len, &self.data.device)
                        .ok()?
                        .unsqueeze(0)
                        .ok()?,
                )
            } else {
                None
            };

            let target = Tensor::from_slice(&ids[1..], len, &self.data.device).ok()?;
            let target_mask = if let Some(mask) = attention_mask {
                Some(
                    Tensor::from_slice(&mask[1..], len, &self.data.device)
                        .ok()?
                        .unsqueeze(0)
                        .ok()?,
                )
            } else {
                None
            };
            inputs.push(input.unsqueeze(0).ok()?);
            targets.push(target.unsqueeze(0).ok()?);
            input_masks.push(input_mask);
            target_masks.push(target_mask);
        }

        let input_masks: Option<Vec<Tensor>> = input_masks.into_iter().collect::<Option<Vec<_>>>();
        let target_masks: Option<Vec<Tensor>> =
            target_masks.into_iter().collect::<Option<Vec<_>>>();

        let inputs = Tensor::cat(&inputs[..], 0).ok()?;
        let targets = Tensor::cat(&targets[..], 0).ok()?;
        let input_masks = if let Some(masks) = input_masks {
            Some(Tensor::cat(&masks[..], 0).ok()?)
        } else {
            None
        };
        let target_masks = if let Some(masks) = target_masks {
            Some(Tensor::cat(&masks[..], 0).ok()?)
        } else {
            None
        };

        Some(DatasetOutput {
            input: InputTensors {
                ids: inputs,
                attention_mask: input_masks,
            },
            target: TargetTensors {
                ids: targets,
                attention_mask: target_masks,
            },
        })
    }
}
