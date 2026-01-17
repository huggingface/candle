# BERT: Fine-tuning for Next Sentence Prediction (Candle/Rust)

This chapter shows how to fine‑tune a compact BERT‑style encoder for the classic Next Sentence Prediction (NSP) objective using Candle and Rust. It mirrors the style of the other BERT chapters in this series: simple tokenizer, compact BERT encoder, an NSP head, and a training/evaluation/inference loop that runs on various devices.

What you will build:
- A simple whitespace tokenizer and a toy multi‑document corpus
- Positive/negative pair construction for NSP with labels {0: is_next, 1: not_next}
- A compact BERT‑like encoder using Candle components
- An NSP classification head over the pooled [CLS] representation
- Training loop, accuracy metric, predict_is_next helper, and checkpointing

Notes:
- NSP was part of original BERT pretraining; many modern variants (e.g., RoBERTa) drop it. Here it's educational.
- For real work, use robust tokenization (tokenizers crate) and pretrained checkpoints.

## 1. Setup and dependencies

Add the necessary dependencies to your `Cargo.toml`:

```toml
[dependencies]
candle-core = "0.3"
candle-nn = "0.3"
rand = "0.8"
```

```rust
use candle_core::{Device, Result, Tensor, DType, IndexOp};
use candle_nn::{Module, VarBuilder, VarMap, Linear};
use std::collections::HashMap;
use rand::{thread_rng, Rng, seq::SliceRandom};

fn main() -> Result<()> {
    println!("BERT NSP Fine-tuning with Candle");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    Ok(())
}
```

## 2. Simple tokenizer and toy multi‑document corpus

We'll use a lowercase whitespace tokenizer. For NSP, we need short documents (lists of sentences). Positive examples come from consecutive sentence pairs within a document; negative examples pair a sentence with a random sentence from a different place.

```rust
// Special tokens
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[MASK]", 3),
];

// Tiny toy documents (each is a list of sentences)
const DOCS: &[&[&str]] = &[
    &[
        "the cat sat on the mat",
        "the mat was warm",
        "the cat purred softly",
    ],
    &[
        "dogs love to play",
        "they run in the park",
        "afterwards they drink water",
    ],
    &[
        "birds can fly",
        "some birds migrate",
        "they travel long distances",
    ],
];

// NSP Tokenizer
pub struct NSPTokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
    pub documents: Vec<Vec<String>>,
}

impl NSPTokenizer {
    pub fn new() -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Convert documents and count words
        let documents: Vec<Vec<String>> = DOCS.iter()
            .map(|doc| doc.iter().map(|s| s.to_string()).collect())
            .collect();
        
        for doc in &documents {
            for sentence in doc {
                for word in sentence.split_whitespace() {
                    let word = word.to_lowercase();
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
        }
        
        // Build vocabulary
        let mut idx = SPECIALS.len();
        for (word, _count) in word_counts.iter() {
            if !vocab.contains_key(word) {
                vocab.insert(word.clone(), idx);
                idx += 1;
            }
        }
        
        // Create inverse mapping
        let itos: HashMap<usize, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        Self { vocab, itos, documents }
    }
    
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let word = word.to_lowercase();
                self.vocab.get(&word)
                    .copied()
                    .unwrap_or_else(|| self.vocab["[MASK]"])
            })
            .collect()
    }
    
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| self.itos.get(&id).cloned().unwrap_or_else(|| "[UNK]".to_string()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}
```

## 3. NSP pair builder: [CLS] A [SEP] B [SEP]

We will construct batches of sentence pairs. Label 0 means B is the actual next sentence after A in its document; label 1 means B is a random non‑consecutive sentence.

```rust
const MAX_LEN: usize = 48; // allow room for two sentences

impl NSPTokenizer {
    pub fn sample_nsp_pair(&self) -> (Vec<usize>, Vec<usize>, usize) {
        let mut rng = thread_rng();
        let doc_idx = rng.gen_range(0..self.documents.len());
        let doc = &self.documents[doc_idx];
        
        if doc.len() < 2 {
            // Edge case: single sentence document
            let a = self.encode(&doc[0]);
            let b = self.encode(&doc[0]);
            return (a, b, 1); // Mark as negative since it's not truly consecutive
        }
        
        let sent_idx = rng.gen_range(0..doc.len()-1);
        let sentence_a = &doc[sent_idx];
        let a_tokens = self.encode(sentence_a);
        
        if rng.gen_bool(0.5) {
            // Positive: next sentence from same document
            let sentence_b = &doc[sent_idx + 1];
            let b_tokens = self.encode(sentence_b);
            (a_tokens, b_tokens, 0) // 0 = is_next
        } else {
            // Negative: random sentence from different location
            loop {
                let other_doc_idx = rng.gen_range(0..self.documents.len());
                let other_doc = &self.documents[other_doc_idx];
                let other_sent_idx = rng.gen_range(0..other_doc.len());
                
                // Ensure it's not the same sentence or consecutive
                if other_doc_idx != doc_idx || 
                   (other_sent_idx != sent_idx && other_sent_idx != sent_idx + 1) {
                    let sentence_b = &other_doc[other_sent_idx];
                    let b_tokens = self.encode(sentence_b);
                    return (a_tokens, b_tokens, 1); // 1 = not_next
                }
            }
        }
    }
    
    pub fn build_nsp_example(&self, max_len: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>, usize) {
        let (tokens_a, tokens_b, label) = self.sample_nsp_pair();
        
        // Build input: [CLS] + A + [SEP] + B + [SEP]
        let mut input_ids = vec![self.vocab["[CLS]"]];
        
        // Truncate sentences to fit in max_len
        let available_len = max_len - 3; // Account for [CLS] and two [SEP]
        let max_a_len = available_len / 2;
        let max_b_len = available_len - max_a_len;
        
        let a_truncated = if tokens_a.len() > max_a_len {
            &tokens_a[..max_a_len]
        } else {
            &tokens_a
        };
        
        let b_truncated = if tokens_b.len() > max_b_len {
            &tokens_b[..max_b_len]
        } else {
            &tokens_b
        };
        
        input_ids.extend(a_truncated);
        input_ids.push(self.vocab["[SEP]"]);
        let sep_idx = input_ids.len() - 1;
        input_ids.extend(b_truncated);
        input_ids.push(self.vocab["[SEP]"]);
        
        // Token type IDs: 0 for sentence A, 1 for sentence B
        let mut token_type_ids = vec![0; sep_idx + 1];
        token_type_ids.extend(vec![1; input_ids.len() - sep_idx - 1]);
        
        // Attention mask
        let attention_mask = vec![1; input_ids.len()];
        
        // Pad sequences
        let mut padded_ids = input_ids;
        let mut padded_token_types = token_type_ids;
        let mut padded_attention = attention_mask;
        
        while padded_ids.len() < max_len {
            padded_ids.push(self.vocab["[PAD]"]);
            padded_token_types.push(0);
            padded_attention.push(0);
        }
        
        padded_ids.truncate(max_len);
        padded_token_types.truncate(max_len);
        padded_attention.truncate(max_len);
        
        (padded_ids, padded_token_types, padded_attention, label)
    }
}
```

## 4. BERT encoder and NSP head

We'll reuse the BERT encoder from previous chapters and add an NSP head for binary classification.

```rust
// Reuse BertConfig and BertEncoder from previous chapters
use super::bert_finetuning::{BertConfig, BertEncoder};

// NSP Head for next sentence prediction
pub struct BertNSPHead {
    classifier: Linear,
}

impl BertNSPHead {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let classifier = candle_nn::linear(cfg.hidden_size, 2, vb.pp("classifier"))?;
        Ok(Self { classifier })
    }
}

impl Module for BertNSPHead {
    fn forward(&self, pooled_output: &Tensor) -> Result<Tensor> {
        self.classifier.forward(pooled_output)
    }
}

// BERT for Next Sentence Prediction
pub struct BertForNextSentencePrediction {
    encoder: BertEncoder,
    nsp_head: BertNSPHead,
}

impl BertForNextSentencePrediction {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;
        let nsp_head = BertNSPHead::new(cfg, vb.pp("nsp_head"))?;
        
        Ok(Self {
            encoder,
            nsp_head,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden_states = self.encoder.forward(input_ids, token_type_ids, attention_mask)?;
        
        // Use [CLS] token (first token) for NSP prediction
        let cls_token = hidden_states.i((.., 0, ..))?;
        self.nsp_head.forward(&cls_token)
    }
}
```

## 5. NSP Dataset

```rust
pub struct NSPDataset {
    pub tokenizer: NSPTokenizer,
    pub max_len: usize,
    pub size: usize,
}

impl NSPDataset {
    pub fn new(size: usize, max_len: usize) -> Self {
        let tokenizer = NSPTokenizer::new();
        Self {
            tokenizer,
            max_len,
            size,
        }
    }
    
    pub fn get_batch(&self, batch_size: usize, device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut input_ids = Vec::new();
        let mut token_type_ids = Vec::new();
        let mut attention_masks = Vec::new();
        let mut labels = Vec::new();
        
        for _ in 0..batch_size {
            let (ids, token_types, attention, label) = 
                self.tokenizer.build_nsp_example(self.max_len);
            
            input_ids.push(ids);
            token_type_ids.push(token_types);
            attention_masks.push(attention);
            labels.push(label);
        }
        
        let seq_len = self.max_len;
        
        // Convert to tensors
        let input_ids_flat: Vec<u32> = input_ids.into_iter().flatten().map(|x| x as u32).collect();
        let token_type_ids_flat: Vec<u32> = token_type_ids.into_iter().flatten().map(|x| x as u32).collect();
        let attention_masks_flat: Vec<u32> = attention_masks.into_iter().flatten().map(|x| x as u32).collect();
        let labels_vec: Vec<u32> = labels.into_iter().map(|x| x as u32).collect();
        
        let input_ids_tensor = Tensor::from_slice(&input_ids_flat, (batch_size, seq_len), device)?;
        let token_type_ids_tensor = Tensor::from_slice(&token_type_ids_flat, (batch_size, seq_len), device)?;
        let attention_masks_tensor = Tensor::from_slice(&attention_masks_flat, (batch_size, seq_len), device)?;
        let labels_tensor = Tensor::from_slice(&labels_vec, batch_size, device)?;
        
        Ok((input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, labels_tensor))
    }
}
```

## 6. Training utilities

```rust
fn compute_nsp_accuracy(logits: &Tensor, labels: &Tensor) -> Result<f64> {
    let predictions = logits.argmax(1)?;
    let correct = predictions.eq(labels)?;
    let accuracy = correct.to_dtype(DType::F64)?.mean_all()?;
    Ok(accuracy.to_vec0()?)
}
```

## 7. Training function

```rust
pub fn train_nsp_model() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create dataset
    let dataset = NSPDataset::new(1000, MAX_LEN);
    
    // Initialize model
    let mut cfg = BertConfig::default();
    cfg.vocab_size = dataset.tokenizer.vocab.len();
    cfg.max_len = MAX_LEN;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BertForNextSentencePrediction::new(&cfg, vb)?;
    
    // Training parameters
    let lr = 3e-4;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr, ..Default::default() })?;
    
    let epochs = 50;
    let batch_size = 4;
    let steps_per_epoch = 25;
    
    println!("BERT NSP model initialized");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Max sequence length: {}", cfg.max_len);
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut num_batches = 0;
        
        for _ in 0..steps_per_epoch {
            let (input_ids, token_type_ids, attention_mask, labels) = 
                dataset.get_batch(batch_size, &device)?;
            
            let logits = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
            
            optimizer.backward_step(&loss)?;
            
            // Compute metrics
            let nsp_accuracy = compute_nsp_accuracy(&logits, &labels)?;
            
            total_loss += loss.to_vec0::<f32>()?;
            total_acc += nsp_accuracy;
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        let avg_acc = total_acc / num_batches as f64;
        
        if epoch % 10 == 0 {
            println!("Epoch {:2} | loss: {:.4} | NSP acc: {:.1}%", 
                     epoch, avg_loss, avg_acc * 100.0);
        }
    }
    
    println!("NSP training completed!");
    Ok(())
}
```

## 8. Inference: predict if B follows A

Given two sentences, build an input and return the probability that B is the next sentence.

```rust
pub fn predict_is_next(model: &BertForNextSentencePrediction, tokenizer: &NSPTokenizer,
                      sent_a: &str, sent_b: &str, max_len: usize, device: &Device) -> Result<f64> {
    let tokens_a = tokenizer.encode(sent_a);
    let tokens_b = tokenizer.encode(sent_b);
    
    // Build input: [CLS] + A + [SEP] + B + [SEP]
    let mut input_ids = vec![tokenizer.vocab["[CLS]"]];
    
    // Truncate to fit in max_len
    let available_len = max_len - 3;
    let max_a_len = available_len / 2;
    let max_b_len = available_len - max_a_len;
    
    let a_truncated = if tokens_a.len() > max_a_len {
        &tokens_a[..max_a_len]
    } else {
        &tokens_a
    };
    
    let b_truncated = if tokens_b.len() > max_b_len {
        &tokens_b[..max_b_len]
    } else {
        &tokens_b
    };
    
    input_ids.extend(a_truncated);
    input_ids.push(tokenizer.vocab["[SEP]"]);
    let sep_idx = input_ids.len() - 1;
    input_ids.extend(b_truncated);
    input_ids.push(tokenizer.vocab["[SEP]"]);
    
    // Token type IDs and attention mask
    let mut token_type_ids = vec![0; sep_idx + 1];
    token_type_ids.extend(vec![1; input_ids.len() - sep_idx - 1]);
    let attention_mask = vec![1; input_ids.len()];
    
    // Pad to max_len
    while input_ids.len() < max_len {
        input_ids.push(tokenizer.vocab["[PAD]"]);
        token_type_ids.push(0);
        attention_mask.push(0);
    }
    
    // Convert to tensors
    let input_ids_tensor = Tensor::from_slice(
        &input_ids.into_iter().map(|x| x as u32).collect::<Vec<_>>()[..max_len],
        (1, max_len),
        device
    )?;
    let token_type_ids_tensor = Tensor::from_slice(
        &token_type_ids.into_iter().map(|x| x as u32).collect::<Vec<_>>()[..max_len],
        (1, max_len),
        device
    )?;
    let attention_mask_tensor = Tensor::from_slice(
        &attention_mask.into_iter().map(|x| x as u32).collect::<Vec<_>>()[..max_len],
        (1, max_len),
        device
    )?;
    
    let logits = model.forward(&input_ids_tensor, &token_type_ids_tensor, &attention_mask_tensor)?;
    let probs = candle_nn::ops::softmax(&logits, 1)?;
    
    // Return probability of "is_next" (class 0)
    let prob_is_next = probs.i((0, 0))?.to_vec0::<f32>()? as f64;
    Ok(prob_is_next)
}
```

## 9. Saving, loading, and checkpointing

```rust
pub fn save_nsp_model(model: &BertForNextSentencePrediction, tokenizer: &NSPTokenizer, 
                     cfg: &BertConfig, path: &str) -> Result<()> {
    // Note: In a real implementation, you'd use candle's save functionality
    // This is a placeholder showing the structure
    println!("Saving NSP model to {}", path);
    println!("Model config: {:?}", cfg);
    println!("Vocab size: {}", tokenizer.vocab.len());
    Ok(())
}

pub fn load_nsp_model(path: &str, device: &Device) -> Result<(BertForNextSentencePrediction, NSPTokenizer, BertConfig)> {
    // Note: In a real implementation, you'd load from saved weights
    // This is a placeholder showing the structure
    let tokenizer = NSPTokenizer::new();
    let mut cfg = BertConfig::default();
    cfg.vocab_size = tokenizer.vocab.len();
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = BertForNextSentencePrediction::new(&cfg, vb)?;
    
    println!("Loading NSP model from {}", path);
    Ok((model, tokenizer, cfg))
}
```

## 10. Complete example

```rust
fn main() -> Result<()> {
    println!("BERT NSP Fine-tuning (Candle)");
    
    // Train the model
    train_nsp_model()?;
    
    println!("Training completed successfully!");
    
    // Example usage
    let tokenizer = NSPTokenizer::new();
    println!("Example documents:");
    for (i, doc) in tokenizer.documents.iter().enumerate() {
        println!("Document {}: {:?}", i, doc);
    }
    
    Ok(())
}
```

## 11. Practical tips and variants

- **Segment IDs**: NSP requires token_type_ids to distinguish sentence A vs B; we set 0 for A and 1 for B
- **Data quality**: For stronger signals, build larger multi‑sentence documents and ensure negatives are not trivial duplicates
- **Mixed precision**: Consider using mixed precision training for better performance on supported hardware
- **Joint objectives**: Original BERT pretraining used MLM + NSP jointly. You could combine both objectives by training on both losses simultaneously
- **Modern alternatives**: Many recent models (RoBERTa, DeBERTa) replace NSP with other objectives like Sentence Order Prediction (SOP)

## 12. Where to go next

- Use the trained NSP model as initialization for downstream tasks that benefit from sentence relationship understanding
- Combine NSP training with MLM for full BERT-style pretraining
- Experiment with other sentence-level objectives like Sentence Order Prediction
- Scale up with larger document collections and longer sequences
- Add evaluation on standard sentence relationship benchmarks