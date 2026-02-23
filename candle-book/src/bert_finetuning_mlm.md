# BERT: Fine-tuning for Masked Language Modeling (Candle/Rust)

This chapter shows how to fine‑tune a compact BERT‑style encoder for the Masked Language Modeling (MLM) objective using Candle and Rust. It mirrors the style of other BERT chapters in this series: simple tokenizer, compact BERT encoder, an MLM head, and a training/evaluation/inference loop that runs on various devices.

What you will build:
- A simple whitespace tokenizer and a toy corpus
- Example construction with dynamic masking (15% of tokens; 80% [MASK], 10% random, 10% original)
- A compact BERT‑like encoder using Candle components
- An MLM head with cross-entropy training and ignore_index for non‑masked positions
- Checkpointing and a predict_masked helper for inference

Notes:
- This is an educational mini setup. For real work, use robust tokenizers (tokenizers crate), large corpora, and pretrained checkpoints.

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
use candle_nn::{Module, VarBuilder, VarMap, Linear, LayerNorm};
use std::collections::HashMap;
use rand::{thread_rng, Rng, seq::SliceRandom};

fn main() -> Result<()> {
    println!("BERT MLM Fine-tuning with Candle");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    Ok(())
}
```

## 2. Simple tokenizer and toy corpus

We'll reuse the simple approach from other chapters: a lowercase whitespace tokenizer and a tiny list of sentences.

```rust
// Special tokens
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[MASK]", 3),
];

// Tiny toy corpus (single sentences; could be grouped into docs too)
const CORPUS: &[&str] = &[
    "the cat sat on the mat",
    "the mat was warm",
    "dogs love to play",
    "they run in the park",
    "birds can fly",
    "some birds migrate",
    "they travel long distances",
];

// MLM Tokenizer
pub struct MLMTokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
}

impl MLMTokenizer {
    pub fn new(corpus: &[&str]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Count words in corpus
        for sentence in corpus {
            for word in sentence.split_whitespace() {
                let word = word.to_lowercase();
                *word_counts.entry(word).or_insert(0) += 1;
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
        
        Self { vocab, itos }
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

## 3. Dynamic masking and example builder (MLM only)

We mask 15% of tokens with the BERT rule: 80% replace with [MASK], 10% random token, 10% keep as is. We also add [CLS] and [SEP] around each sequence and pad/truncate to a fixed max length.

```rust
// MLM constants
const MASK_PROB: f64 = 0.15;
const MASK_TOKEN_PROB: f64 = 0.8;  // 80% -> [MASK]
const RANDOM_TOKEN_PROB: f64 = 0.1; // 10% -> random token
const KEEP_TOKEN_PROB: f64 = 0.1;   // 10% -> keep original
const MAX_LEN: usize = 32;

impl MLMTokenizer {
    pub fn mask_tokens(&self, input_ids: &[usize]) -> (Vec<usize>, Vec<i64>) {
        let mut rng = thread_rng();
        let mut labels = vec![-100i64; input_ids.len()]; // Use -100 as ignore index
        let mut masked = input_ids.to_vec();
        
        for (i, &token_id) in input_ids.iter().enumerate() {
            // Skip special tokens
            if token_id < SPECIALS.len() {
                continue;
            }
            
            if rng.gen_bool(MASK_PROB) {
                labels[i] = token_id as i64; // Store original token for loss
                
                let rand_val = rng.gen::<f64>();
                if rand_val < MASK_TOKEN_PROB {
                    masked[i] = self.vocab["[MASK]"];
                } else if rand_val < MASK_TOKEN_PROB + RANDOM_TOKEN_PROB {
                    // Replace with random token (excluding specials)
                    let random_id = rng.gen_range(SPECIALS.len()..self.vocab.len());
                    masked[i] = random_id;
                }
                // else: keep original token (KEEP_TOKEN_PROB)
            }
        }
        
        (masked, labels)
    }
    
    pub fn prepare_mlm_example(&self, sentence: &str, max_len: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<i64>) {
        let tokens = self.encode(sentence);
        
        // Build sequence: [CLS] + tokens + [SEP]
        let mut input_ids = vec![self.vocab["[CLS]"]];
        input_ids.extend(&tokens);
        input_ids.push(self.vocab["[SEP]"]);
        
        // Apply MLM masking
        let (masked_ids, mut mlm_labels) = self.mask_tokens(&input_ids);
        
        // Create token type ids (all zeros for single sentence)
        let token_type_ids = vec![0; masked_ids.len()];
        
        // Create attention mask
        let attention_mask = vec![1; masked_ids.len()];
        
        // Pad sequences to max_len
        let mut padded_ids = masked_ids;
        let mut padded_token_types = token_type_ids;
        let mut padded_attention = attention_mask;
        
        while padded_ids.len() < max_len {
            padded_ids.push(self.vocab["[PAD]"]);
            padded_token_types.push(0);
            padded_attention.push(0);
            mlm_labels.push(-100); // Ignore padding tokens
        }
        
        padded_ids.truncate(max_len);
        padded_token_types.truncate(max_len);
        padded_attention.truncate(max_len);
        mlm_labels.truncate(max_len);
        
        (padded_ids, padded_token_types, padded_attention, mlm_labels)
    }
}
```

## 4. BERT encoder and MLM head

We'll reuse the BERT encoder from previous chapters and add an MLM head for token-level predictions.

```rust
// Reuse BertConfig and BertEncoder from previous chapters
use super::bert_finetuning::{BertConfig, BertEncoder};

// MLM Head for predicting masked tokens
pub struct BertMLMHead {
    transform: Linear,
    layer_norm: LayerNorm,
    decoder: Linear,
}

impl BertMLMHead {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let transform = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("transform"))?;
        let layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("layer_norm"))?;
        let decoder = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("decoder"))?;
        
        Ok(Self {
            transform,
            layer_norm,
            decoder,
        })
    }
}

impl Module for BertMLMHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.transform.forward(hidden_states)?;
        let hidden_states = hidden_states.gelu()?;
        let hidden_states = self.layer_norm.forward(&hidden_states)?;
        self.decoder.forward(&hidden_states)
    }
}

// BERT for Masked Language Modeling
pub struct BertForMaskedLM {
    encoder: BertEncoder,
    mlm_head: BertMLMHead,
}

impl BertForMaskedLM {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;
        let mlm_head = BertMLMHead::new(cfg, vb.pp("mlm_head"))?;
        
        Ok(Self {
            encoder,
            mlm_head,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden_states = self.encoder.forward(input_ids, token_type_ids, attention_mask)?;
        self.mlm_head.forward(&hidden_states)
    }
}
```

## 5. MLM Dataset

```rust
pub struct MLMDataset {
    pub sentences: Vec<String>,
    pub tokenizer: MLMTokenizer,
    pub max_len: usize,
}

impl MLMDataset {
    pub fn new(sentences: Vec<String>, max_len: usize) -> Self {
        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        let tokenizer = MLMTokenizer::new(&sentence_refs);
        
        Self {
            sentences,
            tokenizer,
            max_len,
        }
    }
    
    pub fn get_batch(&self, indices: &[usize], device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut input_ids = Vec::new();
        let mut token_type_ids = Vec::new();
        let mut attention_masks = Vec::new();
        let mut labels = Vec::new();
        
        for &idx in indices {
            let (ids, token_types, attention, mlm_labels) = 
                self.tokenizer.prepare_mlm_example(&self.sentences[idx], self.max_len);
            
            input_ids.push(ids);
            token_type_ids.push(token_types);
            attention_masks.push(attention);
            labels.push(mlm_labels);
        }
        
        let batch_size = indices.len();
        let seq_len = self.max_len;
        
        // Convert to tensors
        let input_ids_flat: Vec<u32> = input_ids.into_iter().flatten().map(|x| x as u32).collect();
        let token_type_ids_flat: Vec<u32> = token_type_ids.into_iter().flatten().map(|x| x as u32).collect();
        let attention_masks_flat: Vec<u32> = attention_masks.into_iter().flatten().map(|x| x as u32).collect();
        let labels_flat: Vec<i64> = labels.into_iter().flatten().collect();
        
        let input_ids_tensor = Tensor::from_slice(&input_ids_flat, (batch_size, seq_len), device)?;
        let token_type_ids_tensor = Tensor::from_slice(&token_type_ids_flat, (batch_size, seq_len), device)?;
        let attention_masks_tensor = Tensor::from_slice(&attention_masks_flat, (batch_size, seq_len), device)?;
        let labels_tensor = Tensor::from_slice(&labels_flat, (batch_size, seq_len), device)?;
        
        Ok((input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, labels_tensor))
    }
}
```

## 6. Training utilities

```rust
fn compute_mlm_accuracy(logits: &Tensor, labels: &Tensor) -> Result<f64> {
    let predictions = logits.argmax(2)?;
    
    // Create mask for non-ignored labels
    let mask = labels.ne(&Tensor::new(-100i64, labels.device())?)?;
    
    let correct = predictions.eq(labels)?.mul(&mask)?;
    let total = mask.sum_all()?.to_dtype(DType::F64)?;
    let correct_sum = correct.sum_all()?.to_dtype(DType::F64)?;
    
    let total_f64: f64 = total.to_vec0()?;
    let correct_f64: f64 = correct_sum.to_vec0()?;
    
    if total_f64 > 0.0 {
        Ok(correct_f64 / total_f64)
    } else {
        Ok(0.0)
    }
}

fn compute_perplexity(loss: f64) -> f64 {
    loss.exp()
}
```

## 7. Training function

```rust
pub fn train_mlm_model() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create dataset from corpus
    let sentences: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
    let dataset = MLMDataset::new(sentences, MAX_LEN);
    
    // Initialize model
    let mut cfg = BertConfig::default();
    cfg.vocab_size = dataset.tokenizer.vocab.len();
    cfg.max_len = MAX_LEN;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BertForMaskedLM::new(&cfg, vb)?;
    
    // Training parameters
    let lr = 5e-4;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr, ..Default::default() })?;
    
    let epochs = 100;
    let batch_size = 4;
    
    println!("BERT MLM model initialized");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Dataset size: {}", dataset.sentences.len());
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut num_batches = 0;
        
        // Shuffle training indices
        let mut train_indices: Vec<usize> = (0..dataset.sentences.len()).collect();
        train_indices.shuffle(&mut thread_rng());
        
        for batch_start in (0..train_indices.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_indices.len());
            let batch_indices = &train_indices[batch_start..batch_end];
            
            let (input_ids, token_type_ids, attention_mask, labels) = 
                dataset.get_batch(batch_indices, &device)?;
            
            let logits = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
            
            // Flatten for loss computation - ignore -100 labels
            let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &labels.flatten(0, 1)?)?;
            
            optimizer.backward_step(&loss)?;
            
            // Compute metrics
            let mlm_accuracy = compute_mlm_accuracy(&logits, &labels)?;
            let loss_val = loss.to_vec0::<f32>()? as f64;
            
            total_loss += loss_val;
            total_acc += mlm_accuracy;
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f64;
        let avg_acc = total_acc / num_batches as f64;
        let perplexity = compute_perplexity(avg_loss);
        
        if epoch % 25 == 0 {
            println!("Epoch {:3} | loss: {:.4} | acc: {:.1}% | ppl: {:.2}", 
                     epoch, avg_loss, avg_acc * 100.0, perplexity);
        }
    }
    
    println!("MLM training completed!");
    Ok(())
}
```

## 8. Inference: predict masked tokens

```rust
pub fn predict_masked(model: &BertForMaskedLM, tokenizer: &MLMTokenizer, 
                     text: &str, max_len: usize, device: &Device) -> Result<String> {
    let (input_ids, token_type_ids, attention_mask, _) = 
        tokenizer.prepare_mlm_example(text, max_len);
    
    // Convert to tensors
    let input_ids_tensor = Tensor::from_slice(
        &input_ids.into_iter().map(|x| x as u32).collect::<Vec<_>>(), 
        (1, max_len), 
        device
    )?;
    let token_type_ids_tensor = Tensor::from_slice(
        &token_type_ids.into_iter().map(|x| x as u32).collect::<Vec<_>>(), 
        (1, max_len), 
        device
    )?;
    let attention_mask_tensor = Tensor::from_slice(
        &attention_mask.into_iter().map(|x| x as u32).collect::<Vec<_>>(), 
        (1, max_len), 
        device
    )?;
    
    let logits = model.forward(&input_ids_tensor, &token_type_ids_tensor, &attention_mask_tensor)?;
    let predictions = logits.argmax(2)?;
    
    // Extract predictions and rebuild text
    let pred_vec: Vec<u32> = predictions.i((0, ..))?. to_vec1()?;
    let predicted_ids: Vec<usize> = pred_vec.into_iter().map(|x| x as usize).collect();
    
    // Find [MASK] tokens and replace with predictions
    let original_ids: Vec<usize> = tokenizer.encode(text);
    let mut final_ids = vec![tokenizer.vocab["[CLS]"]];
    final_ids.extend(&original_ids);
    final_ids.push(tokenizer.vocab["[SEP]"]);
    
    for (i, &id) in final_ids.iter().enumerate() {
        if id == tokenizer.vocab["[MASK]"] && i < predicted_ids.len() {
            final_ids[i] = predicted_ids[i];
        }
    }
    
    // Skip [CLS] and [SEP] tokens for output
    let result_ids = &final_ids[1..final_ids.len()-1];
    Ok(tokenizer.decode(result_ids))
}

// Helper function to manually mask text
pub fn mask_text(text: &str, mask_positions: &[usize]) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut masked_words = words.clone();
    
    for &pos in mask_positions {
        if pos < masked_words.len() {
            masked_words[pos] = "[MASK]";
        }
    }
    
    masked_words.join(" ")
}
```

## 9. Complete example

```rust
fn main() -> Result<()> {
    println!("BERT MLM Fine-tuning (Candle)");
    
    // Train the model
    train_mlm_model()?;
    
    println!("Training completed successfully!");
    
    // Example usage of masking and prediction
    let text = "the cat sat on the mat";
    let masked = mask_text(text, &[2]); // Mask "sat"
    println!("Original: {}", text);
    println!("Masked: {}", masked);
    
    Ok(())
}
```

## 10. Practical tips

- **Dynamic masking**: Apply different masking patterns for each epoch rather than pre-computing masks
- **Vocabulary size**: Larger vocabularies require more training data to learn good representations
- **Learning rate**: MLM typically uses lower learning rates than classification tasks
- **Batch size**: Larger batches can help with MLM training stability
- **Evaluation**: Monitor both accuracy and perplexity to assess model quality
- **Tokenization**: Use proper subword tokenization (WordPiece/BPE) for real applications

## 11. Where to go next

- Use the trained MLM model as initialization for downstream tasks
- Experiment with whole word masking for better linguistic understanding
- Implement other self-supervised objectives like replaced token detection
- Scale up with larger corpora and model sizes for better representations
- Add evaluation on standard MLM benchmarks and datasets