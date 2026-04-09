# BERT: Fine-tuning for Token Classification (Candle/Rust)

This chapter shows how to fine‑tune a compact BERT‑style encoder for token classification tasks (like Named Entity Recognition) using Candle and Rust. We keep everything device‑agnostic and use pure Candle/Rust implementations, consistent with other BERT chapters in this series.

What you will build:
- A simple tokenizer and a toy NER dataset with token-level labels
- Input construction with proper label alignment and padding
- A compact BERT‑like encoder using Candle components
- A token classification head that predicts labels for each token
- A clean training/evaluation loop with token-level accuracy
- An inference function for predicting entity labels

Notes:
- For real work, use robust tokenizers (tokenizers crate), pretrained encoders, and larger datasets. This chapter focuses on model architecture, APIs, and a minimal fine‑tune recipe.

## 1. Setup and dependencies

Add the necessary dependencies to your `Cargo.toml`:

```toml
[dependencies]
candle-core = "0.9.1"
candle-nn = "0.9.1"
rand = "0.8.5"
```

```rust
use candle_core::{Device, Result, Tensor, DType, IndexOp};
use candle_nn::{Module, VarBuilder, VarMap, Linear, Dropout};
use std::collections::HashMap;
use rand::{thread_rng, seq::SliceRandom};

fn main() -> Result<()> {
    println!("BERT Token Classification Fine-tuning with Candle");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    Ok(())
}
```

## 2. Simple tokenizer and toy NER dataset

We'll use a whitespace tokenizer and define a tiny NER dataset with token-level labels using the BIO (Beginning-Inside-Outside) tagging scheme.

```rust
// Special tokens
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[MASK]", 3),
];

// Token classification item
#[derive(Debug, Clone)]
pub struct TokenClassificationItem {
    pub text: String,
    pub labels: Vec<String>, // One label per token
}

// Toy NER dataset (BIO tagging)
const TOKEN_CLASSIFICATION_ITEMS: &[TokenClassificationItem] = &[
    TokenClassificationItem {
        text: "John Smith works at Microsoft".to_string(),
        labels: vec!["B-PER".to_string(), "I-PER".to_string(), "O".to_string(), "O".to_string(), "B-ORG".to_string()],
    },
    TokenClassificationItem {
        text: "Apple Inc is in California".to_string(), 
        labels: vec!["B-ORG".to_string(), "I-ORG".to_string(), "O".to_string(), "O".to_string(), "B-LOC".to_string()],
    },
    TokenClassificationItem {
        text: "Barack Obama visited New York".to_string(),
        labels: vec!["B-PER".to_string(), "I-PER".to_string(), "O".to_string(), "B-LOC".to_string(), "I-LOC".to_string()],
    },
    TokenClassificationItem {
        text: "Google was founded in Stanford".to_string(),
        labels: vec!["B-ORG".to_string(), "O".to_string(), "O".to_string(), "O".to_string(), "B-LOC".to_string()],
    },
];

// Common NER labels
const NER_LABELS: &[&str] = &["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"];

// Token classification tokenizer
pub struct TokenClassificationTokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
    pub label_to_id: HashMap<String, usize>,
    pub id_to_label: HashMap<usize, String>,
}

impl TokenClassificationTokenizer {
    pub fn new(items: &[TokenClassificationItem]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Count words in all texts
        for item in items {
            for word in item.text.split_whitespace() {
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
        
        // Create inverse mapping for vocabulary
        let itos: HashMap<usize, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        // Create label mappings
        let mut label_to_id = HashMap::new();
        let mut id_to_label = HashMap::new();
        
        for (i, label) in NER_LABELS.iter().enumerate() {
            label_to_id.insert(label.to_string(), i);
            id_to_label.insert(i, label.to_string());
        }
        
        Self { 
            vocab, 
            itos, 
            label_to_id, 
            id_to_label 
        }
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
    
    pub fn encode_labels(&self, labels: &[String]) -> Vec<usize> {
        labels.iter()
            .map(|label| {
                self.label_to_id.get(label)
                    .copied()
                    .unwrap_or(0) // Default to "O" label
            })
            .collect()
    }
    
    pub fn decode_labels(&self, label_ids: &[usize]) -> Vec<String> {
        label_ids.iter()
            .map(|&id| {
                self.id_to_label.get(&id)
                    .cloned()
                    .unwrap_or_else(|| "O".to_string())
            })
            .collect()
    }
    
    pub fn build_token_classification_input(&self, text: &str, labels: &[String], max_len: usize) 
        -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<i64>) {
        
        let tokens = self.encode(text);
        let label_ids = self.encode_labels(labels);
        
        // Ensure tokens and labels have same length
        let min_len = tokens.len().min(label_ids.len());
        let tokens = &tokens[..min_len];
        let label_ids = &label_ids[..min_len];
        
        // Build input: [CLS] + tokens + [SEP]
        let mut input_ids = vec![self.vocab["[CLS]"]];
        input_ids.extend(tokens);
        input_ids.push(self.vocab["[SEP]"]);
        
        // Build labels: ignore_index + labels + ignore_index
        let mut token_labels = vec![-100i64]; // Ignore [CLS]
        token_labels.extend(label_ids.iter().map(|&id| id as i64));
        token_labels.push(-100i64); // Ignore [SEP]
        
        // Create token type ids (all zeros for single sentence)
        let token_type_ids = vec![0; input_ids.len()];
        
        // Create attention mask
        let attention_mask = vec![1; input_ids.len()];
        
        // Pad sequences
        let mut padded_ids = input_ids;
        let mut padded_labels = token_labels;
        let mut padded_token_types = token_type_ids;
        let mut padded_attention = attention_mask;
        
        while padded_ids.len() < max_len {
            padded_ids.push(self.vocab["[PAD]"]);
            padded_labels.push(-100i64); // Ignore padding
            padded_token_types.push(0);
            padded_attention.push(0);
        }
        
        padded_ids.truncate(max_len);
        padded_labels.truncate(max_len);
        padded_token_types.truncate(max_len);
        padded_attention.truncate(max_len);
        
        (padded_ids, padded_token_types, padded_attention, padded_labels)
    }
}
```

## 3. BERT Token Classification Head

```rust
// BERT Token Classification Head
pub struct BertTokenClassificationHead {
    dropout: Dropout,
    classifier: Linear,
}

impl BertTokenClassificationHead {
    pub fn new(hidden_size: usize, num_labels: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let dropout = candle_nn::dropout(dropout)?;
        let classifier = candle_nn::linear(hidden_size, num_labels, vb.pp("classifier"))?;
        
        Ok(Self {
            dropout,
            classifier,
        })
    }
}

impl Module for BertTokenClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dropout.forward(hidden_states, true)?;
        self.classifier.forward(&hidden_states)
    }
}
```

## 4. BERT for Token Classification

```rust
// Reuse BertConfig and BertEncoder from fine-tuning example
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub max_len: usize,
    pub dropout: f64,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 4,
            mlp_ratio: 4.0,
            max_len: 32,
            dropout: 0.1,
        }
    }
}

pub struct BertForTokenClassification {
    encoder: BertEncoder, // Reuse from fine-tuning
    classifier: BertTokenClassificationHead,
}

impl BertForTokenClassification {
    pub fn new(cfg: &BertConfig, num_labels: usize, vb: VarBuilder) -> Result<Self> {
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;
        let classifier = BertTokenClassificationHead::new(cfg.hidden_size, num_labels, cfg.dropout, vb.pp("classifier"))?;
        
        Ok(Self {
            encoder,
            classifier,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor) 
        -> Result<Tensor> {
        let hidden_states = self.encoder.forward(input_ids, token_type_ids, attention_mask)?;
        self.classifier.forward(&hidden_states)
    }
}
```

## 5. Token Classification Dataset

```rust
pub struct TokenClassificationDataset {
    pub items: Vec<TokenClassificationItem>,
    pub tokenizer: TokenClassificationTokenizer,
    pub max_len: usize,
}

impl TokenClassificationDataset {
    pub fn new(items: Vec<TokenClassificationItem>, max_len: usize) -> Self {
        let tokenizer = TokenClassificationTokenizer::new(&items);
        Self {
            items,
            tokenizer,
            max_len,
        }
    }
    
    pub fn get_item(&self, idx: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<i64>) {
        let item = &self.items[idx];
        self.tokenizer.build_token_classification_input(&item.text, &item.labels, self.max_len)
    }
    
    pub fn get_batch(&self, indices: &[usize], device: &Device) 
        -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut input_ids = Vec::new();
        let mut token_type_ids = Vec::new();
        let mut attention_masks = Vec::new();
        let mut labels = Vec::new();
        
        for &idx in indices {
            let (ids, token_types, attention, labs) = self.get_item(idx);
            input_ids.push(ids);
            token_type_ids.push(token_types);
            attention_masks.push(attention);
            labels.push(labs);
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
fn compute_token_accuracy(logits: &Tensor, labels: &Tensor) -> Result<f64> {
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

fn predict_tokens(model: &BertForTokenClassification, tokenizer: &TokenClassificationTokenizer,
                 text: &str, max_len: usize, device: &Device) -> Result<Vec<String>> {
    // Create dummy labels (will be ignored)
    let words: Vec<&str> = text.split_whitespace().collect();
    let dummy_labels: Vec<String> = vec!["O".to_string(); words.len()];
    
    let (input_ids, token_type_ids, attention_mask, _) = 
        tokenizer.build_token_classification_input(text, &dummy_labels, max_len);
    
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
    
    // Extract predictions for actual tokens (skip [CLS] and [SEP])
    let predictions = predictions.i((0, ..))?;
    let pred_vec: Vec<u32> = predictions.to_vec1()?;
    
    // Skip [CLS] token and only take predictions for actual words
    let start_idx = 1; // Skip [CLS]
    let end_idx = (start_idx + words.len()).min(pred_vec.len().saturating_sub(1)); // Before [SEP]
    
    let label_ids: Vec<usize> = pred_vec[start_idx..end_idx].iter().map(|&x| x as usize).collect();
    Ok(tokenizer.decode_labels(&label_ids))
}
```

## 7. Training function

```rust
pub fn train_token_classification_model() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create dataset
    let items = TOKEN_CLASSIFICATION_ITEMS.to_vec();
    let dataset = TokenClassificationDataset::new(items, 32);
    
    // Initialize model
    let mut cfg = BertConfig::default();
    cfg.vocab_size = dataset.tokenizer.vocab.len();
    cfg.max_len = 32;
    
    let num_labels = NER_LABELS.len();
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BertForTokenClassification::new(&cfg, num_labels, vb)?;
    
    // Training parameters
    let lr = 3e-4;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr, ..Default::default() })?;
    
    let epochs = 30;
    let batch_size = 2;
    
    println!("BERT Token Classification model initialized");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Number of labels: {}", num_labels);
    println!("Dataset size: {}", dataset.items.len());
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut num_batches = 0;
        
        // Shuffle training indices
        let mut train_indices: Vec<usize> = (0..dataset.items.len()).collect();
        train_indices.shuffle(&mut thread_rng());
        
        // Simple iteration over all examples
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
            let token_accuracy = compute_token_accuracy(&logits, &labels)?;
            
            total_loss += loss.to_vec0::<f32>()?;
            total_acc += token_accuracy;
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        let avg_acc = total_acc / num_batches as f64;
        
        if epoch % 10 == 0 {
            println!("Epoch {:2} | loss: {:.4} | token acc: {:.1}%", 
                     epoch, avg_loss, avg_acc * 100.0);
        }
    }
    
    // Example prediction
    let test_text = "Apple Inc is located in California";
    let predictions = predict_tokens(&model, &dataset.tokenizer, test_text, 32, &device)?;
    
    println!("\nExample prediction:");
    println!("Text: {}", test_text);
    println!("Predicted labels: {:?}", predictions);
    
    println!("Token classification training completed!");
    Ok(())
}
```

## 8. Evaluation metrics

```rust
fn compute_entity_level_metrics(predicted_labels: &[String], true_labels: &[String]) -> (f64, f64, f64) {
    // Extract entities from BIO labels
    fn extract_entities(labels: &[String]) -> Vec<(String, usize, usize)> {
        let mut entities = Vec::new();
        let mut current_entity: Option<(String, usize)> = None;
        
        for (i, label) in labels.iter().enumerate() {
            if label.starts_with("B-") {
                // Close previous entity if exists
                if let Some((entity_type, start)) = current_entity {
                    entities.push((entity_type, start, i - 1));
                }
                // Start new entity
                current_entity = Some((label[2..].to_string(), i));
            } else if label.starts_with("I-") {
                // Continue current entity (if types match)
                if let Some((ref entity_type, start)) = current_entity {
                    if label[2..] != *entity_type {
                        // Type mismatch, close previous and start new
                        entities.push((entity_type.clone(), start, i - 1));
                        current_entity = Some((label[2..].to_string(), i));
                    }
                    // Otherwise continue the current entity
                } else {
                    // I- without B-, treat as B-
                    current_entity = Some((label[2..].to_string(), i));
                }
            } else {
                // O label, close current entity if exists
                if let Some((entity_type, start)) = current_entity {
                    entities.push((entity_type, start, i - 1));
                    current_entity = None;
                }
            }
        }
        
        // Close final entity if exists
        if let Some((entity_type, start)) = current_entity {
            entities.push((entity_type, start, labels.len() - 1));
        }
        
        entities
    }
    
    let predicted_entities = extract_entities(predicted_labels);
    let true_entities = extract_entities(true_labels);
    
    let predicted_set: std::collections::HashSet<_> = predicted_entities.into_iter().collect();
    let true_set: std::collections::HashSet<_> = true_entities.into_iter().collect();
    
    let intersection = predicted_set.intersection(&true_set).count() as f64;
    let predicted_count = predicted_set.len() as f64;
    let true_count = true_set.len() as f64;
    
    let precision = if predicted_count > 0.0 { intersection / predicted_count } else { 0.0 };
    let recall = if true_count > 0.0 { intersection / true_count } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    
    (precision, recall, f1)
}
```

## 9. Complete example

```rust
fn main() -> Result<()> {
    println!("BERT Token Classification Fine-tuning (Candle)");
    
    // Train the model
    train_token_classification_model()?;
    
    println!("Training completed successfully!");
    Ok(())
}
```

## 10. Practical tips

- **Tokenization**: This whitespace tokenizer is only for demonstration. Use the `tokenizers` crate with WordPiece/BPE for real applications.
- **Label alignment**: In practice, subword tokenization requires careful label alignment (e.g., only predict on first subword of each original token).
- **Class imbalance**: NER datasets often have class imbalance. Consider weighted loss functions or focal loss.
- **Evaluation metrics**: Entity-level F1 score is more informative than token-level accuracy for NER tasks.
- **BIO consistency**: Add constraints to ensure valid BIO tag sequences (e.g., I-PER cannot follow B-ORG).
- **Data augmentation**: Consider techniques like synonym replacement and label-preserving transformations.

## 11. Where to go next

- Explore other BERT fine-tuning tasks in this repository (classification, QA, etc.)
- Replace the simple tokenizer with a learned tokenizer from the tokenizers crate
- Implement Conditional Random Field (CRF) layer on top for better tag sequence modeling
- Experiment with different NER datasets and entity types
- Scale up with pretrained weights and larger datasets for better performance
- Implement advanced evaluation metrics like partial matching and strict/relaxed entity matching