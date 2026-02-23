# BERT: Fine-tuning for Sequence Classification (Candle/Rust)

This chapter shows how to fine‑tune a BERT‑style encoder for sequence classification using Candle and Rust. It pairs naturally with the "BERT: Pre‑training" chapter but can also use randomly initialized weights for learning purposes. We keep everything device‑agnostic and use pure Candle/Rust implementations.

What you will build:
- A simple tokenizer and a toy sentiment‑like dataset
- A compact BERT‑style encoder implemented in Candle
- A classification head on top of the [CLS] representation
- A clean training/evaluation loop with accuracy metrics
- Save/load utilities and a minimal inference function

Notes:
- For real tasks, prefer robust tokenizers (tokenizers crate) and a pretrained encoder. Here we focus on model architecture, APIs, and a clear finetune recipe.

## 1. Setup and dependencies

First, add the necessary dependencies to your `Cargo.toml`:

```toml
[dependencies]
candle-core = "0.3"
candle-nn = "0.3"
rand = "0.8"
```

```rust
use candle_core::{Device, Result, Tensor, DType, IndexOp};
use candle_nn::{Module, VarBuilder, VarMap, Optimizer, AdamW, Linear, LayerNorm, Embedding, Dropout};
use std::collections::HashMap;
use rand::{thread_rng, seq::SliceRandom};

fn main() -> Result<()> {
    println!("Candle BERT Fine-tuning Example");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    Ok(())
}
```

## 2. Simple tokenizer and toy dataset

We'll make a small binary classification dataset with short sentences labeled 0/1. For the tokenizer we use a simple whitespace approach.

```rust
// Special tokens mapping
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[MASK]", 3),
];

// Training data
const TRAIN_TEXTS: &[(&str, usize)] = &[
    ("i love this movie", 1),
    ("this film was great", 1),
    ("what a fantastic experience", 1),
    ("absolutely wonderful acting", 1),
    ("i dislike the pacing", 0),
    ("this movie was boring", 0),
    ("the plot did not work", 0),
    ("terrible sound and weak script", 0),
];

const VAL_TEXTS: &[(&str, usize)] = &[
    ("i loved the film", 1),
    ("boring and long", 0),
    ("wonderful story", 1),
    ("not my taste", 0),
];

// Simple whitespace tokenizer
pub struct SimpleTokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
}

impl SimpleTokenizer {
    pub fn new(texts: &[(&str, usize)]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Count words in all texts
        for (text, _) in texts {
            for word in text.split_whitespace() {
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
    
    pub fn build_input(&self, text: &str, max_len: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut ids = vec![self.vocab["[CLS]"]];
        ids.extend(self.encode(text));
        ids.push(self.vocab["[SEP]"]);
        
        let mut token_type = vec![0; ids.len()];
        let mut attention = vec![1; ids.len()];
        
        // Pad or truncate
        while ids.len() < max_len {
            ids.push(self.vocab["[PAD]"]);
            token_type.push(0);
            attention.push(0);
        }
        
        ids.truncate(max_len);
        token_type.truncate(max_len);
        attention.truncate(max_len);
        
        (ids, token_type, attention)
    }
}

// Create tokenizer from combined train and validation data
// let all_texts: Vec<(&str, usize)> = TRAIN_TEXTS.iter().chain(VAL_TEXTS.iter()).copied().collect();
// let tokenizer = SimpleTokenizer::new(&all_texts);
// let vocab_size = tokenizer.vocab.len();
// println!("Vocab size: {}", vocab_size);
```

## 3. BERT‑style encoder implementation

We implement a compact encoder structure: embeddings + L Transformer blocks using Candle components.

```rust
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
            vocab_size: 100, // Will be set based on actual vocab
            hidden_size: 128,
            num_layers: 2,
            num_heads: 4,
            mlp_ratio: 4.0,
            max_len: 32,
            dropout: 0.1,
        }
    }
}

// BERT Embeddings
pub struct BertEmbeddings {
    token_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let token_embeddings = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embeddings"))?;
        let position_embeddings = candle_nn::embedding(cfg.max_len, cfg.hidden_size, vb.pp("position_embeddings"))?;
        let token_type_embeddings = candle_nn::embedding(2, cfg.hidden_size, vb.pp("token_type_embeddings"))?;
        let layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("layer_norm"))?;
        let dropout = candle_nn::dropout(cfg.dropout)?;
        
        Ok(Self {
            token_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        })
    }
}

impl Module for BertEmbeddings {
    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;
        
        // Create position ids
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
            .unsqueeze(0)?
            .expand(input_ids.dims())?;
        
        // Get embeddings
        let token_embeds = self.token_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;
        let token_type_embeds = self.token_type_embeddings.forward(token_type_ids)?;
        
        // Sum embeddings
        let embeddings = (&token_embeds + &position_embeds)? + &token_type_embeds?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings, false)
    }
}

// Multi-Head Attention
pub struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_heads;
        assert_eq!(cfg.hidden_size % cfg.num_heads, 0);
        
        let query = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("query"))?;
        let key = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("key"))?;
        let value = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("value"))?;
        let output = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("output"))?;
        let dropout = candle_nn::dropout(cfg.dropout)?;
        
        Ok(Self {
            query,
            key,
            value,
            output,
            num_heads: cfg.num_heads,
            head_dim,
            dropout,
        })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        
        let q = self.query.forward(hidden_states)?;
        let k = self.key.forward(hidden_states)?;
        let v = self.value.forward(hidden_states)?;
        
        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        
        // Attention scores
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores / (self.head_dim as f64).sqrt())?;
        
        // Apply attention mask
        let mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
        let mask = (mask - 1.0)? * 10000.0?;
        let scores = (scores + mask)?;
        
        // Softmax and apply to values
        let attention_probs = candle_nn::ops::softmax(&scores, 3)?;
        let attention_probs = self.dropout.forward(&attention_probs, false)?;
        
        let context = attention_probs.matmul(&v)?;
        let context = context.transpose(1, 2)?.reshape((batch_size, seq_len, hidden_size))?;
        
        self.output.forward(&context)
    }
}

// Feed Forward Network
pub struct FeedForward {
    dense: Linear,
    intermediate: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let intermediate_size = (cfg.hidden_size as f64 * cfg.mlp_ratio) as usize;
        let intermediate = candle_nn::linear(cfg.hidden_size, intermediate_size, vb.pp("intermediate"))?;
        let dense = candle_nn::linear(intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        let dropout = candle_nn::dropout(cfg.dropout)?;
        
        Ok(Self {
            dense,
            intermediate,
            dropout,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.intermediate.forward(hidden_states)?;
        let hidden_states = hidden_states.gelu()?;
        let hidden_states = self.dense.forward(&hidden_states)?;
        self.dropout.forward(&hidden_states, false)
    }
}

// Transformer Block
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attention_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,
}

impl TransformerBlock {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let attention = MultiHeadAttention::new(cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(cfg, vb.pp("feed_forward"))?;
        let attention_layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("attention_layer_norm"))?;
        let output_layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("output_layer_norm"))?;
        
        Ok(Self {
            attention,
            feed_forward,
            attention_layer_norm,
            output_layer_norm,
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection and layer norm
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let hidden_states = (hidden_states + &attention_output)?;
        let hidden_states = self.attention_layer_norm.forward(&hidden_states)?;
        
        // Feed forward with residual connection and layer norm
        let feed_forward_output = self.feed_forward.forward(&hidden_states)?;
        let hidden_states = (&hidden_states + &feed_forward_output)?;
        self.output_layer_norm.forward(&hidden_states)
    }
}

// BERT Encoder
pub struct BertEncoder {
    embeddings: BertEmbeddings,
    layers: Vec<TransformerBlock>,
    final_layer_norm: LayerNorm,
}

impl BertEncoder {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = BertEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            let layer = TransformerBlock::new(cfg, vb.pp(&format!("layer_{}", i)))?;
            layers.push(layer);
        }
        let final_layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("final_layer_norm"))?;
        
        Ok(Self {
            embeddings,
            layers,
            final_layer_norm,
        })
    }
}

impl Module for BertEncoder {
    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;
        
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        
        self.final_layer_norm.forward(&hidden_states)
    }
}
```

## 4. Classification head and model wrapper

We take the [CLS] token's final hidden state (position 0) and predict class logits.

```rust
pub struct BertForSequenceClassification {
    encoder: BertEncoder,
    classifier: Linear,
}

impl BertForSequenceClassification {
    pub fn new(cfg: &BertConfig, num_classes: usize, vb: VarBuilder) -> Result<Self> {
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;
        let classifier = candle_nn::linear(cfg.hidden_size, num_classes, vb.pp("classifier"))?;
        
        Ok(Self {
            encoder,
            classifier,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden_states = self.encoder.forward(input_ids, token_type_ids, attention_mask)?;
        
        // Use [CLS] token (first token) for classification
        let cls_token = hidden_states.i((.., 0, ..))?;
        self.classifier.forward(&cls_token)
    }
}
```

## 5. Dataset and training utilities

```rust
pub struct ClassificationDataset {
    pub texts: Vec<String>,
    pub labels: Vec<usize>,
    pub tokenizer: SimpleTokenizer,
    pub max_len: usize,
}

impl ClassificationDataset {
    pub fn new(data: Vec<(String, usize)>, max_len: usize) -> Self {
        let tokenizer = SimpleTokenizer::new(&data.iter().map(|(t, l)| (t.as_str(), *l)).collect::<Vec<_>>());
        let (texts, labels): (Vec<_>, Vec<_>) = data.into_iter().unzip();
        
        Self {
            texts,
            labels,
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
            let (ids, token_types, attention) = self.tokenizer.build_input(&self.texts[idx], self.max_len);
            input_ids.push(ids);
            token_type_ids.push(token_types);
            attention_masks.push(attention);
            labels.push(self.labels[idx]);
        }
        
        let batch_size = indices.len();
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

fn compute_accuracy(logits: &Tensor, targets: &Tensor) -> Result<f64> {
    let predictions = logits.argmax(1)?;
    let correct = predictions.eq(targets)?;
    let accuracy = correct.to_dtype(DType::F64)?.mean_all()?;
    Ok(accuracy.to_vec0()?)
}
```

## 6. Training loop

```rust
pub fn train_model() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create datasets
    let train_data: Vec<(String, usize)> = TRAIN_TEXTS.iter()
        .map(|(text, label)| (text.to_string(), *label))
        .collect();
    let val_data: Vec<(String, usize)> = VAL_TEXTS.iter()
        .map(|(text, label)| (text.to_string(), *label))
        .collect();
    
    let train_dataset = ClassificationDataset::new(train_data, 32);
    let val_dataset = ClassificationDataset::new(val_data, 32);
    
    // Initialize model
    let mut cfg = BertConfig::default();
    cfg.vocab_size = train_dataset.tokenizer.vocab.len();
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BertForSequenceClassification::new(&cfg, 2, vb)?;
    
    // Training parameters
    let lr = 3e-4;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr, ..Default::default() })?;
    
    let epochs = 12;
    let batch_size = 4;
    let mut best_val_acc = 0.0;
    
    for epoch in 0..epochs {
        // Training
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut num_batches = 0;
        
        let mut train_indices: Vec<usize> = (0..train_dataset.texts.len()).collect();
        train_indices.shuffle(&mut thread_rng());
        
        for batch_start in (0..train_indices.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_indices.len());
            let batch_indices = &train_indices[batch_start..batch_end];
            
            let (input_ids, token_type_ids, attention_mask, targets) = 
                train_dataset.get_batch(batch_indices, &device)?;
            
            let logits = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
            
            optimizer.backward_step(&loss)?;
            
            total_loss += loss.to_vec0::<f32>()?;
            total_acc += compute_accuracy(&logits, &targets)?;
            num_batches += 1;
        }
        
        let train_loss = total_loss / num_batches as f32;
        let train_acc = total_acc / num_batches as f64;
        
        // Validation
        let val_indices: Vec<usize> = (0..val_dataset.texts.len()).collect();
        let (input_ids, token_type_ids, attention_mask, targets) = 
            val_dataset.get_batch(&val_indices, &device)?;
        
        let logits = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
        let val_loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
        let val_acc = compute_accuracy(&logits, &targets)?;
        
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            // Save model checkpoint here if needed
        }
        
        println!("Epoch {:2} | train {:.4}/{:.1}% | val {:.4}/{:.1}%", 
                 epoch, train_loss, train_acc * 100.0, 
                 val_loss.to_vec0::<f32>()?, val_acc * 100.0);
    }
    
    println!("Training completed. Best validation accuracy: {:.1}%", best_val_acc * 100.0);
    Ok(())
}
```

## 7. Inference: classify a new sentence

```rust
pub fn predict(model: &BertForSequenceClassification, tokenizer: &SimpleTokenizer, 
              text: &str, max_len: usize, device: &Device) -> Result<(usize, f64)> {
    let (input_ids, token_type_ids, attention_mask) = tokenizer.build_input(text, max_len);
    
    // Convert to tensors
    let input_ids = Tensor::from_slice(&input_ids.into_iter().map(|x| x as u32).collect::<Vec<_>>(), (1, max_len), device)?;
    let token_type_ids = Tensor::from_slice(&token_type_ids.into_iter().map(|x| x as u32).collect::<Vec<_>>(), (1, max_len), device)?;
    let attention_mask = Tensor::from_slice(&attention_mask.into_iter().map(|x| x as u32).collect::<Vec<_>>(), (1, max_len), device)?;
    
    let logits = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
    let probs = candle_nn::ops::softmax(&logits, 1)?;
    
    let probs_vec: Vec<f32> = probs.to_vec2()?[0].clone();
    let prediction = probs_vec.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    
    let confidence = probs_vec[prediction] as f64;
    
    Ok((prediction, confidence))
}

// Example usage:
// let (pred, conf) = predict(&model, &tokenizer, "i really love the story", 32, &device)?;
// println!("Prediction: {}, Confidence: {:.4}", pred, conf);
```

## 8. Complete example

```rust
fn main() -> Result<()> {
    println!("BERT Fine-tuning for Sequence Classification (Candle)");
    
    // Train the model
    train_model()?;
    
    println!("Training completed successfully!");
    Ok(())
}
```

## 9. Practical tips

- **Tokenization**: This whitespace tokenizer is only for demonstration. Use the `tokenizers` crate with WordPiece/BPE for real applications.
- **Sequence handling**: For paired inputs (sentence pairs), set token_type=0 for the first segment and 1 for the second.
- **Regularization**: Adjust dropout, use weight decay in the optimizer, and consider gradient clipping for stability.
- **Device management**: Candle automatically handles CPU/CUDA. Use `Device::cuda_if_available()` for best performance.
- **Checkpointing**: Save model weights using `varmap.save()` for later inference.
- **Batch processing**: Implement proper batching for larger datasets to improve training efficiency.

## 10. Where to go next

- Explore other BERT fine-tuning tasks in this repository (QA, token classification, etc.)
- Replace the simple tokenizer with a learned tokenizer from the tokenizers crate
- Experiment with different model sizes and hyperparameters for better performance
- Load pretrained BERT weights for transfer learning instead of training from scratch