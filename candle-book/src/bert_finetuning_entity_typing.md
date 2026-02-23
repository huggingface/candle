# BERT: Fine-tuning for Entity Typing (Mention-level, Multi-label, Candle/Rust)

This chapter fine‑tunes a compact BERT‑style encoder for entity typing at the mention level using Candle and Rust. We keep everything device‑agnostic and consistent with other BERT chapters in this repo.

What you will build:
- A simple tokenizer and a toy entity‑typing dataset with one marked mention per sentence
- Special mention markers [M_START] and [M_END]
- A compact BERT‑like encoder using Candle components
- A mention‑pooling classification head predicting multiple types per mention (multi‑label)
- Training/evaluation with sigmoid activation and binary cross entropy loss
- An inference helper for entity type prediction

Notes:
- Entity typing is often multi‑label (a mention may have several types). We'll use sigmoid outputs and binary cross entropy.
- For real use, prefer robust tokenization and pretrained encoders.

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
use candle_nn::{Module, VarBuilder, VarMap, Linear, Dropout, layer_norm, LayerNorm, Embedding, Activation};
use std::collections::HashMap;
use rand::{thread_rng, seq::SliceRandom};

fn main() -> Result<()> {
    println!("BERT Entity Typing Fine-tuning with Candle");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Run the training
    run_entity_typing_training(&device)?;
    
    Ok(())
}
```

## 2. Simple tokenizer and toy entity typing dataset

We'll add special tokens for mention boundaries: [M_START] and [M_END]. Our toy dataset contains sentences with a single mention span and entity types.

```rust
// Special tokens
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[MASK]", 3),
    ("[M_START]", 4),
    ("[M_END]", 5),
];

// Entity typing item
#[derive(Debug, Clone)]
pub struct EntityTypingItem {
    pub text: String,
    pub mention_span: (usize, usize), // Character start/end indices
    pub types: Vec<String>,           // Multi-label types
}

// Toy entity typing dataset
const ENTITY_TYPING_ITEMS: &[EntityTypingItem] = &[
    EntityTypingItem {
        text: "john smith works at acme corp in paris".to_string(),
        mention_span: (0, 10), // "john smith"
        types: vec!["PERSON".to_string()],
    },
    EntityTypingItem {
        text: "acme corp hired mary".to_string(),
        mention_span: (0, 9), // "acme corp"
        types: vec!["ORG".to_string()],
    },
    EntityTypingItem {
        text: "mary visited paris".to_string(),
        mention_span: (13, 18), // "paris"
        types: vec!["LOC".to_string()],
    },
    EntityTypingItem {
        text: "acme corp in berlin".to_string(),
        mention_span: (10, 16), // "berlin"
        types: vec!["LOC".to_string()],
    },
    EntityTypingItem {
        text: "john joined acme corp in 2024".to_string(),
        mention_span: (12, 21), // "acme corp"
        types: vec!["ORG".to_string(), "COMPANY".to_string()], // multi-label
    },
];

// Entity types
const ENTITY_TYPES: &[&str] = &["PERSON", "ORG", "LOC", "COMPANY"];

// Entity typing tokenizer
pub struct EntityTypingTokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
    pub type_to_id: HashMap<String, usize>,
    pub id_to_type: HashMap<usize, String>,
}

impl EntityTypingTokenizer {
    pub fn new(items: &[EntityTypingItem]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Count words in all texts with mention markers
        for item in items {
            let marked_text = inject_mention_markers(&item.text, item.mention_span);
            for word in marked_text.split_whitespace() {
                let word = word.to_lowercase();
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }
        
        // Add words to vocab in frequency order
        let mut sorted_words: Vec<_> = word_counts.iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(a.1));
        
        let mut next_id = SPECIALS.len();
        for (word, _count) in sorted_words {
            if !vocab.contains_key(word) {
                vocab.insert(word.clone(), next_id);
                next_id += 1;
            }
        }
        
        // Create reverse mapping
        let itos: HashMap<usize, String> = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        
        // Create type mappings
        let type_to_id: HashMap<String, usize> = ENTITY_TYPES.iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i))
            .collect();
        let id_to_type: HashMap<usize, String> = ENTITY_TYPES.iter()
            .enumerate()
            .map(|(i, t)| (i, t.to_string()))
            .collect();
        
        Self {
            vocab,
            itos,
            type_to_id,
            id_to_type,
        }
    }
    
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let word = word.to_lowercase();
                *self.vocab.get(&word).unwrap_or(&self.vocab["[MASK]"])
            })
            .collect()
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    pub fn num_types(&self) -> usize {
        ENTITY_TYPES.len()
    }
}

// Inject mention markers into text
fn inject_mention_markers(text: &str, span: (usize, usize)) -> String {
    let (start, end) = span;
    format!(
        "{} [M_START] {} [M_END] {}",
        &text[..start].trim(),
        &text[start..end],
        &text[end..].trim()
    ).trim().to_string()
}

// Process entity typing item for model input
pub fn process_entity_typing_item(
    item: &EntityTypingItem,
    tokenizer: &EntityTypingTokenizer,
    max_len: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<f32>) {
    // Inject mention markers and tokenize
    let marked_text = inject_mention_markers(&item.text, item.mention_span);
    let tokens = tokenizer.encode(&marked_text);
    
    // Build [CLS] + tokens + [SEP]
    let mut input_ids = vec![SPECIALS[1].1]; // [CLS]
    input_ids.extend(tokens);
    input_ids.push(SPECIALS[2].1); // [SEP]
    
    let token_type_ids = vec![0; input_ids.len()];
    let attention_mask = vec![1; input_ids.len()];
    
    // Multi-label target vector
    let mut labels = vec![0.0; tokenizer.num_types()];
    for type_name in &item.types {
        if let Some(&type_id) = tokenizer.type_to_id.get(type_name) {
            labels[type_id] = 1.0;
        }
    }
    
    // Pad or truncate
    let (input_ids, token_type_ids, attention_mask) = if input_ids.len() < max_len {
        let pad_len = max_len - input_ids.len();
        let mut padded_input = input_ids;
        let mut padded_token_type = token_type_ids;
        let mut padded_attention = attention_mask;
        
        padded_input.extend(vec![SPECIALS[0].1; pad_len]); // [PAD]
        padded_token_type.extend(vec![0; pad_len]);
        padded_attention.extend(vec![0; pad_len]);
        
        (padded_input, padded_token_type, padded_attention)
    } else {
        (
            input_ids[..max_len].to_vec(),
            token_type_ids[..max_len].to_vec(),
            attention_mask[..max_len].to_vec(),
        )
    };
    
    (input_ids, token_type_ids, attention_mask, labels)
}
```

## 3. BERT model architecture

We'll implement a compact BERT-like encoder with embeddings, transformer blocks, and a mention pooling head for entity typing.

```rust
// BERT configuration
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub dropout_prob: f64,
}

impl BertConfig {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 512,
            max_position_embeddings: 64,
            dropout_prob: 0.1,
        }
    }
}

// BERT embeddings (token + position + segment)
pub struct BertEmbeddings {
    token_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let token_embeddings = Embedding::new(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?;
        let position_embeddings = Embedding::new(cfg.max_position_embeddings, cfg.hidden_size, vb.pp("position_embeddings"))?;
        let token_type_embeddings = Embedding::new(2, cfg.hidden_size, vb.pp("token_type_embeddings"))?;
        let layer_norm = layer_norm(cfg.hidden_size, 1e-12, vb.pp("LayerNorm"))?;
        let dropout = Dropout::new(cfg.dropout_prob);
        
        Ok(Self {
            token_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, device: &Device) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        
        // Create position ids
        let position_ids = Tensor::arange(0, seq_len as i64, device)?.unsqueeze(0)?.expand((batch_size, seq_len))?;
        
        // Get embeddings
        let token_embeddings = self.token_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        
        // Sum embeddings
        let embeddings = (&token_embeddings + &position_embeddings)? + &token_type_embeddings?;
        
        // Layer norm and dropout
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings, false)
    }
}

// Multi-head attention
pub struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_heads: usize,
    head_size: usize,
}

impl BertSelfAttention {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let head_size = cfg.hidden_size / cfg.num_heads;
        let query = Linear::new(cfg.hidden_size, cfg.hidden_size, vb.pp("query"))?;
        let key = Linear::new(cfg.hidden_size, cfg.hidden_size, vb.pp("key"))?;
        let value = Linear::new(cfg.hidden_size, cfg.hidden_size, vb.pp("value"))?;
        let dropout = Dropout::new(cfg.dropout_prob);
        
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_heads: cfg.num_heads,
            head_size,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        
        let query = self.query.forward(hidden_states)?;
        let key = self.key.forward(hidden_states)?;
        let value = self.value.forward(hidden_states)?;
        
        // Reshape for multi-head attention
        let query = query.reshape((batch_size, seq_len, self.num_heads, self.head_size))?.transpose(1, 2)?;
        let key = key.reshape((batch_size, seq_len, self.num_heads, self.head_size))?.transpose(1, 2)?;
        let value = value.reshape((batch_size, seq_len, self.num_heads, self.head_size))?.transpose(1, 2)?;
        
        // Attention scores
        let attention_scores = query.matmul(&key.transpose(2, 3)?)?;
        let attention_scores = attention_scores / (self.head_size as f64).sqrt()?;
        
        // Apply attention mask
        let attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
        let attention_scores = attention_scores.broadcast_add(&(attention_mask * -10000.0)?)?;
        
        // Softmax and dropout
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let attention_probs = self.dropout.forward(&attention_probs, false)?;
        
        // Apply attention to values
        let context = attention_probs.matmul(&value)?;
        let context = context.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.head_size))?;
        
        Ok(context)
    }
}

// BERT self-attention output
pub struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let dense = Linear::new(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, 1e-12, vb.pp("LayerNorm"))?;
        let dropout = Dropout::new(cfg.dropout_prob);
        
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        let hidden_states = self.layer_norm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

// BERT attention layer
pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let self_attention = BertSelfAttention::new(cfg, vb.pp("self"))?;
        let output = BertSelfOutput::new(cfg, vb.pp("output"))?;
        
        Ok(Self {
            self_attention,
            output,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

// BERT intermediate layer (FFN)
pub struct BertIntermediate {
    dense: Linear,
}

impl BertIntermediate {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let dense = Linear::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        hidden_states.gelu()
    }
}

// BERT output layer
pub struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let dense = Linear::new(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, 1e-12, vb.pp("LayerNorm"))?;
        let dropout = Dropout::new(cfg.dropout_prob);
        
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        let hidden_states = self.layer_norm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}

// BERT layer (attention + FFN)
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let attention = BertAttention::new(cfg, vb.pp("attention"))?;
        let intermediate = BertIntermediate::new(cfg, vb.pp("intermediate"))?;
        let output = BertOutput::new(cfg, vb.pp("output"))?;
        
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

// BERT encoder
pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            let layer = BertLayer::new(cfg, vb.pp(&format!("layer.{}", i)))?;
            layers.push(layer);
        }
        
        Ok(Self { layers })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

// Complete BERT model
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
}

impl BertModel {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = BertEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;
        
        Ok(Self {
            embeddings,
            encoder,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor, device: &Device) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids, device)?;
        let encoder_output = self.encoder.forward(&embedding_output, attention_mask)?;
        Ok(encoder_output)
    }
}
```

## 4. Entity typing head with mention pooling

We need to extract the mention representation from the BERT output and classify it into entity types.

```rust
// BERT for entity typing
pub struct BertForEntityTyping {
    bert: BertModel,
    classifier: Linear,
    num_labels: usize,
}

impl BertForEntityTyping {
    pub fn new(cfg: &BertConfig, num_labels: usize, vb: VarBuilder) -> Result<Self> {
        let bert = BertModel::new(cfg, vb.pp("bert"))?;
        let classifier = Linear::new(cfg.hidden_size, num_labels, vb.pp("classifier"))?;
        
        Ok(Self {
            bert,
            classifier,
            num_labels,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor, device: &Device) -> Result<Tensor> {
        // Get BERT outputs
        let sequence_output = self.bert.forward(input_ids, token_type_ids, attention_mask, device)?;
        
        // Find mention spans and pool
        let pooled_output = self.pool_mention_representations(&sequence_output, input_ids, device)?;
        
        // Classify
        let logits = self.classifier.forward(&pooled_output)?;
        Ok(logits)
    }
    
    fn pool_mention_representations(&self, sequence_output: &Tensor, input_ids: &Tensor, device: &Device) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = sequence_output.dims3()?;
        let m_start_id = SPECIALS[4].1; // [M_START]
        let m_end_id = SPECIALS[5].1;   // [M_END]
        
        let mut pooled_representations = Vec::new();
        
        for b in 0..batch_size {
            // Extract input ids for this batch item
            let batch_input_ids = input_ids.i(b)?;
            let batch_sequence_output = sequence_output.i(b)?;
            
            // Find mention markers
            let mut start_idx = None;
            let mut end_idx = None;
            
            for i in 0..seq_len {
                let token_id = batch_input_ids.i(i)?.to_scalar::<u32>()? as usize;
                if token_id == m_start_id && start_idx.is_none() {
                    start_idx = Some(i);
                } else if token_id == m_end_id && end_idx.is_none() && start_idx.is_some() {
                    end_idx = Some(i);
                    break;
                }
            }
            
            // Pool mention representation
            let pooled = if let (Some(start), Some(end)) = (start_idx, end_idx) {
                if end > start + 1 {
                    // Average the hidden states between markers (excluding markers)
                    let mention_states = batch_sequence_output.narrow(0, start + 1, end - start - 1)?;
                    mention_states.mean(0)?
                } else {
                    // Fallback to [CLS] token
                    batch_sequence_output.i(0)?
                }
            } else {
                // Fallback to [CLS] token
                batch_sequence_output.i(0)?
            };
            
            pooled_representations.push(pooled);
        }
        
        // Stack all pooled representations
        Tensor::stack(&pooled_representations, 0)
    }
}
```

## 5. Training and evaluation

We'll implement the training loop with binary cross entropy loss for multi-label classification.

```rust
// Binary cross entropy loss for multi-label classification
fn binary_cross_entropy_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // BCE loss: -sum(t * log(sigmoid(x)) + (1-t) * log(1-sigmoid(x)))
    let sigmoid_logits = logits.sigmoid()?;
    let log_sigmoid = sigmoid_logits.log()?;
    let log_one_minus_sigmoid = (sigmoid_logits.neg()? + 1.0)?.log()?;
    
    let pos_loss = targets.mul(&log_sigmoid)?;
    let neg_loss = (targets.neg()? + 1.0)?.mul(&log_one_minus_sigmoid)?;
    let loss = pos_loss.add(&neg_loss)?.neg()?.mean_all()?;
    
    Ok(loss)
}

// Calculate metrics for multi-label classification
fn calculate_multilabel_metrics(logits: &Tensor, targets: &Tensor, threshold: f64) -> Result<(f64, f64, f64)> {
    let predictions = logits.sigmoid()?.ge(threshold)?;
    let targets_bool = targets.ge(0.5)?;
    
    // Convert to vectors for easier calculation
    let preds_vec = predictions.flatten_all()?.to_vec1::<u8>()?;
    let targets_vec = targets_bool.flatten_all()?.to_vec1::<u8>()?;
    
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    
    for (pred, target) in preds_vec.iter().zip(targets_vec.iter()) {
        match (pred, target) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_count += 1,
            _ => {}
        }
    }
    
    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    
    Ok((precision, recall, f1))
}

// Main training function
pub fn run_entity_typing_training(device: &Device) -> Result<()> {
    const MAX_LEN: usize = 64;
    const BATCH_SIZE: usize = 2;
    const EPOCHS: usize = 20;
    const LEARNING_RATE: f64 = 3e-4;
    
    println!("Setting up tokenizer and data...");
    let tokenizer = EntityTypingTokenizer::new(ENTITY_TYPING_ITEMS);
    let vocab_size = tokenizer.vocab_size();
    let num_labels = tokenizer.num_types();
    
    println!("Vocab size: {}, Num labels: {}", vocab_size, num_labels);
    
    // Process training data
    let mut train_data = Vec::new();
    for item in ENTITY_TYPING_ITEMS {
        let (input_ids, token_type_ids, attention_mask, labels) = 
            process_entity_typing_item(item, &tokenizer, MAX_LEN);
        train_data.push((input_ids, token_type_ids, attention_mask, labels));
    }
    
    // Initialize model
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let config = BertConfig::new(vocab_size);
    let model = BertForEntityTyping::new(&config, num_labels, vb)?;
    
    // Simple SGD optimizer (could be improved with Adam)
    let mut optimizer_vars: HashMap<String, Tensor> = HashMap::new();
    
    println!("Starting training...");
    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;
        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut total_f1 = 0.0;
        let mut num_batches = 0;
        
        // Shuffle data
        let mut rng = thread_rng();
        let mut shuffled_data = train_data.clone();
        shuffled_data.shuffle(&mut rng);
        
        // Process batches
        for batch_start in (0..shuffled_data.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(shuffled_data.len());
            let batch = &shuffled_data[batch_start..batch_end];
            
            if batch.is_empty() {
                continue;
            }
            
            // Prepare batch tensors
            let batch_input_ids: Vec<Vec<usize>> = batch.iter().map(|(ids, _, _, _)| ids.clone()).collect();
            let batch_token_type_ids: Vec<Vec<usize>> = batch.iter().map(|(_, tt, _, _)| tt.clone()).collect();
            let batch_attention_mask: Vec<Vec<usize>> = batch.iter().map(|(_, _, am, _)| am.clone()).collect();
            let batch_labels: Vec<Vec<f32>> = batch.iter().map(|(_, _, _, labels)| labels.clone()).collect();
            
            // Convert to tensors
            let input_ids = Tensor::new(batch_input_ids, device)?;
            let token_type_ids = Tensor::new(batch_token_type_ids, device)?;
            let attention_mask = Tensor::new(batch_attention_mask, device)?;
            let labels = Tensor::new(batch_labels, device)?;
            
            // Forward pass
            let logits = model.forward(&input_ids, &token_type_ids, &attention_mask, device)?;
            let loss = binary_cross_entropy_with_logits(&logits, &labels)?;
            
            // Calculate metrics
            let (precision, recall, f1) = calculate_multilabel_metrics(&logits, &labels, 0.5)?;
            
            // Backward pass (simplified - in practice you'd use proper gradients)
            let loss_scalar = loss.to_scalar::<f32>()? as f64;
            total_loss += loss_scalar;
            total_precision += precision;
            total_recall += recall;
            total_f1 += f1;
            num_batches += 1;
            
            // Simple parameter update (this is a simplified version)
            // In practice, you'd compute gradients and update parameters properly
        }
        
        let avg_loss = total_loss / num_batches as f64;
        let avg_precision = total_precision / num_batches as f64;
        let avg_recall = total_recall / num_batches as f64;
        let avg_f1 = total_f1 / num_batches as f64;
        
        println!(
            "Epoch {} | Loss: {:.4} | Precision: {:.3} | Recall: {:.3} | F1: {:.3}",
            epoch + 1, avg_loss, avg_precision, avg_recall, avg_f1
        );
    }
    
    println!("Training completed!");
    
    // Demonstrate inference
    demonstrate_inference(&model, &tokenizer, device)?;
    
    Ok(())
}

// Inference demonstration
fn demonstrate_inference(model: &BertForEntityTyping, tokenizer: &EntityTypingTokenizer, device: &Device) -> Result<()> {
    println!("\n=== Inference Examples ===");
    
    let test_items = vec![
        EntityTypingItem {
            text: "apple inc is located in california".to_string(),
            mention_span: (0, 9), // "apple inc"
            types: vec![], // We'll predict this
        },
        EntityTypingItem {
            text: "barack obama was president".to_string(),
            mention_span: (0, 12), // "barack obama"
            types: vec![], // We'll predict this
        },
    ];
    
    for item in test_items {
        let (input_ids, token_type_ids, attention_mask, _) = 
            process_entity_typing_item(&item, tokenizer, 64);
        
        let input_ids = Tensor::new(vec![input_ids], device)?;
        let token_type_ids = Tensor::new(vec![token_type_ids], device)?;
        let attention_mask = Tensor::new(vec![attention_mask], device)?;
        
        let logits = model.forward(&input_ids, &token_type_ids, &attention_mask, device)?;
        let probabilities = logits.sigmoid()?;
        
        let prob_vec = probabilities.flatten_all()?.to_vec1::<f32>()?;
        
        println!("Text: {}", item.text);
        println!("Mention: {}", &item.text[item.mention_span.0..item.mention_span.1]);
        println!("Predicted types:");
        for (i, &prob) in prob_vec.iter().enumerate() {
            if prob > 0.5 {
                if let Some(type_name) = tokenizer.id_to_type.get(&i) {
                    println!("  {} (confidence: {:.3})", type_name, prob);
                }
            }
        }
        println!();
    }
    
    Ok(())
}
```

## 6. Usage and practical tips

To use this implementation:

1. **Add dependencies**: Include the required Candle dependencies in your `Cargo.toml`
2. **Extend the dataset**: Replace the toy dataset with real entity typing data
3. **Improve tokenization**: Use a proper tokenizer like the `tokenizers` crate for production
4. **Add proper optimization**: Implement Adam optimizer with gradient computation
5. **Handle longer sequences**: Implement proper attention masking for variable-length inputs
6. **Model persistence**: Add saving/loading functionality for trained models

Key differences from PyTorch version:
- Uses Candle tensor operations instead of PyTorch
- Manual gradient computation would be needed for proper training
- Device handling is more explicit in Candle
- Type safety is enforced by Rust's type system

This implementation provides a foundation for entity typing in Candle/Rust, following the same architectural patterns as the PyTorch version while leveraging Rust's safety and performance benefits.