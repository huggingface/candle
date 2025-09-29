# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Candle/Rust)

This chapter walks through the core ideas and a minimal, runnable implementation of BERT pre-training using Candle and Rust. We keep everything device‑agnostic and compact so you can understand objectives and data flow without heavy compute requirements.

What you will learn:
- The two original pre-training objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
- How to assemble a BERT‑like encoder from building blocks using Candle components
- How to build a tokenizer and a small toy corpus for demonstration
- How to form training examples for MLM + NSP
- A compact training/evaluation loop with MLM perplexity and NSP accuracy

Prerequisites:
- Basic understanding of transformer architecture
- Familiarity with Rust and Candle framework
- Knowledge of BERT's bidirectional attention mechanism

Note: This is an educational, small‑scale implementation to make BERT pre-training concrete. For real pretraining, you need large corpora, substantial compute, and many engineering optimizations.

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
use candle_nn::{Module, VarBuilder, VarMap, Linear, LayerNorm, Embedding, Dropout};
use std::collections::HashMap;
use rand::{thread_rng, Rng, seq::SliceRandom};

fn main() -> Result<()> {
    println!("BERT Pre-training with Candle");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    Ok(())
}
```

## 2. Tokenizer and toy corpus

We'll build a simple whitespace tokenizer with reserved vocabulary and a small corpus of sentence pairs to simulate NSP.

```rust
// Special tokens
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[MASK]", 3),
];

// Toy corpus: list of documents (each doc is a list of sentences)
const CORPUS: &[&[&str]] = &[
    &["the cat sat on the mat", "it was purring softly", "the mat was warm"],
    &["dogs love to play", "they run in the park", "the park is large"],
    &["birds can fly", "some birds migrate", "they travel long distances"],
    &["computers process data", "they use algorithms", "algorithms solve problems"],
    &["books contain knowledge", "reading expands the mind", "knowledge is power"],
];

// Pre-training tokenizer
pub struct BertPretrainingTokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
    pub corpus: Vec<Vec<String>>,
}

impl BertPretrainingTokenizer {
    pub fn new() -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Convert corpus and count words
        let corpus: Vec<Vec<String>> = CORPUS.iter()
            .map(|doc| doc.iter().map(|s| s.to_string()).collect())
            .collect();
            
        for doc in &corpus {
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
        
        Self { vocab, itos, corpus }
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

## 3. Forming MLM + NSP training pairs

BERT pretraining samples two sentences (A, B). With probability 0.5, B is the next sentence; otherwise B is random. Inputs are: [CLS] A [SEP] B [SEP]. For MLM, randomly mask 15% of tokens with the BERT strategy.

```rust
// MLM constants
const MLM_PROBABILITY: f64 = 0.15;
const MASK_TOKEN_PROB: f64 = 0.8;  // 80% -> [MASK]
const RANDOM_TOKEN_PROB: f64 = 0.1; // 10% -> random token
const KEEP_TOKEN_PROB: f64 = 0.1;   // 10% -> keep original

impl BertPretrainingTokenizer {
    pub fn sample_sentence_pair(&self) -> (Vec<usize>, Vec<usize>, usize) {
        let mut rng = thread_rng();
        let doc = self.corpus.choose(&mut rng).unwrap();
        
        if doc.len() < 2 {
            let a = self.encode(&doc[0]);
            let b = self.encode(&doc[0]);
            return (a, b, 0);
        }
        
        let i = rng.gen_range(0..doc.len()-1);
        let a = self.encode(&doc[i]);
        
        if rng.gen_bool(0.5) {
            // Positive: next sentence from same document
            let b = self.encode(&doc[i + 1]);
            (a, b, 1)
        } else {
            // Negative: random sentence from different document
            let other_doc = self.corpus.choose(&mut rng).unwrap();
            let j = rng.gen_range(0..other_doc.len());
            let b = self.encode(&other_doc[j]);
            (a, b, 0)
        }
    }
    
    pub fn apply_mlm_masking(&self, tokens: &[usize]) -> (Vec<usize>, Vec<i64>) {
        let mut rng = thread_rng();
        let mut input_tokens = tokens.to_vec();
        let mut labels = vec![-100i64; tokens.len()]; // Use -100 as ignore index
        
        for (i, &token_id) in tokens.iter().enumerate() {
            // Skip special tokens
            if token_id < SPECIALS.len() {
                continue;
            }
            
            if rng.gen_bool(MLM_PROBABILITY) {
                labels[i] = token_id as i64; // Store original token for loss calculation
                
                let rand_val = rng.gen::<f64>();
                if rand_val < MASK_TOKEN_PROB {
                    input_tokens[i] = self.vocab["[MASK]"];
                } else if rand_val < MASK_TOKEN_PROB + RANDOM_TOKEN_PROB {
                    // Replace with random token (excluding specials)
                    let random_id = rng.gen_range(SPECIALS.len()..self.vocab.len());
                    input_tokens[i] = random_id;
                }
                // else: keep original token (KEEP_TOKEN_PROB)
            }
        }
        
        (input_tokens, labels)
    }
    
    pub fn create_pretraining_example(&self, max_len: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<i64>, usize) {
        let (tokens_a, tokens_b, is_next) = self.sample_sentence_pair();
        
        // Truncate to fit in max_len (accounting for [CLS] and [SEP] tokens)
        let max_seq_len = max_len - 3; // [CLS] + [SEP] + [SEP]
        let max_a = max_seq_len / 2;
        let max_b = max_seq_len - max_a;
        
        let tokens_a = if tokens_a.len() > max_a { tokens_a[..max_a].to_vec() } else { tokens_a };
        let tokens_b = if tokens_b.len() > max_b { tokens_b[..max_b].to_vec() } else { tokens_b };
        
        // Create input sequence: [CLS] + A + [SEP] + B + [SEP]
        let mut tokens = vec![self.vocab["[CLS]"]];
        tokens.extend(&tokens_a);
        tokens.push(self.vocab["[SEP]"]);
        let sep_index = tokens.len() - 1;
        tokens.extend(&tokens_b);
        tokens.push(self.vocab["[SEP]"]);
        
        // Create token type ids (0 for sentence A, 1 for sentence B)
        let mut token_type_ids = vec![0; sep_index + 1];
        token_type_ids.extend(vec![1; tokens.len() - sep_index - 1]);
        
        // Apply MLM masking
        let (masked_tokens, mlm_labels) = self.apply_mlm_masking(&tokens);
        
        // Create attention mask
        let attention_mask = vec![1; masked_tokens.len()];
        
        // Pad sequences
        let mut padded_tokens = masked_tokens;
        let mut padded_token_types = token_type_ids;
        let mut padded_attention = attention_mask;
        let mut padded_mlm_labels = mlm_labels;
        
        while padded_tokens.len() < max_len {
            padded_tokens.push(self.vocab["[PAD]"]);
            padded_token_types.push(0);
            padded_attention.push(0);
            padded_mlm_labels.push(-100); // Ignore index
        }
        
        (padded_tokens, padded_token_types, padded_attention, padded_mlm_labels, is_next)
    }
}
```

## 4. BERT encoder (reusing components)

We'll reuse the BERT encoder components from the fine-tuning example:

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
            vocab_size: 100,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 4,
            mlp_ratio: 4.0,
            max_len: 64,
            dropout: 0.1,
        }
    }
}

// BERT Embeddings (same as fine-tuning)
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
        
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
            .unsqueeze(0)?
            .expand(input_ids.dims())?;
        
        let token_embeds = self.token_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;
        let token_type_embeds = self.token_type_embeddings.forward(token_type_ids)?;
        
        let embeddings = (&token_embeds + &position_embeds)? + &token_type_embeds?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings, false)
    }
}

// Note: MultiHeadAttention, FeedForward, TransformerBlock, and BertEncoder
// are identical to the fine-tuning implementation and can be reused
```

## 5. MLM and NSP heads

```rust
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
```

## 6. BERT for pre-training

```rust
pub struct BertForPretraining {
    pub encoder: BertEncoder, // Reuse from fine-tuning
    pub mlm_head: BertMLMHead,
    pub nsp_head: BertNSPHead,
}

impl BertForPretraining {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;
        let mlm_head = BertMLMHead::new(cfg, vb.pp("mlm_head"))?;
        let nsp_head = BertNSPHead::new(cfg, vb.pp("nsp_head"))?;
        
        Ok(Self {
            encoder,
            mlm_head,
            nsp_head,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor) 
        -> Result<(Tensor, Tensor)> {
        let hidden_states = self.encoder.forward(input_ids, token_type_ids, attention_mask)?;
        
        // MLM predictions for all tokens
        let mlm_logits = self.mlm_head.forward(&hidden_states)?;
        
        // NSP prediction using [CLS] token
        let cls_token = hidden_states.i((.., 0, ..))?;
        let nsp_logits = self.nsp_head.forward(&cls_token)?;
        
        Ok((mlm_logits, nsp_logits))
    }
}
```

## 7. Pre-training dataset

```rust
pub struct PretrainingDataset {
    pub tokenizer: BertPretrainingTokenizer,
    pub max_len: usize,
}

impl PretrainingDataset {
    pub fn new(max_len: usize) -> Self {
        let tokenizer = BertPretrainingTokenizer::new();
        Self {
            tokenizer,
            max_len,
        }
    }
    
    pub fn get_batch(&self, batch_size: usize, device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let mut input_ids = Vec::new();
        let mut token_type_ids = Vec::new();
        let mut attention_masks = Vec::new();
        let mut mlm_labels = Vec::new();
        let mut nsp_labels = Vec::new();
        
        for _ in 0..batch_size {
            let (ids, token_types, attention, mlm_labs, nsp_label) = 
                self.tokenizer.create_pretraining_example(self.max_len);
            
            input_ids.push(ids);
            token_type_ids.push(token_types);
            attention_masks.push(attention);
            mlm_labels.push(mlm_labs);
            nsp_labels.push(nsp_label);
        }
        
        let seq_len = self.max_len;
        
        // Convert to tensors
        let input_ids_flat: Vec<u32> = input_ids.into_iter().flatten().map(|x| x as u32).collect();
        let token_type_ids_flat: Vec<u32> = token_type_ids.into_iter().flatten().map(|x| x as u32).collect();
        let attention_masks_flat: Vec<u32> = attention_masks.into_iter().flatten().map(|x| x as u32).collect();
        let mlm_labels_flat: Vec<i64> = mlm_labels.into_iter().flatten().collect();
        let nsp_labels_vec: Vec<u32> = nsp_labels.into_iter().map(|x| x as u32).collect();
        
        let input_ids_tensor = Tensor::from_slice(&input_ids_flat, (batch_size, seq_len), device)?;
        let token_type_ids_tensor = Tensor::from_slice(&token_type_ids_flat, (batch_size, seq_len), device)?;
        let attention_masks_tensor = Tensor::from_slice(&attention_masks_flat, (batch_size, seq_len), device)?;
        let mlm_labels_tensor = Tensor::from_slice(&mlm_labels_flat, (batch_size, seq_len), device)?;
        let nsp_labels_tensor = Tensor::from_slice(&nsp_labels_vec, batch_size, device)?;
        
        Ok((input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, mlm_labels_tensor, nsp_labels_tensor))
    }
}
```

## 8. Training utilities

```rust
fn compute_mlm_accuracy(logits: &Tensor, labels: &Tensor) -> Result<f64> {
    let predictions = logits.argmax(2)?;
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

fn compute_nsp_accuracy(logits: &Tensor, labels: &Tensor) -> Result<f64> {
    let predictions = logits.argmax(1)?;
    let correct = predictions.eq(labels)?;
    let accuracy = correct.to_dtype(DType::F64)?.mean_all()?;
    Ok(accuracy.to_vec0()?)
}
```

## 9. Training loop

```rust
pub fn pretrain_model() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create dataset
    let dataset = PretrainingDataset::new(64);
    
    // Initialize model
    let mut cfg = BertConfig::default();
    cfg.vocab_size = dataset.tokenizer.vocab.len();
    cfg.max_len = 64;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BertForPretraining::new(&cfg, vb)?;
    
    // Training parameters
    let lr = 1e-4;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr, ..Default::default() })?;
    
    let epochs = 50;
    let batch_size = 4;
    let steps_per_epoch = 10;
    
    println!("BERT pretraining model initialized");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Max sequence length: {}", cfg.max_len);
    
    for epoch in 0..epochs {
        let mut total_mlm_loss = 0.0;
        let mut total_nsp_loss = 0.0;
        let mut total_mlm_acc = 0.0;
        let mut total_nsp_acc = 0.0;
        
        for step in 0..steps_per_epoch {
            let (input_ids, token_type_ids, attention_mask, mlm_labels, nsp_labels) = 
                dataset.get_batch(batch_size, &device)?;
            
            let (mlm_logits, nsp_logits) = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
            
            // Compute losses
            let mlm_loss = candle_nn::loss::cross_entropy(&mlm_logits.flatten_to(1)?, &mlm_labels.flatten(0, 1)?)?;
            let nsp_loss = candle_nn::loss::cross_entropy(&nsp_logits, &nsp_labels)?;
            let total_loss = (&mlm_loss + &nsp_loss)?;
            
            optimizer.backward_step(&total_loss)?;
            
            // Compute accuracy metrics
            let mlm_acc = compute_mlm_accuracy(&mlm_logits, &mlm_labels)?;
            let nsp_acc = compute_nsp_accuracy(&nsp_logits, &nsp_labels)?;
            
            total_mlm_loss += mlm_loss.to_vec0::<f32>()?;
            total_nsp_loss += nsp_loss.to_vec0::<f32>()?;
            total_mlm_acc += mlm_acc;
            total_nsp_acc += nsp_acc;
        }
        
        let avg_mlm_loss = total_mlm_loss / steps_per_epoch as f32;
        let avg_nsp_loss = total_nsp_loss / steps_per_epoch as f32;
        let avg_mlm_acc = total_mlm_acc / steps_per_epoch as f64;
        let avg_nsp_acc = total_nsp_acc / steps_per_epoch as f64;
        
        if epoch % 10 == 0 {
            println!("Epoch {:3} | MLM loss: {:.4}, acc: {:.1}% | NSP loss: {:.4}, acc: {:.1}%",
                     epoch, avg_mlm_loss, avg_mlm_acc * 100.0,
                     avg_nsp_loss, avg_nsp_acc * 100.0);
        }
    }
    
    println!("Pre-training completed!");
    Ok(())
}
```

## 10. Complete example

```rust
fn main() -> Result<()> {
    println!("BERT Pre-training with Candle/Rust");
    
    // Run pre-training
    pretrain_model()?;
    
    println!("Pre-training completed successfully!");
    Ok(())
}
```

## 11. Practical tips

- **Corpus size**: This toy corpus is minimal. Real BERT pretraining uses massive text corpora (BookCorpus, Wikipedia, etc.).
- **Dynamic masking**: In practice, apply different masking patterns for each epoch rather than static masking.
- **Batch optimization**: Use packed sequences and gradient accumulation for larger effective batch sizes.
- **Learning rate scheduling**: Implement warmup and decay schedules for stable training.
- **Checkpointing**: Save model weights regularly during long training runs.
- **Memory efficiency**: Use gradient checkpointing and mixed precision for large models.

## 12. Where to go next

- Use the pretrained encoder for fine-tuning tasks (classification, QA, etc.)
- Experiment with larger vocabularies using subword tokenization
- Implement more sophisticated masking strategies (whole word masking, etc.)
- Scale up the model size and corpus for better representations
- Explore other pre-training objectives like sentence order prediction