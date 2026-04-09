# BERT: Fine-tuning for Sequence Generation (Candle/Rust)

This chapter shows how to fine‑tune a compact BERT‑style encoder together with a Transformer decoder to perform sequence generation (seq2seq) using Candle and Rust. We keep everything device‑agnostic and use pure Candle/Rust implementations, consistent with other BERT chapters in this series.

What you will build:
- A simple whitespace tokenizer and a toy parallel dataset of (source → target) pairs
- A compact BERT‑style encoder reused from previous chapters
- A minimal Transformer decoder with causal self‑attention + cross‑attention to encoder outputs
- Teacher forcing training with cross-entropy loss and greedy decoding for inference
- Save/load utilities for checkpoints

Notes:
- This is an educational mini seq2seq. For real tasks, use robust tokenizers (tokenizers crate), pretrained checkpoints, and larger datasets.

## 1. Setup and dependencies

Add the necessary dependencies to your `Cargo.toml`:

```toml
[dependencies]
candle-core = "0.3"
candle-nn = "0.3"
rand = "0.8"
```

```rust
use candle_core::{Device, Result, Tensor, DType, IndexOp, D};
use candle_nn::{Module, VarBuilder, VarMap, Linear, LayerNorm, Embedding, Dropout};
use std::collections::HashMap;
use rand::{thread_rng, seq::SliceRandom};

fn main() -> Result<()> {
    println!("BERT Sequence Generation Fine-tuning with Candle");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    Ok(())
}
```

## 2. Simple tokenizer and toy parallel dataset

We'll create a tiny synthetic parallel dataset of short phrase mappings (like toy translation). We use a simple whitespace tokenizer and build a joint vocabulary for both source and target.

```rust
// Special tokens
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),   // BOS for decoder
    ("[SEP]", 2),   // EOS
    ("[MASK]", 3),
];

// Toy parallel pairs: source -> target
const PAIRS: &[(&str, &str)] = &[
    ("i like apples", "j aime les pommes"),
    ("i like cats", "j aime les chats"),
    ("i see a dog", "je vois un chien"),
    ("i see a cat", "je vois un chat"),
    ("i eat bread", "je mange du pain"),
];

// Seq2Seq tokenizer with joint vocabulary
pub struct Seq2SeqTokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
}

impl Seq2SeqTokenizer {
    pub fn new(pairs: &[(&str, &str)]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Count words in both source and target
        for (src, tgt) in pairs {
            for word in src.split_whitespace() {
                let word = word.to_lowercase();
                *word_counts.entry(word).or_insert(0) += 1;
            }
            for word in tgt.split_whitespace() {
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
    
    fn pad_to(ids: Vec<usize>, target_len: usize, pad_id: usize) -> Vec<usize> {
        if ids.len() < target_len {
            let mut padded = ids;
            padded.resize(target_len, pad_id);
            padded
        } else {
            ids[..target_len].to_vec()
        }
    }
    
    pub fn prepare_seq2seq_batch(&self, pairs: &[(&str, &str)], max_src_len: usize, max_tgt_len: usize) 
        -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        
        let mut src_batch = Vec::new();
        let mut src_masks = Vec::new();
        let mut tgt_input_batch = Vec::new();
        let mut tgt_output_batch = Vec::new();
        
        for (src, tgt) in pairs {
            let src_ids = self.encode(src);
            let tgt_ids = self.encode(tgt);
            
            // Source: pad to max_src_len
            let padded_src = Self::pad_to(src_ids.clone(), max_src_len, self.vocab["[PAD]"]);
            let src_mask = src_ids.iter().map(|_| 1).chain(std::iter::repeat(0)).take(max_src_len).collect();
            
            // Target input: [CLS] + target (teacher forcing)
            let mut tgt_input = vec![self.vocab["[CLS]"]];
            tgt_input.extend(&tgt_ids);
            let padded_tgt_input = Self::pad_to(tgt_input, max_tgt_len, self.vocab["[PAD]"]);
            
            // Target output: target + [SEP] (labels)
            let mut tgt_output = tgt_ids;
            tgt_output.push(self.vocab["[SEP]"]);
            let padded_tgt_output = Self::pad_to(tgt_output, max_tgt_len, self.vocab["[PAD]"]);
            
            src_batch.push(padded_src);
            src_masks.push(src_mask);
            tgt_input_batch.push(padded_tgt_input);
            tgt_output_batch.push(padded_tgt_output);
        }
        
        (src_batch, src_masks, tgt_input_batch, tgt_output_batch)
    }
}
```

## 3. BERT encoder (reused from previous chapters)

We reuse the compact BERT encoder from the fine-tuning examples:

```rust
// Reuse BertConfig and BertEncoder from previous chapters
use super::bert_finetuning::{BertConfig, BertEncoder};

#[derive(Debug, Clone)]
pub struct Seq2SeqConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub max_src_len: usize,
    pub max_tgt_len: usize,
    pub dropout: f64,
}

impl Default for Seq2SeqConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100,
            hidden_size: 128,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            num_heads: 4,
            mlp_ratio: 4.0,
            max_src_len: 24,
            max_tgt_len: 24,
            dropout: 0.1,
        }
    }
}
```

## 4. Transformer decoder with cross-attention

```rust
// Decoder embeddings
pub struct DecoderEmbeddings {
    token_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl DecoderEmbeddings {
    pub fn new(cfg: &Seq2SeqConfig, vb: VarBuilder) -> Result<Self> {
        let token_embeddings = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embeddings"))?;
        let position_embeddings = candle_nn::embedding(cfg.max_tgt_len, cfg.hidden_size, vb.pp("position_embeddings"))?;
        let layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("layer_norm"))?;
        let dropout = candle_nn::dropout(cfg.dropout)?;
        
        Ok(Self {
            token_embeddings,
            position_embeddings,
            layer_norm,
            dropout,
        })
    }
}

impl Module for DecoderEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;
        
        // Create position ids
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
            .unsqueeze(0)?
            .expand(input_ids.dims())?;
        
        // Get embeddings
        let token_embeds = self.token_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;
        
        // Sum embeddings
        let embeddings = (&token_embeds + &position_embeds)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings, false)
    }
}

// Multi-Head Cross-Attention
pub struct CrossAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl CrossAttention {
    pub fn new(cfg: &Seq2SeqConfig, vb: VarBuilder) -> Result<Self> {
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

impl CrossAttention {
    pub fn forward(&self, decoder_hidden: &Tensor, encoder_hidden: &Tensor, encoder_mask: &Tensor) -> Result<Tensor> {
        let (batch_size, tgt_len, hidden_size) = decoder_hidden.dims3()?;
        let src_len = encoder_hidden.dim(1)?;
        
        // Queries from decoder, keys and values from encoder
        let q = self.query.forward(decoder_hidden)?;
        let k = self.key.forward(encoder_hidden)?;
        let v = self.value.forward(encoder_hidden)?;
        
        // Reshape for multi-head attention
        let q = q.reshape((batch_size, tgt_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch_size, src_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch_size, src_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        
        // Attention scores
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores / (self.head_dim as f64).sqrt())?;
        
        // Apply encoder mask
        let mask = encoder_mask.unsqueeze(1)?.unsqueeze(2)?;
        let mask = (mask - 1.0)? * 10000.0?;
        let scores = (scores + mask)?;
        
        // Softmax and apply to values
        let attention_probs = candle_nn::ops::softmax(&scores, 3)?;
        let attention_probs = self.dropout.forward(&attention_probs, false)?;
        
        let context = attention_probs.matmul(&v)?;
        let context = context.transpose(1, 2)?.reshape((batch_size, tgt_len, hidden_size))?;
        
        self.output.forward(&context)
    }
}

// Causal Self-Attention for decoder
pub struct CausalSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl CausalSelfAttention {
    pub fn new(cfg: &Seq2SeqConfig, vb: VarBuilder) -> Result<Self> {
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
    
    fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        let mut mask_data = vec![vec![0f32; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i][j] = -10000.0;
            }
        }
        let mask_flat: Vec<f32> = mask_data.into_iter().flatten().collect();
        Tensor::from_slice(&mask_flat, (seq_len, seq_len), device)
    }
}

impl CausalSelfAttention {
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
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
        
        // Apply causal mask
        let causal_mask = Self::create_causal_mask(seq_len, hidden_states.device())?;
        let causal_mask = causal_mask.unsqueeze(0)?.unsqueeze(0)?;
        let scores = (scores + causal_mask)?;
        
        // Softmax and apply to values
        let attention_probs = candle_nn::ops::softmax(&scores, 3)?;
        let attention_probs = self.dropout.forward(&attention_probs, false)?;
        
        let context = attention_probs.matmul(&v)?;
        let context = context.transpose(1, 2)?.reshape((batch_size, seq_len, hidden_size))?;
        
        self.output.forward(&context)
    }
}

// Decoder Layer
pub struct DecoderLayer {
    self_attention: CausalSelfAttention,
    cross_attention: CrossAttention,
    feed_forward: super::bert_finetuning::FeedForward,
    self_attn_layer_norm: LayerNorm,
    cross_attn_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,
}

impl DecoderLayer {
    pub fn new(cfg: &Seq2SeqConfig, vb: VarBuilder) -> Result<Self> {
        let bert_cfg = BertConfig {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_layers: cfg.num_decoder_layers,
            num_heads: cfg.num_heads,
            mlp_ratio: cfg.mlp_ratio,
            max_len: cfg.max_tgt_len,
            dropout: cfg.dropout,
        };
        
        let self_attention = CausalSelfAttention::new(cfg, vb.pp("self_attention"))?;
        let cross_attention = CrossAttention::new(cfg, vb.pp("cross_attention"))?;
        let feed_forward = super::bert_finetuning::FeedForward::new(&bert_cfg, vb.pp("feed_forward"))?;
        let self_attn_layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("self_attn_layer_norm"))?;
        let cross_attn_layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("cross_attn_layer_norm"))?;
        let output_layer_norm = candle_nn::layer_norm(cfg.hidden_size, 1e-12, vb.pp("output_layer_norm"))?;
        
        Ok(Self {
            self_attention,
            cross_attention,
            feed_forward,
            self_attn_layer_norm,
            cross_attn_layer_norm,
            output_layer_norm,
        })
    }
}

impl DecoderLayer {
    pub fn forward(&self, hidden_states: &Tensor, encoder_hidden: &Tensor, encoder_mask: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection and layer norm
        let self_attn_output = self.self_attention.forward(hidden_states)?;
        let hidden_states = (hidden_states + &self_attn_output)?;
        let hidden_states = self.self_attn_layer_norm.forward(&hidden_states)?;
        
        // Cross-attention with residual connection and layer norm
        let cross_attn_output = self.cross_attention.forward(&hidden_states, encoder_hidden, encoder_mask)?;
        let hidden_states = (&hidden_states + &cross_attn_output)?;
        let hidden_states = self.cross_attn_layer_norm.forward(&hidden_states)?;
        
        // Feed forward with residual connection and layer norm
        let feed_forward_output = self.feed_forward.forward(&hidden_states)?;
        let hidden_states = (&hidden_states + &feed_forward_output)?;
        self.output_layer_norm.forward(&hidden_states)
    }
}

// Transformer Decoder
pub struct TransformerDecoder {
    embeddings: DecoderEmbeddings,
    layers: Vec<DecoderLayer>,
    output_projection: Linear,
}

impl TransformerDecoder {
    pub fn new(cfg: &Seq2SeqConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = DecoderEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let mut layers = Vec::new();
        for i in 0..cfg.num_decoder_layers {
            let layer = DecoderLayer::new(cfg, vb.pp(&format!("layer_{}", i)))?;
            layers.push(layer);
        }
        let output_projection = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("output_projection"))?;
        
        Ok(Self {
            embeddings,
            layers,
            output_projection,
        })
    }
}

impl TransformerDecoder {
    pub fn forward(&self, input_ids: &Tensor, encoder_hidden: &Tensor, encoder_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;
        
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, encoder_hidden, encoder_mask)?;
        }
        
        self.output_projection.forward(&hidden_states)
    }
}
```

## 5. Sequence-to-Sequence Model

```rust
pub struct BertSeq2Seq {
    encoder: BertEncoder,
    decoder: TransformerDecoder,
}

impl BertSeq2Seq {
    pub fn new(cfg: &Seq2SeqConfig, vb: VarBuilder) -> Result<Self> {
        let bert_cfg = BertConfig {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_layers: cfg.num_encoder_layers,
            num_heads: cfg.num_heads,
            mlp_ratio: cfg.mlp_ratio,
            max_len: cfg.max_src_len,
            dropout: cfg.dropout,
        };
        
        let encoder = BertEncoder::new(&bert_cfg, vb.pp("encoder"))?;
        let decoder = TransformerDecoder::new(cfg, vb.pp("decoder"))?;
        
        Ok(Self {
            encoder,
            decoder,
        })
    }
    
    pub fn forward(&self, src_ids: &Tensor, src_mask: &Tensor, tgt_ids: &Tensor) -> Result<Tensor> {
        // Encode source sequence
        let token_type_ids = Tensor::zeros_like(src_ids)?;
        let encoder_hidden = self.encoder.forward(src_ids, &token_type_ids, src_mask)?;
        
        // Decode target sequence
        let decoder_logits = self.decoder.forward(tgt_ids, &encoder_hidden, src_mask)?;
        
        Ok(decoder_logits)
    }
}
```

## 6. Dataset and training utilities

```rust
pub struct Seq2SeqDataset {
    pub pairs: Vec<(String, String)>,
    pub tokenizer: Seq2SeqTokenizer,
    pub max_src_len: usize,
    pub max_tgt_len: usize,
}

impl Seq2SeqDataset {
    pub fn new(pairs: Vec<(String, String)>, max_src_len: usize, max_tgt_len: usize) -> Self {
        let pair_refs: Vec<(&str, &str)> = pairs.iter().map(|(s, t)| (s.as_str(), t.as_str())).collect();
        let tokenizer = Seq2SeqTokenizer::new(&pair_refs);
        
        Self {
            pairs,
            tokenizer,
            max_src_len,
            max_tgt_len,
        }
    }
    
    pub fn get_batch(&self, indices: &[usize], device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let batch_pairs: Vec<(&str, &str)> = indices.iter()
            .map(|&idx| (self.pairs[idx].0.as_str(), self.pairs[idx].1.as_str()))
            .collect();
        
        let (src_batch, src_masks, tgt_input_batch, tgt_output_batch) = 
            self.tokenizer.prepare_seq2seq_batch(&batch_pairs, self.max_src_len, self.max_tgt_len);
        
        let batch_size = indices.len();
        
        // Convert to tensors
        let src_flat: Vec<u32> = src_batch.into_iter().flatten().map(|x| x as u32).collect();
        let src_mask_flat: Vec<u32> = src_masks.into_iter().flatten().map(|x| x as u32).collect();
        let tgt_input_flat: Vec<u32> = tgt_input_batch.into_iter().flatten().map(|x| x as u32).collect();
        let tgt_output_flat: Vec<u32> = tgt_output_batch.into_iter().flatten().map(|x| x as u32).collect();
        
        let src_tensor = Tensor::from_slice(&src_flat, (batch_size, self.max_src_len), device)?;
        let src_mask_tensor = Tensor::from_slice(&src_mask_flat, (batch_size, self.max_src_len), device)?;
        let tgt_input_tensor = Tensor::from_slice(&tgt_input_flat, (batch_size, self.max_tgt_len), device)?;
        let tgt_output_tensor = Tensor::from_slice(&tgt_output_flat, (batch_size, self.max_tgt_len), device)?;
        
        Ok((src_tensor, src_mask_tensor, tgt_input_tensor, tgt_output_tensor))
    }
}

fn compute_generation_accuracy(logits: &Tensor, targets: &Tensor, pad_token_id: u32) -> Result<f64> {
    let predictions = logits.argmax(D::Minus1)?;
    
    // Create mask to ignore padding tokens
    let pad_tensor = Tensor::new(pad_token_id, targets.device())?;
    let mask = targets.ne(&pad_tensor)?;
    
    let correct = predictions.eq(targets)?.mul(&mask)?;
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
```

## 7. Training loop

```rust
pub fn train_seq2seq_model() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create dataset
    let pairs: Vec<(String, String)> = PAIRS.iter()
        .map(|(src, tgt)| (src.to_string(), tgt.to_string()))
        .collect();
    
    let dataset = Seq2SeqDataset::new(pairs, 24, 24);
    
    // Initialize model
    let mut cfg = Seq2SeqConfig::default();
    cfg.vocab_size = dataset.tokenizer.vocab.len();
    cfg.max_src_len = 24;
    cfg.max_tgt_len = 24;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BertSeq2Seq::new(&cfg, vb)?;
    
    // Training parameters
    let lr = 5e-4;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr, ..Default::default() })?;
    
    let epochs = 100;
    let batch_size = 2;
    let pad_token_id = dataset.tokenizer.vocab["[PAD]"] as u32;
    
    println!("BERT Seq2Seq model initialized");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Dataset size: {}", dataset.pairs.len());
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut num_batches = 0;
        
        // Shuffle training indices
        let mut train_indices: Vec<usize> = (0..dataset.pairs.len()).collect();
        train_indices.shuffle(&mut thread_rng());
        
        for batch_start in (0..train_indices.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_indices.len());
            let batch_indices = &train_indices[batch_start..batch_end];
            
            let (src_tensor, src_mask_tensor, tgt_input_tensor, tgt_output_tensor) = 
                dataset.get_batch(batch_indices, &device)?;
            
            let logits = model.forward(&src_tensor, &src_mask_tensor, &tgt_input_tensor)?;
            
            // Flatten for loss computation
            let logits_flat = logits.flatten_to(1)?;
            let targets_flat = tgt_output_tensor.flatten(0, 1)?;
            let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
            
            optimizer.backward_step(&loss)?;
            
            // Compute metrics
            let acc = compute_generation_accuracy(&logits, &tgt_output_tensor, pad_token_id)?;
            
            total_loss += loss.to_vec0::<f32>()?;
            total_acc += acc;
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        let avg_acc = total_acc / num_batches as f64;
        
        if epoch % 20 == 0 {
            println!("Epoch {:3} | loss: {:.4} | acc: {:.1}%", 
                     epoch, avg_loss, avg_acc * 100.0);
        }
    }
    
    println!("Seq2Seq training completed!");
    Ok(())
}
```

## 8. Greedy decoding for inference

```rust
pub fn generate_sequence(model: &BertSeq2Seq, tokenizer: &Seq2SeqTokenizer, 
                        src_text: &str, max_src_len: usize, max_tgt_len: usize, 
                        device: &Device) -> Result<String> {
    let src_ids = tokenizer.encode(src_text);
    let padded_src = Seq2SeqTokenizer::pad_to(src_ids.clone(), max_src_len, tokenizer.vocab["[PAD]"]);
    let src_mask: Vec<usize> = src_ids.iter().map(|_| 1).chain(std::iter::repeat(0)).take(max_src_len).collect();
    
    // Convert to tensors
    let src_tensor = Tensor::from_slice(
        &padded_src.into_iter().map(|x| x as u32).collect::<Vec<_>>(), 
        (1, max_src_len), 
        device
    )?;
    let src_mask_tensor = Tensor::from_slice(
        &src_mask.into_iter().map(|x| x as u32).collect::<Vec<_>>(), 
        (1, max_src_len), 
        device
    )?;
    
    // Encode source
    let token_type_ids = Tensor::zeros_like(&src_tensor)?;
    let encoder_hidden = model.encoder.forward(&src_tensor, &token_type_ids, &src_mask_tensor)?;
    
    // Greedy decoding
    let mut generated_ids = vec![tokenizer.vocab["[CLS]"]];
    let bos_id = tokenizer.vocab["[CLS]"];
    let eos_id = tokenizer.vocab["[SEP]"];
    
    for _ in 0..max_tgt_len {
        // Prepare decoder input
        let padded_tgt_input = Seq2SeqTokenizer::pad_to(generated_ids.clone(), max_tgt_len, tokenizer.vocab["[PAD]"]);
        let tgt_input_tensor = Tensor::from_slice(
            &padded_tgt_input.into_iter().map(|x| x as u32).collect::<Vec<_>>(), 
            (1, max_tgt_len), 
            device
        )?;
        
        // Get decoder logits
        let decoder_logits = model.decoder.forward(&tgt_input_tensor, &encoder_hidden, &src_mask_tensor)?;
        
        // Get next token prediction (greedy)
        let next_token_logits = decoder_logits.i((0, generated_ids.len() - 1, ..))?;
        let next_token_id = next_token_logits.argmax(0)?.to_vec0::<u32>()? as usize;
        
        // Stop if EOS token is generated
        if next_token_id == eos_id {
            break;
        }
        
        generated_ids.push(next_token_id);
    }
    
    // Decode the generated sequence (skip BOS token)
    let generated_text = tokenizer.decode(&generated_ids[1..]);
    Ok(generated_text)
}
```

## 9. Complete example

```rust
fn main() -> Result<()> {
    println!("BERT Sequence Generation Fine-tuning (Candle)");
    
    // Train the model
    train_seq2seq_model()?;
    
    println!("Training completed successfully!");
    Ok(())
}
```

## 10. Practical tips

- **Tokenization**: This whitespace tokenizer is only for demonstration. Use the `tokenizers` crate with BPE/WordPiece for real applications.
- **Teacher forcing**: During training, the decoder receives the ground truth previous tokens. During inference, it uses its own predictions.
- **Attention masks**: Proper masking is crucial for both encoder (padding) and decoder (causal + padding).
- **Beam search**: For better generation quality, implement beam search instead of greedy decoding.
- **Length normalization**: Normalize sequence probabilities by length to avoid bias toward shorter sequences.
- **Vocabulary sharing**: Source and target can use separate vocabularies if needed for different languages.

## 11. Where to go next

- Implement beam search for better generation quality
- Add attention visualization to understand encoder-decoder interactions
- Experiment with different decoder architectures (e.g., pointer networks)
- Scale up with larger datasets and pretrained encoder weights
- Implement other generation tasks like summarization or dialogue
- Add evaluation metrics like BLEU score for translation quality assessment