# BERT: Fine-tuning for Question Answering (Candle/Rust)

This chapter shows how to fine‑tune a compact BERT‑style encoder for extractive question answering (SQuAD‑like) using Candle and Rust. We keep everything device‑agnostic and use pure Candle/Rust implementations, consistent with other BERT chapters in this series.

What you will build:
- A simple whitespace tokenizer and a toy QA dataset with character‑level answers mapped to token spans
- Input construction: [CLS] question [SEP] context [SEP], token_type ids, attention masks
- A compact BERT‑like encoder using Candle components
- A QA head that predicts start and end logits for each token
- A clean training/evaluation loop with span accuracy and simple token‑level metrics
- An inference function to extract the best answer span

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
use candle_nn::{Module, VarBuilder, VarMap, Linear};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("BERT QA Fine-tuning with Candle");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    Ok(())
}
```

## 2. Simple tokenizer and toy QA dataset

We'll use a whitespace tokenizer and define a tiny QA dataset with contexts, questions, and answers given by character offsets. We will map character offsets to token indices for training.

```rust
// Special tokens
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[MASK]", 3),
];

// QA dataset item
#[derive(Debug, Clone)]
pub struct QAItem {
    pub context: String,
    pub question: String,
    pub answer_text: String,
    pub answer_start: usize, // Character position in context
}

// Toy QA examples
pub fn get_qa_items() -> Vec<QAItem> {
    vec![
        QAItem {
            context: "the cat sat on the mat in the sunny room".to_string(),
            question: "where did the cat sit?".to_string(), 
            answer_text: "on the mat".to_string(),
            answer_start: 12,
        },
        QAItem {
            context: "dogs love to play in the park near the river".to_string(),
            question: "where do dogs play?".to_string(),
            answer_text: "in the park".to_string(), 
            answer_start: 19,
        },
        QAItem {
            context: "the weather is sunny and warm today".to_string(),
            question: "what is the weather like?".to_string(),
            answer_text: "sunny and warm".to_string(),
            answer_start: 15,
        },
        QAItem {
            context: "alice went to the store to buy groceries".to_string(), 
            question: "where did alice go?".to_string(),
            answer_text: "to the store".to_string(),
            answer_start: 11,
        },
    ]
}

// Token with character offsets
#[derive(Debug, Clone)]
pub struct TokenWithOffsets {
    pub token: String,
    pub start: usize,
    pub end: usize,
}

// QA Tokenizer with character offset tracking
pub struct QATokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
}

impl QATokenizer {
    pub fn new(qa_items: &[QAItem]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Count words in contexts, questions, and answers
        for item in qa_items {
            for word in item.context.split_whitespace() {
                let word = word.to_lowercase();
                *word_counts.entry(word).or_insert(0) += 1;
            }
            for word in item.question.split_whitespace() {
                let word = word.to_lowercase();
                *word_counts.entry(word).or_insert(0) += 1;
            }
            for word in item.answer_text.split_whitespace() {
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
    
    pub fn tokenize_with_offsets(&self, text: &str) -> Vec<TokenWithOffsets> {
        let mut tokens = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        
        while i < chars.len() {
            // Skip whitespace
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            
            if i >= chars.len() {
                break;
            }
            
            // Find end of token
            let start = i;
            while i < chars.len() && !chars[i].is_whitespace() {
                i += 1;
            }
            
            let token_str: String = chars[start..i].iter().collect();
            tokens.push(TokenWithOffsets {
                token: token_str,
                start,
                end: i,
            });
        }
        
        tokens
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
    
    pub fn find_answer_span(&self, context: &str, answer_start: usize, answer_text: &str) -> (usize, usize) {
        let tokens = self.tokenize_with_offsets(context);
        let answer_end = answer_start + answer_text.len();
        
        let mut start_token_idx = None;
        let mut end_token_idx = None;
        
        // Find tokens that overlap with the answer span
        for (i, token) in tokens.iter().enumerate() {
            // Check if token starts within or overlaps with answer span
            if token.start >= answer_start && token.start < answer_end {
                if start_token_idx.is_none() {
                    start_token_idx = Some(i);
                }
                end_token_idx = Some(i);
            }
            // Check if token ends within answer span
            else if token.end > answer_start && token.end <= answer_end {
                if start_token_idx.is_none() {
                    start_token_idx = Some(i);
                }
                end_token_idx = Some(i);
            }
            // Check if token completely contains answer span
            else if token.start <= answer_start && token.end >= answer_end {
                if start_token_idx.is_none() {
                    start_token_idx = Some(i);
                }
                end_token_idx = Some(i);
            }
        }
        
        let start_idx = start_token_idx.unwrap_or(0);
        let end_idx = end_token_idx.unwrap_or(tokens.len().saturating_sub(1));
        
        (start_idx, end_idx)
    }
    
    pub fn build_qa_input(&self, question: &str, context: &str, max_len: usize) 
        -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        
        let question_tokens = self.encode(question);
        let context_tokens = self.encode(context);
        
        // Calculate available space
        let available_len = max_len - 3; // [CLS] + 2 * [SEP]
        let question_len = question_tokens.len().min(available_len / 3); // Reserve space for context
        let context_len = (available_len - question_len).min(context_tokens.len());
        
        // Build input: [CLS] + question + [SEP] + context + [SEP]
        let mut input_ids = vec![self.vocab["[CLS]"]];
        input_ids.extend(&question_tokens[..question_len]);
        input_ids.push(self.vocab["[SEP]"]);
        
        let question_sep_idx = input_ids.len() - 1;
        input_ids.extend(&context_tokens[..context_len]);
        input_ids.push(self.vocab["[SEP]"]);
        
        // Create token type ids (0 for question, 1 for context)
        let mut token_type_ids = vec![0; question_sep_idx + 1];
        token_type_ids.extend(vec![1; input_ids.len() - question_sep_idx - 1]);
        
        // Create attention mask
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
        
        (padded_ids, padded_token_types, padded_attention)
    }
}
```

## 3. BERT QA head

```rust
// BERT QA Head
pub struct BertQAHead {
    qa_outputs: Linear,
}

impl BertQAHead {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let qa_outputs = candle_nn::linear(hidden_size, 2, vb.pp("qa_outputs"))?;
        Ok(Self { qa_outputs })
    }
}

impl Module for BertQAHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.qa_outputs.forward(hidden_states)
    }
}
```

## 4. BERT for Question Answering

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
            max_len: 64,
            dropout: 0.1,
        }
    }
}

pub struct BertForQuestionAnswering {
    encoder: BertEncoder, // Reuse from fine-tuning
    qa_head: BertQAHead,
}

impl BertForQuestionAnswering {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;
        let qa_head = BertQAHead::new(cfg.hidden_size, vb.pp("qa_head"))?;
        
        Ok(Self {
            encoder,
            qa_head,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor) 
        -> Result<(Tensor, Tensor)> {
        let hidden_states = self.encoder.forward(input_ids, token_type_ids, attention_mask)?;
        let logits = self.qa_head.forward(&hidden_states)?;
        
        // Split into start and end logits
        let start_logits = logits.i((.., .., 0))?;
        let end_logits = logits.i((.., .., 1))?;
        
        Ok((start_logits, end_logits))
    }
}
```

## 5. QA Dataset

```rust
pub struct QADataset {
    pub items: Vec<QAItem>,
    pub tokenizer: QATokenizer,
    pub max_len: usize,
}

impl QADataset {
    pub fn new(items: Vec<QAItem>, max_len: usize) -> Self {
        let tokenizer = QATokenizer::new(&items);
        Self {
            items,
            tokenizer,
            max_len,
        }
    }
    
    pub fn get_item(&self, idx: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>, usize, usize) {
        let item = &self.items[idx];
        let (input_ids, token_type_ids, attention_mask) = 
            self.tokenizer.build_qa_input(&item.question, &item.context, self.max_len);
        
        let (start_pos, end_pos) = self.tokenizer.find_answer_span(
            &item.context, 
            item.answer_start, 
            &item.answer_text
        );
        
        // Adjust positions for the input format: [CLS] + question + [SEP] + context + [SEP]
        let question_tokens = self.tokenizer.encode(&item.question);
        let offset = 1 + question_tokens.len().min((self.max_len - 3) / 3) + 1; // [CLS] + question + [SEP]
        let adjusted_start = (start_pos + offset).min(self.max_len - 1);
        let adjusted_end = (end_pos + offset).min(self.max_len - 1);
        
        (input_ids, token_type_ids, attention_mask, adjusted_start, adjusted_end)
    }
    
    pub fn get_batch(&self, indices: &[usize], device: &Device) 
        -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let mut input_ids = Vec::new();
        let mut token_type_ids = Vec::new();
        let mut attention_masks = Vec::new();
        let mut start_positions = Vec::new();
        let mut end_positions = Vec::new();
        
        for &idx in indices {
            let (ids, token_types, attention, start_pos, end_pos) = self.get_item(idx);
            input_ids.push(ids);
            token_type_ids.push(token_types);
            attention_masks.push(attention);
            start_positions.push(start_pos);
            end_positions.push(end_pos);
        }
        
        let batch_size = indices.len();
        let seq_len = self.max_len;
        
        // Convert to tensors
        let input_ids_flat: Vec<u32> = input_ids.into_iter().flatten().map(|x| x as u32).collect();
        let token_type_ids_flat: Vec<u32> = token_type_ids.into_iter().flatten().map(|x| x as u32).collect();
        let attention_masks_flat: Vec<u32> = attention_masks.into_iter().flatten().map(|x| x as u32).collect();
        let start_positions_vec: Vec<u32> = start_positions.into_iter().map(|x| x as u32).collect();
        let end_positions_vec: Vec<u32> = end_positions.into_iter().map(|x| x as u32).collect();
        
        let input_ids_tensor = Tensor::from_slice(&input_ids_flat, (batch_size, seq_len), device)?;
        let token_type_ids_tensor = Tensor::from_slice(&token_type_ids_flat, (batch_size, seq_len), device)?;
        let attention_masks_tensor = Tensor::from_slice(&attention_masks_flat, (batch_size, seq_len), device)?;
        let start_positions_tensor = Tensor::from_slice(&start_positions_vec, batch_size, device)?;
        let end_positions_tensor = Tensor::from_slice(&end_positions_vec, batch_size, device)?;
        
        Ok((input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, start_positions_tensor, end_positions_tensor))
    }
}
```

## 6. Training utilities

```rust
fn compute_span_accuracy(start_logits: &Tensor, end_logits: &Tensor, 
                        start_positions: &Tensor, end_positions: &Tensor) -> Result<f64> {
    let predicted_starts = start_logits.argmax(1)?;
    let predicted_ends = end_logits.argmax(1)?;
    
    let start_correct = predicted_starts.eq(start_positions)?;
    let end_correct = predicted_ends.eq(end_positions)?;
    let both_correct = start_correct.mul(&end_correct)?;
    
    let accuracy = both_correct.to_dtype(DType::F64)?.mean_all()?;
    Ok(accuracy.to_vec0()?)
}

fn extract_answer_span(start_logits: &Tensor, end_logits: &Tensor) -> Result<(usize, usize)> {
    let start_idx = start_logits.argmax(0)?.to_vec0::<u32>()? as usize;
    let end_idx = end_logits.argmax(0)?.to_vec0::<u32>()? as usize;
    
    // Ensure end >= start
    let end_idx = if end_idx < start_idx { start_idx } else { end_idx };
    
    Ok((start_idx, end_idx))
}
```

## 7. Training function

```rust
pub fn train_qa_model() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create dataset
    let qa_items = get_qa_items();
    let dataset = QADataset::new(qa_items, 64);
    
    // Initialize model
    let mut cfg = BertConfig::default();
    cfg.vocab_size = dataset.tokenizer.vocab.len();
    cfg.max_len = 64;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BertForQuestionAnswering::new(&cfg, vb)?;
    
    // Training parameters
    let lr = 3e-4;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr, ..Default::default() })?;
    
    let epochs = 20;
    let batch_size = 2;
    
    println!("BERT QA model initialized");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Dataset size: {}", dataset.items.len());
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut num_batches = 0;
        
        // Simple iteration over all examples
        for batch_start in (0..dataset.items.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(dataset.items.len());
            let indices: Vec<usize> = (batch_start..batch_end).collect();
            
            let (input_ids, token_type_ids, attention_mask, start_positions, end_positions) = 
                dataset.get_batch(&indices, &device)?;
            
            let (start_logits, end_logits) = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
            
            // Compute losses
            let start_loss = candle_nn::loss::cross_entropy(&start_logits, &start_positions)?;
            let end_loss = candle_nn::loss::cross_entropy(&end_logits, &end_positions)?;
            let total_loss_batch = (&start_loss + &end_loss)?;
            
            optimizer.backward_step(&total_loss_batch)?;
            
            // Compute metrics
            let span_accuracy = compute_span_accuracy(&start_logits, &end_logits, &start_positions, &end_positions)?;
            
            total_loss += total_loss_batch.to_vec0::<f32>()?;
            total_acc += span_accuracy;
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        let avg_acc = total_acc / num_batches as f64;
        
        if epoch % 5 == 0 {
            println!("Epoch {:2} | loss: {:.4} | span acc: {:.1}%", 
                     epoch, avg_loss, avg_acc * 100.0);
        }
    }
    
    println!("QA training completed!");
    Ok(())
}
```

## 8. Inference function

```rust
pub fn predict_answer(model: &BertForQuestionAnswering, tokenizer: &QATokenizer, 
                     question: &str, context: &str, max_len: usize, device: &Device) -> Result<String> {
    let (input_ids, token_type_ids, attention_mask) = 
        tokenizer.build_qa_input(question, context, max_len);
    
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
    
    let (start_logits, end_logits) = model.forward(&input_ids_tensor, &token_type_ids_tensor, &attention_mask_tensor)?;
    
    let start_logits = start_logits.i((0, ..))?;
    let end_logits = end_logits.i((0, ..))?;
    let (start_idx, end_idx) = extract_answer_span(&start_logits, &end_logits)?;
    
    // Extract answer tokens (simplified - would need proper decoding in practice)
    let context_tokens: Vec<&str> = context.split_whitespace().collect();
    let question_len = question.split_whitespace().count();
    
    // Adjust indices to account for [CLS] + question + [SEP]
    let context_start_offset = 1 + question_len + 1;
    
    if start_idx >= context_start_offset && end_idx >= start_idx {
        let relative_start = start_idx - context_start_offset;
        let relative_end = end_idx - context_start_offset;
        
        if relative_end < context_tokens.len() {
            let answer_tokens = &context_tokens[relative_start..=relative_end.min(context_tokens.len() - 1)];
            Ok(answer_tokens.join(" "))
        } else {
            Ok("".to_string())
        }
    } else {
        Ok("".to_string())
    }
}
```

## 9. Complete example

```rust
fn main() -> Result<()> {
    println!("BERT Question Answering Fine-tuning (Candle)");
    
    // Train the model
    train_qa_model()?;
    
    println!("Training completed successfully!");
    Ok(())
}
```

## 10. Practical tips

- **Tokenization**: This whitespace tokenizer is only for demonstration. Use the `tokenizers` crate with WordPiece/BPE for real applications.
- **Answer span mapping**: The character-to-token mapping here is simplified. Real implementations need more robust alignment.
- **Input formatting**: For question-context pairs, ensure proper token type assignments (0 for question, 1 for context).
- **Span constraints**: In practice, add constraints to ensure end_position >= start_position during training and inference.
- **Evaluation metrics**: Implement proper F1 score and exact match metrics for thorough evaluation.
- **Data augmentation**: Consider techniques like back-translation and paraphrasing for better generalization.

## 11. Where to go next

- Explore other BERT fine-tuning tasks in this repository (classification, token classification, etc.)
- Replace the simple tokenizer with a learned tokenizer from the tokenizers crate
- Implement proper evaluation metrics (F1, exact match) for QA tasks
- Experiment with different answer selection strategies beyond simple argmax
- Scale up with larger datasets and pretrained weights for better performance