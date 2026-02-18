# BERT: Fine-tuning for Multiple Choice (Candle/Rust)

This chapter shows how to fine‑tune a compact BERT‑style encoder for multiple‑choice tasks using Candle and Rust. We keep everything device‑agnostic and use pure Candle/Rust implementations, consistent with other BERT chapters in this series.

What you will build:
- A simple whitespace tokenizer and a toy multiple‑choice dataset
- Input construction per choice: [CLS] question [SEP] choice [SEP] (optionally with context)
- A compact BERT‑like encoder using Candle components
- A multiple‑choice head that scores each choice using cross-entropy loss over choices
- A clean training/evaluation loop with accuracy and simple inference function

Notes:
- For real tasks (e.g., RACE/SWAG/PIQA), use robust tokenizers (tokenizers crate) and pretrained encoders. This chapter focuses on model architecture, APIs, and a minimal fine‑tune recipe.

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
use rand::{thread_rng, seq::SliceRandom};

fn main() -> Result<()> {
    println!("BERT Multiple Choice Fine-tuning with Candle");
    
    // Select device (CUDA if available, else CPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    Ok(())
}
```

## 2. Simple tokenizer and toy multiple‑choice dataset

We'll define a small dataset with questions and 3–4 choices each. The correct answer is the index of the right choice.

```rust
// Special tokens
const SPECIALS: &[(&str, usize)] = &[
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[MASK]", 3),
];

// Multiple choice item
#[derive(Debug, Clone)]
pub struct MultipleChoiceItem {
    pub question: String,
    pub choices: Vec<String>,
    pub answer: usize,
    pub context: Option<String>,
}

// Toy multiple choice dataset
const MC_ITEMS: &[MultipleChoiceItem] = &[
    MultipleChoiceItem {
        question: "the sky is".to_string(),
        choices: vec!["green".to_string(), "blue".to_string(), "yellow".to_string()],
        answer: 1,
        context: None,
    },
    MultipleChoiceItem {
        question: "cats like to".to_string(),
        choices: vec!["bark".to_string(), "meow".to_string(), "quack".to_string()],
        answer: 1,
        context: None,
    },
    MultipleChoiceItem {
        question: "water is typically".to_string(),
        choices: vec!["solid".to_string(), "liquid".to_string(), "gas".to_string()],
        answer: 1,
        context: None,
    },
    MultipleChoiceItem {
        question: "sun rises in the".to_string(),
        choices: vec!["north".to_string(), "east".to_string(), "south".to_string(), "west".to_string()],
        answer: 1,
        context: None,
    },
];

// Multiple choice tokenizer
pub struct MultipleChoiceTokenizer {
    pub vocab: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
}

impl MultipleChoiceTokenizer {
    pub fn new(items: &[MultipleChoiceItem]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        
        // Add special tokens
        for (token, id) in SPECIALS {
            vocab.insert(token.to_string(), *id);
        }
        
        // Count words in questions and choices
        for item in items {
            // Question words
            for word in item.question.split_whitespace() {
                let word = word.to_lowercase();
                *word_counts.entry(word).or_insert(0) += 1;
            }
            
            // Choice words
            for choice in &item.choices {
                for word in choice.split_whitespace() {
                    let word = word.to_lowercase();
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
            
            // Context words (if present)
            if let Some(ref context) = item.context {
                for word in context.split_whitespace() {
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
    
    pub fn build_choice_input(&self, question: &str, choice: &str, context: Option<&str>, max_len: usize) 
        -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        
        let question_tokens = self.encode(question);
        let choice_tokens = self.encode(choice);
        
        // Build input: [CLS] + question + [SEP] + choice + [SEP] (+ context if provided)
        let mut input_ids = vec![self.vocab["[CLS]"]];
        input_ids.extend(&question_tokens);
        input_ids.push(self.vocab["[SEP]"]);
        
        let mut token_type_ids = vec![0; input_ids.len()];
        
        input_ids.extend(&choice_tokens);
        input_ids.push(self.vocab["[SEP]"]);
        token_type_ids.extend(vec![1; choice_tokens.len() + 1]);
        
        // Add context if provided
        if let Some(ctx) = context {
            let context_tokens = self.encode(ctx);
            input_ids.extend(&context_tokens);
            token_type_ids.extend(vec![1; context_tokens.len()]);
        }
        
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
    
    pub fn prepare_mc_item(&self, item: &MultipleChoiceItem, max_len: usize) 
        -> (Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>, usize) {
        
        let mut choice_inputs = Vec::new();
        
        for choice in &item.choices {
            let (input_ids, token_type_ids, attention_mask) = self.build_choice_input(
                &item.question, 
                choice, 
                item.context.as_deref(), 
                max_len
            );
            choice_inputs.push((input_ids, token_type_ids, attention_mask));
        }
        
        (choice_inputs, item.answer)
    }
}
```

## 3. BERT Multiple Choice Head

```rust
// Multiple choice head that scores each choice
pub struct BertMultipleChoiceHead {
    dropout: candle_nn::Dropout,
    classifier: Linear,
}

impl BertMultipleChoiceHead {
    pub fn new(hidden_size: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let dropout = candle_nn::dropout(dropout)?;
        let classifier = candle_nn::linear(hidden_size, 1, vb.pp("classifier"))?;
        
        Ok(Self {
            dropout,
            classifier,
        })
    }
}

impl Module for BertMultipleChoiceHead {
    fn forward(&self, pooled_output: &Tensor) -> Result<Tensor> {
        let pooled_output = self.dropout.forward(pooled_output, true)?;
        self.classifier.forward(&pooled_output)
    }
}
```

## 4. BERT for Multiple Choice

```rust
// Reuse BertConfig and BertEncoder from previous chapters
use super::bert_finetuning::{BertConfig, BertEncoder};

pub struct BertForMultipleChoice {
    encoder: BertEncoder,
    classifier: BertMultipleChoiceHead,
}

impl BertForMultipleChoice {
    pub fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = BertEncoder::new(cfg, vb.pp("encoder"))?;
        let classifier = BertMultipleChoiceHead::new(cfg.hidden_size, cfg.dropout, vb.pp("classifier"))?;
        
        Ok(Self {
            encoder,
            classifier,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (batch_size, num_choices, seq_len) = input_ids.dims3()?;
        
        // Flatten to process all choices at once: (batch_size * num_choices, seq_len)
        let flat_input_ids = input_ids.flatten(0, 1)?;
        let flat_token_type_ids = token_type_ids.flatten(0, 1)?;
        let flat_attention_mask = attention_mask.flatten(0, 1)?;
        
        // Get encoder output
        let encoder_output = self.encoder.forward(&flat_input_ids, &flat_token_type_ids, &flat_attention_mask)?;
        
        // Use [CLS] token (first token) for classification
        let cls_output = encoder_output.i((.., 0, ..))?; // (batch_size * num_choices, hidden_size)
        
        // Get choice scores
        let choice_scores = self.classifier.forward(&cls_output)?; // (batch_size * num_choices, 1)
        
        // Reshape back to (batch_size, num_choices)
        let scores = choice_scores.squeeze(1)?.reshape((batch_size, num_choices))?;
        
        Ok(scores)
    }
}
```

## 5. Multiple Choice Dataset

```rust
pub struct MultipleChoiceDataset {
    pub items: Vec<MultipleChoiceItem>,
    pub tokenizer: MultipleChoiceTokenizer,
    pub max_len: usize,
}

impl MultipleChoiceDataset {
    pub fn new(items: Vec<MultipleChoiceItem>, max_len: usize) -> Self {
        let tokenizer = MultipleChoiceTokenizer::new(&items);
        Self {
            items,
            tokenizer,
            max_len,
        }
    }
    
    pub fn get_item(&self, idx: usize) -> (Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>, usize) {
        let item = &self.items[idx];
        self.tokenizer.prepare_mc_item(item, self.max_len)
    }
    
    pub fn get_batch(&self, indices: &[usize], device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut all_input_ids = Vec::new();
        let mut all_token_type_ids = Vec::new();
        let mut all_attention_masks = Vec::new();
        let mut all_labels = Vec::new();
        
        let mut max_choices = 0;
        
        // First pass: determine max number of choices
        for &idx in indices {
            let item = &self.items[idx];
            max_choices = max_choices.max(item.choices.len());
        }
        
        // Second pass: prepare batched data
        for &idx in indices {
            let (choice_inputs, label) = self.get_item(idx);
            let num_choices = choice_inputs.len();
            
            let mut batch_input_ids = Vec::new();
            let mut batch_token_type_ids = Vec::new();
            let mut batch_attention_masks = Vec::new();
            
            // Add actual choices
            for (input_ids, token_type_ids, attention_mask) in choice_inputs {
                batch_input_ids.push(input_ids);
                batch_token_type_ids.push(token_type_ids);
                batch_attention_masks.push(attention_mask);
            }
            
            // Pad to max_choices with dummy choices if needed
            while batch_input_ids.len() < max_choices {
                let dummy_choice = vec![self.tokenizer.vocab["[PAD]"]; self.max_len];
                batch_input_ids.push(dummy_choice.clone());
                batch_token_type_ids.push(dummy_choice.clone());
                batch_attention_masks.push(vec![0; self.max_len]);
            }
            
            all_input_ids.push(batch_input_ids);
            all_token_type_ids.push(batch_token_type_ids);
            all_attention_masks.push(batch_attention_masks);
            all_labels.push(label);
        }
        
        let batch_size = indices.len();
        
        // Convert to tensors
        let input_ids_flat: Vec<u32> = all_input_ids.into_iter()
            .flatten()
            .flatten()
            .map(|x| x as u32)
            .collect();
        let token_type_ids_flat: Vec<u32> = all_token_type_ids.into_iter()
            .flatten()
            .flatten()
            .map(|x| x as u32)
            .collect();
        let attention_masks_flat: Vec<u32> = all_attention_masks.into_iter()
            .flatten()
            .flatten()
            .map(|x| x as u32)
            .collect();
        let labels_vec: Vec<u32> = all_labels.into_iter().map(|x| x as u32).collect();
        
        let input_ids_tensor = Tensor::from_slice(&input_ids_flat, (batch_size, max_choices, self.max_len), device)?;
        let token_type_ids_tensor = Tensor::from_slice(&token_type_ids_flat, (batch_size, max_choices, self.max_len), device)?;
        let attention_masks_tensor = Tensor::from_slice(&attention_masks_flat, (batch_size, max_choices, self.max_len), device)?;
        let labels_tensor = Tensor::from_slice(&labels_vec, batch_size, device)?;
        
        Ok((input_ids_tensor, token_type_ids_tensor, attention_masks_tensor, labels_tensor))
    }
}
```

## 6. Training utilities

```rust
fn compute_mc_accuracy(logits: &Tensor, labels: &Tensor) -> Result<f64> {
    let predictions = logits.argmax(1)?;
    let correct = predictions.eq(labels)?;
    let accuracy = correct.to_dtype(DType::F64)?.mean_all()?;
    Ok(accuracy.to_vec0()?)
}

fn predict_choice(model: &BertForMultipleChoice, tokenizer: &MultipleChoiceTokenizer, 
                 item: &MultipleChoiceItem, max_len: usize, device: &Device) -> Result<usize> {
    let (choice_inputs, _) = tokenizer.prepare_mc_item(item, max_len);
    let num_choices = choice_inputs.len();
    
    // Convert to tensors
    let mut input_ids_batch = Vec::new();
    let mut token_type_ids_batch = Vec::new();
    let mut attention_masks_batch = Vec::new();
    
    for (input_ids, token_type_ids, attention_mask) in choice_inputs {
        input_ids_batch.push(input_ids);
        token_type_ids_batch.push(token_type_ids);
        attention_masks_batch.push(attention_mask);
    }
    
    let input_ids_flat: Vec<u32> = input_ids_batch.into_iter().flatten().map(|x| x as u32).collect();
    let token_type_ids_flat: Vec<u32> = token_type_ids_batch.into_iter().flatten().map(|x| x as u32).collect();
    let attention_masks_flat: Vec<u32> = attention_masks_batch.into_iter().flatten().map(|x| x as u32).collect();
    
    let input_ids_tensor = Tensor::from_slice(&input_ids_flat, (1, num_choices, max_len), device)?;
    let token_type_ids_tensor = Tensor::from_slice(&token_type_ids_flat, (1, num_choices, max_len), device)?;
    let attention_masks_tensor = Tensor::from_slice(&attention_masks_flat, (1, num_choices, max_len), device)?;
    
    let logits = model.forward(&input_ids_tensor, &token_type_ids_tensor, &attention_masks_tensor)?;
    let prediction = logits.argmax(1)?.to_vec0::<u32>()? as usize;
    
    Ok(prediction)
}
```

## 7. Training function

```rust
pub fn train_multiple_choice_model() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create dataset
    let items = MC_ITEMS.to_vec();
    let dataset = MultipleChoiceDataset::new(items, 48);
    
    // Initialize model
    let mut cfg = BertConfig::default();
    cfg.vocab_size = dataset.tokenizer.vocab.len();
    cfg.max_len = 48;
    
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BertForMultipleChoice::new(&cfg, vb)?;
    
    // Training parameters
    let lr = 3e-4;
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr, ..Default::default() })?;
    
    let epochs = 50;
    let batch_size = 2;
    
    println!("BERT Multiple Choice model initialized");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Dataset size: {}", dataset.items.len());
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut num_batches = 0;
        
        // Shuffle training indices
        let mut train_indices: Vec<usize> = (0..dataset.items.len()).collect();
        train_indices.shuffle(&mut thread_rng());
        
        for batch_start in (0..train_indices.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_indices.len());
            let batch_indices = &train_indices[batch_start..batch_end];
            
            let (input_ids, token_type_ids, attention_mask, labels) = 
                dataset.get_batch(batch_indices, &device)?;
            
            let logits = model.forward(&input_ids, &token_type_ids, &attention_mask)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
            
            optimizer.backward_step(&loss)?;
            
            // Compute metrics
            let mc_accuracy = compute_mc_accuracy(&logits, &labels)?;
            
            total_loss += loss.to_vec0::<f32>()?;
            total_acc += mc_accuracy;
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        let avg_acc = total_acc / num_batches as f64;
        
        if epoch % 10 == 0 {
            println!("Epoch {:2} | loss: {:.4} | acc: {:.1}%", 
                     epoch, avg_loss, avg_acc * 100.0);
        }
    }
    
    // Example predictions
    println!("\nExample predictions:");
    for (i, item) in dataset.items.iter().enumerate() {
        let prediction = predict_choice(&model, &dataset.tokenizer, item, 48, &device)?;
        let correct = if prediction == item.answer { "✓" } else { "✗" };
        println!("{} Q: '{}' | Predicted: {} ({}), Correct: {}", 
                 correct, item.question, prediction, 
                 item.choices.get(prediction).unwrap_or(&"?".to_string()), 
                 item.answer);
    }
    
    println!("Multiple choice training completed!");
    Ok(())
}
```

## 8. Advanced features

For more sophisticated multiple choice tasks, you might want to add:

```rust
// Support for passage-based questions
pub struct PassageBasedMCItem {
    pub passage: String,
    pub question: String,
    pub choices: Vec<String>,
    pub answer: usize,
}

impl MultipleChoiceTokenizer {
    pub fn build_passage_choice_input(&self, passage: &str, question: &str, choice: &str, max_len: usize) 
        -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        
        let passage_tokens = self.encode(passage);
        let question_tokens = self.encode(question);
        let choice_tokens = self.encode(choice);
        
        // Build input: [CLS] + passage + [SEP] + question + choice + [SEP]
        let mut input_ids = vec![self.vocab["[CLS]"]];
        
        // Add passage (truncated if needed)
        let max_passage_len = max_len / 2;
        let passage_slice = if passage_tokens.len() > max_passage_len { 
            &passage_tokens[..max_passage_len] 
        } else { 
            &passage_tokens 
        };
        input_ids.extend(passage_slice);
        input_ids.push(self.vocab["[SEP]"]);
        
        let mut token_type_ids = vec![0; input_ids.len()];
        
        // Add question + choice
        input_ids.extend(&question_tokens);
        input_ids.extend(&choice_tokens);
        input_ids.push(self.vocab["[SEP]"]);
        token_type_ids.extend(vec![1; question_tokens.len() + choice_tokens.len() + 1]);
        
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

## 9. Complete example

```rust
fn main() -> Result<()> {
    println!("BERT Multiple Choice Fine-tuning (Candle)");
    
    // Train the model
    train_multiple_choice_model()?;
    
    println!("Training completed successfully!");
    Ok(())
}
```

## 10. Practical tips

- **Input formatting**: Each choice is processed as a separate input sequence. The model learns to score each choice independently.
- **Choice balancing**: Ensure training data has balanced answer distributions across choice positions.
- **Context handling**: For reading comprehension tasks, include passage context before the question.
- **Negative sampling**: Consider adding plausible but incorrect choices to improve model discrimination.
- **Evaluation metrics**: Beyond accuracy, consider metrics like choice distribution and confidence scores.
- **Efficiency**: For large choice sets, consider using shared encoders or hierarchical selection.

## 11. Where to go next

- Implement passage-based multiple choice for reading comprehension tasks
- Add confidence estimation and uncertainty quantification
- Experiment with different choice encoding strategies (e.g., choice-aware attention)
- Scale up with larger datasets like RACE, SWAG, or CommonsenseQA
- Implement ensemble methods for improved accuracy
- Add explainability features to understand model reasoning
- Explore few-shot learning approaches for new domains