# 25. Fine-tuning Models

## Introduction

Fine-tuning is a powerful technique that allows you to adapt pretrained models to your specific tasks and domains. Rather than training a model from scratch, which requires large amounts of data and computational resources, fine-tuning leverages the knowledge already captured in pretrained models and adjusts it for your particular needs.

This chapter explores:
- The concept and benefits of fine-tuning
- When to fine-tune versus when to use other transfer learning approaches
- Step-by-step guide to fine-tuning different types of models in Candle
- Advanced fine-tuning techniques and strategies
- Practical examples with code
- Best practices and troubleshooting tips

## Understanding Fine-tuning

### What is Fine-tuning?

Fine-tuning is a specific form of transfer learning where you take a model that has been pretrained on a large dataset and continue training it on a smaller, task-specific dataset. The key difference between fine-tuning and other transfer learning approaches is that in fine-tuning, you update the weights of the pretrained model, rather than just using it as a fixed feature extractor.

The process typically involves:
1. Starting with a pretrained model
2. Replacing the task-specific layers (usually the output layers)
3. Training the model on your dataset, allowing some or all of the pretrained weights to be updated

### Why Fine-tuning Works

Fine-tuning works because many deep learning models learn hierarchical representations:

1. **Lower layers** capture generic features (edges, textures, basic language patterns)
2. **Middle layers** capture domain-specific features (object parts, phrase structures)
3. **Higher layers** capture task-specific features (object categories, semantic meanings)

By fine-tuning, you preserve the general knowledge in the lower and middle layers while adapting the higher layers to your specific task. This approach is particularly effective because:

- The model has already learned useful feature representations from a large dataset
- These representations are often transferable to related tasks
- You need much less task-specific data than training from scratch
- Training converges faster and often achieves better performance

### When to Fine-tune

Fine-tuning is particularly beneficial in the following scenarios:

1. **Limited task-specific data**: When you have a small dataset for your target task
2. **Similar domains**: When your task domain is related to the pretraining domain
3. **Complex tasks**: When your task requires understanding complex patterns that would be difficult to learn from scratch
4. **Time and resource constraints**: When you don't have the resources to train a model from scratch

However, fine-tuning may not always be the best approach. Consider these alternatives in certain situations:

1. **Feature extraction**: If your dataset is very small or very different from the pretraining data
2. **Full retraining**: If your dataset is very large and significantly different from the pretraining data
3. **Prompt engineering**: For large language models when you need quick adaptation without training

## Fine-tuning in Candle

Candle provides the tools and flexibility needed to fine-tune various types of pretrained models. Let's explore how to implement fine-tuning for different model architectures.

### General Fine-tuning Process

Regardless of the specific model type, the general process for fine-tuning in Candle follows these steps:

1. **Load the pretrained model**: Import the model architecture and weights
2. **Modify the model**: Replace or adapt the output layers for your task
3. **Prepare your dataset**: Process and format your data appropriately
4. **Configure training**: Set up optimizers, learning rates, and other hyperparameters
5. **Train the model**: Update the weights using your dataset
6. **Evaluate and iterate**: Assess performance and refine as needed

Let's implement this process in Rust using Candle:

```rust
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder, VarMap, Optimizer};
use std::path::Path;

// Generic fine-tuning function
fn fine_tune<M: Module>(
    model: &mut M,
    train_data: (&Tensor, &Tensor),
    val_data: (&Tensor, &Tensor),
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    device: &Device,
) -> Result<()> {
    let (train_x, train_y) = train_data;
    let (val_x, val_y) = val_data;
    
    // Create variable map for optimization
    let mut varmap = VarMap::new();
    let vars = varmap.all_vars();
    
    // Create optimizer
    let mut optimizer = candle_nn::AdamW::new(vars, learning_rate)?;
    
    // Training loop
    for epoch in 0..epochs {
        // Training phase
        let mut train_loss = 0.0;
        let n_batches = train_x.dim(0)? / batch_size;
        
        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (batch_idx + 1) * batch_size;
            
            // Get batch
            let batch_x = train_x.narrow(0, start_idx, end_idx - start_idx)?;
            let batch_y = train_y.narrow(0, start_idx, end_idx - start_idx)?;
            
            // Forward pass
            let logits = model.forward(&batch_x)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &batch_y)?;
            
            // Backward pass and optimize
            optimizer.backward_step(&loss)?;
            
            train_loss += loss.to_scalar::<f32>()?;
        }
        
        train_loss /= n_batches as f32;
        
        // Validation phase
        let val_logits = model.forward(val_x)?;
        let val_loss = candle_nn::loss::cross_entropy(&val_logits, val_y)?;
        
        println!(
            "Epoch {}/{}: Train Loss = {:.6}, Val Loss = {:.6}",
            epoch + 1,
            epochs,
            train_loss,
            val_loss.to_scalar::<f32>()?
        );
    }
    
    Ok(())
}
```

### Fine-tuning Language Models

Language models like BERT, GPT, and T5 are commonly fine-tuned for specific NLP tasks. Let's look at how to fine-tune a BERT model for text classification:

```rust
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use candle_transformers::models::bert::{BertConfig, BertModel};

// BERT classifier for fine-tuning
struct BertForClassification {
    bert: BertModel,
    classifier: candle_nn::Linear,
}

impl BertForClassification {
    fn new(bert: BertModel, num_labels: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_size = bert.config().hidden_size;
        let classifier = candle_nn::linear(hidden_size, num_labels, vb)?;
        
        Ok(Self {
            bert,
            classifier,
        })
    }
    
    fn from_pretrained(model_path: &Path, num_labels: usize, device: &Device) -> Result<Self> {
        // Load config
        let config_path = model_path.parent().unwrap().join("config.json");
        let config_str = std::fs::read_to_string(config_path)?;
        let config: BertConfig = serde_json::from_str(&config_str)?;
        
        // Load pretrained weights
        let mut varmap = VarMap::new();
        varmap.load(model_path)?;
        
        // Create BERT model
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let bert = BertModel::new(&config, vb)?;
        
        // Create classifier with new random weights
        let classifier_vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, device);
        Self::new(bert, num_labels, classifier_vb)
    }
}

impl Module for BertForClassification {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Get attention mask (1 for real tokens, 0 for padding)
        let attention_mask = input_ids.ne(0)?;
        
        // Forward pass through BERT
        let bert_output = self.bert.forward(input_ids, &attention_mask, None)?;
        
        // Use the [CLS] token representation (first token)
        let cls_output = bert_output.hidden_states.get(0)?;
        
        // Forward through classifier
        self.classifier.forward(&cls_output)
    }
}

// Example usage
fn fine_tune_bert_for_classification() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Load pretrained BERT
    let model_path = Path::new("models/bert-base-uncased/model.safetensors");
    let mut model = BertForClassification::from_pretrained(model_path, 2, &device)?; // Binary classification
    
    // Load and preprocess your dataset
    let (train_ids, train_labels) = load_and_preprocess_dataset("train.csv", &device)?;
    let (val_ids, val_labels) = load_and_preprocess_dataset("val.csv", &device)?;
    
    // Fine-tune
    fine_tune(
        &mut model,
        (&train_ids, &train_labels),
        (&val_ids, &val_labels),
        2e-5, // Lower learning rate for fine-tuning
        3,    // Typically 2-4 epochs is enough
        16,   // Batch size
        &device,
    )?;
    
    Ok(())
}
```

## Complete Fine-tuning Example: Sentiment Analysis with BERT

Let's put everything together in a complete example of fine-tuning BERT for sentiment analysis:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use candle_transformers::models::bert::{BertConfig, BertModel, BertTokenizer};
use std::path::Path;

// BERT for sentiment analysis
struct BertForSentiment {
    bert: BertModel,
    classifier: candle_nn::Linear,
}

impl BertForSentiment {
    fn new(bert: BertModel, vb: VarBuilder) -> Result<Self> {
        let hidden_size = bert.config().hidden_size;
        let classifier = candle_nn::linear(hidden_size, 2, vb)?; // Binary classification
        
        Ok(Self {
            bert,
            classifier,
        })
    }
    
    fn from_pretrained(model_path: &Path, device: &Device) -> Result<Self> {
        // Load config
        let config_path = model_path.parent().unwrap().join("config.json");
        let config_str = std::fs::read_to_string(config_path)?;
        let config: BertConfig = serde_json::from_str(&config_str)?;
        
        // Load pretrained weights
        let mut varmap = VarMap::new();
        varmap.load(model_path)?;
        
        // Create BERT model
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let bert = BertModel::new(&config, vb)?;
        
        // Create classifier with new random weights
        let classifier_vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, device);
        Self::new(bert, classifier_vb)
    }
}

impl Module for BertForSentiment {
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Forward pass through BERT
        let bert_output = self.bert.forward(input_ids, attention_mask, None)?;
        
        // Use the [CLS] token representation (first token)
        let batch_size = input_ids.dim(0)?;
        let cls_output = bert_output.hidden_states.narrow(1, 0, 1)?.reshape((batch_size, -1))?;
        
        // Forward through classifier
        self.classifier.forward(&cls_output)
    }
}

// Dataset preparation
fn prepare_sentiment_dataset(
    texts: &[String],
    labels: &[u32],
    tokenizer: &BertTokenizer,
    max_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut input_ids = Vec::new();
    let mut attention_masks = Vec::new();
    let mut label_tensors = Vec::new();
    
    for (text, &label) in texts.iter().zip(labels.iter()) {
        // Tokenize
        let encoding = tokenizer.encode(text, max_length)?;
        
        // Add to batches
        input_ids.push(encoding.ids);
        attention_masks.push(encoding.attention_mask);
        label_tensors.push(label);
    }
    
    // Convert to tensors
    let input_ids = Tensor::new(input_ids.as_slice(), device)?;
    let attention_masks = Tensor::new(attention_masks.as_slice(), device)?;
    let labels = Tensor::new(label_tensors.as_slice(), device)?;
    
    Ok((input_ids, attention_masks, labels))
}

// Fine-tuning function
fn fine_tune_bert_sentiment(
    model: &mut BertForSentiment,
    train_data: (&Tensor, &Tensor, &Tensor),
    val_data: (&Tensor, &Tensor, &Tensor),
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    device: &Device,
) -> Result<()> {
    let (train_ids, train_masks, train_labels) = train_data;
    let (val_ids, val_masks, val_labels) = val_data;
    
    // Create variable map for optimization
    let mut varmap = VarMap::new();
    let vars = varmap.all_vars();
    
    // Create optimizer
    let mut optimizer = candle_nn::AdamW::new(vars, learning_rate)?;
    
    // Training loop
    for epoch in 0..epochs {
        // Training phase
        let mut train_loss = 0.0;
        let n_batches = train_ids.dim(0)? / batch_size;
        
        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (batch_idx + 1) * batch_size;
            
            // Get batch
            let batch_ids = train_ids.narrow(0, start_idx, end_idx - start_idx)?;
            let batch_masks = train_masks.narrow(0, start_idx, end_idx - start_idx)?;
            let batch_labels = train_labels.narrow(0, start_idx, end_idx - start_idx)?;
            
            // Forward pass
            let logits = model.forward(&batch_ids, &batch_masks)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &batch_labels)?;
            
            // Backward pass and optimize
            optimizer.backward_step(&loss)?;
            
            train_loss += loss.to_scalar::<f32>()?;
        }
        
        train_loss /= n_batches as f32;
        
        // Validation phase
        let val_logits = model.forward(val_ids, val_masks)?;
        let val_loss = candle_nn::loss::cross_entropy(&val_logits, val_labels)?;
        
        // Calculate accuracy
        let val_predictions = val_logits.argmax(1)?;
        let correct = val_predictions.eq(val_labels)?.sum_all()?.to_scalar::<f32>()?;
        let accuracy = correct / val_labels.dim(0)? as f32;
        
        println!(
            "Epoch {}/{}: Train Loss = {:.6}, Val Loss = {:.6}, Val Accuracy = {:.4}",
            epoch + 1,
            epochs,
            train_loss,
            val_loss.to_scalar::<f32>()?,
            accuracy
        );
    }
    
    Ok(())
}

// Main function
fn main() -> Result<()> {
    // Set up device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Load pretrained BERT
    let model_path = Path::new("models/bert-base-uncased/model.safetensors");
    let mut model = BertForSentiment::from_pretrained(model_path, &device)?;
    
    // Load tokenizer
    let tokenizer_path = model_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = BertTokenizer::from_file(&tokenizer_path)?;
    
    // Load your sentiment dataset
    let (train_texts, train_labels) = load_sentiment_dataset("train.csv")?;
    let (val_texts, val_labels) = load_sentiment_dataset("val.csv")?;
    
    // Prepare dataset
    let train_data = prepare_sentiment_dataset(
        &train_texts,
        &train_labels,
        &tokenizer,
        128, // Max sequence length
        &device,
    )?;
    
    let val_data = prepare_sentiment_dataset(
        &val_texts,
        &val_labels,
        &tokenizer,
        128, // Max sequence length
        &device,
    )?;
    
    // Fine-tune
    fine_tune_bert_sentiment(
        &mut model,
        (&train_data.0, &train_data.1, &train_data.2),
        (&val_data.0, &val_data.1, &val_data.2),
        2e-5, // Learning rate
        3,    // Epochs
        16,   // Batch size
        &device,
    )?;
    
    // Save fine-tuned model
    let mut varmap = VarMap::new();
    // Add model parameters to varmap
    // ...
    varmap.save("models/bert-sentiment/model.safetensors")?;
    
    println!("Fine-tuning complete! Model saved to models/bert-sentiment/model.safetensors");
    
    Ok(())
}
```

## Best Practices for Fine-tuning

### Hyperparameter Selection

Choosing the right hyperparameters is crucial for successful fine-tuning:

1. **Learning rate**: Use a smaller learning rate than when training from scratch (typically 2e-5 to 5e-5 for transformers)
2. **Batch size**: Smaller batch sizes often work better for fine-tuning (16-32)
3. **Number of epochs**: Fine-tuning typically requires fewer epochs (2-4 for most tasks)
4. **Weight decay**: Use moderate weight decay (0.01-0.1) to prevent overfitting

### Data Preparation

Proper data preparation can significantly impact fine-tuning results:

1. **Data cleaning**: Remove noise and irrelevant information
2. **Augmentation**: Use task-appropriate data augmentation techniques
3. **Class balancing**: Address class imbalance issues in your dataset
4. **Preprocessing**: Apply the same preprocessing used during pretraining

### Preventing Overfitting

Fine-tuning on small datasets can lead to overfitting. Here are strategies to prevent it:

1. **Early stopping**: Monitor validation performance and stop when it starts degrading
2. **Layer freezing**: Freeze lower layers and only train upper layers
3. **Dropout**: Increase dropout rates in the fine-tuned layers
4. **Regularization**: Apply stronger regularization than during pretraining
5. **Gradient clipping**: Limit gradient magnitudes to prevent large weight updates

```rust
// Example of applying overfitting prevention techniques
fn configure_for_fine_tuning<M: Module>(model: &mut M) -> Result<()> {
    // Increase dropout
    set_dropout_rate(model, 0.3)?;
    
    // Freeze lower layers
    freeze_layers(model, &["embeddings.", "encoder.layer.0.", "encoder.layer.1."])?;
    
    // Configure optimizer with weight decay
    let mut optimizer = candle_nn::AdamW::new_with_weight_decay(
        model.parameters(),
        0.0,      // No bias decay
        0.01,     // Weight decay
        2e-5,     // Learning rate
        (0.9, 0.999), // Betas
        1e-8,     // Epsilon
    )?;
    
    // Set up gradient clipping
    optimizer.set_gradient_clip_norm(1.0)?;
    
    Ok(())
}
```

### Evaluating Fine-tuned Models

Proper evaluation is essential to ensure your fine-tuned model performs well:

1. **Multiple metrics**: Use task-appropriate metrics beyond accuracy (F1, precision, recall)
2. **Cross-validation**: For small datasets, use k-fold cross-validation
3. **Test set**: Keep a separate test set that is never used during development
4. **Error analysis**: Analyze where your model makes mistakes to identify improvement areas

```rust
// Example of comprehensive model evaluation
fn evaluate_model<M: Module>(
    model: &M,
    test_data: (&Tensor, &Tensor, &Tensor),
    device: &Device,
) -> Result<()> {
    let (test_ids, test_masks, test_labels) = test_data;
    
    // Get predictions
    let logits = model.forward(test_ids, test_masks)?;
    let predictions = logits.argmax(1)?;
    
    // Calculate metrics
    let correct = predictions.eq(test_labels)?.sum_all()?.to_scalar::<f32>()?;
    let accuracy = correct / test_labels.dim(0)? as f32;
    
    // Calculate precision, recall, F1 for each class
    let num_classes = logits.dim(1)?;
    
    for class in 0..num_classes {
        let class_tensor = Tensor::new(&[class as u32], device)?;
        
        // True positives: predicted class and actual class match
        let true_positives = predictions.eq(&class_tensor)?.logical_and(&test_labels.eq(&class_tensor)?)?
            .sum_all()?.to_scalar::<f32>()?;
        
        // False positives: predicted class but actual class doesn't match
        let false_positives = predictions.eq(&class_tensor)?.logical_and(&test_labels.ne(&class_tensor)?)?
            .sum_all()?.to_scalar::<f32>()?;
        
        // False negatives: didn't predict class but actual class matches
        let false_negatives = predictions.ne(&class_tensor)?.logical_and(&test_labels.eq(&class_tensor)?)?
            .sum_all()?.to_scalar::<f32>()?;
        
        // Calculate metrics
        let precision = true_positives / (true_positives + false_positives + 1e-10);
        let recall = true_positives / (true_positives + false_negatives + 1e-10);
        let f1 = 2.0 * precision * recall / (precision + recall + 1e-10);
        
        println!("Class {}: Precision = {:.4}, Recall = {:.4}, F1 = {:.4}", class, precision, recall, f1);
    }
    
    println!("Overall Accuracy: {:.4}", accuracy);
    
    Ok(())
}
```

## Troubleshooting Common Issues

### Catastrophic Forgetting

Catastrophic forgetting occurs when fine-tuning causes the model to "forget" knowledge from pretraining:

**Solutions:**
1. Use a smaller learning rate
2. Implement elastic weight consolidation (EWC)
3. Use layer freezing or progressive unfreezing
4. Apply stronger regularization

### Overfitting

Overfitting happens when the model performs well on training data but poorly on validation data:

**Solutions:**
1. Use more data or data augmentation
2. Freeze more layers
3. Increase dropout and regularization
4. Reduce the number of training epochs
5. Use early stopping

### Underfitting

Underfitting occurs when the model fails to learn the patterns in the training data:

**Solutions:**
1. Train for more epochs
2. Increase the learning rate
3. Unfreeze more layers
4. Reduce regularization
5. Use a larger model

### Training Instability

Training instability manifests as large fluctuations in loss or performance:

**Solutions:**
1. Reduce the learning rate
2. Use gradient clipping
3. Use a learning rate scheduler
4. Increase batch size if possible
5. Check for data issues or label noise

## Conclusion

Fine-tuning pretrained models is a powerful technique that allows you to leverage the knowledge captured in large models and adapt it to your specific tasks with relatively little data and computational resources. In this chapter, we've explored:

- The concept and benefits of fine-tuning
- When to fine-tune versus when to use other transfer learning approaches
- Step-by-step guide to fine-tuning different types of models in Candle
- Advanced fine-tuning techniques and strategies
- Practical examples with code
- Best practices and troubleshooting tips

By applying these techniques, you can efficiently adapt state-of-the-art models to your specific needs, achieving high performance even with limited resources. Fine-tuning bridges the gap between general-purpose pretrained models and specialized applications, making advanced AI capabilities accessible for a wide range of tasks.

## Further Reading

- "How to Fine-Tune BERT for Text Classification" by Devlin et al.
- "ULMFiT: Universal Language Model Fine-tuning for Text Classification" by Howard and Ruder
- "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks" by Gururangan et al.
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" by Raffel et al.
- "A Primer in BERTology: What We Know About How BERT Works" by Rogers et al.
- "Revisiting Few-sample BERT Fine-tuning" by Mosbach et al.