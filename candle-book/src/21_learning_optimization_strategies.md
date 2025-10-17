# Chapter 10: Learning Optimization

## Introduction

One of the most frustrating experiences in deep learning is watching your model struggle to learn. You've set up your neural network, prepared your data, and started training, only to find that the loss barely decreases, oscillates wildly, or your model's predictions are completely off target. These issues are more common than you might think, and fortunately, there are proven strategies and tactics to overcome them.

This chapter addresses the critical question: "What do you do when your model isn't learning effectively?" We'll explore the most common reasons why neural networks fail to converge and provide practical solutions that you can implement immediately.

The strategies covered in this chapter fall into several categories:

1. **Learning Rate Optimization**: Finding the sweet spot between too slow and too fast learning
2. **Gradient Management**: Preventing exploding and vanishing gradients
3. **Loss Function Engineering**: Choosing and designing better loss functions
4. **Regularization Techniques**: Preventing overfitting and improving generalization
5. **Training Dynamics**: Early stopping, batch size optimization, and curriculum learning
6. **Architecture Improvements**: Model design choices that facilitate learning
7. **Data Preprocessing**: Ensuring your data helps rather than hinders learning

Each strategy will be illustrated with practical examples using the Candle library, showing you exactly how to implement these techniques in your own projects.

## Understanding Why Learning Fails

Before diving into solutions, it's crucial to understand the common reasons why neural networks fail to learn effectively:

### 1. Learning Rate Issues

The learning rate is perhaps the most critical hyperparameter in neural network training. When it's too high, the model overshoots optimal solutions and may never converge. When it's too low, training becomes painfully slow and may get stuck in poor local minima.

**Symptoms of poor learning rate:**
- Loss oscillates wildly (too high)
- Loss decreases extremely slowly (too low)
- Loss suddenly explodes to infinity (too high)
- Model seems to "forget" previous learning (too high)

### 2. Gradient Problems

Gradients are the driving force behind neural network learning. When they become too large (exploding gradients) or too small (vanishing gradients), learning becomes ineffective.

**Symptoms of gradient problems:**
- Loss suddenly jumps to very large values (exploding)
- Loss plateaus early in training (vanishing)
- Deeper layers learn much slower than shallow layers (vanishing)

### 3. Poor Loss Function Choice

The loss function defines what the model is trying to optimize. A poorly chosen loss function can make learning difficult or impossible.

**Symptoms of poor loss function:**
- Model converges to trivial solutions
- Loss doesn't correlate with actual performance
- Training is unstable despite reasonable hyperparameters

### 4. Overfitting and Underfitting

Models that are too complex may memorize training data without learning generalizable patterns, while models that are too simple may not have enough capacity to learn the underlying patterns.

**Symptoms:**
- Large gap between training and validation loss (overfitting)
- Both training and validation loss remain high (underfitting)
- Model performs well on training data but poorly on new data (overfitting)

## Strategy 1: Learning Rate Optimization

The learning rate determines how big steps the optimizer takes when updating model parameters. Getting this right is crucial for effective learning.

### Learning Rate Scheduling

Instead of using a fixed learning rate throughout training, learning rate scheduling adjusts the learning rate during training to improve convergence.

```rust
use candle_core::{Device, Result, Tensor};
use candle_nn::{loss, VarBuilder, Optimizer, VarMap};

struct LearningRateScheduler {
    initial_lr: f64,
    decay_rate: f64,
    decay_steps: usize,
    current_step: usize,
}

impl LearningRateScheduler {
    fn new(initial_lr: f64, decay_rate: f64, decay_steps: usize) -> Self {
        Self {
            initial_lr,
            decay_rate,
            decay_steps,
            current_step: 0,
        }
    }
    
    fn get_lr(&mut self) -> f64 {
        self.current_step += 1;
        let decay_factor = (self.current_step / self.decay_steps) as f64;
        self.initial_lr * self.decay_rate.powf(decay_factor)
    }
}

// Example usage in training loop
fn train_with_lr_scheduling() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    // Create your model here
    // let model = YourModel::new(vs.clone())?;
    
    let params = varmap.all_vars();
    let mut optimizer = candle_nn::SGD::new(params, 0.1)?; // Initial learning rate
    let mut scheduler = LearningRateScheduler::new(0.1, 0.95, 100);
    
    for epoch in 0..1000 {
        // Your training step here
        // let loss = compute_loss(&model, &batch)?;
        // optimizer.backward_step(&loss)?;
        
        // Update learning rate every 10 epochs
        if epoch % 10 == 0 {
            let new_lr = scheduler.get_lr();
            // Note: In practice, you'd need to create a new optimizer with the new learning rate
            // This is a simplified example
            println!("Epoch {}: Learning rate = {:.6}", epoch, new_lr);
        }
    }
    
    Ok(())
}
```

### Adaptive Learning Rate Methods

While SGD with learning rate scheduling is effective, adaptive methods like Adam automatically adjust learning rates for each parameter:

```rust
use candle_nn::AdamW;

fn train_with_adaptive_optimizer() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    // Create your model
    // let model = YourModel::new(vs.clone())?;
    
    let params = varmap.all_vars();
    
    // AdamW with weight decay for better generalization
    let mut optimizer = candle_nn::AdamW::new(
        params,
        candle_nn::ParamsAdamW {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    )?;
    
    // Training loop remains the same
    for epoch in 0..1000 {
        // Your training logic here
        // let loss = compute_loss(&model, &batch)?;
        // optimizer.backward_step(&loss)?;
    }
    
    Ok(())
}
```

### Learning Rate Finding

Before training, you can systematically find a good learning rate by gradually increasing it and observing the loss:

```rust
fn find_learning_rate() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    // Create your model and data
    // let model = YourModel::new(vs.clone())?;
    // let training_data = prepare_data()?;
    
    let params = varmap.all_vars();
    let mut learning_rates = Vec::new();
    let mut losses = Vec::new();
    
    // Test learning rates from 1e-6 to 1e-1
    let mut lr = 1e-6;
    while lr < 1e-1 {
        let mut optimizer = candle_nn::SGD::new(params.clone(), lr)?;
        
        // Run a few training steps
        let mut total_loss = 0.0;
        for _ in 0..10 {
            // Your training step here
            // let loss = compute_loss(&model, &batch)?;
            // optimizer.backward_step(&loss)?;
            // total_loss += loss.to_scalar::<f32>()?;
        }
        
        learning_rates.push(lr);
        losses.push(total_loss / 10.0);
        
        println!("LR: {:.2e}, Loss: {:.4}", lr, total_loss / 10.0);
        
        lr *= 1.2; // Increase learning rate by 20%
    }
    
    // The optimal learning rate is typically where loss decreases fastest
    // Look for the steepest negative slope in the loss curve
    
    Ok(())
}
```

## Strategy 2: Gradient Management

Gradient problems are among the most common causes of training failure, especially in deep networks and recurrent neural networks.

### Gradient Clipping

Gradient clipping prevents exploding gradients by limiting the magnitude of gradients during backpropagation:

```rust
use candle_core::Tensor;

fn clip_gradients(params: &[Tensor], max_norm: f64) -> Result<()> {
    // Calculate the total gradient norm
    let mut total_norm_squared = 0.0;
    
    for param in params {
        if let Some(grad) = param.grad() {
            let grad_norm_squared = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            total_norm_squared += grad_norm_squared as f64;
        }
    }
    
    let total_norm = total_norm_squared.sqrt();
    
    // If gradient norm exceeds threshold, scale down all gradients
    if total_norm > max_norm {
        let scale_factor = max_norm / total_norm;
        
        for param in params {
            if let Some(grad) = param.grad() {
                let scaled_grad = (grad * scale_factor)?;
                // Update the gradient (this is conceptual - actual implementation depends on Candle's API)
                // param.set_grad(scaled_grad)?;
            }
        }
        
        println!("Gradients clipped: norm {:.4} -> {:.4}", total_norm, max_norm);
    }
    
    Ok(())
}

// Usage in training loop
fn train_with_gradient_clipping() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    // Create model
    // let model = YourModel::new(vs.clone())?;
    
    let params = varmap.all_vars();
    let mut optimizer = candle_nn::SGD::new(params.clone(), 0.01)?;
    
    for epoch in 0..1000 {
        // Forward pass and loss computation
        // let loss = compute_loss(&model, &batch)?;
        
        // Backward pass
        // loss.backward()?;
        
        // Clip gradients before optimizer step
        clip_gradients(&params, 1.0)?; // Clip to max norm of 1.0
        
        // Optimizer step
        // optimizer.step()?;
        // optimizer.zero_grad()?;
    }
    
    Ok(())
}
```

### Gradient Monitoring

Monitoring gradient statistics helps you understand what's happening during training:

```rust
fn monitor_gradients(params: &[Tensor], epoch: usize) -> Result<()> {
    let mut grad_norms = Vec::new();
    let mut grad_means = Vec::new();
    
    for (i, param) in params.iter().enumerate() {
        if let Some(grad) = param.grad() {
            // Calculate gradient norm
            let grad_norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            grad_norms.push(grad_norm);
            
            // Calculate gradient mean
            let grad_mean = grad.mean_all()?.to_scalar::<f32>()?;
            grad_means.push(grad_mean);
        }
    }
    
    if epoch % 100 == 0 {
        println!("Epoch {}: Gradient norms: {:?}", epoch, grad_norms);
        println!("Epoch {}: Gradient means: {:?}", epoch, grad_means);
        
        // Check for potential problems
        let max_norm = grad_norms.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_norm = grad_norms.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        if max_norm > 10.0 {
            println!("Warning: Large gradients detected (max: {:.4})", max_norm);
        }
        if min_norm < 1e-6 {
            println!("Warning: Very small gradients detected (min: {:.4e})", min_norm);
        }
    }
    
    Ok(())
}
```

## Strategy 3: Loss Function Engineering

The choice of loss function significantly impacts learning dynamics and final performance.

### Custom Loss Functions

Sometimes standard loss functions aren't sufficient for your problem. Here's how to create custom loss functions:

```rust
use candle_core::Tensor;

// Focal Loss - good for imbalanced classification
fn focal_loss(predictions: &Tensor, targets: &Tensor, alpha: f64, gamma: f64) -> Result<Tensor> {
    // Convert targets to one-hot if needed
    let num_classes = predictions.dim(1)?;
    
    // Compute softmax probabilities
    let probs = predictions.softmax(1)?;
    
    // Get probabilities for correct classes
    let target_probs = probs.gather(targets, 1)?;
    
    // Compute focal loss: -alpha * (1 - p)^gamma * log(p)
    let one_minus_p = (1.0 - &target_probs)?;
    let focal_weight = one_minus_p.powf(gamma)?;
    let log_prob = target_probs.log()?;
    let loss = (focal_weight * log_prob * (-alpha))?;
    
    loss.mean_all()
}

// Smooth L1 Loss - less sensitive to outliers than MSE
fn smooth_l1_loss(predictions: &Tensor, targets: &Tensor, beta: f64) -> Result<Tensor> {
    let diff = (predictions - targets)?;
    let abs_diff = diff.abs()?;
    
    // If |diff| < beta, use 0.5 * diff^2 / beta
    // Otherwise, use |diff| - 0.5 * beta
    let quadratic = (diff.sqr()? * (0.5 / beta))?;
    let linear = (abs_diff - (beta * 0.5))?;
    
    // Create mask for quadratic vs linear regions
    let mask = abs_diff.lt(beta)?;
    let loss = mask.where_cond(&quadratic, &linear)?;
    
    loss.mean_all()
}

// Label Smoothing - prevents overconfident predictions
fn label_smoothed_cross_entropy(predictions: &Tensor, targets: &Tensor, smoothing: f64) -> Result<Tensor> {
    let num_classes = predictions.dim(1)? as f64;
    let confidence = 1.0 - smoothing;
    let smooth_value = smoothing / (num_classes - 1.0);
    
    // Create smooth targets
    let one_hot = targets.one_hot(predictions.dim(1)?)?;
    let smooth_targets = (one_hot * confidence + smooth_value)?;
    
    // Compute cross entropy with smooth targets
    let log_probs = predictions.log_softmax(1)?;
    let loss = -(smooth_targets * log_probs)?.sum(1)?;
    
    loss.mean_all()
}
```

### Loss Function Selection Guide

Different problems require different loss functions:

```rust
// Classification problems
fn choose_classification_loss(num_classes: usize, is_balanced: bool) -> String {
    match (num_classes, is_balanced) {
        (2, true) => "Binary Cross Entropy".to_string(),
        (2, false) => "Focal Loss or Weighted Binary Cross Entropy".to_string(),
        (n, true) if n > 2 => "Categorical Cross Entropy".to_string(),
        (n, false) if n > 2 => "Focal Loss or Weighted Categorical Cross Entropy".to_string(),
        _ => "Custom Loss".to_string(),
    }
}

// Regression problems
fn choose_regression_loss(has_outliers: bool, distribution: &str) -> String {
    match (has_outliers, distribution) {
        (false, "normal") => "Mean Squared Error (MSE)".to_string(),
        (true, "normal") => "Smooth L1 Loss or Huber Loss".to_string(),
        (false, "heavy_tailed") => "Mean Absolute Error (MAE)".to_string(),
        (true, "heavy_tailed") => "Quantile Loss".to_string(),
        _ => "Custom Loss based on domain knowledge".to_string(),
    }
}
```

## Strategy 4: Regularization Techniques

Regularization prevents overfitting and improves generalization by constraining the model's complexity.

### Dropout Implementation

Dropout randomly sets some neurons to zero during training, preventing co-adaptation:

```rust
use candle_core::{Tensor, Device};
use rand::Rng;

struct Dropout {
    p: f64, // Dropout probability
    training: bool,
}

impl Dropout {
    fn new(p: f64) -> Self {
        Self { p, training: true }
    }
    
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !self.training || self.p == 0.0 {
            return Ok(x.clone());
        }
        
        // Create dropout mask
        let shape = x.shape();
        let device = x.device();
        
        // Generate random mask (1 with probability 1-p, 0 with probability p)
        let mut rng = rand::thread_rng();
        let mask_data: Vec<f32> = (0..x.elem_count())
            .map(|_| if rng.gen::<f64>() > self.p { 1.0 / (1.0 - self.p) as f32 } else { 0.0 })
            .collect();
        
        let mask = Tensor::from_vec(mask_data, shape, device)?;
        x * mask
    }
}

// Usage in a neural network
struct RegularizedNetwork {
    layer1: candle_nn::Linear,
    dropout1: Dropout,
    layer2: candle_nn::Linear,
    dropout2: Dropout,
    output: candle_nn::Linear,
}

impl RegularizedNetwork {
    fn new(vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            layer1: candle_nn::linear(784, 256, vs.pp("layer1"))?,
            dropout1: Dropout::new(0.5),
            layer2: candle_nn::linear(256, 128, vs.pp("layer2"))?,
            dropout2: Dropout::new(0.3),
            output: candle_nn::linear(128, 10, vs.pp("output"))?,
        })
    }
    
    fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let mut dropout1 = self.dropout1;
        let mut dropout2 = self.dropout2;
        dropout1.set_training(training);
        dropout2.set_training(training);
        
        let x = self.layer1.forward(x)?.relu()?;
        let x = dropout1.forward(&x)?;
        let x = self.layer2.forward(&x)?.relu()?;
        let x = dropout2.forward(&x)?;
        self.output.forward(&x)
    }
}
```

### Weight Decay and L2 Regularization

Weight decay adds a penalty term to the loss function to prevent weights from becoming too large:

```rust
fn compute_l2_penalty(params: &[Tensor], weight_decay: f64) -> Result<Tensor> {
    let mut l2_norm = None;
    
    for param in params {
        let param_l2 = param.sqr()?.sum_all()?;
        l2_norm = match l2_norm {
            None => Some(param_l2),
            Some(norm) => Some((norm + param_l2)?),
        };
    }
    
    match l2_norm {
        Some(norm) => Ok((norm * weight_decay)?),
        None => Ok(Tensor::zeros(&[], candle_core::DType::F32, &Device::Cpu)?),
    }
}

// Training with L2 regularization
fn train_with_regularization() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    // Create model
    let model = RegularizedNetwork::new(vs.clone())?;
    
    let params = varmap.all_vars();
    let mut optimizer = candle_nn::SGD::new(params.clone(), 0.01)?;
    let weight_decay = 0.001;
    
    for epoch in 0..1000 {
        // Forward pass
        // let predictions = model.forward(&batch_x, true)?;
        // let base_loss = loss::cross_entropy(&predictions, &batch_y)?;
        
        // Add L2 regularization
        // let l2_penalty = compute_l2_penalty(&params, weight_decay)?;
        // let total_loss = (base_loss + l2_penalty)?;
        
        // Backward pass
        // optimizer.backward_step(&total_loss)?;
        
        if epoch % 100 == 0 {
            // println!("Epoch {}: Loss = {:.4}, L2 Penalty = {:.4}", 
            //          epoch, base_loss.to_scalar::<f32>()?, l2_penalty.to_scalar::<f32>()?);
        }
    }
    
    Ok(())
}
```

## Strategy 5: Training Dynamics

Optimizing the training process itself can significantly improve learning outcomes.

### Early Stopping

Early stopping prevents overfitting by monitoring validation loss and stopping when it starts to increase:

```rust
struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    best_loss: f64,
    wait: usize,
    stopped: bool,
}

impl EarlyStopping {
    fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait: 0,
            stopped: false,
        }
    }
    
    fn should_stop(&mut self, val_loss: f64) -> bool {
        if val_loss < self.best_loss - self.min_delta {
            self.best_loss = val_loss;
            self.wait = 0;
        } else {
            self.wait += 1;
        }
        
        if self.wait >= self.patience {
            self.stopped = true;
        }
        
        self.stopped
    }
    
    fn is_stopped(&self) -> bool {
        self.stopped
    }
}

// Training with early stopping
fn train_with_early_stopping() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    // Create model
    // let model = YourModel::new(vs.clone())?;
    
    let params = varmap.all_vars();
    let mut optimizer = candle_nn::SGD::new(params, 0.01)?;
    let mut early_stopping = EarlyStopping::new(10, 0.001); // Patience of 10 epochs
    
    for epoch in 0..1000 {
        // Training phase
        // let train_loss = train_epoch(&model, &train_data, &mut optimizer)?;
        
        // Validation phase
        // let val_loss = validate(&model, &val_data)?;
        
        // Check early stopping
        // if early_stopping.should_stop(val_loss) {
        //     println!("Early stopping at epoch {}", epoch);
        //     break;
        // }
        
        if epoch % 10 == 0 {
            // println!("Epoch {}: Train Loss = {:.4}, Val Loss = {:.4}", 
            //          epoch, train_loss, val_loss);
        }
    }
    
    Ok(())
}
```

### Batch Size Optimization

The batch size affects both training stability and convergence speed:

```rust
fn find_optimal_batch_size() -> Result<()> {
    let device = Device::Cpu;
    let batch_sizes = vec![16, 32, 64, 128, 256];
    let mut results = Vec::new();
    
    for &batch_size in &batch_sizes {
        println!("Testing batch size: {}", batch_size);
        
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        
        // Create model
        // let model = YourModel::new(vs.clone())?;
        
        let params = varmap.all_vars();
        let mut optimizer = candle_nn::SGD::new(params, 0.01)?;
        
        // Train for a fixed number of epochs
        let mut final_loss = 0.0;
        for epoch in 0..100 {
            // Training with current batch size
            // let loss = train_epoch_with_batch_size(&model, &train_data, &mut optimizer, batch_size)?;
            // final_loss = loss;
        }
        
        results.push((batch_size, final_loss));
        println!("Batch size {}: Final loss = {:.4}", batch_size, final_loss);
    }
    
    // Find best batch size
    let best = results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    if let Some((best_batch_size, best_loss)) = best {
        println!("Best batch size: {} (loss: {:.4})", best_batch_size, best_loss);
    }
    
    Ok(())
}
```

### Curriculum Learning

Curriculum learning presents training examples in order of increasing difficulty:

```rust
struct CurriculumLearning {
    difficulty_scores: Vec<f64>,
    current_threshold: f64,
    increment: f64,
}

impl CurriculumLearning {
    fn new(difficulty_scores: Vec<f64>, initial_threshold: f64, increment: f64) -> Self {
        Self {
            difficulty_scores,
            current_threshold: initial_threshold,
            increment,
        }
    }
    
    fn get_training_indices(&self, epoch: usize) -> Vec<usize> {
        // Gradually increase difficulty threshold
        let threshold = self.current_threshold + (epoch as f64 * self.increment);
        
        self.difficulty_scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score <= threshold)
            .map(|(idx, _)| idx)
            .collect()
    }
    
    fn update_threshold(&mut self, epoch: usize) {
        self.current_threshold += self.increment;
        // Cap at maximum difficulty
        self.current_threshold = self.current_threshold.min(1.0);
    }
}

// Example: Training with curriculum learning
fn train_with_curriculum() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    // Create model
    // let model = YourModel::new(vs.clone())?;
    
    // Assume we have difficulty scores for each training example
    let difficulty_scores = vec![0.1, 0.3, 0.5, 0.7, 0.9]; // Easy to hard
    let mut curriculum = CurriculumLearning::new(difficulty_scores, 0.2, 0.1);
    
    let params = varmap.all_vars();
    let mut optimizer = candle_nn::SGD::new(params, 0.01)?;
    
    for epoch in 0..100 {
        // Get training indices for current difficulty level
        let training_indices = curriculum.get_training_indices(epoch);
        
        println!("Epoch {}: Training on {} examples", epoch, training_indices.len());
        
        // Train only on selected examples
        // for &idx in &training_indices {
        //     let (x, y) = get_training_example(idx)?;
        //     let loss = compute_loss(&model, &x, &y)?;
        //     optimizer.backward_step(&loss)?;
        // }
        
        // Update curriculum
        curriculum.update_threshold(epoch);
    }
    
    Ok(())
}
```

## Strategy 6: Architecture Improvements

Sometimes the issue isn't with training parameters but with the model architecture itself.

### Residual Connections

Residual connections help with gradient flow in deep networks:

```rust
use candle_core::{Tensor, Module};
use candle_nn::VarBuilder;

struct ResidualBlock {
    layer1: candle_nn::Linear,
    layer2: candle_nn::Linear,
    shortcut: Option<candle_nn::Linear>,
}

impl ResidualBlock {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, vs: VarBuilder) -> Result<Self> {
        let layer1 = candle_nn::linear(input_dim, hidden_dim, vs.pp("layer1"))?;
        let layer2 = candle_nn::linear(hidden_dim, output_dim, vs.pp("layer2"))?;
        
        // Add shortcut connection if dimensions don't match
        let shortcut = if input_dim != output_dim {
            Some(candle_nn::linear(input_dim, output_dim, vs.pp("shortcut"))?)
        } else {
            None
        };
        
        Ok(Self { layer1, layer2, shortcut })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = match &self.shortcut {
            Some(shortcut) => shortcut.forward(x)?,
            None => x.clone(),
        };
        
        let out = self.layer1.forward(x)?.relu()?;
        let out = self.layer2.forward(&out)?;
        
        // Add residual connection
        (out + residual)?.relu()
    }
}

struct ResNet {
    blocks: Vec<ResidualBlock>,
    output: candle_nn::Linear,
}

impl ResNet {
    fn new(layer_dims: Vec<usize>, vs: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::new();
        
        for i in 0..layer_dims.len()-2 {
            let block = ResidualBlock::new(
                layer_dims[i], 
                layer_dims[i+1], 
                layer_dims[i+1], 
                vs.pp(&format!("block_{}", i))
            )?;
            blocks.push(block);
        }
        
        let output = candle_nn::linear(
            *layer_dims.last().unwrap(), 
            1, 
            vs.pp("output")
        )?;
        
        Ok(Self { blocks, output })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        
        self.output.forward(&x)
    }
}
```

### Attention Mechanisms

Attention can help models focus on relevant parts of the input:

```rust
struct SimpleAttention {
    query: candle_nn::Linear,
    key: candle_nn::Linear,
    value: candle_nn::Linear,
    scale: f64,
}

impl SimpleAttention {
    fn new(input_dim: usize, hidden_dim: usize, vs: VarBuilder) -> Result<Self> {
        let query = candle_nn::linear(input_dim, hidden_dim, vs.pp("query"))?;
        let key = candle_nn::linear(input_dim, hidden_dim, vs.pp("key"))?;
        let value = candle_nn::linear(input_dim, hidden_dim, vs.pp("value"))?;
        let scale = 1.0 / (hidden_dim as f64).sqrt();
        
        Ok(Self { query, key, value, scale })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;
        
        // Compute attention scores
        let scores = q.matmul(&k.transpose(1, 2)?)?;
        let scaled_scores = (scores * self.scale)?;
        let attention_weights = scaled_scores.softmax(2)?;
        
        // Apply attention to values
        attention_weights.matmul(&v)
    }
}
```

### Batch Normalization

Batch normalization stabilizes training by normalizing layer inputs:

```rust
struct BatchNorm1d {
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
    momentum: f64,
    training: bool,
}

impl BatchNorm1d {
    fn new(num_features: usize, vs: VarBuilder, device: &Device) -> Result<Self> {
        let gamma = vs.get((num_features,), "gamma")?;
        let beta = vs.get((num_features,), "beta")?;
        let running_mean = Tensor::zeros((num_features,), candle_core::DType::F32, device)?;
        let running_var = Tensor::ones((num_features,), candle_core::DType::F32, device)?;
        
        Ok(Self {
            gamma,
            beta,
            running_mean,
            running_var,
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        })
    }
    
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
    
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        if self.training {
            // Compute batch statistics
            let mean = x.mean(0)?;
            let var = x.var(0)?;
            
            // Update running statistics
            self.running_mean = ((1.0 - self.momentum) * &self.running_mean + self.momentum * &mean)?;
            self.running_var = ((1.0 - self.momentum) * &self.running_var + self.momentum * &var)?;
            
            // Normalize using batch statistics
            let normalized = ((x - &mean)? / (var + self.eps)?.sqrt()?)?;
            (normalized * &self.gamma + &self.beta)
        } else {
            // Use running statistics for inference
            let normalized = ((x - &self.running_mean)? / (self.running_var + self.eps)?.sqrt()?)?;
            (normalized * &self.gamma + &self.beta)
        }
    }
}
```

## Strategy 7: Data Preprocessing and Augmentation

Poor data quality or insufficient data diversity can severely hamper learning.

### Data Normalization

Proper data normalization is crucial for stable training:

```rust
use candle_core::{Tensor, Device};

struct DataNormalizer {
    mean: Tensor,
    std: Tensor,
}

impl DataNormalizer {
    fn fit(data: &Tensor) -> Result<Self> {
        let mean = data.mean(0)?;
        let std = data.std(0)?;
        
        // Prevent division by zero
        let std = std.clamp(1e-8, f64::INFINITY)?;
        
        Ok(Self { mean, std })
    }
    
    fn transform(&self, data: &Tensor) -> Result<Tensor> {
        ((data - &self.mean)? / &self.std)
    }
    
    fn inverse_transform(&self, data: &Tensor) -> Result<Tensor> {
        (data * &self.std + &self.mean)
    }
}

// Different normalization strategies
fn normalize_data(data: &Tensor, method: &str) -> Result<Tensor> {
    match method {
        "z_score" => {
            let mean = data.mean_all()?;
            let std = data.std(0)?.mean_all()?;
            ((data - mean)? / std)
        },
        "min_max" => {
            let min_val = data.min(0)?;
            let max_val = data.max(0)?;
            let range = (max_val - &min_val)?;
            ((data - min_val)? / range)
        },
        "robust" => {
            // Use median and IQR for robust normalization
            let median = data.median(0)?;
            let q75 = data.quantile(0.75, 0, false)?;
            let q25 = data.quantile(0.25, 0, false)?;
            let iqr = (q75 - q25)?;
            ((data - median)? / iqr)
        },
        _ => Ok(data.clone()),
    }
}
```

### Data Augmentation

Data augmentation increases dataset diversity and improves generalization:

```rust
use rand::Rng;

struct DataAugmenter {
    noise_std: f64,
    dropout_prob: f64,
}

impl DataAugmenter {
    fn new(noise_std: f64, dropout_prob: f64) -> Self {
        Self { noise_std, dropout_prob }
    }
    
    fn add_noise(&self, data: &Tensor) -> Result<Tensor> {
        let mut rng = rand::thread_rng();
        let shape = data.shape();
        let device = data.device();
        
        let noise_data: Vec<f32> = (0..data.elem_count())
            .map(|_| rng.gen::<f32>() * self.noise_std as f32)
            .collect();
        
        let noise = Tensor::from_vec(noise_data, shape, device)?;
        data + noise
    }
    
    fn feature_dropout(&self, data: &Tensor) -> Result<Tensor> {
        let mut rng = rand::thread_rng();
        let shape = data.shape();
        let device = data.device();
        
        let mask_data: Vec<f32> = (0..data.elem_count())
            .map(|_| if rng.gen::<f64>() > self.dropout_prob { 1.0 } else { 0.0 })
            .collect();
        
        let mask = Tensor::from_vec(mask_data, shape, device)?;
        data * mask
    }
    
    fn augment(&self, data: &Tensor) -> Result<Tensor> {
        let data = self.add_noise(data)?;
        self.feature_dropout(&data)
    }
}

// Training with data augmentation
fn train_with_augmentation() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    
    // Create model and augmenter
    // let model = YourModel::new(vs.clone())?;
    let augmenter = DataAugmenter::new(0.1, 0.1); // 10% noise, 10% dropout
    
    let params = varmap.all_vars();
    let mut optimizer = candle_nn::SGD::new(params, 0.01)?;
    
    for epoch in 0..1000 {
        // for batch in training_data {
        //     // Augment training data
        //     let augmented_batch = augmenter.augment(&batch.x)?;
        //     
        //     // Forward pass with augmented data
        //     let predictions = model.forward(&augmented_batch)?;
        //     let loss = loss::mse(&predictions, &batch.y)?;
        //     
        //     // Backward pass
        //     optimizer.backward_step(&loss)?;
        // }
    }
    
    Ok(())
}
```

## Putting It All Together: A Complete Training Framework

Here's how to combine multiple strategies into a comprehensive training framework:

```rust
use candle_core::{Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap, Optimizer};

struct TrainingConfig {
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
    weight_decay: f64,
    gradient_clip_norm: f64,
    early_stopping_patience: usize,
    dropout_rate: f64,
    use_batch_norm: bool,
    data_augmentation: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 1000,
            weight_decay: 0.01,
            gradient_clip_norm: 1.0,
            early_stopping_patience: 10,
            dropout_rate: 0.5,
            use_batch_norm: true,
            data_augmentation: true,
        }
    }
}

struct Trainer {
    config: TrainingConfig,
    early_stopping: EarlyStopping,
    lr_scheduler: LearningRateScheduler,
    augmenter: Option<DataAugmenter>,
}

impl Trainer {
    fn new(config: TrainingConfig) -> Self {
        let early_stopping = EarlyStopping::new(config.early_stopping_patience, 0.001);
        let lr_scheduler = LearningRateScheduler::new(config.learning_rate, 0.95, 100);
        let augmenter = if config.data_augmentation {
            Some(DataAugmenter::new(0.1, 0.1))
        } else {
            None
        };
        
        Self {
            config,
            early_stopping,
            lr_scheduler,
            augmenter,
        }
    }
    
    fn train<M: Module>(&mut self, model: &M, train_data: &[(Tensor, Tensor)], val_data: &[(Tensor, Tensor)]) -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        
        // Setup optimizer
        // let params = model.parameters();
        // let mut optimizer = candle_nn::AdamW::new(params, candle_nn::ParamsAdamW {
        //     lr: self.config.learning_rate,
        //     weight_decay: self.config.weight_decay,
        //     ..Default::default()
        // })?;
        
        for epoch in 0..self.config.epochs {
            // Training phase
            let train_loss = self.train_epoch(model, train_data)?;
            
            // Validation phase
            let val_loss = self.validate(model, val_data)?;
            
            // Update learning rate
            let new_lr = self.lr_scheduler.get_lr();
            
            // Check early stopping
            if self.early_stopping.should_stop(val_loss) {
                println!("Early stopping at epoch {}", epoch);
                break;
            }
            
            if epoch % 10 == 0 {
                println!("Epoch {}: Train Loss = {:.4}, Val Loss = {:.4}, LR = {:.6}", 
                         epoch, train_loss, val_loss, new_lr);
            }
        }
        
        Ok(())
    }
    
    fn train_epoch<M: Module>(&self, model: &M, train_data: &[(Tensor, Tensor)]) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        for (x, y) in train_data {
            // Apply data augmentation if enabled
            let x = if let Some(ref augmenter) = self.augmenter {
                augmenter.augment(x)?
            } else {
                x.clone()
            };
            
            // Forward pass
            // let predictions = model.forward(&x)?;
            // let loss = compute_loss_with_regularization(&predictions, y, &model.parameters(), self.config.weight_decay)?;
            
            // Backward pass with gradient clipping
            // loss.backward()?;
            // clip_gradients(&model.parameters(), self.config.gradient_clip_norm)?;
            // optimizer.step()?;
            // optimizer.zero_grad()?;
            
            // total_loss += loss.to_scalar::<f64>()?;
            num_batches += 1;
        }
        
        Ok(total_loss / num_batches as f64)
    }
    
    fn validate<M: Module>(&self, model: &M, val_data: &[(Tensor, Tensor)]) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Set model to evaluation mode (disable dropout, etc.)
        for (x, y) in val_data {
            // Forward pass only
            // let predictions = model.forward(x)?;
            // let loss = compute_loss(&predictions, y)?;
            // total_loss += loss.to_scalar::<f64>()?;
            num_batches += 1;
        }
        
        Ok(total_loss / num_batches as f64)
    }
}

// Usage example
fn main() -> Result<()> {
    let config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 64,
        epochs: 500,
        weight_decay: 0.01,
        gradient_clip_norm: 1.0,
        early_stopping_patience: 15,
        dropout_rate: 0.3,
        use_batch_norm: true,
        data_augmentation: true,
    };
    
    let mut trainer = Trainer::new(config);
    
    // Load your data
    // let (train_data, val_data) = load_data()?;
    
    // Create your model
    // let model = YourModel::new()?;
    
    // Train the model
    // trainer.train(&model, &train_data, &val_data)?;
    
    Ok(())
}
```

## Debugging Checklist

When your model isn't learning effectively, work through this systematic checklist:

### 1. Data Issues
- [ ] Is your data properly normalized?
- [ ] Are there any NaN or infinite values?
- [ ] Is the data distribution reasonable?
- [ ] Are input and target shapes correct?
- [ ] Is there sufficient data diversity?

### 2. Model Architecture
- [ ] Is the model capacity appropriate for the problem?
- [ ] Are activation functions suitable?
- [ ] Do tensor shapes match between layers?
- [ ] Are there any gradient flow issues?

### 3. Loss Function
- [ ] Is the loss function appropriate for the problem?
- [ ] Does the loss correlate with actual performance?
- [ ] Are there numerical stability issues?

### 4. Optimization
- [ ] Is the learning rate in a reasonable range (1e-5 to 1e-1)?
- [ ] Are gradients flowing properly (not too large or small)?
- [ ] Is the optimizer appropriate for the problem?
- [ ] Are you using appropriate regularization?

### 5. Training Process
- [ ] Is the batch size reasonable?
- [ ] Are you training for enough epochs?
- [ ] Is validation loss being monitored?
- [ ] Are you using appropriate data augmentation?

## Conclusion

Effective neural network training is both an art and a science. When your model isn't learning as expected, systematic application of the strategies covered in this chapter will help you identify and resolve the issues.

Remember these key principles:

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Monitor Everything**: Track loss, gradients, learning rates, and validation metrics
3. **Be Systematic**: Change one thing at a time to understand what works
4. **Use Domain Knowledge**: Leverage understanding of your specific problem
5. **Be Patient**: Good models often require experimentation and iteration

The strategies in this chapter provide a comprehensive toolkit for improving neural network training. By combining multiple techniques and systematically debugging issues, you can overcome most learning problems and build models that perform well on your specific tasks.

The key is to understand that training neural networks is an iterative process. Each problem is unique, and the optimal combination of strategies will depend on your specific data, model architecture, and objectives. Use this chapter as a guide, but don't be afraid to experiment and adapt these techniques to your particular situation.