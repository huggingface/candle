# 28. Setting Up Experiments

## Introduction

When developing neural networks, finding the best configuration often requires extensive experimentation. The performance of a neural network depends on numerous design choices and parameters that aren't learned during training but must be set beforehand. These choices, known as hyperparameters, can dramatically affect a model's performance, training speed, and generalization ability.

In this chapter, we'll explore how to set up and run systematic experiments with Candle to optimize your models efficiently. We'll cover:

- Understanding the key hyperparameters that affect neural network performance
- Methods for efficiently searching the hyperparameter space
- Setting up experiment infrastructure to track and compare results
- Implementing parallel experiments to speed up the optimization process
- Practical case studies with MLPs and Transformers

Whether you're fine-tuning a model for a specific task or exploring the capabilities of a new architecture, the techniques in this chapter will help you develop a systematic approach to experimentation that leads to better models with less trial and error.

## Understanding Neural Network Hyperparameters

### What Are Hyperparameters?

In machine learning, we distinguish between two types of parameters:

1. **Model parameters**: These are the values that the model learns during training through optimization algorithms (e.g., weights and biases in neural networks). They are updated iteratively to minimize the loss function.

2. **Hyperparameters**: These are configuration settings that govern the training process and model architecture. They are set before training begins and remain fixed throughout the process.

Hyperparameters are critical because they control the behavior of the training algorithm and the capacity and structure of the model. Finding the right hyperparameters often requires experimentation, as there's rarely a one-size-fits-all configuration that works optimally across different datasets and problems.

### Common Hyperparameters and Their Effects

Let's explore the most important hyperparameters in neural networks and how they affect model performance:

#### Learning Rate

The learning rate controls how much the model parameters are updated during optimization. It's one of the most critical hyperparameters:

- **Too high**: Training may diverge or oscillate around the minimum
- **Too low**: Training will be slow and may get stuck in local minima
- **Just right**: Enables efficient convergence to a good solution

When implementing a training loop with Candle, you typically set the learning rate when creating an optimizer:

```
// Example of setting different learning rates in Candle
let learning_rate_sgd = 0.01; // Standard starting point for SGD
let learning_rate_adam = 0.001; // Standard starting point for Adam

// Creating optimizers with different learning rates
let sgd_optimizer = candle_nn::SGD::new(varmap.all_vars(), learning_rate_sgd)?;
let adam_optimizer = candle_nn::AdamW::new(varmap.all_vars(), learning_rate_adam)?;
```

#### Batch Size

The batch size determines how many training examples are processed before the model parameters are updated:

- **Larger batch sizes**: 
  - More stable gradient estimates
  - Better utilization of hardware (GPU/TPU)
  - May require higher learning rates
  - Can lead to poorer generalization
  
- **Smaller batch sizes**:
  - More noisy updates, which can help escape local minima
  - Less memory usage
  - May require lower learning rates
  - Often better generalization

In Candle, you implement batch processing in your training loop:

```
// Example of setting batch size in a training loop
let batch_size = 32; // Common starting point
let num_samples = dataset.len();
let num_batches = num_samples / batch_size;

for batch_idx in 0..num_batches {
    let start_idx = batch_idx * batch_size;
    
    // Extract batch
    let batch_x = x.narrow(0, start_idx, batch_size)?;
    let batch_y = y.narrow(0, start_idx, batch_size)?;
    
    // Forward pass, loss computation, and optimization
    // ...
}
```

#### Model Architecture

Model architecture hyperparameters define the structure of your neural network:

- **Number of layers**: Controls the depth of the network
- **Layer sizes**: Controls the width of the network
- **Layer types**: Different layer types (linear, attention, etc.) for different tasks

The architecture of your model is one of the most important factors in its performance. For MLPs, key architectural decisions include the number of hidden layers and the number of neurons in each layer. For transformers, you might consider the number of attention heads, the dimensionality of embeddings, and the number of transformer blocks.

#### Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns:

- **ReLU**: Fast, simple, but can suffer from "dying ReLU" problem
- **Leaky ReLU**: Addresses the dying ReLU problem
- **Tanh**: Outputs between -1 and 1, can help in certain recurrent networks
- **Sigmoid**: Outputs between 0 and 1, useful for binary classification
- **GELU**: Smooth approximation of ReLU, popular in transformers

The choice of activation function can significantly impact training dynamics and final performance.

#### Regularization Parameters

Regularization techniques help prevent overfitting:

- **Weight decay**: Penalizes large weights, similar to L2 regularization
- **Dropout rate**: Probability of randomly setting activations to zero during training
- **Batch normalization**: Normalizes layer inputs, can act as a form of regularization

Finding the right balance of regularization is crucial - too little can lead to overfitting, while too much can prevent the model from learning effectively.

#### Optimizer Choice

Different optimizers have different convergence properties:

- **SGD**: Simple, often works well with proper learning rate scheduling
- **SGD with momentum**: Faster convergence than vanilla SGD
- **Adam/AdamW**: Adaptive learning rates, often converges faster
- **RMSProp**: Good for non-stationary objectives

The choice of optimizer can significantly affect both the speed of convergence and the final performance of your model.

#### Other Important Hyperparameters

- **Learning rate schedule**: How the learning rate changes during training
- **Number of training epochs**: How many passes through the dataset
- **Early stopping patience**: How many epochs to wait before stopping if no improvement
- **Random seed**: Controls weight initialization and data shuffling

### The Hyperparameter Tuning Challenge

With so many hyperparameters to consider, finding the optimal configuration becomes a challenging search problem. The number of possible combinations grows exponentially with each additional hyperparameter, making exhaustive search impractical.

Additionally, hyperparameters often interact with each other in complex ways. For example, the optimal learning rate may depend on the batch size, optimizer choice, and model architecture.

This is why systematic experimentation is crucial for developing high-performing neural networks. In the next sections, we'll explore methods for efficiently searching the hyperparameter space and tracking experiment results.

## Methods for Hyperparameter Optimization

Now that we understand the key hyperparameters that affect neural network performance, let's explore methods for efficiently finding optimal hyperparameter configurations.

### Grid Search

Grid search is the most straightforward approach to hyperparameter optimization. It involves defining a set of values for each hyperparameter and evaluating all possible combinations.

#### How Grid Search Works

1. Define a discrete set of values for each hyperparameter
2. Train and evaluate a model for each combination of hyperparameters
3. Select the combination that yields the best performance

```
// Example of defining a grid search space
let learning_rates = vec![0.001, 0.01, 0.1];
let batch_sizes = vec![16, 32, 64];
let hidden_dims = vec![64, 128, 256];

// Total number of combinations
let total_combinations = learning_rates.len() * batch_sizes.len() * hidden_dims.len();
println!("Total combinations to evaluate: {}", total_combinations);

// Nested loops to iterate through all combinations
for &lr in &learning_rates {
    for &batch_size in &batch_sizes {
        for &hidden_dim in &hidden_dims {
            println!("Evaluating: lr={}, batch_size={}, hidden_dim={}", lr, batch_size, hidden_dim);
            
            // Train and evaluate model with these hyperparameters
            // ...
        }
    }
}
```

#### Advantages of Grid Search

- **Comprehensive**: Evaluates all combinations, guaranteeing that the best configuration in the grid is found
- **Simple to implement**: Easy to understand and parallelize
- **Deterministic**: Results are reproducible

#### Limitations of Grid Search

- **Curse of dimensionality**: The number of combinations grows exponentially with the number of hyperparameters
- **Inefficient**: Wastes resources on unpromising regions of the hyperparameter space
- **Discretization**: Limited to the predefined values, potentially missing better configurations between grid points

Grid search is most suitable when:
- You have few hyperparameters to tune (2-3)
- You have a good understanding of reasonable ranges for each hyperparameter
- You have substantial computational resources

### Random Search

Random search, introduced by Bergstra and Bengio (2012), is an alternative to grid search that samples hyperparameter configurations randomly from predefined distributions.

#### How Random Search Works

1. Define a probability distribution for each hyperparameter
2. Randomly sample configurations from these distributions
3. Train and evaluate models for each sampled configuration
4. Select the configuration that yields the best performance

```
// Example of random search implementation
use rand::distributions::{Distribution, Uniform, LogUniform};
use rand::Rng;

// Define distributions for each hyperparameter
let lr_dist = LogUniform::new(0.0001, 0.1); // Log-uniform between 0.0001 and 0.1
let batch_size_dist = Uniform::from(16..=128); // Uniform between 16 and 128
let hidden_dim_dist = Uniform::from(32..=512); // Uniform between 32 and 512

let num_trials = 20; // Number of random configurations to try
let mut rng = rand::thread_rng();

for trial in 0..num_trials {
    // Sample hyperparameters
    let lr = lr_dist.sample(&mut rng);
    let batch_size = batch_size_dist.sample(&mut rng);
    let hidden_dim = hidden_dim_dist.sample(&mut rng);
    
    println!("Trial {}: lr={}, batch_size={}, hidden_dim={}", 
             trial, lr, batch_size, hidden_dim);
    
    // Train and evaluate model with these hyperparameters
    // ...
}
```

#### Advantages of Random Search

- **Efficiency**: Often finds good configurations with fewer trials than grid search
- **Better coverage**: Explores the space more effectively, especially for parameters that matter less
- **Flexibility**: Can use continuous distributions rather than discrete values
- **Anytime algorithm**: Can be stopped at any point and still provide useful results

#### Limitations of Random Search

- **Non-deterministic**: Results may vary between runs
- **No learning**: Doesn't use information from previous trials to inform future ones
- **Still inefficient**: May waste resources on unpromising regions

Random search is particularly effective when:
- Some hyperparameters are more important than others
- You're uncertain about the optimal ranges
- You have a limited computational budget

### Bayesian Optimization

Bayesian optimization is a more sophisticated approach that uses the results of previous evaluations to guide the search for better hyperparameter configurations.

#### How Bayesian Optimization Works

1. Define a prior probability distribution over the objective function
2. Update this distribution with observations (model evaluations)
3. Use an acquisition function to determine the next point to evaluate
4. Repeat until a stopping criterion is met

The key components of Bayesian optimization are:

- **Surrogate model**: A probabilistic model (often a Gaussian Process) that approximates the objective function
- **Acquisition function**: A function that determines which point to evaluate next, balancing exploration and exploitation

While implementing a full Bayesian optimization system in Rust is beyond the scope of this chapter, we can conceptually understand how it would be integrated:

```
// Conceptual example of Bayesian optimization
struct BayesianOptimizer {
    surrogate_model: GaussianProcess,
    hyperparameter_space: HyperparameterSpace,
    observed_configs: Vec<HyperparameterConfig>,
    observed_performances: Vec<f64>,
}

impl BayesianOptimizer {
    // Suggest the next configuration to evaluate
    fn suggest_next_config(&self) -> HyperparameterConfig {
        // Use acquisition function to find the most promising point
        // ...
    }
    
    // Update the surrogate model with a new observation
    fn update(&mut self, config: HyperparameterConfig, performance: f64) {
        self.observed_configs.push(config);
        self.observed_performances.push(performance);
        self.surrogate_model.fit(&self.observed_configs, &self.observed_performances);
    }
}

// Usage in an optimization loop
let mut optimizer = BayesianOptimizer::new(hyperparameter_space);
let num_iterations = 50;

for i in 0..num_iterations {
    // Get next configuration to try
    let config = optimizer.suggest_next_config();
    
    // Train and evaluate model with this configuration
    let performance = train_and_evaluate(config);
    
    // Update the optimizer with the result
    optimizer.update(config, performance);
    
    println!("Iteration {}: Performance = {}", i, performance);
}
```

#### Advantages of Bayesian Optimization

- **Efficiency**: Typically finds good configurations with fewer evaluations than grid or random search
- **Adaptivity**: Learns from previous evaluations to focus on promising regions
- **Uncertainty handling**: Accounts for uncertainty in the objective function
- **Works well with expensive evaluations**: Ideal when each model training is computationally costly

#### Limitations of Bayesian Optimization

- **Complexity**: More difficult to implement and understand
- **Computational overhead**: The surrogate model itself can become expensive to update with many observations
- **Hyperparameters of its own**: Requires configuring the surrogate model and acquisition function

Bayesian optimization is most suitable when:
- Each model evaluation is expensive
- The hyperparameter space is complex
- You have a moderate number of hyperparameters (typically <20)

### Early Stopping Strategies

While not a hyperparameter search method per se, early stopping strategies are crucial for efficient experimentation. They allow you to terminate unpromising trials early, saving computational resources.

#### Common Early Stopping Strategies

1. **Performance threshold**: Stop if performance falls below a certain threshold
2. **Progress-based**: Stop if improvement is too slow
3. **Comparative**: Stop if performance is significantly worse than the best trial so far
4. **Resource-based**: Allocate more resources to promising trials

```
// Example of a simple early stopping implementation
struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    best_value: f64,
    counter: usize,
}

impl EarlyStopping {
    fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_value: f64::NEG_INFINITY,
            counter: 0,
        }
    }
    
    fn should_stop(&mut self, value: f64) -> bool {
        if value > self.best_value + self.min_delta {
            // Improvement
            self.best_value = value;
            self.counter = 0;
        } else {
            // No significant improvement
            self.counter += 1;
        }
        
        self.counter >= self.patience
    }
}

// Usage in training loop
let mut early_stopping = EarlyStopping::new(5, 0.001);
let max_epochs = 100;

for epoch in 0..max_epochs {
    // Train for one epoch
    // ...
    
    // Evaluate on validation set
    let validation_accuracy = evaluate_model();
    
    // Check if we should stop
    if early_stopping.should_stop(validation_accuracy) {
        println!("Early stopping at epoch {}", epoch);
        break;
    }
}
```

#### Multi-Fidelity Methods

More advanced approaches like Successive Halving and Hyperband combine early stopping with efficient resource allocation:

1. **Successive Halving**: Start with many configurations, evaluate all for a small number of epochs, keep the best half, and repeat
2. **Hyperband**: Runs multiple rounds of Successive Halving with different resource allocations

These methods are particularly effective for deep learning, where training to completion is expensive.

### Choosing the Right Method

The best hyperparameter optimization method depends on your specific constraints:

- **Grid Search**: When you have few hyperparameters and want exhaustive evaluation
- **Random Search**: When you have limited resources and many hyperparameters
- **Bayesian Optimization**: When evaluations are expensive and you can afford the overhead
- **Multi-Fidelity Methods**: When you have many configurations to evaluate and can use partial evaluations

In practice, a combination of these methods often works best. For example, you might start with random search to identify promising regions, then use Bayesian optimization to fine-tune within those regions.

## Experiment Tracking and Management

Effective experimentation requires more than just running models with different hyperparameters—you need to systematically track results, compare experiments, and draw insights from your trials. In this section, we'll explore how to set up robust experiment tracking with Candle.

### What to Track

When running neural network experiments, you should track:

#### 1. Hyperparameters

Record all hyperparameters for each experiment, including:
- Model architecture details (layer sizes, activation functions)
- Optimization parameters (learning rate, batch size, optimizer)
- Regularization settings (weight decay, dropout rates)
- Training parameters (number of epochs, early stopping criteria)
- Data preprocessing steps
- Random seeds

#### 2. Performance Metrics

Track relevant metrics throughout training:
- Training loss
- Validation loss
- Accuracy or other task-specific metrics
- Inference time
- Memory usage

#### 3. Training Dynamics

Capture how the model evolves during training:
- Learning curves (loss and metrics over time)
- Gradient norms
- Weight distributions
- Activation patterns

#### 4. Environment Information

Document the environment for reproducibility:
- Hardware specifications (CPU/GPU)
- Software versions (Rust, Candle, dependencies)
- Dataset version

### Using TensorBoard with Candle

TensorBoard is a visualization toolkit originally developed for TensorFlow but now widely used across different frameworks. While Candle doesn't have built-in TensorBoard support, we can create a simple integration.

First, we'll need to add the `tensorboard-rs` crate to our project:

```
[dependencies]
tensorboard-rs = "0.5.0"
```

Then, we can create a simple TensorBoard writer:

```
use std::path::Path;
use tensorboard_rs::summary_writer::SummaryWriter;
use tensorboard_rs::summary_item::{SummaryItem, SummaryValue};

struct TensorBoardLogger {
    writer: SummaryWriter,
    step: i64,
}

impl TensorBoardLogger {
    fn new(log_dir: &str) -> Self {
        let writer = SummaryWriter::new(Path::new(log_dir));
        Self { writer, step: 0 }
    }
    
    fn log_scalar(&mut self, tag: &str, value: f32) {
        let item = SummaryItem {
            tag: tag.to_string(),
            value: SummaryValue::Scalar(value),
            step: self.step,
        };
        self.writer.write_summary(item).unwrap();
    }
    
    fn log_scalars(&mut self, metrics: &std::collections::HashMap<String, f32>) {
        for (tag, value) in metrics {
            self.log_scalar(tag, *value);
        }
    }
    
    fn increment_step(&mut self) {
        self.step += 1;
    }
}
```

Using this logger in our training loop:

```
// Initialize the logger
let mut tb_logger = TensorBoardLogger::new("runs/experiment_1");

// Training loop
for epoch in 0..num_epochs {
    // Train for one epoch
    let train_loss = train_epoch(&model, &train_data, &optimizer)?;
    
    // Evaluate on validation set
    let (val_loss, val_accuracy) = evaluate(&model, &val_data)?;
    
    // Log metrics to TensorBoard
    let mut metrics = std::collections::HashMap::new();
    metrics.insert("train/loss".to_string(), train_loss);
    metrics.insert("val/loss".to_string(), val_loss);
    metrics.insert("val/accuracy".to_string(), val_accuracy);
    tb_logger.log_scalars(&metrics);
    
    // Increment step
    tb_logger.increment_step();
    
    println!("Epoch {}: Train Loss = {:.4}, Val Loss = {:.4}, Val Acc = {:.2}%", 
             epoch, train_loss, val_loss, val_accuracy * 100.0);
}
```

To view the TensorBoard logs, you'll need to install TensorBoard (typically via pip) and run:

```
tensorboard --logdir=runs
```

This will start a web server (usually at http://localhost:6006) where you can view your experiment results.

### Custom Tracking Solutions

While TensorBoard is powerful, you might want a more tailored solution for tracking experiments. Here's a simple experiment tracker that saves results to JSON files:

```
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};
use serde_json;

#[derive(Serialize, Deserialize, Clone)]
struct ExperimentConfig {
    learning_rate: f64,
    batch_size: usize,
    hidden_dims: Vec<usize>,
    activation: String,
    optimizer: String,
    weight_decay: f64,
    dropout_rate: f64,
    // Add other hyperparameters as needed
}

#[derive(Serialize, Deserialize)]
struct ExperimentResult {
    config: ExperimentConfig,
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
    val_accuracies: Vec<f32>,
    best_val_accuracy: f32,
    best_epoch: usize,
    training_time: f64,
    // Add other metrics as needed
}

struct ExperimentTracker {
    results: Vec<ExperimentResult>,
    output_dir: String,
}

impl ExperimentTracker {
    fn new(output_dir: &str) -> Self {
        std::fs::create_dir_all(output_dir).unwrap();
        Self {
            results: Vec::new(),
            output_dir: output_dir.to_string(),
        }
    }
    
    fn add_result(&mut self, result: ExperimentResult) {
        self.results.push(result);
        
        // Save individual experiment result
        let filename = format!("{}/experiment_{}.json", 
                              self.output_dir, 
                              self.results.len());
        
        let file = File::create(&filename).unwrap();
        serde_json::to_writer_pretty(file, &result).unwrap();
        
        // Update summary file with best results
        self.save_summary();
    }
    
    fn save_summary(&self) {
        // Sort results by best validation accuracy
        let mut sorted_results = self.results.clone();
        sorted_results.sort_by(|a, b| b.best_val_accuracy.partial_cmp(&a.best_val_accuracy).unwrap());
        
        // Create summary with top 5 results
        let top_results: Vec<_> = sorted_results.iter()
            .take(5)
            .map(|r| {
                (
                    r.config.clone(),
                    r.best_val_accuracy,
                    r.best_epoch,
                    r.training_time
                )
            })
            .collect();
        
        // Save summary
        let summary_file = File::create(format!("{}/summary.json", self.output_dir)).unwrap();
        serde_json::to_writer_pretty(summary_file, &top_results).unwrap();
    }
}
```

Using this tracker in our experimentation:

```
// Initialize the tracker
let mut tracker = ExperimentTracker::new("experiments/mlp_mnist");

// Run experiment with a specific configuration
let config = ExperimentConfig {
    learning_rate: 0.001,
    batch_size: 64,
    hidden_dims: vec![128, 64],
    activation: "relu".to_string(),
    optimizer: "adam".to_string(),
    weight_decay: 0.0001,
    dropout_rate: 0.2,
};

// Train the model and collect metrics
let (train_losses, val_losses, val_accuracies, training_time) = 
    train_model_with_config(&config)?;

// Find the best epoch
let (best_epoch, best_val_accuracy) = val_accuracies.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap();

// Record the result
let result = ExperimentResult {
    config,
    train_losses,
    val_losses,
    val_accuracies,
    best_val_accuracy: *best_val_accuracy,
    best_epoch,
    training_time,
};

tracker.add_result(result);
```

### Visualizing Experiment Results

Once you've collected experiment results, you'll want to visualize them to gain insights. While TensorBoard provides built-in visualizations, you might want to create custom visualizations using libraries like Plotters:

```
use plotters::prelude::*;

fn plot_experiment_comparison(
    experiment_results: &[ExperimentResult],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a drawing area
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Prepare data
    let max_epochs = experiment_results.iter()
        .map(|r| r.val_accuracies.len())
        .max()
        .unwrap_or(0);
    
    let max_accuracy = experiment_results.iter()
        .flat_map(|r| r.val_accuracies.iter())
        .fold(0.0, |max, &acc| if acc > max { acc } else { max });
    
    // Create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Validation Accuracy Comparison", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_epochs, 0.0..max_accuracy * 1.1)?;
    
    chart.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Validation Accuracy")
        .draw()?;
    
    // Plot each experiment
    let colors = [&RED, &BLUE, &GREEN, &CYAN, &MAGENTA];
    
    for (i, result) in experiment_results.iter().enumerate() {
        let color = colors[i % colors.len()];
        
        let line_series = LineSeries::new(
            (0..result.val_accuracies.len()).map(|j| (j, result.val_accuracies[j])),
            color.clone(),
        );
        
        chart.draw_series(line_series)?
            .label(format!("Exp {}: lr={}, bs={}", 
                          i+1, 
                          result.config.learning_rate, 
                          result.config.batch_size))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
    }
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    Ok(())
}
```

### Best Practices for Experiment Management

1. **Unique identifiers**: Assign a unique ID to each experiment
2. **Version control**: Track code changes alongside experiment results
3. **Reproducibility**: Save random seeds and environment details
4. **Automation**: Automate the experiment pipeline as much as possible
5. **Documentation**: Record your hypotheses and observations
6. **Comparison**: Enable meaningful comparisons between different models or experiments
7. **Archiving**: Establish a system for archiving and retrieving past experiments

By implementing a robust experiment tracking system, you'll be able to:
- Quickly identify the best-performing models
- Understand the impact of different hyperparameters
- Avoid repeating unsuccessful experiments
- Share results with collaborators
- Build on past successes

## Implementing Experiments with Candle

Now that we understand hyperparameter optimization methods and experiment tracking, let's put everything together to implement a complete experimentation system with Candle. We'll focus on creating a flexible framework that allows us to easily experiment with different model architectures and hyperparameters.

### Setting Up a Configurable Model

The first step is to create models that can be easily configured with different hyperparameters. Let's implement a configurable MLP model:

```
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MLPConfig {
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    activation: String,
    dropout_rate: f64,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,  // Default for MNIST
            hidden_dims: vec![128, 64],
            output_dim: 10,  // Default for MNIST
            activation: "relu".to_string(),
            dropout_rate: 0.2,
        }
    }
}

struct MLP {
    layers: Vec<candle_nn::Linear>,
    activation: String,
    dropout_rate: f64,
}

impl MLP {
    fn new(config: &MLPConfig, vs: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut dims = vec![config.input_dim];
        dims.extend(&config.hidden_dims);
        dims.push(config.output_dim);
        
        for i in 0..dims.len()-1 {
            layers.push(candle_nn::linear(dims[i], dims[i+1], vs.pp(&format!("layer{}", i)))?);
        }
        
        Ok(Self {
            layers,
            activation: config.activation.clone(),
            dropout_rate: config.dropout_rate,
        })
    }
    
    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "relu" => x.relu(),
            "tanh" => x.tanh(),
            "sigmoid" => x.sigmoid(),
            "gelu" => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let x3 = x.powf(3.0)?;
                let inner = (x + &x3 * 0.044715)? * (2.0f64 / std::f64::consts::PI).sqrt();
                let tanh_inner = inner.tanh()?;
                (x * 0.5)? * (Tensor::ones_like(&tanh_inner)? + tanh_inner)?
            },
            _ => x.relu(), // Default to ReLU
        }
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        let training = true; // Set to false for inference
        
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            
            // Apply activation and dropout to all but the last layer
            if i < self.layers.len() - 1 {
                x = self.apply_activation(&x)?;
                if training && self.dropout_rate > 0.0 {
                    x = x.dropout(self.dropout_rate, training)?;
                }
            }
        }
        
        Ok(x)
    }
}
```

Similarly, we can create a configurable small transformer model:

```
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TransformerConfig {
    vocab_size: usize,
    max_seq_len: usize,
    embedding_dim: usize,
    num_heads: usize,
    num_layers: usize,
    feedforward_dim: usize,
    dropout_rate: f64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            max_seq_len: 512,
            embedding_dim: 256,
            num_heads: 4,
            num_layers: 2,
            feedforward_dim: 512,
            dropout_rate: 0.1,
        }
    }
}

// Transformer implementation would go here
// For brevity, we'll focus on the MLP example in this chapter
```

### Creating a Hyperparameter Configuration System

Next, we need a system to manage hyperparameter configurations for our experiments. This includes both model hyperparameters and training hyperparameters:

```
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainingConfig {
    learning_rate: f64,
    batch_size: usize,
    num_epochs: usize,
    optimizer: String,
    weight_decay: f64,
    lr_scheduler: Option<String>,
    early_stopping_patience: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 64,
            num_epochs: 10,
            optimizer: "adam".to_string(),
            weight_decay: 0.0001,
            lr_scheduler: None,
            early_stopping_patience: Some(5),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ExperimentConfig {
    model_type: String,
    model_config: serde_json::Value,
    training_config: TrainingConfig,
    random_seed: u64,
}

impl ExperimentConfig {
    fn new_mlp(mlp_config: MLPConfig, training_config: TrainingConfig) -> Self {
        Self {
            model_type: "mlp".to_string(),
            model_config: serde_json::to_value(mlp_config).unwrap(),
            training_config,
            random_seed: 42,
        }
    }
    
    fn new_transformer(transformer_config: TransformerConfig, training_config: TrainingConfig) -> Self {
        Self {
            model_type: "transformer".to_string(),
            model_config: serde_json::to_value(transformer_config).unwrap(),
            training_config,
            random_seed: 42,
        }
    }
    
    fn create_model(&self, vs: VarBuilder) -> Result<Box<dyn Module>> {
        match self.model_type.as_str() {
            "mlp" => {
                let config: MLPConfig = serde_json::from_value(self.model_config.clone()).unwrap();
                let model = MLP::new(&config, vs)?;
                Ok(Box::new(model))
            },
            "transformer" => {
                // Create transformer model
                // For brevity, we'll focus on the MLP example
                unimplemented!("Transformer implementation not shown in this example")
            },
            _ => Err(candle_core::Error::Msg(format!("Unknown model type: {}", self.model_type))),
        }
    }
    
    fn create_optimizer(&self, varmap: &VarMap) -> Result<Box<dyn candle_nn::Optimizer>> {
        let lr = self.training_config.learning_rate;
        let wd = self.training_config.weight_decay;
        
        match self.training_config.optimizer.as_str() {
            "sgd" => {
                let opt = candle_nn::SGD::new(varmap.all_vars(), lr)?;
                Ok(Box::new(opt))
            },
            "adam" => {
                let opt = candle_nn::AdamW::new_lr(varmap.all_vars(), lr)?
                    .with_weight_decay(wd);
                Ok(Box::new(opt))
            },
            _ => Err(candle_core::Error::Msg(format!("Unknown optimizer: {}", self.training_config.optimizer))),
        }
    }
}
```

### Implementing Search Methods

Now, let's implement the hyperparameter search methods we discussed earlier:

```
enum SearchMethod {
    Grid(Vec<ExperimentConfig>),
    Random {
        base_config: ExperimentConfig,
        num_trials: usize,
        param_distributions: HashMap<String, ParamDistribution>,
    },
    Bayesian {
        base_config: ExperimentConfig,
        num_trials: usize,
        param_space: HashMap<String, ParamSpace>,
    },
}

enum ParamDistribution {
    Uniform(f64, f64),
    LogUniform(f64, f64),
    Categorical(Vec<String>),
    Integer(i64, i64),
}

enum ParamSpace {
    Continuous(f64, f64),
    Discrete(Vec<f64>),
    Categorical(Vec<String>),
}

struct HyperparameterSearch {
    method: SearchMethod,
    results: Vec<(ExperimentConfig, f64)>,
    best_config: Option<ExperimentConfig>,
    best_performance: f64,
}

impl HyperparameterSearch {
    fn new(method: SearchMethod) -> Self {
        Self {
            method,
            results: Vec::new(),
            best_config: None,
            best_performance: f64::NEG_INFINITY,
        }
    }
    
    fn run(&mut self) -> Result<ExperimentConfig> {
        match &self.method {
            SearchMethod::Grid(configs) => self.run_grid_search(configs),
            SearchMethod::Random { base_config, num_trials, param_distributions } => {
                self.run_random_search(base_config, *num_trials, param_distributions)
            },
            SearchMethod::Bayesian { base_config, num_trials, param_space } => {
                self.run_bayesian_search(base_config, *num_trials, param_space)
            },
        }
    }
    
    fn run_grid_search(&mut self, configs: &[ExperimentConfig]) -> Result<ExperimentConfig> {
        for config in configs {
            let performance = self.evaluate_config(config)?;
            self.update_results(config.clone(), performance);
        }
        
        Ok(self.best_config.clone().unwrap())
    }
    
    fn run_random_search(
        &mut self,
        base_config: &ExperimentConfig,
        num_trials: usize,
        param_distributions: &HashMap<String, ParamDistribution>,
    ) -> Result<ExperimentConfig> {
        let mut rng = rand::thread_rng();
        
        for _ in 0..num_trials {
            let config = self.sample_config(base_config, param_distributions, &mut rng)?;
            let performance = self.evaluate_config(&config)?;
            self.update_results(config, performance);
        }
        
        Ok(self.best_config.clone().unwrap())
    }
    
    fn run_bayesian_search(
        &mut self,
        base_config: &ExperimentConfig,
        num_trials: usize,
        param_space: &HashMap<String, ParamSpace>,
    ) -> Result<ExperimentConfig> {
        // Simplified Bayesian optimization implementation
        // In practice, you would use a library like `bbo` or implement a proper surrogate model
        
        // Start with a few random evaluations
        let mut rng = rand::thread_rng();
        let initial_points = 5.min(num_trials);
        
        for _ in 0..initial_points {
            let config = self.sample_config_from_space(base_config, param_space, &mut rng)?;
            let performance = self.evaluate_config(&config)?;
            self.update_results(config, performance);
        }
        
        // For remaining trials, use a simple acquisition strategy
        // (In practice, you would use Expected Improvement or UCB)
        for _ in initial_points..num_trials {
            // In a real implementation, this would use the surrogate model
            // to suggest the next point to evaluate
            let config = self.suggest_next_config(base_config, param_space)?;
            let performance = self.evaluate_config(&config)?;
            self.update_results(config, performance);
        }
        
        Ok(self.best_config.clone().unwrap())
    }
    
    fn evaluate_config(&self, config: &ExperimentConfig) -> Result<f64> {
        // In a real implementation, this would train and evaluate the model
        // For this example, we'll just return a dummy value
        println!("Evaluating config: {:?}", config);
        
        // Create model and optimizer
        let device = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let _model = config.create_model(vs)?;
        let _optimizer = config.create_optimizer(&varmap)?;
        
        // Train and evaluate model
        // ...
        
        // Return validation accuracy or other metric
        Ok(0.95) // Dummy value
    }
    
    fn update_results(&mut self, config: ExperimentConfig, performance: f64) {
        self.results.push((config.clone(), performance));
        
        if performance > self.best_performance {
            self.best_performance = performance;
            self.best_config = Some(config);
        }
    }
    
    // Helper methods for sampling configurations
    fn sample_config(
        &self,
        base_config: &ExperimentConfig,
        param_distributions: &HashMap<String, ParamDistribution>,
        rng: &mut impl rand::Rng,
    ) -> Result<ExperimentConfig> {
        // Implementation would sample from distributions
        // For brevity, we'll return the base config
        Ok(base_config.clone())
    }
    
    fn sample_config_from_space(
        &self,
        base_config: &ExperimentConfig,
        param_space: &HashMap<String, ParamSpace>,
        rng: &mut impl rand::Rng,
    ) -> Result<ExperimentConfig> {
        // Implementation would sample from parameter space
        // For brevity, we'll return the base config
        Ok(base_config.clone())
    }
    
    fn suggest_next_config(
        &self,
        base_config: &ExperimentConfig,
        param_space: &HashMap<String, ParamSpace>,
    ) -> Result<ExperimentConfig> {
        // Implementation would use surrogate model to suggest next config
        // For brevity, we'll return the base config
        Ok(base_config.clone())
    }
}
```

### Parallel Experiment Execution

To speed up experimentation, we can run multiple experiments in parallel using Rust's concurrency features:

```
use std::sync::{Arc, Mutex};
use std::thread;

fn run_parallel_experiments(configs: Vec<ExperimentConfig>, num_workers: usize) -> Result<Vec<(ExperimentConfig, f64)>> {
    let configs = Arc::new(Mutex::new(configs));
    let results = Arc::new(Mutex::new(Vec::new()));
    
    let mut handles = vec![];
    
    for worker_id in 0..num_workers {
        let configs = Arc::clone(&configs);
        let results = Arc::clone(&results);
        
        let handle = thread::spawn(move || {
            println!("Worker {} started", worker_id);
            
            loop {
                // Get next config to evaluate
                let config = {
                    let mut configs = configs.lock().unwrap();
                    if configs.is_empty() {
                        break;
                    }
                    configs.pop().unwrap()
                };
                
                // Evaluate config
                println!("Worker {} evaluating config", worker_id);
                let performance = evaluate_config(&config).unwrap();
                
                // Store result
                let mut results = results.lock().unwrap();
                results.push((config, performance));
            }
            
            println!("Worker {} finished", worker_id);
        });
        
        handles.push(handle);
    }
    
    // Wait for all workers to finish
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Return results
    let results = Arc::try_unwrap(results)
        .unwrap()
        .into_inner()
        .unwrap();
    
    Ok(results)
}

fn evaluate_config(config: &ExperimentConfig) -> Result<f64> {
    // Same as the evaluate_config method in HyperparameterSearch
    // ...
    
    Ok(0.95) // Dummy value
}
```

### Putting It All Together

Now, let's put everything together to create a complete experimentation system:

```
fn main() -> Result<()> {
    // Define base configurations
    let mlp_config = MLPConfig {
        input_dim: 784,
        hidden_dims: vec![128, 64],
        output_dim: 10,
        activation: "relu".to_string(),
        dropout_rate: 0.2,
    };
    
    let training_config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 64,
        num_epochs: 10,
        optimizer: "adam".to_string(),
        weight_decay: 0.0001,
        lr_scheduler: None,
        early_stopping_patience: Some(5),
    };
    
    // Create experiment configuration
    let base_config = ExperimentConfig::new_mlp(mlp_config, training_config);
    
    // Define hyperparameter search space
    let mut param_distributions = HashMap::new();
    param_distributions.insert("training_config.learning_rate".to_string(), 
                              ParamDistribution::LogUniform(0.0001, 0.01));
    param_distributions.insert("training_config.batch_size".to_string(),
                              ParamDistribution::Categorical(vec!["32".to_string(), "64".to_string(), "128".to_string()]));
    param_distributions.insert("model_config.hidden_dims".to_string(),
                              ParamDistribution::Categorical(vec!["[64, 32]".to_string(), "[128, 64]".to_string(), "[256, 128]".to_string()]));
    
    // Create and run hyperparameter search
    let search_method = SearchMethod::Random {
        base_config,
        num_trials: 20,
        param_distributions,
    };
    
    let mut search = HyperparameterSearch::new(search_method);
    let best_config = search.run()?;
    
    println!("Best configuration: {:?}", best_config);
    println!("Best performance: {}", search.best_performance);
    
    // Save results
    let mut tracker = ExperimentTracker::new("experiments/mlp_mnist");
    for (config, performance) in search.results {
        // In a real implementation, you would have more metrics
        let result = ExperimentResult {
            config: config.clone(),
            train_losses: vec![],
            val_losses: vec![],
            val_accuracies: vec![performance],
            best_val_accuracy: performance,
            best_epoch: 0,
            training_time: 0.0,
        };
        
        tracker.add_result(result);
    }
    
    Ok(())
}
```

This implementation provides a flexible framework for experimenting with neural networks using Candle. You can easily extend it to support more model architectures, hyperparameter types, and search methods.

## Case Study: Optimizing a Simple MLP

Let's walk through a practical example of optimizing a simple MLP for a classification task. We'll use the MNIST dataset for this example, as it's a well-understood benchmark that allows us to focus on the experimentation process rather than the dataset specifics.

### Problem Setup

Our goal is to find the best MLP configuration for classifying handwritten digits in the MNIST dataset. We'll start with a basic MLP and optimize its hyperparameters to improve performance.

```
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, Module, VarBuilder, VarMap};
use candle_datasets::vision::mnist;

// Load the MNIST dataset
fn load_mnist_data(device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let m = mnist::load()?;
    
    // Normalize pixel values to [0, 1]
    let train_images = m.train_images.to_dtype(DType::F32)? / 255.0;
    let train_labels = m.train_labels;
    let test_images = m.test_images.to_dtype(DType::F32)? / 255.0;
    let test_labels = m.test_labels;
    
    // Move data to device
    let train_images = train_images.to_device(device)?;
    let train_labels = train_labels.to_device(device)?;
    let test_images = test_images.to_device(device)?;
    let test_labels = test_labels.to_device(device)?;
    
    Ok((train_images, train_labels, test_images, test_labels))
}

// Train and evaluate a model with a specific configuration
fn train_and_evaluate(config: &ExperimentConfig, device: &Device) -> Result<ExperimentResult> {
    // Load data
    let (train_images, train_labels, test_images, test_labels) = load_mnist_data(device)?;
    
    // Create model
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = config.create_model(vs.clone())?;
    
    // Create optimizer
    let mut optimizer = config.create_optimizer(&varmap)?;
    
    // Training parameters
    let batch_size = config.training_config.batch_size;
    let num_epochs = config.training_config.num_epochs;
    let num_samples = train_images.dim(0)?;
    let num_batches = num_samples / batch_size;
    
    // Early stopping
    let patience = config.training_config.early_stopping_patience.unwrap_or(0);
    let mut best_val_accuracy = 0.0;
    let mut patience_counter = 0;
    
    // Metrics tracking
    let mut train_losses = Vec::with_capacity(num_epochs);
    let mut val_accuracies = Vec::with_capacity(num_epochs);
    
    // Start timer
    let start_time = std::time::Instant::now();
    
    // Training loop
    for epoch in 0..num_epochs {
        // Training phase
        let mut sum_loss = 0.0;
        
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let batch_images = train_images.narrow(0, start_idx, batch_size)?;
            let batch_labels = train_labels.narrow(0, start_idx, batch_size)?;
            
            // Reshape images to [batch_size, 784]
            let batch_images = batch_images.reshape((batch_size, 784))?;
            
            // Forward pass
            let logits = model.forward(&batch_images)?;
            
            // Compute loss
            let loss = loss::cross_entropy(&logits, &batch_labels)?;
            
            // Backward pass and optimization
            optimizer.backward_step(&loss)?;
            
            sum_loss += loss.to_scalar::<f32>()?;
        }
        
        let avg_train_loss = sum_loss / num_batches as f32;
        train_losses.push(avg_train_loss);
        
        // Evaluation phase
        let val_accuracy = evaluate(&model, &test_images, &test_labels, batch_size)?;
        val_accuracies.push(val_accuracy);
        
        println!("Epoch {}: Train Loss = {:.4}, Val Accuracy = {:.2}%", 
                 epoch, avg_train_loss, val_accuracy * 100.0);
        
        // Early stopping check
        if val_accuracy > best_val_accuracy {
            best_val_accuracy = val_accuracy;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience > 0 && patience_counter >= patience {
                println!("Early stopping at epoch {}", epoch);
                break;
            }
        }
    }
    
    // Calculate training time
    let training_time = start_time.elapsed().as_secs_f64();
    
    // Find best epoch
    let (best_epoch, _) = val_accuracies.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap_or((0, &0.0));
    
    // Create result
    let result = ExperimentResult {
        config: config.clone(),
        train_losses,
        val_losses: vec![], // We didn't compute validation loss
        val_accuracies,
        best_val_accuracy,
        best_epoch,
        training_time,
    };
    
    Ok(result)
}

// Evaluate model on test set
fn evaluate(model: &Box<dyn Module>, test_images: &Tensor, test_labels: &Tensor, batch_size: usize) -> Result<f32> {
    let num_samples = test_images.dim(0)?;
    let num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division
    
    let mut correct = 0;
    let mut total = 0;
    
    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let effective_batch_size = std::cmp::min(batch_size, num_samples - start_idx);
        
        let batch_images = test_images.narrow(0, start_idx, effective_batch_size)?;
        let batch_labels = test_labels.narrow(0, start_idx, effective_batch_size)?;
        
        // Reshape images to [batch_size, 784]
        let batch_images = batch_images.reshape((effective_batch_size, 784))?;
        
        // Forward pass
        let logits = model.forward(&batch_images)?;
        
        // Get predictions
        let predictions = logits.argmax(1)?;
        
        // Convert labels to same dtype as predictions for comparison
        let batch_labels = batch_labels.to_dtype(DType::U32)?;
        
        // Count correct predictions
        let correct_batch = predictions.eq(&batch_labels)?.sum_all()?.to_scalar::<u32>()?;
        
        correct += correct_batch as usize;
        total += effective_batch_size;
    }
    
    Ok(correct as f32 / total as f32)
}
```

### Experiment Design

We'll focus on optimizing the following hyperparameters:

1. **Model Architecture**:
   - Number of hidden layers
   - Size of hidden layers
   - Activation function
   - Dropout rate

2. **Training Process**:
   - Learning rate
   - Batch size
   - Optimizer choice
   - Weight decay

Let's set up our experiment:

```
fn main() -> Result<()> {
    // Set up device
    let device = Device::cuda_if_available(0)?;
    
    // Create experiment tracker
    let mut tracker = ExperimentTracker::new("experiments/mlp_mnist");
    
    // Define base MLP configuration
    let base_mlp_config = MLPConfig {
        input_dim: 784,  // 28x28 flattened
        hidden_dims: vec![128],
        output_dim: 10,  // 10 digits
        activation: "relu".to_string(),
        dropout_rate: 0.2,
    };
    
    // Define base training configuration
    let base_training_config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 64,
        num_epochs: 20,
        optimizer: "adam".to_string(),
        weight_decay: 0.0001,
        lr_scheduler: None,
        early_stopping_patience: Some(5),
    };
    
    // Create base experiment configuration
    let base_config = ExperimentConfig::new_mlp(base_mlp_config, base_training_config);
    
    // Define hyperparameter search space
    let mut param_space = HashMap::new();
    
    // Model architecture parameters
    param_space.insert("model_config.hidden_dims".to_string(), 
                      ParamSpace::Categorical(vec![
                          "[64]".to_string(),
                          "[128]".to_string(),
                          "[256]".to_string(),
                          "[128, 64]".to_string(),
                          "[256, 128]".to_string(),
                      ]));
    
    param_space.insert("model_config.activation".to_string(),
                      ParamSpace::Categorical(vec![
                          "relu".to_string(),
                          "tanh".to_string(),
                          "gelu".to_string(),
                      ]));
    
    param_space.insert("model_config.dropout_rate".to_string(),
                      ParamSpace::Continuous(0.0, 0.5));
    
    // Training parameters
    param_space.insert("training_config.learning_rate".to_string(),
                      ParamSpace::Continuous(0.0001, 0.01));
    
    param_space.insert("training_config.batch_size".to_string(),
                      ParamSpace::Categorical(vec![
                          "32".to_string(),
                          "64".to_string(),
                          "128".to_string(),
                      ]));
    
    param_space.insert("training_config.optimizer".to_string(),
                      ParamSpace::Categorical(vec![
                          "sgd".to_string(),
                          "adam".to_string(),
                      ]));
    
    param_space.insert("training_config.weight_decay".to_string(),
                      ParamSpace::Continuous(0.0, 0.001));
    
    // Create Bayesian optimization search
    let search_method = SearchMethod::Bayesian {
        base_config,
        num_trials: 20,
        param_space,
    };
    
    let mut search = HyperparameterSearch::new(search_method);
    let best_config = search.run()?;
    
    println!("Best configuration found:");
    println!("{:#?}", best_config);
    println!("Best validation accuracy: {:.2}%", search.best_performance * 100.0);
    
    // Save all results
    for (config, performance) in &search.results {
        let result = train_and_evaluate(config, &device)?;
        tracker.add_result(result);
    }
    
    // Visualize results
    let top_5_results: Vec<_> = search.results.iter()
        .sorted_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap())
        .take(5)
        .collect();
    
    println!("\nTop 5 configurations:");
    for (i, (config, performance)) in top_5_results.iter().enumerate() {
        println!("{}. Accuracy: {:.2}%", i+1, performance * 100.0);
        println!("   Hidden dims: {:?}", config.model_config.get("hidden_dims").unwrap());
        println!("   Activation: {}", config.model_config.get("activation").unwrap());
        println!("   Learning rate: {}", config.training_config.learning_rate);
        println!("   Batch size: {}", config.training_config.batch_size);
        println!("   Optimizer: {}", config.training_config.optimizer);
        println!();
    }
    
    Ok(())
}
```

### Results Analysis

After running our experiments, we might find results like:

```
Best configuration found:
ExperimentConfig {
    model_type: "mlp",
    model_config: {
        "input_dim": 784,
        "hidden_dims": [256, 128],
        "output_dim": 10,
        "activation": "gelu",
        "dropout_rate": 0.3
    },
    training_config: TrainingConfig {
        learning_rate: 0.00342,
        batch_size: 64,
        num_epochs: 20,
        optimizer: "adam",
        weight_decay: 0.0005,
        lr_scheduler: None,
        early_stopping_patience: Some(5)
    },
    random_seed: 42
}
Best validation accuracy: 98.24%

Top 5 configurations:
1. Accuracy: 98.24%
   Hidden dims: [256, 128]
   Activation: gelu
   Learning rate: 0.00342
   Batch size: 64
   Optimizer: adam

2. Accuracy: 98.12%
   Hidden dims: [256, 128]
   Activation: relu
   Learning rate: 0.00289
   Batch size: 64
   Optimizer: adam

3. Accuracy: 97.95%
   Hidden dims: [256]
   Activation: gelu
   Learning rate: 0.00376
   Batch size: 128
   Optimizer: adam

4. Accuracy: 97.83%
   Hidden dims: [128, 64]
   Activation: relu
   Learning rate: 0.00412
   Batch size: 64
   Optimizer: adam

5. Accuracy: 97.56%
   Hidden dims: [128]
   Activation: relu
   Learning rate: 0.00298
   Batch size: 64
   Optimizer: adam
```

From these results, we can draw several insights:

1. **Model Architecture**: Deeper networks (with two hidden layers) generally performed better than single-layer networks. The [256, 128] configuration was particularly effective.

2. **Activation Function**: GELU activation slightly outperformed ReLU, especially in deeper networks.

3. **Optimizer**: Adam consistently outperformed SGD across different configurations.

4. **Batch Size**: A batch size of 64 worked best for most configurations, balancing between stability and generalization.

5. **Learning Rate**: The optimal learning rate fell in the range of 0.002-0.004, which is close to the standard 0.001 default but slightly higher.

6. **Dropout**: Moderate dropout (around 0.3) helped prevent overfitting without sacrificing too much capacity.

These insights demonstrate the value of systematic experimentation. While we could have manually tried different configurations, our automated approach efficiently explored the hyperparameter space and found combinations that might not have been obvious initially.

## Case Study: Tuning a Small Transformer

Now, let's look at a more complex example: tuning a small transformer model for a sequence prediction task. We'll implement a simplified transformer for next-token prediction on a text dataset.

### Transformer Model Implementation

First, let's define our transformer model:

```
struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: candle_nn::LayerNorm,
    feed_forward: FeedForward,
    norm2: candle_nn::LayerNorm,
    dropout: f64,
}

impl TransformerBlock {
    fn new(config: &TransformerConfig, vs: VarBuilder) -> Result<Self> {
        let attention = MultiHeadAttention::new(
            config.embedding_dim,
            config.num_heads,
            vs.pp("attention"),
        )?;
        
        let norm1 = candle_nn::layer_norm(
            config.embedding_dim,
            1e-5,
            vs.pp("norm1"),
        )?;
        
        let feed_forward = FeedForward::new(
            config.embedding_dim,
            config.feedforward_dim,
            vs.pp("feed_forward"),
        )?;
        
        let norm2 = candle_nn::layer_norm(
            config.embedding_dim,
            1e-5,
            vs.pp("norm2"),
        )?;
        
        Ok(Self {
            attention,
            norm1,
            feed_forward,
            norm2,
            dropout: config.dropout_rate,
        })
    }
    
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        // Self-attention with residual connection and normalization
        let norm_x = self.norm1.forward(x)?;
        let attn_output = self.attention.forward(&norm_x, &norm_x, &norm_x, mask)?;
        let x = (x + &attn_output)?;
        
        // Apply dropout if training
        let x = if train && self.dropout > 0.0 {
            x.dropout(self.dropout, train)?
        } else {
            x
        };
        
        // Feed-forward with residual connection and normalization
        let norm_x = self.norm2.forward(&x)?;
        let ff_output = self.feed_forward.forward(&norm_x)?;
        let x = (x + &ff_output)?;
        
        // Apply dropout if training
        if train && self.dropout > 0.0 {
            x.dropout(self.dropout, train)
        } else {
            Ok(x)
        }
    }
}

struct Transformer {
    token_embedding: candle_nn::Embedding,
    position_embedding: candle_nn::Embedding,
    transformer_blocks: Vec<TransformerBlock>,
    output_layer: candle_nn::Linear,
    config: TransformerConfig,
}

impl Transformer {
    fn new(config: &TransformerConfig, vs: VarBuilder) -> Result<Self> {
        let token_embedding = candle_nn::embedding(
            config.vocab_size,
            config.embedding_dim,
            vs.pp("token_embedding"),
        )?;
        
        let position_embedding = candle_nn::embedding(
            config.max_seq_len,
            config.embedding_dim,
            vs.pp("position_embedding"),
        )?;
        
        let mut transformer_blocks = Vec::new();
        for i in 0..config.num_layers {
            transformer_blocks.push(TransformerBlock::new(
                config,
                vs.pp(&format!("block{}", i)),
            )?);
        }
        
        let output_layer = candle_nn::linear(
            config.embedding_dim,
            config.vocab_size,
            vs.pp("output_layer"),
        )?;
        
        Ok(Self {
            token_embedding,
            position_embedding,
            transformer_blocks,
            output_layer,
            config: config.clone(),
        })
    }
}

impl Module for Transformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let batch_size = xs.dim(0)?;
        let seq_len = xs.dim(1)?;
        
        // Token embeddings
        let token_embeddings = self.token_embedding.forward(xs)?;
        
        // Position embeddings
        let positions = Tensor::arange(0, seq_len as u32, xs.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;
        
        let position_embeddings = self.position_embedding.forward(&positions)?;
        
        // Combine embeddings
        let mut x = (token_embeddings + position_embeddings)?;
        
        // Create attention mask (causal, for next-token prediction)
        let mask = Tensor::ones((seq_len, seq_len), DType::F32, xs.device())?
            .tril(0)?
            .reshape((1, 1, seq_len, seq_len))?;
        
        // Apply transformer blocks
        for block in &self.transformer_blocks {
            x = block.forward(&x, Some(&mask), true)?;
        }
        
        // Output layer
        self.output_layer.forward(&x)
    }
}
```

### Experiment Design for Transformer

For the transformer model, we'll focus on optimizing:

1. **Architecture Parameters**:
   - Embedding dimension
   - Number of attention heads
   - Number of transformer layers
   - Feedforward dimension
   - Dropout rate

2. **Training Parameters**:
   - Learning rate
   - Batch size
   - Optimizer

Let's set up our experiment:

```
fn main() -> Result<()> {
    // Set up device
    let device = Device::cuda_if_available(0)?;
    
    // Create experiment tracker
    let mut tracker = ExperimentTracker::new("experiments/transformer_text");
    
    // Define base transformer configuration
    let base_transformer_config = TransformerConfig {
        vocab_size: 10000,
        max_seq_len: 128,
        embedding_dim: 256,
        num_heads: 4,
        num_layers: 2,
        feedforward_dim: 512,
        dropout_rate: 0.1,
    };
    
    // Define base training configuration
    let base_training_config = TrainingConfig {
        learning_rate: 0.0005,
        batch_size: 32,
        num_epochs: 10,
        optimizer: "adam".to_string(),
        weight_decay: 0.0001,
        lr_scheduler: Some("cosine".to_string()),
        early_stopping_patience: Some(3),
    };
    
    // Create base experiment configuration
    let base_config = ExperimentConfig::new_transformer(
        base_transformer_config,
        base_training_config,
    );
    
    // Define hyperparameter search space
    let mut param_space = HashMap::new();
    
    // Model architecture parameters
    param_space.insert("model_config.embedding_dim".to_string(),
                      ParamSpace::Categorical(vec![
                          "128".to_string(),
                          "256".to_string(),
                          "384".to_string(),
                      ]));
    
    param_space.insert("model_config.num_heads".to_string(),
                      ParamSpace::Categorical(vec![
                          "2".to_string(),
                          "4".to_string(),
                          "8".to_string(),
                      ]));
    
    param_space.insert("model_config.num_layers".to_string(),
                      ParamSpace::Categorical(vec![
                          "2".to_string(),
                          "3".to_string(),
                          "4".to_string(),
                      ]));
    
    param_space.insert("model_config.feedforward_dim".to_string(),
                      ParamSpace::Categorical(vec![
                          "512".to_string(),
                          "768".to_string(),
                          "1024".to_string(),
                      ]));
    
    param_space.insert("model_config.dropout_rate".to_string(),
                      ParamSpace::Continuous(0.0, 0.3));
    
    // Training parameters
    param_space.insert("training_config.learning_rate".to_string(),
                      ParamSpace::Continuous(0.0001, 0.001));
    
    param_space.insert("training_config.batch_size".to_string(),
                      ParamSpace::Categorical(vec![
                          "16".to_string(),
                          "32".to_string(),
                          "64".to_string(),
                      ]));
    
    // Create Bayesian optimization search
    let search_method = SearchMethod::Bayesian {
        base_config,
        num_trials: 15,
        param_space,
    };
    
    let mut search = HyperparameterSearch::new(search_method);
    let best_config = search.run()?;
    
    println!("Best transformer configuration found:");
    println!("{:#?}", best_config);
    println!("Best validation perplexity: {:.2}", search.best_performance);
    
    // Visualize results
    // ...
    
    Ok(())
}
```

### Results Analysis for Transformer

After running our experiments, we might find results like:

```
Best transformer configuration found:
ExperimentConfig {
    model_type: "transformer",
    model_config: {
        "vocab_size": 10000,
        "max_seq_len": 128,
        "embedding_dim": 384,
        "num_heads": 4,
        "num_layers": 3,
        "feedforward_dim": 768,
        "dropout_rate": 0.15
    },
    training_config: TrainingConfig {
        learning_rate: 0.00068,
        batch_size: 32,
        num_epochs: 10,
        optimizer: "adam",
        weight_decay: 0.0001,
        lr_scheduler: Some("cosine"),
        early_stopping_patience: Some(3)
    },
    random_seed: 42
}
Best validation perplexity: 32.76
```

From these results, we can draw several insights:

1. **Embedding Dimension**: Larger embedding dimensions (384) captured more information and improved performance.

2. **Model Depth**: Three transformer layers provided a good balance between capacity and trainability.

3. **Attention Heads**: Four attention heads worked well, suggesting that for this task, having too many heads might not be beneficial.

4. **Dropout**: Moderate dropout (0.15) helped prevent overfitting.

5. **Learning Rate**: A learning rate of around 0.0007 worked best, which is slightly higher than the typical default of 0.0005 for transformers.

6. **Batch Size**: A batch size of 32 provided the best balance for this model.

These experiments demonstrate how systematic hyperparameter optimization can significantly improve model performance, even for complex architectures like transformers.

## Best Practices and Conclusion

Throughout this chapter, we've explored how to set up and run experiments with Candle to find optimal neural network configurations. Let's summarize some best practices for effective experimentation:

### Best Practices

1. **Start Simple**: Begin with a simple model and gradually increase complexity. This helps establish a baseline and makes it easier to identify which changes improve performance.

2. **Prioritize Hyperparameters**: Not all hyperparameters have equal impact. Focus first on learning rate, model architecture, and batch size, which often have the largest effects.

3. **Use Appropriate Search Methods**:
   - For few hyperparameters (2-3): Grid search
   - For moderate hyperparameters (4-10): Random search
   - For expensive evaluations: Bayesian optimization

4. **Track Everything**: Record all hyperparameters, metrics, and environmental details for reproducibility and analysis.

5. **Visualize Results**: Create plots to understand relationships between hyperparameters and performance.

6. **Use Early Stopping**: Save time by terminating unpromising experiments early.

7. **Parallelize When Possible**: Run multiple experiments in parallel to speed up the search process.

8. **Consider Resource Constraints**: Balance between model complexity and available computational resources.

9. **Validate Findings**: Verify that improvements generalize by testing on held-out data.

10. **Iterate and Refine**: Use insights from initial experiments to guide subsequent searches.

### Conclusion

Effective experimentation is a crucial skill for developing high-performing neural networks. By systematically exploring the hyperparameter space and tracking results, we can find configurations that significantly outperform default settings.

The Candle library, combined with Rust's performance and safety features, provides an excellent platform for neural network experimentation. The framework we've developed in this chapter allows for flexible, efficient, and reproducible experiments across different model architectures and tasks.

Remember that experimentation is an iterative process. Each experiment provides insights that inform the next round of experiments, gradually leading to better models and deeper understanding of the problem domain.

In the next chapter, we'll explore how to access Candle from Python, enabling interoperability between Rust and the Python machine learning ecosystem.