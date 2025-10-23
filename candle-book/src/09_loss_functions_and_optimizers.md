# 8. Loss Functions and Optimizers

## Introduction

Loss functions and optimizers are two fundamental components of the neural network training process. A loss function quantifies how well a model is performing by measuring the difference between its predictions and the actual target values. Optimizers, on the other hand, are algorithms that adjust the model's parameters to minimize this loss. Together, they form the backbone of the learning process in neural networks.

This chapter explores:
- The mathematical foundations of common loss functions
- How to implement and use loss functions in Candle
- Popular optimization algorithms and their characteristics
- Practical considerations for choosing and tuning loss functions and optimizers
- Implementation examples for various scenarios

## Loss Functions

A loss function, also called a cost function or objective function, measures how far the model's predictions are from the actual values. The goal of training is to minimize this function.

### Mean Squared Error (MSE)

Mean Squared Error is one of the most common loss functions for regression problems. It calculates the average of the squared differences between predicted and actual values:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{\mkern-3mu y}_i)^2
$$

Where:
- \\( n \\)  is the number of samples
- \\( y_i \\)  is the actual value
- \\( \hat{\mkern-3mu y}_i \\)  is the predicted value

MSE heavily penalizes large errors due to the squaring operation, making it particularly sensitive to outliers.

#### Implementation in Candle

```rust
use candle_core::{Tensor, Result};

fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Calculate the squared difference
    let diff = predictions.sub(targets)?;
    let squared_diff = diff.sqr()?;

    // Take the mean
    let loss = squared_diff.mean_all()?;

    Ok(loss)
}

// Using Candle's built-in MSE loss
fn using_candle_mse(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let loss = candle_nn::loss::mse(predictions, targets)?;
    Ok(loss)
}
```

### Mean Absolute Error (MAE)

Mean Absolute Error calculates the average of the absolute differences between predicted and actual values:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{\mkern-3mu y}_i|
$$

MAE is less sensitive to outliers compared to MSE because it doesn't square the errors.

#### Implementation in Candle

```rust
fn mae_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Calculate the absolute difference
    let diff = predictions.sub(targets)?;
    let abs_diff = diff.abs()?;

    // Take the mean
    let loss = abs_diff.mean_all()?;

    Ok(loss)
}
```

### Binary Cross-Entropy Loss

Binary Cross-Entropy is used for binary classification problems where the output is a probability between 0 and 1:

$$
\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{\mkern-3mu y}_i) + (1 - y_i) \log(1 - \hat{\mkern-3mu y}_i)]
$$

Where:
- \\( y_i \\)  is the true label (0 or 1)
- \\( \hat{\mkern-3mu y}_i \\)  is the predicted probability

#### Implementation in Candle

```rust
fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Clip predictions to avoid log(0)
    let eps = 1e-7;
    let predictions = predictions.clamp(eps, 1.0 - eps)?;

    // Calculate BCE
    let term1 = targets.mul(&predictions.log()?)?;
    let term2 = targets.neg_add(1.0)?.mul(&predictions.neg_add(1.0)?.log()?)?;
    let loss = term1.add(&term2)?.neg()?.mean_all()?;

    Ok(loss)
}

// Using Candle's built-in BCE loss
fn using_candle_bce(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let loss = candle_nn::loss::binary_cross_entropy(predictions, targets)?;
    Ok(loss)
}
```

### Categorical Cross-Entropy Loss

Categorical Cross-Entropy is used for multi-class classification problems:

$$
\text{CCE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{C} y_{ij} \log(\hat{\mkern-3mu y}_{ij})
$$

Where:
- \\( C \\)is the number of classes
- \\( y_{ij} \\)  is 1 if sample \\( i \\)  belongs to class \\( j \\)  and 0 otherwise (one-hot encoding)
- \\( \hat{\mkern-3mu y}_{ij} \\) is the predicted probability that sample \\( i \\) belongs to class \\( j \\)

#### Implementation in Candle

```rust
fn categorical_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Apply softmax to get probabilities
    let log_softmax = candle_nn::ops::log_softmax(logits, 1)?;

    // Calculate cross-entropy
    let loss = targets.mul(&log_softmax)?.neg()?.sum_all()?.div_scalar(targets.dim(0)? as f64)?;

    Ok(loss)
}

// Using Candle's built-in cross-entropy loss
fn using_candle_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let loss = candle_nn::loss::cross_entropy(logits, targets)?;
    Ok(loss)
}
```


## Optimizers

Optimizers are algorithms that adjust the model's parameters to minimize the loss function. They implement different strategies for updating parameters based on the gradients computed during backpropagation.

### Gradient Descent

Gradient Descent is the most basic optimization algorithm. It updates parameters in the direction of the negative gradient of the loss function:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)
$$

Where:
- \\( \theta_t \\)  is the parameter at step \\( t \\) (t)
- \\( \alpha \\)  is the learning rate
- \\( \nabla_\theta J(\theta_t) \\) is the gradient of the loss function with respect to the parameters

#### Variants of Gradient Descent

1. **Batch Gradient Descent**: Uses the entire dataset to compute the gradient
2. **Stochastic Gradient Descent (SGD)**: Uses a single sample to compute the gradient
3. **Mini-Batch Gradient Descent**: Uses a small batch of samples to compute the gradient (most common)

#### Implementation in Candle

```rust
use candle_nn::{Optimizer, VarMap};

fn train_with_sgd(model: &mut impl Module, x: &Tensor, y: &Tensor, learning_rate: f64) -> Result<()> {
    // Create a variable map to track model parameters
    let mut varmap = VarMap::new();
    let vars = varmap.all_vars();

    // Create SGD optimizer
    let mut optimizer = candle_nn::SGD::new(vars, learning_rate)?;

    // Forward pass
    let predictions = model.forward(x)?;

    // Compute loss
    let loss = candle_nn::loss::mse(&predictions, y)?;

    // Backward pass and optimize
    optimizer.backward_step(&loss)?;

    Ok(())
}
```

### SGD with Momentum

Momentum accelerates convergence by accumulating a velocity vector in the direction of persistent reduction in the loss:

$$
\begin{align}
v_{t+1} &= \gamma v_t + \alpha \nabla_\theta J(\theta_t) \\\\
\theta_{t+1} &= \theta_t - v_{t+1}
\end{align}
$$

Where:
- \\( v_t \\)  is the velocity at step \\( t \\) (t)
- \\( \gamma \\)  is the momentum coefficient (typically 0.9)

#### Implementation in Candle

```rust
fn train_with_sgd_momentum(model: &mut impl Module, x: &Tensor, y: &Tensor, learning_rate: f64, momentum: f64) -> Result<()> {
    let mut varmap = VarMap::new();
    let vars = varmap.all_vars();

    // Create SGD optimizer with momentum
    let mut optimizer = candle_nn::SGD::new(vars, learning_rate)?
        .with_momentum(momentum);

    // Forward pass
    let predictions = model.forward(x)?;

    // Compute loss
    let loss = candle_nn::loss::mse(&predictions, y)?;

    // Backward pass and optimize
    optimizer.backward_step(&loss)?;

    Ok(())
}
```


### Adam (Adaptive Moment Estimation)

Adam is one of the most popular and effective optimization algorithms in deep learning. It was introduced by Diederik Kingma and Jimmy Ba in 2014 and combines the best aspects of two other optimization methods: AdaGrad's adaptive learning rates and RMSProp's exponential moving averages, while also incorporating momentum.

#### Why Adam Works So Well

Adam addresses several key challenges in neural network optimization:

1. **Adaptive Learning Rates**: Different parameters may need different learning rates. Adam automatically adapts the learning rate for each parameter based on the historical gradients.

2. **Momentum**: Like SGD with momentum, Adam maintains a "velocity" that helps accelerate convergence and navigate past local minima.

3. **Bias Correction**: Adam corrects for the bias introduced by initializing the moment estimates to zero, which is particularly important in the early stages of training.

4. **Sparse Gradients**: Adam works well even when gradients are sparse, making it suitable for a wide range of problems including natural language processing.

#### The Adam Algorithm

Adam maintains two moving averages for each parameter:
- **First moment estimate (m_t)**: The exponential moving average of the gradient (momentum)
- **Second moment estimate (v_t)**: The exponential moving average of the squared gradient (uncentered variance)

The complete Adam update equations are:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta_t)$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta J(\theta_t))^2$$

$$\hat{\mkern-3mu m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{\mkern-3mu v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\alpha \hat{\mkern-3mu m}_t}{\sqrt{\hat{\mkern-3mu v}_t} + \epsilon}$$


Where:
- \\( m_t \\) and \\( v_t \\) are the first and second moment estimates
- \\( \beta_1 \\) and \\( \beta_2 \\) are decay rates (typically 0.9 and 0.999)
- \\( \hat{\mkern-3mu m}_t \\) and \\( \hat{\mkern-3mu v}_t \\) are bias-corrected moment estimates
- \\( \alpha \\) is the learning rate (typically 0.001)
- \\( \epsilon \\) is a small constant for numerical stability (typically 1e-8)

#### Understanding the Components

1. **First moment (m_t)**: This is similar to momentum in SGD, providing a "memory" of previous gradients that helps smooth out noisy updates and accelerate convergence in consistent directions.

2. **Second moment (v_t)**: This tracks the magnitude of recent gradients, allowing Adam to use smaller effective learning rates for parameters with large gradients and larger effective learning rates for parameters with small gradients.

3. **Bias correction**: The terms \\( \hat{\mkern-3mu m}_t \\) and \\( \hat{\mkern-3mu v}_t \\) correct for the fact that \\( m_t \\) and \\( v_t \\) are initialized to zero, which would otherwise bias them toward zero, especially in early training steps.

4. **Adaptive step size**: The final update divides the bias-corrected momentum by the square root of the bias-corrected second moment, creating an adaptive step size for each parameter.

#### Default Hyperparameters

The original Adam paper suggests these default values, which work well across many problems:
- \\( \alpha = 0.001 \\) (learning rate)
- \\( \beta_1 = 0.9 \\) (exponential decay rate for first moment)
- \\( \beta_2 = 0.999 \\) (exponential decay rate for second moment)
- \\( \epsilon = 10^{-8} \\) (small constant for numerical stability)

#### Implementation in Candle

```rust
fn train_with_adam(model: &mut impl Module, x: &Tensor, y: &Tensor, learning_rate: f64) -> Result<()> {
   let mut varmap = VarMap::new();
   let vars = varmap.all_vars();

   // Create Adam optimizer with default parameters
   // Note: Candle uses AdamW by default, which includes weight decay
   let mut optimizer = candle_nn::AdamW::new(vars, learning_rate)?;

   // Forward pass
   let predictions = model.forward(x)?;

   // Compute loss
   let loss = candle_nn::loss::mse(&predictions, y)?;

   // Backward pass and optimize
   optimizer.backward_step(&loss)?;

   Ok(())
}

// For more control over Adam parameters
fn train_with_custom_adam(
   model: &mut impl Module,
   x: &Tensor,
   y: &Tensor,
   learning_rate: f64,
   beta1: f64,
   beta2: f64
) -> Result<()> {
   let mut varmap = VarMap::new();
   let vars = varmap.all_vars();

   // Create Adam optimizer with custom parameters
   let mut optimizer = candle_nn::AdamW::new_lr(vars, learning_rate)?
           .with_beta1(beta1)
           .with_beta2(beta2);

   let predictions = model.forward(x)?;
   let loss = candle_nn::loss::mse(&predictions, y)?;
   optimizer.backward_step(&loss)?;

   Ok(())
}
```
### AdamW

AdamW is a variant of Adam that implements weight decay correctly by decoupling it from the gradient updates:

$$
\theta_{t+1} = \theta_t - \alpha \left( \frac{\hat{\mkern-3mu m}_t}{\sqrt{\hat{\mkern-3mu v}_t} + \epsilon} + \lambda \theta_t \right)
$$

Where \\( \lambda \\)  is the weight decay coefficient.

#### Implementation in Candle

```rust
fn train_with_adamw(model: &mut impl Module, x: &Tensor, y: &Tensor, learning_rate: f64, weight_decay: f64) -> Result<()> {
    let mut varmap = VarMap::new();
    let vars = varmap.all_vars();

    // Create AdamW optimizer with weight decay
    let mut optimizer = candle_nn::AdamW::new_lr_wd(vars, learning_rate, weight_decay)?;

    // Forward pass
    let predictions = model.forward(x)?;

    // Compute loss
    let loss = candle_nn::loss::mse(&predictions, y)?;

    // Backward pass and optimize
    optimizer.backward_step(&loss)?;

    Ok(())
}
```

## Practical Considerations

### Choosing the Right Loss Function

The choice of loss function depends on the task:

1. **Regression Tasks**:
   - MSE: Good general-purpose loss, but sensitive to outliers
   - MAE: More robust to outliers, but may converge slower
   - Huber: Combines benefits of MSE and MAE

2. **Classification Tasks**:
   - Binary Cross-Entropy: For binary classification
   - Categorical Cross-Entropy: For multi-class classification
   - Focal Loss: For imbalanced datasets

3. **Special Cases**:
   - Custom loss functions for specific requirements
   - Combined loss functions for multi-task learning

### Choosing the Right Optimizer

The choice of optimizer affects convergence speed and final performance:

1. **SGD**: Simple and works well with large datasets, but may converge slowly
2. **SGD with Momentum**: Faster convergence than plain SGD
3. **Adam/AdamW**: Adaptive learning rates, generally works well across many problems
4. **RMSProp**: Good for non-stationary objectives and RNNs

### Learning Rate Scheduling

Learning rate scheduling can improve convergence and final performance:

```rust
fn train_with_lr_scheduler(model: &mut impl Module, x: &Tensor, y: &Tensor, 
                          initial_lr: f64, epochs: usize) -> Result<()> {
    let mut varmap = VarMap::new();
    let vars = varmap.all_vars();

    for epoch in 0..epochs {
        // Decay learning rate over time
        let lr = initial_lr / (1.0 + 0.1 * epoch as f64);

        // Create optimizer with current learning rate
        let mut optimizer = candle_nn::AdamW::new_lr(vars, lr)?;

        // Forward pass
        let predictions = model.forward(x)?;

        // Compute loss
        let loss = candle_nn::loss::mse(&predictions, y)?;

        // Backward pass and optimize
        optimizer.backward_step(&loss)?;

        println!("Epoch {}: Loss = {:.6}, LR = {:.6}", epoch, loss.to_scalar::<f32>()?, lr);
    }

    Ok(())
}
```

### Gradient Clipping

Gradient clipping prevents exploding gradients, especially in recurrent networks:

```rust
fn train_with_gradient_clipping(model: &mut impl Module, x: &Tensor, y: &Tensor, 
                               learning_rate: f64, max_norm: f64) -> Result<()> {
    let mut varmap = VarMap::new();
    let vars = varmap.all_vars();

    let mut optimizer = candle_nn::AdamW::new(vars, learning_rate)?;

    // Forward pass
    let predictions = model.forward(x)?;

    // Compute loss
    let loss = candle_nn::loss::mse(&predictions, y)?;

    // Backward pass
    optimizer.backward(&loss)?;

    // Clip gradients
    optimizer.clip_grad_norm(max_norm)?;

    // Update parameters
    optimizer.step()?;

    Ok(())
}
```

## Complete Training Example

Let's put everything together in a complete training example for a simple regression problem:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap, Optimizer};

struct SimpleRegression {
    layer: Linear,
}

impl SimpleRegression {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let layer = candle_nn::linear(in_dim, out_dim, vb)?;
        Ok(Self { layer })
    }
}

impl Module for SimpleRegression {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.layer.forward(x)
    }
}

fn main() -> Result<()> {
    // Set up device
    let device = Device::cuda_if_available(0)?;

    // Generate synthetic data: y = 2x + 1 + noise
    let x_data: Vec<f32> = (0..100).map(|i| i as f32 / 10.0).collect();
    let y_data: Vec<f32> = x_data.iter()
        .map(|&x| 2.0 * x + 1.0 + (rand::random::<f32>() - 0.5) * 0.2)
        .collect();

    let x = Tensor::from_slice(&x_data, (100, 1), &device)?;
    let y = Tensor::from_slice(&y_data, (100, 1), &device)?;

    // Create model
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SimpleRegression::new(1, 1, vb)?;

    // Training parameters
    let learning_rate = 0.01;
    let epochs = 200;

    // Create optimizer
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), learning_rate)?;

    // Training loop
    for epoch in 0..epochs {
        // Forward pass
        let predictions = model.forward(&x)?;

        // Compute loss (MSE)
        let loss = candle_nn::loss::mse(&predictions, &y)?;

        // Backward pass and optimize
        optimizer.backward_step(&loss)?;

        if (epoch + 1) % 20 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch + 1, loss.to_scalar::<f32>()?);
        }
    }

    // Test the model
    let test_x = Tensor::from_slice(&[0.0f32, 5.0, 10.0], (3, 1), &device)?;
    let predictions = model.forward(&test_x)?;

    println!("\nModel predictions:");
    println!("x = 0.0, predicted y = {:.4}, expected y ≈ 1.0", 
             predictions.get(0)?.to_scalar::<f32>()?);
    println!("x = 5.0, predicted y = {:.4}, expected y ≈ 11.0", 
             predictions.get(1)?.to_scalar::<f32>()?);
    println!("x = 10.0, predicted y = {:.4}, expected y ≈ 21.0", 
             predictions.get(2)?.to_scalar::<f32>()?);

    Ok(())
}
```

## Conclusion

Loss functions and optimizers are essential components of the neural network training process. The choice of loss function depends on the specific task, while the choice of optimizer affects convergence speed and final performance.

In this chapter, we've explored:
- Common loss functions for regression and classification tasks
- Popular optimization algorithms and their characteristics
- Practical considerations for choosing and tuning loss functions and optimizers
- Implementation examples in Rust using Candle

Understanding these components allows you to make informed decisions when designing and training neural networks, leading to better performance and faster convergence.

## Further Reading

- "Deep Learning" by Goodfellow, Bengio, and Courville - Comprehensive coverage of loss functions and optimization algorithms
- "An overview of gradient descent optimization algorithms" by Sebastian Ruder - Detailed explanation of various optimizers
- "Why Momentum Really Works" by Gabriel Goh - Insights into momentum-based optimization
- "Adam: A Method for Stochastic Optimization" by Kingma and Ba - Original paper introducing the Adam optimizer
