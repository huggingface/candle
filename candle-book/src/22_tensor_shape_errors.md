# 23. Debugging Tensors

## Introduction to Tensor Shape Errors

Tensor shape errors are arguably the most common and frustrating runtime errors encountered in deep learning development. Unlike compilation errors that are caught before your program runs, shape mismatches occur during execution and can be particularly challenging to debug, especially in complex neural network architectures where tensors flow through multiple layers and transformations.

The fundamental issue stems from the fact that tensor operations in deep learning frameworks like Candle have strict requirements about the dimensions and shapes of their inputs. When these requirements aren't met, the program crashes with often cryptic error messages that can be difficult to interpret, especially for beginners.

Understanding tensor shapes is crucial because:

1. **Mathematical Correctness**: Operations like matrix multiplication have strict mathematical requirements
2. **Memory Layout**: Tensors must have compatible memory layouts for efficient computation
3. **Broadcasting Rules**: Element-wise operations follow specific broadcasting rules
4. **Performance**: Incorrect shapes can lead to inefficient memory usage and computation
5. **Model Architecture**: Neural network layers expect specific input and output shapes

This chapter will explore the most common types of tensor shape errors, provide debugging strategies, and offer practical solutions based on real examples from neural network implementations.

## Understanding Tensor Shapes and Dimensions

### Tensor Fundamentals

A tensor is a multi-dimensional array with a specific shape that defines its structure. In Candle, tensors have several key properties:

- **Shape**: A tuple describing the size of each dimension, e.g., `(batch_size, channels, height, width)`
- **Rank**: The number of dimensions (0D scalar, 1D vector, 2D matrix, etc.)
- **Size**: The total number of elements in the tensor
- **Data Type**: The type of data stored (f32, f64, i32, etc.)

### Common Tensor Layouts in Neural Networks

Different neural network components expect specific tensor layouts:

1. **Fully Connected Layers**: `[batch_size, features]`
2. **Convolutional Layers**: `[batch_size, channels, height, width]` (NCHW format)
3. **Recurrent Layers**: `[batch_size, sequence_length, features]` or `[sequence_length, batch_size, features]`
4. **Attention Mechanisms**: `[batch_size, num_heads, sequence_length, head_dim]`

### Shape Notation and Conventions

Throughout this chapter, we'll use the following notation:
- `N` or `batch_size`: Batch dimension
- `C` or `channels`: Channel dimension
- `H` or `height`: Height dimension
- `W` or `width`: Width dimension
- `L` or `seq_len`: Sequence length
- `D` or `features`: Feature dimension

## Types of Tensor Shape Errors

### 1. Matrix Multiplication (MatMul) Errors

Matrix multiplication is one of the most common sources of shape errors. The fundamental rule is that for matrices A and B to be multiplied (A × B), the number of columns in A must equal the number of rows in B.

#### Common MatMul Error Patterns

**Error Type**: Incompatible inner dimensions

    // This will fail: [batch_size, 128] × [64, 10]
    let input = Tensor::randn(&[32, 128], DType::F32, &device)?;
    let weight = Tensor::randn(&[64, 10], DType::F32, &device)?;
    let output = input.matmul(&weight)?; // ERROR: 128 ≠ 64

**Solution**: Ensure inner dimensions match

    // Correct: [batch_size, 128] × [128, 10] = [batch_size, 10]
    let input = Tensor::randn(&[32, 128], DType::F32, &device)?;
    let weight = Tensor::randn(&[128, 10], DType::F32, &device)?;
    let output = input.matmul(&weight)?; // SUCCESS: [32, 10]

#### Real-World Example from CNN Implementation

From the `simple_cnn.rs` file, we see careful dimension calculation:

    // Calculate the size after convolutions and pooling
    // Input: 28x28 -> Conv1: 28x28 -> Pool1: 14x14 -> Conv2: 14x14 -> Pool2: 7x7
    // So the flattened size is 64 * 8 * 8 = 4096
    let fc1 = candle_nn::linear(64 * 8 * 8, 128, vb.pp("fc1"))?;

    // In the forward pass:
    let batch_size = x.dim(0)?;
    let features = x.dim(1)? * x.dim(2)? * x.dim(3)?;
    let x = x.reshape((batch_size, features))?;

This example shows how to properly calculate the flattened dimension size to avoid MatMul errors when transitioning from convolutional to fully connected layers.

### 2. Broadcasting Errors

Broadcasting allows tensors with different but compatible shapes to be used in element-wise operations. However, broadcasting rules are strict and can lead to confusing errors.

#### Broadcasting Rules

1. Tensors are aligned from the rightmost dimension
2. Dimensions of size 1 can be broadcast to any size
3. Missing dimensions are treated as size 1
4. Incompatible dimensions (neither equal nor one of them is 1) cause errors

#### Common Broadcasting Error Patterns

**Error Type**: Incompatible dimensions for broadcasting
```rust
// This will fail: [32, 64] + [32, 128]
let a = Tensor::randn(&[32, 64], DType::F32, &device)?;
let b = Tensor::randn(&[32, 128], DType::F32, &device)?;
let result = a.add(&b)?; // ERROR: 64 ≠ 128
```

**Solution**: Reshape or use proper broadcasting
```rust
// Option 1: Make dimensions compatible
let a = Tensor::randn(&[32, 64], DType::F32, &device)?;
let b = Tensor::randn(&[32, 1], DType::F32, &device)?; // Can broadcast
let result = a.add(&b)?; // SUCCESS: broadcasts to [32, 64]

// Option 2: Use explicit broadcasting
let a = Tensor::randn(&[32, 64], DType::F32, &device)?;
let b = Tensor::randn(&[64], DType::F32, &device)?; // Can broadcast
let result = a.broadcast_add(&b)?; // SUCCESS: broadcasts to [32, 64]
```

#### Real-World Example from CNN Bias Addition

From the `simple_cnn.rs` file:

    // Add bias - reshape bias for proper broadcasting
    let bias = self.bias.reshape((1, self.bias.dim(0)?, 1, 1))?;
    let x = x.broadcast_add(&bias)?;

This shows how to properly reshape a 1D bias tensor to broadcast with a 4D feature map tensor.

### 3. Dimension Mismatch Errors

These occur when operations expect tensors with specific numbers of dimensions, but receive tensors with different ranks.

#### Common Dimension Mismatch Patterns

**Error Type**: Wrong number of dimensions
```rust
// Conv2d expects 4D input: [batch, channels, height, width]
let conv = candle_nn::conv2d(1, 32, 3, Default::default(), vb)?;
let input_2d = Tensor::randn(&[28, 28], DType::F32, &device)?; // Only 2D!
let output = conv.forward(&input_2d)?; // ERROR: Expected 4D, got 2D
```

**Solution**: Add missing dimensions
```rust
// Add batch and channel dimensions
let input_4d = input_2d.unsqueeze(0)?.unsqueeze(0)?; // Now [1, 1, 28, 28]
let output = conv.forward(&input_4d)?; // SUCCESS
```

#### Real-World Example from Mamba Implementation

From the `simple_mamba_nn.rs` file:

    fn selective_scan(&self, x: &Tensor, dt: &Tensor, b: &Tensor, c: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, dim) = x.dims3()?; // Expects exactly 3D

        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?; // [batch_size, dim]
            let b_t = b.narrow(1, t, 1)?.squeeze(1)?; // [batch_size, d_state]

            let b_expanded = b_t.unsqueeze(1)?; // [batch_size, 1, d_state]
            let x_expanded = x_t.unsqueeze(2)?; // [batch_size, dim, 1]
        }
    }

This shows careful dimension management with `squeeze` and `unsqueeze` operations to maintain proper tensor shapes throughout the computation.

### 4. Indexing and Slicing Errors

These errors occur when trying to access tensor elements or slices with invalid indices or when the resulting shapes are incompatible with subsequent operations.

#### Common Indexing Error Patterns

**Error Type**: Index out of bounds
```rust
let tensor = Tensor::randn(&[10, 20], DType::F32, &device)?;
let slice = tensor.i(15)?; // ERROR: Index 15 >= dimension size 10
```

**Error Type**: Incompatible slice shapes
```rust
let tensor = Tensor::randn(&[10, 20, 30], DType::F32, &device)?;
let slice1 = tensor.i((0, .., 0..10))?; // Shape: [20, 10]
let slice2 = tensor.i((1, .., 0..15))?; // Shape: [20, 15]
let combined = slice1.add(&slice2)?; // ERROR: [20, 10] + [20, 15]
```

**Solution**: Ensure consistent slicing
```rust
let slice1 = tensor.i((0, .., 0..10))?; // Shape: [20, 10]
let slice2 = tensor.i((1, .., 0..10))?; // Shape: [20, 10] - same size
let combined = slice1.add(&slice2)?; // SUCCESS
```

#### Real-World Example from Mamba Implementation

    // Get current timestep inputs with proper bounds checking
    let x_t = x.narrow(1, t, 1)?.squeeze(1)?; // [batch_size, dim]
    let dt_t = dt.narrow(1, t, 1)?.squeeze(1)?; // [batch_size, dt_rank]
    let b_t = b.narrow(1, t, 1)?.squeeze(1)?; // [batch_size, d_state]
    let c_t = c.narrow(1, t, 1)?.squeeze(1)?; // [batch_size, d_state]

This shows safe indexing using `narrow` with explicit bounds rather than direct indexing.

### 5. Reshaping Errors

Reshaping errors occur when trying to change a tensor's shape to an incompatible configuration.

#### Common Reshaping Error Patterns

**Error Type**: Incompatible total size
```rust
let tensor = Tensor::randn(&[10, 20], DType::F32, &device)?; // 200 elements
let reshaped = tensor.reshape(&[15, 15])?; // ERROR: 225 ≠ 200 elements
```

**Solution**: Ensure total elements match
```rust
let tensor = Tensor::randn(&[10, 20], DType::F32, &device)?; // 200 elements
let reshaped = tensor.reshape(&[8, 25])?; // SUCCESS: 200 elements
```

**Error Type**: Dynamic dimension calculation errors
```rust
// Incorrect calculation of flattened size
let conv_output = Tensor::randn(&[32, 64, 7, 7], DType::F32, &device)?;
let batch_size = conv_output.dim(0)?;
// Wrong: forgetting one dimension
let features = conv_output.dim(1)? * conv_output.dim(2)?; // Missing dim(3)
let flattened = conv_output.reshape(&[batch_size, features])?; // ERROR
```

**Solution**: Include all dimensions in calculation
```rust
let batch_size = conv_output.dim(0)?;
let features = conv_output.dim(1)? * conv_output.dim(2)? * conv_output.dim(3)?;
let flattened = conv_output.reshape(&[batch_size, features])?; // SUCCESS
```

### 6. Concatenation and Stacking Errors

These errors occur when trying to combine tensors with incompatible shapes.

#### Common Concatenation Error Patterns

**Error Type**: Incompatible dimensions for concatenation
```rust
let tensor1 = Tensor::randn(&[10, 20], DType::F32, &device)?;
let tensor2 = Tensor::randn(&[10, 25], DType::F32, &device)?;
let combined = Tensor::cat(&[&tensor1, &tensor2], 0)?; // ERROR: dim 1 mismatch
```

**Solution**: Concatenate along the correct dimension
```rust
let tensor1 = Tensor::randn(&[10, 20], DType::F32, &device)?;
let tensor2 = Tensor::randn(&[10, 25], DType::F32, &device)?;
let combined = Tensor::cat(&[&tensor1, &tensor2], 1)?; // SUCCESS: along dim 1
```

#### Real-World Example from Mamba Implementation

    // Stack outputs along sequence dimension
    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        // ... process timestep ...
        outputs.push(y_t.unsqueeze(1)?); // Ensure consistent shape
    }
    Tensor::cat(&outputs, 1) // Concatenate along sequence dimension

This shows how to ensure all tensors have compatible shapes before concatenation.

## Debugging Strategies and Tools

### 1. Shape Inspection and Logging

The most fundamental debugging technique is to inspect tensor shapes at various points in your code.

#### Basic Shape Inspection

```rust
// Print tensor shape for debugging
println!("Tensor shape: {:?}", tensor.dims());

// More detailed inspection
println!("Input shape: {:?}, dtype: {:?}", 
         input.dims(), input.dtype());

// Check specific dimensions
let (batch_size, seq_len, features) = input.dims3()?;
println!("Batch: {}, Seq: {}, Features: {}", batch_size, seq_len, features);
```

#### Systematic Shape Logging

```rust
fn debug_tensor_shape(tensor: &Tensor, name: &str) -> candle_core::Result<()> {
    println!("{}: shape={:?}, dtype={:?}, device={:?}", 
             name, tensor.dims(), tensor.dtype(), tensor.device());
    Ok(())
}

// Usage in forward pass
fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    debug_tensor_shape(x, "input")?;

    let x = self.layer1.forward(x)?;
    debug_tensor_shape(&x, "after_layer1")?;

    let x = self.layer2.forward(&x)?;
    debug_tensor_shape(&x, "after_layer2")?;

    Ok(x)
}
```

### 2. Dimension Validation Functions

Create helper functions to validate tensor shapes before operations:

```rust
fn validate_matmul_shapes(a: &Tensor, b: &Tensor) -> candle_core::Result<()> {
    let a_dims = a.dims();
    let b_dims = b.dims();

    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Err(candle_core::Error::Msg(
            format!("MatMul requires at least 2D tensors, got {:?} and {:?}", 
                   a_dims, b_dims)));
    }

    let a_cols = a_dims[a_dims.len() - 1];
    let b_rows = b_dims[b_dims.len() - 2];

    if a_cols != b_rows {
        return Err(candle_core::Error::Msg(
            format!("MatMul dimension mismatch: {} != {}", a_cols, b_rows)));
    }

    Ok(())
}

// Usage
validate_matmul_shapes(&input, &weight)?;
let output = input.matmul(&weight)?;
```

### 3. Shape-Aware Wrapper Functions

Create wrapper functions that handle common shape transformations:

```rust
fn safe_linear_forward(
    input: &Tensor, 
    weight: &Tensor, 
    bias: Option<&Tensor>
) -> candle_core::Result<Tensor> {
    // Ensure input is 2D for linear layer
    let original_shape = input.dims();
    let input_2d = if original_shape.len() > 2 {
        let batch_size = original_shape[0];
        let features: usize = original_shape[1..].iter().product();
        input.reshape(&[batch_size, features])?
    } else {
        input.clone()
    };

    // Perform linear transformation
    let output = input_2d.matmul(weight)?;
    let output = match bias {
        Some(b) => output.broadcast_add(b)?,
        None => output,
    };

    // Reshape back if needed
    if original_shape.len() > 2 {
        let mut new_shape = original_shape[..original_shape.len()-1].to_vec();
        new_shape.push(weight.dim(1)?);
        output.reshape(&new_shape)
    } else {
        Ok(output)
    }
}
```

### 4. Error Message Interpretation

Understanding common error messages can help quickly identify the issue:

#### Candle Error Patterns

- **"Dimension mismatch"**: Usually indicates incompatible tensor shapes for an operation
- **"Index out of bounds"**: Trying to access invalid tensor indices
- **"Cannot broadcast"**: Broadcasting rules violated in element-wise operations
- **"Invalid reshape"**: Total number of elements doesn't match in reshape operation

#### Creating Informative Error Messages

```rust
fn informative_matmul(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let a_shape = a.dims();
    let b_shape = b.dims();

    match a.matmul(b) {
        Ok(result) => Ok(result),
        Err(e) => Err(candle_core::Error::Msg(
            format!("MatMul failed: {} × {} - Original error: {}", 
                   format!("{:?}", a_shape), 
                   format!("{:?}", b_shape), 
                   e)))
    }
}
```

### 5. Interactive Debugging Techniques

#### Step-by-Step Shape Tracking

```rust
fn debug_forward_pass(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    println!("=== Forward Pass Debug ===");
    println!("Input: {:?}", x.dims());

    let x = self.conv1.forward(x)?;
    println!("After conv1: {:?}", x.dims());

    let x = x.relu()?;
    println!("After relu: {:?}", x.dims());

    let x = self.pool1.forward(&x)?;
    println!("After pool1: {:?}", x.dims());

    // Continue for all layers...
    Ok(x)
}
```

#### Conditional Shape Checking

```rust
fn conditional_debug<T: std::fmt::Debug>(
    tensor: &Tensor, 
    name: &str, 
    expected_shape: Option<&[usize]>
) -> candle_core::Result<()> {
    let actual_shape = tensor.dims();
    println!("{}: {:?}", name, actual_shape);

    if let Some(expected) = expected_shape {
        if actual_shape != expected {
            println!("WARNING: Expected {:?}, got {:?}", expected, actual_shape);
        }
    }
    Ok(())
}
```

## Prevention Techniques and Best Practices

### 1. Design Patterns for Shape Safety

#### Shape-Aware Layer Design

```rust
struct ShapeAwareLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    input_features: usize,
    output_features: usize,
}

impl ShapeAwareLinear {
    fn new(input_features: usize, output_features: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get((output_features, input_features), "weight")?;
        let bias = vb.get(output_features, "bias").ok();

        Ok(Self {
            weight,
            bias,
            input_features,
            output_features,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Validate input shape
        let input_dims = x.dims();
        if input_dims[input_dims.len() - 1] != self.input_features {
            return Err(candle_core::Error::Msg(
                format!("Expected {} input features, got {}", 
                       self.input_features, 
                       input_dims[input_dims.len() - 1])));
        }

        // Perform forward pass with guaranteed shape compatibility
        let output = x.matmul(&self.weight.t()?)?;
        match &self.bias {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output),
        }
    }
}
```

#### Builder Pattern for Complex Architectures

```rust
struct ModelBuilder {
    layers: Vec<Box<dyn Module>>,
    expected_shapes: Vec<Vec<usize>>,
}

impl ModelBuilder {
    fn new() -> Self {
        Self {
            layers: Vec::new(),
            expected_shapes: Vec::new(),
        }
    }

    fn add_layer<L: Module + 'static>(
        mut self, 
        layer: L, 
        expected_output_shape: Vec<usize>
    ) -> Self {
        self.layers.push(Box::new(layer));
        self.expected_shapes.push(expected_output_shape);
        self
    }

    fn build(self) -> ShapeValidatedModel {
        ShapeValidatedModel {
            layers: self.layers,
            expected_shapes: self.expected_shapes,
        }
    }
}
```

### 2. Documentation and Comments

#### Shape Documentation Standards

```rust
impl Module for TransformerBlock {
    /// Forward pass through transformer block
    /// 
    /// # Arguments
    /// * `x` - Input tensor with shape [batch_size, seq_len, d_model]
    /// 
    /// # Returns
    /// * Output tensor with shape [batch_size, seq_len, d_model]
    /// 
    /// # Shape Transformations
    /// 1. Input: [batch_size, seq_len, d_model]
    /// 2. After attention: [batch_size, seq_len, d_model]
    /// 3. After feedforward: [batch_size, seq_len, d_model]
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [batch_size, seq_len, d_model]
        let attn_output = self.attention.forward(x)?;
        // attn_output: [batch_size, seq_len, d_model]

        let x = (x + attn_output)?;
        // x: [batch_size, seq_len, d_model]

        let ff_output = self.feedforward.forward(&x)?;
        // ff_output: [batch_size, seq_len, d_model]

        Ok((x + ff_output)?)
        // output: [batch_size, seq_len, d_model]
    }
}
```

### 3. Testing Strategies

#### Shape-Focused Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_shapes() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let layer = ShapeAwareLinear::new(128, 64, vb)?;

        // Test various input shapes
        let test_cases = vec![
            vec![32, 128],        // Standard 2D input
            vec![16, 10, 128],    // 3D input (batch, seq, features)
            vec![8, 5, 3, 128],   // 4D input
        ];

        for input_shape in test_cases {
            let input = Tensor::randn(&input_shape, DType::F32, &device)?;
            let output = layer.forward(&input)?;

            // Verify output shape
            let expected_output_shape = {
                let mut shape = input_shape.clone();
                *shape.last_mut().unwrap() = 64;
                shape
            };

            assert_eq!(output.dims(), expected_output_shape.as_slice());
        }

        Ok(())
    }

    #[test]
    fn test_invalid_input_shapes() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device).unwrap();

        let layer = ShapeAwareLinear::new(128, 64, vb).unwrap();

        // Test invalid input shape
        let invalid_input = Tensor::randn(&[32, 64], DType::F32, &device).unwrap(); // Wrong feature size
        let result = layer.forward(&invalid_input);

        assert!(result.is_err());
    }
}
```

### 4. Runtime Shape Validation

#### Assertion-Based Validation

```rust
fn assert_shape(tensor: &Tensor, expected_shape: &[usize], name: &str) -> candle_core::Result<()> {
    let actual_shape = tensor.dims();
    if actual_shape != expected_shape {
        return Err(candle_core::Error::Msg(
            format!("{}: expected shape {:?}, got {:?}", 
                   name, expected_shape, actual_shape)));
    }
    Ok(())
}

// Usage in forward pass
fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    assert_shape(x, &[self.batch_size, self.seq_len, self.input_dim], "input")?;

    let hidden = self.rnn.forward(x)?;
    assert_shape(&hidden, &[self.batch_size, self.seq_len, self.hidden_dim], "hidden")?;

    let output = self.output_layer.forward(&hidden)?;
    assert_shape(&output, &[self.batch_size, self.seq_len, self.output_dim], "output")?;

    Ok(output)
}
```

## Real-World Examples and Solutions

### Example 1: CNN to RNN Transition

A common issue occurs when transitioning from convolutional layers to recurrent layers, where the tensor needs to be reshaped from 4D to 3D.

#### Problem Code

```rust
// This will fail due to shape mismatch
let conv_output = self.conv_layers.forward(x)?; // Shape: [batch, channels, height, width]
let rnn_output = self.rnn.forward(&conv_output)?; // ERROR: RNN expects 3D input
```

#### Solution

```rust
fn conv_to_rnn_transition(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    // x: [batch_size, channels, height, width]
    let conv_output = self.conv_layers.forward(x)?;

    // Reshape for RNN: flatten spatial dimensions, treat as sequence
    let (batch_size, channels, height, width) = conv_output.dims4()?;
    let seq_len = height * width;
    let features = channels;

    // Reshape to [batch_size, seq_len, features]
    let rnn_input = conv_output
        .transpose(1, 2)?  // [batch, height, channels, width]
        .transpose(2, 3)?  // [batch, height, width, channels]
        .reshape(&[batch_size, seq_len, features])?;

    let rnn_output = self.rnn.forward(&rnn_input)?;
    Ok(rnn_output)
}
```

### Example 2: Attention Mechanism Shape Management

Attention mechanisms involve complex tensor reshaping and matrix operations that are prone to shape errors.

#### Problem: Multi-Head Attention Implementation

```rust
fn multi_head_attention(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    let (batch_size, seq_len, d_model) = x.dims3()?;

    // Generate Q, K, V
    let q = self.q_proj.forward(x)?; // [batch, seq_len, d_model]
    let k = self.k_proj.forward(x)?; // [batch, seq_len, d_model]
    let v = self.v_proj.forward(x)?; // [batch, seq_len, d_model]

    // Reshape for multi-head attention
    let head_dim = d_model / self.num_heads;

    // Reshape to [batch, seq_len, num_heads, head_dim]
    let q = q.reshape(&[batch_size, seq_len, self.num_heads, head_dim])?;
    let k = k.reshape(&[batch_size, seq_len, self.num_heads, head_dim])?;
    let v = v.reshape(&[batch_size, seq_len, self.num_heads, head_dim])?;

    // Transpose to [batch, num_heads, seq_len, head_dim]
    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    let v = v.transpose(1, 2)?;

    // Compute attention scores: Q @ K^T
    let scores = q.matmul(&k.transpose(-2, -1)?)?; // [batch, num_heads, seq_len, seq_len]

    // Scale scores
    let scale = (head_dim as f64).sqrt();
    let scores = (scores / scale)?;

    // Apply softmax
    let attn_weights = scores.softmax(-1)?;

    // Apply attention to values
    let attn_output = attn_weights.matmul(&v)?; // [batch, num_heads, seq_len, head_dim]

    // Transpose back and reshape
    let attn_output = attn_output.transpose(1, 2)?; // [batch, seq_len, num_heads, head_dim]
    let attn_output = attn_output.reshape(&[batch_size, seq_len, d_model])?;

    // Final projection
    self.out_proj.forward(&attn_output)
}
```

This example shows careful shape management throughout a complex operation with multiple reshaping and transposition steps.

### Example 3: Batch Processing with Variable Sequence Lengths

Handling variable-length sequences in batches requires careful padding and masking.

#### Problem and Solution

```rust
fn process_variable_length_batch(
    &self, 
    sequences: Vec<Tensor>, 
    max_len: usize
) -> candle_core::Result<Tensor> {
    let batch_size = sequences.len();
    let feature_dim = sequences[0].dim(1)?;

    // Create padded batch tensor
    let mut batch_data = Vec::new();

    for seq in sequences {
        let seq_len = seq.dim(0)?;

        if seq_len <= max_len {
            // Pad sequence to max_len
            let padding_size = max_len - seq_len;
            let padding = Tensor::zeros(&[padding_size, feature_dim], seq.dtype(), seq.device())?;
            let padded_seq = Tensor::cat(&[&seq, &padding], 0)?;
            batch_data.push(padded_seq);
        } else {
            // Truncate sequence to max_len
            let truncated_seq = seq.narrow(0, 0, max_len)?;
            batch_data.push(truncated_seq);
        }
    }

    // Stack into batch tensor
    let batch_refs: Vec<&Tensor> = batch_data.iter().collect();
    let batch_tensor = Tensor::stack(&batch_refs, 0)?; // [batch_size, max_len, feature_dim]

    Ok(batch_tensor)
}
```

## Advanced Debugging Techniques

### 1. Shape Profiling and Monitoring

Create a shape profiler to track tensor shapes throughout your model:

```rust
struct ShapeProfiler {
    shapes: std::collections::HashMap<String, Vec<Vec<usize>>>,
}

impl ShapeProfiler {
    fn new() -> Self {
        Self {
            shapes: std::collections::HashMap::new(),
        }
    }

    fn record(&mut self, name: &str, tensor: &Tensor) {
        let shape = tensor.dims().to_vec();
        self.shapes.entry(name.to_string()).or_insert_with(Vec::new).push(shape);
    }

    fn print_summary(&self) {
        for (name, shapes) in &self.shapes {
            println!("{}: {:?}", name, shapes);
        }
    }
}

// Usage in model
fn forward_with_profiling(&self, x: &Tensor, profiler: &mut ShapeProfiler) -> candle_core::Result<Tensor> {
    profiler.record("input", x);

    let x = self.layer1.forward(x)?;
    profiler.record("after_layer1", &x);

    let x = self.layer2.forward(&x)?;
    profiler.record("after_layer2", &x);

    Ok(x)
}
```

### 2. Automatic Shape Inference

Implement automatic shape inference for complex models:

```rust
trait ShapeInference {
    fn infer_output_shape(&self, input_shape: &[usize]) -> Vec<usize>;
}

impl ShapeInference for candle_nn::Linear {
    fn infer_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let mut output_shape = input_shape.to_vec();
        *output_shape.last_mut().unwrap() = self.weight().dim(0).unwrap();
        output_shape
    }
}

fn validate_model_shapes<M: Module + ShapeInference>(
    model: &M, 
    input_shape: &[usize]
) -> candle_core::Result<Vec<usize>> {
    let predicted_output_shape = model.infer_output_shape(input_shape);

    // Create dummy input to test actual shapes
    let device = Device::Cpu;
    let dummy_input = Tensor::zeros(input_shape, DType::F32, &device)?;
    let actual_output = model.forward(&dummy_input)?;
    let actual_output_shape = actual_output.dims();

    if predicted_output_shape != actual_output_shape {
        return Err(candle_core::Error::Msg(
            format!("Shape inference mismatch: predicted {:?}, actual {:?}",
                   predicted_output_shape, actual_output_shape)));
    }

    Ok(predicted_output_shape)
}
```

## Common Pitfalls and How to Avoid Them

### 1. Forgetting Batch Dimensions

**Problem**: Implementing layers that work with single samples but fail with batches.

```rust
// This works for single samples but fails for batches
fn naive_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let scores = q.matmul(&k.t()?)?; // Assumes 2D tensors
    let weights = scores.softmax(1)?;
    weights.matmul(v)
}
```

**Solution**: Always design for batch processing.

```rust
fn batch_aware_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    // Handle both 2D and 3D tensors (with batch dimension)
    let k_t = k.transpose(-2, -1)?; // Transpose last two dimensions
    let scores = q.matmul(&k_t)?;
    let weights = scores.softmax(-1)?; // Softmax over last dimension
    weights.matmul(v)
}
```

### 2. Hardcoded Dimension Assumptions

**Problem**: Assuming specific tensor dimensions without validation.

```rust
// Dangerous: assumes input is always 4D
fn unsafe_conv_forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    let (batch, channels, height, width) = x.dims4()?; // Will panic if not 4D
    // ... rest of implementation
}
```

**Solution**: Validate dimensions or handle multiple cases.

```rust
fn safe_conv_forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    match dims.len() {
        4 => {
            let (batch, channels, height, width) = x.dims4()?;
            // Handle 4D case
        },
        3 => {
            // Add batch dimension
            let x = x.unsqueeze(0)?;
            let result = self.safe_conv_forward(&x)?;
            result.squeeze(0) // Remove batch dimension
        },
        _ => Err(candle_core::Error::Msg(
            format!("Expected 3D or 4D tensor, got {}D", dims.len())
        ))
    }
}
```

### 3. Inconsistent Tensor Layouts

**Problem**: Mixing different tensor layout conventions (NCHW vs NHWC, etc.).

**Solution**: Establish and document consistent conventions.

```rust
// Document tensor layout expectations clearly
/// Convolution layer expecting NCHW format
/// Input: [batch_size, in_channels, height, width]
/// Output: [batch_size, out_channels, new_height, new_width]
struct Conv2dNCHW {
    // implementation
}

/// Utility function to convert between layouts
fn nchw_to_nhwc(tensor: &Tensor) -> candle_core::Result<Tensor> {
    // [N, C, H, W] -> [N, H, W, C]
    tensor.permute(&[0, 2, 3, 1])
}
```

## Performance Considerations

### 1. Memory-Efficient Reshaping

Some reshaping operations can be expensive. Understanding when reshaping creates copies vs. views is important:

```rust
// Efficient: creates a view (no data copying)
let reshaped = tensor.reshape(&[new_shape])?;

// Less efficient: may require data copying
let transposed = tensor.transpose(0, 1)?;
let then_reshaped = transposed.reshape(&[new_shape])?;

// More efficient approach
let reshaped_first = tensor.reshape(&[intermediate_shape])?;
let then_transposed = reshaped_first.transpose(0, 1)?;
```

### 2. Batch Size Optimization

Choose batch sizes that work well with your tensor operations:

```rust
fn optimal_batch_size_for_matmul(
    input_features: usize, 
    output_features: usize
) -> usize {
    // Prefer batch sizes that are multiples of common SIMD widths
    let preferred_multiples = [32, 64, 128, 256];

    // Choose based on memory constraints and computational efficiency
    for &multiple in &preferred_multiples {
        if multiple * input_features * 4 < 1_000_000 { // Rough memory estimate
            return multiple;
        }
    }

    32 // Default fallback
}
```

## Conclusion

Tensor shape errors are an inevitable part of deep learning development, but with proper understanding, debugging techniques, and prevention strategies, they can be managed effectively. The key principles to remember are:

1. **Always validate tensor shapes** at critical points in your code
2. **Document expected shapes** in comments and function signatures
3. **Use systematic debugging approaches** rather than trial-and-error
4. **Design shape-aware abstractions** that handle common cases automatically
5. **Test with various input shapes** to ensure robustness
6. **Understand the mathematical requirements** of each operation

By following these guidelines and using the techniques outlined in this chapter, you can significantly reduce the time spent debugging shape errors and build more robust neural network implementations. Remember that shape errors, while frustrating, often indicate deeper architectural issues that, when resolved, lead to better and more maintainable code.

The examples and patterns shown in this chapter are based on real issues encountered in neural network development. As you gain experience, you'll develop an intuition for common shape problems and their solutions, making you a more effective deep learning practitioner.
