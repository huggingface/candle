# 3. Introduction to Neural Networks

## What Are Neural Networks?

Neural networks are computational models inspired by the structure and function of the human brain. At their core, they are systems of interconnected "neurons" that can learn patterns from data without being explicitly programmed with rules. This ability to learn from examples makes neural networks powerful tools for solving complex problems in areas such as image recognition, natural language processing, and game playing.

The fundamental idea behind neural networks is simple yet profound: by connecting many simple computational units (neurons) and adjusting the strength of these connections (weights), we can create systems that can approximate almost any function. This universal approximation capability allows neural networks to learn complex patterns and relationships in data.

### The Biological Inspiration

The artificial neurons in neural networks are loosely modeled after biological neurons in the brain. In the human brain, a neuron receives signals from other neurons through dendrites, processes these signals in the cell body, and if the combined input exceeds a certain threshold, sends an output signal through the axon to other neurons.

Similarly, an artificial neuron:
1. Receives input signals from other neurons
2. Applies weights to these inputs
3. Sums the weighted inputs
4. Passes this sum through an activation function
5. Produces an output that can be sent to other neurons

### From Single Neurons to Networks

While a single artificial neuron can only perform simple computations, the power of neural networks comes from connecting many neurons in layers. A typical neural network consists of:

1. **Input Layer**: Neurons that receive the initial data
2. **Hidden Layers**: Intermediate layers that perform computations
3. **Output Layer**: Neurons that provide the final result

The "deep" in deep learning refers to networks with multiple hidden layers, which can learn increasingly abstract representations of the data.

## Anatomy of a Neural Network Program

To understand how neural networks are implemented in practice, let's examine the structure of a typical neural network program. We'll use a simple example from the Candle library: a neural network that learns to add two numbers.

This example demonstrates all the essential components of a neural network application:

1. Creating the network architecture
2. Generating or loading input data
3. Training the network
4. Using the trained network for inference

Let's explore each of these components in detail.

## Creating a Neural Network

The first step in building a neural network application is defining the network architecture. This involves specifying the number of layers, the number of neurons in each layer, and how these neurons are connected.

### Network Architecture

For our addition example, we'll use a simple feedforward neural network with one hidden layer. This architecture is sufficient for learning the addition operation:

```rust
// Simple feedforward neural network for addition
struct AdditionNetwork {
    layer1: candle_nn::Linear,
    layer2: candle_nn::Linear,
}
```

This network has:
- An input layer with 2 neurons (one for each number to be added)
- A hidden layer with 16 neurons
- An output layer with 1 neuron (for the sum)

### Initializing the Network

When creating a neural network, we need to initialize its parameters (weights and biases). In the Candle library, this is done using a `VarBuilder`:

```rust
impl AdditionNetwork {
    fn new(_device: &Device, vb: VarBuilder) -> Result<Self> {
        let layer1 = candle_nn::linear(INPUT_SIZE, HIDDEN_SIZE, vb.pp("layer1"))?;
        let layer2 = candle_nn::linear(HIDDEN_SIZE, OUTPUT_SIZE, vb.pp("layer2"))?;
        Ok(Self { layer1, layer2 })
    }
}
```

The `VarBuilder` handles the creation and initialization of the network parameters, typically using random values drawn from specific distributions designed to help the network learn effectively.

### Forward Pass

The forward pass defines how input data flows through the network to produce an output. This is where the actual computation happens:

```rust
fn forward(&self, input: &Tensor) -> Result<Tensor> {
    let hidden = self.layer1.forward(input)?;
    let hidden = hidden.relu()?;
    let output = self.layer2.forward(&hidden)?;
    // Reshape to ensure we get a 1D tensor
    let batch_size = input.dim(0)?;
    let output = output.reshape((batch_size,))?;
    Ok(output)
}
```

In this forward pass:
1. The input is passed through the first layer
2. The ReLU activation function is applied to introduce non-linearity
3. The result is passed through the second layer
4. The output is reshaped to the expected format

The activation function (ReLU in this case) is crucial as it allows the network to learn non-linear relationships. Without activation functions, a neural network would only be capable of learning linear transformations.

## Preparing Input Data

Neural networks learn from data, so preparing appropriate training data is a critical step in the process.

### Data Generation

For our addition example, we generate random pairs of numbers and their sums:

```rust
fn generate_batch(batch_size: usize, device: &Device, rng: &mut StdRng) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::with_capacity(batch_size * INPUT_SIZE);
    let mut targets = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        // Generate two random numbers between 0 and NUM_RANGE
        let a = rng.gen::<f32>() * NUM_RANGE;
        let b = rng.gen::<f32>() * NUM_RANGE;

        // Calculate the sum
        let sum = a + b;

        // Add to inputs and targets
        inputs.push(a);
        inputs.push(b);
        targets.push(sum);
    }

    // Create tensors
    let inputs = Tensor::from_slice(&inputs, (batch_size, INPUT_SIZE), device)?;
    let targets = Tensor::from_slice(&targets, (batch_size,), device)?;

    Ok((inputs, targets))
}
```

This function:
1. Generates random number pairs
2. Calculates their sums
3. Converts the data into tensors suitable for the neural network

### Data Representation

In neural networks, data is typically represented as tensors (multi-dimensional arrays). For our addition example:
- Inputs are represented as a tensor of shape `[batch_size, 2]`, where each row contains two numbers to be added
- Targets (expected outputs) are represented as a tensor of shape `[batch_size]`, where each element is the sum of the corresponding input pair

The batch dimension allows us to process multiple examples simultaneously, which improves training efficiency.

## Training the Neural Network

Training is the process by which a neural network learns from data. It involves repeatedly showing examples to the network, comparing its predictions with the expected outputs, and adjusting the network's parameters to reduce the error.

### The Training Loop

The training loop is the heart of the learning process:

```rust
// Training loop
println!("Training the addition network...");
for epoch in tqdm(0..EPOCHS) {
    let mut epoch_loss = 0.0;
    let num_batches = 20; // Number of batches per epoch

    for _ in 0..num_batches {
        // Generate batch
        let (inputs, targets) = generate_batch(BATCH_SIZE, &device, &mut rng)?;

        // Forward pass
        let predictions = model.forward(&inputs)?;

        // Calculate loss (mean squared error)
        let loss = candle_nn::loss::mse(&predictions, &targets)?;

        // Backward pass and optimize
        optimizer.backward_step(&loss)?;

        epoch_loss += loss.to_scalar::<f32>()?;
    }

    epoch_loss /= num_batches as f32;

    if (epoch + 1) % 10 == 0 || epoch == 0 {
        println!("Epoch {}: Loss = {:.6}", epoch + 1, epoch_loss);
    }
}
```

This loop:
1. Iterates through a specified number of epochs (complete passes through the training data)
2. For each epoch, processes multiple batches of data
3. For each batch:
   - Performs a forward pass to get predictions
   - Calculates the loss (error) between predictions and targets
   - Performs a backward pass to compute gradients
   - Updates the network parameters using the optimizer
4. Tracks and reports the average loss for each epoch

### Loss Function

The loss function quantifies how far the network's predictions are from the expected outputs. For regression problems like our addition example, mean squared error (MSE) is a common choice:

```rust
let loss = candle_nn::loss::mse(&predictions, &targets)?;
```

MSE calculates the average of the squared differences between predicted and actual values:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true},i} - y_{\text{pred},i})^2
$$

### Backpropagation and Optimization

After calculating the loss, we need to update the network's parameters to reduce this loss. This is done through:

1. **Backpropagation**: Computing the gradient of the loss with respect to each parameter
2. **Optimization**: Updating the parameters using these gradients

In Candle, this is handled by the optimizer:

```rust
optimizer.backward_step(&loss)?;
```

The optimizer (AdamW in our example) uses the gradients to update the parameters in a way that minimizes the loss. Different optimizers use different strategies for this update, but all aim to find the parameter values that minimize the loss function.

## Inference: Using the Trained Network

Once the network is trained, we can use it to make predictions on new data. This process is called inference.

### Testing with Examples

To verify that our network has learned to add numbers correctly, we test it with specific examples:

```rust
// Test the model with some examples
println!("\nTesting the addition network:");

// Generate some test cases
let test_cases = [
    (3.0, 5.0),
    (2.5, 7.5),
    (1.2, 3.4),
    (8.0, 9.0),
    (0.0, 0.0),
    (NUM_RANGE, NUM_RANGE),  // Test edge case
];

for (a, b) in test_cases {
    // Create input tensor
    let input = Tensor::from_slice(&[a, b], (1, INPUT_SIZE), &device)?;

    // Get prediction
    let prediction = model.forward(&input)?;
    let predicted_sum = prediction.get(0)?.to_scalar::<f32>()?;

    // Calculate actual sum
    let actual_sum = a + b;

    // Calculate error
    let error = (predicted_sum - actual_sum).abs();

    println!("{:.1} + {:.1} = {:.4} (predicted) vs {:.1} (actual), error: {:.4}", 
             a, b, predicted_sum, actual_sum, error);
}
```

This code:
1. Defines a set of test cases
2. For each case, creates an input tensor
3. Performs a forward pass to get the prediction
4. Compares the prediction with the actual sum
5. Reports the result and the error

### Generalization

A key aspect of neural networks is their ability to generalize from the training data to new, unseen examples. To test this, we can evaluate the network on inputs outside the range it was trained on:

```rust
// Test with random numbers outside the training range
println!("\nTesting with numbers outside training range:");

let mut rng = StdRng::seed_from_u64(100);
for _ in 0..3 {
    let a = rng.gen::<f32>() * NUM_RANGE * 2.0;  // Generate numbers up to 2x the training range
    let b = rng.gen::<f32>() * NUM_RANGE * 2.0;

    // Create input tensor
    let input = Tensor::from_slice(&[a, b], (1, INPUT_SIZE), &device)?;

    // Get prediction
    let prediction = model.forward(&input)?;
    let predicted_sum = prediction.get(0)?.to_scalar::<f32>()?;

    // Calculate actual sum
    let actual_sum = a + b;

    // Calculate error
    let error = (predicted_sum - actual_sum).abs();

    println!("{:.1} + {:.1} = {:.4} (predicted) vs {:.1} (actual), error: {:.4}", 
             a, b, predicted_sum, actual_sum, error);
}
```

This tests the network's ability to extrapolate beyond its training distribution, which is an important aspect of its practical utility.

## Complete Program Structure

Let's step back and look at the complete structure of our neural network program:

1. **Imports and Setup**:
   - Import necessary libraries
   - Define constants and hyperparameters

2. **Model Definition**:
   - Define the network architecture
   - Implement the forward pass

3. **Data Preparation**:
   - Create functions to generate or load data
   - Convert data to the appropriate format

4. **Training**:
   - Initialize the model and optimizer
   - Implement the training loop
   - Track and report progress

5. **Inference**:
   - Use the trained model to make predictions
   - Evaluate the model's performance

This structure is common to most neural network applications, though the specific implementation details will vary depending on the task and the complexity of the model.

## From Simple Addition to Complex Tasks

While our example of learning to add two numbers is deliberately simple, the same fundamental principles apply to much more complex neural network applications:

1. **Image Classification**: Networks like CNNs (Convolutional Neural Networks) that can identify objects in images
2. **Natural Language Processing**: Models like Transformers that can understand and generate human language
3. **Reinforcement Learning**: Systems that can learn to play games or control robots through trial and error
4. **Generative Models**: Networks like GANs (Generative Adversarial Networks) that can create new images, music, or text

The key differences lie in:
- The architecture of the network (more layers, specialized layer types)
- The amount and complexity of the training data
- The specific loss functions and optimization strategies
- The computational resources required

But at their core, all these applications follow the same pattern: define a network, prepare data, train the network, and use it for inference.

## Conclusion

Neural networks represent a powerful paradigm for machine learning, allowing computers to learn complex patterns from data without explicit programming. In this chapter, we've explored the fundamental concepts of neural networks and the structure of a neural network program, using a simple addition example to illustrate these principles.

We've seen how to:
- Create a neural network with appropriate architecture
- Generate training data
- Train the network using backpropagation and optimization
- Use the trained network for inference

These foundational concepts will serve as building blocks as we explore more advanced neural network architectures and applications in the following chapters. The simple addition network may seem far removed from cutting-edge AI applications, but the core principles remain the same, scaled up to handle more complex tasks.
