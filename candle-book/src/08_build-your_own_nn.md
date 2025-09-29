# 7. Building a Neural Network

In this chapter, we'll build a complete neural network from scratch using the Candle framework. We'll implement a Multi-Layer Perceptron (MLP) for classifying Iris flowers, a classic machine learning task. This example will demonstrate all the essential components of neural network development, from data loading to model evaluation.

The Iris dataset contains measurements of 150 iris flowers from three different species: Setosa, Versicolor, and Virginica. Each sample has four features: sepal length, sepal width, petal length, and petal width. Our goal is to train a neural network that can correctly classify the species based on these measurements.
![iris_mlp_architecture.svg](images/iris_mlp_architecture.svg)

## 1. Imports and Setup

Let's start by importing the necessary libraries and defining our hyperparameters:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor, IndexOp};
use candle_nn::{VarBuilder, VarMap, Module, Optimizer};
use rand::{rngs::StdRng, SeedableRng, Rng};
use tqdm::tqdm;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// Define hyperparameters
const INPUT_SIZE: usize = 4;  // Iris has 4 features
const HIDDEN_SIZE: usize = 32;
const OUTPUT_SIZE: usize = 3;  // Iris has 3 classes
const BATCH_SIZE: usize = 32;
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 100;
const PRINT_EVERY: usize = 10;
```

Here's what each import does:
- `anyhow`: Provides error handling utilities
- `candle_core`: Core functionality of the Candle framework, including tensors and devices
- `candle_nn`: Neural network components like layers and optimizers
- `rand`: Random number generation for data shuffling
- `tqdm`: Progress bar for tracking training
- Standard library components for file I/O

Our hyperparameters define:
- The network architecture (input size, hidden layer size, output size)
- Training parameters (batch size, learning rate, number of epochs)
- How often to print progress updates

## 2. Model Definition

Next, we'll define our neural network architecture. For this task, we'll use a simple MLP with one hidden layer:

```rust
// Simple MLP for Iris classification
struct IrisClassifier {
    layer1: candle_nn::Linear,
    layer2: candle_nn::Linear,
}

impl IrisClassifier {
    fn new(_device: &Device, vb: VarBuilder) -> Result<Self> {
        let layer1 = candle_nn::linear(INPUT_SIZE, HIDDEN_SIZE, vb.pp("layer1"))?;
        let layer2 = candle_nn::linear(HIDDEN_SIZE, OUTPUT_SIZE, vb.pp("layer2"))?;
        Ok(Self { layer1, layer2 })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.layer1.forward(input)?;
        let hidden = hidden.relu()?;
        let output = self.layer2.forward(&hidden)?;
        Ok(output)
    }
}
```

Our `IrisClassifier` struct has two linear layers:
1. The first layer (`layer1`) transforms the 4 input features to 32 hidden units
2. The second layer (`layer2`) transforms the 32 hidden units to 3 output units (one for each class)

The `forward` method defines how data flows through the network:
1. The input passes through the first linear layer
2. The ReLU activation function is applied to introduce non-linearity
3. The result passes through the second linear layer to produce the final output

This is a simple feed-forward architecture, but it's powerful enough for our classification task.

## 3. Data Preparation

Now we need to load and prepare our data. We'll create functions to load the Iris dataset from a CSV file and generate batches for training:

```rust
// Load the Iris dataset from file
fn load_iris_dataset(device: &Device) -> Result<(Tensor, Tensor)> {
    // Path to the Iris dataset CSV file
    let file_path = Path::new("data/iris.csv");

    // Open the file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // Vectors to store features and labels
    let mut features_data: Vec<f32> = Vec::new();
    let mut labels_data: Vec<u32> = Vec::new();

    // Read the file line by line
    for (i, line_result) in reader.lines().enumerate() {
        // Skip the header line
        if i == 0 {
            continue;
        }

        let line = line_result?;
        let values: Vec<&str> = line.split(',').collect();

        if values.len() < 5 {
            return Err(anyhow::anyhow!("Invalid data format in line {}: {}", i, line));
        }

        // Parse the 4 feature values
        for j in 0..4 {
            let value = values[j].parse::<f32>()
                .map_err(|_| anyhow::anyhow!("Failed to parse feature value: {}", values[j]))?;
            features_data.push(value);
        }

        // Parse the label (species)
        let label = match values[4] {
            "Iris-setosa" => 0,
            "Iris-versicolor" => 1,
            "Iris-virginica" => 2,
            _ => return Err(anyhow::anyhow!("Unknown species: {}", values[4])),
        };
        labels_data.push(label);
    }

    // Create tensors and normalize features
    let num_samples = labels_data.len();
    let features = Tensor::from_vec(features_data, (num_samples, 4), device)?;
    let labels = Tensor::from_slice(&labels_data, (num_samples,), device)?;
    
    // Normalize features using min-max scaling
    let features_min = features.min(0)?.reshape((1, 4))?;
    let features_max = features.max(0)?.reshape((1, 4))?;
    let features_range = features_max.sub(&features_min)?;
    let normalized_features = features.broadcast_sub(&features_min)?
                                     .broadcast_div(&features_range)?;

    Ok((normalized_features, labels))
}
```

We also implement functions for generating training batches and calculating accuracy. The `generate_batches` function shuffles the data and creates batches of the specified size. The `calculate_accuracy` function compares predicted classes with true labels to compute the accuracy.

```rust
// Generate batches for training
fn generate_batches(features: &Tensor, labels: &Tensor, batch_size: usize, device: &Device, rng: &mut StdRng) -> Result<Vec<(Tensor, Tensor)>> {
    let num_samples = features.dim(0)?;
    let num_batches = (num_samples + batch_size - 1) / batch_size;

    // Create indices and shuffle them
    let mut indices: Vec<usize> = (0..num_samples).collect();
    for i in (1..indices.len()).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }

    let mut batches = Vec::with_capacity(num_batches);

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = std::cmp::min(start_idx + batch_size, num_samples);
        let batch_indices = &indices[start_idx..end_idx];

        let mut batch_features = Vec::with_capacity(batch_indices.len() * 4);
        let mut batch_labels = Vec::with_capacity(batch_indices.len());

        for &idx in batch_indices {
            let feature = features.i(idx)?;
            let feature_vec = feature.to_vec1::<f32>()?;
            batch_features.extend_from_slice(&feature_vec);

            let label = labels.i(idx)?.to_scalar::<u32>()?;
            batch_labels.push(label);
        }

        let batch_size = batch_indices.len();
        let batch_features_tensor = Tensor::from_slice(&batch_features, (batch_size, 4), device)?;
        let batch_labels_tensor = Tensor::from_slice(&batch_labels, (batch_size,), device)?;

        batches.push((batch_features_tensor, batch_labels_tensor));
    }

    Ok(batches)
}

// Calculate classification accuracy
fn calculate_accuracy(predictions: &Tensor, targets: &Tensor) -> Result<f32> {
    let pred_indices = predictions.argmax(1)?;
    let num_samples = targets.dim(0)?;

    let mut correct = 0;
    for i in 0..num_samples {
        let pred_idx = pred_indices.i(i)?.to_scalar::<u32>()?;
        let target_idx = targets.i(i)?.to_scalar::<u32>()?;
        if pred_idx == target_idx {
            correct += 1;
        }
    }

    Ok(correct as f32 / num_samples as f32)
}
```

The data preparation involves several steps:

1. **Loading the dataset**:
   - We read the CSV file line by line
   - Parse the feature values and convert species names to numeric labels
   - Create tensors for features and labels

2. **Normalizing the features**:
   - We apply min-max scaling to normalize each feature to the range [0, 1]
   - This helps the neural network converge faster and perform better

3. **Generating batches**:
   - We shuffle the data to prevent the model from learning the order of samples
   - Create batches of the specified size for efficient training
   - Each batch contains both features and corresponding labels

4. **Calculating accuracy**:
   - We define a helper function to evaluate model performance
   - It compares predicted classes with true labels and calculates the accuracy

These data preparation steps are crucial for effective training and evaluation of our neural network.

## 4. Training

Now we're ready to set up the training process. We'll initialize our model and optimizer, then implement the training loop.

### Device Setup

First, we need to set up the device for computation. Candle supports multiple backends:
- CUDA for NVIDIA GPUs
- Metal for Apple GPUs
- CPU as a fallback option

The code tries to use the most efficient available device, falling back to CPU if necessary:

```rust
// Set up device
let device = Device::cuda_if_available(0).unwrap_or_else(|_| {
    println!("CUDA device not available, trying Metal...");
    Device::new_metal(0).unwrap_or_else(|_| {
        println!("Metal device not available, falling back to CPU");
        Device::Cpu
    })
});
println!("Using device: {:?}", device);
```

### Model and Optimizer Initialization

Next, we initialize our model and set up the optimizer:

```rust
// Load iris dataset
let (features, labels) = load_iris_dataset(&device)?;
println!("Loaded Iris dataset: {} samples", features.dim(0)?);

// Create model
let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
let model = IrisClassifier::new(&device, vb)?;

// Set up optimizer
let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;

// Set up RNG for reproducibility
let mut rng = StdRng::seed_from_u64(42);
```

We create a `VarMap` to store the model parameters, initialize a `VarBuilder` with the appropriate data type and device, create our `IrisClassifier` model, and set up the AdamW optimizer with our specified learning rate.

The AdamW optimizer is a variant of Adam that includes weight decay regularization. This optimizer adapts the learning rate for each parameter based on historical gradients.

### Training Loop

The training loop is where the model learns from the data:

```rust
// Training loop
println!("Starting training...");
for epoch in tqdm(0..EPOCHS) {
    // Generate batches
    let batches = generate_batches(&features, &labels, BATCH_SIZE, &device, &mut rng)?;

    let mut epoch_loss = 0.0;
    let mut epoch_accuracy = 0.0;

    for (batch_features, batch_labels) in &batches {
        // Forward pass
        let logits = model.forward(batch_features)?;

        // Calculate loss (cross-entropy)
        let loss = candle_nn::loss::cross_entropy(&logits, batch_labels)?;

        // Backward pass and optimize
        optimizer.backward_step(&loss)?;

        // Calculate accuracy
        let batch_accuracy = calculate_accuracy(&logits, batch_labels)?;

        epoch_loss += loss.to_scalar::<f32>()?;
        epoch_accuracy += batch_accuracy;
    }

    epoch_loss /= batches.len() as f32;
    epoch_accuracy /= batches.len() as f32;

    // Print epoch summary
    if epoch % PRINT_EVERY == 0 || epoch == EPOCHS - 1 {
        println!("Epoch {}/{}: Loss = {:.4}, Accuracy = {:.4}",
                 epoch + 1, EPOCHS, epoch_loss, epoch_accuracy);
    }
}
```

For each epoch, we:

1. Generate new shuffled batches
2. Process each batch through the model
3. Calculate the loss and update the model parameters
4. Track and report the progress

For each batch, we perform the following steps:

1. **Forward Pass**: Pass the batch through the model to get predictions
2. **Loss Calculation**: Calculate the cross-entropy loss between predictions and true labels
3. **Backward Pass and Optimization**: Update the model parameters to minimize the loss
4. **Metrics Tracking**: Calculate and track the accuracy and loss

After each epoch, we print the average loss and accuracy if it's a reporting epoch. This allows us to monitor the training progress and ensure the model is learning effectively.

This training process allows our model to learn the patterns in the Iris dataset and improve its classification accuracy over time.

## 5. Inference

After training, we use our model to make predictions and evaluate its performance. This involves several steps:

### Overall Evaluation

First, we evaluate the model on the entire dataset by:

```rust
// Evaluate on the full dataset
let logits = model.forward(&features)?;
let accuracy = calculate_accuracy(&logits, &labels)?;
println!("\nFinal classification accuracy: {:.4}", accuracy);
```

This gives us a quantitative measure of how well our model performs.

### Confusion Matrix

Next, we create a confusion matrix to see how well the model performs for each class:

```rust
// Get class predictions
let predictions = logits.argmax(1)?;

// Print confusion matrix
println!("\nConfusion Matrix:");
let mut confusion_matrix = vec![vec![0; OUTPUT_SIZE]; OUTPUT_SIZE];

for i in 0..features.dim(0)? {
    let pred_idx = predictions.i(i)?.to_scalar::<u32>()? as usize;
    let true_idx = labels.i(i)?.to_scalar::<u32>()? as usize;
    confusion_matrix[true_idx][pred_idx] += 1;
}

println!("True\\Pred | Setosa | Versicolor | Virginica");
println!("---------|--------|------------|----------");
println!("Setosa    | {:6} | {:10} | {:9}", 
         confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[0][2]);
println!("Versicolor| {:6} | {:10} | {:9}", 
         confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[1][2]);
println!("Virginica | {:6} | {:10} | {:9}", 
         confusion_matrix[2][0], confusion_matrix[2][1], confusion_matrix[2][2]);
```

A confusion matrix shows:
- How many samples of each class were correctly classified (diagonal elements)
- How many samples were misclassified as another class (off-diagonal elements)

This helps us identify which classes the model struggles with and understand the types of errors it makes.

### Sample Predictions

Finally, we print some example predictions to see how the model performs on individual samples:

```rust
// Print some example predictions
println!("\nSample predictions:");
for class_id in 0..OUTPUT_SIZE {
    println!("Class {} ({}): ", class_id, match class_id {
        0 => "Iris-setosa",
        1 => "Iris-versicolor",
        2 => "Iris-virginica",
        _ => "Unknown",
    });

    let mut count = 0;
    for i in 0..features.dim(0)? {
        let true_label = labels.i(i)?.to_scalar::<u32>()?;
        let pred_label = predictions.i(i)?.to_scalar::<u32>()?;

        if true_label == class_id as u32 && count < 3 {
            let feature = features.i(i)?;
            let feature_vec = feature.to_vec1::<f32>()?;

            println!("  Sample {}: Features = [{:.2}, {:.2}, {:.2}, {:.2}], Predicted = {}",
                     i, feature_vec[0], feature_vec[1], feature_vec[2], feature_vec[3], 
                     match pred_label {
                         0 => "Iris-setosa",
                         1 => "Iris-versicolor",
                         2 => "Iris-virginica",
                         _ => "Unknown",
                     });
            count += 1;
        }
    }
}
```

For each class, we show a few examples with:
- The input features
- The true class
- The predicted class


```text
Using device: Cpu
Loaded 150 samples from data/iris.csv
Loaded Iris dataset: 150 samples
Starting training...
.. epochs deleted ..
Final classification accuracy: 0.9733

Confusion Matrix:
True\Pred | Setosa | Versicolor | Virginica
---------|--------|------------|----------
Setosa    |     50 |          0 |         0
Versicolor|      0 |         47 |         3
Virginica |      0 |          1 |        49

Sample predictions:
Class 0 (Iris-setosa): 
  Sample 0: Features = [0.22, 0.62, 0.07, 0.04], Predicted = Iris-setosa
  Sample 1: Features = [0.17, 0.42, 0.07, 0.04], Predicted = Iris-setosa
  Sample 2: Features = [0.11, 0.50, 0.05, 0.04], Predicted = Iris-setosa
Class 1 (Iris-versicolor): 
  Sample 50: Features = [0.75, 0.50, 0.63, 0.54], Predicted = Iris-versicolor
  Sample 51: Features = [0.58, 0.50, 0.59, 0.58], Predicted = Iris-versicolor
  Sample 52: Features = [0.72, 0.46, 0.66, 0.58], Predicted = Iris-versicolor
Class 2 (Iris-virginica): 
  Sample 100: Features = [0.56, 0.54, 0.85, 1.00], Predicted = Iris-virginica
  Sample 101: Features = [0.42, 0.29, 0.69, 0.75], Predicted = Iris-virginica
  Sample 102: Features = [0.78, 0.42, 0.83, 0.83], Predicted = Iris-virginica
```

This gives us a more intuitive understanding of the model's behavior and helps us verify that it's making reasonable predictions.

## Conclusion

In this chapter, we've built a complete neural network for Iris flower classification using the Candle framework. We've covered all the essential components of neural network development:

1. **Imports and Setup**: Setting up the environment and defining hyperparameters
2. **Model Definition**: Creating a neural network architecture
3. **Data Preparation**: Loading, preprocessing, and batching data
4. **Training**: Implementing the training loop with optimization
5. **Inference**: Making predictions and evaluating performance

This example demonstrates how to use Candle to build practical neural networks for real-world tasks. The principles we've covered here can be extended to more complex models and datasets.

In the next chapter, we'll explore more advanced neural network architectures and techniques for handling different types of data.