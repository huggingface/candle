# 27. Jupyter Notebooks

## Introduction

Visualization is a crucial aspect of machine learning that helps us understand, debug, and communicate our models and results. Effective visualizations can reveal patterns in data, track training progress, identify issues, and provide insights into model behavior that might otherwise remain hidden in raw numbers.

In this chapter, we'll explore how to use Jupyter notebooks with Rust and Candle to visualize model training progress. We'll focus on one of the most common visualization techniques in machine learning: plotting loss and accuracy curves during model training.

## Setting Up Jupyter Notebooks with Rust

[Jupyter notebooks](https://jupyter.org/) are interactive documents that combine code, visualizations, and narrative text. While traditionally associated with Python, Jupyter notebooks can be used with Rust through the [evcxr Jupyter kernel](https://github.com/google/evcxr/tree/main/evcxr_jupyter).

### Installing the evcxr Jupyter Kernel

To use Rust in Jupyter notebooks, you'll need to install the evcxr kernel:

1. First, ensure you have Jupyter installed:
   ```
   pip install jupyter
   ```

2. Install the evcxr Jupyter kernel:
   ```
   cargo install evcxr_jupyter
   ```

3. Register the kernel with Jupyter:
   ```
   evcxr_jupyter --install
   ```

4. Launch Jupyter:
   ```
   jupyter notebook
   ```

5. Create a new notebook with the Rust kernel by selecting "Rust" from the "New" dropdown menu.

### Basic Usage in Jupyter

Here's a simple example of using Rust in a Jupyter notebook:

```
// This is a cell in a Jupyter notebook
println!("Hello, Candle!");

// You can define variables and use them in subsequent cells
let x = 42;
x * 2
```

The evcxr kernel supports many Rust features, including:
- Loading crates with `:dep` directives
- Displaying custom output
- Defining functions and structs
- Using external dependencies

## Visualizing Model Training in Jupyter Notebooks

Now, let's explore how to visualize model training progress using Candle in a Jupyter notebook. We'll focus on plotting loss and accuracy curves, which are essential for monitoring training progress and diagnosing issues like overfitting.

### Required Dependencies

First, we need to load the necessary dependencies in our Jupyter notebook:

```
// In a Jupyter notebook cell, load dependencies with :dep
:dep candle-core = { version = "0.3.0" }
:dep candle-nn = { version = "0.3.0" }
:dep plotters = "0.3.5"
:dep rand = "0.8.5"
```

### Creating a Simple Model for Demonstration

For our visualization example, we'll create a simple CNN model similar to the one we implemented in Chapter 14 for MNIST digit classification. We'll track the loss and accuracy during training and visualize them.

```
// In a Jupyter notebook cell
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, AdamW, Module, Optimizer, VarBuilder, VarMap};
use plotters::prelude::*;
use rand::prelude::*;

// Define a simple CNN model
struct SimpleCNN {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl SimpleCNN {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 3, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 3, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(64 * 5 * 5, 128, vs.pp("l1"))?;
        let fc2 = candle_nn::linear(128, 10, vs.pp("l2"))?;

        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
        })
    }
}

impl Module for SimpleCNN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv1.forward(xs)?.relu()?;
        let xs = xs.max_pool2d_with_stride(2, 2)?;
        
        let xs = self.conv2.forward(&xs)?.relu()?;
        let xs = xs.max_pool2d_with_stride(2, 2)?;
        
        let xs = xs.flatten_from(1)?;
        
        let xs = self.fc1.forward(&xs)?.relu()?;
        self.fc2.forward(&xs)
    }
}
```

### Generating Synthetic Data for Demonstration

Since we're focusing on visualization rather than model performance, we'll generate synthetic data for our example:

```
// In a Jupyter notebook cell
// Generate synthetic data for demonstration
fn generate_synthetic_data(device: &Device) -> Result<(Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(42);
    
    // Generate 1000 random 28x28 images
    let mut images_data = vec![0f32; 1000 * 28 * 28];
    rng.fill(&mut images_data[..]);
    
    // Generate random labels (0-9)
    let mut labels_data = vec![0u8; 1000];
    for label in &mut labels_data {
        *label = rng.gen_range(0..10);
    }
    
    let images = Tensor::from_vec(images_data, (1000, 1, 28, 28), device)?;
    let labels = Tensor::from_vec(labels_data, (1000,), device)?;
    
    Ok((images, labels))
}
```

### Training Loop with Metric Collection

Now, let's implement a training loop that collects loss and accuracy metrics for visualization:

```
// In a Jupyter notebook cell
// Train the model and collect metrics
fn train_model(
    model: &SimpleCNN,
    optimizer: &mut AdamW,
    images: &Tensor,
    labels: &Tensor,
    batch_size: usize,
    epochs: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut losses = Vec::with_capacity(epochs);
    let mut accuracies = Vec::with_capacity(epochs);
    
    let num_samples = images.dim(0)?;
    let num_batches = num_samples / batch_size;
    
    for epoch in 0..epochs {
        let mut sum_loss = 0f32;
        let mut correct_predictions = 0;
        
        // Create a random permutation for shuffling
        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut thread_rng());
        
        for batch_idx in 0..num_batches {
            let batch_indices = &indices[batch_idx * batch_size..(batch_idx + 1) * batch_size];
            
            // Extract batch data
            let batch_images = images.index_select(&Tensor::from_vec(
                batch_indices.iter().map(|&i| i as u32).collect(),
                (batch_size,),
                images.device()
            )?)?;
            
            let batch_labels = labels.index_select(&Tensor::from_vec(
                batch_indices.iter().map(|&i| i as u32).collect(),
                (batch_size,),
                labels.device()
            )?)?;
            
            // Forward pass
            let logits = model.forward(&batch_images)?;
            
            // Compute loss
            let loss = loss::cross_entropy(&logits, &batch_labels)?;
            
            // Backward pass and optimization
            optimizer.backward_step(&loss)?;
            
            // Calculate accuracy
            let predictions = logits.argmax(1)?;
            let batch_labels_u32 = batch_labels.to_dtype(DType::U32)?;
            let correct = predictions.eq(&batch_labels_u32)?.sum_all()?.to_scalar::<u32>()?;
            
            correct_predictions += correct as usize;
            sum_loss += loss.to_scalar::<f32>()?;
        }
        
        let avg_loss = sum_loss / num_batches as f32;
        let accuracy = correct_predictions as f32 / (num_batches * batch_size) as f32;
        
        println!("Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", epoch, avg_loss, accuracy * 100.0);
        
        losses.push(avg_loss);
        accuracies.push(accuracy);
    }
    
    Ok((losses, accuracies))
}
```

### Visualizing Training Metrics

Now, let's create a function to visualize the training metrics using the Plotters library:

```
// In a Jupyter notebook cell
// Visualize training metrics
fn plot_training_metrics(
    losses: &[f32],
    accuracies: &[f32],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a drawing area
    let root = BitMapBackend::new("training_metrics.png", (width, height)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Split the drawing area into two parts for loss and accuracy
    let areas = root.split_evenly((1, 2));
    
    // Plot loss
    {
        let mut chart = ChartBuilder::on(&areas[0])
            .caption("Training Loss", ("sans-serif", 20).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0..losses.len(),
                0f32..*losses.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&1.0) * 1.1,
            )?;
        
        chart.configure_mesh()
            .x_desc("Epoch")
            .y_desc("Loss")
            .draw()?;
        
        chart.draw_series(LineSeries::new(
            (0..losses.len()).map(|i| (i, losses[i])),
            &RED,
        ))?
        .label("Training Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }
    
    // Plot accuracy
    {
        let mut chart = ChartBuilder::on(&areas[1])
            .caption("Training Accuracy", ("sans-serif", 20).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0..accuracies.len(),
                0f32..1f32,
            )?;
        
        chart.configure_mesh()
            .x_desc("Epoch")
            .y_desc("Accuracy")
            .y_label_formatter(&|v| format!("{:.0}%", v * 100.0))
            .draw()?;
        
        chart.draw_series(LineSeries::new(
            (0..accuracies.len()).map(|i| (i, accuracies[i])),
            &BLUE,
        ))?
        .label("Training Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }
    
    // Save the plot
    root.present()?;
    println!("Plot saved as training_metrics.png");
    
    Ok(())
}
```

### Putting It All Together

Now, let's put everything together in a complete example:

```
// In a Jupyter notebook cell
// Set up the device
let device = Device::Cpu;

// Generate synthetic data
let (images, labels) = generate_synthetic_data(&device)?;

// Create the model
let varmap = VarMap::new();
let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
let model = SimpleCNN::new(vs.clone())?;

// Set up the optimizer
let mut optimizer = AdamW::new_lr(varmap.all_vars(), 1e-3)?;

// Train the model and collect metrics
let (losses, accuracies) = train_model(&model, &mut optimizer, &images, &labels, 32, 10)?;

// Visualize the metrics
plot_training_metrics(&losses, &accuracies, 800, 600)?;
```

## Displaying Plots in Jupyter Notebooks

In a Jupyter notebook, we can display the generated plot directly in the notebook. The evcxr kernel supports displaying images using the `display_png` function:

```
// In a Jupyter notebook cell
// Display the plot in the notebook
let plot_data = std::fs::read("training_metrics.png")?;
evcxr_runtime::mime_type::image_png(&plot_data)
```

## Interactive Visualization in Jupyter Notebooks

One of the advantages of using Jupyter notebooks is the ability to create interactive visualizations. While Rust's ecosystem for interactive visualization is still developing, we can use libraries like Plotters to create static visualizations and update them as training progresses.

Here's an example of how to create an interactive training loop that updates the visualization after each epoch:

```
// In a Jupyter notebook cell
// Interactive training with visualization updates
fn interactive_training(
    model: &SimpleCNN,
    optimizer: &mut AdamW,
    images: &Tensor,
    labels: &Tensor,
    batch_size: usize,
    epochs: usize,
) -> Result<()> {
    let mut losses = Vec::with_capacity(epochs);
    let mut accuracies = Vec::with_capacity(epochs);
    
    let num_samples = images.dim(0)?;
    let num_batches = num_samples / batch_size;
    
    for epoch in 0..epochs {
        let mut sum_loss = 0f32;
        let mut correct_predictions = 0;
        
        // Training code (same as before)
        // ...
        
        // Update metrics
        let avg_loss = sum_loss / num_batches as f32;
        let accuracy = correct_predictions as f32 / (num_batches * batch_size) as f32;
        
        println!("Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", epoch, avg_loss, accuracy * 100.0);
        
        losses.push(avg_loss);
        accuracies.push(accuracy);
        
        // Update visualization after each epoch
        plot_training_metrics(&losses, &accuracies, 800, 600)?;
        
        // Display the updated plot in the notebook
        let plot_data = std::fs::read("training_metrics.png")?;
        evcxr_runtime::mime_type::image_png(&plot_data);
        
        // Add a small delay to see the update
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    
    Ok(())
}
```

## Advanced Visualization Techniques

While loss and accuracy curves are the most common visualizations for model training, there are many other visualization techniques that can provide insights into your models:

1. **Learning Rate Visualization**: Plot learning rate schedules and their effects on training.
2. **Gradient Magnitude Visualization**: Track the magnitude of gradients during training to detect vanishing or exploding gradients.
3. **Weight Distribution Visualization**: Plot histograms of model weights to understand how they evolve during training.
4. **Confusion Matrix Visualization**: Visualize model predictions across different classes.
5. **Feature Map Visualization**: Visualize the activations of convolutional layers to understand what features the model is learning.

Let's implement a simple example of visualizing weight distributions:

```
// In a Jupyter notebook cell
// Visualize weight distributions
fn plot_weight_distribution(model: &SimpleCNN) -> Result<(), Box<dyn std::error::Error>> {
    // Extract weights from the first convolutional layer
    let conv1_weights = model.conv1.weight().flatten_all()?;
    let weights_vec: Vec<f32> = conv1_weights.to_vec1()?;
    
    // Create a drawing area
    let root = BitMapBackend::new("weight_distribution.png", (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Create a histogram
    let mut chart = ChartBuilder::on(&root)
        .caption("Weight Distribution (Conv1)", ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            -0.5f32..0.5f32,
            0..100u32,
        )?;
    
    chart.configure_mesh()
        .x_desc("Weight Value")
        .y_desc("Count")
        .draw()?;
    
    // Create histogram data
    let bin_width = 0.05;
    let mut histogram = vec![0; 20];
    
    for &weight in &weights_vec {
        let bin = ((weight + 0.5) / bin_width).floor() as usize;
        if bin < histogram.len() {
            histogram[bin] += 1;
        }
    }
    
    // Draw the histogram
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(GREEN.filled())
            .margin(0)
            .data(histogram.iter().enumerate().map(|(i, &count)| {
                ((-0.5 + bin_width * i as f32, count as u32), bin_width)
            }))
    )?;
    
    // Save the plot
    root.present()?;
    println!("Weight distribution plot saved as weight_distribution.png");
    
    Ok(())
}
```

## Conclusion

Visualization is an essential tool for understanding, debugging, and improving machine learning models. In this chapter, we've explored how to use Jupyter notebooks with Rust and Candle to visualize model training progress, focusing on loss and accuracy curves as a common example.

Key takeaways:
- Jupyter notebooks provide an interactive environment for Rust code and visualizations
- The evcxr kernel enables Rust support in Jupyter notebooks
- Plotters is a powerful library for creating visualizations in Rust
- Visualizing training metrics helps identify issues like overfitting and underfitting
- Advanced visualization techniques can provide deeper insights into model behavior

By combining the performance and safety of Rust with the interactive nature of Jupyter notebooks, you can create powerful visualizations that help you understand and improve your machine learning models.

In the next chapter, we'll explore how to access Candle from Python, enabling interoperability between Rust and the Python machine learning ecosystem.