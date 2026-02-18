# 22. Data Preprocessing

## Introduction

Data loading and preprocessing are critical steps in any machine learning workflow. Before a model can learn patterns or make predictions, it needs properly formatted, cleaned, and prepared data. In deep learning, how you prepare your data can significantly impact model performance, training speed, and convergence.

This chapter explores:
- The importance of data loading and preprocessing in machine learning
- Candle's approach to handling different data formats
- Techniques for loading various data types (images, text, tabular data)
- Creating and manipulating tensors from raw data
- Common preprocessing techniques (normalization, standardization, one-hot encoding)
- Data augmentation strategies for improving model generalization
- Building efficient data pipelines
- Batching and mini-batch processing
- Memory optimization for large datasets
- Practical examples with complete code
- Best practices and common pitfalls

## The Importance of Data Preprocessing

### Why Preprocessing Matters

Data preprocessing serves several critical functions in the machine learning pipeline:

1. **Data Cleaning**: Removing or correcting errors, handling missing values, and addressing outliers
2. **Feature Engineering**: Creating new features or transforming existing ones to improve model performance
3. **Normalization**: Ensuring features are on similar scales to prevent some features from dominating others
4. **Dimensionality Reduction**: Reducing the number of features to improve training efficiency
5. **Format Conversion**: Converting data into the tensor format required by deep learning models

Proper preprocessing can:
- Improve model accuracy and generalization
- Reduce training time
- Prevent numerical issues during training
- Make models more robust to variations in input data

### Common Preprocessing Steps

A typical preprocessing pipeline might include:

1. **Data Collection**: Gathering raw data from various sources
2. **Data Cleaning**: Handling missing values, removing duplicates, correcting errors
3. **Feature Selection/Engineering**: Choosing relevant features or creating new ones
4. **Data Transformation**: Normalizing, standardizing, or encoding categorical variables
5. **Data Splitting**: Dividing data into training, validation, and test sets
6. **Data Augmentation**: Creating variations of existing data to improve generalization
7. **Batching**: Organizing data into mini-batches for efficient training

## Data Handling in Candle

Candle provides several ways to load and preprocess data, with a focus on efficiency and integration with Rust's ecosystem.

### Candle's Data Representation

At the core of Candle's data handling is the `Tensor` struct, which represents multi-dimensional arrays of numeric values. Tensors are the fundamental data structure used throughout the framework for:

- Input data
- Model parameters
- Intermediate activations
- Model outputs

Tensors in Candle have:
- A shape (dimensions)
- A data type (e.g., f32, f64, i64)
- A device location (CPU or GPU)

### Creating Tensors from Raw Data

There are several ways to create tensors from raw data:

```rust
use candle_core::{Device, Tensor, Result};

fn create_tensors() -> Result<()> {
    let device = Device::Cpu;
    
    // From a vector
    let vec_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor_1d = Tensor::new(&vec_data, &device)?;
    println!("1D tensor: {}", tensor_1d);
    
    // Reshape to 2D
    let tensor_2d = tensor_1d.reshape((2, 3))?;
    println!("2D tensor: {}", tensor_2d);
    
    // From nested vectors
    let nested_data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ];
    let tensor_2d_direct = Tensor::new(&nested_data, &device)?;
    println!("2D tensor from nested vec: {}", tensor_2d_direct);
    
    // Random tensors
    let random_uniform = Tensor::rand(-1.0, 1.0, &[2, 3], &device)?;
    println!("Random uniform tensor: {}", random_uniform);
    
    let random_normal = Tensor::randn(0.0, 1.0, &[2, 3], &device)?;
    println!("Random normal tensor: {}", random_normal);
    
    // Constant tensors
    let ones = Tensor::ones((2, 3), candle_core::DType::F32, &device)?;
    println!("Ones tensor: {}", ones);
    
    let zeros = Tensor::zeros((2, 3), candle_core::DType::F32, &device)?;
    println!("Zeros tensor: {}", zeros);
    
    Ok(())
}
```

## Loading Different Data Formats

### Loading CSV Data

CSV (Comma-Separated Values) is a common format for tabular data. Here's how to load and preprocess CSV data with Candle:

```rust
use candle_core::{Device, Tensor, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_csv_data(file_path: &str, has_header: bool) -> Result<(Tensor, Tensor)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Skip header if present
    if has_header {
        lines.next();
    }
    
    let mut features = Vec::new();
    let mut labels = Vec::new();
    
    for line in lines {
        let line = line?;
        let values: Vec<&str> = line.split(',').collect();
        
        // Assume the last column is the label
        let label = values.last().unwrap().parse::<f32>()?;
        labels.push(label);
        
        // Convert feature columns to f32
        let feature_values: Vec<f32> = values[..values.len()-1]
            .iter()
            .map(|v| v.parse::<f32>().unwrap_or(0.0))
            .collect();
        
        features.push(feature_values);
    }
    
    // Create tensors
    let device = Device::Cpu;
    let features_tensor = Tensor::new(&features, &device)?;
    let labels_tensor = Tensor::new(&labels, &device)?;
    
    Ok((features_tensor, labels_tensor))
}

// Example usage
fn process_csv_example() -> Result<()> {
    let (features, labels) = load_csv_data("data/iris.csv", true)?;
    
    println!("Features shape: {:?}", features.shape());
    println!("Labels shape: {:?}", labels.shape());
    
    // Normalize features
    let mean = features.mean_dim(0, true)?;
    let std = features.std_dim(0, true)?;
    let normalized_features = features.broadcast_sub(&mean)?.broadcast_div(&std)?;
    
    println!("Normalized features: {}", normalized_features.get(0)?);
    
    Ok(())
}
```

### Loading Image Data

Images require special handling due to their multi-dimensional nature. Here's how to load and preprocess image data:

```rust
use candle_core::{Device, Tensor, Result};
use image::{GenericImageView, DynamicImage};
use std::path::Path;

fn load_image(path: &str, size: (u32, u32)) -> Result<Tensor> {
    let img = image::open(Path::new(path))?;
    
    // Resize image
    let img = img.resize_exact(size.0, size.1, image::imageops::FilterType::Triangle);
    
    // Convert to RGB tensor
    let (width, height) = img.dimensions();
    let mut tensor_data = Vec::with_capacity((width * height * 3) as usize);
    
    // Extract RGB values and normalize to [0, 1]
    for pixel in img.pixels() {
        let rgb = pixel.2;
        tensor_data.push(rgb[0] as f32 / 255.0);
        tensor_data.push(rgb[1] as f32 / 255.0);
        tensor_data.push(rgb[2] as f32 / 255.0);
    }
    
    // Create tensor with shape [channels, height, width]
    let tensor = Tensor::from_vec(
        tensor_data,
        (3, height as usize, width as usize),
        &Device::Cpu,
    )?;
    
    Ok(tensor)
}

// Load a batch of images from a directory
fn load_image_batch(dir_path: &str, image_paths: &[String], size: (u32, u32)) -> Result<Tensor> {
    let mut images = Vec::new();
    
    for path in image_paths {
        let full_path = format!("{}/{}", dir_path, path);
        let img_tensor = load_image(&full_path, size)?;
        images.push(img_tensor);
    }
    
    // Stack tensors along a new batch dimension
    Tensor::stack(&images, 0)
}

// Example usage
fn process_image_example() -> Result<()> {
    // Load a single image
    let img = load_image("data/images/sample.jpg", (224, 224))?;
    println!("Image shape: {:?}", img.shape());
    
    // Load a batch of images
    let image_paths = vec![
        "img1.jpg".to_string(),
        "img2.jpg".to_string(),
        "img3.jpg".to_string(),
    ];
    
    let batch = load_image_batch("data/images", &image_paths, (224, 224))?;
    println!("Batch shape: {:?}", batch.shape());
    
    // Apply normalization with ImageNet mean and std
    let mean = Tensor::new(&[0.485, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.229, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
    
    let normalized_batch = batch.broadcast_sub(&mean)?.broadcast_div(&std)?;
    println!("Normalized batch shape: {:?}", normalized_batch.shape());
    
    Ok(())
}
```

### Loading Text Data

Text data requires tokenization and conversion to numerical representations:

```rust
use candle_core::{Device, Tensor, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

// Simple tokenizer for text data
struct SimpleTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: Vec<String>,
}

impl SimpleTokenizer {
    fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            reverse_vocab: Vec::new(),
        }
    }
    
    fn build_vocab(&mut self, texts: &[String]) {
        // Add special tokens
        self.add_token("<PAD>");
        self.add_token("<UNK>");
        
        // Add tokens from texts
        for text in texts {
            for word in text.split_whitespace() {
                self.add_token(word);
            }
        }
    }
    
    fn add_token(&mut self, token: &str) {
        if !self.vocab.contains_key(token) {
            let idx = self.vocab.len();
            self.vocab.insert(token.to_string(), idx);
            self.reverse_vocab.push(token.to_string());
        }
    }
    
    fn encode(&self, text: &str, max_length: usize) -> Vec<usize> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = Vec::with_capacity(max_length);
        
        // Convert words to token IDs
        for word in words.iter().take(max_length) {
            let token_id = self.vocab.get(*word).copied().unwrap_or(1); // 1 is <UNK>
            tokens.push(token_id);
        }
        
        // Pad to max_length
        while tokens.len() < max_length {
            tokens.push(0); // 0 is <PAD>
        }
        
        tokens
    }
    
    fn decode(&self, token_ids: &[usize]) -> String {
        token_ids.iter()
            .filter(|&&id| id > 0) // Skip padding
            .map(|&id| {
                if id < self.reverse_vocab.len() {
                    self.reverse_vocab[id].clone()
                } else {
                    "<UNK>".to_string()
                }
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// Load text data from file
fn load_text_data(file_path: &str) -> Result<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let lines: Result<Vec<String>, _> = reader.lines().collect();
    Ok(lines?)
}

// Process text data for a classification task
fn process_text_classification(
    texts: &[String],
    labels: &[usize],
    max_length: usize,
) -> Result<(Tensor, Tensor)> {
    // Create and build tokenizer
    let mut tokenizer = SimpleTokenizer::new();
    tokenizer.build_vocab(texts);
    
    // Encode texts
    let mut encoded_texts = Vec::with_capacity(texts.len());
    for text in texts {
        encoded_texts.push(tokenizer.encode(text, max_length));
    }
    
    // Create tensors
    let device = Device::Cpu;
    let input_tensor = Tensor::new(&encoded_texts, &device)?;
    let label_tensor = Tensor::new(labels, &device)?;
    
    Ok((input_tensor, label_tensor))
}

// Example usage
fn process_text_example() -> Result<()> {
    // Sample data
    let texts = vec![
        "this movie was great".to_string(),
        "the acting was terrible".to_string(),
        "i loved the cinematography".to_string(),
    ];
    
    let labels = vec![1, 0, 1]; // 1 for positive, 0 for negative
    
    // Process data
    let (input_tensor, label_tensor) = process_text_classification(&texts, &labels, 10)?;
    
    println!("Input tensor shape: {:?}", input_tensor.shape());
    println!("Label tensor shape: {:?}", label_tensor.shape());
    
    Ok(())
}
```

## Common Preprocessing Techniques

### Normalization and Standardization

Normalization and standardization are essential for ensuring features are on similar scales:

```rust
use candle_core::{Tensor, Result};

// Min-Max Normalization: scales data to [0, 1] range
fn min_max_normalize(tensor: &Tensor) -> Result<Tensor> {
    let min = tensor.min_all()?;
    let max = tensor.max_all()?;
    let range = max.sub(&min)?;
    
    // (x - min) / (max - min)
    tensor.sub(&min)?.div(&range)
}

// Z-score Standardization: transforms data to have mean=0, std=1
fn standardize(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    let mean = tensor.mean_dim(dim, true)?;
    let std = tensor.std_dim(dim, true)?;
    
    // (x - mean) / std
    tensor.broadcast_sub(&mean)?.broadcast_div(&std)
}

// Example usage
fn normalization_example() -> Result<()> {
    use candle_core::Device;
    
    let device = Device::Cpu;
    let data = Tensor::new(&[
        [1.0f32, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], &device)?;
    
    // Min-max normalization
    let normalized = min_max_normalize(&data)?;
    println!("Min-max normalized:\n{}", normalized);
    
    // Z-score standardization (along columns)
    let standardized = standardize(&data, 0)?;
    println!("Standardized (along columns):\n{}", standardized);
    
    Ok(())
}
```

### One-Hot Encoding

One-hot encoding is used to represent categorical variables:

```rust
use candle_core::{Tensor, Result, Device};

// One-hot encode categorical variables
fn one_hot_encode(tensor: &Tensor, num_classes: usize) -> Result<Tensor> {
    tensor.one_hot(num_classes)
}

// Example usage
fn one_hot_example() -> Result<()> {
    let device = Device::Cpu;
    
    // Class labels: 0, 1, 2, 1, 0
    let labels = Tensor::new(&[0, 1, 2, 1, 0], &device)?;
    
    // One-hot encode with 3 classes
    let one_hot = one_hot_encode(&labels, 3)?;
    println!("Labels: {}", labels);
    println!("One-hot encoded:\n{}", one_hot);
    
    Ok(())
}
```

### Handling Missing Values

Missing values can be handled in several ways:

```rust
use candle_core::{Tensor, Result, Device};

// Replace missing values (represented as NaN) with mean
fn impute_with_mean(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    // Calculate mean, ignoring NaN values
    let mask = tensor.is_nan()?.logical_not()?;
    let masked_tensor = tensor.where_cond(&mask, &Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.device())?)?;
    let count = mask.sum_dim(dim, true)?;
    let sum = masked_tensor.sum_dim(dim, true)?;
    let mean = sum.div(&count)?;
    
    // Replace NaN with mean
    let is_nan = tensor.is_nan()?;
    tensor.where_cond(&is_nan.logical_not()?, &mean.broadcast_like(tensor)?)
}

// Example usage
fn missing_values_example() -> Result<()> {
    let device = Device::Cpu;
    
    // Create tensor with some NaN values
    let data = Tensor::new(&[
        [1.0f32, 2.0, f32::NAN],
        [4.0, f32::NAN, 6.0],
        [7.0, 8.0, 9.0],
    ], &device)?;
    
    // Impute missing values with column means
    let imputed = impute_with_mean(&data, 0)?;
    println!("Original data:\n{}", data);
    println!("Imputed data:\n{}", imputed);
    
    Ok(())
}
```

## Data Augmentation

Data augmentation is a powerful technique to increase the diversity of your training data and improve model generalization.

### Image Augmentation

Common image augmentation techniques include:

```rust
use candle_core::{Tensor, Result, Device};
use rand::Rng;

// Random horizontal flip
fn random_horizontal_flip(image: &Tensor, p: f32) -> Result<Tensor> {
    let mut rng = rand::thread_rng();
    
    if rng.gen::<f32>() < p {
        // Assuming image shape is [channels, height, width]
        image.flip(2)
    } else {
        Ok(image.clone())
    }
}

// Random crop
fn random_crop(image: &Tensor, crop_size: (usize, usize)) -> Result<Tensor> {
    let (_, height, width) = image.dims3()?;
    let (crop_height, crop_width) = crop_size;
    
    if crop_height > height || crop_width > width {
        return Err(candle_core::Error::Msg("Crop size larger than image".to_string()));
    }
    
    let mut rng = rand::thread_rng();
    let top = rng.gen_range(0..height - crop_height + 1);
    let left = rng.gen_range(0..width - crop_width + 1);
    
    // Extract crop
    let cropped = image.narrow(1, top, crop_height)?.narrow(2, left, crop_width)?;
    
    Ok(cropped)
}

// Random rotation
fn random_rotation(image: &Tensor, max_angle: f32) -> Result<Tensor> {
    // This is a simplified implementation
    // In practice, you would use a proper image rotation function
    
    let mut rng = rand::thread_rng();
    let angle = rng.gen_range(-max_angle..max_angle);
    
    // Placeholder for actual rotation implementation
    println!("Rotating image by {} degrees", angle);
    
    // For now, just return the original image
    Ok(image.clone())
}

// Color jitter
fn color_jitter(image: &Tensor, brightness: f32, contrast: f32, saturation: f32) -> Result<Tensor> {
    let mut rng = rand::thread_rng();
    let mut result = image.clone();
    
    // Apply brightness adjustment
    if brightness > 0.0 {
        let factor = 1.0 + rng.gen_range(-brightness..brightness);
        result = result.mul_scalar(factor)?;
    }
    
    // Apply contrast adjustment
    if contrast > 0.0 {
        let factor = 1.0 + rng.gen_range(-contrast..contrast);
        let mean = result.mean_all()?;
        result = result.broadcast_sub(&mean)?.mul_scalar(factor)?.broadcast_add(&mean)?;
    }
    
    // Note: Saturation adjustment would require converting to HSV color space
    // This is a simplified implementation
    
    Ok(result)
}

// Apply a series of augmentations
fn augment_image(image: &Tensor) -> Result<Tensor> {
    let mut result = image.clone();
    
    // Apply random horizontal flip
    result = random_horizontal_flip(&result, 0.5)?;
    
    // Apply random crop and resize
    let (_, height, width) = result.dims3()?;
    let crop_size = (height * 9 / 10, width * 9 / 10);
    result = random_crop(&result, crop_size)?;
    
    // Apply color jitter
    result = color_jitter(&result, 0.2, 0.2, 0.2)?;
    
    Ok(result)
}

// Example usage
fn image_augmentation_example() -> Result<()> {
    let device = Device::Cpu;
    
    // Create a dummy image tensor [channels, height, width]
    let image = Tensor::rand(0.0, 1.0, (3, 224, 224), &device)?;
    
    // Apply augmentation
    let augmented = augment_image(&image)?;
    
    println!("Original shape: {:?}", image.shape());
    println!("Augmented shape: {:?}", augmented.shape());
    
    Ok(())
}
```

### Text Augmentation

Text augmentation techniques can include:

```rust
use rand::{Rng, seq::SliceRandom};

// Random word deletion
fn random_word_deletion(text: &str, p: f32) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut rng = rand::thread_rng();
    
    let filtered_words: Vec<&str> = words.iter()
        .filter(|_| rng.gen::<f32>() >= p)
        .copied()
        .collect();
    
    if filtered_words.is_empty() {
        // Ensure we don't delete all words
        return words.choose(&mut rng).unwrap_or(&"").to_string();
    }
    
    filtered_words.join(" ")
}

// Random word swap
fn random_word_swap(text: &str, n: usize) -> String {
    let mut words: Vec<&str> = text.split_whitespace().collect();
    let mut rng = rand::thread_rng();
    
    for _ in 0..n.min(words.len().saturating_sub(1)) {
        let idx1 = rng.gen_range(0..words.len());
        let idx2 = rng.gen_range(0..words.len());
        words.swap(idx1, idx2);
    }
    
    words.join(" ")
}

// Example usage
fn text_augmentation_example() {
    let text = "this is an example of text augmentation techniques";
    
    // Apply random word deletion
    let deleted = random_word_deletion(text, 0.2);
    println!("Original: {}", text);
    println!("After deletion: {}", deleted);
    
    // Apply random word swap
    let swapped = random_word_swap(text, 2);
    println!("After swapping: {}", swapped);
}
```

## Building Efficient Data Pipelines

### Dataset and DataLoader Implementation

Creating efficient data pipelines involves implementing dataset and dataloader abstractions:

```rust
use candle_core::{Tensor, Result, Device};
use std::sync::Arc;
use rand::seq::SliceRandom;

// Dataset trait
trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)>;
}

// Simple in-memory dataset
struct InMemoryDataset {
    features: Tensor,
    labels: Tensor,
}

impl InMemoryDataset {
    fn new(features: Tensor, labels: Tensor) -> Self {
        Self { features, labels }
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.features.dim(0).unwrap_or(0)
    }
    
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        let feature = self.features.get(index)?;
        let label = self.labels.get(index)?;
        Ok((feature, label))
    }
}

// DataLoader for batching and shuffling
struct DataLoader {
    dataset: Arc<dyn Dataset>,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_index: usize,
}

impl DataLoader {
    fn new(dataset: Arc<dyn Dataset>, batch_size: usize, shuffle: bool) -> Self {
        let dataset_len = dataset.len();
        let indices: Vec<usize> = (0..dataset_len).collect();
        
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_index: 0,
        }
    }
    
    fn shuffle_indices(&mut self) {
        if self.shuffle {
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }
    
    fn reset(&mut self) {
        self.current_index = 0;
        self.shuffle_indices();
    }
    
    fn next_batch(&mut self) -> Option<Result<(Tensor, Tensor)>> {
        if self.current_index >= self.dataset.len() {
            return None;
        }
        
        let end_idx = (self.current_index + self.batch_size).min(self.dataset.len());
        let batch_indices = &self.indices[self.current_index..end_idx];
        
        // Get individual samples
        let mut features = Vec::with_capacity(batch_indices.len());
        let mut labels = Vec::with_capacity(batch_indices.len());
        
        for &idx in batch_indices {
            match self.dataset.get(idx) {
                Ok((feature, label)) => {
                    features.push(feature);
                    labels.push(label);
                }
                Err(e) => return Some(Err(e)),
            }
        }
        
        // Stack into batches
        let result = match (Tensor::stack(&features, 0), Tensor::stack(&labels, 0)) {
            (Ok(f), Ok(l)) => Ok((f, l)),
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        
        self.current_index = end_idx;
        
        Some(result)
    }
}

// Example usage
fn dataloader_example() -> Result<()> {
    let device = Device::Cpu;
    
    // Create dummy data
    let features = Tensor::rand(0.0, 1.0, (100, 10), &device)?;
    let labels = Tensor::randint(0, 5, &[100], &device)?;
    
    // Create dataset
    let dataset = Arc::new(InMemoryDataset::new(features, labels)) as Arc<dyn Dataset>;
    
    // Create dataloader
    let mut dataloader = DataLoader::new(dataset, 16, true);
    
    // Iterate through batches
    dataloader.reset();
    let mut batch_count = 0;
    
    while let Some(batch_result) = dataloader.next_batch() {
        let (batch_x, batch_y) = batch_result?;
        println!("Batch {}: X shape: {:?}, Y shape: {:?}", 
                 batch_count, batch_x.shape(), batch_y.shape());
        batch_count += 1;
    }
    
    Ok(())
}
```

### Lazy Loading for Large Datasets

For large datasets that don't fit in memory, lazy loading is essential:

```rust
use candle_core::{Tensor, Result, Device};
use std::sync::Arc;
use std::path::Path;

// Lazy-loading image dataset
struct ImageDataset {
    image_paths: Vec<String>,
    labels: Vec<usize>,
    image_size: (u32, u32),
    device: Device,
}

impl ImageDataset {
    fn new(
        image_dir: &str,
        image_paths: Vec<String>,
        labels: Vec<usize>,
        image_size: (u32, u32),
        device: Device,
    ) -> Self {
        Self {
            image_paths: image_paths.iter().map(|p| format!("{}/{}", image_dir, p)).collect(),
            labels,
            image_size,
            device,
        }
    }
}

impl Dataset for ImageDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        // Load image on demand
        let img_tensor = load_image(&self.image_paths[index], self.image_size)?;
        
        // Create label tensor
        let label = Tensor::new(&[self.labels[index]], &self.device)?;
        
        Ok((img_tensor, label))
    }
}

// Example usage
fn lazy_loading_example() -> Result<()> {
    let device = Device::Cpu;
    
    // Sample data
    let image_paths = vec![
        "img1.jpg".to_string(),
        "img2.jpg".to_string(),
        "img3.jpg".to_string(),
    ];
    
    let labels = vec![0, 1, 2];
    
    // Create lazy-loading dataset
    let dataset = Arc::new(ImageDataset::new(
        "data/images",
        image_paths,
        labels,
        (224, 224),
        device.clone(),
    )) as Arc<dyn Dataset>;
    
    // Create dataloader with small batch size to demonstrate lazy loading
    let mut dataloader = DataLoader::new(dataset, 1, false);
    
    // Load first batch
    if let Some(batch_result) = dataloader.next_batch() {
        let (batch_x, batch_y) = batch_result?;
        println!("Loaded batch: X shape: {:?}, Y shape: {:?}", 
                 batch_x.shape(), batch_y.shape());
    }
    
    Ok(())
}
```

## Memory Optimization for Large Datasets

### Memory Mapping

Memory mapping allows working with large files without loading them entirely into memory:

```rust
use candle_core::{Tensor, Result, Device, DType};
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{BufReader, Read};

// Load a large tensor using memory mapping
fn load_large_tensor_mmap(file_path: &str, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let file = File::open(file_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    
    // Create tensor from memory-mapped data
    // Note: This is a simplified example. In practice, you would need to handle
    // data conversion based on the dtype.
    let tensor = match dtype {
        DType::F32 => {
            let data = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr() as *const f32,
                    mmap.len() / std::mem::size_of::<f32>(),
                )
            };
            Tensor::from_vec(data.to_vec(), shape, &Device::Cpu)?
        },
        // Handle other data types...
        _ => return Err(candle_core::Error::Msg("Unsupported dtype".to_string())),
    };
    
    Ok(tensor)
}

// Example usage
fn memory_mapping_example() -> Result<()> {
    // This is a placeholder example
    // In practice, you would have a large binary file containing tensor data
    println!("Memory mapping example (placeholder)");
    
    // let tensor = load_large_tensor_mmap(
    //     "data/large_tensor.bin",
    //     &[10000, 1000],
    //     DType::F32,
    // )?;
    // println!("Loaded tensor shape: {:?}", tensor.shape());
    
    Ok(())
}
```

### Chunked Processing

For datasets too large to process at once, chunked processing is useful:

```rust
use candle_core::{Tensor, Result, Device};

// Process a large dataset in chunks
fn process_large_dataset_in_chunks<F>(
    dataset_size: usize,
    chunk_size: usize,
    mut process_fn: F,
) -> Result<()>
where
    F: FnMut(usize, usize) -> Result<()>,
{
    let num_chunks = (dataset_size + chunk_size - 1) / chunk_size;
    
    for chunk_idx in 0..num_chunks {
        let start_idx = chunk_idx * chunk_size;
        let end_idx = (start_idx + chunk_size).min(dataset_size);
        
        println!("Processing chunk {}/{}: indices {} to {}", 
                 chunk_idx + 1, num_chunks, start_idx, end_idx);
        
        // Process this chunk
        process_fn(start_idx, end_idx)?;
    }
    
    Ok(())
}

// Example usage
fn chunked_processing_example() -> Result<()> {
    let device = Device::Cpu;
    
    // Simulate a large dataset
    let dataset_size = 10000;
    let feature_dim = 100;
    
    // Process in chunks of 1000 samples
    process_large_dataset_in_chunks(dataset_size, 1000, |start_idx, end_idx| {
        // In a real scenario, you would load this chunk from disk
        let chunk_size = end_idx - start_idx;
        let chunk_features = Tensor::rand(0.0, 1.0, (chunk_size, feature_dim), &device)?;
        
        // Perform some computation on the chunk
        let chunk_mean = chunk_features.mean_dim(0, false)?;
        println!("  Chunk mean shape: {:?}", chunk_mean.shape());
        
        Ok(())
    })?;
    
    Ok(())
}
```

## Case Studies

### Case Study 1: MNIST Image Classification

Let's implement a complete data loading and preprocessing pipeline for MNIST:

```rust
use candle_core::{Tensor, Result, Device, DType};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use flate2::read::GzDecoder;
use byteorder::{BigEndian, ReadBytesExt};

// MNIST dataset loader
struct MnistDataset {
    images: Tensor,
    labels: Tensor,
}

impl MnistDataset {
    fn new(images_path: &str, labels_path: &str, device: &Device) -> Result<Self> {
        // Load images
        let images = load_mnist_images(images_path, device)?;
        
        // Load labels
        let labels = load_mnist_labels(labels_path, device)?;
        
        Ok(Self { images, labels })
    }
    
    fn len(&self) -> usize {
        self.images.dim(0).unwrap_or(0)
    }
    
    fn get_batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let batch_images = self.images.index_select(&Tensor::new(indices, self.images.device())?, 0)?;
        let batch_labels = self.labels.index_select(&Tensor::new(indices, self.labels.device())?, 0)?;
        
        Ok((batch_images, batch_labels))
    }
}

// Load MNIST images from file
fn load_mnist_images(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(BufReader::new(file));
    
    // Read header
    let magic_number = decoder.read_u32::<BigEndian>()?;
    if magic_number != 2051 {
        return Err(candle_core::Error::Msg("Invalid MNIST image file".to_string()));
    }
    
    let num_images = decoder.read_u32::<BigEndian>()? as usize;
    let num_rows = decoder.read_u32::<BigEndian>()? as usize;
    let num_cols = decoder.read_u32::<BigEndian>()? as usize;
    
    // Read image data
    let mut buffer = vec![0u8; num_images * num_rows * num_cols];
    decoder.read_exact(&mut buffer)?;
    
    // Convert to f32 and normalize to [0, 1]
    let float_data: Vec<f32> = buffer.iter().map(|&x| x as f32 / 255.0).collect();
    
    // Create tensor with shape [num_images, 1, num_rows, num_cols]
    let tensor = Tensor::from_vec(
        float_data,
        (num_images, 1, num_rows, num_cols),
        device,
    )?;
    
    Ok(tensor)
}

// Load MNIST labels from file
fn load_mnist_labels(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(BufReader::new(file));
    
    // Read header
    let magic_number = decoder.read_u32::<BigEndian>()?;
    if magic_number != 2049 {
        return Err(candle_core::Error::Msg("Invalid MNIST label file".to_string()));
    }
    
    let num_items = decoder.read_u32::<BigEndian>()? as usize;
    
    // Read label data
    let mut buffer = vec![0u8; num_items];
    decoder.read_exact(&mut buffer)?;
    
    // Convert to u32
    let labels: Vec<u32> = buffer.iter().map(|&x| x as u32).collect();
    
    // Create tensor
    let tensor = Tensor::new(&labels, device)?;
    
    Ok(tensor)
}

// MNIST data preprocessing
fn preprocess_mnist(images: &Tensor) -> Result<Tensor> {
    // Normalize to mean=0, std=1
    let mean = images.mean_all()?;
    let std = images.std_all()?;
    
    images.sub(&mean)?.div(&std)
}

// Example usage
fn mnist_example() -> Result<()> {
    let device = Device::Cpu;
    
    // Load MNIST dataset
    let mnist = MnistDataset::new(
        "data/mnist/train-images-idx3-ubyte.gz",
        "data/mnist/train-labels-idx1-ubyte.gz",
        &device,
    )?;
    
    println!("Loaded MNIST dataset with {} samples", mnist.len());
    
    // Create random indices for a batch
    let mut rng = rand::thread_rng();
    let indices: Vec<usize> = (0..64).map(|_| rng.gen_range(0..mnist.len())).collect();
    
    // Get and preprocess a batch
    let (batch_images, batch_labels) = mnist.get_batch(&indices)?;
    let preprocessed_images = preprocess_mnist(&batch_images)?;
    
    println!("Batch images shape: {:?}", batch_images.shape());
    println!("Batch labels shape: {:?}", batch_labels.shape());
    println!("Preprocessed images shape: {:?}", preprocessed_images.shape());
    
    Ok(())
}
```

### Case Study 2: Text Classification with Embeddings

Let's implement a data pipeline for text classification using word embeddings:

```rust
use candle_core::{Tensor, Result, Device, DType};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// Text classification dataset
struct TextClassificationDataset {
    texts: Vec<String>,
    labels: Vec<usize>,
    tokenizer: SimpleTokenizer,
    max_length: usize,
    device: Device,
}

impl TextClassificationDataset {
    fn new(
        texts: Vec<String>,
        labels: Vec<usize>,
        max_length: usize,
        device: Device,
    ) -> Self {
        let mut tokenizer = SimpleTokenizer::new();
        tokenizer.build_vocab(&texts);
        
        Self {
            texts,
            labels,
            tokenizer,
            max_length,
            device,
        }
    }
    
    fn len(&self) -> usize {
        self.texts.len()
    }
    
    fn get_batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let mut batch_tokens = Vec::with_capacity(indices.len());
        let mut batch_labels = Vec::with_capacity(indices.len());
        
        for &idx in indices {
            let tokens = self.tokenizer.encode(&self.texts[idx], self.max_length);
            batch_tokens.push(tokens);
            batch_labels.push(self.labels[idx]);
        }
        
        let token_tensor = Tensor::new(&batch_tokens, &self.device)?;
        let label_tensor = Tensor::new(&batch_labels, &self.device)?;
        
        Ok((token_tensor, label_tensor))
    }
    
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

// Load GloVe embeddings
fn load_glove_embeddings(
    path: &str,
    vocab: &HashMap<String, usize>,
    embedding_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    // Initialize embedding matrix with random values
    let vocab_size = vocab.len();
    let mut embeddings = vec![0.0; vocab_size * embedding_dim];
    
    // Track which words were found in the GloVe file
    let mut found_words = vec![false; vocab_size];
    
    // Parse GloVe file
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() != embedding_dim + 1 {
            continue;
        }
        
        let word = parts[0];
        if let Some(&idx) = vocab.get(word) {
            found_words[idx] = true;
            
            // Parse embedding values
            for i in 0..embedding_dim {
                if let Ok(value) = parts[i + 1].parse::<f32>() {
                    embeddings[idx * embedding_dim + i] = value;
                }
            }
        }
    }
    
    // Initialize random embeddings for words not found in GloVe
    let mut rng = rand::thread_rng();
    for i in 0..vocab_size {
        if !found_words[i] {
            for j in 0..embedding_dim {
                embeddings[i * embedding_dim + j] = rng.gen::<f32>() * 0.1 - 0.05;
            }
        }
    }
    
    // Create embedding tensor
    Tensor::from_vec(embeddings, (vocab_size, embedding_dim), device)
}

// Example usage
fn text_classification_example() -> Result<()> {
    let device = Device::Cpu;
    
    // Sample data
    let texts = vec![
        "this movie was great".to_string(),
        "the acting was terrible".to_string(),
        "i loved the cinematography".to_string(),
        "the plot made no sense".to_string(),
        "the soundtrack was amazing".to_string(),
    ];
    
    let labels = vec![1, 0, 1, 0, 1]; // 1 for positive, 0 for negative
    
    // Create dataset
    let dataset = TextClassificationDataset::new(
        texts,
        labels,
        20, // max_length
        device.clone(),
    );
    
    println!("Dataset size: {}", dataset.len());
    println!("Vocabulary size: {}", dataset.vocab_size());
    
    // Get a batch
    let indices = vec![0, 2, 4];
    let (batch_tokens, batch_labels) = dataset.get_batch(&indices)?;
    
    println!("Batch tokens shape: {:?}", batch_tokens.shape());
    println!("Batch labels shape: {:?}", batch_labels.shape());
    
    // Load embeddings (placeholder - in practice you would load from a file)
    // let embedding_dim = 100;
    // let embeddings = load_glove_embeddings(
    //     "data/glove.6B.100d.txt",
    //     &dataset.tokenizer.vocab,
    //     embedding_dim,
    //     &device,
    // )?;
    
    // println!("Embeddings shape: {:?}", embeddings.shape());
    
    Ok(())
}
```

## Best Practices and Common Pitfalls

### Best Practices

1. **Understand Your Data**: Explore and visualize your data before preprocessing to identify patterns, outliers, and potential issues.

2. **Normalize Appropriately**: Choose the right normalization technique for your data and model. For example, images often use [0, 1] or [-1, 1] normalization, while tabular data might benefit from standardization.

3. **Handle Missing Values**: Develop a strategy for missing values that makes sense for your data, such as imputation with mean/median or using a special indicator.

4. **Split Data Properly**: Ensure your train/validation/test splits are representative and don't leak information between sets.

5. **Use Data Augmentation Wisely**: Apply augmentations that make sense for your domain and task. Not all augmentations are appropriate for all data types.

6. **Batch Size Considerations**: Choose a batch size that balances between computational efficiency and model convergence. Larger batches may be faster but can affect generalization.

7. **Efficient Data Loading**: Use lazy loading, memory mapping, or chunked processing for large datasets to avoid memory issues.

8. **Caching Preprocessed Data**: Consider caching preprocessed data to disk to avoid redundant computation during training.

9. **Reproducibility**: Set random seeds for shuffling and augmentation to ensure reproducible results.

10. **Validate Your Pipeline**: Test your data pipeline thoroughly to ensure it's working as expected before training models.

### Common Pitfalls

1. **Data Leakage**: Accidentally including information from the test set in the training process, leading to overly optimistic performance estimates.

2. **Inconsistent Preprocessing**: Applying different preprocessing to training and test data, causing a distribution shift.

3. **Inappropriate Normalization**: Using the wrong normalization technique for your data or model architecture.

4. **Memory Issues**: Loading too much data into memory at once, causing out-of-memory errors.

5. **Ignoring Class Imbalance**: Not addressing class imbalance in classification tasks, leading to biased models.

6. **Over-augmentation**: Applying too aggressive augmentations that distort the data beyond realistic variations.

7. **Slow Data Loading**: Creating bottlenecks in the training process due to inefficient data loading.

8. **Incorrect Tensor Shapes**: Not properly reshaping tensors to match the expected input format of your model.

9. **Forgetting to Shuffle**: Training on data in a fixed order, which can bias the learning process.

10. **Ignoring Data Quality**: Not cleaning or validating your data, leading to garbage-in-garbage-out problems.

## Conclusion

Data loading and preprocessing are foundational steps in the machine learning pipeline that can significantly impact model performance. In this chapter, we've explored:

- The importance of data preprocessing in machine learning
- Candle's approach to handling different data formats
- Techniques for loading and preprocessing various data types
- Creating and manipulating tensors from raw data
- Common preprocessing techniques and data augmentation strategies
- Building efficient data pipelines with datasets and dataloaders
- Memory optimization for large datasets
- Practical examples and case studies
- Best practices and common pitfalls

By mastering these techniques, you'll be able to build efficient, effective data pipelines that prepare your data optimally for training models with Candle. Remember that good data preparation is often the difference between a model that fails to learn and one that achieves state-of-the-art performance.

## Further Reading

- "Feature Engineering for Machine Learning" by Alice Zheng and Amanda Casari
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Data Cleaning: A Practical Guide" by Megan Squire
- "Practical Data Science with R" by Nina Zumel and John Mount
- "Python Data Science Handbook" by Jake VanderPlas
- "Efficient Processing of Deep Neural Networks" by Vivienne Sze et al.