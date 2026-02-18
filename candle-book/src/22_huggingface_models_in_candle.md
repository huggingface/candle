# 24. Pretrained Models

## Introduction to Hugging Face Models

Hugging Face has become the central hub for sharing and discovering machine learning models, particularly in the field of natural language processing (NLP) and increasingly in computer vision and audio processing. The Hugging Face Hub hosts thousands of pretrained models that can be used for a wide variety of tasks.

Candle, as a Rust-native deep learning framework, provides the ability to use these pretrained models efficiently. This chapter will guide you through the process of using Hugging Face models with Candle, from understanding what models are available to running them on your own computer.

## Available Pretrained Models

Hugging Face hosts a wide variety of models that can be used with Candle. These models span different architectures and are designed for various tasks:

### Language Models

1. **GPT Family**
   - GPT-2: A smaller version of the GPT architecture (124M to 1.5B parameters)
   - GPT-Neo/GPT-J: Open-source alternatives to GPT-3 (125M to 6B parameters)
   - LLaMA and LLaMA 2: Meta's Large Language Models (7B to 70B parameters)
   - Mistral: Efficient language models with strong performance (7B parameters)

2. **BERT Family**
   - BERT: Bidirectional Encoder Representations from Transformers (110M to 340M parameters)
   - RoBERTa: Optimized version of BERT (125M to 355M parameters)
   - DistilBERT: Distilled version of BERT (66M parameters)

3. **T5 Family**
   - T5: Text-to-Text Transfer Transformer (60M to 11B parameters)
   - FLAN-T5: Instruction-tuned version of T5 (80M to 11B parameters)

### Vision Models

1. **Image Classification**
   - ResNet: Residual Networks (11M to 60M parameters)
   - ViT: Vision Transformer (86M to 632M parameters)
   - CLIP: Contrastive Language-Image Pre-training (150M to 400M parameters)

2. **Image Generation**
   - Stable Diffusion: Text-to-image diffusion models (860M to 1.5B parameters)
   - DALL-E: Text-to-image generation models

### Multimodal Models

1. **Vision-Language Models**
   - CLIP: Connects images and text (150M to 400M parameters)
   - LLaVA: Language-and-Vision Assistant (7B to 13B parameters)

## Model Sizes and Resource Requirements

Understanding the size and resource requirements of models is crucial for running them efficiently on your hardware:

### Model Size Categories

1. **Small Models (< 500M parameters)**
   - Memory requirement: 1-2 GB RAM
   - Storage: 0.5-2 GB
   - Can run on CPU or modest GPUs
   - Examples: DistilBERT, BERT-base, smaller ResNets

2. **Medium Models (500M - 3B parameters)**
   - Memory requirement: 4-8 GB RAM
   - Storage: 2-10 GB
   - Benefit from GPU acceleration
   - Examples: GPT-2 Large, RoBERTa Large, CLIP

3. **Large Models (3B - 10B parameters)**
   - Memory requirement: 8-16 GB RAM
   - Storage: 10-30 GB
   - Require GPU with 8+ GB VRAM for efficient inference
   - Examples: LLaMA-7B, Mistral-7B, FLAN-T5 Large

4. **Very Large Models (10B+ parameters)**
   - Memory requirement: 16+ GB RAM
   - Storage: 30+ GB
   - Require high-end GPUs or multi-GPU setups
   - Examples: LLaMA-13B, LLaMA-70B, FLAN-T5 XXL

### Quantization Options

To reduce memory requirements, Candle supports various quantization techniques:

1. **FP16 (Half Precision)**
   - Reduces memory usage by ~50% compared to FP32
   - Minimal impact on model quality
   - Supported by most modern GPUs

2. **INT8 Quantization**
   - Reduces memory usage by ~75% compared to FP32
   - Some impact on model quality, but often acceptable
   - Enables running larger models on consumer hardware

3. **INT4 Quantization**
   - Reduces memory usage by ~87.5% compared to FP32
   - More noticeable impact on model quality
   - Allows running very large models on consumer hardware

## Running Models on Your Computer

This section provides a step-by-step guide to running Hugging Face models with Candle on your local machine.

### Prerequisites

Before you begin, ensure you have:

1. Rust installed (latest stable version recommended)
2. Candle dependencies installed:
   - For GPU support: CUDA or Metal development tools
   - For CPU-only: No additional dependencies

3. Add Candle to your project:

```toml
[dependencies]
candle-core = "0.9.1"
candle-nn = "0.9.1"
candle-transformers = "0.9.1"  # For transformer-based models
```

### Downloading Model Weights

Hugging Face models can be downloaded directly from the Hub:

```rust
use std::path::Path;
use candle_core::utils::download;

fn download_model(model_id: &str, filename: &str, dest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("https://huggingface.co/{}/resolve/main/{}", model_id, filename);

    if !dest_path.exists() {
        println!("Downloading {} to {:?}", url, dest_path);
        download(&url, dest_path)?;
    } else {
        println!("Model already downloaded at {:?}", dest_path);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create models directory if it doesn't exist
    std::fs::create_dir_all("models")?;

    // Download model weights
    let model_id = "bert-base-uncased";
    let filename = "model.safetensors";
    let dest_path = Path::new("models").join(model_id).join(filename);

    // Create parent directory
    std::fs::create_dir_all(dest_path.parent().unwrap())?;

    // Download the model
    download_model(model_id, filename, &dest_path)?;

    Ok(())
}
```

### Loading a Pretrained Model

Once you've downloaded the model weights, you can load them into Candle:

```rust
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;

// Example for loading a BERT model
fn load_bert_model(model_path: &Path, device: &Device) -> Result<BertModel, Box<dyn std::error::Error>> {
    // Load the model configuration
    let config_path = model_path.parent().unwrap().join("config.json");
    let config_str = std::fs::read_to_string(config_path)?;
    let config: BertConfig = serde_json::from_str(&config_str)?;

    // Load the model weights
    let mut varmap = VarMap::new();
    varmap.load(model_path)?;

    // Create the model
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = BertModel::new(&config, vb)?;

    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Select device (CPU or GPU)
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Load the model
    let model_path = Path::new("models/bert-base-uncased/model.safetensors");
    let model = load_bert_model(model_path, &device)?;

    println!("Model loaded successfully!");

    Ok(())
}
```

### Running Inference

After loading the model, you can use it for inference:

```rust
use candle_core::{Tensor, Device};
use candle_nn::Module;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load tokenizer and model (from previous example)
    let device = Device::cuda_if_available(0)?;
    let model_path = Path::new("models/bert-base-uncased/model.safetensors");
    let model = load_bert_model(model_path, &device)?;
    let tokenizer = BertTokenizer::from_file(model_path.parent().unwrap().join("tokenizer.json"))?;

    // Prepare input
    let text = "Hello, world!";
    let tokens = tokenizer.encode(text)?;

    // Convert tokens to tensor
    let input_ids = Tensor::new(&tokens.ids, &device)?;
    let attention_mask = Tensor::new(&tokens.attention_mask, &device)?;

    // Run inference
    let output = model.forward(&input_ids, &attention_mask, None)?;

    // Process output
    println!("Model output shape: {:?}", output.hidden_states.shape());

    Ok(())
}
```

## Example: Text Generation with GPT-2

Let's walk through a complete example of using GPT-2 for text generation:

```rust
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::gpt2::{Config, GPT2Model, GPT2Tokenizer};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Load model and tokenizer
    let model_id = "gpt2";
    let model_path = Path::new("models").join(model_id);

    // Ensure model directory exists
    std::fs::create_dir_all(&model_path)?;

    // Download model if needed
    let weights_path = model_path.join("model.safetensors");
    if !weights_path.exists() {
        download_model(model_id, "model.safetensors", &weights_path)?;
    }

    // Load tokenizer
    let tokenizer_path = model_path.join("tokenizer.json");
    if !tokenizer_path.exists() {
        download_model(model_id, "tokenizer.json", &tokenizer_path)?;
    }
    let tokenizer = GPT2Tokenizer::from_file(&tokenizer_path)?;

    // Load config
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        download_model(model_id, "config.json", &config_path)?;
    }
    let config_str = std::fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;

    // Load model weights
    let mut varmap = VarMap::new();
    varmap.load(&weights_path)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT2Model::new(&config, vb)?;

    // Generate text
    let prompt = "Once upon a time";
    let tokens = tokenizer.encode(prompt)?;
    let mut input_ids = Tensor::new(&tokens, &device)?;

    // Generate 50 new tokens
    for _ in 0..50 {
        // Get model prediction
        let logits = model.forward(&input_ids, None)?;

        // Get the last token's logits
        let logits = logits.get(logits.dim(0)? - 1)?;

        // Sample from the logits
        let next_token = sample_token(&logits, 0.8)?;

        // Append the new token to input_ids
        let next_token_tensor = Tensor::new(&[next_token], &device)?;
        input_ids = Tensor::cat(&[input_ids, next_token_tensor], 0)?;

        // Break if we generate an EOS token
        if next_token == tokenizer.eos_token_id() {
            break;
        }
    }

    // Decode the generated tokens
    let output_text = tokenizer.decode(&input_ids.to_vec1::<u32>()?)?;
    println!("Generated text: {}", output_text);

    Ok(())
}

// Helper function to sample a token from logits with temperature
fn sample_token(logits: &Tensor, temperature: f32) -> Result<u32, Box<dyn std::error::Error>> {
    // Apply temperature
    let logits = logits.div_scalar(temperature)?;

    // Apply softmax to get probabilities
    let probs = candle_nn::ops::softmax(logits, 0)?;

    // Sample from the distribution
    let probs_vec = probs.to_vec1::<f32>()?;
    let distr = rand::distributions::WeightedIndex::new(&probs_vec)?;
    let mut rng = rand::thread_rng();
    let token_id = distr.sample(&mut rng) as u32;

    Ok(token_id)
}
```

## Example: Image Classification with ResNet

Here's an example of using a pretrained ResNet model for image classification:

```rust
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::resnet::{ResNet50Config, ResNet};
use image::{self, GenericImageView};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Load model
    let model_id = "microsoft/resnet-50";
    let model_path = Path::new("models").join("resnet-50");

    // Ensure model directory exists
    std::fs::create_dir_all(&model_path)?;

    // Download model if needed
    let weights_path = model_path.join("model.safetensors");
    if !weights_path.exists() {
        download_model(model_id, "model.safetensors", &weights_path)?;
    }

    // Load config
    let config = ResNet50Config::default();

    // Load model weights
    let mut varmap = VarMap::new();
    varmap.load(&weights_path)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = ResNet::new(&config, vb)?;

    // Load and preprocess image
    let img_path = "path/to/your/image.jpg";
    let img = image::open(img_path)?;

    // Resize to 224x224
    let img = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);

    // Convert to RGB tensor and normalize
    let mut tensor_data = Vec::with_capacity(3 * 224 * 224);
    for pixel in img.pixels() {
        let rgb = pixel.2.0;
        // Normalize using ImageNet mean and std
        tensor_data.push((rgb[0] as f32 / 255.0 - 0.485) / 0.229);
        tensor_data.push((rgb[1] as f32 / 255.0 - 0.456) / 0.224);
        tensor_data.push((rgb[2] as f32 / 255.0 - 0.406) / 0.225);
    }

    // Create input tensor
    let input = Tensor::from_vec(tensor_data, (1, 3, 224, 224), &device)?;

    // Run inference
    let output = model.forward(&input)?;

    // Get top 5 predictions
    let (top5_values, top5_indices) = output.topk(5, 1, true, true)?;

    // Load class labels
    let labels_path = model_path.join("imagenet_classes.txt");
    if !labels_path.exists() {
        download_model(model_id, "imagenet_classes.txt", &labels_path)?;
    }
    let labels = std::fs::read_to_string(labels_path)?;
    let labels: Vec<&str> = labels.lines().collect();

    // Print predictions
    let values = top5_values.to_vec1::<f32>()?;
    let indices = top5_indices.to_vec1::<u32>()?;

    println!("Top 5 predictions:");
    for i in 0..5 {
        let idx = indices[i] as usize;
        let confidence = values[i];
        println!("{}: {} - {:.2}%", i+1, labels[idx], confidence * 100.0);
    }

    Ok(())
}
```

## Best Practices and Optimization Tips

To get the most out of pretrained models in Candle, consider these best practices:

### Memory Optimization

1. **Use Quantization**: For large models, use quantization to reduce memory requirements:

```rust
// Load with INT8 quantization
let vb = VarBuilder::from_varmap_quantized(&varmap, DType::I8, &device);
```

2. **Batch Processing**: Process multiple inputs in batches to maximize throughput:

```rust
// Create a batch of inputs
let batch_input = Tensor::cat(&[input1, input2, input3], 0)?;
```

3. **Memory Mapping**: For very large models, use memory mapping to avoid loading the entire model into RAM:

```rust
// Memory-map the model weights
let vb = VarBuilder::from_mmaped_safetensors(&[model_path], DType::F16, &device)?;
```

### Performance Optimization

1. **Use GPU Acceleration**: When available, use GPU for faster inference:

```rust
let device = Device::cuda_if_available(0)?;
```

2. **Mixed Precision**: Use FP16 for faster computation with minimal accuracy loss:

```rust
let vb = VarBuilder::from_varmap(&varmap, DType::F16, &device);
```

3. **Caching**: Cache intermediate results for repeated operations:

```rust
// Cache the key-value pairs in transformer models
let cached_kv = Some(model.cache_key_values(seq_len, num_layers)?);
```

### Model Selection

1. **Start Small**: Begin with smaller models to test your pipeline before scaling up.

2. **Task-Specific Models**: Choose models fine-tuned for your specific task when available.

3. **Quantized Models**: Look for models that have been specifically quantized for efficiency.

## Conclusion

In this chapter, we've explored how to use pretrained Hugging Face models with Candle. We've covered the types of models available, their resource requirements, and provided practical examples for loading and running these models on your own computer.

Pretrained models offer a powerful way to leverage state-of-the-art AI capabilities without the need for extensive training resources. By combining the efficiency of Candle with the vast ecosystem of Hugging Face models, you can build sophisticated AI applications in Rust that are both performant and production-ready.

In the next chapter, we'll explore how to fine-tune these pretrained models on your own data to adapt them for specific tasks and domains.
