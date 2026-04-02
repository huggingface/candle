# 16. Tokenizers

## Introduction

Tokenization is a fundamental process in natural language processing (NLP) and is especially critical for transformer-based models like GPT (Generative Pre-trained Transformer). In this chapter, we'll explore what tokenizers are, the different types available, how to use Hugging Face tokenizers, and how to implement a tokenizer from scratch in Rust using the Candle library.

## What is a Tokenizer?

A tokenizer is a component that splits text into smaller units called tokens. These tokens serve as the input to neural language models. The tokenization process transforms human-readable text into a numerical format that models can process.

### The Role of Tokenizers in NLP

Tokenizers bridge the gap between human language and machine understanding by:

1. **Breaking down text**: Converting sentences or documents into smaller units
2. **Creating a vocabulary**: Establishing a fixed set of tokens the model recognizes
3. **Encoding/decoding**: Converting between text and numerical representations
4. **Handling out-of-vocabulary words**: Managing words not seen during training

### Why Tokenization Matters

The choice of tokenization strategy significantly impacts model performance:

- It determines the granularity of language understanding
- It affects the model's vocabulary size and memory requirements
- It influences how well the model handles rare words or new terms
- It can impact the model's ability to understand context and semantics

## Types of Tokenizers

There are several approaches to tokenization, each with its own advantages and limitations:

### Word-Based Tokenizers

Word-based tokenizers split text at word boundaries, typically using spaces and punctuation.

**Advantages:**
- Intuitive and straightforward
- Preserves word meanings

**Limitations:**
- Large vocabulary size (potentially millions of tokens)
- Poor handling of out-of-vocabulary words
- Language-dependent (doesn't work well for languages without clear word boundaries)

**Example:**
```
"The quick brown fox jumps." → ["The", "quick", "brown", "fox", "jumps", "."]
```

### Character-Based Tokenizers

Character-based tokenizers treat each character as a separate token.

**Advantages:**
- Tiny vocabulary size
- No out-of-vocabulary issues

**Limitations:**
- Very long sequences
- Loss of word-level semantics
- Inefficient for capturing higher-level patterns

**Example:**
```
"Hello" → ["H", "e", "l", "l", "o"]
```

### Subword Tokenizers

Subword tokenizers strike a balance between word and character tokenization by breaking words into meaningful subunits.

**Advantages:**
- Manageable vocabulary size
- Better handling of rare words and morphology
- More efficient representation

**Limitations:**
- More complex implementation
- May split words in unintuitive ways

**Popular Subword Tokenization Algorithms:**

1. **Byte-Pair Encoding (BPE)**
   - Used by GPT models
   - Starts with characters and iteratively merges most frequent pairs
   
2. **WordPiece**
   - Used by BERT
   - Similar to BPE but uses likelihood rather than frequency for merges
   
3. **Unigram**
   - Used by some T5 models
   - Starts with a large vocabulary and iteratively removes tokens to maximize likelihood
   
4. **SentencePiece**
   - Language-agnostic approach that treats the text as a sequence of Unicode characters
   - Can implement BPE or Unigram algorithms

**Example (BPE):**
```
"unhappiness" → ["un", "happiness"]
```

## Hugging Face Tokenizers

The Hugging Face `tokenizers` library provides fast, state-of-the-art implementations of various tokenization algorithms.

### Key Features of Hugging Face Tokenizers

1. **Performance**: Implemented in Rust for speed
2. **Flexibility**: Supports multiple tokenization algorithms
3. **Pre-trained**: Provides tokenizers trained on large corpora
4. **Pipeline design**: Modular components for pre-processing, tokenization, and post-processing

### Using Hugging Face Tokenizers in Rust

While the original `tokenizers` library is implemented in Rust, it's most commonly used via Python. However, we can use it directly in Rust or through Candle's integration.

Here's how you might use a pre-trained tokenizer with Candle:

```rust
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config};
use candle_transformers::tokenizers::{Tokenizer, TokenizerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a pre-trained tokenizer
    let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;
    
    // Tokenize some text
    let encoding = tokenizer.encode("Hello, world!", true)?;
    
    // Get the token IDs
    let token_ids = encoding.get_ids();
    println!("Token IDs: {:?}", token_ids);
    
    // Convert to a tensor for model input
    let device = Device::cuda_if_available(0)?;
    let input_ids = Tensor::new(&token_ids, &device)?;
    
    // Use with a model
    // ...
    
    Ok(())
}
```

## Writing a Tokenizer from Scratch

Now, let's implement a simple Byte-Pair Encoding (BPE) tokenizer from scratch in Rust. BPE is the algorithm used by GPT models and provides a good balance between vocabulary size and token meaningfulness.

### Step 1: Define the Tokenizer Structure

```rust
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

struct BPETokenizer {
    // Vocabulary: mapping from token to ID
    vocab: HashMap<String, usize>,
    // Reverse mapping: ID to token
    id_to_token: HashMap<usize, String>,
    // Merges: pairs of tokens that should be merged
    merges: HashMap<(String, String), usize>,
    // Special tokens
    unk_token: String,
    unk_token_id: usize,
}

impl BPETokenizer {
    fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        
        // Add special tokens
        let unk_token = "<unk>".to_string();
        let unk_token_id = 0;
        
        vocab.insert(unk_token.clone(), unk_token_id);
        id_to_token.insert(unk_token_id, unk_token.clone());
        
        BPETokenizer {
            vocab,
            id_to_token,
            merges: HashMap::new(),
            unk_token,
            unk_token_id,
        }
    }
}
```

### Step 2: Implement Training Logic

```rust
impl BPETokenizer {
    // ... previous code ...
    
    fn train(&mut self, texts: &[String], vocab_size: usize) -> Result<(), Box<dyn std::error::Error>> {
        // Start with character-level tokens
        let mut vocab: HashSet<char> = HashSet::new();
        for text in texts {
            for c in text.chars() {
                vocab.insert(c);
            }
        }
        
        // Initialize vocabulary with characters
        let mut token_id = self.vocab.len();
        for c in vocab {
            let token = c.to_string();
            if !self.vocab.contains_key(&token) {
                self.vocab.insert(token.clone(), token_id);
                self.id_to_token.insert(token_id, token);
                token_id += 1;
            }
        }
        
        // Tokenize the corpus into characters
        let mut tokenized_texts: Vec<Vec<String>> = texts
            .iter()
            .map(|text| text.chars().map(|c| c.to_string()).collect())
            .collect();
        
        // Perform BPE training until we reach the desired vocabulary size
        while self.vocab.len() < vocab_size {
            // Count pair frequencies
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            
            for tokens in &tokenized_texts {
                for i in 0..tokens.len() - 1 {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }
            
            // Find the most frequent pair
            if let Some((best_pair, _count)) = pair_counts.iter().max_by_key(|&(_, count)| count) {
                let (first, second) = best_pair;
                let new_token = format!("{}{}", first, second);
                
                // Add the new token to the vocabulary
                if !self.vocab.contains_key(&new_token) {
                    self.vocab.insert(new_token.clone(), token_id);
                    self.id_to_token.insert(token_id, new_token.clone());
                    token_id += 1;
                }
                
                // Record the merge
                self.merges.insert((first.clone(), second.clone()), self.vocab[&new_token]);
                
                // Apply the merge to all tokenized texts
                for tokens in &mut tokenized_texts {
                    let mut i = 0;
                    while i < tokens.len() - 1 {
                        if tokens[i] == *first && tokens[i + 1] == *second {
                            tokens[i] = new_token.clone();
                            tokens.remove(i + 1);
                        } else {
                            i += 1;
                        }
                    }
                }
            } else {
                // No more pairs to merge
                break;
            }
        }
        
        Ok(())
    }
}
```

### Step 3: Implement Tokenization and Encoding

```rust
impl BPETokenizer {
    // ... previous code ...
    
    fn tokenize(&self, text: &str) -> Vec<String> {
        // Start with character-level tokenization
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        
        // Apply merges iteratively
        let mut i = 0;
        while i < tokens.len() - 1 {
            let pair = (tokens[i].clone(), tokens[i + 1].clone());
            
            if let Some(&_) = self.merges.get(&pair) {
                tokens[i] = format!("{}{}", pair.0, pair.1);
                tokens.remove(i + 1);
            } else {
                i += 1;
            }
        }
        
        tokens
    }
    
    fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenize(text);
        
        // Convert tokens to IDs
        tokens
            .iter()
            .map(|token| *self.vocab.get(token).unwrap_or(&self.unk_token_id))
            .collect()
    }
    
    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| self.id_to_token.get(&id).unwrap_or(&self.unk_token).clone())
            .collect::<Vec<String>>()
            .join("")
    }
}
```

### Step 4: Implement Save and Load Functions

```rust
impl BPETokenizer {
    // ... previous code ...
    
    fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        
        // Save vocabulary
        writeln!(file, "# Vocabulary")?;
        for (token, id) in &self.vocab {
            writeln!(file, "{}\t{}", token, id)?;
        }
        
        // Save merges
        writeln!(file, "# Merges")?;
        for ((first, second), _) in &self.merges {
            writeln!(file, "{} {}", first, second)?;
        }
        
        Ok(())
    }
    
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut tokenizer = BPETokenizer::new();
        let mut section = "";
        
        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') {
                section = line.trim_start_matches('#').trim();
                continue;
            }
            
            if line.trim().is_empty() {
                continue;
            }
            
            match section {
                "Vocabulary" => {
                    let parts: Vec<&str> = line.split('\t').collect();
                    if parts.len() == 2 {
                        let token = parts[0].to_string();
                        let id = parts[1].parse::<usize>()?;
                        tokenizer.vocab.insert(token.clone(), id);
                        tokenizer.id_to_token.insert(id, token);
                    }
                }
                "Merges" => {
                    let parts: Vec<&str> = line.split(' ').collect();
                    if parts.len() == 2 {
                        let first = parts[0].to_string();
                        let second = parts[1].to_string();
                        let merged = format!("{}{}", first, second);
                        if let Some(&id) = tokenizer.vocab.get(&merged) {
                            tokenizer.merges.insert((first, second), id);
                        }
                    }
                }
                _ => {}
            }
        }
        
        Ok(tokenizer)
    }
}
```

### Step 5: Example Usage

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new tokenizer
    let mut tokenizer = BPETokenizer::new();
    
    // Sample training data
    let texts = vec![
        "Hello world!".to_string(),
        "How are you doing today?".to_string(),
        "Natural language processing is fascinating.".to_string(),
        "Tokenization is a fundamental step in NLP.".to_string(),
    ];
    
    // Train the tokenizer
    tokenizer.train(&texts, 100)?;
    
    // Save the tokenizer
    tokenizer.save(Path::new("my_tokenizer.txt"))?;
    
    // Load the tokenizer
    let loaded_tokenizer = BPETokenizer::load(Path::new("my_tokenizer.txt"))?;
    
    // Tokenize and encode a text
    let text = "Hello, how are you?";
    let tokens = loaded_tokenizer.tokenize(text);
    let ids = loaded_tokenizer.encode(text);
    
    println!("Text: {}", text);
    println!("Tokens: {:?}", tokens);
    println!("IDs: {:?}", ids);
    
    // Decode back to text
    let decoded = loaded_tokenizer.decode(&ids);
    println!("Decoded: {}", decoded);
    
    Ok(())
}
```

## Integrating with Candle

To use our custom tokenizer with Candle models, we need to ensure it can produce tensors in the format expected by the models:

```rust
use candle_core::{Device, Tensor};

impl BPETokenizer {
    // ... previous code ...
    
    fn encode_for_model(&self, text: &str, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
        let ids = self.encode(text);
        Tensor::new(&ids, device).map_err(|e| e.into())
    }
    
    fn batch_encode_for_model(&self, texts: &[&str], device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
        let batch: Vec<Vec<usize>> = texts.iter().map(|&text| self.encode(text)).collect();
        
        // Find the maximum sequence length
        let max_len = batch.iter().map(|seq| seq.len()).max().unwrap_or(0);
        
        // Pad sequences to the same length
        let padded_batch: Vec<Vec<usize>> = batch
            .into_iter()
            .map(|mut seq| {
                seq.resize(max_len, self.unk_token_id);
                seq
            })
            .collect();
        
        // Convert to a 2D tensor
        let flat: Vec<usize> = padded_batch.into_iter().flatten().collect();
        let batch_size = texts.len();
        
        Tensor::new(&flat, device)?
            .reshape(&[batch_size as i64, max_len as i64])
            .map_err(|e| e.into())
    }
}
```

## Conclusion

Tokenizers are a critical component in the NLP pipeline, especially for transformer-based models like GPT. In this chapter, we've explored:

1. What tokenizers are and why they're important
2. Different types of tokenizers and their trade-offs
3. How to use Hugging Face tokenizers with Candle
4. How to implement a BPE tokenizer from scratch in Rust

In the next chapters of our "Build Your Own GPT" series, we'll explore token embeddings, positional embeddings, transformer architectures, and attention mechanisms - all essential components for building a complete GPT-style model.

## Further Reading

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- [BPE Original Paper: "Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909)
- [The Illustrated GPT-2: Visualizing Transformer Language Models](https://jalammar.github.io/illustrated-gpt2/)
- [Tokenizers: How machines read](https://blog.floydhub.com/tokenization-nlp/)