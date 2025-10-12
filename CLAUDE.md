# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Candle is a minimalist ML framework for Rust focused on performance and ease of use. It provides GPU support via CUDA and Metal, WASM capabilities for browser deployment, and includes implementations of many popular ML models.

## Architecture

The project is structured as a Rust workspace with multiple crates:

- **candle-core**: Core tensor operations, devices (CPU/CUDA/Metal), and fundamental data structures
- **candle-nn**: Neural network building blocks (layers, activations, optimizers)
- **candle-transformers**: Transformer model implementations and utilities  
- **candle-examples**: Comprehensive examples for various ML models (LLMs, vision, audio)
- **candle-datasets**: Data loading and preprocessing utilities
- **candle-kernels**: Custom CUDA kernels for optimized operations
- **candle-metal-kernels**: Metal compute shaders for Apple Silicon
- **candle-flash-attn**: Flash attention v2 implementation
- **candle-onnx**: ONNX model support
- **candle-wasm-examples**: WebAssembly demo applications
- **candle-pyo3**: Python bindings

## Common Commands

### Building and Testing
- `cargo test` - Run all tests
- `cargo build --release` - Build release version
- `cargo run --example <name> --release` - Run specific example
- `make test` - Run tests via Makefile
- `make clean` - Clean build artifacts
- `make clean-ptx` - Clean CUDA PTX files and rebuild kernels

### GPU Support
- Add `--features cuda` for CUDA support
- Add `--features cudnn` for cuDNN acceleration (requires cuDNN installation)
- Metal support is automatically available on macOS

### Example Usage
Examples are located in `candle-examples/examples/`. Run with:
```bash
cargo run --example quantized --release
cargo run --example llama --release
cargo run --example stable-diffusion --release --features cuda
```

## Key Design Patterns

### Device Abstraction
The framework uses a `Device` enum to abstract hardware:
- `Device::Cpu` for CPU execution
- `Device::new_cuda(0)?` for GPU execution
- Metal automatically detected on Apple platforms

### Tensor Operations
Tensors are the core data structure. Operations follow PyTorch-like patterns but with Rust ownership:
- Use `&tensor` for borrowing in operations
- Chain operations with `?` for error handling
- Reshape with `tensor.reshape((2, 2))?`

### Model Loading
Models typically support:
- SafeTensors format (preferred)
- PyTorch `.pth` files  
- GGML quantized formats
- Hub integration via `hf-hub` crate

### Error Handling
Use `anyhow::Result` for error propagation. Set `RUST_BACKTRACE=1` for debugging.

## Development Notes

### Performance Features
- Intel MKL support: Add `extern crate intel_mkl_src;` to enable
- Apple Accelerate: Add `extern crate accelerate_src;` for optimized BLAS
- Flash attention available for transformer models

### Quantization Support
The framework supports llama.cpp compatible quantization formats (Q4_0, Q4_1, Q8_0, etc.) for efficient inference.

### WASM Deployment
Many examples can be compiled to WebAssembly for browser deployment using `trunk serve`.

### Custom Operations
The framework supports custom CUDA kernels and Metal compute shaders for specialized operations.