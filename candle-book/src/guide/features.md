# Features

<!--- ANCHOR: features --->

- Simple syntax, looks and feels like PyTorch.
    - Model training.
    - Embed user-defined ops/kernels, such as [flash-attention v2](https://github.com/huggingface/candle/blob/89ba005962495f2bfbda286e185e9c3c7f5300a3/candle-flash-attn/src/lib.rs#L152).
- Backends.
    - Optimized CPU backend with optional MKL support for x86 and Accelerate for macs.
    - CUDA backend for efficiently running on GPUs, multiple GPU distribution via NCCL.
    - WASM support, run your models in a browser.
- Included models.
    - Language Models.
        - LLaMA v1 and v2.
        - Falcon.
        - StarCoder.
        - Phi v1.5.
        - Mistral 7b v0.1.
        - StableLM-3B-4E1T.
        - Replit-code-v1.5-3B.
        - Bert.
        - Yi-6B and Yi-34B.
    - Quantized LLMs.
        - Llama 7b, 13b, 70b, as well as the chat and code variants.
        - Mistral 7b, and 7b instruct.
        - Zephyr 7b a and b (Mistral based).
        - OpenChat 3.5 (Mistral based).
    - Text to text.
        - T5 and its variants: FlanT5, UL2, MADLAD400 (translation), CoEdit (Grammar correction).
        - Marian MT (Machine Translation).
    - Whisper (multi-lingual support).
    - Text to image.
        - Stable Diffusion v1.5, v2.1, XL v1.0.
        - Wurstchen v2.
    - Image to text.
        - BLIP.
    - Computer Vision Models.
        - DINOv2, ConvMixer, EfficientNet, ResNet, ViT.
        - yolo-v3, yolo-v8.
        - Segment-Anything Model (SAM).
- File formats: load models from safetensors, npz, ggml, or PyTorch files.
- Serverless (on CPU), small and fast deployments.
- Quantization support using the llama.cpp quantized types.

<!--- ANCHOR_END: features --->