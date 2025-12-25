# candle-donut

[Donut](https://huggingface.co/naver-clova-ix/donut-base) (Document Understanding Transformer) is an OCR-free document understanding model that combines a Swin Transformer encoder for image processing with a BART decoder for text generation.

## Architecture

```
Document Image → [Swin Encoder] → Visual Features → [BART Decoder] → Structured Text
     ↑                                                    ↑
  1280×960                                          Cross-attention
  3 channels                                        to visual features
```

The model processes document images directly without requiring OCR, learning to extract structured information end-to-end.

## Supported Tasks

- **cord-v2**: Receipt parsing (extracts structured data from receipts)
- **docvqa**: Document Visual Question Answering
- **rvlcdip**: Document classification
- **zhtrainticket**: Chinese train ticket parsing

## Running the Examples

### Receipt Parsing (CORD-v2)

```bash
cargo run --example donut --release -- \
    --image candle-examples/examples/donut/cake-receipt.jpeg \
    --task cord-v2
```

### Document Classification (RVL-CDIP)

```bash
cargo run --example donut --release -- \
    --model-id naver-clova-ix/donut-base-finetuned-rvlcdip \
    --image candle-examples/examples/donut/cake-receipt.jpeg \
    --task rvlcdip
```

### Document VQA

```bash
cargo run --example donut --release -- \
    --model-id naver-clova-ix/donut-base-finetuned-docvqa \
    --image candle-examples/examples/donut/cake-receipt.jpeg \
    --task docvqa \
    --question "What is the total amount?"
```

## Command Line Options

- `--image`: Path to the document image (required)
- `--task`: Task type: `cord-v2`, `docvqa`, `rvlcdip`, `zhtrainticket` (default: `cord-v2`)
- `--question`: Question for DocVQA task (required when using `docvqa`)
- `--model-id`: HuggingFace model repository (default: `naver-clova-ix/donut-base-finetuned-cord-v2`)
- `--max-length`: Maximum generation length (default: 512)
- `--cpu`: Run on CPU instead of GPU

## Model Weights

Weights are automatically downloaded from HuggingFace Hub:

| Task | Model ID |
|------|----------|
| Receipt parsing | `naver-clova-ix/donut-base-finetuned-cord-v2` |
| Document VQA | `naver-clova-ix/donut-base-finetuned-docvqa` |
| Document classification | `naver-clova-ix/donut-base-finetuned-rvlcdip` |
| Chinese train tickets | `naver-clova-ix/donut-base-finetuned-zhtrainticket` |
