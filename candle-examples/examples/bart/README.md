# BART Example

This example demonstrates BART (Bidirectional and Auto-Regressive Transformers) for sequence-to-sequence tasks like summarization and translation.

## Supported Models

- **facebook/bart-large-cnn** - Summarization (CNN/DailyMail)
- **facebook/bart-large-xsum** - Summarization (XSum)
- **facebook/mbart-large-50-many-to-many-mmt** - Multilingual translation
- **naver-clova-ix/donut-base** - Document understanding (VisionEncoderDecoder)

## Usage

### Text Summarization

```bash
# Beam search decoding (recommended for summarization)
cargo run --example bart --release -- \
    --model-id facebook/bart-large-cnn \
    --prompt "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930." \
    --beam-size 4 \
    --length-penalty 2.0 \
    --min-length 30 \
    --sample-len 100

# Sampling-based generation
cargo run --example bart --release -- \
    --model-id facebook/bart-large-cnn \
    --prompt "Your article text here..." \
    --sample-len 100 \
    --temperature 0.7
```

### Multilingual Translation (mBART)

mBART models require converting the SentencePiece tokenizer first:

```bash
# Step 1: Convert tokenizer
cd candle-examples/examples/bart
pip install transformers sentencepiece
python convert_mbart_tokenizer.py --model-id facebook/mbart-large-50-many-to-many-mmt

# Step 2: Run translation (English to French)
cargo run --example bart --release -- \
    --model-id facebook/mbart-large-50-many-to-many-mmt \
    --prompt "Hello, how are you today?" \
    --source-lang en_XX \
    --target-lang fr_XX \
    --sample-len 50
```

### VisionEncoderDecoder (Donut)

For full Donut document understanding with real images, see the [donut example](../donut/).

```bash
# Test decoder with dummy encoder output
cargo run --example bart --release -- \
    --model-id naver-clova-ix/donut-base \
    --use-dummy-encoder \
    --sample-len 50
```

## Important Notes

### Input Length for Summarization

BART-large-cnn was trained on CNN/DailyMail articles (typically 500-1000 words). For best results:

- **Short inputs (1-2 sentences)**: Model may copy/repeat the input since there's nothing to summarize
- **Paragraph+ inputs (100+ words)**: Model produces proper abstractive summaries

### Beam Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--beam-size` | 1 | Number of beams (1 = greedy, 4 recommended for quality) |
| `--length-penalty` | 2.0 | Higher values favor longer outputs |
| `--min-length` | 10 | Minimum tokens before EOS is allowed |
| `--no-repeat-ngram-size` | 3 | Block n-gram repetition (0 = disabled) |

## Architecture

```
Input Text → [Encoder] → Hidden States → [Decoder + Cross-Attention] → Summary
                ↑                              ↑
           BartEncoder                  BartDecoder (autoregressive)
```

The encoder processes the full input bidirectionally, while the decoder generates output tokens one at a time, attending to the encoder's hidden states via cross-attention.
