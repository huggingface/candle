# BART Batch Inference Example

This example demonstrates efficient batch inference with BART models using batched beam search. All `batch_size × beam_size` hypotheses are processed in parallel with KV cache reuse and reordering.

## Usage

```bash
cargo run --example bart_batch --release -- \
    --model-id facebook/bart-large-cnn \
    --input-file prompts.txt \
    --batch-size 4 \
    --beam-size 4 \
    --length-penalty 2.0 \
    --min-length 30 \
    --max-length 100
```

## Input File Format

Create a text file with one article per line. For best summarization results, use paragraph-length inputs (100+ words per line):

```text
Article 1 text here...
Article 2 text here...
Article 3 text here...
```

See `prompts.txt` for example articles.

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | facebook/bart-large-cnn | HuggingFace model ID |
| `--input-file` | (required) | Path to file with one prompt per line |
| `--batch-size` | 4 | Number of inputs to process in parallel |
| `--beam-size` | 4 | Beam width for beam search |
| `--max-length` | 50 | Maximum generation length |
| `--min-length` | 0 | Minimum length before EOS allowed |
| `--length-penalty` | 1.0 | Length normalization (higher = longer) |
| `--no-repeat-ngram-size` | 3 | Block n-gram repetition |

## How Batched Beam Search Works

1. **Encode**: All inputs encoded in parallel → `(batch_size, seq_len, hidden_dim)`
2. **Expand**: Encoder output expanded for beams → `(batch_size × beam_size, seq_len, hidden_dim)`
3. **Decode Loop**:
   - Forward pass processes all `batch_size × beam_size` hypotheses at once
   - KV cache is reordered (not recomputed) when beams are pruned
   - Cross-attention cache computed once per encoder output
4. **Collect**: Best hypothesis returned for each input

## Performance

Batched inference provides significant speedup over sequential processing:

- **Memory**: KV cache shared across beams via efficient reordering
- **Compute**: All hypotheses processed in single forward pass
- **I/O**: Encoder output computed once, reused across all beams

## Example Output

```
Processing batch 1/1

Prompt: The tower is 324 metres (1,063 ft) tall...
Output (score=-0.234): The Eiffel Tower is 324 metres tall and the tallest structure in Paris.

Prompt: Scientists have discovered a high-redshift black hole...
Output (score=-0.189): A supermassive black hole has been discovered 13 billion light-years from Earth.
```

## Important Notes

### Input Length Matters

BART-large-cnn expects article-length inputs. Short inputs (1-2 sentences) may be copied rather than summarized. Use the provided `prompts.txt` as a reference for appropriate input lengths.

### Memory Usage

Batch size × beam size determines memory usage. For a 4×4 configuration:
- 16 decoder states maintained simultaneously
- Cross-attention cache: `16 × num_layers × seq_len × hidden_dim`

Reduce batch size if running out of memory.
