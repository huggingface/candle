# candle-modernbert

ModernBERT is a bidirectional encoder-only language model. In this example it is used for the fill-mask task:

## Usage

```bash
cargo run --example modernbert --release  -- --model modern-bert-large --prompt 'The capital of France is [MASK].'
```
```markdown
Sentence: 1 : The capital of France is Paris.
```
