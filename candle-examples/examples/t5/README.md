# candle-t5

Candle implementations of the T5 family of translation models.

## Encoder-decoder example:

```bash
$ cargo run --example t5 --release -- --model-id "t5-small" --prompt "translate to German: A beautiful candle." --decode
...
 Eine schöne Kerze.
9 tokens generated (2.42 token/s)
```

Variants such as [flan-t5](https://huggingface.co/google/flan-t5-small), [flan-ul2](https://huggingface.co/google/flan-ul2) (with `--revision "refs/pr/25"`), and [Co-EdIT](https://huggingface.co/grammarly/coedit-large) are also supported.

## Translation with [MADLAD-400](https://arxiv.org/abs/2309.04662)

MADLAD-400 is a series of multilingual machine translation T5 models trained on 250 billion tokens covering over 450 languages using publicly available data. These models are competitive with significantly larger models.

```bash
cargo run --example t5 --release  -- \
  --model-id "jbochi/madlad400-3b-mt" \
  --prompt "<2de> How are you, my friend?" \
  --decode --temperature 0
...
 Wie geht es dir, mein Freund?
```

## Sentence embedding example

```bash
$ cargo run --example t5 --release -- --model-id "t5-small" --prompt "A beautiful candle."
...
[[[ 0.0515, -0.0541, -0.0761, ..., -0.0392,  0.1511, -0.0265],
  [-0.0974,  0.0998, -0.1659, ..., -0.2450,  0.1738, -0.0164],
  [ 0.0624, -0.1024,  0.0430, ..., -0.1388,  0.0564, -0.2962],
  [-0.0389, -0.1173,  0.0026, ...,  0.1064, -0.1065,  0.0990],
  [ 0.1300,  0.0027, -0.0326, ...,  0.0026, -0.0317,  0.0851]]]
Tensor[[1, 5, 512], f32]
Took 303.766583ms
```
