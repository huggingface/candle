# candle-marian-mt

`marian-mt` is a neural machine translation model. In this example it is used to
translate text from French to English. See the associated [model
card](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-fr-en) for details on
the model itself.

## Running an example

```bash
cargo run --example marian-mt --release -- \
    --text "Demain, dès l'aube, à l'heure où blanchit la campagne, Je partirai. Vois-tu, je sais que tu m'attends. J'irai par la forêt, j'irai par la montagne. Je ne puis demeurer loin de toi plus longtemps."
```

```
Tomorrow, at dawn, at the time when the country is whitening, I will go. See, I
know you are waiting for me. I will go through the forest, I will go through the
mountain. I cannot stay far from you any longer.
```

### Changing model and language pairs

```bash
$ cargo run --example marian-mt --release -- --text "hello, how are you." --which base --language-pair en-zh

你好,你好吗?
```

## Tokenizers

The tokenizer for each `marian-mt` model was trained independently, meaning each
model needs unique tokenizer encoders and decoders. These are built on the fly
from the `source.spm`, `target.spm` and `vocab.json` files of the model repo, so
adding a new language pair does not require any conversion step. Pre-built
`tokenizer.json` files can be used instead via `--tokenizer` and
`--tokenizer-dec`.
