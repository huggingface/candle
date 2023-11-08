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
<NIL> Tomorrow, at dawn, at the time when the country is whitening, I will go. See,
I know you are waiting for me. I will go through the forest, I will go through the
mountain. I cannot stay far from you any longer.</s>
```

## Generating the tokenizer.json files

You can use the following script to generate the `tokenizer.json` config files
from the hf-hub repos. This requires the `tokenizers` and `sentencepiece`
packages to be install and use the `convert_slow_tokenizer.py` script from this
directory.

```python
from convert_slow_tokenizer import MarianConverter
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en", use_fast=False)
fast_tokenizer = MarianConverter(tokenizer, index=0).converted()
fast_tokenizer.save(f"tokenizer-marian-base-fr.json")
fast_tokenizer = MarianConverter(tokenizer, index=1).converted()
fast_tokenizer.save(f"tokenizer-marian-base-en.json")
```
