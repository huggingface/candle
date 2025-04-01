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

### Changing model and language pairs

```bash
$ cargo run --example marian-mt --release -- --text "hello, how are you." --which base --language-pair en-zh

你好,你好吗?
```

## Generating the tokenizer.json files

The tokenizer for each `marian-mt` model was trained independently, 
meaning each new model needs unique tokenizer encoders and decoders.
You can use the following script to generate the `tokenizer.json` config files
from the hf-hub repos. This requires the `tokenizers` and `sentencepiece`
packages.

```python
from pathlib import Path
import warnings

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import SpmConverter, requires_backends, import_protobuf

class MarianConverter(SpmConverter):
    def __init__(self, *args, index: int = 0):
        requires_backends(self, "protobuf")

        super(SpmConverter, self).__init__(*args)

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        print(self.original_tokenizer.spm_files)
        with open(self.original_tokenizer.spm_files[index], "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m
        print(self.original_tokenizer)
        #with open(self.original_tokenizer.vocab_path, "r") as f:
        dir_path = Path(self.original_tokenizer.spm_files[0]).parents[0]
        with open(dir_path / "vocab.json", "r") as f:
            import json
            self._vocab = json.load(f)

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                    " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                    " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                    "unknown tokens into a sequence of byte tokens matching the original piece of text."
                )

    def vocab(self, proto):
        vocab_size = max(self._vocab.values()) + 1
        vocab = [("<NIL>", -100) for _ in range(vocab_size)]
        for piece in proto.pieces:
            try:
                index = self._vocab[piece.piece]
            except Exception:
                print(f"Ignored missing piece {piece.piece}")
            vocab[index] = (piece.piece, piece.score)
        return vocab


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en", use_fast=False)
fast_tokenizer = MarianConverter(tokenizer, index=0).converted()
fast_tokenizer.save(f"tokenizer-marian-base-fr.json")
fast_tokenizer = MarianConverter(tokenizer, index=1).converted()
fast_tokenizer.save(f"tokenizer-marian-base-en.json")
```
