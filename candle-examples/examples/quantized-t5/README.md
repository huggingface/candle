# candle-quantized-t5

Candle implementation for quantizing and running T5 translation models.

## Seq2Seq example

This example uses a quantized version of the t5 model.

```bash
$ cargo run --example quantized-t5 --release -- --prompt "translate to German: A beautiful candle."
...
 Eine schöne Kerze.
```

## Generating Quantized weight files

The weight file is automatically retrieved from the hub. It is also possible to
generate quantized weight files from the original safetensors file by using the
`tensor-tools` command line utility via:

```bash
$ cargo run --bin tensor-tools --release -- quantize --quantization q6k PATH/TO/T5/model.safetensors /tmp/model.gguf
```

## Using custom models

To use a different model, specify the `model-id`.

For example, for text editing, you can use quantized [CoEdit models](https://huggingface.co/jbochi/candle-coedit-quantized).

```bash
$ cargo run --example quantized-t5 --release  -- \
  --model-id "jbochi/candle-coedit-quantized" \
  --prompt "Make this text coherent: Their flight is weak. They run quickly through the tree canopy." \
  --temperature 0
...
 Although their flight is weak, they run quickly through the tree canopy.
```

By default, it will look for `model.gguf` and `config.json`, but you can specify
custom local or remote `weight-file` and `config-file`s:

```bash
cargo run --example quantized-t5 --release  -- \
  --model-id "jbochi/candle-coedit-quantized" \
  --weight-file "model-xl.gguf" \
  --config-file "config-xl.json" \
  --prompt "Rewrite to make this easier to understand: Note that a storm surge is what forecasters consider a hurricane's most treacherous aspect." \
  --temperature 0
...
 Note that a storm surge is what forecasters consider a hurricane's most dangerous part.
```

### [MADLAD-400](https://arxiv.org/abs/2309.04662)

MADLAD-400 is a series of multilingual machine translation T5 models trained on 250 billion tokens covering over 450 languages using publicly available data. These models are competitive with significantly larger models.

```bash
cargo run --example quantized-t5 --release  -- \
  --model-id "jbochi/madlad400-3b-mt" --weight-file "model-q4k.gguf" \
  --prompt "<2de> How are you, my friend?" \
  --temperature 0
...
 Wie geht es dir, mein Freund?
```
