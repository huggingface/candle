# candle-quantized-t5

This example uses a quantized version of the t5 model.

```bash
$ cargo run --example quantized-t5 --release -- --prompt "translate to German: A beautiful candle."
...
 Eine schöne Kerze.
```

The weight file is automatically retrieved from the hub. It is also possible to
generate quantized weight files from the original safetensors file by using the
`tensor-tools` command line utility via:

```bash
$ cargo run --example tensor-tools --release -- quantize --quantization q6k PATH/TO/T5/model.safetensors /tmp/model.gguf
```

To use a different model, specify the `model-id`. For example, you can use
quantized [CoEdit models](https://huggingface.co/jbochi/candle-coedit-quantized).

```bash
$ cargo run --example quantized-t5 --release  -- \
  --model-id "jbochi/candle-coedit-quantized" \
  --prompt "Make this text coherent: Their flight is weak. They run quickly through the tree canopy." \
  --temperature 0
...
 Although their flight is weak, they run quickly through the tree canopy.

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
