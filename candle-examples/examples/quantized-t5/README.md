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
cargo run --example tensor-tools --release -- quantize --quantization q6k PATH/TO/T5/model.safetensors /tmp/model.gguf
```
