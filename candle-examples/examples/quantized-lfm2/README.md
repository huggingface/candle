# candle-quantized-lfm2

Candle implementation of the quantized (GGUF) LFM2 and LFM2.5 text models,
including the LFM2.5-8B-A1B mixture of experts.

`--which` picks the model and `--quant` the GGUF file, e.g. `Q4_0`, `Q4_K_M`,
`Q5_K_M`, `Q6_K`, `Q8_0` or `F16`. The file is fetched from
`LiquidAI/<model>-GGUF`.

```bash
cargo run --example quantized-lfm2 --release -- --which lfm2.5-8b-a1b --quant Q4_K_M
```

## Running an example

```bash
$ cargo run --example quantized-lfm2 --release -- --which lfm2-2.6b --prompt "Tell me a story in 100 words."
avx: false, neon: true, simd128: false, f16c: false
temp: 0.80 repeat-penalty: 1.10 repeat-last-n: 64
Running on CPU, to run on GPU(metal), build this example with `--features metal`
loaded 266 tensors (1.56GB) in 0.13s
model ready
Starting the inference loop:
Tell me a story in 100 words.

A quiet town nestled between rolling hills, where every springtime arrives with laughter and blossoms. Clara, the town’s beloved baker, opens her shop at dawn—cinnamon swirling into warm air, fresh pastries glowing on wooden racks. Each customer greets her with a smile, sharing tales while savoring sweet treats. One day, an old man hands her a faded photo: him and Clara, decades ago, when she’d kneaded dough for his wedding cake. Now he waits in silence, unseen. Clara bakes him another batch—hope rising from the oven, turning cold hearts into laughter again.

  10 prompt tokens processed: 39.28 token/s
 133 tokens generated: 43.34 token/s
```