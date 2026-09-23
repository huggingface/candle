# lfm2-moe

LFM2-MoE is the Mixture-of-Experts variant of [LFM2](../lfm2) from
[LiquidAI](https://www.liquid.ai/). It keeps the hybrid attention +
short-convolution backbone of LFM2 and replaces the dense feed-forward of the
later layers with a sparse MoE block (DeepSeek-V3 style sigmoid routing with a
per-expert selection bias). Both `LFM2.5-8B-A1B` and `LFM2-8B-A1B` have 8.3B
total parameters but only ~1.5B active per token (top-4 of 32 experts).

## Running the example

```bash
$ cargo run --example lfm2-moe --release -- --prompt "The capital of France is"
```

The default model is `LFM2.5-8B-A1B`; use `--which lfm2-8b-a1b` for the older
LFM2 checkpoint.

Sample output:

```
The capital of France is Paris.
```

A chat-formatted prompt for the instruct usage looks like:

```bash
$ cargo run --example lfm2-moe --release -- \
    --prompt "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
```

Use `--model-id <hub-id>` to point at another LFM2-MoE checkpoint, and `--cpu`
to force CPU execution.
