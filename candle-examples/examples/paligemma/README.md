# PaliGemma

[HuggingFace Model Card](https://huggingface.co/google/paligemma-3b-pt-224) -
[Model Page](https://ai.google.dev/gemma/docs/paligemma)

```bash
cargo run --features cuda --release --example paligemma -- \
    --prompt "caption fr" --image candle-examples/examples/yolo-v8/assets/bike.jpg
```

```
loaded image with shape Tensor[dims 1, 3, 224, 224; bf16, cuda:0]
loaded the model in 1.267744448s
caption fr. Un groupe de cyclistes qui sont dans la rue.
13 tokens generated (56.52 token/s)
```

```bash
cargo run --features cuda --release --example paligemma -- \
    --prompt "caption fr" --image candle-examples/examples/flux/assets/flux-robot.jpg
```

```
loaded image with shape Tensor[dims 1, 3, 224, 224; bf16, cuda:0]
loaded the model in 1.271492621s
caption fr une image d' un robot sur la plage avec le mot rouillé
15 tokens generated (62.78 token/s)
```
