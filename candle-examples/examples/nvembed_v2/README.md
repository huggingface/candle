# NV-Embed-v2

Candle implementation (inference only) of [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2), a text embedding model that ranks No. 1 (as of Nov 25 2024) on the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) benchmark with a score of 72.31 across 56 text embedding tasks.

## Running an example: Retrieval
```bash
cargo run --example nvembed_v2 --release
> scores: [[87.4269,  0.4629],
>         [ 0.9653, 86.0372]]
> Tensor[[2, 2], f32]
```
In this example, we have two queries and two passages (the corresponding answers). The output tensor represents the similarity scores between each query-passage pair. The scores are computed by taking the dot product of the query and passage embeddings and scaling the result by 100.
```rust
let queries = [
    "are judo throws allowed in wrestling?",
    "how to become a radiology technician in michigan?",
];
let query_instruction =
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
        .to_string();
        
let passages = [
    "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
    "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
];
let passage_instruction = "".to_string();
```

If you already have the model and tokenizer files, you can use the `--tokenizer` and `--model-files` options to specify their full paths, instead of downloading them from the hub.

## Running an example: Sentence embedding
```bash
cargo run --example nvembed_v2 --release -- --prompt "Here is a test sentence"
> Embedding: [[ 0.0066, -0.0048,  0.0066, ..., -0.0096,  0.0119, -0.0052]]
> Tensor[[1, 4096], f32]
```
In this example, we pass a prompt to the model and it outputs the vector encoding of the prompt.

## Hardware Requirements
29.25GB at fp32

## License
CC-BY-NC-4.0. This model should not be used for any commercial purpose. Refer the [license](https://spdx.org/licenses/CC-BY-NC-4.0) for the detailed terms.
