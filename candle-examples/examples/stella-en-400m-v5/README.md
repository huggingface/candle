---
model-index:
- name: stella_en_400M_v5
license: mit
---


# Introduction

The models are trained based on `Alibaba-NLP/gte-large-en-v1.5` and `Alibaba-NLP/gte-Qwen2-1.5B-instruct`. Thanks for
their contributions!

**We simplify usage of prompts, providing two prompts for most general tasks, one is for s2p, another one is for s2s.**



The following reproduces the example in the [model card](https://huggingface.co/dunzhang/stella_en_400M_v5).for a retrieval task (s2p). The sample queries and docs are hardcoded in the example.

```bash
$ cargo run --example stella-en-v5 --release --features <metal | cuda>

Similarity========
 [[0.8398, 0.2990],
 [0.3282, 0.8095]]

Score: 0.83975387
Query: What are some ways to reduce stress?
Answer: There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.



Score: 0.8095451
Query: What are the benefits of drinking green tea?
Answer: Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.


```


The models are finally trained by [MRL](https://arxiv.org/abs/2205.13147), so they have multiple dimensions: 512, 768,
1024, 2048, 4096, 6144 and 8192.

## Supported options:
- `Stella_en_400m_v5` supports 256, 768, 1024, 2048, 4096, 6144 and 8192 embedding dimensions (though the model card mentions 512, I couldn't find weights for the same). In the example run this is supported with `--embed-dim` option. E.g. `... --embed-dim 4096`. Defaults to `1024`.

- As per the [model card](https://huggingface.co/dunzhang/stella_en_400M_v5), the model has been primarily trained on `s2s` (similarity) and `s2p` (retrieval) tasks. These require a slightly different `query` preprocessing (a different prompt template for each). In this example this is enabled though `--task` option.