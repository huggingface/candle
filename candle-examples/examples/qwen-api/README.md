# 使用candle+axum开发一个简单的copilot服务器
## 缘起
自己实现一个copilot服务器，是不是很酷?!
我偶然发现在candle下面的qwen 7B 模型，可以支持辅助编码，于是就有了写了一个简单的服务器的想法。
说干就干！撸起袖子玩命干！

## 步骤

### 1. 创建一个axum项目
我是在candel的candle-examples的examples目录下创建的，起名qwen-api。

### 2. 运行
```shell
curl --location '127.0.0.1:7568/ssecode' \
--header 'Content-Type: application/json' \
--data '{"prompt":"C语言写一个冒泡排序算法，并解释其运行原理."}'
```

#### n. docker启动
```shell
$ sudo docker run  --runtime=nvidia  --gpus all  --name rust-dev --restart=always -d -p 7568:7568 -v /home/sunny/work/算法 相关/candle:/opt/candle -it  harbor.cloudminds.com/crss/rust/rust_cuda_ubuntu:dev-20240418

```

## 参考网址
[candle主站](https://github.com/huggingface/candle)
[Streamlining Serverless ML Inference: Unleashing Candle Framework’s Power in Rust](https://towardsdatascience.com/streamlining-serverless-ml-inference-unleashing-candle-frameworks-power-in-rust-c6775d558545)
[Rust使用axum结合Actor模型实现异步发送SSE](https://juejin.cn/post/7236591682615525431)
[I made a Copilot in Rust 🦀 , here is what I have learned... ](https://dev.to/chenhunghan/i-made-a-copilot-in-rust-here-is-what-i-have-learned-as-a-typescript-dev-52md)
[code llama](https://huggingface.co/docs/transformers/main/model_doc/code_llama)