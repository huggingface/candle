# Llama Multinode

This project implements a distributed version of the Llama language model using Rust, CUDA, and NCCL for multi-node, multi-GPU inference.

## TL;DR

To quickly set up and run the project on the master node, use this single command:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
. "$HOME/.cargo/env" && \
git clone -b chore/llama_multinode https://github.com/b0xtch/candle.git && \
cd candle && \
git submodule update --init --recursive && \
pip install --upgrade huggingface_hub && \
echo '<hf_token>' | huggingface-cli login && \
source nccl_env_vars.sh && \
RUST_BACKTRACE=1 cargo run --example llama_multinode --release --features="cuda nccl" -- \
    --num-nodes 2 \
    --node-rank 0 \
    --master-addr 10.0.10.30 \
    --master-port 29500 \
    --num-gpus-per-node 1 \
    --model-id "meta-llama/Meta-Llama-3-8B" \
    --dtype bf16 \
    --prompt "Once upon a time"
```

Note: Replace `10.0.10.30` with the private IP of your master node.

## Prerequisites

- CUDA-capable GPUs
- CUDA Toolkit 12.1 or later
- Rust toolchain
- Docker (for containerized setup)

## Setup on AWS Nodes

Follow these steps to set up and run the project on AWS nodes:

1. Install Rust:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   source "$HOME/.cargo/env"
   ```

2. Clone the repository and update submodules:
   ```bash
   git clone -b chore/llama_multinode https://github.com/b0xtch/candle.git
   cd candle
   git submodule update --init --recursive
   ```

3. Install and set up Hugging Face CLI:
   ```bash
   pip install --upgrade huggingface_hub
   echo '<hf_token>' | huggingface-cli login
   ```

4. Set up NCCL environment variables:
   ```bash
   source nccl_env_vars.sh
   ```

## Running the Distributed Llama Model

### On the Master Node

Run the following command, replacing `10.0.10.30` with the private IP of your master node:

```bash
RUST_BACKTRACE=1 cargo run --example llama_multinode --release --features="cuda nccl" -- \
    --num-nodes 2 \
    --node-rank 0 \
    --master-addr 10.0.10.30 \
    --master-port 29500 \
    --num-gpus-per-node 1 \
    --model-id "meta-llama/Meta-Llama-3-8B" \
    --dtype bf16 \
    --prompt "Once upon a time"
```

### On Worker Nodes

Run the following command on each worker node, replacing `10.0.10.30` with the private IP of your master node:

```bash
RUST_BACKTRACE=1 cargo run --example llama_multinode --release --features="cuda nccl" -- \
    --num-nodes 2 \
    --node-rank 1 \
    --master-addr 10.0.10.30 \
    --master-port 29500 \
    --num-gpus-per-node 1 \
    --model-id "meta-llama/Meta-Llama-3-8B" \
    --dtype bf16 \
    --prompt "Once upon a time"
```

Note: Increment the `--node-rank` for each additional worker node.

## Troubleshooting

- Ensure all nodes can communicate with each other over the specified port (29500 in this example).
- Check that the CUDA and NCCL versions are compatible across all nodes.
- Verify that the Hugging Face token has the necessary permissions to access the model.

## Misc

> Optional: Run the NVIDIA CUDA container:
   ```bash
   docker run -it --gpus all nvidia/cuda:12.1.1-devel-ubuntu20.04 /bin/bash
   ```

> Optional: Inside the container, install necessary dependencies:
   ```bash
   apt-get -y update && \
   apt-get -y install curl git pkg-config libssl-dev
   ```
---