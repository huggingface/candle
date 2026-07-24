# Usage: 

## GPU — auto-downloads model from HF, uses sample image
cargo run --example lightonocr_2_1B --features cuda

## CPU
cargo run --example lightonocr_2_1B -- --cpu

## With your own image
cargo run --example lightonocr_2_1B --features cuda -- --image-location /path/to/doc.png

## Full local control
cargo run --example lightonocr_2_1B --features cuda -- \
  --weights ./model.safetensors --config ./config.json --tokenizer ./tokenizer.json \
  --image-location ./doc.png --max-new-tokens 512 --max-edge 1024