set -e
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals" cargo build --target wasm32-unknown-unknown --release
wasm-bindgen ../../target/wasm32-unknown-unknown/release/m.wasm --out-dir build --target web --split-linked-modules
wasm-opt -Oz --enable-threads --enable-bulk-memory build/m_bg.wasm -o build/m_bg.wasm
python server.py
