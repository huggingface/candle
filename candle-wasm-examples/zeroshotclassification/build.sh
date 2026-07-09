RUSTFLAGS='--cfg getrandom_backend="wasm_js"' cargo build --target wasm32-unknown-unknown --release
wasm-bindgen ../../target/wasm32-unknown-unknown/release/m.wasm --out-dir build --target web
