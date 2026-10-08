set -e

if [ ! -f tailwind-3.4.17.js ]; then
    echo "Downloading tailwind-3.4.17.js..."
    curl -L -o tailwind-3.4.17.js https://cdn.tailwindcss.com/3.4.17
fi

cargo build --target wasm32-unknown-unknown --release
wasm-bindgen ../../target/wasm32-unknown-unknown/release/m.wasm --out-dir build --target web
python server.py