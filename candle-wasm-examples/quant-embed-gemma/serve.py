# reference - https://github.com/huggingface/candle/blob/main/candle-wasm-examples/quant-qwen3/serve.py

#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: Required packages not installed", file=sys.stderr)
    print("Install with: pip install huggingface-hub tqdm", file=sys.stderr)
    sys.exit(1)

HOME = Path.home()
HF_CACHE = HOME / ".cache/huggingface/hub"

# Base quantized models (GGUF)
MODELS = {
    "q8": {
        "repo": "unsloth/embeddinggemma-300m-GGUF",
        "filename": "embeddinggemma-300M-Q8_0.gguf",
        "size": "~329MB",
    },
    "q4": {
        "repo": "unsloth/embeddinggemma-300m-GGUF",
        "filename": "embeddinggemma-300m-Q4_0.gguf",
        "size": "~278MB",
    },
}

# Dense layers + tokenizer/config
TOKENIZER_REPO = "google/embeddinggemma-300m"

DENSE1_PATH = "2_Dense/model.safetensors"
DENSE2_PATH = "3_Dense/model.safetensors"
TOKENIZER_JSON = "tokenizer.json"
CONFIG_JSON = "config.json"


def hf_download(repo_id: str, filename: str, cache_dir: Path) -> Path:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            resume_download=True,
        )
        return Path(path)
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {e}", file=sys.stderr)
        sys.exit(1)


def download_assets():
    # Download BOTH quant models
    model_paths = {}
    for key, cfg in MODELS.items():
        print(f"Downloading base model ({key}) {cfg['filename']} ...")
        model_paths[key] = hf_download(cfg["repo"], cfg["filename"], HF_CACHE)

    # Download Dense1 / Dense2 + tokenizer/config from TOKENIZER_REPO
    print("Downloading tokenizer/config + dense layers ...")
    dense1 = hf_download(TOKENIZER_REPO, DENSE1_PATH, HF_CACHE)
    dense2 = hf_download(TOKENIZER_REPO, DENSE2_PATH, HF_CACHE)
    tok = hf_download(TOKENIZER_REPO, TOKENIZER_JSON, HF_CACHE)
    cfg = hf_download(TOKENIZER_REPO, CONFIG_JSON, HF_CACHE)

    print("Q8:", model_paths["q8"])
    print("Q4:", model_paths["q4"])
    print("Dense1:", dense1)
    print("Dense2:", dense2)
    print("Tokenizer:", tok)
    print("Config:", cfg)

    return model_paths, dense1, dense2, tok, cfg


class CustomHandler(SimpleHTTPRequestHandler):
    # Filled in at runtime
    model_paths = None  # dict: {"q8": Path, "q4": Path}
    dense1_path = None
    dense2_path = None
    tokenizer_path = None
    config_path = None

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".gguf": "application/octet-stream",
        ".safetensors": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_GET(self):
        # Fixed endpoints (frontend uses these)
        if self.path == "/model-q8.gguf":
            return self.send_file(self.model_paths["q8"], "application/octet-stream")
        if self.path == "/model-q4.gguf":
            return self.send_file(self.model_paths["q4"], "application/octet-stream")
        if self.path == "/dense1.safetensors":
            return self.send_file(self.dense1_path, "application/octet-stream")
        if self.path == "/dense2.safetensors":
            return self.send_file(self.dense2_path, "application/octet-stream")
        if self.path == "/tokenizer.json":
            return self.send_file(self.tokenizer_path, "application/json")
        if self.path == "/config.json":
            return self.send_file(self.config_path, "application/json")

        # Otherwise serve the static site (index.html, pkg/, main.js, etc)
        return super().do_GET()

    def send_file(self, filepath: Path, content_type: str):
        try:
            data = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(404, f"File not found: {e}")

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve EmbeddingGemma WASM demo")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    args = parser.parse_args()

    print("=" * 60)
    print("Quantized EmbeddingGemma WASM Server")
    print("=" * 60)

    model_paths, dense1, dense2, tok, cfg = download_assets()

    CustomHandler.model_paths = model_paths
    CustomHandler.dense1_path = dense1
    CustomHandler.dense2_path = dense2
    CustomHandler.tokenizer_path = tok
    CustomHandler.config_path = cfg

    print("\nServing from:", os.getcwd())
    print(f"Port: {args.port}")
    print("Endpoints:")
    print("  /model-q8.gguf")
    print("  /model-q4.gguf")
    print("  /dense1.safetensors")
    print("  /dense2.safetensors")
    print("  /tokenizer.json")
    print("  /config.json")
    print("=" * 60)
    print(f"\nServer running at http://localhost:{args.port}\nPress Ctrl+C to stop\n")

    server = HTTPServer(("", args.port), CustomHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
