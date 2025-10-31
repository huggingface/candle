#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

try:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed", file=sys.stderr)
    print("Install with: pip install huggingface-hub tqdm", file=sys.stderr)
    sys.exit(1)

HOME = Path.home()
HF_CACHE = HOME / '.cache/huggingface/hub'

# Model configurations
MODELS = {
    '0.6b-q8': {
        'repo': 'unsloth/Qwen3-0.6B-GGUF',
        'filename': 'Qwen3-0.6B-Q8_0.gguf',
        'size': '~645MB',
        'description': '8-bit quantization (good quality and fastest)'
    },
    '0.6b-q4': {
        'repo': 'unsloth/Qwen3-0.6B-GGUF',
        'filename': 'Qwen3-0.6B-Q4_K_M.gguf',
        'size': '~380MB',
        'description': '4-bit quantization (smaller, less accurate, slower in WASM SIMD)'
    }
}

TOKENIZER_REPO = 'Qwen/Qwen3-0.6B'


def download_with_progress(repo_id, filename, cache_dir):
    """Download a file from HuggingFace with progress bar"""
    print(f"\nDownloading {filename} from {repo_id}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True
        )
        print(f"✓ Downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"Error downloading {filename}: {e}", file=sys.stderr)
        sys.exit(1)


def find_or_download_model(model_key, custom_path=None):
    """Find model in cache or download it"""
    if custom_path:
        custom_path = Path(custom_path)
        if not custom_path.exists():
            print(f"Error: Custom path does not exist: {custom_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using custom model: {custom_path}")
        return custom_path

    model_config = MODELS[model_key]
    repo_id = model_config['repo']
    filename = model_config['filename']

    # Check cache first
    repo_cache = HF_CACHE / f"models--{repo_id.replace('/', '--')}"
    if repo_cache.exists():
        snapshots = list((repo_cache / 'snapshots').glob('*'))
        if snapshots:
            model_path = snapshots[0] / filename
            if model_path.exists():
                print(f"✓ Found model in cache: {model_path}")
                return model_path

    # Download if not found
    print(f"Model not found in cache")
    print(f"Size: {model_config['size']} - {model_config['description']}")
    return download_with_progress(repo_id, filename, HF_CACHE)


def find_or_download_tokenizer():
    """Find tokenizer files or download them"""
    repo_cache = HF_CACHE / f"models--{TOKENIZER_REPO.replace('/', '--')}"

    if repo_cache.exists():
        snapshots = list((repo_cache / 'snapshots').glob('*'))
        if snapshots:
            tokenizer_path = snapshots[0] / 'tokenizer.json'
            config_path = snapshots[0] / 'config.json'
            if tokenizer_path.exists() and config_path.exists():
                print(f"✓ Found tokenizer in cache: {snapshots[0]}")
                return snapshots[0]

    print("Tokenizer not found in cache")
    print("Downloading tokenizer and config...")

    tokenizer_path = download_with_progress(TOKENIZER_REPO, 'tokenizer.json', HF_CACHE)
    config_path = download_with_progress(TOKENIZER_REPO, 'config.json', HF_CACHE)

    return tokenizer_path.parent


class CustomHandler(SimpleHTTPRequestHandler):
    model_path = None
    tokenizer_dir = None

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
    }

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        SimpleHTTPRequestHandler.end_headers(self)

    def do_GET(self):
        # Serve model file
        if self.path.endswith('.gguf'):
            self.send_file(self.model_path, 'application/octet-stream')
        elif self.path == '/tokenizer.json':
            self.send_file(self.tokenizer_dir / 'tokenizer.json', 'application/json')
        elif self.path == '/config.json':
            self.send_file(self.tokenizer_dir / 'config.json', 'application/json')
        else:
            SimpleHTTPRequestHandler.do_GET(self)

    def send_file(self, filepath, content_type):
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(404, f"File not found: {e}")

    def log_message(self, format, *args):
        # Suppress default logging for cleaner output
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Serve Qwen3 WASM model with automatic downloads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Q8_0 model
  %(prog)s

  # Use Q4 model (smaller, less accurate, slower in WASM SIMD)
  %(prog)s --model 0.6b-q4

  # Use custom model file
  %(prog)s --path /path/to/model.gguf

  # Change port
  %(prog)s --port 3000
        """
    )

    parser.add_argument(
        '--model', '-m',
        choices=list(MODELS.keys()),
        default='0.6b-q8',
        help='Model to use (default: 0.6b-q8)'
    )

    parser.add_argument(
        '--path', '-p',
        type=str,
        help='Path to custom GGUF model file'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Server port (default: 8080)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for key, config in MODELS.items():
            print(f"\n  {key}:")
            print(f"    Size: {config['size']}")
            print(f"    Description: {config['description']}")
            print(f"    File: {config['filename']}")
        return

    print("=" * 60)
    print("Qwen3 WASM Server")
    print("=" * 60)

    # Find or download model
    model_path = find_or_download_model(args.model, args.path)
    tokenizer_dir = find_or_download_tokenizer()

    # Set paths for handler
    CustomHandler.model_path = model_path
    CustomHandler.tokenizer_dir = tokenizer_dir

    print("\n" + "=" * 60)
    print(f"Model: {model_path.name}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"Serving from: {os.getcwd()}")
    print(f"Port: {args.port}")
    print("=" * 60)
    print(f"\n Server running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")

    try:
        server = HTTPServer(('', args.port), CustomHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()


if __name__ == '__main__':
    main()