#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# HuggingFace cache locations
HOME = Path.home()

# GGUF model from unsloth
GGUF_BASE = HOME / '.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF'
gguf_snapshots = list((GGUF_BASE / 'snapshots').glob('*'))
if not gguf_snapshots:
    print(f"Error: No snapshots found in {GGUF_BASE / 'snapshots'}", file=sys.stderr)
    print("Run: huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf")
    sys.exit(1)
GGUF_SNAPSHOT = gguf_snapshots[0]

# Tokenizer and config from original Qwen3 repo
QWEN_BASE = HOME / '.cache/huggingface/hub/models--Qwen--Qwen3-0.6B'
qwen_snapshots = list((QWEN_BASE / 'snapshots').glob('*'))
if not qwen_snapshots:
    print(f"Error: No snapshots found in {QWEN_BASE / 'snapshots'}", file=sys.stderr)
    print("Run: huggingface-cli download Qwen/Qwen3-0.6B tokenizer.json config.json")
    sys.exit(1)
QWEN_SNAPSHOT = qwen_snapshots[0]

print(f"GGUF model location: {GGUF_SNAPSHOT}")
print(f"Tokenizer/config location: {QWEN_SNAPSHOT}")

# Verify files exist
GGUF_FILE = 'Qwen3-0.6B-Q8_0.gguf'      # 8-bit quantization
#GGUF_FILE = 'Qwen3-0.6B-Q4_K_M.gguf'  # 4-bit K-quants (smaller, faster)

files_to_check = [
    (GGUF_SNAPSHOT, GGUF_FILE),
    (QWEN_SNAPSHOT, 'tokenizer.json'),
    (QWEN_SNAPSHOT, 'config.json'),
]

for base_path, filename in files_to_check:
    filepath = base_path / filename
    if not filepath.exists():
        print(f"Error: {filename} not found at {base_path}", file=sys.stderr)
        print(f"\nAvailable files in {base_path}:")
        for f in sorted(base_path.glob('*')):
            print(f"  - {f.name}")
        sys.exit(1)
    else:
        print(f"✓ Found: {filename}")


class CustomHandler(SimpleHTTPRequestHandler):
    # Add .wasm MIME type support
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
    }

    def end_headers(self):
        # Add CORS headers for all responses
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        SimpleHTTPRequestHandler.end_headers(self)

    def do_GET(self):
        # Serve model files from HuggingFace cache
        if self.path == '/' + GGUF_FILE:
            self.send_file(GGUF_SNAPSHOT / GGUF_FILE, 'application/octet-stream')
        elif self.path == '/tokenizer.json':
            self.send_file(QWEN_SNAPSHOT / 'tokenizer.json', 'application/json')
        elif self.path == '/config.json':
            self.send_file(QWEN_SNAPSHOT / 'config.json', 'application/json')
        else:
            # Serve everything else from current directory (pkg/, index.html)
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


if __name__ == '__main__':
    PORT = 8080
    print(f"\nServing WASM from: {os.getcwd()}")
    print(f"Serving Q4_K_M GGUF from: {GGUF_SNAPSHOT}")
    print(f"Serving tokenizer/config from: {QWEN_SNAPSHOT}")
    print(f"\n🚀 Server running at http://localhost:{PORT}\n")

    server = HTTPServer(('', PORT), CustomHandler)
    server.serve_forever()