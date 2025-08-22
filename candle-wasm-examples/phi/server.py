#!/usr/bin/env python3
import socketserver
from http.server import SimpleHTTPRequestHandler


class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "same-origin")
        super().end_headers()


with socketserver.TCPServer(("", 8000), CustomHTTPRequestHandler) as httpd:
    print("Server running at http://localhost:8000/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user (Ctrl-C)")
