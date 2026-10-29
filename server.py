"""
Localhost Server for House Price Predictor
Run: python server.py
Uses standard library only - no external packages needed!
"""
import http.server
import socketserver
import webbrowser
import os
import sys

# Ensure UTF-8 output if possible, or fallback safely
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Clean terminal logging
        print(f"[Localhost] {self.address_string()} - {args[0]}")

def start_server():
    os.chdir(DIRECTORY)
    
    # Try PORT 8000, fallback to 8080 or 8001 if occupied
    current_port = PORT
    httpd = None
    
    for p in [8000, 8080, 8001, 8002]:
        try:
            httpd = socketserver.TCPServer(("", p), CustomHandler)
            current_port = p
            break
        except OSError:
            continue

    if httpd is None:
        print("Error: Could not bind to an open port.")
        return

    url = f"http://localhost:{current_port}/frontend/index.html"

    print("=" * 60)
    print("  House Price Predictor - Localhost Server is Running!")
    print(f"  Local URL: {url}")
    print("  Opening your web browser automatically...")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    try:
        webbrowser.open(url)
    except Exception:
        pass
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nLocalhost server stopped gracefully.")
        httpd.server_close()
        sys.exit(0)

if __name__ == "__main__":
    start_server()
