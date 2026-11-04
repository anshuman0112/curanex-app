"""Serves the exported web build (../dist) with single-page fallback, so links like /doctor open directly."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIST = Path(__file__).resolve().parents[2] / "dist"


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        if not Path(self.translate_path(self.path)).exists():
            self.path = "/index.html"
        return super().send_head()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8090), partial(Handler, directory=str(DIST))).serve_forever()
