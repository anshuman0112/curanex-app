"""Serves the exported web build (../dist) with single-page fallback, so links like /doctor open directly.

/api/... is forwarded to the Curanex API on localhost:8001, so a phone needs ONE link (Cloudflare tunnel to this server):
the app and its API share the same https address, and nothing has to be remembered on the phone.
"""
import http.client
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIST = Path(__file__).resolve().parents[2] / "dist"
API_HOST, API_PORT = "127.0.0.1", 8001
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        if not Path(self.translate_path(self.path)).exists():
            self.path = "/index.html"
        return super().send_head()

    # ---- /api proxy ----
    def _proxy(self) -> None:
        path = self.path[len("/api"):] or "/"
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            conn = http.client.HTTPConnection(API_HOST, API_PORT, timeout=900)
            conn.request(self.command, path, body=body, headers=headers)
            r = conn.getresponse()
            data = r.read()
        except OSError:
            self.send_error(502, "Curanex API is not running")
            return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        conn.close()

    def _route(self, fallback) -> None:
        if self.path == "/api" or self.path.startswith("/api/"):
            self._proxy()
        else:
            fallback()

    def do_GET(self):  # noqa: N802
        self._route(super().do_GET)

    def do_HEAD(self):  # noqa: N802
        self._route(super().do_HEAD)

    def do_POST(self):  # noqa: N802
        self._route(lambda: self.send_error(405))

    def do_DELETE(self):  # noqa: N802
        self._route(lambda: self.send_error(405))

    def do_OPTIONS(self):  # noqa: N802
        self._route(lambda: self.send_error(405))


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8090), partial(Handler, directory=str(DIST))).serve_forever()
