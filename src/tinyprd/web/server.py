"""En lille webserver, så man kan prøve TinyPRD i browseren.

Start den med:   tinyprd serve        og åbn  http://localhost:8000

Den bruger kun Pythons indbyggede `http.server`, så der er ingen ekstra
pakker at installere. Der er tre adresser:

    GET  /               selve hjemmesiden (static/index.html)
    GET  /api/info       hvilke modeller der er indlæst, eksempler på idéer
    POST /api/generate   {"idea": "...", "temperature": 0, "compare": true}  →  PRD, niveauer og kritik
    POST /api/critique   {"text": "## problem ..."}                        →  kritikerens point

Serveren er lavet til at køre på din egen maskine, ikke på internettet.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .service import PrdService

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 64_000


def make_handler(service: PrdService) -> type[BaseHTTPRequestHandler]:
    """Lav en request-handler, der kender vores `service`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._send_json(service.info())
            else:
                self._send_json({"error": "Siden findes ikke."}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                if self.path == "/api/generate":
                    temperature = min(max(float(body.get("temperature", 0.0)), 0.0), 2.0)
                    self._send_json(service.generate(str(body.get("idea", "")), temperature,
                                                     bool(body.get("compare", False))))
                elif self.path == "/api/critique":
                    self._send_json(service.critique(str(body.get("text", ""))[:MAX_BODY_BYTES]))
                else:
                    self._send_json({"error": "Siden findes ikke."}, HTTPStatus.NOT_FOUND)
            except (ValueError, TypeError) as error:  # dårligt input fra brugeren
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        # ---- hjælpere -------------------------------------------------------

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError("Teksten er for lang.")
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as error:
                raise ValueError("Ugyldig JSON.") from error
            if not isinstance(data, dict):
                raise TypeError("Forventede et JSON-objekt.")
            return data

        def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(data).encode(), "application/json", status)

        def _send_file(self, path: Path, content_type: str) -> None:
            self._send(path.read_bytes(), content_type)

        def _send(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            pass  # hold terminalen ren

    return Handler


def make_server(service: PrdService, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(service))


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> None:
    print("Indlæser modellerne …")
    service = PrdService.from_checkpoints()
    service.generate("warm up", compare=True)  # første kald på GPU'en er langsomt, så tiderne bliver fair
    server = make_server(service, host, port)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{server.server_port}"
    print(f"TinyPRD kører på {url}   (stop med Ctrl+C)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFarvel!")
    finally:
        server.server_close()
