"""GET /api/workspace: the public, read-only React dashboard projection."""
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
import json
from urllib.parse import parse_qsl, urlsplit

from frontend.workspace_data import workspace


FILTERS = frozenset({"from", "to", "model", "currency", "rate"})


def _options(target):
    query = urlsplit(target).query
    if len(query) > 2048:
        raise ValueError("Invalid filters.")
    pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True,
                      errors="strict", max_num_fields=len(FILTERS))
    options = dict(pairs)
    if (len(options) != len(pairs) or options.keys() - FILTERS
            or any(len(value) > 128 for value in options.values())):
        raise ValueError("Invalid filters.")
    return options


class handler(BaseHTTPRequestHandler):
    """Vercel's standard-library Python function entry point."""

    def _reply(self, status, payload):
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == HTTPStatus.METHOD_NOT_ALLOWED:
            self.send_header("Allow", "GET, HEAD")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        try:
            options = _options(self.path)
        except (UnicodeError, ValueError):
            self._reply(HTTPStatus.BAD_REQUEST, {"error": "Invalid workspace filters."})
            return
        try:
            self._reply(HTTPStatus.OK, workspace(options, include_local_history=False))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self._reply(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Workspace unavailable."})

    do_HEAD = do_GET

    def send_error(self, code, message=None, explain=None):
        # The base class dispatches unsupported methods through send_error(501).
        if code == HTTPStatus.NOT_IMPLEMENTED:
            self._reply(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "Method not allowed."})
        else:
            self._reply(code, {"error": "Invalid request."})

    def log_message(self, format, *args):
        # Request targets may contain sensitive values supplied by callers.
        pass
