from __future__ import annotations

"""
Minimal subset of the ``requests`` API used by ``pipeline.llm.client``.

Implements:
  - post(url, timeout=..., headers=..., json=...)
  - RequestException
  - Response.json()
"""

import json as _json
from dataclasses import dataclass
from typing import Any, Dict
import urllib.error
import urllib.request


class RequestException(Exception):
    """Base exception for HTTP errors in this minimal client."""


@dataclass
class Response:
    status_code: int
    text: str

    def json(self) -> Any:
        return _json.loads(self.text)


def post(url: str, *, timeout: float, headers: Dict[str, str], json: Any) -> Response:
    """Perform a minimal HTTP POST request with a JSON body."""
    data = _json.dumps(json).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            status = resp.getcode() or 0
            body_bytes = resp.read()
            text = body_bytes.decode("utf-8", errors="replace")
            return Response(status_code=status, text=text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return Response(status_code=exc.code, text=body)
    except Exception as exc:  # pragma: no cover
        raise RequestException(str(exc)) from exc

