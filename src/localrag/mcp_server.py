"""Minimal stdio MCP server exposing localrag as a ``search_docs`` tool.

Any MCP client (Claude Desktop, Claude Code, OpenCode, ...) can call it. Speaks
newline-delimited JSON-RPC 2.0 over stdio. Zero third-party dependencies; the
search itself uses the same index built by ``localrag build``.

    python -m localrag.mcp_server
"""
from __future__ import annotations

import json
import os
import sys

from . import core

TOOLS = [{
    "name": "search_docs",
    "description": (
        "Semantically search the user's own indexed notes and documents. "
        "Use it when you need to recall what was written or decided earlier. "
        "Natural-language queries are fine."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "what to look for (natural language)"},
            "k": {"type": "integer", "description": "number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}]


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _result(rid, res):
    _send({"jsonrpc": "2.0", "id": rid, "result": res})


def _error(rid, code, msg):
    _send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}})


def _do_search(args: dict) -> str:
    hits = core.search(args.get("query", ""), int(args.get("k", 5) or 5))
    if not hits:
        return "(no matching documents)"
    out = []
    for i, h in enumerate(hits, 1):
        try:
            loc = os.path.relpath(h["file"])
        except ValueError:
            loc = h["file"]
        out.append(f"[{i}] {loc}\n{h['text']}")
    return "\n\n".join(out)


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        method = req.get("method")
        rid = req.get("id")
        if method == "initialize":
            pv = (req.get("params") or {}).get("protocolVersion", "2025-06-18")
            _result(rid, {
                "protocolVersion": pv,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "localrag", "version": "0.1.0"},
            })
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _result(rid, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params") or {}
            if params.get("name") == "search_docs":
                try:
                    text = _do_search(params.get("arguments") or {})
                    _result(rid, {"content": [{"type": "text", "text": text}]})
                except Exception as e:
                    _result(rid, {"content": [{"type": "text", "text": f"search error: {e}"}],
                                  "isError": True})
            else:
                _error(rid, -32601, f"unknown tool: {params.get('name')}")
        elif method == "ping":
            _result(rid, {})
        elif rid is not None:
            _error(rid, -32601, f"unknown method: {method}")


if __name__ == "__main__":
    main()
