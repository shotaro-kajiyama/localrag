"""Local semantic search over your own files. Zero third-party dependencies.

Embeddings and (optional) answers are produced by any **OpenAI-compatible**
endpoint — Ollama (`http://localhost:11434/v1`), LM Studio, llama.cpp, vLLM, or
the real OpenAI API. Nothing leaves the machine unless you point it at a remote
endpoint. The index is a single JSON file; search is plain cosine similarity.

Configure via environment (all optional):

    LOCALRAG_EMBED_URL    base URL incl. /v1   (default http://localhost:11434/v1)
    LOCALRAG_EMBED_MODEL  embedding model      (default nomic-embed-text)
    LOCALRAG_CHAT_URL     base URL for --answer (default = LOCALRAG_EMBED_URL)
    LOCALRAG_CHAT_MODEL   chat model           (default qwen3)
    LOCALRAG_API_KEY      bearer token if the server needs one (default none)
    LOCALRAG_INDEX        index file path      (default ./localrag-index.json)
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request

CHUNK_CHARS = 800
OVERLAP = 150
TEXT_EXT = (".md", ".txt", ".rst")
SKIP_RE = re.compile(r"(?:^|/)(?:\.git|node_modules|\.venv|venv|__pycache__|\.env)", re.I)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def index_path() -> str:
    return _env("LOCALRAG_INDEX", os.path.join(os.getcwd(), "localrag-index.json"))


def _post(base: str, path: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("LOCALRAG_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        base.rstrip("/") + path, data=json.dumps(payload).encode(), headers=headers
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings (batched) via the OpenAI-compatible endpoint."""
    base = _env("LOCALRAG_EMBED_URL", "http://localhost:11434/v1")
    model = _env("LOCALRAG_EMBED_MODEL", "nomic-embed-text")
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        r = _post(base, "/embeddings", {"model": model, "input": texts[i:i + 32]})
        out.extend(d["embedding"] for d in r["data"])
    return out


def chunk(text: str) -> list[str]:
    """Group paragraphs into ~CHUNK_CHARS blocks with a little overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= CHUNK_CHARS:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-OVERLAP:] + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def iter_files(dirs: list[str]) -> "list[str]":
    found = []
    for d in dirs:
        if os.path.isfile(d):
            found.append(d)
            continue
        for root, _, files in os.walk(d):
            for f in files:
                full = os.path.join(root, f)
                if f.endswith(TEXT_EXT) and not SKIP_RE.search(full):
                    found.append(full)
    return found


def build(dirs: list[str]) -> int:
    """Index every text file under ``dirs``. Returns the chunk count."""
    items = []
    for full in iter_files(dirs):
        try:
            text = open(full, encoding="utf-8").read()
        except Exception:
            continue
        for ch in chunk(text):
            items.append({"file": os.path.abspath(full), "text": ch})
    if not items:
        return 0
    for it, v in zip(items, embed([it["text"] for it in items])):
        it["vec"] = v
    model = _env("LOCALRAG_EMBED_MODEL", "nomic-embed-text")
    with open(index_path(), "w", encoding="utf-8") as f:
        json.dump({"model": model, "items": items}, f, ensure_ascii=False)
    return len(items)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def search(query: str, k: int = 5) -> list[dict]:
    """Return the top-k most similar chunks from the index."""
    with open(index_path(), encoding="utf-8") as f:
        idx = json.load(f)
    qv = embed([query])[0]
    scored = [(_cos(qv, it["vec"]), it) for it in idx["items"]]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"file": it["file"], "text": it["text"], "score": s} for s, it in scored[:k]]


def answer(query: str, k: int = 5) -> str:
    """Grounded answer: retrieve, then ask a chat model to answer from context."""
    hits = search(query, k)
    ctx = "\n\n---\n\n".join(h["text"] for h in hits)
    base = _env("LOCALRAG_CHAT_URL", _env("LOCALRAG_EMBED_URL", "http://localhost:11434/v1"))
    model = _env("LOCALRAG_CHAT_MODEL", "qwen3")
    prompt = (
        "Answer the question using ONLY the excerpts below. "
        "If the answer is not in them, say so.\n\n"
        f"Excerpts:\n{ctx}\n\nQuestion: {query}"
    )
    r = _post(base, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    })
    return r["choices"][0]["message"]["content"].strip()
