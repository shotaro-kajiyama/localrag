"""Tests for localrag. A fake deterministic embedder stands in for a live server."""
import json
import subprocess
import sys

import localrag
from localrag import core


def _fake_embed(texts):
    """Bag-of-words hashed into a fixed-width vector — deterministic, offline."""
    dim = 64
    vecs = []
    for t in texts:
        v = [0.0] * dim
        for w in t.lower().split():
            v[hash(w) % dim] += 1.0
        vecs.append(v)
    return vecs


def test_chunk_overlap_and_size():
    text = "\n\n".join(f"paragraph number {i} " + "x" * 300 for i in range(6))
    chunks = core.chunk(text)
    assert len(chunks) > 1
    assert all(len(c) <= core.CHUNK_CHARS + core.OVERLAP + 50 for c in chunks)


def test_build_and_search(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("The deployment uses a blue green strategy with manual approval.")
    (tmp_path / "b.md").write_text("Breakfast recipes: pancakes, eggs, and coffee brewing tips.")
    monkeypatch.setenv("LOCALRAG_INDEX", str(tmp_path / "idx.json"))
    monkeypatch.setattr(core, "embed", _fake_embed)

    n = core.build([str(tmp_path)])
    assert n >= 2

    hits = core.search("deployment approval strategy", k=2)
    assert hits, "expected search hits"
    assert "blue green" in hits[0]["text"]  # the ops doc ranks first, not the recipes


def test_build_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALRAG_INDEX", str(tmp_path / "idx.json"))
    monkeypatch.setattr(core, "embed", _fake_embed)
    assert core.build([str(tmp_path)]) == 0


def test_mcp_server_handshake_and_list():
    """Drive the stdio MCP server: initialize -> tools/list must return search_docs."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "localrag.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    reqs = (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n"
    )
    out, _ = proc.communicate(reqs, timeout=15)
    lines = [json.loads(x) for x in out.splitlines() if x.strip()]
    init, tools = lines[0], lines[1]
    assert init["result"]["serverInfo"]["name"] == "localrag"
    names = [t["name"] for t in tools["result"]["tools"]]
    assert "search_docs" in names
