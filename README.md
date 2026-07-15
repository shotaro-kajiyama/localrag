# localrag

Zero-dependency **local semantic search** over your own files — plus a tiny
**MCP server** so Claude (or any MCP client) can search them as a tool.

- **Nothing leaves your machine.** Embeddings come from any OpenAI-compatible
  endpoint; point it at [Ollama](https://ollama.com), LM Studio, llama.cpp, or
  vLLM running locally.
- **No dependencies.** Pure standard library — `urllib`, `json`, `math`. The
  index is one JSON file; search is plain cosine similarity.
- **Two ways in:** a `localrag` CLI, and an MCP server (`search_docs` tool).

## Install

```bash
pip install localrag
```

## Quick start

Assuming Ollama with an embedding model:

```bash
ollama pull nomic-embed-text

localrag build ~/notes ~/docs        # index your files (.md/.txt/.rst)
localrag query "what did I decide about the deploy pipeline?"
localrag query "deploy pipeline" --answer   # retrieve + let a chat model answer
```

Output:

```
=== top 5 for: deploy pipeline ===

[1] (0.812) notes/ops.md
    We settled on blue/green with a manual approval gate before cutover...
```

## Use it from Claude / any MCP client (the `search_docs` tool)

Run the server:

```bash
python -m localrag.mcp_server
```

Register it with an MCP client. For **Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "localrag": {
      "command": "python",
      "args": ["-m", "localrag.mcp_server"],
      "env": { "LOCALRAG_INDEX": "/home/you/notes/localrag-index.json" }
    }
  }
}
```

Now the model can call `search_docs("...")` to ground its answers in your notes.

## Configuration

All optional — sensible local defaults out of the box.

| Env var | Meaning | Default |
|---|---|---|
| `LOCALRAG_EMBED_URL` | OpenAI-compatible base URL (incl. `/v1`) | `http://localhost:11434/v1` |
| `LOCALRAG_EMBED_MODEL` | embedding model | `nomic-embed-text` |
| `LOCALRAG_CHAT_URL` | base URL for `--answer` | = `LOCALRAG_EMBED_URL` |
| `LOCALRAG_CHAT_MODEL` | chat model for `--answer` | `qwen3` |
| `LOCALRAG_API_KEY` | bearer token, if your server needs one | (none) |
| `LOCALRAG_INDEX` | index file path | `./localrag-index.json` |

Using a hosted endpoint instead of local? Point `LOCALRAG_EMBED_URL` at it and
set `LOCALRAG_API_KEY` — the same code path works with the OpenAI API.

## How it works

1. **Chunk** — files are split on blank lines into ~800-char blocks with a small
   overlap so context isn't cut mid-thought.
2. **Embed** — each chunk is embedded once and stored with its vector in a JSON
   index.
3. **Search** — the query is embedded and ranked against every chunk by cosine
   similarity. For `--answer`, the top chunks become the sole context for a
   grounded reply.

Small corpora (thousands of chunks) are the sweet spot: no database, no server,
just a file you can commit or delete.

## License

MIT — see [LICENSE](LICENSE).
