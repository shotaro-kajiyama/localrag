"""localrag — zero-dependency local semantic search over your own files, with an MCP server."""
from .core import answer, build, chunk, embed, index_path, search

__all__ = ["build", "search", "answer", "embed", "chunk", "index_path"]
__version__ = "0.1.0"
