"""``localrag`` command line: build an index and query it."""
from __future__ import annotations

import argparse
import os
import sys

from . import core


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="localrag", description=core.__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="index text files under the given paths")
    b.add_argument("paths", nargs="+", help="directories or files to index")

    q = sub.add_parser("query", help="semantic search over the index")
    q.add_argument("text", help="what to look for (natural language)")
    q.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    q.add_argument("--answer", action="store_true", help="have a chat model answer from the hits")

    args = ap.parse_args(argv)

    if args.cmd == "build":
        n = core.build(args.paths)
        if n == 0:
            print("No text files found to index.", file=sys.stderr)
            return 1
        print(f"Indexed {n} chunks -> {core.index_path()}")
        return 0

    if not os.path.exists(core.index_path()):
        print(f"No index at {core.index_path()} — run `localrag build` first.", file=sys.stderr)
        return 1

    hits = core.search(args.text, args.k)
    print(f"\n=== top {len(hits)} for: {args.text} ===\n")
    for i, h in enumerate(hits, 1):
        snippet = h["text"].replace("\n", " ")[:200]
        print(f"[{i}] ({h['score']:.3f}) {os.path.relpath(h['file'])}\n    {snippet}...\n")
    if args.answer:
        print("=== answer ===")
        print(core.answer(args.text, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
