#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["google-genai>=1.0", "redis>=5.0"]
# ///
"""
test_vecstore.py - Embed text with Google GenAI and store/query via Redis Vector Sets.

Generates an embedding, stores it with Q8 quantization in a Redis vector set,
then performs a similarity search to verify the round-trip. Prints structured
JSON to stdout; progress goes to stderr.

Requires Redis 8+ (vector sets are native - no RediSearch module needed).

Usage:
    # Store and self-query
    python3 test_vecstore.py --text "Redis vector sets use HNSW for ANN"

    # Store with a label, then query with a different string
    python3 test_vecstore.py --text "Redis HNSW ANN" --source runbook --query "nearest neighbour search"

    # Store with truncated dimensions (faster, smaller)
    python3 test_vecstore.py --text "..." --dim 512

    # Filter search to a source tag
    python3 test_vecstore.py --query "vector search" --filter '.source == "runbook"'

    # List all stored elements
    python3 test_vecstore.py --list

    # Wipe the vector set
    python3 test_vecstore.py --flush

Exits 0 on success, 1 on any error.
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("Error: google-genai is required -- pip install google-genai", file=sys.stderr)
    sys.exit(1)

try:
    import redis as redis_lib
    from redis.commands.vectorset.commands import QuantizationOptions
except ImportError:
    print("Error: redis>=5.0 is required -- pip install redis", file=sys.stderr)
    sys.exit(1)

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIM = 768
DEFAULT_VSET_KEY = "genai:embeddings"
DEFAULT_REDIS_URL = "redis://localhost:6379"

SAMPLE_TEXT = "Redis vector sets use HNSW for approximate nearest-neighbour search."
SAMPLE_QUERY = "nearest neighbour search in Redis"

VECSTORE_NOTES = """
Redis Vector Set tuning reference
----------------------------------
Quantization  Size vs FP32  Recall    Use when
NOQUANT       1x            Perfect   Exact precision required
Q8            4x smaller    ~99.99%   Default sweet spot  <-- this script uses Q8
BIN           32x smaller   Lower     Speed/memory critical, recall less important

Q8 reduces memory ~75%, improves search speed ~30%, recall loss is negligible.

Dimensions   Model examples
384-dim      all-MiniLM-L6-v2 -- very memory efficient
768-dim      most BERT-family models, solid general purpose  <-- this script default
1536-dim     OpenAI text-embedding-3-small
3072-dim     OpenAI text-embedding-3-large, Gemini embedding-001 max

HNSW graph: each node costs ~M*2 + M*0.33 pointers at 8 bytes each (default M=16).
Lower M to shrink per-node memory at the cost of recall.

Performance ballpark: ~50k similarity queries/s for 3M items at 300-dim with Q8.
VSIM scales nearly linearly with CPU cores up to 32 threads.

TL;DR for RAG/semantic search: Q8 + 768-dim + M=16 -- Redis-recommended baseline.
"""

EMBEDDING_MODELS: dict[str, str] = {
    "gemini-embedding-001": "stable, recommended (3072 dims max)",
    "gemini-embedding-2-preview": "latest generation, multimodal 8k context (3072 dims max)",
    "text-embedding-004": "stable, lower-cost (768 dims max)",
}


def _slug(text: str) -> str:
    """Convert arbitrary text to a short Redis-safe element ID."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())[:48].strip("-")
    return slug or "doc"


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


async def _embed(client: genai.Client, model: str, text: str, dim: int) -> list[float]:
    """Return an embedding vector, optionally truncated to dim dimensions.

    Args:
        client: configured GenAI client.
        model: embedding model name.
        text: text to embed.
        dim: output dimensionality; passed to the API as output_dimensionality
            so truncation is done server-side (cheaper than slicing locally).

    Returns:
        List of float32 values of length dim.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: client.models.embed_content(
            model=model,
            contents=text,
            config=genai_types.EmbedContentConfig(output_dimensionality=dim),
        ),
    )
    if not result.embeddings:
        raise RuntimeError("API returned no embeddings")
    return list(result.embeddings[0].values)


def _store(
    r: redis_lib.Redis,
    vset_key: str,
    element_id: str,
    vector: list[float],
    metadata: dict[str, Any],
) -> None:
    """Store a vector and its metadata in a Redis vector set.

    Uses Q8 quantization (4x memory reduction, minimal accuracy loss).

    Args:
        r: Redis client.
        vset_key: name of the vector set key.
        element_id: unique element identifier.
        vector: embedding values.
        metadata: JSON-serialisable dict stored as element attributes.
    """
    r.vset().vadd(vset_key, vector, element_id, quantization=QuantizationOptions.Q8)
    r.vset().vsetattr(vset_key, element_id, json.dumps(metadata, ensure_ascii=False))


def _search(
    r: redis_lib.Redis,
    vset_key: str,
    query_vector: list[float],
    top_k: int,
    filter_expr: str | None,
) -> list[dict[str, Any]]:
    """Search the vector set and return ranked results with attributes.

    Args:
        r: Redis client.
        vset_key: name of the vector set key.
        query_vector: embedding to search against.
        top_k: number of nearest neighbours to return.
        filter_expr: optional Redis filter expression, e.g. '.source == "runbook"'.

    Returns:
        List of dicts with keys: element, score, attrs.
    """
    hits: dict[str, float] = r.vset().vsim(
        vset_key,
        query_vector,
        count=top_k,
        with_scores=True,
        filter=filter_expr,
    )
    results = []
    for element, score in hits.items():
        raw_attrs = r.vset().vgetattr(vset_key, element)
        attrs = raw_attrs if isinstance(raw_attrs, dict) else (json.loads(raw_attrs) if raw_attrs else {})
        results.append({"element": element, "score": round(float(score), 6), "attrs": attrs})
    return results


async def _async_main() -> None:
    """Async entry point: parse args, embed, store, and search."""
    parser = argparse.ArgumentParser(
        description="Embed text with Google GenAI and store/query via Redis Vector Sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Google API key (default: $GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=DEFAULT_MODEL,
        choices=sorted(EMBEDDING_MODELS),
        help=(
            f"Embedding model (default: {DEFAULT_MODEL}). "
            + " | ".join(f"{k}: {v}" for k, v in sorted(EMBEDDING_MODELS.items()))
        ),
    )
    parser.add_argument(
        "--dim",
        metavar="N",
        type=int,
        default=DEFAULT_DIM,
        help=(
            f"Output dimensionality (default: {DEFAULT_DIM}). "
            "Passed to the API as output_dimensionality -- truncation is server-side. "
            "Lower values (e.g. 512, 256) are faster and use less memory with moderate recall loss."
        ),
    )
    parser.add_argument(
        "--text",
        metavar="TEXT",
        help="Text to embed and store in the vector set",
    )
    parser.add_argument(
        "--source",
        metavar="TAG",
        default="cli",
        help="Source label stored as metadata (default: cli). Used for filter expressions.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Text to embed and search against the vector set (default: same as --text)",
    )
    parser.add_argument(
        "--filter",
        metavar="EXPR",
        dest="filter_expr",
        default=None,
        help=(
            "Redis filter expression applied during VSIM search. "
            "Examples: '(.source == \"runbook\")' or '(.stored_at > 1700000000)'"
        ),
    )
    parser.add_argument(
        "--top-k",
        metavar="N",
        type=int,
        default=5,
        help="Number of nearest neighbours to return (default: 5)",
    )
    parser.add_argument(
        "--key",
        metavar="KEY",
        default=DEFAULT_VSET_KEY,
        help=f"Redis vector set key (default: {DEFAULT_VSET_KEY})",
    )
    parser.add_argument(
        "--redis-url",
        metavar="URL",
        default=DEFAULT_REDIS_URL,
        help=f"Redis connection URL (default: {DEFAULT_REDIS_URL})",
    )
    parser.add_argument(
        "--list",
        dest="list_members",
        action="store_true",
        help="List all elements in the vector set with their attributes and exit",
    )
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Delete the entire vector set and exit",
    )
    parser.add_argument(
        "--json",
        dest="force_json",
        action="store_true",
        help="Suppress stderr progress output (stdout JSON only)",
    )
    args = parser.parse_args()

    def log(msg: str) -> None:
        if not args.force_json:
            print(msg, file=sys.stderr)

    try:
        r: redis_lib.Redis = redis_lib.Redis.from_url(args.redis_url, decode_responses=True)
        r.ping()
        log(f"  redis:   {args.redis_url}")
    except redis_lib.RedisError as exc:
        print(f"Error: cannot connect to Redis -- {exc}", file=sys.stderr)
        sys.exit(1)

    if args.flush:
        deleted = r.delete(args.key)
        print(
            json.dumps({"ok": True, "action": "flush", "key": args.key, "deleted": bool(deleted)}, ensure_ascii=False)
        )
        return

    if args.list_members:
        try:
            members = r.execute_command("VRANDMEMBER", args.key, 1000) or []
            card = r.execute_command("VCARD", args.key)
            items = []
            for m in members:
                raw = r.vset().vgetattr(args.key, m)
                attrs = raw if isinstance(raw, dict) else (json.loads(raw) if raw else {})
                items.append({"element": m, "attrs": attrs})
        except redis_lib.ResponseError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            sys.exit(1)
        print(
            json.dumps(
                {"ok": True, "action": "list", "key": args.key, "cardinality": card, "members": items},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not args.text and not args.query:
        args.text = SAMPLE_TEXT
        args.query = SAMPLE_QUERY
        log("  (no --text/--query given; running built-in sample round-trip)")
        log(VECSTORE_NOTES)

    if not args.api_key:
        print("Error: --api-key is required or set $GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=args.api_key)
    query_text = args.query or args.text
    output: dict[str, Any] = {"ok": True}

    # Store path
    if args.text:
        element_id = _slug(args.text)
        log(f"  model:   {args.model}  dim={args.dim}")
        log(f"  text:    {args.text[:72]}")
        log(f"  element: {element_id}")
        log("  embedding ...")

        t0 = time.perf_counter()
        try:
            doc_vector = await _embed(client, args.model, args.text, args.dim)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            sys.exit(1)
        embed_ms = round((time.perf_counter() - t0) * 1000)
        log(f"  dimensions={len(doc_vector)}, norm={_norm(doc_vector):.4f}, {embed_ms} ms")

        metadata: dict[str, Any] = {
            "text": args.text,
            "source": args.source,
            "model": args.model,
            "dim": len(doc_vector),
            "stored_at": int(time.time()),
        }

        log(f"  storing in '{args.key}' with Q8 quantization ...")
        try:
            _store(r, args.key, element_id, doc_vector, metadata)
            cardinality = r.execute_command("VCARD", args.key)
        except redis_lib.RedisError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            sys.exit(1)

        output["stored"] = {
            "element": element_id,
            "key": args.key,
            "cardinality": cardinality,
            "dimensions": len(doc_vector),
            "norm": round(_norm(doc_vector), 6),
            "quantization": "Q8",
            "embed_ms": embed_ms,
            "metadata": metadata,
        }

    # Search path
    if query_text:
        log(f"  embedding query: {query_text[:72]} ...")
        t1 = time.perf_counter()
        try:
            query_vector = await _embed(client, args.model, query_text, args.dim)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            sys.exit(1)
        query_embed_ms = round((time.perf_counter() - t1) * 1000)

        log(f"  searching top-{args.top_k}" + (f" filter={args.filter_expr!r}" if args.filter_expr else "") + " ...")
        try:
            results = _search(r, args.key, query_vector, args.top_k, args.filter_expr)
        except redis_lib.RedisError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            sys.exit(1)

        output["query"] = {
            "text": query_text,
            "embed_ms": query_embed_ms,
            "top_k": args.top_k,
            "filter": args.filter_expr,
            "results": results,
        }

        if results:
            top = results[0]
            log(f"\n  ok  top={top['element']}  score={top['score']:.4f}")

    print(json.dumps(output, indent=2, ensure_ascii=False))


def main() -> None:
    """Synchronous CLI entry point."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
