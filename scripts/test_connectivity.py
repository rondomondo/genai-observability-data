#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["google-genai>=1.0", "httpx>=0.27"]
# ///
"""
test_connectivity.py - Test API key connectivity for Google GenAI (query or embedding).

Sends a single request to verify the API key and model are reachable. Prints
a structured JSON result to stdout; progress and errors go to stderr.

Embedding models available:
  gemini-embedding-001     (stable, recommended)
  gemini-embedding-2       (latest)
  text-embedding-004       (stable, lower-cost)

Query model example:
  gemini-2.0-flash

Exits 0 on success, 1 on any error.

Usage:
    # Embedding via default Google endpoint
    python3 test_connectivity.py --api-key AIzaSy... --model gemini-embedding-001 --mode embed

    # Query via Braintrust gateway
    python3 test_connectivity.py --api-key AIzaSy... --model gemini-2.0-flash --mode query \\
        --braintrust-key bt-...

    # Pipe the JSON result into another tool
    python3 test_connectivity.py --api-key AIzaSy... --model gemini-embedding-001 | jq .dimensions

"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("Error: google-genai is required -- pip install google-genai", file=sys.stderr)
    sys.exit(1)

BRAINTRUST_BASE_URL = "https://braintrustproxy.com/v1"

EMBEDDING_MODELS: dict[str, str] = {
    "gemini-embedding-001": "stable, recommended for production use",
    "gemini-embedding-2": "latest generation, may have breaking changes",
    "text-embedding-004": "stable, lower-cost, good for high-volume workloads",
}

PROBE_TEXT = "The quick brown fox jumps over the lazy dog."
PROBE_PROMPT = "Reply with exactly the word: connected"
VECTOR_PREVIEW_LEN = 10
_SORTED_EMBEDDING_MODELS = sorted(EMBEDDING_MODELS)
print(_SORTED_EMBEDDING_MODELS)
def _build_client(api_key: str, braintrust_key: str | None) -> genai.Client:
    """Build a GenAI client, optionally routed through the Braintrust gateway.

    Args:
        api_key: Google API key.
        braintrust_key: Braintrust API key; when supplied, all requests are
            proxied through the Braintrust gateway for observability.

    Returns:
        Configured genai.Client.
    """
    if braintrust_key:
        print(f"  gateway: Braintrust proxy ({BRAINTRUST_BASE_URL})", file=sys.stderr)
        http_options = genai_types.HttpOptions(
            base_url=BRAINTRUST_BASE_URL,
            headers={"x-bt-use-cache": "never", "Authorization": f"Bearer {braintrust_key}"},
        )
        return genai.Client(api_key=api_key, http_options=http_options)
    return genai.Client(api_key=api_key)


async def _run_embed(client: genai.Client, model: str) -> dict[str, Any]:
    """Run an embedding probe and return a professional-grade result payload.

    Args:
        client: configured GenAI client.
        model: embedding model name (must be in EMBEDDING_MODELS).

    Returns:
        Dict with keys:
            - dimensions: length of the embedding vector
            - vector: full list of float32 values (use for ingestion)
            - norm: L2 norm of the vector (sanity-check; should be ~1.0 for normalised models)
            - first_5_values: first 5 values rounded to 6 dp (human-readable preview)
            - text_preview: first 80 chars of the probe text
            - task_type: embedding task type reported by the API (if present)
            - elapsed_ms: wall-clock latency in milliseconds
    """
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    result = await loop.run_in_executor(
        None,
        lambda: client.models.embed_content(model=model, contents=PROBE_TEXT),
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    embedding_obj = result.embeddings[0] if result.embeddings else None
    vector: list[float] = list(embedding_obj.values) if embedding_obj else []

    norm = round(math.sqrt(sum(v * v for v in vector)), 6) if vector else 0.0

    return {
        "elapsed_ms": elapsed_ms,
        "dimensions": len(vector),
        "norm": norm,
        "vector": vector,
        "first_5_values": [round(v, 6) for v in vector[:5]],
        "text_preview": PROBE_TEXT[:80],
        "task_type": getattr(embedding_obj, "task_type", None),
    }


async def _run_query(client: genai.Client, model: str) -> dict[str, Any]:
    """Run a minimal generative query probe and return result fields.

    Args:
        client: configured GenAI client.
        model: generative model name.

    Returns:
        Dict with keys:
            - response_text: stripped text from the first candidate
            - finish_reason: string representation of the finish reason
            - elapsed_ms: wall-clock latency in milliseconds
    """
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    result = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(model=model, contents=PROBE_PROMPT),
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    candidate = result.candidates[0] if result.candidates else None
    response_text = candidate.content.parts[0].text.strip() if candidate else ""
    finish_reason = str(candidate.finish_reason) if candidate else "unknown"

    return {
        "elapsed_ms": elapsed_ms,
        "response_text": response_text,
        "finish_reason": finish_reason,
    }


def _emit_model_help() -> str:
    """Build the --model help string dynamically from EMBEDDING_MODELS."""
    lines = ["Model to use (default: gemini-embedding-001)."]
    lines.append("Embedding models:")
    for name, desc in sorted(EMBEDDING_MODELS.items()):
        lines.append(f"  {name}: {desc}")
    lines.append("Query models: gemini-2.0-flash, gemini-2.0-pro, etc. (any generative model).")
    return " ".join(lines)


async def _async_main() -> None:
    """Async entry point; handles argument parsing, validation, and probe dispatch."""
    parser = argparse.ArgumentParser(
        description="Test Google GenAI API key connectivity (embed or query)",
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
        default="gemini-embedding-001",
        choices=_SORTED_EMBEDDING_MODELS,
        help=_emit_model_help(),
    )
    parser.add_argument(
        "--mode",
        choices=["embed", "query", "auto"],
        default="auto",
        help=(
            "Request type: embed, query, or auto (default: auto -- inferred from model name). "
            f"Known embedding models: {', '.join(_SORTED_EMBEDDING_MODELS)}."
        ),
    )
    parser.add_argument(
        "--braintrust-key",
        metavar="KEY",
        default=None,
        required=False,
        help=(
            "Braintrust API key to route requests via the Braintrust gateway for observability. "
            "Must be supplied explicitly; never read from the environment."
        ),
    )
    parser.add_argument(
        "--vector-preview",
        metavar="N",
        type=int,
        default=VECTOR_PREVIEW_LEN,
        help=(
            "Truncate the vector in JSON output to the first N values (e.g. 5, 8, 10). "
            "Omit to include the full vector; pass 0 to exclude it entirely."
        ),
    )
    parser.add_argument(
        "--json",
        dest="force_json",
        action="store_true",
        help="Suppress all stderr progress output (stdout JSON only)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key is required or set $GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)

    mode = args.mode
    if mode == "auto":
        mode = "embed" if args.model in EMBEDDING_MODELS else "query"

    if mode == "query" and args.model in EMBEDDING_MODELS:
        print(
            f"Warning: model '{args.model}' is an embedding model but --mode query was requested.",
            file=sys.stderr,
        )

    def log(msg: str) -> None:
        if not args.force_json:
            print(msg, file=sys.stderr)

    log(f"  model:   {args.model}")
    log(f"  mode:    {mode}")
    log(f"  api_key: {args.api_key[:8]}{'*' * max(0, len(args.api_key) - 8)}")

    try:
        client = _build_client(args.api_key, args.braintrust_key)
    except Exception as exc:
        print(f"Error: failed to build client -- {exc}", file=sys.stderr)
        sys.exit(1)

    log("  sending probe request ...")

    try:
        if mode == "embed":
            probe_result = await _run_embed(client, args.model)
        else:
            probe_result = await _run_query(client, args.model)
    except Exception as exc:
        output: dict[str, Any] = {
            "ok": False,
            "model": args.model,
            "mode": mode,
            "error": str(exc),
        }
        print(json.dumps(output, ensure_ascii=False))
        log(f"\nFAILED: {exc}")
        sys.exit(1)

    if args.vector_preview is not None and "vector" in probe_result:
        probe_result["vector"] = probe_result["vector"][:args.vector_preview]

    output = {
        "ok": True,
        "model": args.model,
        "mode": mode,
        "via_braintrust": bool(args.braintrust_key),
        **probe_result,
    }

    print(json.dumps(output, ensure_ascii=False))
    log(f"\n  ok: {probe_result['elapsed_ms']} ms")


def main() -> None:
    """Synchronous entry point for CLI invocation."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
