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
  gemini-2.5-flash

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

QUERY_MODELS: dict[str, str] = {
    "gemini-2.5-flash": "fast, cost-efficient, 1M token context",
}

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
SAMPLE_PROMPT = """Endianness considerations for FP32 format 
When using the FP32 blob format with vector set commands like VADD and VSIM, the binary data must be encoded in little-endian byte order. This is important for cross-platform compatibility, as some ARM variants and other architectures may use different endianness.

If your platform uses big-endian or mixed-endian encoding, you have two options:

Manually convert the byte order to little-endian before passing the blob to Redis.
Use the VALUES syntax instead, which accepts floating-point numbers as strings and is platform-independent."""



VECTOR_PREVIEW_LEN = 10
SAMPLE_TIMEOUT_SECONDS = 30.0

_SORTED_EMBEDDING_MODELS = sorted(EMBEDDING_MODELS)
_SORTED_QUERY_MODELS = sorted(QUERY_MODELS)
_ALL_KNOWN_MODELS = sorted({*EMBEDDING_MODELS, *QUERY_MODELS})


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
    """Run an embedding sample and return a professional-grade result payload.

    Args:
        client: configured GenAI client.
        model: embedding model name (must be in EMBEDDING_MODELS).

    Returns:
        Dict with keys:
            - dimensions: length of the embedding vector
            - vector: full list of float32 values (use for ingestion)
            - norm: L2 norm of the vector (sanity-check; should be ~1.0 for normalised models)
            - first_5_values: first 5 values rounded to 6 dp (human-readable preview)
            - text_preview: first 80 chars of the sample text
            - task_type: embedding task type reported by the API (if present)
            - elapsed_ms: wall-clock latency in milliseconds

    Raises:
        RuntimeError: if the API returns an empty or missing embedding.
        asyncio.TimeoutError: if the request exceeds SAMPLE_TIMEOUT_SECONDS.
    """
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: client.models.embed_content(model=model, contents=SAMPLE_TEXT),
        ),
        timeout=SAMPLE_TIMEOUT_SECONDS,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    embedding_obj = result.embeddings[0] if result.embeddings else None
    if embedding_obj is None:
        raise RuntimeError(f"API returned no embeddings for model '{model}'")

    vector: list[float] = list(embedding_obj.values) if embedding_obj.values else []
    if not vector:
        raise RuntimeError(f"API returned an empty embedding vector for model '{model}'")

    norm = round(math.sqrt(sum(v * v for v in vector)), 6)

    return {
        "elapsed_ms": elapsed_ms,
        "dimensions": len(vector),
        "norm": norm,
        "vector": vector,
        "first_5_values": [round(v, 6) for v in vector[:5]],
        "text_preview": SAMPLE_TEXT[:80],
        "task_type": getattr(embedding_obj, "task_type", None),
    }


async def _run_query(client: genai.Client, model: str) -> dict[str, Any]:
    """Run a minimal generative query sample and return result fields.

    Args:
        client: configured GenAI client.
        model: generative model name.

    Returns:
        Dict with keys:
            - response_text: stripped text from the first candidate
            - finish_reason: finish reason name from the API
            - elapsed_ms: wall-clock latency in milliseconds

    Raises:
        RuntimeError: if the API returns no candidates or an empty response.
        asyncio.TimeoutError: if the request exceeds SAMPLE_TIMEOUT_SECONDS.
    """
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: client.models.generate_content(model=model, contents=SAMPLE_PROMPT),
        ),
        timeout=SAMPLE_TIMEOUT_SECONDS,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    candidate = result.candidates[0] if result.candidates else None
    if candidate is None:
        raise RuntimeError(f"API returned no candidates for model '{model}'")

    response_text = (
        candidate.content.parts[0].text.strip()
        if candidate.content and candidate.content.parts
        else ""
    )
    if not response_text:
        raise RuntimeError(f"API returned an empty response text for model '{model}'")

    # Use .name for a clean string like "STOP" rather than "FinishReason.STOP"
    finish_reason = (
        candidate.finish_reason.name
        if hasattr(candidate.finish_reason, "name")
        else str(candidate.finish_reason)
    )

    return {
        "elapsed_ms": elapsed_ms,
        "response_text": response_text,
        "finish_reason": finish_reason,
    }


def _emit_model_help() -> str:
    """Build the --model help string dynamically from known model dicts."""
    lines = ["Model to use. Defaults to gemini-embedding-001 for embed mode, gemini-2.5-flash for query mode."]
    lines.append("Embedding models:")
    for name, desc in sorted(EMBEDDING_MODELS.items()):
        lines.append(f"  {name}: {desc}")
    lines.append("Query models:")
    for name, desc in sorted(QUERY_MODELS.items()):
        lines.append(f"  {name}: {desc}")
    return " ".join(lines)


async def _async_main() -> None:
    """Async entry point; handles argument parsing, validation, and sample dispatch."""
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
        default=None,
        choices=_ALL_KNOWN_MODELS,
        help=_emit_model_help(),
    )
    parser.add_argument(
        "--mode",
        choices=["embed", "query", "auto"],
        default="auto",
        help=(
            "Request type: embed, query, or auto (default: auto -- inferred from model name). "
            f"Known embedding models: {', '.join(_SORTED_EMBEDDING_MODELS)}. "
            f"Known query models: {', '.join(_SORTED_QUERY_MODELS)}."
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
            f"Truncate the vector in JSON output to the first N values (default: {VECTOR_PREVIEW_LEN}). "
            "Pass 0 to exclude the vector field entirely. "
            "Pass -1 to include the full vector."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all stderr progress output (stdout JSON only)",
    )
    parser.add_argument(
        "--timeout",
        metavar="SECONDS",
        type=float,
        default=SAMPLE_TIMEOUT_SECONDS,
        help=f"Request timeout in seconds (default: {SAMPLE_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key is required or set $GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Resolve mode, then default model based on mode when not explicitly given.
    mode = args.mode
    if mode == "query":
        model = args.model or _SORTED_QUERY_MODELS[0]
    elif mode == "embed":
        model = args.model or _SORTED_EMBEDDING_MODELS[0]
    else:
        # auto: infer from model, or default to embed
        model = args.model or _SORTED_EMBEDDING_MODELS[0]
        mode = "embed" if model in EMBEDDING_MODELS else "query"
    args.model = model

    # Validate: embedding mode requires a known embedding model.
    if mode == "embed" and args.model not in EMBEDDING_MODELS:
        print(
            f"Error: unknown embedding model '{args.model}'. "
            f"Known models: {', '.join(_SORTED_EMBEDDING_MODELS)}. "
            "Use --mode query for generative models.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Warn: embedding model passed with --mode query is almost certainly a mistake.
    if mode == "query" and args.model in EMBEDDING_MODELS:
        print(
            f"Warning: '{args.model}' is an embedding model but --mode query was requested. "
            "This will likely fail.",
            file=sys.stderr,
        )

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    log(f"  model:   {args.model}")
    log(f"  mode:    {mode}")
    log(f"  timeout: {args.timeout}s")
    log(f"  api_key: {args.api_key[:8]}{'*' * max(0, len(args.api_key) - 8)}")

    client = _build_client(args.api_key, args.braintrust_key)

    log("  sending sample request ...")

    try:
        if mode == "embed":
            SAMPLE_result = await _run_embed(client, args.model)
        else:
            SAMPLE_result = await _run_query(client, args.model)
    except asyncio.TimeoutError:
        output: dict[str, Any] = {
            "ok": False,
            "model": args.model,
            "mode": mode,
            "error": f"Request timed out after {args.timeout}s",
        }
        print(json.dumps(output, ensure_ascii=False))
        log(f"\nFAILED: timed out after {args.timeout}s")
        sys.exit(1)
    except Exception as exc:
        output = {
            "ok": False,
            "model": args.model,
            "mode": mode,
            "error": str(exc),
        }
        print(json.dumps(output, ensure_ascii=False))
        log(f"\nFAILED: {exc}")
        sys.exit(1)

    # Apply --vector-preview: 0 → omit field, -1 → full vector, N → first N values.
    if "vector" in SAMPLE_result:
        n = args.vector_preview
        if n == 0:
            del SAMPLE_result["vector"]
        elif n > 0:
            SAMPLE_result["vector"] = SAMPLE_result["vector"][:n]
        # n < 0: leave full vector intact

    output = {
        "ok": True,
        "model": args.model,
        "mode": mode,
        "via_braintrust": bool(args.braintrust_key),
        **SAMPLE_result,
    }

    print(json.dumps(output, ensure_ascii=False))
    if mode == "query" and "response_text" in SAMPLE_result:
        log(f"\n--- response ({SAMPLE_result['elapsed_ms']} ms, finish={SAMPLE_result.get('finish_reason', '?')}) ---")
        log(SAMPLE_result["response_text"])
        log("---")
    else:
        log(f"\n  ok: {SAMPLE_result['elapsed_ms']} ms")


def main() -> None:
    """Synchronous entry point for CLI invocation."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()