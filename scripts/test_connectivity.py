#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["google-genai>=1.0", "google-auth>=2.0", "httpx>=0.27", "anthropic>=0.40"]
# ///
"""
test_connectivity.py - Test API key / bearer token connectivity for Google GenAI.

Sends a single request to verify credentials and model are reachable. Prints
a structured JSON result to stdout; progress and errors go to stderr.

Auth modes (in priority order):
  1. --api-key KEY          Gemini API key (google.* endpoint)
  2. --access-token TOKEN   Explicit bearer token (Vertex AI / google-cloud endpoint)
  3. (neither)              Application Default Credentials via google.auth.default()

For modes 2 and 3, --project is required (Vertex endpoints are project-scoped).
--location defaults to "us-central1".

Embedding models available:
  gemini-embedding-001     (stable, recommended)
  gemini-embedding-2-preview (latest, multimodal 8k context)
  text-embedding-004       (stable, lower-cost)

04JUN2026
text-embedding-large-exp-03-07

PROJECT_ID=YOUR_PROJECT_ID
Query model example:
  gemini-2.5-flash

Exits 0 on success, 1 on any error.

Usage:
    # API key (Gemini endpoint)
    python3 test_connectivity.py --api-key AIzaSy... --model gemini-embedding-001 --mode embed

    # Explicit bearer token (Vertex AI)
    python3 test_connectivity.py --access-token $(gcloud auth print-access-token) \\
        --project my-gcp-project --model gemini-embedding-001

    # Application Default Credentials (Vertex AI)
    python3 test_connectivity.py --project my-gcp-project --model gemini-2.5-flash --mode query

    # API key via Braintrust gateway
    python3 test_connectivity.py --api-key AIzaSy... --model gemini-2.5-flash --mode query \\
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

ANTHROPIC_MODELS: set[str] = {"claude-sonnet-4-6"}
BRAINTRUST_BASE_URL = "https://braintrustproxy.com/v1"
DEFAULT_LOCATION = "us-central1"

EMBEDDING_MODELS: dict[str, str] = {
    # Gemini Series (High dimension, advanced reasoning)
    "gemini-embedding-001": "Stable first-gen text embedding; outputs up to 3072 dimensions.",
    "gemini-embedding-2-preview": "Next-gen multimodal embedding (text, image, audio, video) with an 8k context window.",
    
    # Text-Embedding Series (Cost-effective, elastic dimensions via Matryoshka learning)
    "text-embedding-004": "Stable text-only model; 768 dimensions with elastic sizing down to 256.",
    "text-embedding-005": "Optimized text-only model; enhanced retrieval performance with elastic sizing.",
}


QUERY_MODELS: dict[str, str] = {
    # Google Gemini Series (Multimodal, massive context, built-in reasoning)
    "gemini-2.5-flash-lite": "Ultra-low-cost, high-speed model ($0.10/1M input tokens); ideal for high-volume text or basic analytics.",
    "gemini-2.5-flash":      "Balanced price-performance workhorse ($0.30/1M input tokens); features enhanced reasoning, coding, and thinking tokens.",
    
    # Anthropic Claude Series (Frontier reasoning and agentic tasks)
    "claude-sonnet-4-6":     "Premium agentic tier; features a native 1M context window with elite multi-step execution and structural trace log analysis.",
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


def _build_client_api_key(api_key: str, braintrust_key: str | None) -> genai.Client:
    """Build a GenAI client authenticated with a Google API key.

    Args:
        api_key: Google API key.
        braintrust_key: when supplied, requests are proxied through Braintrust.

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


def _build_client_vertex(
    project: str,
    location: str,
    access_token: str | None,
) -> genai.Client:
    """Build a GenAI client targeting Vertex AI, using a bearer token or ADC.

    Args:
        project: GCP project ID.
        location: GCP region (e.g. "us-central1").
        access_token: explicit bearer token; when None, google.auth.default() is used.

    Returns:
        Configured genai.Client pointed at Vertex AI.

    Raises:
        SystemExit: if google-auth is missing or ADC lookup fails.
    """
    try:
        import google.auth
        import google.auth.credentials
        import google.oauth2.credentials
    except ImportError:
        print("Error: google-auth is required for bearer token / ADC auth -- pip install google-auth", file=sys.stderr)
        sys.exit(1)

    if access_token:
        credentials: google.auth.credentials.Credentials = google.oauth2.credentials.Credentials(
            token=access_token
        )
        print("  auth:    explicit bearer token", file=sys.stderr)
    else:
        try:
            credentials, detected_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except google.auth.exceptions.DefaultCredentialsError as exc:
            print(f"Error: ADC lookup failed -- {exc}", file=sys.stderr)
            print(
                "Hint: run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS",
                file=sys.stderr,
            )
            sys.exit(1)
        if not project and detected_project:
            project = detected_project
        print("  auth:    Application Default Credentials", file=sys.stderr)

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials,
    )


async def _run_embed(client: genai.Client, model: str) -> dict[str, Any]:
    """Run an embedding sample and return a result payload.

    Args:
        client: configured GenAI client.
        model: embedding model name (must be in EMBEDDING_MODELS).

    Returns:
        Dict with keys: dimensions, vector, norm, first_5_values, text_preview,
        task_type, elapsed_ms.

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
        Dict with keys: response_text, finish_reason, elapsed_ms.

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

    response_text = candidate.content.parts[0].text.strip() if candidate.content and candidate.content.parts else ""
    if not response_text:
        raise RuntimeError(f"API returned an empty response text for model '{model}'")

    finish_reason = (
        candidate.finish_reason.name if hasattr(candidate.finish_reason, "name") else str(candidate.finish_reason)
    )

    return {
        "elapsed_ms": elapsed_ms,
        "response_text": response_text,
        "finish_reason": finish_reason,
    }


async def _run_query_anthropic(api_key: str, model: str) -> dict[str, Any]:
    """Run a minimal generative query via the Anthropic SDK and return result fields.

    Args:
        api_key: Anthropic API key.
        model: Anthropic model ID (e.g. "claude-sonnet-4-6").

    Returns:
        Dict with keys: response_text, finish_reason, elapsed_ms, input_tokens, output_tokens.

    Raises:
        ImportError: if the anthropic package is not installed.
        RuntimeError: if the API returns no content.
        asyncio.TimeoutError: if the request exceeds SAMPLE_TIMEOUT_SECONDS.
    """
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic is required for Claude models -- pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": SAMPLE_PROMPT}],
            ),
        ),
        timeout=SAMPLE_TIMEOUT_SECONDS,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    if not result.content:
        raise RuntimeError(f"API returned no content for model '{model}'")

    response_text = result.content[0].text.strip() if hasattr(result.content[0], "text") else ""
    if not response_text:
        raise RuntimeError(f"API returned empty response text for model '{model}'")

    return {
        "elapsed_ms": elapsed_ms,
        "response_text": response_text,
        "finish_reason": result.stop_reason or "unknown",
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
    }


def _emit_model_help() -> str:
    lines = ["Model to use. Defaults to gemini-embedding-001 for embed mode, gemini-2.5-flash for query mode."]
    lines.append("Embedding models:")
    for name, desc in sorted(EMBEDDING_MODELS.items()):
        lines.append(f"  {name}: {desc}")
    lines.append("Query models:")
    for name, desc in sorted(QUERY_MODELS.items()):
        lines.append(f"  {name}: {desc}")
    return " ".join(lines)


async def _async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Google GenAI connectivity (API key, bearer token, or ADC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    auth_group = parser.add_argument_group("auth (pick one, or omit for ADC)")
    auth_group.add_argument(
        "--api-key",
        metavar="KEY",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Google API key for the Gemini endpoint (default: $GEMINI_API_KEY)",
    )
    auth_group.add_argument(
        "--anthropic-key",
        metavar="KEY",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key for Claude models (default: $ANTHROPIC_API_KEY)",
    )
    auth_group.add_argument(
        "--access-token",
        metavar="TOKEN",
        default=None,
        help=(
            "Explicit bearer token for Vertex AI (e.g. $(gcloud auth print-access-token)). "
            "Requires --project."
        ),
    )

    vertex_group = parser.add_argument_group("Vertex AI options (required for bearer token / ADC auth)")
    vertex_group.add_argument(
        "--project",
        metavar="PROJECT_ID",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="GCP project ID (default: $GOOGLE_CLOUD_PROJECT)",
    )
    vertex_group.add_argument(
        "--location",
        metavar="REGION",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION),
        help=f"GCP region for Vertex AI (default: {DEFAULT_LOCATION})",
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
        help=(
            "Braintrust API key to proxy requests for observability. "
            "Only valid with --api-key; ignored for Vertex AI paths."
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

    # Resolve mode then default model
    mode = args.mode
    if mode == "query":
        _SORTED_QUERY_MODELS.reverse()
        model = args.model or _SORTED_QUERY_MODELS[0]
    elif mode == "embed":
        model = args.model or _SORTED_EMBEDDING_MODELS[0]
    else:
        model = args.model or _SORTED_EMBEDDING_MODELS[0]
        mode = "embed" if model in EMBEDDING_MODELS else "query"
    args.model = model

    if mode == "embed" and args.model not in EMBEDDING_MODELS:
        print(
            f"Error: unknown embedding model '{args.model}'. "
            f"Known models: {', '.join(_SORTED_EMBEDDING_MODELS)}. "
            "Use --mode query for generative models.",
            file=sys.stderr,
        )
        sys.exit(1)

    if mode == "query" and args.model in EMBEDDING_MODELS:
        print(
            f"Warning: '{args.model}' is an embedding model but --mode query was requested. This will likely fail.",
            file=sys.stderr,
        )

    # Determine provider from model name
    use_anthropic = args.model in ANTHROPIC_MODELS
    use_vertex = False

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    log(f"  model:   {args.model}")
    log(f"  mode:    {mode}")
    log(f"  timeout: {args.timeout}s")

    if use_anthropic:
        if not args.anthropic_key:
            print(
                "Error: --anthropic-key (or $ANTHROPIC_API_KEY) is required for Claude models",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.braintrust_key:
            print("Warning: --braintrust-key is not supported for Anthropic models and will be ignored", file=sys.stderr)
        log(f"  api_key: {args.anthropic_key[:8]}{'*' * max(0, len(args.anthropic_key) - 8)}")
        auth_mode = "anthropic_api_key"
        client = None
    else:
        # Determine Google auth mode
        use_vertex = args.access_token is not None or args.api_key is None
        if use_vertex and not args.api_key and not args.project:
            print(
                "Error: --project (or $GOOGLE_CLOUD_PROJECT) is required for bearer token / ADC auth",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.braintrust_key and use_vertex:
            print("Warning: --braintrust-key is ignored when using Vertex AI auth", file=sys.stderr)
        if use_vertex:
            log(f"  project: {args.project}")
            log(f"  location:{args.location}")
            if args.access_token:
                log(f"  token:   {args.access_token[:12]}{'*' * max(0, len(args.access_token) - 12)}")
            client = _build_client_vertex(args.project, args.location, args.access_token)
            auth_mode = "access_token" if args.access_token else "adc"
        else:
            log(f"  api_key: {args.api_key[:8]}{'*' * max(0, len(args.api_key) - 8)}")
            client = _build_client_api_key(args.api_key, args.braintrust_key)
            auth_mode = "api_key"

    log("  sending sample request ...")

    try:
        if use_anthropic:
            sample_result = await _run_query_anthropic(args.anthropic_key, args.model)
        elif mode == "embed":
            sample_result = await _run_embed(client, args.model)
        else:
            sample_result = await _run_query(client, args.model)
    except TimeoutError:
        output: dict[str, Any] = {
            "ok": False,
            "model": args.model,
            "mode": mode,
            "auth": auth_mode,
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
            "auth": auth_mode,
            "error": str(exc),
        }
        print(json.dumps(output, ensure_ascii=False))
        log(f"\nFAILED: {exc}")
        sys.exit(1)

    if "vector" in sample_result:
        n = args.vector_preview
        if n == 0:
            del sample_result["vector"]
        elif n > 0:
            sample_result["vector"] = sample_result["vector"][:n]

    output = {
        "ok": True,
        "model": args.model,
        "mode": mode,
        "auth": auth_mode,
        "via_braintrust": bool(args.braintrust_key) and not use_anthropic,
        **({"project": args.project, "location": args.location} if not use_anthropic and use_vertex else {}),
        **sample_result,
    }

    print(json.dumps(output, ensure_ascii=False))
    if mode == "query" and "response_text" in sample_result:
        log(f"\n--- response ({sample_result['elapsed_ms']} ms, finish={sample_result.get('finish_reason', '?')}) ---")
        log(sample_result["response_text"])
        log("---")
    else:
        log(f"\n  ok: {sample_result['elapsed_ms']} ms")


def main() -> None:
    """Synchronous entry point for CLI invocation."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
