"""Cohere-compatible endpoints."""

from flask import Blueprint, request

bp = Blueprint("cohere", __name__)


@bp.route("/v2/chat", methods=["POST"])
def cohere_chat():
    return {
        "id": "cohere-mock-001",
        "finish_reason": "COMPLETE",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "This is a response from the mock server."}],
        },
        "usage": {
            "billed_units": {"input_tokens": 25, "output_tokens": 12},
            "tokens": {"input_tokens": 25, "output_tokens": 12},
        },
    }


@bp.route("/v1/chat", methods=["POST"])
def cohere_chat_v1():
    return {
        "text": "This is a response from the mock server.",
        "generation_id": "cohere-mock-001",
        "finish_reason": "COMPLETE",
        "meta": {
            "tokens": {"input_tokens": 25, "output_tokens": 12},
            "billed_units": {"input_tokens": 25, "output_tokens": 12},
        },
    }


@bp.route("/v2/embed", methods=["POST"])
def cohere_embed():
    body = request.get_json(silent=True) or {}
    texts = body.get("texts", ["Hello, world!"])
    embeddings = [[0.001] * 256 for _ in texts]
    return {
        "id": "cohere-embed-mock-001",
        "embeddings": {"float": embeddings},
        "texts": texts,
        "meta": {
            "api_version": {"version": "2"},
            "billed_units": {"input_tokens": 8},
        },
    }


@bp.route("/v1/embed", methods=["POST"])
def cohere_embed_v1():
    body = request.get_json(silent=True) or {}
    texts = body.get("texts", ["Hello, world!"])
    embeddings = [[0.001] * 256 for _ in texts]
    return {
        "response_type": "embeddings_floats",
        "id": "cohere-embed-mock-001",
        "embeddings": embeddings,
        "texts": texts,
        "meta": {
            "api_version": {"version": "1"},
            "billed_units": {"input_tokens": 8},
        },
    }
