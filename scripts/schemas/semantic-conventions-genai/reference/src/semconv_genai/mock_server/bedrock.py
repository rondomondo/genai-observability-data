"""AWS Bedrock-compatible endpoints."""

import json

from flask import Blueprint, Response

from ._common import encode_aws_event_stream_message

bp = Blueprint("bedrock", __name__)


CONVERSE_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "This is a response from the mock server."}],
        }
    },
    "stopReason": "end_turn",
    "usage": {
        "inputTokens": 25,
        "outputTokens": 12,
        "totalTokens": 37,
    },
    "metrics": {"latencyMs": 100},
}


def _stream_converse():
    """Yield Bedrock ConverseStream event-stream chunks in binary format."""
    events = []
    events.append(("messageStart", {"role": "assistant"}))
    for word in ["This ", "is ", "a ", "mock ", "streamed ", "response."]:
        events.append(("contentBlockDelta", {"delta": {"text": word}, "contentBlockIndex": 0}))
    events.append(("contentBlockStop", {"contentBlockIndex": 0}))
    events.append(("messageStop", {"stopReason": "end_turn"}))
    events.append(
        (
            "metadata",
            {
                "usage": {"inputTokens": 25, "outputTokens": 6, "totalTokens": 31},
                "metrics": {"latencyMs": 100},
            },
        )
    )
    for event_type, body in events:
        payload = json.dumps(body).encode("utf-8")
        yield encode_aws_event_stream_message(event_type, payload)


@bp.route("/model/<path:model_id>/converse", methods=["POST"])
def bedrock_converse(model_id):
    return CONVERSE_RESPONSE


@bp.route("/model/<path:model_id>/converse-stream", methods=["POST"])
def bedrock_converse_stream(model_id):
    return Response(_stream_converse(), mimetype="application/vnd.amazon.eventstream")


@bp.route("/model/<path:model_id>/invoke", methods=["POST"])
def bedrock_invoke(model_id):
    """Handle Bedrock InvokeModel — used for Titan Embeddings."""
    # Amazon Titan Embeddings response format
    resp = {
        "embedding": [0.001] * 256,
        "inputTextTokenCount": 8,
    }
    return Response(json.dumps(resp), mimetype="application/json")
