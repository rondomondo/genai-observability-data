FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /usr/local/bin/uv

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Cache dependencies independently of source so rebuilds are fast when only
# app code changes. sentence-transformers pulls torch which is large.
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --extra model --no-install-project --no-cache

COPY app/ ./app/

# Pre-download the model into the image so the container starts without
# network access at runtime. The HF_HOME path is set explicitly so the
# non-root user can write to it.
ENV HF_HOME=/app/.cache/huggingface
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}')" \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8001

CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--no-access-log"]
