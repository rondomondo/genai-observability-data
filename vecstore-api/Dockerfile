FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /usr/local/bin/uv

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies from lock file before copying source so this layer is
# cache-valid as long as pyproject.toml and uv.lock are unchanged.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

COPY app/ ./app/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
