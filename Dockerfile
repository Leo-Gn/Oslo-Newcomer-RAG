FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "oslo_newcomer_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
