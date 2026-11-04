# syntax=docker/dockerfile:1.7

# --- Build stage: install dependencies with uv ---
FROM python:3.12-slim AS builder

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /uvx /bin/

# Copy the project files needed to resolve dependencies.
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtualenv at /app/.venv.
# NOTE: the extra names (`mcp`, `retrieval`, `graph`) are the ones in
# `pyproject.toml`. If your project uses different extra names, update
# this line.
RUN uv sync --frozen --no-dev --extra mcp --extra retrieval --extra graph

# --- Runtime stage ---
FROM python:3.12-slim AS runtime

# Non-root user.
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Copy the virtualenv and the source.
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/

# Make the venv the default Python.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Run as the non-root user.
USER app

EXPOSE 8000

# Healthcheck: the MCP endpoint responds with 401 (no token) if auth is on,
# which is a non-error for the connection check. If auth is off, it responds
# 200. We accept either: we just need the socket to be up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import socket; s=socket.create_connection(('127.0.0.1', 8000), timeout=3); s.close()" || exit 1

ENTRYPOINT ["python", "-m", "copilot.mcp_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]