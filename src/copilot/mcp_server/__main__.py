"""Entry point for the MCP server (mcp 1.x).

Usage:
    python -m copilot.mcp_server --port 8000

DNS rebinding protection is configured on the `FastMCP` constructor (in
`server.py`), not on `streamable_http_app()`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from copilot.mcp_server.server import mcp


def _configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Act Compliance Copilot MCP server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="Bind host (default 0.0.0.0)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    args = parser.parse_args()

    _configure_logging()
    logging.getLogger(__name__).info(
        "starting MCP server on %s:%d (auth disabled, rate limit %s req / %s s)",
        args.host,
        args.port,
        os.environ.get("MCP_RATE_LIMIT", "30"),
        os.environ.get("MCP_RATE_WINDOW", "60"),
    )

    from copilot.mcp_server.server import RateLimitMiddleware

    app = mcp.streamable_http_app()
    app.add_middleware(RateLimitMiddleware)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()