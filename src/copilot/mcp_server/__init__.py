"""MCP server for the AI Act Compliance Copilot.

Exposes the corpus and the graph as MCP tools, so an external LLM agent
(Claude Desktop, Claude.ai via a remote connector, LangChain, ...) can
search the regulation, read exact articles, and assess an AI system.

Transport: Streamable HTTP only. stdio is intentionally not implemented:
the goal is to expose the server to claude.ai via a remote connector, and
HTTP is the only transport Claude supports for remote connectors. Keeping
stdio would add a code path that is never exercised in production.
"""
