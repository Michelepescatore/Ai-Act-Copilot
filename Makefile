.PHONY: install lint format typecheck test test-slow index eval-retrieval

install:
	uv sync --frozen --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/copilot

test:
	uv run pytest

index:
	uv run python -m copilot.retrieval.index

eval-retrieval:
	uv run python evals/retrieval_eval.py

eval-plot:
	uv run python evals/timing_plot.py