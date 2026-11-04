"""Generate evals/results/timing.png with average latency bars per node.

Runs the graph on a fixed case N times, captures the JSON node logs via a
custom handler, and plots average ms per node. Requires DEEPSEEK_API_KEY
and a reachable Qdrant instance.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402

from copilot.graph.graph import compile_graph  # noqa: E402
from copilot.llm import LLMClient  # noqa: E402
from copilot.observability import setup_logging  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CASE = "my company wants to use an AI product for CV screening"
DEFAULT_RUNS = 3


class _NodeLogCollector(logging.Handler):
    """Handler that collects node logs carrying a `node` field."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        extra = getattr(record, "extra_fields", None)
        if extra and "node" in extra:
            self.rows.append(dict(extra))


def _run_and_collect(case: str, runs: int) -> list[dict]:
    """Run the graph N times and collect one row per node execution."""
    collector = _NodeLogCollector()
    logging.getLogger("copilot.nodes").addHandler(collector)
    logging.getLogger("copilot.nodes").setLevel(logging.INFO)

    llm = LLMClient()
    app = compile_graph(llm=llm)
    for i in range(runs):
        config = {"configurable": {"thread_id": f"plot:{i}"}}
        try:
            app.invoke({"user_input": case}, config=config)
        except Exception as exc:
            print(f"Run {i} failed: {exc}", file=sys.stderr)

    logging.getLogger("copilot.nodes").removeHandler(collector)
    return collector.rows


def _format_ms(value: float) -> str:
    """Human-readable ms label (e.g. 950 ms, 1.2 s)."""
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.0f} ms"


def _plot(rows: list[dict], out: Path) -> None:
    """Horizontal bar chart of average latency per node with value labels."""
    by_node: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_node[r["node"]].append(float(r["ms"]))

    nodes = list(by_node.keys())
    means = [statistics.mean(by_node[n]) for n in nodes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(nodes, means, color="#4a9eff")

    # Value labels to the right of each bar.
    max_mean = max(means) if means else 1.0
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_width() + max_mean * 0.01,
            bar.get_y() + bar.get_height() / 2,
            _format_ms(mean),
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )

    # Leave some room on the right so labels don't get clipped.
    ax.set_xlim(0, max_mean * 1.15)
    ax.set_xlabel("Average latency (ms)")
    ax.set_title("Average latency per node")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "timing.png",
    )
    args = parser.parse_args(argv)

    setup_logging("WARNING")
    rows = _run_and_collect(args.case, args.runs)
    if not rows:
        print(
            "No logs collected. Check the Qdrant connection and the API key.",
            file=sys.stderr,
        )
        return 1

    # Also save raw data as JSON for inspection.
    raw = args.out.with_suffix(".json")
    raw.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Raw data: {raw}")

    _plot(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())