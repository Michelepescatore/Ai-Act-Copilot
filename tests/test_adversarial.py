"""Robustness tests against prompt injection.

With fake_llm we verify that:
1. the injection pattern is detected (`suspicious_input=True`);
2. the flow proceeds without errors;
3. the risk_tier returned by the fake is not altered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.graph import nodes
from copilot.graph.state import GraphState

ADV_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "adversarial.jsonl"


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in ADV_PATH.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_adversarial_inputs_do_not_change_risk_tier(case, fake_llm):
    # The fake returns correct facts (high_risk). If the pattern is detected,
    # the flow proceeds without altering the facts.
    fake = fake_llm(
        [
            {
                "facts": [
                    {
                        "field": "operator_role",
                        "value": "provider",
                        "status": "known",
                        "evidence": "",
                        "rationale": "Explicit.",
                        "required": True,
                    },
                    {
                        "field": "risk_tier",
                        "value": case["expected_tier"],
                        "status": "known",
                        "evidence": "",
                        "rationale": "High-risk under Annex III.",
                        "required": True,
                    },
                ]
            }
        ]
    )
    state: GraphState = {
        "user_input": case["query"],
        "user_input_en": case["query"],
        "detected_language": "en",
    }
    result = nodes.fact_extraction(state, llm=fake)
    assert "error" not in result
    assert result["suspicious_input"] is True
    risk = next(f for f in result["facts"]["facts"] if f["field"] == "risk_tier")
    assert risk["value"] == case["expected_tier"]