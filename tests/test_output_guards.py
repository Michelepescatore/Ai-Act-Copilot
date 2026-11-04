"""Tests for the output guardrails: forbidden phrases neutralized + disclaimer."""

from __future__ import annotations

import pytest

from copilot.graph import nodes
from copilot.graph.prompts import REPORT_DISCLAIMER


@pytest.mark.parametrize("forbidden", [
    "Your system is compliant.",
    "You are fully compliant.",
    "You have nothing to do.",
    "No obligations apply.",
    "Certainly compliant.",
])
def test_forbidden_phrases_are_neutralized(forbidden: str):
    # _apply_report_guards neutralizes + appends disclaimer.
    raw = {
        "summary": forbidden,
        "facts": [],
        "articles": [],
        "paragraphs": [],
        "citations": [],
        "open_questions": [],
        "unverified_citations": [],
    }
    report = nodes._apply_report_guards(raw)

    for pat in nodes.FORBIDDEN_PATTERNS:
        assert not pat.search(report["summary"]), f"Pattern still present: {pat.pattern}"
    assert REPORT_DISCLAIMER in report["summary"]


def test_disclaimer_always_added():
    raw = {
        "summary": "A neutral analysis.",
        "facts": [], "articles": [], "paragraphs": [], "citations": [],
        "open_questions": [], "unverified_citations": [],
    }
    report = nodes._apply_report_guards(raw)
    assert report["summary"].endswith(REPORT_DISCLAIMER)


def test_unverified_warning_added():
    raw = {
        "summary": "A neutral analysis.",
        "facts": [], "articles": [], "paragraphs": [], "citations": [],
        "open_questions": [], "unverified_citations": ["art_x", "art_y"],
    }
    report = nodes._apply_report_guards(raw)
    assert "2 citation(s) could not be verified" in report["summary"]


def test_report_node_includes_disclaimer():
    state = {
        "facts": {"facts": [], "user_input": "x"},
        "candidate_articles": [],
        "paragraphs": [],
        "synthesis": {
            "summary": "Analysis.",
            "citations": [],
            "has_compliance_issue": True,
        },
        "detected_language": "en",
    }
    result = nodes.report_node(state)
    assert REPORT_DISCLAIMER in result["report"]["summary"]