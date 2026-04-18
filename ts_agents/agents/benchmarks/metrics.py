"""Metrics for evaluating agent performance.

This module provides functions to measure and score agent responses
against expected outcomes.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .scenarios import ExpectedOutcome, BenchmarkScenario


@dataclass
class EvaluationResult:
    """Result of evaluating an agent response against expected outcome."""

    # Scores (0.0 to 1.0)
    tool_score: float = 0.0
    content_score: float = 0.0
    reasoning_score: float = 0.0
    format_score: float = 0.0
    overall_score: float = 0.0

    # Details
    tools_used: List[str] = field(default_factory=list)
    required_tools_used: List[str] = field(default_factory=list)
    required_tools_missed: List[str] = field(default_factory=list)
    forbidden_tools_used: List[str] = field(default_factory=list)

    content_matches: List[str] = field(default_factory=list)
    content_misses: List[str] = field(default_factory=list)
    reasoning_matches: List[str] = field(default_factory=list)
    reasoning_misses: List[str] = field(default_factory=list)

    # Pass/fail
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)


def evaluate_response(
    response: str,
    tool_calls: List[str],
    expected: ExpectedOutcome,
    strict: bool = False,
) -> EvaluationResult:
    """Evaluate an agent response against expected outcome.

    Parameters
    ----------
    response : str
        The agent's text response
    tool_calls : List[str]
        Names of tools that were called
    expected : ExpectedOutcome
        Expected outcome specification
    strict : bool
        If True, all required tools must be used

    Returns
    -------
    EvaluationResult
        Detailed evaluation results
    """
    result = EvaluationResult()
    result.tools_used = tool_calls

    # Evaluate tool usage
    tool_score, tool_details = _evaluate_tool_usage(tool_calls, expected, strict)
    result.tool_score = tool_score
    result.required_tools_used = tool_details["used"]
    result.required_tools_missed = tool_details["missed"]
    result.forbidden_tools_used = tool_details["forbidden"]

    # Evaluate content
    content_score, content_details = _evaluate_content(response, expected)
    result.content_score = content_score
    result.content_matches = content_details["matches"]
    result.content_misses = content_details["misses"]

    # Evaluate reasoning quality
    reasoning_score, reasoning_details = _evaluate_reasoning(response, expected)
    result.reasoning_score = reasoning_score
    result.reasoning_matches = reasoning_details["matches"]
    result.reasoning_misses = reasoning_details["misses"]

    # Evaluate format
    format_score = _evaluate_format(response, expected)
    result.format_score = format_score

    # Calculate overall score (weighted average)
    result.overall_score = (
        0.3 * tool_score +
        0.3 * content_score +
        0.2 * reasoning_score +
        0.2 * format_score
    )

    # Determine pass/fail
    result.passed = _determine_pass(result, expected)

    # Collect failure reasons
    if result.required_tools_missed:
        result.failure_reasons.append(
            f"Missing required tools: {result.required_tools_missed}"
        )
    if result.forbidden_tools_used:
        result.failure_reasons.append(
            f"Used forbidden tools: {result.forbidden_tools_used}"
        )
    if result.content_misses:
        result.failure_reasons.append(
            f"Missing required content: {result.content_misses}"
        )
    if result.reasoning_misses:
        result.failure_reasons.append(
            f"Missing required reasoning cues: {result.reasoning_misses}"
        )

    return result


def _evaluate_tool_usage(
    tool_calls: List[str],
    expected: ExpectedOutcome,
    strict: bool,
) -> tuple:
    """Evaluate tool usage against expectations."""
    details = {"used": [], "missed": [], "forbidden": []}
    tool_set = set(tool_calls)

    # Check required tools (at least one must be used unless strict)
    required_used = []
    for tool in expected.required_tools:
        if tool in tool_set:
            required_used.append(tool)

    if expected.required_tools:
        if strict:
            # All required tools must be used
            details["used"] = required_used
            details["missed"] = [t for t in expected.required_tools if t not in tool_set]
            required_score = len(required_used) / len(expected.required_tools)
        else:
            # At least one required tool must be used
            details["used"] = required_used
            if required_used:
                required_score = 1.0
            else:
                details["missed"] = expected.required_tools
                required_score = 0.0
    else:
        required_score = 1.0  # No required tools

    # Check forbidden tools
    for tool in expected.forbidden_tools:
        if tool in tool_set:
            details["forbidden"].append(tool)

    forbidden_penalty = len(details["forbidden"]) * 0.2

    # Check tool call count bounds
    count_penalty = 0.0
    if len(tool_calls) < expected.min_tool_calls:
        count_penalty = 0.2
    elif len(tool_calls) > expected.max_tool_calls:
        count_penalty = 0.1

    score = max(0.0, required_score - forbidden_penalty - count_penalty)
    return score, details


def _evaluate_content(
    response: str,
    expected: ExpectedOutcome,
) -> tuple:
    """Evaluate response content against expectations."""
    details = {"matches": [], "misses": []}
    response_lower = response.lower()

    # Check must_contain (all must be present)
    must_contain_score = 0.0
    if expected.must_contain:
        matches = []
        for term in expected.must_contain:
            if term.lower() in response_lower:
                matches.append(term)
        details["matches"].extend(matches)
        details["misses"] = [t for t in expected.must_contain if t.lower() not in response_lower]
        must_contain_score = len(matches) / len(expected.must_contain)
    else:
        must_contain_score = 1.0

    # Check should_contain (bonus for having these)
    should_contain_score = 0.0
    if expected.should_contain:
        matches = []
        for term in expected.should_contain:
            if term.lower() in response_lower:
                matches.append(term)
        details["matches"].extend(matches)
        should_contain_score = len(matches) / len(expected.should_contain)
    else:
        should_contain_score = 1.0

    # Check must_not_contain (penalty for having these)
    must_not_penalty = 0.0
    for term in expected.must_not_contain:
        if term.lower() in response_lower:
            must_not_penalty += 0.2

    # Weighted score
    score = (0.7 * must_contain_score + 0.3 * should_contain_score) - must_not_penalty
    return max(0.0, score), details


def _evaluate_reasoning(
    response: str,
    expected: ExpectedOutcome,
) -> tuple:
    """Evaluate reasoning quality cues against expectations."""
    details = {"matches": [], "misses": []}
    response_lower = response.lower()

    must_contain_score = 0.0
    if expected.reasoning_must_contain:
        matches = []
        for term in expected.reasoning_must_contain:
            if term.lower() in response_lower:
                matches.append(term)
        details["matches"].extend(matches)
        details["misses"] = [
            term for term in expected.reasoning_must_contain if term.lower() not in response_lower
        ]
        must_contain_score = len(matches) / len(expected.reasoning_must_contain)
    else:
        must_contain_score = 1.0

    should_contain_score = 0.0
    if expected.reasoning_should_contain:
        matches = []
        for term in expected.reasoning_should_contain:
            if term.lower() in response_lower:
                matches.append(term)
        details["matches"].extend(matches)
        should_contain_score = len(matches) / len(expected.reasoning_should_contain)
    else:
        should_contain_score = 1.0

    must_not_penalty = 0.0
    for term in expected.reasoning_must_not_contain:
        if term.lower() in response_lower:
            must_not_penalty += 0.2

    score = (0.7 * must_contain_score + 0.3 * should_contain_score) - must_not_penalty
    return max(0.0, score), details


def _evaluate_format(
    response: str,
    expected: ExpectedOutcome,
) -> float:
    """Evaluate response format against expectations."""
    scores = []

    if expected.expects_number:
        # Check if response contains a number
        has_number = bool(re.search(r'\d+\.?\d*', response))
        scores.append(1.0 if has_number else 0.0)

    if expected.expects_list:
        # Check if response contains list markers
        has_list = bool(re.search(r'(?:^|\n)\s*[-*\d]+[.)\s]', response))
        scores.append(1.0 if has_list else 0.5)

    if expected.expects_table:
        # Check if response contains table-like structure
        has_table = '|' in response or bool(re.search(r'\t.*\t', response))
        scores.append(1.0 if has_table else 0.5)

    if expected.expects_recommendation:
        # Check if response contains recommendation language
        rec_patterns = [
            r'\brecommend\b', r'\bbest\b', r'\bsuggested?\b',
            r'\bshould\b', r'\bprefer\b', r'\badvise\b',
        ]
        has_rec = any(re.search(p, response.lower()) for p in rec_patterns)
        scores.append(1.0 if has_rec else 0.5)

    if not scores:
        return 1.0  # No format requirements

    return sum(scores) / len(scores)


def _determine_pass(
    result: EvaluationResult,
    expected: ExpectedOutcome,
) -> bool:
    """Determine if the evaluation result constitutes a pass."""
    # Fail if forbidden tools were used
    if result.forbidden_tools_used:
        return False

    # Fail if no required tools were used (when there are requirements)
    if expected.required_tools and not result.required_tools_used:
        return False

    # Fail if must_contain content is missing
    if result.content_misses:
        # Check if these are from must_contain
        must_misses = [m for m in result.content_misses if m in expected.must_contain]
        if must_misses:
            return False

    if result.reasoning_misses:
        return False

    # Pass if overall score is above threshold
    return result.overall_score >= 0.5


def compute_agent_metrics(
    responses: List[Dict[str, Any]],
    scenario: BenchmarkScenario,
) -> Dict[str, Any]:
    """Compute aggregate metrics for multiple agent responses.

    Parameters
    ----------
    responses : List[Dict[str, Any]]
        List of response data with 'response', 'tool_calls', 'duration_ms'
    scenario : BenchmarkScenario
        The scenario being evaluated

    Returns
    -------
    Dict[str, Any]
        Aggregate metrics
    """
    if not responses:
        return {"error": "No responses to evaluate"}

    evaluations = []
    for resp_data in responses:
        eval_result = evaluate_response(
            response=resp_data.get("response", ""),
            tool_calls=resp_data.get("tool_calls", []),
            expected=scenario.expected,
        )
        evaluations.append(eval_result)

    # Aggregate scores
    tool_scores = [e.tool_score for e in evaluations]
    content_scores = [e.content_score for e in evaluations]
    reasoning_scores = [e.reasoning_score for e in evaluations]
    overall_scores = [e.overall_score for e in evaluations]
    pass_count = sum(1 for e in evaluations if e.passed)

    # Collect all tools used
    all_tools: Set[str] = set()
    tool_frequency: Dict[str, int] = {}
    for e in evaluations:
        for tool in e.tools_used:
            all_tools.add(tool)
            tool_frequency[tool] = tool_frequency.get(tool, 0) + 1

    durations = [r.get("duration_ms", 0) for r in responses]
    token_counts = [r.get("token_count", 0) for r in responses if r.get("token_count")]

    return {
        "scenario": scenario.name,
        "difficulty": scenario.difficulty.value,
        "category": scenario.category,

        "n_evaluations": len(evaluations),
        "pass_rate": pass_count / len(evaluations),
        "pass_count": pass_count,

        "tool_score_mean": sum(tool_scores) / len(tool_scores),
        "content_score_mean": sum(content_scores) / len(content_scores),
        "reasoning_score_mean": sum(reasoning_scores) / len(reasoning_scores),
        "overall_score_mean": sum(overall_scores) / len(overall_scores),

        "tool_score_min": min(tool_scores),
        "tool_score_max": max(tool_scores),
        "reasoning_score_min": min(reasoning_scores),
        "reasoning_score_max": max(reasoning_scores),
        "overall_score_min": min(overall_scores),
        "overall_score_max": max(overall_scores),

        "tools_used": list(all_tools),
        "tool_frequency": tool_frequency,
        "avg_tool_calls": sum(len(e.tools_used) for e in evaluations) / len(evaluations),

        "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
        "total_duration_ms": sum(durations),
        "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
    }


def summarize_evaluations(
    evaluations: List[EvaluationResult],
) -> Dict[str, Any]:
    """Summarize a list of evaluation results.

    Parameters
    ----------
    evaluations : List[EvaluationResult]
        List of evaluation results

    Returns
    -------
    Dict[str, Any]
        Summary statistics
    """
    if not evaluations:
        return {"error": "No evaluations to summarize"}

    pass_count = sum(1 for e in evaluations if e.passed)

    all_failure_reasons: Dict[str, int] = {}
    for e in evaluations:
        for reason in e.failure_reasons:
            all_failure_reasons[reason] = all_failure_reasons.get(reason, 0) + 1

    return {
        "total": len(evaluations),
        "passed": pass_count,
        "failed": len(evaluations) - pass_count,
        "pass_rate": pass_count / len(evaluations),

        "avg_tool_score": sum(e.tool_score for e in evaluations) / len(evaluations),
        "avg_content_score": sum(e.content_score for e in evaluations) / len(evaluations),
        "avg_reasoning_score": sum(e.reasoning_score for e in evaluations) / len(evaluations),
        "avg_format_score": sum(e.format_score for e in evaluations) / len(evaluations),
        "avg_overall_score": sum(e.overall_score for e in evaluations) / len(evaluations),

        "failure_reasons": all_failure_reasons,
    }
