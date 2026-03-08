"""Response parsing utilities for Litmus benchmark."""

import re


def extract_answer(response: str) -> str | None:
    """Extract the answer from <answer>...</answer> tags.

    Falls back to the last non-empty line if no tags are found.

    Parameters
    ----------
    response : str
        The full model response text.

    Returns
    -------
    str | None
        The extracted answer string, or None if response is empty.
    """
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: last non-empty line
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return None


def parse_float(answer: str) -> float | None:
    """Try to parse a float from the answer string.

    Handles common formats like "3.14", "-2.5", "1.2e-3", and
    strips surrounding text.

    Parameters
    ----------
    answer : str
        The answer string to parse.

    Returns
    -------
    float | None
        The parsed float, or None if parsing fails.
    """
    if not answer:
        return None

    # Try direct parse first
    try:
        return float(answer)
    except ValueError:
        pass

    # Try to find a float pattern in the string
    match = re.search(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", answer)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass

    return None


def parse_label(answer: str, choices: list[str]) -> str | None:
    """Match the answer to one of the valid choices.

    Tries exact match first, then case-insensitive, then substring matching.

    Parameters
    ----------
    answer : str
        The answer string to match.
    choices : list[str]
        Valid choice labels.

    Returns
    -------
    str | None
        The matched choice, or None if no match is found.
    """
    if not answer or not choices:
        return None

    answer_lower = answer.lower().strip()

    # Exact match
    for choice in choices:
        if answer_lower == choice.lower():
            return choice

    # Substring: answer contains a choice
    for choice in choices:
        if choice.lower() in answer_lower:
            return choice

    # Substring: choice contains the answer
    for choice in choices:
        if answer_lower in choice.lower():
            return choice

    return None


def parse_multilabel(answer: str, choices: list[str]) -> list[str]:
    """Parse a multilabel answer — expects comma-separated labels.

    Parameters
    ----------
    answer : str
        The answer string, e.g. "label1, label2, label3".
    choices : list[str]
        Valid choice labels.

    Returns
    -------
    list[str]
        List of matched labels (may be empty).
    """
    if not answer or not choices:
        return []

    # Split by commas, semicolons, or newlines
    parts = re.split(r"[,;\n]", answer)
    matched = []
    for part in parts:
        label = parse_label(part.strip(), choices)
        if label and label not in matched:
            matched.append(label)

    return matched
