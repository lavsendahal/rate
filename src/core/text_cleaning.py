from __future__ import annotations

import re


_HEADING_RE = re.compile(
    r"(?im)^\s*(history|clinical history|indication|technique|comparison|findings|impression)\s*:\s*"
)


def extract_findings_impression(text: str) -> str:
    """Extract a Findings(+Impression) region from mixed-format radiology report text.

    Rules:
    - If a "Findings:" heading exists, return text from that heading through the end,
      which naturally includes "Impression:" if present later.
    - Otherwise, return the original text unchanged.

    This is intentionally conservative: it does not attempt to reconstruct missing headings.
    """
    if not text:
        return ""

    m = re.search(r"(?i)\bfindings\s*:\s*", text)
    if not m:
        return text

    # Keep from "Findings:" to end (includes Impression if present).
    return text[m.start() :].strip()


def remove_history_indication_blocks(text: str) -> str:
    """Remove History/Indication/Clinical History blocks when they appear as headed sections.

    This is useful when the input already contains Findings/Impression, but may embed
    other headed sections.
    """
    if not text:
        return ""

    lines = text.splitlines()
    out_lines = []
    skipping = False

    def is_start(line: str) -> bool:
        return bool(re.match(r"(?i)^\s*(history|clinical history|indication)\s*:?\s*$", line.strip()))

    def is_heading(line: str) -> bool:
        # Any heading like "FINDINGS:" or "IMPRESSION:" etc.
        return bool(re.match(r"(?i)^\s*(history|clinical history|indication|technique|comparison|findings|impression)\s*:?\s*$", line.strip()))

    for line in lines:
        if is_start(line):
            skipping = True
            continue
        if skipping and is_heading(line):
            skipping = False
        if not skipping:
            out_lines.append(line)

    return "\n".join(out_lines).strip()

