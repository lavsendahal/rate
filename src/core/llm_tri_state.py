from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .merlin_diseases import DISEASE_TO_QUESTION, MERLIN_DISEASES
from .tri_state_rules import TriLabel, extract_merlin_30


@dataclass
class TriStateResult:
    labels: Dict[str, int]  # disease -> {-1,0,1}
    evidence: Dict[str, List[str]]  # disease -> list of quoted substrings
    raw_payload: Dict[str, Any]


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_merlin_prompt(findings_text: str) -> str:
    disease_keys = ", ".join(MERLIN_DISEASES)
    schema_example = {
        "labels": {k: 0 for k in MERLIN_DISEASES},
        "evidence": {k: [] for k in MERLIN_DISEASES},
    }
    # The model is instructed to produce strict JSON only (no prose, no markdown).
    return (
        "You are extracting document-level disease labels from a radiology report's FINDINGS text.\n"
        "Return JSON only. No explanation, no markdown, no extra keys.\n\n"
        "Label meaning:\n"
        "  1 = present (ONLY if you can quote explicit supporting evidence)\n"
        "  0 = absent (explicitly negated or clearly stated absent)\n"
        " -1 = uncertain / equivocal / limited evaluation / cannot determine\n\n"
        "Global rules:\n"
        "- Use ONLY the provided FINDINGS text as evidence.\n"
        "- If evidence is uncertain (may/likely/possible/indeterminate/TSTC/limited eval), use -1 unless a definitive positive exists.\n"
        "- If both positive and negative statements exist, choose 1 only if there is a clear affirmative statement; otherwise -1.\n"
        "- For label=1 you MUST include at least one exact quote substring from the FINDINGS text in evidence.<disease>.\n\n"
        "Disease-specific rules:\n"
        "- coronary_calcification: label 1 ONLY if FINDINGS contains explicit 'coronary' AND a calcification cue.\n"
        "- renal_hypodensities vs renal_cyst: if renal hypodensities are interpreted as cysts, set renal_cyst=1 and renal_hypodensities=0.\n\n"
        "Output schema (keys must exactly match):\n"
        f"{json.dumps(schema_example, indent=2)}\n\n"
        f"Disease keys: {disease_keys}\n\n"
        "FINDINGS TEXT:\n"
        "-----\n"
        f"{findings_text}\n"
        "-----\n"
    )


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _evidence_in_text(evidence: str, text: str) -> bool:
    if not evidence:
        return False
    # Whitespace-normalized, case-insensitive containment check.
    ev = _normalize_ws(evidence).lower()
    tx = _normalize_ws(text).lower()
    return ev in tx


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def enforce_merlin_constraints(
    findings_text: str, labels: Dict[str, int], evidence: Dict[str, List[str]]
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """Enforce non-hallucination and cross-label gating rules on LLM output."""
    out_labels = {k: int(labels.get(k, 0)) for k in MERLIN_DISEASES}
    out_evidence: Dict[str, List[str]] = {k: list(evidence.get(k, [])) for k in MERLIN_DISEASES}

    # 1) Evidence requirement: label 1 requires at least one evidence quote that exists in text.
    for d in MERLIN_DISEASES:
        if out_labels[d] != 1:
            continue
        valid_quotes = [q for q in out_evidence[d] if _evidence_in_text(q, findings_text)]
        if not valid_quotes:
            out_labels[d] = int(TriLabel.UNCERTAIN)
            out_evidence[d] = []
        else:
            out_evidence[d] = valid_quotes[:3]

    # 2) coronary_calcification: require 'coronary' + 'calcif' in evidence (or in text as backup).
    if out_labels["coronary_calcification"] == 1:
        joined = " ".join(out_evidence["coronary_calcification"]) or findings_text
        if not (re.search(r"\bcoronary\b", joined, re.IGNORECASE) and re.search(r"\bcalcif", joined, re.IGNORECASE)):
            out_labels["coronary_calcification"] = int(TriLabel.UNCERTAIN)
            out_evidence["coronary_calcification"] = []

    # 3) renal_hypodensities gating with renal_cyst
    cyst_joined = " ".join(out_evidence["renal_cyst"])
    hyp_joined = " ".join(out_evidence["renal_hypodensities"])
    cyst_like = bool(re.search(r"\bcyst", cyst_joined, re.IGNORECASE)) or bool(
        re.search(r"\blikely\b.*\bcyst", hyp_joined, re.IGNORECASE)
    )
    if cyst_like and out_labels["renal_cyst"] != 1:
        out_labels["renal_cyst"] = 1
    if out_labels["renal_cyst"] == 1 and re.search(r"\bcyst", hyp_joined, re.IGNORECASE):
        out_labels["renal_hypodensities"] = 0
        out_evidence["renal_hypodensities"] = []

    return out_labels, out_evidence


def parse_merlin_llm_output(findings_text: str, model_text: str) -> TriStateResult:
    payload = _extract_json(model_text) or {}
    labels = payload.get("labels") if isinstance(payload.get("labels"), dict) else {}
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}

    # Coerce
    coerced_labels: Dict[str, int] = {}
    coerced_evidence: Dict[str, List[str]] = {}
    for d in MERLIN_DISEASES:
        v = labels.get(d, 0)
        try:
            iv = int(v)
        except Exception:
            iv = 0
        if iv not in (-1, 0, 1):
            iv = 0
        coerced_labels[d] = iv

        ev = evidence.get(d, [])
        if isinstance(ev, list):
            coerced_evidence[d] = [str(x) for x in ev if str(x).strip()]
        elif isinstance(ev, str):
            coerced_evidence[d] = [ev] if ev.strip() else []
        else:
            coerced_evidence[d] = []

    enforced_labels, enforced_evidence = enforce_merlin_constraints(findings_text, coerced_labels, coerced_evidence)

    # Optional: if LLM output is empty/invalid, fall back to rule baseline.
    if not payload:
        baseline = extract_merlin_30(findings_text)
        enforced_labels = {k: int(v[0]) for k, v in baseline.items()}
        enforced_evidence = {k: list(v[1]) for k, v in baseline.items()}

    return TriStateResult(labels=enforced_labels, evidence=enforced_evidence, raw_payload=payload)
