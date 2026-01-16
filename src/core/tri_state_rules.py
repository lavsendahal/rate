from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List, Sequence, Tuple


class TriLabel(IntEnum):
    ABSENT = 0
    PRESENT = 1
    UNCERTAIN = -1


@dataclass(frozen=True)
class Evidence:
    text: str
    kind: str  # "present" | "absent" | "uncertain"


NEGATION_CUES = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bno evidence of\b",
    r"\bno ct evidence of\b",
    r"\babsent\b",
    r"\bfree of\b",
    r"\bnot seen\b",
]

UNCERTAINTY_CUES = [
    r"\bcannot exclude\b",
    r"\bcan(?:not|'t) rule out\b",
    r"\bmay represent\b",
    r"\bpossibly\b",
    r"\bpossible\b",
    r"\bsuspicious for\b",
    r"\bconcerning for\b",
    r"\bsuggests\b",
    r"\blikely\b",
    r"\bprobable\b",
    r"\bquestionable\b",
    r"\bindeterminate\b",
    r"\bequivocal\b",
    r"\blimited evaluation\b",
    r"\bnot adequately evaluated\b",
    r"\bpoorly visualized\b",
    r"\bearly arterial phase limits evaluation\b",
    r"\btoo small to characterize\b",
    r"\btstc\b",
]

NEG_RE = re.compile("|".join(NEGATION_CUES), re.IGNORECASE)
UNC_RE = re.compile("|".join(UNCERTAINTY_CUES), re.IGNORECASE)

SECTION_START_RE = re.compile(r"^\s*(history|indication|clinical history)\s*:?\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*[A-Z][A-Z /]{2,}:\s*$")


def _strip_history_blocks(text: str) -> str:
    """Remove History/Indication blocks if they appear inside the provided text."""
    lines = text.splitlines()
    out: List[str] = []
    skipping = False
    for line in lines:
        if SECTION_START_RE.match(line.strip()):
            skipping = True
            continue
        if skipping and HEADING_RE.match(line.strip()):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def _sentences(text: str) -> Iterable[str]:
    # Lightweight segmentation that preserves evidence substrings.
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    for p in parts:
        s = p.strip()
        if s:
            yield s


def _is_negated(sentence: str, match_start: int) -> bool:
    window = sentence[max(0, match_start - 80) : match_start]
    if NEG_RE.search(window):
        return True
    # Post-match "is absent" style
    post = sentence[match_start : min(len(sentence), match_start + 120)]
    if re.search(r"\bis absent\b", post, re.IGNORECASE):
        return True
    return False


def _is_uncertain(sentence: str) -> bool:
    return bool(UNC_RE.search(sentence))


def _classify_mentions(
    text: str,
    mention_res: Sequence[re.Pattern],
    *,
    require_all: Sequence[re.Pattern] = (),
    forbid_any: Sequence[re.Pattern] = (),
    always_present_phrases: Sequence[re.Pattern] = (),
    special_uncertain_phrases: Sequence[re.Pattern] = (),
) -> Tuple[TriLabel, List[Evidence]]:
    """Generic document-level tri-state classification from sentence-level mentions."""
    present_ev: List[str] = []
    absent_ev: List[str] = []
    uncertain_ev: List[str] = []

    for sent in _sentences(text):
        if any(p.search(sent) for p in forbid_any):
            continue
        if require_all and not all(p.search(sent) for p in require_all):
            continue

        # Hard-coded always-present phrases (even if they contain uncertainty cues like "likely")
        if any(p.search(sent) for p in always_present_phrases):
            present_ev.append(sent)
            continue

        for pat in mention_res:
            m = pat.search(sent)
            if not m:
                continue

            if _is_negated(sent, m.start()):
                absent_ev.append(sent)
                continue

            # Disease-specific phrases that should map to uncertain even without generic uncertainty cues.
            if any(p.search(sent) for p in special_uncertain_phrases):
                uncertain_ev.append(sent)
                continue

            if _is_uncertain(sent):
                uncertain_ev.append(sent)
            else:
                present_ev.append(sent)

    # Conflict resolution
    if present_ev:
        return TriLabel.PRESENT, [Evidence(t, "present") for t in _dedupe(present_ev)]
    if uncertain_ev and not absent_ev:
        return TriLabel.UNCERTAIN, [Evidence(t, "uncertain") for t in _dedupe(uncertain_ev)]
    if uncertain_ev and absent_ev:
        # Negation wins if the only positives are uncertain.
        return TriLabel.ABSENT, [Evidence(t, "absent") for t in _dedupe(absent_ev)]
    if absent_ev:
        return TriLabel.ABSENT, [Evidence(t, "absent") for t in _dedupe(absent_ev)]
    return TriLabel.ABSENT, []


def _dedupe(items: List[str], limit: int = 3) -> List[str]:
    seen = set()
    out = []
    for it in items:
        key = it.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out


def extract_merlin_30(text: str) -> Dict[str, Tuple[int, List[str]]]:
    """Return {disease: (label, evidence_strings)} for Merlin 30 disease set."""
    cleaned = _strip_history_blocks(text or "")

    compiled: Dict[str, Tuple[int, List[str]]] = {}

    # --- Simple direct diseases ---
    simple_patterns: Dict[str, Sequence[str]] = {
        "submucosal_edema": [r"\bsubmucosal edema\b", r"\benterocolitis\b", r"\bgastritis\b"],
        "aortic_valve_calcification": [r"\baortic (valve|valvular) calcif"],
        "thrombosis": [r"\bthrombosis\b", r"\bthrombus\b"],
        "pancreatic_atrophy": [r"\bpancreatic atrophy\b", r"\batrophic pancreas\b"],
        "osteopenia": [r"\bosteopenia\b", r"\bdemineral(?:ized|ization)\b"],
        "surgically_absent_gallbladder": [r"\bsurgically absent\b.*\bgallbladder\b", r"\bstatus post cholecystectomy\b"],
        "atelectasis": [r"\batelectasis\b"],
        "abdominal_aortic_aneurysm": [r"\babdominal aortic aneurysm\b", r"\baaa\b"],
        "anasarca": [r"\banasarca\b", r"\bdiffuse (?:subcutaneous )?edema\b"],
        "hiatal_hernia": [r"\bhiatal hernia\b"],
        "lymphadenopathy": [r"\blymphadenopathy\b", r"\benlarged lymph nodes\b"],
        "prostatomegaly": [r"\bprostatomegaly\b", r"\benlarged prostate\b"],
        "biliary_ductal_dilation": [r"\bbiliary (?:ductal )?dilat", r"\bintrahepatic biliary.*dilat", r"\bcommon bile duct.*dilat"],
        "cardiomegaly": [r"\bcardiomegaly\b", r"\benlarged heart\b"],
        "splenomegaly": [r"\bsplenomegaly\b"],
        "hepatomegaly": [r"\bhepatomegaly\b"],
        "atherosclerosis": [r"\batherosclero"],
        "ascites": [r"\bascites\b"],
        "pleural_effusion": [r"\bpleural effusion\b"],
        "hepatic_steatosis": [r"\bhepatic steatosis\b", r"\bfatty liver\b", r"\bfatty infiltration\b"],
        "appendicitis": [r"\bappendicitis\b"],
        "gallstones": [r"\bgallstones\b", r"\bcholelithiasis\b"],
        "bowel_obstruction": [r"\bbowel obstruction\b", r"\bsmall bowel obstruction\b", r"\bsbo\b"],
        "free_air": [r"\bfree air\b", r"\bpneumoperitoneum\b"],
        "fracture": [r"\bfracture\b", r"\bcompression fracture\b", r"\brib fracture\b", r"\bvertebral.*fracture\b"],
    }

    for disease, pats in simple_patterns.items():
        label, ev = _classify_mentions(
            cleaned,
            [re.compile(p, re.IGNORECASE) for p in pats],
        )
        compiled[disease] = (int(label), [e.text for e in ev])

    # --- coronary_calcification (requires coronary + calcification) ---
    coronary_label, coronary_ev = _classify_mentions(
        cleaned,
        [re.compile(r"\bcoronary\b", re.IGNORECASE), re.compile(r"\bcalcif", re.IGNORECASE)],
        require_all=[re.compile(r"\bcoronary\b", re.IGNORECASE), re.compile(r"\bcalcif", re.IGNORECASE)],
        forbid_any=[re.compile(r"\bmitral annular calcif", re.IGNORECASE), re.compile(r"\baortic\b.*\bcalcif", re.IGNORECASE)],
    )
    compiled["coronary_calcification"] = (int(coronary_label), [e.text for e in coronary_ev])

    # --- renal cyst vs renal hypodensities gating ---
    renal_cyst_label, renal_cyst_ev = _classify_mentions(
        cleaned,
        [
            re.compile(r"\brenal cyst", re.IGNORECASE),
            re.compile(r"\bcysts?\b.*\b(kidney|renal)\b", re.IGNORECASE),
        ],
        always_present_phrases=[
            re.compile(r"\bhypodens(?:e|ities)\b.*\blikely\b.*\bcysts?\b", re.IGNORECASE),
            re.compile(r"\btoo small to characterize\b.*\blikely\b.*\bcysts?\b", re.IGNORECASE),
            re.compile(r"\bsimple renal cyst", re.IGNORECASE),
            re.compile(r"\bconsistent with\b.*\bcysts?\b", re.IGNORECASE),
        ],
    )
    compiled["renal_cyst"] = (int(renal_cyst_label), [e.text for e in renal_cyst_ev])

    # Renal hypodensities: only count if not characterized as cyst
    hypodensity_mentions = [
        re.compile(r"\b(hypodens(?:e|ity|ities)|renal (?:mass|lesion)|renal (?:hypodense )?lesion)\b", re.IGNORECASE),
    ]
    hypodensity_label, hypodensity_ev = _classify_mentions(
        cleaned,
        hypodensity_mentions,
        special_uncertain_phrases=[
            re.compile(r"\btoo small to characterize\b", re.IGNORECASE),
            re.compile(r"\bindeterminate\b", re.IGNORECASE),
        ],
        forbid_any=[
            re.compile(r"\blikely\b.*\bcysts?\b", re.IGNORECASE),
            re.compile(r"\bsimple cyst", re.IGNORECASE),
            re.compile(r"\bconsistent with\b.*\bcysts?\b", re.IGNORECASE),
        ],
    )
    # Cross-label gating: if cyst is present and hypodensity only explained as cysts, force hypodensities absent.
    if compiled["renal_cyst"][0] == int(TriLabel.PRESENT) and hypodensity_label != TriLabel.PRESENT:
        # If we didn't find a non-cystic renal hypodensity, keep it absent.
        hypodensity_label = TriLabel.ABSENT
        hypodensity_ev = []
    compiled["renal_hypodensities"] = (int(hypodensity_label), [e.text for e in hypodensity_ev])

    # --- hydronephrosis (include synonyms; treat mild prominence as uncertain) ---
    hydronephrosis_label, hydronephrosis_ev = _classify_mentions(
        cleaned,
        [
            re.compile(r"\bhydronephrosis\b", re.IGNORECASE),
            re.compile(r"\bhydroureteronephrosis\b", re.IGNORECASE),
            re.compile(r"\bhydroureter\b", re.IGNORECASE),
            re.compile(r"\bpelviectasis\b|\bpyelectasis\b", re.IGNORECASE),
            re.compile(r"\bcollecting system (?:dilat|dilatation|dilation)\b", re.IGNORECASE),
            re.compile(r"\bdilated collecting system\b", re.IGNORECASE),
        ],
        special_uncertain_phrases=[
            re.compile(r"\bmild prominence of (?:the )?collecting system\b", re.IGNORECASE),
        ],
    )
    compiled["hydronephrosis"] = (int(hydronephrosis_label), [e.text for e in hydronephrosis_ev])

    # --- metastatic_disease ---
    metastatic_label, metastatic_ev = _classify_mentions(
        cleaned,
        [
            re.compile(r"\bmetastatic\b", re.IGNORECASE),
            re.compile(r"\bmetastases\b", re.IGNORECASE),
            re.compile(r"\bperitoneal carcinomatosis\b", re.IGNORECASE),
            re.compile(r"\bomental caking\b", re.IGNORECASE),
            re.compile(r"\btumor deposits?\b", re.IGNORECASE),
        ],
    )
    compiled["metastatic_disease"] = (int(metastatic_label), [e.text for e in metastatic_ev])

    return compiled

