#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import yaml

# === Paths (user-provided) ===
LABELS_CSV_PATH = Path("/home/ld258/ipredict/neuro_symbolic/radioprior_v2/zero_shot_findings_disease_cls.csv")
PREDICTIONS_JSON_PATH = Path("/home/ld258/ipredict/rate/output_ct_merlin_30/final_results.json")

# This script expects the modality config that produced the predictions.
MODALITY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "modalities" / "abdomen_ct_merlin.yaml"


DISEASE_TO_QUESTION = {
    "submucosal_edema": "Is there submucosal edema?",
    "renal_hypodensities": "Are there renal hypodensities?",
    "aortic_valve_calcification": "Is there aortic valve calcification?",
    "coronary_calcification": "Is there coronary calcification?",
    "thrombosis": "Is there thrombosis?",
    "metastatic_disease": "Is there metastatic disease?",
    "pancreatic_atrophy": "Is there pancreatic atrophy?",
    "renal_cyst": "Is there a renal cyst?",
    "osteopenia": "Is there osteopenia?",
    "surgically_absent_gallbladder": "Is the gallbladder surgically absent?",
    "atelectasis": "Is there atelectasis?",
    "abdominal_aortic_aneurysm": "Is there abdominal aortic aneurysm?",
    "anasarca": "Is there anasarca?",
    "hiatal_hernia": "Is there hiatal hernia?",
    "lymphadenopathy": "Is there lymphadenopathy?",
    "prostatomegaly": "Is there prostatomegaly?",
    "biliary_ductal_dilation": "Is there biliary ductal dilation?",
    "cardiomegaly": "Is there cardiomegaly?",
    "splenomegaly": "Is there splenomegaly?",
    "hepatomegaly": "Is there hepatomegaly?",
    "atherosclerosis": "Is there atherosclerosis?",
    "ascites": "Is there ascites?",
    "pleural_effusion": "Is there pleural effusion?",
    "hepatic_steatosis": "Is there hepatic steatosis?",
    "appendicitis": "Is there appendicitis?",
    "gallstones": "Are there gallstones?",
    "hydronephrosis": "Is there hydronephrosis?",
    "bowel_obstruction": "Is there bowel obstruction?",
    "free_air": "Is there free air (pneumoperitoneum)?",
    "fracture": "Is there a fracture?",
}


YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
UNCERTAIN_RE = re.compile(r"\b(uncertain|unknown|cannot determine|can't determine|unsure)\b", re.IGNORECASE)


def parse_yes_no(answer: object) -> Optional[int]:
    """Parse a model answer into {1,0,None} for Yes/No/Unknown."""
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    if UNCERTAIN_RE.search(text):
        return None
    # Prefer explicit yes/no signals; allow extra text (e.g., "Yes." or "No - ...").
    if YES_RE.search(text) and not NO_RE.search(text):
        return 1
    if NO_RE.search(text) and not YES_RE.search(text):
        return 0
    # If both appear, it's ambiguous.
    return None


def load_questions_from_config(path: Path) -> Iterable[str]:
    cfg = yaml.safe_load(path.read_text())
    categories = cfg.get("categories", {})
    for cat in categories.values():
        for q in cat.get("questions", []):
            if isinstance(q, dict) and "question" in q:
                yield str(q["question"])


def build_pred_map_for_case(case_obj: Dict) -> Dict[str, object]:
    """Map question text -> raw answer for a single case."""
    qa = case_obj.get("qa_results") or case_obj.get("answers") or {}
    out: Dict[str, object] = {}
    if isinstance(qa, dict):
        for _, qa_list in qa.items():
            if isinstance(qa_list, list):
                for item in qa_list:
                    if isinstance(item, dict):
                        for q, a in item.items():
                            out[str(q)] = a
    return out


@dataclass
class DiseaseStats:
    n_labeled: int = 0
    n_predicted: int = 0
    n_correct: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def accuracy(self) -> float:
        return (self.n_correct / self.n_labeled) if self.n_labeled else float("nan")

    @property
    def coverage(self) -> float:
        return (self.n_predicted / self.n_labeled) if self.n_labeled else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return (self.tp / denom) if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom else float("nan")

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if p != p or r != r:  # NaN
            return float("nan")
        denom = p + r
        return (2 * p * r / denom) if denom else float("nan")


def main() -> int:
    if not LABELS_CSV_PATH.exists():
        raise SystemExit(f"Labels CSV not found: {LABELS_CSV_PATH}")
    if not PREDICTIONS_JSON_PATH.exists():
        raise SystemExit(f"Predictions JSON not found: {PREDICTIONS_JSON_PATH}")
    if not MODALITY_CONFIG_PATH.exists():
        raise SystemExit(f"Modality config not found: {MODALITY_CONFIG_PATH}")

    # Validate that our mapping questions exist in the config (helps catch drift).
    config_questions = set(load_questions_from_config(MODALITY_CONFIG_PATH))
    missing_questions = sorted({q for q in DISEASE_TO_QUESTION.values() if q not in config_questions})
    if missing_questions:
        raise SystemExit(
            "These questions are not present in the modality config; fix DISEASE_TO_QUESTION or the YAML:\n"
            + "\n".join(missing_questions)
        )

    labels = pd.read_csv(LABELS_CSV_PATH)
    if "case_id" not in labels.columns:
        raise SystemExit("Labels CSV must contain a 'case_id' column")

    diseases = [c for c in labels.columns if c != "case_id"]
    missing_mapping = sorted([d for d in diseases if d not in DISEASE_TO_QUESTION])
    if missing_mapping:
        raise SystemExit(
            "These label columns are missing from DISEASE_TO_QUESTION mapping:\n" + "\n".join(missing_mapping)
        )

    preds = json.loads(PREDICTIONS_JSON_PATH.read_text())

    # Precompute prediction maps for speed.
    pred_by_case: Dict[str, Dict[str, object]] = {}
    for case_id, case_obj in preds.items():
        if isinstance(case_obj, dict):
            pred_by_case[str(case_id)] = build_pred_map_for_case(case_obj)

    # Compute metrics.
    per_disease: Dict[str, DiseaseStats] = {d: DiseaseStats() for d in diseases}
    total_labeled = 0
    total_predicted = 0
    total_correct = 0

    # Optional: per-case accuracy for troubleshooting.
    per_case_rows = []

    for _, row in labels.iterrows():
        case_id = str(row["case_id"])
        pred_q_map = pred_by_case.get(case_id)
        if pred_q_map is None:
            continue

        case_labeled = 0
        case_correct = 0

        for disease in diseases:
            gt = row[disease]
            if gt not in (0, 1):
                continue

            case_labeled += 1
            total_labeled += 1

            question = DISEASE_TO_QUESTION[disease]
            pred_raw = pred_q_map.get(question)
            pred = parse_yes_no(pred_raw)

            per_disease[disease].n_labeled += 1

            if pred is not None:
                per_disease[disease].n_predicted += 1
                total_predicted += 1

                if pred == 1 and int(gt) == 1:
                    per_disease[disease].tp += 1
                elif pred == 1 and int(gt) == 0:
                    per_disease[disease].fp += 1
                elif pred == 0 and int(gt) == 0:
                    per_disease[disease].tn += 1
                elif pred == 0 and int(gt) == 1:
                    per_disease[disease].fn += 1

            if pred == int(gt):
                per_disease[disease].n_correct += 1
                total_correct += 1
                case_correct += 1

        if case_labeled:
            per_case_rows.append(
                {
                    "case_id": case_id,
                    "n_labeled": case_labeled,
                    "n_correct": case_correct,
                    "accuracy": case_correct / case_labeled,
                }
            )

    overall_acc = (total_correct / total_labeled) if total_labeled else float("nan")
    overall_cov = (total_predicted / total_labeled) if total_labeled else float("nan")

    print("=== Overall ===")
    print(f"cases_with_predictions: {len(per_case_rows)}")
    print(f"total_labeled: {total_labeled}")
    print(f"total_predicted_yesno: {total_predicted}")
    print(f"coverage: {overall_cov:.4f}")
    print(f"accuracy: {overall_acc:.4f}")

    print("\n=== Per-disease ===")
    rows = []
    for disease in diseases:
        st = per_disease[disease]
        n_no_pred = st.n_labeled - st.n_predicted
        rows.append(
            {
                "disease": disease,
                "n_labeled": st.n_labeled,
                "n_predicted_yesno": st.n_predicted,
                "n_no_pred": n_no_pred,
                "coverage": st.coverage,
                "accuracy": st.accuracy,
                "tp": st.tp,
                "fp": st.fp,
                "tn": st.tn,
                "fn": st.fn,
                "precision_ppv": st.precision,
                "recall_sensitivity": st.recall,
                "f1": st.f1,
            }
        )
    df_out = pd.DataFrame(rows).sort_values(["f1", "n_labeled"], ascending=[True, False])
    with pd.option_context("display.max_rows", 200, "display.width", 140):
        print(df_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Write per-case csv next to predictions for convenience.
    per_case_path = PREDICTIONS_JSON_PATH.parent / "per_case_accuracy.csv"
    pd.DataFrame(per_case_rows).to_csv(per_case_path, index=False)
    print(f"\nWrote per-case accuracy to: {per_case_path}")
    per_disease_path = PREDICTIONS_JSON_PATH.parent / "per_disease_metrics.csv"
    df_out.to_csv(per_disease_path, index=False)
    print(f"Wrote per-disease metrics to: {per_disease_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
