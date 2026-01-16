#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate rule-based Merlin tri-state predictions vs labels (ignore -1).")
    p.add_argument(
        "--labels-csv",
        default="/home/ld258/ipredict/neuro_symbolic/radioprior_v2/zero_shot_findings_disease_cls.csv",
        help="Ground-truth label CSV (default: provided path)",
    )
    p.add_argument(
        "--preds-csv",
        default="output_merlin_rules/merlin_tri_state_preds.csv",
        help="Predictions CSV from merlin_tri_state_extract.py",
    )
    p.add_argument(
        "--out-csv",
        default="output_merlin_rules/per_disease_metrics_rules.csv",
        help="Output metrics CSV",
    )
    p.add_argument("--id-col-labels", default="case_id", help="ID column in labels CSV (default: case_id)")
    p.add_argument("--id-col-preds", default="report_id", help="ID column in predictions CSV (default: report_id)")
    return p.parse_args()


def safe_div(num: int, den: int) -> float:
    return (num / den) if den else float("nan")


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels_csv)
    preds_path = Path(args.preds_csv)
    out_path = Path(args.out_csv)

    labels = pd.read_csv(labels_path)
    preds = pd.read_csv(preds_path)

    if args.id_col_labels not in labels.columns:
        raise SystemExit(f"Missing labels id column: {args.id_col_labels}")
    if args.id_col_preds not in preds.columns:
        raise SystemExit(f"Missing preds id column: {args.id_col_preds}")

    diseases: List[str] = [c for c in labels.columns if c != args.id_col_labels]
    missing = [d for d in diseases if d not in preds.columns]
    if missing:
        raise SystemExit(f"Predictions missing disease columns: {missing}")

    # Because disease columns exist in both inputs, merged dataframe will contain:
    # - ground truth: <disease>_gt
    # - predictions:  <disease>
    df = labels.merge(preds, left_on=args.id_col_labels, right_on=args.id_col_preds, how="inner", suffixes=("_gt", ""))
    if df.empty:
        raise SystemExit("No matching case IDs between labels and predictions.")

    rows = []
    total_tp = total_fp = total_tn = total_fn = 0
    total_eval = 0

    for disease in diseases:
        gt_col = f"{disease}_gt"
        pr_col = disease
        if gt_col not in df.columns:
            raise SystemExit(f"Missing ground-truth column after merge: {gt_col}")
        if pr_col not in df.columns:
            raise SystemExit(f"Missing prediction column after merge: {pr_col}")

        gt = df[gt_col]
        pr = df[pr_col]

        # Evaluate only where GT is 0/1 and prediction is 0/1 (ignore -1 on either side)
        mask = gt.isin([0, 1]) & pr.isin([0, 1])
        n_eval = int(mask.sum())
        if n_eval == 0:
            continue

        tp = int(((pr[mask] == 1) & (gt[mask] == 1)).sum())
        fp = int(((pr[mask] == 1) & (gt[mask] == 0)).sum())
        tn = int(((pr[mask] == 0) & (gt[mask] == 0)).sum())
        fn = int(((pr[mask] == 0) & (gt[mask] == 1)).sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)

        rows.append(
            {
                "disease": disease,
                "n_eval": n_eval,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision_ppv": precision,
                "recall_sensitivity": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        total_eval += n_eval

    out = pd.DataFrame(rows).sort_values(["f1", "n_eval"], ascending=[True, False])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    overall_precision = safe_div(total_tp, total_tp + total_fp)
    overall_recall = safe_div(total_tp, total_tp + total_fn)
    overall_f1 = safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn)
    overall_acc = safe_div(total_tp + total_tn, total_tp + total_tn + total_fp + total_fn)

    print(f"matched_cases: {df[args.id_col_labels].nunique()}")
    print(f"total_eval: {total_eval}")
    print(f"overall_precision_ppv: {overall_precision:.4f}")
    print(f"overall_recall_sensitivity: {overall_recall:.4f}")
    print(f"overall_f1: {overall_f1:.4f}")
    print(f"overall_accuracy: {overall_acc:.4f}")
    print(f"wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
