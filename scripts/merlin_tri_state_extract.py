#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.core.tri_state_rules import extract_merlin_30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-based tri-state extraction for Merlin 30 disease labels.")
    p.add_argument("--input-csv", required=True, help="CSV containing report IDs and Findings text")
    p.add_argument("--out-csv", required=True, help="Output CSV path")
    p.add_argument("--out-json", default="", help="Optional output JSON path")
    p.add_argument("--id-col", default="study id", help="Report ID column name (default: 'study id')")
    p.add_argument("--text-col", default="Findings", help="Findings text column name (default: 'Findings')")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json) if args.out_json else None

    df = pd.read_csv(in_path)
    if args.id_col not in df.columns or args.text_col not in df.columns:
        raise SystemExit(f"Missing required columns. Have: {df.columns.tolist()}")

    rows: List[Dict] = []
    json_out: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        report_id = str(row[args.id_col])
        text = "" if pd.isna(row[args.text_col]) else str(row[args.text_col])
        disease_map = extract_merlin_30(text)

        out_row: Dict[str, object] = {"report_id": report_id}
        for disease, (label, evidence_list) in disease_map.items():
            out_row[disease] = int(label)
            out_row[f"evidence_{disease}"] = " ||| ".join(evidence_list)

        rows.append(out_row)
        json_out[report_id] = {
            "report_id": report_id,
            "labels": {d: int(v[0]) for d, v in disease_map.items()},
            "evidence": {d: v[1] for d, v in disease_map.items()},
        }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(json_out, indent=2))
        print(f"Wrote: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

