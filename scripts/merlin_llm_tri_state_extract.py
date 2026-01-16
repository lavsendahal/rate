#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.llm_tri_state import build_merlin_prompt, parse_merlin_llm_output  # noqa: E402
from core.merlin_diseases import MERLIN_DISEASES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM tri-state extraction (Merlin 30) with evidence + rule enforcement.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", default="")
    p.add_argument("--id-col", default="study id")
    p.add_argument("--text-col", default="Findings")
    p.add_argument("--config", default="config/default_config.yaml", help="Path to default_config.yaml")
    p.add_argument("--max-concurrency", type=int, default=32, help="HTTP request concurrency")
    p.add_argument("--request-timeout-s", type=int, default=180)
    return p.parse_args()


def get_content(resp: Dict[str, Any]) -> str:
    # Prefer standard OpenAI content; fall back to reasoning_content if needed.
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if content:
                return str(content)
            rc = msg.get("reasoning_content")
            if rc:
                return str(rc)
    return ""


def post_chat(base_url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def main() -> int:
    args = parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base_url = f"{cfg['server']['base_url']}:{cfg['server']['port']}"
    model = cfg["model"]["name"]
    temperature = cfg["model"].get("temperature", 0.0)
    top_p = cfg["model"].get("top_p", 1.0)
    max_tokens = int(cfg["model"].get("max_tokens", 256))

    df = pd.read_csv(args.input_csv)
    if args.id_col not in df.columns or args.text_col not in df.columns:
        raise SystemExit(f"Missing required columns. Have: {df.columns.tolist()}")

    rows: List[Dict[str, Any]] = []
    json_out: Dict[str, Any] = {}

    def run_one(report_id: str, findings: str) -> Tuple[str, Dict[str, Any]]:
        prompt = build_merlin_prompt(findings)
        req_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return JSON only. Do not include reasoning."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        resp = post_chat(base_url, req_payload, args.request_timeout_s)
        content = get_content(resp)
        parsed = parse_merlin_llm_output(findings, content)

        out_row: Dict[str, Any] = {"report_id": report_id}
        for d in MERLIN_DISEASES:
            out_row[d] = int(parsed.labels.get(d, 0))
            out_row[f"evidence_{d}"] = " ||| ".join(parsed.evidence.get(d, []))

        out_debug = {
            "report_id": report_id,
            "labels": parsed.labels,
            "evidence": parsed.evidence,
            "raw_payload": parsed.raw_payload,
        }
        return report_id, {"row": out_row, "debug": out_debug}

    # Concurrency
    max_workers = max(1, min(args.max_concurrency, len(df)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for _, r in df.iterrows():
            rid = str(r[args.id_col])
            txt = "" if pd.isna(r[args.text_col]) else str(r[args.text_col])
            futs.append(ex.submit(run_one, rid, txt))

        for fut in as_completed(futs):
            rid, out = fut.result()
            rows.append(out["row"])
            json_out[rid] = out["debug"]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("report_id").to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(json_out, indent=2))
        print(f"Wrote: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

