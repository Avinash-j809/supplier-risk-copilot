"""
Supplier Risk Co-pilot — Weekly digest generator.

Loads extracted complaints, computes a weighted top-10 manufacturer watch list,
calls Gemini to write a 2-3 paragraph narrative, and writes
output/digest_YYYY-MM-DD.{md,json}.

Usage:
    python weekly_digest.py
    python weekly_digest.py --data examples/sample_output.csv
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

MODEL_ID = "gemini-3.5-flash-lite"
DATA_PATHS = ["output/extracted_risks.csv", "examples/sample_output.csv"]
# Weighted risk score per complaint: critical carries the most urgency
SEVERITY_WEIGHTS = {"critical": 3, "high": 2, "medium": 1, "low": 0}


class DigestNarrative(BaseModel):
    executive_summary: str   # 2-3 sentences: totals, biggest risk, severity trend
    top_manufacturers: str   # paragraph with CMPLID callouts for top 3
    recommendation: str      # one actionable sentence for procurement team


_gen_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=DigestNarrative,
)


def load_data(override: str | None = None) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, source_path), falling back through DATA_PATHS."""
    paths = ([override] if override else []) + DATA_PATHS
    for path in paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            if not df.empty:
                return df, path
    raise FileNotFoundError(
        "No data file found. Run: python extract_risks.py --input data/sample_nhtsa.txt"
    )


def compute_watch_list(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Group by MFR_NAME, weight complaints by severity, return top-n descending."""
    rows = []
    for mfr, grp in df.groupby("MFR_NAME"):
        counts = grp["severity"].value_counts().to_dict()
        score = sum(SEVERITY_WEIGHTS.get(sev, 0) * cnt for sev, cnt in counts.items())
        rows.append({
            "MFR_NAME": str(mfr),
            "critical": int(counts.get("critical", 0)),
            "high":     int(counts.get("high",     0)),
            "medium":   int(counts.get("medium",   0)),
            "low":      int(counts.get("low",      0)),
            "weighted_score": int(score),
        })
    rows.sort(key=lambda r: r["weighted_score"], reverse=True)
    return rows[:top_n]


def _watch_list_md_table(watch_list: list[dict]) -> str:
    lines = ["Rank | Manufacturer | Critical | High | Medium | Low | Score",
             "-----|-------------|----------|------|--------|-----|------"]
    for i, r in enumerate(watch_list, 1):
        lines.append(
            f"{i:4d} | {r['MFR_NAME']:<40} | {r['critical']:8d} | "
            f"{r['high']:4d} | {r['medium']:6d} | {r['low']:3d} | {r['weighted_score']}"
        )
    return "\n".join(lines)


def _sample_cmplids(df: pd.DataFrame, mfr: str, n: int = 3) -> list[str]:
    """Return up to n CMPLIDs for mfr, preferring critical rows."""
    sub = df[df["MFR_NAME"] == mfr]
    critical = sub[sub["severity"] == "critical"]["CMPLID"].astype(str).head(n).tolist()
    if len(critical) >= n:
        return critical
    others = sub[sub["severity"] != "critical"]["CMPLID"].astype(str).head(n - len(critical)).tolist()
    return critical + others


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def _call_gemini(client: genai.Client, prompt: str) -> DigestNarrative:
    resp = client.models.generate_content(model=MODEL_ID, contents=prompt, config=_gen_config)
    result = resp.parsed
    if result is None:
        raise ValueError("Null parsed response from Gemini")
    return result


def generate_narrative(
    df: pd.DataFrame, watch_list: list[dict], client: genai.Client
) -> DigestNarrative:
    top3 = "\n".join(
        f"  - {r['MFR_NAME']}: example CMPLIDs {', '.join(_sample_cmplids(df, r['MFR_NAME']))} "
        f"(score={r['weighted_score']}, critical={r['critical']}, high={r['high']})"
        for r in watch_list[:3]
    )
    prompt = (
        "You are a supplier risk analyst writing a weekly digest for a procurement team.\n\n"
        f"Top-10 manufacturer risk watch list (weighted score: critical=3, high=2, medium=1, low=0):\n"
        f"{_watch_list_md_table(watch_list)}\n\n"
        f"Top 3 manufacturers with example complaint IDs:\n{top3}\n\n"
        "Write a digest:\n"
        "- executive_summary: 2-3 sentences covering total complaints, highest-risk manufacturer, "
        "and overall severity trend.\n"
        "- top_manufacturers: 2-3 sentences per manufacturer for the top 3, naming specific CMPLIDs "
        "and the failure patterns observed.\n"
        "- recommendation: One actionable sentence for the procurement team.\n"
        "Be specific and cite only the data provided. Do not invent complaint details."
    )
    return _call_gemini(client, prompt)


def write_outputs(
    watch_list: list[dict], narrative: DigestNarrative, source: str
) -> tuple[str, str]:
    today = date.today().isoformat()
    Path("output").mkdir(exist_ok=True)

    md_path = f"output/digest_{today}.md"
    md = "\n".join([
        f"# Supplier Risk Digest — {today}", "",
        f"_Source: {source}_", "",
        "## Executive Summary", "",
        narrative.executive_summary, "",
        "## Top Manufacturers", "",
        narrative.top_manufacturers, "",
        "## Recommendation", "",
        narrative.recommendation, "",
        "## Watch List", "",
        "| Rank | Manufacturer | Critical | High | Medium | Low | Score |",
        "|------|-------------|----------|------|--------|-----|-------|",
    ] + [
        f"| {i} | {r['MFR_NAME']} | {r['critical']} | {r['high']} "
        f"| {r['medium']} | {r['low']} | {r['weighted_score']} |"
        for i, r in enumerate(watch_list, 1)
    ])
    Path(md_path).write_text(md, encoding="utf-8")

    json_path = f"output/digest_{today}.json"
    Path(json_path).write_text(
        json.dumps({
            "date": today,
            "source": source,
            "top_10": watch_list,
            "narrative": narrative.model_dump(),
        }, indent=2),
        encoding="utf-8",
    )
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weekly supplier risk digest.")
    parser.add_argument("--data", default=None, help="Override data file path.")
    args = parser.parse_args()

    df, source = load_data(args.data)
    watch_list = compute_watch_list(df)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    narrative = generate_narrative(df, watch_list, client)
    md_path, _ = write_outputs(watch_list, narrative, source)
    print(f"Digest generated: {md_path}, {len(watch_list)} manufacturers tracked")


if __name__ == "__main__":
    main()
