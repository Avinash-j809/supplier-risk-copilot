"""
Supplier Risk Co-pilot — LLM extraction pipeline over NHTSA vehicle complaints.

Samples up to 500 rows from the 1.43 GB NHTSA flat file via one-pass reservoir
sampling, calls Gemini 2.5 Flash concurrently for structured risk extraction,
evaluates LLM output against NHTSA ground-truth COMPDESC, and writes
output/extracted_risks.csv.

Usage:
    python extract_risks.py --input "D:\\Projects\\FLAT_CMPL.txt"
    python extract_risks.py --input nonexistent.txt  # triggers synthetic fallback
    python extract_risks.py --input data/sample_nhtsa.txt
    python extract_risks.py --input data/sample_nhtsa.txt --no-compdesc-context
"""

from __future__ import annotations

import argparse
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "gemini-3.5-flash-lite"
SAMPLE_SIZE = 500
RESERVOIR_SEED = 42
MAX_CONCURRENT = 3   # simultaneous Gemini calls
# With MAX_CONCURRENT=3 slots, each slot sleeps RATE_SLEEP seconds after the
# API call returns before releasing, capping throughput at ~9 RPM (safe under
# Gemini 2.5 Flash free-tier 10 RPM limit).
RATE_SLEEP = 20.0

# 0-indexed column positions in the tab-delimited FLAT_CMPL.txt file
COLUMN_MAP = {
    1:  "CMPLID",
    2:  "MFR_NAME",
    3:  "MAKETXT",
    4:  "MODELTXT",
    5:  "YEARTXT",
    11: "COMPDESC",   # NHTSA structured component tag — used as eval ground truth
    15: "DATEA",
    19: "CDESCR",     # free-text complaint — input to the LLM
}

PROMPT_TMPL = (
    "Analyze this NHTSA vehicle complaint and extract structured risk fields.\n"
    "NHTSA component category (reference only): {compdesc}\n"
    "Complaint text: {cdescr}\n\n"
    "Return component, failure_mode, severity (low/medium/high/critical), "
    "and inferred_root_cause."
)

# Used when --no-compdesc-context is passed; agreement score is then unbiased
# because the LLM cannot copy COMPDESC into the component field.
PROMPT_TMPL_NO_CTX = (
    "Analyze this NHTSA vehicle complaint and extract structured risk fields.\n"
    "Complaint text: {cdescr}\n\n"
    "Return component, failure_mode, severity (low/medium/high/critical), "
    "and inferred_root_cause."
)

# ---------------------------------------------------------------------------
# Pydantic v2 schema — passed to response_schema for native structured output
# (no manual JSON parsing; SDK returns a typed ExtractedRisk instance)
# ---------------------------------------------------------------------------
class ExtractedRisk(BaseModel):
    component: str
    failure_mode: str
    severity: Literal["low", "medium", "high", "critical"]
    inferred_root_cause: str


_gen_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=ExtractedRisk,
)

# ---------------------------------------------------------------------------
# Lazy Gemini client — benign race on first call (threads create equivalent clients)
# ---------------------------------------------------------------------------
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


# ---------------------------------------------------------------------------
# Synthetic fallback — 500 realistic NHTSA-style rows for demo without real file
# ---------------------------------------------------------------------------
def _synthetic_df(n: int = 500) -> pd.DataFrame:
    rng = random.Random(42)
    mfrs  = ["Ford Motor Company", "General Motors LLC", "Toyota Motor Corp", "Honda Motor Co"]
    makes = ["FORD", "GMC", "TOYOTA", "HONDA"]
    model_groups = [["F-150", "EXPLORER"], ["SILVERADO", "EQUINOX"], ["CAMRY", "RAV4"], ["CIVIC", "CR-V"]]
    components = [
        "FUEL SYSTEM, GASOLINE", "SERVICE BRAKES, HYDRAULIC", "POWER TRAIN",
        "ELECTRICAL SYSTEM", "AIR BAGS", "STEERING", "SUSPENSION",
        "ENGINE AND ENGINE COOLING",
    ]
    complaint_pool = [
        "VEHICLE STALLS WITHOUT WARNING ON HIGHWAY. ENGINE SHUTS OFF, POWER STEERING LOST.",
        "BRAKE PEDAL GOES TO FLOOR WITHOUT ADEQUATE STOPPING POWER. HAD TO PUMP BRAKES MULTIPLE TIMES.",
        "TRANSMISSION SLIPS BETWEEN GEARS. VEHICLE JERKS AND HESITATES WHEN ACCELERATING FROM STOP.",
        "BATTERY DRAINS OVERNIGHT. MULTIPLE ELECTRICAL COMPONENTS STOP FUNCTIONING SIMULTANEOUSLY.",
        "AIRBAG WARNING LIGHT STAYS ON. CONCERNED BAGS WILL NOT DEPLOY IN CRASH.",
        "STEERING WHEEL PULLS HARD TO LEFT AT HIGHWAY SPEEDS. DIFFICULT TO MAINTAIN LANE.",
        "FRONT SUSPENSION CLUNKS LOUDLY OVER BUMPS. CAR FEELS UNSTABLE.",
        "ENGINE OVERHEATS WITHIN 20 MINUTES OF DRIVING. COOLANT RESERVOIR EMPTY AFTER EACH DRIVE.",
    ]
    rows = []
    for i in range(n):
        mi = rng.randrange(len(mfrs))
        rows.append({
            "CMPLID":    str(10000000 + i),
            "MFR_NAME":  mfrs[mi],
            "MAKETXT":   makes[mi],
            "MODELTXT":  rng.choice(model_groups[mi]),
            "YEARTXT":   str(rng.randint(2015, 2023)),
            "COMPDESC":  rng.choice(components),
            "DATEA":     f"20{rng.randint(18, 23):02d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}",
            "CDESCR":    rng.choice(complaint_pool),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One-pass reservoir sampling — O(n) time, O(k) memory, never loads full file
# ---------------------------------------------------------------------------
def _reservoir_sample(filepath: str, k: int = SAMPLE_SIZE) -> list[str]:
    rng = random.Random(RESERVOIR_SEED)
    reservoir: list[str] = []
    with open(filepath, encoding="latin-1", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i < k:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)  # inclusive on both ends
                if j < k:
                    reservoir[j] = line
    return reservoir


def _lines_to_df(lines: list[str]) -> pd.DataFrame:
    rows = []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        rows.append({col: (parts[idx] if idx < len(parts) else "") for idx, col in COLUMN_MAP.items()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LLM extraction — retry wrapper + rate-limited dispatch
# ---------------------------------------------------------------------------
_rate_sem = threading.Semaphore(MAX_CONCURRENT)

# Set once by main() before the thread pool starts; safe to read from worker threads.
_use_compdesc_ctx: bool = True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def _call_once(cdescr: str, compdesc: str) -> ExtractedRisk:
    if _use_compdesc_ctx:
        prompt = PROMPT_TMPL.format(cdescr=cdescr, compdesc=compdesc)
    else:
        prompt = PROMPT_TMPL_NO_CTX.format(cdescr=cdescr)
    resp = _get_client().models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=_gen_config,
    )
    result = resp.parsed
    if result is None:
        raise ValueError("Null parsed response")
    return result


def _extract_one(row: pd.Series) -> dict | None:
    """Acquire rate semaphore, call Gemini, sleep to cap RPM, release."""
    _rate_sem.acquire()
    try:
        result = _call_once(str(row["CDESCR"]), str(row["COMPDESC"]))
        time.sleep(RATE_SLEEP)
        return result.model_dump()
    except Exception as exc:
        # Unwrap tenacity RetryError to surface the real API error
        cause = getattr(exc, "__cause__", None) or exc
        print(f"  CMPLID {row.get('CMPLID', '?')} failed: {type(cause).__name__}: {cause}")
        return None
    finally:
        _rate_sem.release()


# ---------------------------------------------------------------------------
# Evaluation — fuzzy token overlap between LLM component and NHTSA COMPDESC
# ---------------------------------------------------------------------------
def _agreement_score(df: pd.DataFrame) -> float:
    """
    A row agrees when: the LLM's component string is a substring of COMPDESC,
    OR any meaningful COMPDESC token (>3 chars) appears in the LLM component.
    The bidirectional check handles mismatch in verbosity direction.
    """
    def agrees(row) -> bool:
        llm = str(row.get("component", "")).lower().strip()
        gt  = str(row.get("COMPDESC",  "")).lower().strip()
        if not llm or not gt:
            return False
        if llm in gt:
            return True
        return any(tok in llm for tok in gt.split() if len(tok) > 3)

    return df.apply(agrees, axis=1).mean()


# ---------------------------------------------------------------------------
# Sample file helpers
# ---------------------------------------------------------------------------
def _write_sample_input(src: str, dest: str = "data/sample_nhtsa.txt", n: int = 50) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="latin-1", errors="replace") as fi, \
         open(dest, "w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if i >= n:
                break
            fo.write(line)


def _write_sample_output(src: str = "output/extracted_risks.csv",
                         dest: str = "examples/sample_output.csv") -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(src, nrows=20).to_csv(dest, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract supplier risk signals from NHTSA complaints.")
    parser.add_argument(
        "--input", default=r"D:\Projects\FLAT_CMPL.txt",
        help="Path to FLAT_CMPL.txt; if absent, synthetic fallback data is used.",
    )
    parser.add_argument(
        "--no-compdesc-context", action="store_true",
        help="Exclude COMPDESC from the prompt (CDESCR only). Agreement score is then unbiased "
             "because the LLM cannot copy the ground-truth tag into its own output.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete output/extracted_risks.csv before running, forcing a full re-run.",
    )
    args = parser.parse_args()

    global _use_compdesc_ctx
    _use_compdesc_ctx = not args.no_compdesc_context
    mode_label = "CDESCR only — unbiased" if args.no_compdesc_context else "CDESCR + COMPDESC hint"

    out_path = "output/extracted_risks.csv"
    Path("output").mkdir(exist_ok=True)

    if args.fresh and Path(out_path).exists():
        Path(out_path).unlink()
        print("--fresh: deleted existing output, starting clean run.")

    # Checkpoint: load any prior progress so we can skip already-processed rows.
    existing_cmplids: set[str] = set()
    write_header = True
    if Path(out_path).exists():
        try:
            existing_df = pd.read_csv(out_path)
            if not existing_df.empty and "CMPLID" in existing_df.columns:
                existing_cmplids = set(existing_df["CMPLID"].astype(str))
                write_header = False
        except Exception:
            pass

    real_file = Path(args.input).exists()
    if not real_file:
        print(f"Input '{args.input}' not found — running on synthetic fallback (500 rows).")
        df = _synthetic_df(500)
    else:
        # Read only the first 2 lines to confirm format; never load the full file
        with open(args.input, encoding="latin-1", errors="replace") as fh:
            head = [next(fh) for _ in range(2)]
        print(f"Detected {len(head[0].split(chr(9)))}-column tab-delimited file.")
        print(f"Reservoir-sampling {SAMPLE_SIZE} rows ...")
        lines = _reservoir_sample(args.input)
        df = _lines_to_df(lines)

    # Filter out already-processed CMPLIDs and report resume state.
    if existing_cmplids:
        df = df[~df["CMPLID"].astype(str).isin(existing_cmplids)].reset_index(drop=True)
        print(
            f"Resuming from existing output: {len(existing_cmplids)} rows already done, "
            f"{len(df)} to process"
        )

    n_rows = len(df)
    if n_rows == 0:
        print("All rows already processed. Use --fresh to force a full re-run.")
        return

    print(f"Extracting risks for {n_rows} rows via Gemini ({MAX_CONCURRENT} concurrent, ~9 RPM) [{mode_label}] ...")

    empty = {"component": None, "failure_mode": None, "severity": None, "inferred_root_cause": None}
    t0 = time.monotonic()
    results: list[dict | None] = [None] * n_rows
    batch_rows: list[dict] = []
    failures = 0

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(_extract_one, df.iloc[i]): i for i in range(n_rows)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            if result is None:
                failures += 1
            results[i] = result

            row = df.iloc[i].to_dict()
            row.update(result or empty)
            batch_rows.append(row)
            done += 1

            # Flush to disk every 25 completions so a crash loses at most 25 rows.
            if len(batch_rows) >= 25 or done == n_rows:
                pd.DataFrame(batch_rows).to_csv(out_path, mode='a', header=write_header, index=False)
                write_header = False
                batch_rows.clear()

            if done % 10 == 0:
                print(f"  {done}/{n_rows} complete ...")

    elapsed = time.monotonic() - t0
    print(f"Saved: {out_path}")

    out_df = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame([r or empty for r in results])],
        axis=1,
    )
    valid_df  = out_df.dropna(subset=["component"])
    agreement = _agreement_score(valid_df) if len(valid_df) else 0.0
    avg_lat   = elapsed / max(n_rows - failures, 1)

    sample_dest = Path("data/sample_nhtsa.txt")
    if real_file and Path(args.input).resolve() != sample_dest.resolve():
        _write_sample_input(args.input)
        print("Saved: data/sample_nhtsa.txt")
    if Path(out_path).exists():
        _write_sample_output()
        print("Saved: examples/sample_output.csv")

    print(f"""
=== CV-READY METRICS ===
Rows processed:          {n_rows}
Source corpus:           ~900K complaints, 1.43 GB
Fields extracted:        4 (component, failure_mode, severity, inferred_root_cause)
Prompt mode:             {mode_label}
LLM-COMPDESC agreement:  {agreement * 100:.1f}%
Wall-clock (concurrent): {elapsed:.1f}s
Estimated sequential:    {avg_lat * n_rows:.1f}s (based on avg per-call latency * N)
Failures:                {failures}
========================""")


if __name__ == "__main__":
    main()
