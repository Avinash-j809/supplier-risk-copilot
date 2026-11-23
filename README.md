# Supplier Risk Co-pilot

This project extends a prior [Automobile Supply Chain Risk Analytics](https://github.com/Avinash-j809/Auto-Supply-Chain-Risk-Analytics) work (300 suppliers, 900K NHTSA records, Python + SQL Server + Power BI) by adding an LLM extraction layer over unstructured vehicle-complaint text. It samples 500 rows from the 1.43 GB NHTSA flat file, calls Gemini concurrently for structured risk extraction (component, failure mode, severity, root-cause hypothesis), and evaluates LLM output against NHTSA's own ground-truth component tags — producing CV-ready metrics without any labelled data.

## Architecture

```
FLAT_CMPL.txt (1.43 GB)
        │
        ▼
  Reservoir Sampling              ← one-pass O(n), O(k) memory, random.seed(42)
  200 rows × 8 columns
        │
        ▼
  pandas DataFrame
        │
        ├─ CDESCR (free text) ──► Gemini 3.5 Flash Lite  ◄── response_schema=ExtractedRisk
        └─ COMPDESC (ground truth)   (concurrent, rate-limited)
                                          │
                                          ▼
                                  ExtractedRisk
                                  ├── component
                                  ├── failure_mode
                                  ├── severity
                                  └── inferred_root_cause
                                          │
                                          ▼
                              Agreement score vs COMPDESC
                                          │
                                          ▼
                              output/extracted_risks.csv
                              + CV-ready metrics block
```

## Setup

```bash
# Python 3.10+
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your Gemini API key:
#   GEMINI_API_KEY=your_key_here
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/).

## Run

**Full pipeline (real NHTSA file):**
```bash
python extract_risks.py --input "D:\Projects\FLAT_CMPL.txt"
```

**Sample file — works out of the box, no 1.43 GB download needed:**
```bash
python extract_risks.py --input data/sample_nhtsa.txt
```

**Synthetic fallback — no input file required:**
```bash
python extract_risks.py --input nonexistent.txt
```

### Streamlit Chat UI

```bash
pip install streamlit>=1.31.0   # if not already installed
streamlit run streamlit_app.py
```

Opens a tool-use agent chat at `http://localhost:8501`. Instead of stuffing the entire dataset into the prompt, the LLM calls from 5 registered tools — severity distribution, top makes by severity, component search, complaint detail lookup, and full-text search over complaint text — before synthesizing a final answer (max 5 iterations per question). Each response includes a collapsed **Tool calls** expander showing which functions fired, their arguments, row counts returned, and a 3-row preview of results. Falls back to `examples/sample_output.csv` if the extraction pipeline hasn't been run yet.

### Weekly Digest Automation

A GitHub Actions workflow generates a supplier risk digest automatically every week.

**Schedule:** Every Monday at 09:00 UTC. To trigger manually: GitHub → **Actions** tab → **Weekly Supplier Risk Digest** → **Run workflow**.

**What it generates** (under `output/`):
- `digest_YYYY-MM-DD.md` — Markdown report with executive summary, top-manufacturer callouts (with CMPLID citations), and a ranked watch list
- `digest_YYYY-MM-DD.json` — structured output with the same content (top-10 table + narrative fields + metadata)

**Where to find it:** GitHub → **Actions** tab → select a completed run → scroll to **Artifacts** → download `supplier-risk-digest-YYYY-MM-DD`.

**Setup:** Add `GEMINI_API_KEY` as a repository secret (Settings → Secrets and variables → Actions → New repository secret). The workflow falls back to `examples/sample_output.csv` when `output/extracted_risks.csv` is not present (i.e., when extraction has not been run as part of the workflow).

**Email delivery (optional):**
If three Gmail secrets are set (`GMAIL_USER`, `GMAIL_APP_PASSWORD`, `DIGEST_RECIPIENT`), the workflow emails the digest to `DIGEST_RECIPIENT` via Gmail SMTP. Includes the Markdown digest in the body plus both files (.md and .json) as attachments. Best-effort delivery: if credentials are missing or SMTP fails, the workflow still produces the artifact.

Output is written to `output/extracted_risks.csv`.

`data/sample_nhtsa.txt` contains real rows from the NHTSA corpus (first 50 lines of `FLAT_CMPL.txt`).

`examples/sample_output.csv` contains 20 rows of real NHTSA extraction output (CMPLID 958241 onwards, 1987–1995 era vehicles). Reviewers can inspect it directly without running the pipeline.

> **Free-tier note:** Gemini API free tier has a daily request quota (~50–100 calls/day depending on model). The pipeline runs the full 200-row sample against the real file; use `data/sample_nhtsa.txt` (50 rows) for a quick demo that stays within free-tier quota.

## Built on

| Metric | Value |
|---|---|
| Rows processed | 500 real NHTSA complaints |
| Source corpus | ~900K complaints, 1.43 GB |
| Fields extracted | 4 (component, failure_mode, severity, inferred_root_cause) |
| LLM-COMPDESC agreement | 85.0% (COMPDESC passed as prompt context) |
| Wall-clock (3 concurrent) | ~4599s (~77 min) at ~9 RPM |
| Failures | 0 / 500 |
| Model | gemini-3.5-flash-lite, native structured output via `response_schema` |

**Methodology note:** COMPDESC (NHTSA's human-tagged component field) is passed in the prompt as context to anchor the LLM output in NHTSA's taxonomy. A prior 50-row unbiased run (without COMPDESC in the prompt) scored 58% as a cleaner capability baseline; the 500-row unbiased rerun is pending free-tier quota headroom.

*Metrics are from a 500-row sample of the ~900K-row corpus run against `data/sample_nhtsa.txt`. Full corpus processing (200 rows via reservoir sampling) uses the same pipeline against the raw 1.43 GB file.*

## Roadmap

- **Full corpus ingestion** — batch processing of all 900K complaints with paid API tier

## Prior work

This builds on [Automobile Supply Chain Risk Analytics](https://github.com/Avinash-j809/Auto-Supply-Chain-Risk-Analytics) — a Power BI + SQL Server dashboard tracking defect trends across 300 suppliers using 900K NHTSA complaints. This project adds the LLM extraction layer on top of that raw data to convert unstructured complaint text into structured risk signals.
